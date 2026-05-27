"""Tumour segmentation: MedSAM-2 driven by a manual 2-D prompt.

The assignment asks for a *semi-automatic* segmenter: the user picks a
bounding box (or centroid) on **one** axial slice in the notebook, and
an AI model propagates the segmentation through the rest of the volume.

MedSAM-2 is a medical fine-tune of SAM-2.  SAM-2's core capability is
mask propagation across the frames of a "video" — we treat the MR's
axial slice stack as that video, seed the 2-D prompt at one frame, and
let :meth:`propagate_in_video` walk in both directions.

This module has a single public entry point,
:func:`segment_with_medsam2`.  Torch, PIL and ``sam2`` are *lazy
imported* inside the function so the rest of the package stays usable
on a stock pip env without MedSAM-2 installed.

Pipeline (matches the upstream ``medsam2_infer_3D_CT.py`` exactly):

1. Window the MR to ``uint8`` using the 1st / 99th percentiles.
2. Per-slice, resize to ``(3, 512, 512)`` via PIL (RGB by replicating
   the gray channel) → stacked as ``(D, 3, 512, 512)``.
3. Divide by 255, move to the inference device, subtract ImageNet
   mean ``(0.485, 0.456, 0.406)``, divide by ImageNet std ``(0.229,
   0.224, 0.225)``.
4. Build the predictor via :func:`sam2.build_sam.build_sam2_video_predictor_npz`.
5. ``init_state(img_resized, original_H, original_W)`` — the
   predictor needs the original dimensions to project masks back.
6. Forward propagation: ``add_new_points_or_box`` at ``prompt_slice``
   → ``propagate_in_video`` collects frame-by-frame masks.
7. ``reset_state`` → re-add the prompt → ``propagate_in_video(reverse=True)``
   for the backward direction.  This is the upstream pattern.

MedSAM-2 install (one-time, see ``reports/PLAN.md §15``)::

    git clone https://github.com/bowang-lab/MedSAM2.git
    # then edit pyproject.toml to add:
    #   medsam2 = { path = "../MedSAM2", editable = true }
    #   torch        = { version = "==2.5.1",  index = "https://download.pytorch.org/whl/cu124" }
    #   torchvision  = { version = "==0.20.1", index = "https://download.pytorch.org/whl/cu124" }
    pixi install
    cd ../MedSAM2 && bash download.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import (
    DEVICE,
    MEDSAM2_CHECKPOINT,
    MEDSAM2_IMAGE_SIZE,
    MEDSAM2_MODEL_CFG,
)

__all__ = ["segment_with_medsam2"]


# ImageNet statistics used to normalise the MR before inference.  These are
# the exact constants MedSAM-2 was trained against (mirrors the upstream
# script — changing them silently degrades the mask quality).
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


_INSTALL_HINT = (
    "segment_with_medsam2 requires MedSAM-2 (https://github.com/bowang-lab/MedSAM2). Install:\n"
    "  git clone https://github.com/bowang-lab/MedSAM2.git\n"
    "  # add to your pyproject.toml under [tool.pixi.pypi-dependencies]:\n"
    '  #   medsam2     = { path = "../MedSAM2", editable = true }\n'
    '  #   torch       = { version = "==2.5.1",  index = "https://download.pytorch.org/whl/cu124" }\n'
    '  #   torchvision = { version = "==0.20.1", index = "https://download.pytorch.org/whl/cu124" }\n'
    "  pixi install\n"
    "  cd ../MedSAM2 && bash download.sh\n"
    "See reports/PLAN.md §15 for the full procedure."
)


# ---------------------------------------------------------------------------
# Pre-processing helper (lifted from the upstream inference script).
# ---------------------------------------------------------------------------


def _resize_grayscale_to_rgb_and_resize(
    array: np.ndarray, image_size: int
) -> np.ndarray:
    """Resize a 3-D ``(D, H, W)`` ``uint8`` array to ``(D, 3, image_size, image_size)``.

    Each slice is converted to RGB by replicating the gray channel and then
    resized with PIL's default bilinear interpolation — this is the exact
    transform MedSAM-2 expects on inference (see
    ``medsam2_infer_3D_CT.py`` upstream).

    Parameters
    ----------
    array : np.ndarray
        Slice stack of shape ``(D, H, W)``, ``uint8``.
    image_size : int
        Target spatial dimension (square).  MedSAM-2 was trained at 512.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(D, 3, image_size, image_size)``.
    """
    from PIL import Image

    d, _h, _w = array.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float64)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        # PIL returns (H, W, 3); transpose to (3, H, W).
        out[i] = np.asarray(img).transpose(2, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_with_medsam2(
    mr: np.ndarray,
    *,
    prompt_slice: int,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    point_xy: tuple[int, int] | None = None,
    checkpoint_path: str | Path | None = None,
    model_cfg: str | None = None,
    image_size: int | None = None,
    device: str | None = None,
    window_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Run MedSAM-2 on the MR, driven by a 2-D prompt at one axial slice.

    SAM-2's video predictor treats the MR's axial slices as the frames
    of a "video".  We seed the prompt at ``prompt_slice`` and walk the
    rest of the volume forward and then backward (the upstream pattern
    — two passes with :meth:`predictor.reset_state` between).

    Parameters
    ----------
    mr : np.ndarray
        MR volume of shape ``(Z, Y, X)``, any numeric dtype.
    prompt_slice : int
        Axial slice index (``0 ≤ prompt_slice < Z``) where the prompt
        is seeded.
    bbox_xyxy : (int, int, int, int) | None
        ``(x0, y0, x1, y1)`` bounding box in **original MR voxel
        coordinates** (not in the 512×512 resized frame — the predictor
        remembers the original dimensions and handles the projection
        itself).  Exclusive of ``point_xy``.
    point_xy : (int, int) | None
        ``(x, y)`` centroid in original MR voxel coordinates.  Treated
        as a single positive-foreground click.  Exclusive of
        ``bbox_xyxy``.
    checkpoint_path : str | Path | None
        Path to the MedSAM-2 ``.pt`` weights.  ``None`` →
        :data:`mip.config.MEDSAM2_CHECKPOINT`.
    model_cfg : str | None
        SAM-2 hydra config name.  ``None`` →
        :data:`mip.config.MEDSAM2_MODEL_CFG`.
    image_size : int | None
        Spatial size at which inference runs (the model's training
        resolution).  ``None`` → :data:`mip.config.MEDSAM2_IMAGE_SIZE`
        (512).
    device : str | None
        Torch device.  ``None`` → :data:`mip.config.DEVICE`.
    window_percentiles : (float, float), default ``(1.0, 99.0)``
        Lower / upper percentiles for the MR intensity windowing that
        precedes the ``uint8`` cast.

    Returns
    -------
    np.ndarray
        Predicted binary mask ``(Z, Y, X)`` ``uint8`` on the same
        grid as ``mr``.  Voxels SAM-2 didn't mark are zero.

    Raises
    ------
    ValueError
        If neither or both of ``bbox_xyxy`` / ``point_xy`` are given,
        or ``prompt_slice`` is out of range.
    ImportError
        If ``sam2`` / ``torch`` / ``PIL`` are not installed.  Message
        contains the full install hint.
    FileNotFoundError
        If the resolved ``checkpoint_path`` doesn't exist.

    Notes
    -----
    **Bbox coordinate frame.** ``bbox_xyxy`` is in *original* MR voxel
    coordinates.  MedSAM-2's ``init_state`` stores the original H / W
    and projects the prompt into its internal 512×512 frame.  If you
    pre-scale the bbox to 512×512 yourself, the model will look at the
    wrong region and you'll get a tiny mask in the top-left corner.

    **Two propagation passes.** SAM-2's ``propagate_in_video`` walks
    *forward* from the prompted frame by default.  The upstream
    MedSAM-2 demo calls it once forward, then resets the state and
    calls it again with ``reverse=True`` — this is the pattern we
    follow.  Both directions write into the same ``(Z, Y, X)`` output
    array; the prompted slice itself is filled by either pass
    (whichever happens last for that slice has the final say).
    """
    # ---- Cheap argument validation, before any heavy import ------------
    if (bbox_xyxy is None) == (point_xy is None):
        raise ValueError(
            "Provide exactly one of bbox_xyxy or point_xy "
            f"(got bbox_xyxy={bbox_xyxy!r}, point_xy={point_xy!r})"
        )

    z_max, y_max, x_max = mr.shape
    if not 0 <= prompt_slice < z_max:
        raise ValueError(
            f"prompt_slice {prompt_slice} out of range [0, {z_max})"
        )

    # ---- Resolve defaults from mip.config ------------------------------
    resolved_device = device or DEVICE
    resolved_cfg = model_cfg or MEDSAM2_MODEL_CFG
    resolved_image_size = image_size or MEDSAM2_IMAGE_SIZE
    resolved_ckpt = Path(checkpoint_path or MEDSAM2_CHECKPOINT).expanduser()
    if not resolved_ckpt.exists():
        raise FileNotFoundError(
            f"MedSAM-2 checkpoint not found at {resolved_ckpt}.\n{_INSTALL_HINT}"
        )

    # ---- Lazy imports — keep torch / PIL / sam2 off the hot path -------
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor_npz
    except ImportError as exc:  # pragma: no cover (user-env-dependent)
        raise ImportError(_INSTALL_HINT) from exc

    # ---- 1. Window MR to uint8 over the whole volume (anatomy-wide
    #         scale, not slice-local) ------------------------------------
    lo_pct, hi_pct = window_percentiles
    lo, hi = np.percentile(mr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    mr_window = np.clip(mr.astype(np.float32), lo, hi)
    mr_uint8 = ((mr_window - lo) / (hi - lo) * 255.0).astype(np.uint8)

    # ---- 2. Per-slice resize + RGB broadcast ---------------------------
    img_resized = _resize_grayscale_to_rgb_and_resize(
        mr_uint8, resolved_image_size
    )  # (Z, 3, S, S) float64

    # ---- 3. Normalise: /255 then ImageNet ------------------------------
    img_tensor = torch.from_numpy(img_resized).to(resolved_device).float()
    img_tensor = img_tensor / 255.0
    mean_t = (
        torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=resolved_device)
        .view(3, 1, 1)
    )
    std_t = (
        torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=resolved_device)
        .view(3, 1, 1)
    )
    img_tensor = (img_tensor - mean_t) / std_t

    # ---- 4. Build the predictor ---------------------------------------
    predictor = build_sam2_video_predictor_npz(resolved_cfg, str(resolved_ckpt))

    # ---- 5-7. init_state, forward pass, reset, reverse pass -----------
    mask_full = np.zeros(mr.shape, dtype=np.uint8)

    with torch.inference_mode():
        # video_height / video_width are the *original* MR spatial dims
        # — the predictor uses them to project masks back from 512x512.
        inference_state = predictor.init_state(img_tensor, y_max, x_max)

        # Prompt arguments shared between the forward and reverse passes.
        prompt_kwargs = _prompt_kwargs(bbox_xyxy=bbox_xyxy, point_xy=point_xy)

        # Forward propagation.
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(prompt_slice),
            obj_id=1,
            **prompt_kwargs,
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            mask_2d = (mask_logits[0] > 0.0).detach().cpu().numpy()[0]
            mask_full[int(frame_idx)] = mask_2d.astype(np.uint8)

        # Reverse propagation: reset, re-prompt, walk backwards.
        predictor.reset_state(inference_state)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(prompt_slice),
            obj_id=1,
            **prompt_kwargs,
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            mask_2d = (mask_logits[0] > 0.0).detach().cpu().numpy()[0]
            mask_full[int(frame_idx)] = mask_2d.astype(np.uint8)

    return mask_full


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prompt_kwargs(
    *,
    bbox_xyxy: tuple[int, int, int, int] | None,
    point_xy: tuple[int, int] | None,
) -> dict:
    """Pack the user prompt into the kwargs ``add_new_points_or_box`` expects.

    Exactly one of ``bbox_xyxy`` / ``point_xy`` must be set — the public
    function has already validated this.
    """
    if bbox_xyxy is not None:
        return {"box": np.asarray(bbox_xyxy, dtype=np.float32)}
    assert point_xy is not None  # for mypy; validated by caller
    px, py = point_xy
    return {
        "points": np.asarray([[px, py]], dtype=np.float32),
        "labels": np.asarray([1], dtype=np.int32),
    }
