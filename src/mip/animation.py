"""GIF animations

Two public functions are provided:

* :func:`animate_three_planes` — three median planes x T time frames, from
  a :class:`mip.io.PETStudy`.
* :func:`rotating_mip` — a 360° rotating Maximum Intensity Projection of a
  3-D volume.

Both functions write a GIF via ``matplotlib.animation.PillowWriter`` and
return the resulting :class:`pathlib.Path`.  Frame counts are intentionally
capped (≤60) to keep file sizes reasonable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .config import FIGURE_DPI, GIF_FPS, MIP_FPS, MIP_N_ANGLES
from .io import PETStudy
from .viz import _aspect_for_plane

__all__ = ["animate_three_planes", "rotating_mip"]


def animate_three_planes(
    study: PETStudy,
    out: Path | str,
    *,
    fps: int | None = None,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Animate the 3 median planes (axial / coronal / sagittal) across time.

    One animation frame per PET time frame; intensity scale is fixed
    across frames (defaults to the global ``min``/``max`` of the whole
    4-D study) so brightness differences between early and late frames
    are visible rather than auto-normalised away.

    Parameters
    ----------
    study : PETStudy
        Source study; ``study.pixel_array`` is iterated along ``axis=0``.
    out : Path | str
        Output GIF path.  Parent dirs are created on demand.
    fps : int | None, default ``None``
        Frame rate.  Falls back to :data:`mip.config.GIF_FPS`.
    cmap : str, default ``"hot"``
        Colormap for all three panels.
    vmin, vmax : float | None
        Global intensity bounds.  ``None`` → computed from the whole 4-D
        study so the scale is comparable across time frames.

    Returns
    -------
    Path
        The written GIF path.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = fps if fps is not None else GIF_FPS

    volume0 = study.pixel_array[0]
    z, y, x = volume0.shape
    iz, iy, ix = z // 2, y // 2, x // 2

    if vmin is None:
        vmin = float(study.pixel_array.min())
    if vmax is None:
        vmax = float(study.pixel_array.max())

    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4.4), dpi=FIGURE_DPI, constrained_layout=True
    )
    labels = ("Axial", "Coronal", "Sagittal")
    images = []
    for ax, axis_idx, label in zip(axes, (0, 1, 2), labels):
        data = (
            volume0[iz, :, :] if axis_idx == 0
            else volume0[:, iy, :] if axis_idx == 1
            else volume0[:, :, ix]
        )
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=_aspect_for_plane(axis_idx, study.voxel_spacing),
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        images.append(im)

    title = fig.suptitle("PET — frame 0", fontsize=12)

    def _update(t_idx: int) -> tuple:
        vol = study.pixel_array[t_idx]
        images[0].set_data(vol[iz, :, :])
        images[1].set_data(vol[:, iy, :])
        images[2].set_data(vol[:, :, ix])
        t_s = float(study.frame_start_times_s[t_idx])
        d_s = float(study.frame_durations_s[t_idx])
        title.set_text(f"PET — frame {t_idx}   t = {t_s:.1f} s   Δt = {d_s:.1f} s")
        return (*images, title)

    anim = FuncAnimation(
        fig,
        _update,
        frames=study.n_frames,
        interval=1000 / fps,
        blit=False,
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def rotating_mip(
    volume: np.ndarray,
    out: Path | str,
    *,
    voxel_spacing: tuple[float, float, float] | None = None,
    n_angles: int | None = None,
    fps: int | None = None,
    cmap: str = "gray",
    overlay: np.ndarray | None = None,
    overlay_cmap: str = "hot",
    alpha: float = 0.5,
) -> Path:
    """Render a 360° rotating Maximum Intensity Projection as a GIF.

    The rotation axis is the SI (z) axis; at angle θ the volume is
    rotated around that axis and the maximum is projected along the AP
    direction.  When ``overlay`` is given, it is rotated and projected in
    lockstep and α-composited on top of the primary volume.

    Notes
    -----
    Algorithm:

    1. Optionally rescale the volume (and overlay) to isotropic spacing
       so the rotation is physically consistent.  For volumes already
       isotropic (e.g. coregistered onto a 1×1×1 mm MR), this is a no-op.
    2. Precompute the projection for every angle before opening the
       writer — ``scipy.ndimage.rotate`` is the expensive step; doing it
       once per angle and caching the resulting 2-D MIP keeps the
       per-frame ``imshow`` update cheap.
    3. Animate the precomputed stack with ``FuncAnimation`` +
       ``PillowWriter``.
    """
    from scipy.ndimage import rotate, zoom

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_angles = n_angles if n_angles is not None else MIP_N_ANGLES
    fps = fps if fps is not None else MIP_FPS

    if overlay is not None and overlay.shape != volume.shape:
        raise ValueError(
            f"overlay shape {overlay.shape} != volume shape {volume.shape}"
        )

    # 1. Make voxels isotropic so the AP-direction projection is unbiased.
    vol = volume
    ov = overlay
    if voxel_spacing is not None:
        z_mm, y_mm, x_mm = voxel_spacing
        target = min(z_mm, y_mm, x_mm)
        zoom_factors = (z_mm / target, y_mm / target, x_mm / target)
        if not all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
            vol = zoom(vol, zoom_factors, order=1)
            if ov is not None:
                ov = zoom(ov, zoom_factors, order=1)

    # 2. Precompute projections at each angle (the slow step).
    angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    proj_stack: list[np.ndarray] = []
    overlay_stack: list[np.ndarray] | None = [] if ov is not None else None

    for theta in angles:
        rot = rotate(
            vol, theta, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0
        )
        proj_stack.append(rot.max(axis=1))
        if ov is not None:
            rot_ov = rotate(
                ov, theta, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0
            )
            assert overlay_stack is not None
            overlay_stack.append(rot_ov.max(axis=1))

    # Robust intensity bounds (1st/99th percentile) — avoids a single
    # bright voxel washing out the dynamic range across all frames.
    base = np.stack(proj_stack)
    vmin = float(np.percentile(base, 1))
    vmax = float(np.percentile(base, 99))
    ov_vmin = 0.0
    ov_vmax = 1.0
    if overlay_stack is not None:
        ov_base = np.stack(overlay_stack)
        # Ignore zero voxels so the overlay scale isn't dominated by
        # background after rotation padding.
        nonzero = ov_base[ov_base > 0]
        ov_vmin = float(np.percentile(nonzero, 1)) if nonzero.size else 0.0
        ov_vmax = float(np.percentile(ov_base, 99))

    # 3. Render the animation from the precomputed stack.
    fig, ax = plt.subplots(figsize=(5, 6), dpi=FIGURE_DPI, constrained_layout=True)
    ax.set_axis_off()
    im = ax.imshow(
        proj_stack[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        origin="lower",
    )
    im_ov = None
    if overlay_stack is not None:
        im_ov = ax.imshow(
            overlay_stack[0],
            cmap=overlay_cmap,
            vmin=ov_vmin,
            vmax=ov_vmax,
            alpha=alpha,
            aspect="equal",
            interpolation="nearest",
            origin="lower",
        )
    title = ax.set_title(f"angle = {angles[0]:.0f}°", fontsize=10)

    def _update(i: int) -> tuple:
        im.set_data(proj_stack[i])
        artists: tuple = (im,)
        if im_ov is not None:
            assert overlay_stack is not None
            im_ov.set_data(overlay_stack[i])
            artists = (im, im_ov)
        title.set_text(f"angle = {angles[i]:.0f}°")
        return (*artists, title)

    anim = FuncAnimation(
        fig, _update, frames=n_angles, interval=1000 / fps, blit=False
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
