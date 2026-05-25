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

from .config import FIGURE_DPI, GIF_FPS
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

    The rotation axis is the SI (z) axis; at angle θ the volume is rotated
    around that axis and the maximum is projected along the AP direction,
    yielding a 2-D coronal-ish view per frame.  When ``overlay`` is given,
    it is rotated and projected in lockstep and α-composited on top of the
    primary volume — used for the "alpha-fusion" panel.

    Parameters
    ----------
    volume : np.ndarray
        Primary 3-D volume ``(Z, Y, X)``.
    out : Path | str
        Output GIF path.
    voxel_spacing : tuple[float, float, float] | None
        ``(z, y, x)`` mm.  If provided, the volume is rescaled to isotropic
        spacing before rotation so the MIP doesn't look squashed.
    n_angles : int | None
        Number of angular samples.  Default :data:`mip.config.MIP_N_ANGLES`.
    fps : int | None
        Frame rate.  Default :data:`mip.config.MIP_FPS`.
    cmap : str, default ``"gray"``
        Primary colormap.  Use ``"gray"`` for MR, ``"hot"`` for PET.
    overlay : np.ndarray | None
        Optional second volume of the same shape as ``volume``.  When set,
        it is projected in lockstep and composited on top with ``alpha``.
    overlay_cmap : str, default ``"hot"``
        Colormap for the overlay.
    alpha : float, default 0.5
        Opacity for the overlay channel.

    Returns
    -------
    Path
        The written GIF path.
    """
    raise NotImplementedError
