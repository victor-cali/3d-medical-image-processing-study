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

import numpy as np

from .io import PETStudy

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

    One animation frame per PET time frame; intensity scale is fixed across
    frames (default to global ``min``/``max`` of the study) so brightness
    differences over time are visible rather than auto-normalised away.

    Parameters
    ----------
    study : PETStudy
        Source study; ``study.pixel_array`` is iterated along ``axis=0``.
    out : Path | str
        Output GIF path.  Parent dirs created on demand.
    fps : int | None, default ``None``
        Frames per second.  Defaults to :data:`mip.config.GIF_FPS`.
    cmap : str, default ``"hot"``
        Colormap for all three panels.
    vmin, vmax : float | None
        Global intensity bounds.  If ``None``, computed from the whole 4-D
        study so the scale is comparable across time frames.

    Returns
    -------
    Path
        The written GIF path.
    """
    raise NotImplementedError


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
