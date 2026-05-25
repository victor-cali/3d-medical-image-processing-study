"""Static figure helpers for the deck and the report.

In the visualisation layer each function returns a ``matplotlib.figure.Figure`` 
(or draws on a user-supplied ``Axes``).

Conventions
-----------
* All volumes are ``(Z, Y, X)`` ``float32``; for 4-D PET the caller has
  already picked a frame or computed a mean.
* All ``axis`` arguments select the plane normal: ``axis=0`` → axial,
  ``axis=1`` → coronal, ``axis=2`` → sagittal.  Mosaics tile slices along
  this axis.
* Colormaps: ``"gray"`` for MR (anatomical), ``"hot"`` for PET (functional).
  Overlays use ``alpha`` blending with the PET cmap on top of MR gray.
* **Anisotropic voxels** are honoured via the ``voxel_spacing`` argument
  on the three-planes view: with ``z_mm = 3.27`` and ``x_mm = y_mm =
  1.17``, coronal / sagittal planes would otherwise look horizontally
  squashed.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import FIGURE_DPI
from .io import PETStudy

__all__ = [
    "show_last_frame",
    "show_mean_frame",
    "show_three_median_planes",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return ``volume`` sliced perpendicular to ``axis`` at ``index``.

    Always returns a 2-D view; collapses the chosen axis.
    """
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")

def _aspect_for_plane(
    axis: int, voxel_spacing: tuple[float, float, float]
) -> float:
    """Matplotlib ``aspect`` value so a 2-D slice is shown with physical proportions.

    Parameters
    ----------
    axis : int
        0 (axial), 1 (coronal), or 2 (sagittal).
    voxel_spacing : tuple[float, float, float]
        ``(z_mm, y_mm, x_mm)``.

    Returns
    -------
    float
        ``height_mm / width_mm`` for the resulting image, suitable to
        pass as ``imshow(..., aspect=...)``.  The default ``aspect=1`` is
        only correct for isotropic voxels; on FORISI PET (3.27 × 1.17 ×
        1.17 mm) the axial plane is square but coronal / sagittal are
        roughly 2.8× taller than wide.
    """
    z_mm, y_mm, x_mm = voxel_spacing
    if axis == 0:
        return y_mm / x_mm
    if axis == 1:
        return z_mm / x_mm
    if axis == 2:
        return z_mm / y_mm
    raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")

def _mosaic(
    volume: np.ndarray,
    *,
    axis: int,
    n_cols: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    voxel_spacing: tuple[float, float, float] | None,
    title: str | None,
) -> Figure:
    """Tile every slice along ``axis`` into an ``n_rows × n_cols`` grid.

    Each tile is one ``imshow`` call.  Empty tiles in the last row are
    hidden so the figure has a clean rectangular look.
    """
    n_slices = volume.shape[axis]
    n_rows = math.ceil(n_slices / n_cols)
    aspect = _aspect_for_plane(axis, voxel_spacing) if voxel_spacing else 1.0

    # Per-tile figure size in inches; tweak for readable mosaics.
    tile_w, tile_h = 1.4, 1.4 * aspect
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(tile_w * n_cols, tile_h * n_rows),
        dpi=FIGURE_DPI,
        squeeze=False,
        constrained_layout=True,
    )

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        ax.set_axis_off()
        if k < n_slices:
            ax.imshow(
                _take_slice(volume, axis, k),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect=aspect,
                interpolation="nearest",
            )

    if title:
        fig.suptitle(title, fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_last_frame(
    study: PETStudy,
    *,
    axis: int = 0,
    n_cols: int = 8,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = "PET — last frame",
) -> Figure:
    """Render every slice of the last PET frame as a mosaic.

    Parameters
    ----------
    study : PETStudy
        Source study; uses ``study.last_frame``.
    axis : int, default 0
        Slicing axis (0=axial, 1=coronal, 2=sagittal).
    n_cols : int, default 8
        Tiles per row.
    cmap : str, default ``"hot"``
        Matplotlib colormap (``"hot"`` is the conventional PET palette).
    vmin, vmax : float | None
        Intensity bounds.  ``None`` lets matplotlib autoscale per-tile,
        which can look prettier per-slice but obscures the global dynamic
        range.  Pass shared bounds to compare across mosaics.
    title : str | None
        Figure suptitle.  Pass ``None`` to suppress.

    Returns
    -------
    Figure
    """
    return _mosaic(
        study.last_frame,
        axis=axis,
        n_cols=n_cols,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        voxel_spacing=study.voxel_spacing,
        title=title,
    )


def show_mean_frame(
    study: PETStudy,
    *,
    axis: int = 0,
    n_cols: int = 8,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = "PET — mean of frames",
) -> Figure:
    """Render every slice of the time-averaged PET volume as a mosaic.

    Same parameters as :func:`show_last_frame`; uses ``study.mean_volume``.
    """
    return _mosaic(
        study.mean_volume,
        axis=axis,
        n_cols=n_cols,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        voxel_spacing=study.voxel_spacing,
        title=title,
    )


def show_three_median_planes(
    volume: np.ndarray,
    *,
    voxel_spacing: tuple[float, float, float] | None = None,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> Figure:
    """Show the three median planes (axial / coronal / sagittal) side-by-side.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of shape ``(Z, Y, X)``.
    voxel_spacing : tuple[float, float, float] | None
        ``(z_mm, y_mm, x_mm)``.  When provided, the coronal and sagittal
        panels are rendered with the correct physical aspect ratio
        (otherwise they look horizontally squashed for the FORISI
        anisotropy of 3.27 vs 1.17 mm).
    cmap, vmin, vmax, title
        Standard matplotlib options.

    Returns
    -------
    Figure
    """
    z, y, x = volume.shape
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4.2), dpi=FIGURE_DPI, constrained_layout=True
    )
    labels = ("Axial (z = Z/2)", "Coronal (y = Y/2)", "Sagittal (x = X/2)")
    indices = (z // 2, y // 2, x // 2)

    for ax, axis_idx, index, label in zip(axes, (0, 1, 2), indices, labels):
        ax.imshow(
            _take_slice(volume, axis_idx, index),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=_aspect_for_plane(axis_idx, voxel_spacing) if voxel_spacing else 1.0,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.4,
    mask_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    title: str | None = None,
    ax: Axes | None = None,
    aspect: float | str = 1.0,
) -> Figure:
    """Show a 2-D anatomical image with a binary mask alpha-composited on top.

    Parameters
    ----------
    image : np.ndarray
        Background grayscale image of shape ``(Y, X)``.
    mask : np.ndarray
        Binary mask, same shape as ``image``.  Truthy voxels are overlaid.
    alpha : float, default 0.4
        Mask opacity (``0`` = invisible, ``1`` = opaque).
    mask_color : tuple[float, float, float], default ``(1.0, 0.2, 0.2)``
        RGB color of the mask overlay (0-1 range).
    title : str | None
        Optional axes title.
    ax : Axes | None
        If provided, draw into this axes (no new figure created).
    aspect : float | str, default 1.0
        Pass-through to ``imshow``.

    Returns
    -------
    Figure
        The matplotlib Figure (the parent of ``ax`` when ``ax`` is given).
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image shape {image.shape} != mask shape {mask.shape}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=FIGURE_DPI, constrained_layout=True)
    else:
        fig = ax.figure

    ax.imshow(image, cmap="gray", aspect=aspect, interpolation="nearest")

    # Build an RGBA overlay where alpha is 0 outside the mask.
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = mask_color[0]
    rgba[..., 1] = mask_color[1]
    rgba[..., 2] = mask_color[2]
    rgba[..., 3] = mask.astype(bool).astype(np.float32) * alpha
    ax.imshow(rgba, aspect=aspect, interpolation="nearest")

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return fig
