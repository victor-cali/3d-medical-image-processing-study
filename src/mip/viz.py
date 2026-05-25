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
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .io import PETStudy

__all__ = [
    "show_last_frame",
    "show_mean_frame",
    "show_three_median_planes",
]


def show_last_frame(
    study: PETStudy,
    *,
    axis: int = 0,
    n_cols: int = 8,
    cmap: str = "hot",
    ax: Axes | None = None,
) -> Figure:
    """Render every slice of the last PET frame as a single mosaic figure.

    Parameters
    ----------
    study : PETStudy
        The dynamic PET study; uses ``study.last_frame``.
    axis : int, default 0
        Slicing axis (0=axial, 1=coronal, 2=sagittal).
    n_cols : int, default 8
        Number of tiles per row in the mosaic.
    cmap : str, default ``"hot"``
        Matplotlib colormap.  ``"hot"`` is the conventional PET palette.
    ax : Axes | None, default ``None``
        If provided, draw the mosaic into this axes (single ``imshow`` of a
        composed mosaic image).  Otherwise a fresh figure is created.

    Returns
    -------
    Figure
        The matplotlib Figure (caller saves / displays).
    """
    raise NotImplementedError


def show_mean_frame(
    study: PETStudy,
    *,
    axis: int = 0,
    n_cols: int = 8,
    cmap: str = "hot",
    ax: Axes | None = None,
) -> Figure:
    """Render every slice of the mean-of-frames PET volume as a mosaic.

    Parameters mirror :func:`show_last_frame`; uses ``study.mean_volume``.
    """
    raise NotImplementedError


def show_three_median_planes(
    volume: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "hot",
) -> Figure:
    """Show the three median planes (axial / coronal / sagittal) side-by-side.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of shape ``(Z, Y, X)``.
    title : str | None, default ``None``
        Optional suptitle for the figure.
    cmap : str, default ``"hot"``
        Colormap.  Use ``"gray"`` for MR.

    Returns
    -------
    Figure
        A 1×3 figure with axial, coronal, sagittal panels.
    """
    raise NotImplementedError