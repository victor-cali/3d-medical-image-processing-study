"""Quantitative assessment helpers for 3-D segmentation masks.

Self-contained — pure NumPy.  No torch / SAM-2 dependency, so the
metrics can be unit-tested without any AI install.

Function :func:`volume_mm3` takes two ``(Z, Y, X)`` masks (any dtype — they get cast
to ``bool``) and never modify the inputs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["volume_mm3"]


def volume_mm3(
    mask: np.ndarray, voxel_spacing: tuple[float, float, float]
) -> float:
    """Physical volume of a binary mask in cubic millimetres.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape ``(Z, Y, X)``.
    voxel_spacing : tuple[float, float, float]
        ``(z_mm, y_mm, x_mm)`` voxel side lengths.

    Returns
    -------
    float
        ``mask.sum() * z_mm * y_mm * x_mm``.
    """
    z, y, x = voxel_spacing
    return float(mask.astype(bool, copy=False).sum() * z * y * x)
