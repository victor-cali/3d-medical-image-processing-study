"""DICOM I/O for the dynamic PET study and the MR reference.

The two DICOM files in ``data/FORISI/`` are *enhanced multi-frame* objects:
PyDicom returns a flat ``pixel_array`` of shape ``(n_frames, rows, cols)``
where ``n_frames = T * Z`` for the PET and ``Z`` for the MR.  The
reconstruction job here is to:

1. Read the geometry headers (rows, columns, slice positions, pixel spacing,
   spacing between slices).
2. Read the timing headers, including the vendor-private vectors, for the 
   dynamic PET.
3. Group the flat slice stack into a 4-D ``(T, Z, Y, X)`` array, so downstream 
   code never has to think about the ordering again.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pydicom


@dataclass(frozen=True, slots=True)
class PETStudy:
    """A dynamic PET reconstruction with timing metadata.

    Parameters
    ----------
    pixel_array : np.ndarray
        Volume of shape ``(T, Z, Y, X)``, ``float32``.  Reordered from
        PyDicom's flat ``(T*Z, Y, X)`` stack using the slice-position vector
        so that ``pixel_array[t]`` is a single 3-D PET volume acquired at
        frame ``t``.
    voxel_spacing : tuple[float, float, float]
        ``(z_mm, y_mm, x_mm)``.  ``z`` from ``(0018, 0088) SpacingBetweenSlices``,
        ``y`` / ``x`` from ``(0028, 0030) PixelSpacing``.
    frame_start_times_ms : np.ndarray
        Shape ``(T,)``.  Acquisition start times in milliseconds, from the
        private tag ``(0055, 1001)``.
    frame_durations_ms : np.ndarray
        Shape ``(T,)``.  Per-frame integration durations in ms, from
        ``(0055, 1004)``.
    slice_positions_mm : np.ndarray
        Shape ``(Z,)``.  Unique slice positions along the SI axis (mm),
        derived from ``(0055, 1002) FramePositionsVector``.
    affine : np.ndarray
        Shape ``(4, 4)``.  Voxel-to-world affine in DICOM patient
        coordinates, built from ``ImagePositionPatient`` and
        ``ImageOrientationPatient``.  Drives SimpleITK image creation.
    raw_headers : pydicom.Dataset
        The PyDicom dataset as-loaded.  Kept out of equality / hashing
        (see ``compare=False`` below) so two ``PETStudy`` instances compare
        on numeric content.
    """

    pixel_array: np.ndarray
    voxel_spacing: tuple[float, float, float]
    frame_start_times_ms: np.ndarray
    frame_durations_ms: np.ndarray
    slice_positions_mm: np.ndarray
    affine: np.ndarray
    raw_headers: pydicom.Dataset = field(repr=False, compare=False)

    @property
    def n_frames(self) -> int:
        """Number of time frames ``T``."""
        return int(self.pixel_array.shape[0])

    @property
    def n_slices(self) -> int:
        """Number of axial slices ``Z`` per frame."""
        return int(self.pixel_array.shape[1])

    @property
    def mean_volume(self) -> np.ndarray:
        """The time-averaged 3-D volume of shape ``(Z, Y, X)``, ``float32``."""
        return self.pixel_array.mean(axis=0, dtype=np.float32)

    @property
    def last_frame(self) -> np.ndarray:
        """The last time frame, shape ``(Z, Y, X)``, ``float32``."""
        return self.pixel_array[-1]


@dataclass(frozen=True, slots=True)
class MRVolume:
    """A 3-D anatomical MR reference volume.

    Parameters
    ----------
    pixel_array : np.ndarray
        Shape ``(Z, Y, X)``, ``float32``.
    voxel_spacing : tuple[float, float, float]
        ``(z_mm, y_mm, x_mm)``.
    affine : np.ndarray
        Shape ``(4, 4)``.  Voxel-to-world affine.
    raw_headers : pydicom.Dataset
        Original PyDicom dataset; debug / appendix only.
    """

    pixel_array: np.ndarray
    voxel_spacing: tuple[float, float, float]
    affine: np.ndarray
    raw_headers: pydicom.Dataset = field(repr=False, compare=False)