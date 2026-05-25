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
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

# Vendor-private DICOM tags (anchored by Private Creator at (0055, 0010)).
# Tag numbers verified against the FORISI files.
_TAG_FRAME_START_TIMES = (0x0055, 0x1001)   # FD, length T, **seconds**
_TAG_FRAME_POSITIONS = (0x0055, 0x1002)     # FD, length 3 * n_frames, (x,y,z)/slice
_TAG_FRAME_DURATIONS = (0x0055, 0x1004)     # FD, length T, **milliseconds**
_TAG_RESCALE_SLOPES = (0x0055, 0x1005)      # FL, length n_frames (per slice)

@dataclass(frozen=True, slots=True)
class PETStudy:
    """A dynamic PET reconstruction with timing metadata.

    Parameters
    ----------
    pixel_array : np.ndarray
        Volume of shape ``(T, Z, Y, X)``, ``float32``.  Reordered from
        PyDicom's flat ``(T*Z, Y, X)`` stack and rescaled per slice using
        ``(0055, 1005)`` so values are in physical PET counts.
        ``pixel_array[t]`` is the 3-D PET volume acquired at frame ``t``.
    voxel_spacing : tuple[float, float, float]
        ``(z_mm, y_mm, x_mm)``.  ``z`` from ``(0018, 0088)
        SpacingBetweenSlices``; ``y``/``x`` from ``(0028, 0030)
        PixelSpacing``.
    frame_start_times_s : np.ndarray
        Shape ``(T,)``.  Start times in **seconds**, taken verbatim from
        ``(0055, 1001)`` (vendor packs them in seconds despite the FD VR).
    frame_durations_s : np.ndarray
        Shape ``(T,)``.  Per-frame integration durations in **seconds**,
        converted from ``(0055, 1004)`` (vendor packs that one in ms).
    slice_positions_mm : np.ndarray
        Shape ``(Z,)``.  Unique slice z positions (mm), the z column of the
        reshaped ``(0055, 1002) FramePositionsVector``.
    affine : np.ndarray
        Shape ``(4, 4)``.  Voxel-to-world affine.  ``ImagePositionPatient``
        and ``ImageOrientationPatient`` are absent from FORISI, so this is
        the axial-orientation fallback documented in PLAN decision D18.
    raw_headers : pydicom.Dataset
        The PyDicom dataset as-loaded.  Kept out of equality / hashing
        (``compare=False``) so two ``PETStudy`` instances compare on
        numeric content.
    """

    pixel_array: np.ndarray
    voxel_spacing: tuple[float, float, float]
    frame_start_times_s: np.ndarray
    frame_durations_s: np.ndarray
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

    def late_mean_volume(self, fraction: float = 1 / 3) -> np.ndarray:
        """Mean of the last ``fraction`` of time frames.

        Used to derive the segmentation prompt: late frames after the
        tracer has plateaued have much better SNR than the last frame
        alone (see PLAN decision D7).

        Parameters
        ----------
        fraction : float, default ``1/3``
            Fraction of trailing frames to average.  Must be in ``(0, 1]``.

        Returns
        -------
        np.ndarray
            Volume of shape ``(Z, Y, X)``, ``float32``.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        k = max(1, int(round(self.n_frames * fraction)))
        return self.pixel_array[-k:].mean(axis=0, dtype=np.float32)


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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_path(path: Path | str) -> Path:
    """Resolve and existence-check a filesystem path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _read_private_array(ds: pydicom.Dataset, tag: tuple[int, int]) -> np.ndarray:
    """Read a private DICOM tag and return its value as a NumPy array.

    Raises
    ------
    KeyError
        If the tag is missing.
    """
    if tag not in ds:
        raise KeyError(
            f"Required private tag ({tag[0]:#06x}, {tag[1]:#06x}) absent from DICOM"
        )
    return np.asarray(ds[tag].value, dtype=np.float64)


def _axial_affine(
    voxel_spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a 4×4 voxel-to-world affine for an axially-oriented volume.

    Used when ``ImageOrientationPatient`` is absent (PLAN decision D18).
    The diagonal is ``(x_mm, y_mm, z_mm, 1)`` so multiplying by a voxel
    index ``(i_x, i_y, i_z, 1)`` produces a world coordinate in mm.
    """
    z_mm, y_mm, x_mm = voxel_spacing
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = x_mm
    affine[1, 1] = y_mm
    affine[2, 2] = z_mm
    affine[:3, 3] = origin
    return affine


def _apply_rescale_per_slice(
    raw: np.ndarray, slopes: np.ndarray
) -> np.ndarray:
    """Multiply each axial slice of ``raw`` by its rescale slope.

    Parameters
    ----------
    raw : np.ndarray
        Flat ``(N, Y, X)`` integer slice stack from PyDicom.
    slopes : np.ndarray
        Per-slice slopes from ``(0055, 1005)``, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        ``float32`` array of the same shape as ``raw``.
    """
    if slopes.shape != (raw.shape[0],):
        raise ValueError(
            f"slope vector shape {slopes.shape} does not match slice count {raw.shape[0]}"
        )
    # Broadcasting: (N, 1, 1) * (N, Y, X) → (N, Y, X) in float32.
    return (raw.astype(np.float32) * slopes.astype(np.float32)[:, None, None])


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_dynamic_pet(path: Path | str) -> PETStudy:
    """Load a dynamic PET multi-frame DICOM and reshape it to ``(T, Z, Y, X)``.

    Parameters
    ----------
    path : Path | str
        Filesystem path to the enhanced multi-frame DICOM file.

    Returns
    -------
    PETStudy
        Fully assembled dataclass — see :class:`PETStudy`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If any required private tag is missing.
    ValueError
        If the flat slice count is not ``T * Z`` (i.e. the reshape would
        lose data), or if the z grid is inconsistent across time frames.

    Notes
    -----
    See PLAN §11 for the tag-level reasoning and decisions D16–D18 for the
    unit / orientation conventions.
    """
    p = _require_path(path)
    ds = pydicom.dcmread(str(p))

    n_frames_flat = int(ds.NumberOfFrames)
    rows, cols = int(ds.Rows), int(ds.Columns)

    # Per-slice geometry: 3 doubles per slice — (x, y, z).  Z column drives
    # the reshape; x/y are constant across slices for this axial study.
    positions = _read_private_array(ds, _TAG_FRAME_POSITIONS).reshape(-1, 3)
    if positions.shape[0] != n_frames_flat:
        raise ValueError(
            f"FramePositionsVector has {positions.shape[0]} rows, "
            f"NumberOfFrames says {n_frames_flat}"
        )

    # Frame timings (verbatim length = T).
    start_times_s = _read_private_array(ds, _TAG_FRAME_START_TIMES)  # already seconds (D16)
    durations_ms = _read_private_array(ds, _TAG_FRAME_DURATIONS)
    durations_s = durations_ms / 1000.0  # vendor packs in ms (D16)

    t = int(start_times_s.shape[0])
    if n_frames_flat % t != 0:
        raise ValueError(
            f"NumberOfFrames {n_frames_flat} is not divisible by T={t}; "
            "cannot reshape to (T, Z, Y, X)"
        )
    z = n_frames_flat // t

    # Sanity: the z grid in the first frame must repeat in subsequent
    # frames; if it doesn't, the simple stride-Z reshape would mis-align
    # slices and we should abort rather than silently corrupt the volume.
    z_per_slice = positions[:, 2]
    first_frame_z = z_per_slice[:z]
    for f_idx in range(1, t):
        block = z_per_slice[f_idx * z : (f_idx + 1) * z]
        if not np.allclose(block, first_frame_z):
            raise ValueError(
                f"Slice z-grid mismatch at frame {f_idx}: simple reshape unsafe"
            )

    # Per-slice rescale (constant here, but stored per-slice in the file).
    rescale = _read_private_array(ds, _TAG_RESCALE_SLOPES)
    flat = _apply_rescale_per_slice(ds.pixel_array, rescale)
    pixel = flat.reshape(t, z, rows, cols)

    voxel_spacing: tuple[float, float, float] = (
        float(ds.SpacingBetweenSlices),
        float(ds.PixelSpacing[0]),
        float(ds.PixelSpacing[1]),
    )

    affine = _axial_affine(
        voxel_spacing,
        origin=(
            float(positions[0, 0]),
            float(positions[0, 1]),
            float(first_frame_z[0]),
        ),
    )

    return PETStudy(
        pixel_array=pixel,
        voxel_spacing=voxel_spacing,
        frame_start_times_s=start_times_s.astype(np.float64),
        frame_durations_s=durations_s.astype(np.float64),
        slice_positions_mm=first_frame_z.astype(np.float64),
        affine=affine,
        raw_headers=ds,
    )


def load_mr(path: Path | str) -> MRVolume:
    """Load the 3-D anatomical MR DICOM.

    Parameters
    ----------
    path : Path | str
        Filesystem path to the (enhanced multi-frame) DICOM file.

    Returns
    -------
    MRVolume
        See :class:`MRVolume`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If the per-slice rescale tag is missing.
    """
    p = _require_path(path)
    ds = pydicom.dcmread(str(p))

    raw = ds.pixel_array
    rescale = _read_private_array(ds, _TAG_RESCALE_SLOPES)
    pixel = _apply_rescale_per_slice(raw, rescale)

    voxel_spacing: tuple[float, float, float] = (
        float(ds.SpacingBetweenSlices),
        float(ds.PixelSpacing[0]),
        float(ds.PixelSpacing[1]),
    )

    affine = _axial_affine(voxel_spacing)

    return MRVolume(
        pixel_array=pixel,
        voxel_spacing=voxel_spacing,
        affine=affine,
        raw_headers=ds,
    )


def to_sitk_image(
    study: PETStudy | MRVolume,
    *,
    frame: int | None = None,
) -> sitk.Image:
    """Convert a :class:`PETStudy` or :class:`MRVolume` to a ``SimpleITK.Image``.

    Parameters
    ----------
    study : PETStudy | MRVolume
        The dataclass to convert.
    frame : int | None, default ``None``
        For a :class:`PETStudy`, the time frame to extract.  If ``None``,
        the time-averaged volume (:attr:`PETStudy.mean_volume`) is used.
        Ignored for :class:`MRVolume`.

    Returns
    -------
    sitk.Image
        A 3-D float32 image with the correct origin, spacing, and direction
        cosines pulled from ``study.affine`` so that registration uses
        physical coordinates rather than voxel indices.
    """
    raise NotImplementedError