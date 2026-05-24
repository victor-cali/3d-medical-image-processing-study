"""Project-wide paths and runtime constants for the ``mip`` package.

Importing this module is the *single* place where the on-disk layout and the
default tunables are known.  Nothing else in the package should hardcode a
path or a magic number — everything routes through here, which makes the code
portable (Windows ↔ Linux) and easy to override from a notebook with a one-
liner like ``mip.config.FIGURE_DPI = 200``.

Override the data directory with the ``MIP_DATA_DIR`` environment variable
(useful when running on a remote Linux host where the FORISI study lives at
``/data/FORISI`` instead of ``<repo>/data/FORISI``).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FORISI_DIR",
    "PET_PATH",
    "MR_PATH",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "RANDOM_SEED",
    "GIF_FPS",
    "MIP_N_ANGLES",
    "MIP_FPS",
    "FIGURE_DPI",
    "MEDSAM2_CHECKPOINT",
    "DEVICE",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at <PROJECT_ROOT>/src/mip/config.py, so two parents up is
# the package directory and three is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("MIP_DATA_DIR", PROJECT_ROOT / "data"))
FORISI_DIR: Path = DATA_DIR / "FORISI"

# The two FORISI DICOMs are single multi-frame files with stable names; if
# they ever get renamed, this is the single point of change.
PET_PATH: Path = FORISI_DIR / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"
MR_PATH: Path = FORISI_DIR / "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------

#: Reproducibility seed used by the SimpleITK random sampler and any NumPy
#: RNG inside the package.  Surfaced here so a single override changes every
#: stochastic step at once.
RANDOM_SEED: int = 42

#: Default frame rate for ``animate_three_planes``.  Dynamic PET typically
#: has 20–30 time frames; 8 fps gives a ~3-second loop.
GIF_FPS: int = 8

#: Number of angular samples around the z (SI) axis when generating a
#: rotating MIP.  36 → 10° steps → a smooth 360° loop.
MIP_N_ANGLES: int = 36
MIP_FPS: int = 15

#: Matplotlib figure DPI for static figures.  130 is a sweet spot between
#: notebook responsiveness and LaTeX-figure quality.
FIGURE_DPI: int = 130

#: HuggingFace model id for MedSAM-2.  Verified during Obj 3 implementation.
MEDSAM2_CHECKPOINT: str = "wanglab/MedSAM2"


def _detect_device() -> str:
    """Return ``"cuda"`` if a working CUDA build of torch is importable, else ``"cpu"``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


#: Inference device for MedSAM-2.  Lazy-resolved so importing ``mip.config``
#: never pays the torch import cost when torch isn't installed yet.
DEVICE: str = _detect_device()
