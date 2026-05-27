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
    "MEDSAM2_DIR",
    "MEDSAM2_CHECKPOINT",
    "MEDSAM2_MODEL_CFG",
    "MEDSAM2_IMAGE_SIZE",
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

#: Root of the locally-cloned MedSAM-2 repo.  Default convention: the user
#: clones ``https://github.com/bowang-lab/MedSAM2.git`` *next to* this
#: project, i.e. ``<PROJECT_ROOT.parent>/MedSAM2``.  Override the
#: ``MEDSAM2_DIR`` env var (or this constant) to point somewhere else.
MEDSAM2_DIR: Path = Path(
    os.environ.get("MEDSAM2_DIR", PROJECT_ROOT.parent / "MedSAM2")
)

#: Filesystem path to the MedSAM-2 checkpoint (downloaded by
#: ``MedSAM2/download.sh`` into ``MedSAM2/checkpoints/``).  ``MedSAM2_latest.pt``
#: is the variant the upstream inference script (``medsam2_infer_3D_CT.py``)
#: uses and is what we target here.
MEDSAM2_CHECKPOINT: Path = MEDSAM2_DIR / "checkpoints" / "MedSAM2_latest.pt"

#: Hydra config name for the SAM-2.1 tiny-512 backbone, the one MedSAM-2 was
#: fine-tuned against.  ``build_sam2_video_predictor_npz`` accepts either a
#: relative path within the ``sam2`` package's ``configs/`` directory or a
#: Hydra-style dotted name; this is the form used in the upstream demo.
MEDSAM2_MODEL_CFG: str = "configs/sam2.1_hiera_t512.yaml"

#: Spatial size MedSAM-2 was trained on.  Both H and W of every slice are
#: resized to this before inference, then masks are projected back to the
#: original MR grid by the predictor (using the ``video_height`` /
#: ``video_width`` it stored at ``init_state`` time).
MEDSAM2_IMAGE_SIZE: int = 512


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
