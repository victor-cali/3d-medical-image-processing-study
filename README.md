# 3D Medical Image Processing Study

End-to-end pipeline for a single brain PET–MR study (FORISI 2016-09-26):
load a dynamic PET, rigidly coregister it to a 3-D T1 MR, and segment
the tumour on the MR with MedSAM-2 driven by a manual bounding-box prompt.

> **Course:** UIB Master in Intelligent Systems · 2025–2026

> **Author:** V. A. Canales Lima

> **Report:** [`reports/report.tex`](reports/report.tex) (5 pages, compile with `pdflatex` / Tectonic / Overleaf)

The original assignment brief is preserved in [`docs/assignment.md`](docs/assignment.md).

---

## Objectives

1. **DICOM loading & visualisation.** Read both DICOMs with PyDicom,
   reshape the dynamic PET's flat slice stack into a 4-D `(T, Z, Y, X)`
   array, and render the last frame, the time average, three median
   planes and an animated GIF.
2. **Rigid PET → MR coregistration.** Optimise a 6-DoF Versor rigid
   transform with Mattes mutual information, plot the convergence
   curve, and animate a 36-angle rotating MIP (reference,
   coregistered, α-fused).
3. **AI-prompted tumour segmentation.** Pick one 2-D bounding box or
   centroid on one axial slice via `ipywidgets`, then run MedSAM-2's
   video predictor to propagate the mask through the whole MR volume.
   Assess by predicted volume / z-extent / bbox-fill ratio.

---

## Repository layout

```
3d-medical-image-processing-study/
├─ data/FORISI/                       ← the two DICOMs (not committed)
├─ src/mip/                           ← the Python package
│  ├─ io.py            DICOM loaders + dataclasses
│  ├─ viz.py           static mosaics, three-planes, overlay
│  ├─ animation.py     animate_three_planes, rotating_mip
│  ├─ registration.py  rigid_coregister + RegistrationResult
│  ├─ segmentation.py  segment_with_medsam2 (MedSAM-2 wrapper)
│  ├─ metrics.py       volume_mm3
│  └─ config.py        paths, defaults, MedSAM-2 checkpoint location
├─ notebooks/
│  └─ 00_presentation.ipynb           ← run-everything notebook + RISE deck
└─ reports/
   ├─ report.tex                     
   └─ figures/                        ← PNG / GIF artefacts
```

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/victor-cali/3d-medical-image-processing-study.git
cd 3d-medical-image-processing-study
```

### 2. Drop the dataset in

The two DICOM files (provided by the professor) go under
`data/FORISI/` keeping their original names:

```
data/FORISI/02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916
data/FORISI/15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916
```

Override the directory with `MIP_DATA_DIR=...` if you keep the data
elsewhere (e.g. `/data/FORISI`).

### 3. Install the pixi environment

```bash
pixi install
```

This resolves `pyproject.toml`'s `[tool.pixi.pypi-dependencies]`
(MedSAM-2 included — see step 4 below for the prerequisite) and
creates `.pixi/envs/default/`.

### 4. Install MedSAM-2 (required for Objective 3 only)

MedSAM-2 must be cloned **next to** this project, not inside it.
Layout after cloning:

```
<parent>/
├─ 3d-medical-image-processing-study/   ← this repo
└─ MedSAM2/                              ← the MedSAM-2 repo
```

```bash
# from <parent>/
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
bash download.sh         # ~2 GB, fetches MedSAM2_latest.pt and friends
cd -
```

`pyproject.toml` already lists MedSAM-2 as an editable PyPI
dependency pointing at `../MedSAM2`, so `pixi install` picks it up
automatically once the directory exists.  GPU prerequisites:

- A CUDA 12.4-capable driver (`nvidia-smi` reports "CUDA Version: 12.4"
  or higher; the bundled PyTorch wheel index is `cu124`).
- ~4 GB VRAM during inference (the `sam2.1_hiera_t512` variant).

Set `MEDSAM2_DIR=/some/other/path` if you cloned MedSAM-2 somewhere
other than `../MedSAM2`.

### 5. Run the notebook

```bash
pixi run jupyter lab notebooks/00_presentation.ipynb
```

Open the notebook, run all cells top-to-bottom.  Two flags at the top
of the setup section control freshness:

| Flag | Default | Effect |
|---|---|---|
| `RUN_REGISTRATION` | `False` | recompute Obj 2 coregistration (~15 s) |
| `RUN_MIPS`         | `False` | regenerate the three rotating MIPs (~60 s) |
| `RUN_SEGMENTATION` | `False` | call MedSAM-2 on the MR (Obj 3 cell) |

Default `False` reloads cached `.npy` / `.png` / `.gif` artefacts so
the whole notebook plays through in under 30 seconds.  Flip a flag to
`True` once on a fresh checkout to populate the caches, then back to
`False` for normal use.

### 6. Present as slides (optional)

`jupyterlab-rise` is pre-installed.  Inside JupyterLab, after running
all cells, click the **RISE Slideshow** button (or hit `Alt-R`) to
enter the presentation mode.  Use space / shift-space to walk slides.

### 7. Build the report

```bash
cd reports
pdflatex report.tex && pdflatex report.tex   # twice for cross-refs
```

Or open `reports/report.tex` in Overleaf and let it compile.

---

## Outputs

After a full notebook run with all `RUN_*` flags set to `True` once,
the following artefacts live under `reports/figures/` (all committed):

| File | What it shows |
|---|---|
| `pet_last_frame.png` | mosaic of all 47 axial slices, last PET frame |
| `pet_mean_frame.png` | same mosaic, time-averaged PET |
| `pet_mean_three_planes.png` | axial / coronal / sagittal of the mean PET |
| `pet_three_planes.gif` | 36-frame animation of the three median planes |
| `registration_metric_curve.png` | Mattes-MI convergence across the 3 pyramid levels |
| `coregistration_axial_panels.png` | MR \| PET \| α-fused at the middle slice |
| `rotating_mip_reference.gif` | 360° MR-alone MIP |
| `rotating_mip_coregistered.gif` | 360° PET-coregistered MIP |
| `rotating_mip_fused.gif` | 360° α-fused MR + PET MIP |
| `tumor_prompt_on_mr.png` | user-picked bbox over the MR (Obj 3) |
| `segmentation_overlay.png` | MedSAM-2 mask around the prompt slice |

The cached intermediate `.npy` files (coregistered PET, metric trace,
segmentation mask) live next to the figures and are gitignored.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pixi install` fails resolving MedSAM-2 | Did you clone `MedSAM2/` next to the project? `ls ../MedSAM2/setup.py` should exist. |
| `segment_with_medsam2` → `FileNotFoundError: MedSAM2_latest.pt` | Run `bash download.sh` inside `../MedSAM2/`. |
| `import torch` reports CPU-only | Check `nvidia-smi`; update the driver to a CUDA 12.4-capable one or downgrade the PyTorch wheel index in `pyproject.toml` to `cu121` / `cu118`. |
| RISE button missing in JupyterLab | Restart the lab server after `pixi install` so `jupyterlab-rise` registers its extension. |
| Notebook can't find the data | Set the `MIP_DATA_DIR` env var to wherever the FORISI directory lives. |

---

## License & acknowledgements

- **MedSAM-2** by Wang Lab (Toronto), used under its repository
  license: <https://github.com/bowang-lab/MedSAM2>.
- **SAM 2** by Meta AI, used transitively through MedSAM-2:
  <https://github.com/facebookresearch/sam2>.
- **SimpleITK** for the registration backbone.
- Dataset: FORISI series provided by the course instructor for
  educational use; not redistributed in this repository.
