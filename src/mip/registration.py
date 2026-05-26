"""3-D rigid coregistration of dynamic PET → anatomical MR via SimpleITK.

* **Metric: Mattes mutual information.**  PET and MR intensities live in
  unrelated value domains (tracer counts vs proton density), so any
  metric that assumes a linear relationship (SSD, NCC) is unsuitable.
  Mattes MI is the de-facto standard for multi-modal medical registration.
* **Transform: Versor rigid 3-D (6 DoF).**  The assignment explicitly
  asks for *rigid* registration.  Versor parameterization avoids the
  gimbal lock and quaternion-norm issues you get with Euler angles, and
  SimpleITK's optimizer handles its non-Euclidean parameter space
  natively.
* **Multi-resolution pyramid: 4x / 2x / 1x with σ = 2.0 / 1.0 / 0.0.**
  MI landscapes are bumpy; coarser scales widen the convergence basin
  before fine alignment.  These factors are the SimpleITK textbook
  defaults and have been shown to be robust on brain PET-MR pairs.
* **Optimizer: Regular-step gradient descent.**  Simple, deterministic,
  and parameter-light; ample for 6-DoF rigid problems.  We log the
  metric per iteration so we can plot convergence in the report.
* **Sampling: random 20 %.**  Full-image sampling is slow and unnecessary
  for rigid registration; 20 % balances speed and gradient stability.
* **Initialization: centered geometry alignment.**  Aligns volume
  centroids before optimization so the optimizer starts inside the MI
  basin instead of a flat region of parameter space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import SimpleITK as sitk

__all__ = ["RegistrationResult", "rigid_coregister"]


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """The output of :func:`rigid_coregister`.

    Parameters
    ----------
    transform : sitk.Transform
        The final rigid transform mapping ``moving`` into ``fixed`` space.
    resampled : np.ndarray
        ``moving`` resampled onto ``fixed``'s grid using ``transform``.
        Shape and spacing match ``fixed``.
    metric_values : list[float]
        Per-iteration Mattes MI value (lower is better in SimpleITK's
        sign convention).  Used to plot the convergence curve.
    final_metric : float
        Last entry of ``metric_values``.
    n_iterations : int
        Iterations actually run before the stop condition triggered.
    optimizer_stop_condition : str
        Human-readable reason the optimizer terminated.
    """

    transform: sitk.Transform = field(compare=False)
    resampled: np.ndarray
    metric_values: list[float]
    final_metric: float
    n_iterations: int
    optimizer_stop_condition: str


def rigid_coregister(
    moving: sitk.Image,
    fixed: sitk.Image,
    *,
    n_histogram_bins: int = 50,
    sampling_percentage: float = 0.20,
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
    lr: float = 1.0,
    min_step: float = 1e-4,
    n_iterations: int = 100,
    relaxation_factor: float = 0.5,
    seed: int | None = None,
    metric_callback: Callable[[float, int], None] | None = None,
) -> RegistrationResult:
    """Register ``moving`` onto ``fixed`` with a rigid 3-D transform via Mattes MI.

    Parameters
    ----------
    moving : sitk.Image
        Image to be transformed (e.g. mean-PET).
    fixed : sitk.Image
        Reference image; defines the output grid (e.g. MR).
    n_histogram_bins : int, default 50
        Mattes MI histogram bin count.  50 is the SimpleITK default; 64 is
        a reasonable bump for high-bit-depth inputs.
    sampling_percentage : float, default 0.20
        Fraction of voxels randomly sampled per iteration to estimate the
        metric and its gradient.
    shrink_factors : tuple[int, ...], default ``(4, 2, 1)``
        Multi-resolution downsampling factors per level (coarse → fine).
    smoothing_sigmas : tuple[float, ...], default ``(2.0, 1.0, 0.0)``
        Gaussian σ (voxels) per level.  Same length as ``shrink_factors``.
    lr : float, default 1.0
        Initial learning rate for the regular-step gradient descent.
    min_step : float, default 1e-4
        Stop criterion: step length below this value terminates the
        optimizer at the current resolution level.
    n_iterations : int, default 100
        Maximum iterations per resolution level.
    relaxation_factor : float, default 0.5
        Factor by which the step length is reduced when the gradient
        changes sign.
    seed : int | None, default ``None``
        Random sampler seed; defaults to :data:`mip.config.RANDOM_SEED`
        when ``None``.
    metric_callback : callable | None, default ``None``
        Optional ``(metric_value, iteration_index) -> None`` callback,
        invoked from inside the SimpleITK iteration command.  Useful when
        the caller wants to display a live progress bar instead of (or in
        addition to) the recorded ``metric_values`` list.

    Returns
    -------
    RegistrationResult
        See :class:`RegistrationResult`.

    Notes
    -----
    Pipeline implemented here:

    1. ``CenteredTransformInitializer(fixed, moving, VersorRigid3DTransform, GEOMETRY)``
       so the optimizer starts at aligned centroids.
    2. ``ImageRegistrationMethod`` with Mattes MI, random sampling at
       ``sampling_percentage``, and the Regular Step Gradient Descent
       optimizer.
    3. Multi-resolution via
       ``SetShrinkFactorsPerLevel`` / ``SetSmoothingSigmasPerLevel``.
    4. ``AddCommand(sitkIterationEvent, ...)`` to capture per-iteration MI.
    5. ``Execute`` then resample with a linear interpolator onto ``fixed``.
    """
    raise NotImplementedError