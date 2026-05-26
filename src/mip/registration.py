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

import numpy as np
import SimpleITK as sitk


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
