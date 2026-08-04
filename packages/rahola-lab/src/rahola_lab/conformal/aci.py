"""Adaptive conformal inference under distribution shift.

Uses equation (2) of Gibbs & Candès (2021), *Adaptive Conformal Inference Under
Distribution Shift*: alpha[t+1] = alpha[t] + gamma * (alpha - err[t]). The
working level is deliberately not projected; their Lemma 4.1 relies on infinite
and empty prediction sets outside [0, 1], and Proposition 4.1 gives the pathwise
long-run error-frequency bound. Source: https://arxiv.org/abs/2106.00170.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from rahola_lab.conformal.cqr import conformal_quantile

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ACIResult:
    upper_bounds: FloatArray
    working_alpha: FloatArray
    errors: NDArray[np.bool_]


def adaptive_conformal_bounds(
    calibration_scores: FloatArray,
    raw_upper: FloatArray,
    targets: FloatArray,
    *,
    alpha: float,
    gamma: float,
    initial_alpha: float | None = None,
    feedback_delay_steps: int = 0,
) -> ACIResult:
    """Run ACI, optionally delaying updates until forecast outcomes resolve."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    raw = np.asarray(raw_upper, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    if raw.shape != observed.shape or raw.ndim != 1:
        raise ValueError("raw bounds and targets must be matching vectors")
    if not 0.0 < alpha < 1.0 or gamma <= 0:
        raise ValueError("alpha must lie in (0,1) and gamma must be positive")
    if isinstance(feedback_delay_steps, bool) or not isinstance(
        feedback_delay_steps, Integral
    ) or feedback_delay_steps < 0:
        raise ValueError("feedback_delay_steps must be a nonnegative integer")
    working = alpha if initial_alpha is None else initial_alpha
    bounds = np.empty_like(raw)
    history = np.empty_like(raw)
    errors = np.empty(len(raw), dtype=np.bool_)
    for index, (prediction, target) in enumerate(zip(raw, observed, strict=True)):
        if feedback_delay_steps and index >= feedback_delay_steps:
            working += gamma * (
                alpha - float(errors[index - feedback_delay_steps])
            )
        history[index] = working
        bounds[index] = prediction + conformal_quantile(scores, working)
        errors[index] = target > bounds[index]
        if feedback_delay_steps == 0:
            working += gamma * (alpha - float(errors[index]))
    return ACIResult(upper_bounds=bounds, working_alpha=history, errors=errors)
