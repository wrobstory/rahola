"""Online conformal adapters for general distribution shifts.

DtACI implements deterministic Algorithm 2 of Gibbs & Candès (2024),
https://jmlr.org/papers/v25/22-1218.html. Sliding score recalibration is an
explicitly nonexchangeable recent-weighting convention motivated by Foygel
Barber et al. (2023), https://doi.org/10.1214/23-AOS2276; it does not inherit
ordinary split-conformal coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from rahola_lab.conformal.aci import ACIResult
from rahola_lab.conformal.cqr import conformal_quantile


@dataclass(frozen=True)
class DtACIResult:
    upper_bounds: np.ndarray
    working_alpha: np.ndarray
    errors: np.ndarray
    final_expert_weights: np.ndarray


def _empirical_beta(scores: np.ndarray, observed_score: float) -> float:
    """Supremum level whose inflated empirical upper set contains the score."""
    position = int(np.searchsorted(np.sort(scores), observed_score, side="left"))
    return 1.0 - position / (len(scores) + 1.0)


def _pinball(beta: float, theta: np.ndarray, alpha: float) -> np.ndarray:
    difference = beta - theta
    return alpha * difference - np.minimum(0.0, difference)


def dynamically_tuned_aci_bounds(
    calibration_scores: np.ndarray,
    raw_upper: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    gamma_experts: tuple[float, ...],
    target_interval: int = 500,
    feedback_delay_steps: int = 0,
) -> DtACIResult:
    """Run deterministic DtACI Algorithm 2 with the paper's fixed heuristic."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    raw = np.asarray(raw_upper, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    gammas = np.asarray(gamma_experts, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores must be a non-empty finite vector")
    if raw.shape != observed.shape or raw.ndim != 1:
        raise ValueError("raw bounds and targets must be matching vectors")
    if not 0.0 < alpha < 1.0 or np.any(gammas <= 0.0) or target_interval < 1:
        raise ValueError("invalid alpha, expert grid, or target interval")
    if isinstance(feedback_delay_steps, bool) or not isinstance(
        feedback_delay_steps, Integral
    ) or feedback_delay_steps < 0:
        raise ValueError("feedback_delay_steps must be a nonnegative integer")
    expert_alpha = np.full(len(gammas), alpha, dtype=np.float64)
    weights = np.ones(len(gammas), dtype=np.float64)
    sigma = 1.0 / (2.0 * target_interval)
    expected_loss_square = (1.0 - alpha) ** 2 * alpha**2 / 3.0
    eta = math.sqrt(
        (math.log(2.0 * len(gammas) * target_interval) + 1.0)
        / (target_interval * expected_loss_square)
    )
    bounds = np.empty_like(raw)
    history = np.empty_like(raw)
    errors = np.empty(len(raw), dtype=np.bool_)
    pending_losses = np.empty((len(raw), len(gammas)), dtype=np.float64)
    pending_expert_errors = np.empty((len(raw), len(gammas)), dtype=np.bool_)

    def apply_feedback(loss: np.ndarray, expert_errors: np.ndarray) -> None:
        nonlocal weights, expert_alpha
        shifted_loss = loss - np.min(loss)
        reweighted = weights * np.exp(-eta * shifted_loss)
        total = reweighted.sum()
        weights = (1.0 - sigma) * reweighted + total * sigma / len(gammas)
        expert_alpha += gammas * (alpha - expert_errors.astype(np.float64))

    for index, (prediction, target) in enumerate(zip(raw, observed, strict=True)):
        if feedback_delay_steps and index >= feedback_delay_steps:
            resolved = index - feedback_delay_steps
            apply_feedback(pending_losses[resolved], pending_expert_errors[resolved])
        probabilities = weights / weights.sum()
        working_alpha = float(probabilities @ expert_alpha)
        history[index] = working_alpha
        bounds[index] = prediction + conformal_quantile(scores, working_alpha)
        errors[index] = target > bounds[index]
        observed_score = target - prediction
        beta = _empirical_beta(scores, observed_score)
        loss = _pinball(beta, expert_alpha, alpha)
        expert_bounds = np.array(
            [prediction + conformal_quantile(scores, level) for level in expert_alpha]
        )
        expert_errors = target > expert_bounds
        pending_losses[index] = loss
        pending_expert_errors[index] = expert_errors
        if feedback_delay_steps == 0:
            apply_feedback(loss, expert_errors)
    return DtACIResult(bounds, history, errors, weights / weights.sum())


def sliding_recalibrated_aci_bounds(
    calibration_scores: np.ndarray,
    raw_upper: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    window_size: int,
    feedback_delay_steps: int = 0,
) -> ACIResult:
    """Run scalar ACI while replacing its score distribution with recent scores."""
    initial = np.asarray(calibration_scores, dtype=np.float64)
    raw = np.asarray(raw_upper, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    if raw.shape != observed.shape or raw.ndim != 1:
        raise ValueError("raw bounds and targets must be matching vectors")
    if window_size < 1 or not 0.0 < alpha < 1.0 or gamma <= 0.0:
        raise ValueError("window, alpha, and gamma must be positive and valid")
    if isinstance(feedback_delay_steps, bool) or not isinstance(
        feedback_delay_steps, Integral
    ) or feedback_delay_steps < 0:
        raise ValueError("feedback_delay_steps must be a nonnegative integer")
    recent = list(initial[-window_size:])
    if not recent or not np.all(np.isfinite(recent)):
        raise ValueError("calibration scores must be non-empty and finite")
    working = alpha
    bounds = np.empty_like(raw)
    history = np.empty_like(raw)
    errors = np.empty(len(raw), dtype=np.bool_)
    for index, (prediction, target) in enumerate(zip(raw, observed, strict=True)):
        if feedback_delay_steps and index >= feedback_delay_steps:
            resolved = index - feedback_delay_steps
            recent.append(float(observed[resolved] - raw[resolved]))
            if len(recent) > window_size:
                recent.pop(0)
            working += gamma * (alpha - float(errors[resolved]))
        history[index] = working
        bounds[index] = prediction + conformal_quantile(np.asarray(recent), working)
        errors[index] = target > bounds[index]
        if feedback_delay_steps == 0:
            recent.append(float(target - prediction))
            if len(recent) > window_size:
                recent.pop(0)
            working += gamma * (alpha - float(errors[index]))
    return ACIResult(bounds, history, errors)
