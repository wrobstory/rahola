"""Frozen statistics and scoring helpers for the final F1 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from rahola.config import SimulationConfig
from rahola.dataset import SimulationDataset, TangentRollout
from rahola.simulate import simulate_tangent_batch
from rahola_lab.detectors import NormalizationMode, extract_detector_windows
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.forecast import fit_piecewise_linear_restoring

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FrozenLogistic:
    mean: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    intercept: float

    def predict(self, features: FloatArray) -> FloatArray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return expit(standardized @ self.coefficients + self.intercept)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FrozenLogistic:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
        )


@dataclass(frozen=True)
class StatisticRows:
    trajectory_indices: NDArray[np.int64]
    end_times_s: FloatArray
    labels: NDArray[np.int8]
    features: dict[str, FloatArray]
    scores: dict[str, FloatArray]


def fit_logistic(features: FloatArray, labels: NDArray[np.int8]) -> FrozenLogistic:
    """Fit one fixed two-feature logistic regression without a model grid."""
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    finite = np.all(np.isfinite(values), axis=1)
    values, targets = values[finite], targets[finite]
    if values.shape[1] != 2 or len(np.unique(targets)) != 2:
        raise ValueError("logistic fit requires two features and both labels")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1e-12)
    standardized = (values - mean) / scale

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        linear = standardized @ parameters[:2] + parameters[2]
        loss = np.sum(np.logaddexp(0.0, linear) - targets * linear)
        residual = expit(linear) - targets
        gradient = np.r_[standardized.T @ residual, np.sum(residual)]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 500},
    )
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return FrozenLogistic(
        mean=mean,
        scale=scale,
        coefficients=np.asarray(result.x[:2], dtype=np.float64),
        intercept=float(result.x[2]),
    )


def reproduce_tangent(
    dataset: SimulationDataset, *, require_stored_match: bool = True
) -> TangentRollout:
    """Re-integrate configured seeds and optionally prove stored-motion identity."""
    rollout = simulate_tangent_batch(SimulationConfig.from_dict(dataset.config), dataset.seeds)
    if require_stored_match and not np.array_equal(
        dataset.angle_rad, rollout.dataset.angle_rad, equal_nan=True
    ):
        raise AssertionError("tangent rollout did not reproduce stored angle trajectory bitwise")
    if require_stored_match and not np.array_equal(
        dataset.rate_rad_s, rollout.dataset.rate_rad_s, equal_nan=True
    ):
        raise AssertionError("tangent rollout did not reproduce stored rate trajectory bitwise")
    return rollout


def _window_rows(
    dataset: SimulationDataset,
    *,
    first_endpoint_s: float | None = None,
    last_endpoint_s: float | None = None,
) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.int8]]:
    windows = extract_detector_windows(
        dataset,
        stride_s=10.0,
        normalization_mode=NormalizationMode.PHYSICAL,
    )
    selected = np.ones(len(windows.labels), dtype=bool)
    if first_endpoint_s is not None:
        selected &= windows.end_times_s >= first_endpoint_s
    if last_endpoint_s is not None:
        selected &= windows.end_times_s <= last_endpoint_s
    return (
        windows.trajectory_indices[selected],
        windows.end_times_s[selected],
        windows.labels[selected],
    )


def _saddle_energies(
    config: SimulationConfig, stiffness: FloatArray
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(stiffness, dtype=np.float64)
    if config.bias_moment == 0.0:
        if config.quintic_coefficient == 0.0:
            saddle = 1.0
        else:
            roots = np.roots([config.quintic_coefficient, -1.0, 1.0])
            positive = [
                math.sqrt(float(root.real))
                for root in roots
                if abs(root.imag) < 1e-10 and root.real > 0
            ]
            saddle = min(positive)
        potential = (
            0.5 * saddle**2
            - 0.25 * saddle**4
            + config.quintic_coefficient * saddle**6 / 6.0
        )
        energy = x * potential
        return energy, energy
    if not np.all(x == x.flat[0]):
        raise ValueError("biased time-varying stiffness is outside the frozen F1 families")
    coefficients = [
        float(x.flat[0]) * config.quintic_coefficient,
        0.0,
        -float(x.flat[0]),
        0.0,
        float(x.flat[0]),
        -config.bias_moment,
    ]
    roots = sorted(
        float(root.real) for root in np.roots(coefficients) if abs(root.imag) < 1e-9
    )
    stable = min(
        (
            root
            for root in roots
            if float(x.flat[0])
            * (1.0 - 3.0 * root**2 + 5.0 * config.quintic_coefficient * root**4)
            > 0.0
        ),
        key=lambda root: abs(root - config.initial_angle_rad / config.escape_angle_rad),
    )
    negative = max(root for root in roots if root < stable)
    positive = min(root for root in roots if root > stable)

    def potential(position: float) -> float:
        shape = (
            0.5 * position**2
            - 0.25 * position**4
            + config.quintic_coefficient * position**6 / 6.0
        )
        return float(x.flat[0]) * shape - config.bias_moment * position

    return np.full_like(x, potential(positive)), np.full_like(x, potential(negative))


def _raw_statistics(
    rollout: TangentRollout, *, setting: str
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    dataset = rollout.dataset
    config = SimulationConfig.from_dict(dataset.config)
    if setting not in {"oracle", "operational"}:
        raise ValueError("setting must be oracle or operational")
    dt_s = float(np.median(np.diff(dataset.time_s)))
    dt_tau = config.omega_n_rad_s * dt_s
    x = dataset.angle_rad / config.escape_angle_rad
    velocity = dataset.rate_rad_s / (config.escape_angle_rad * config.omega_n_rad_s)
    stiffness = (
        rollout.effective_stiffness
        if setting == "oracle"
        else np.ones_like(rollout.effective_stiffness)
    )

    fit = fit_piecewise_linear_restoring(dataset.config)
    margin = fit.safety_margin(dataset.angle_rad, dataset.rate_rad_s)
    period_samples = round(config.natural_period_s / dt_s)
    margin_rate = np.full_like(margin, np.nan)
    margin_rate[:, period_samples:] = (
        margin[:, period_samples:] - margin[:, :-period_samples]
    ) / config.natural_period_s
    closure = -margin_rate
    time_to_closure = np.where(closure > 0.0, margin / np.maximum(closure, 1e-12), np.nan)

    shape = x - x**3 + config.quintic_coefficient * x**5
    potential_shape = (
        0.5 * x**2 - 0.25 * x**4 + config.quintic_coefficient * x**6 / 6.0
    )
    potential = stiffness * potential_shape - config.bias_moment * x
    energy = 0.5 * velocity**2 + potential
    positive_saddle, negative_saddle = _saddle_energies(config, stiffness)
    reserve = np.minimum(positive_saddle - energy, negative_saddle - energy)
    selected_saddle = np.where(
        positive_saddle - energy <= negative_saddle - energy,
        positive_saddle,
        negative_saddle,
    )
    acceleration = np.full_like(velocity, np.nan)
    acceleration[:, 1:] = np.diff(velocity, axis=1) / dt_tau
    stiffness_rate = np.zeros_like(stiffness)
    stiffness_rate[:, 1:] = np.diff(stiffness, axis=1) / dt_tau
    force_estimate = (
        acceleration
        + 2.0 * config.damping_ratio * velocity
        + config.quadratic_damping * velocity * np.abs(velocity)
        + stiffness * shape
        - config.bias_moment
    )
    energy_rate_tau = (
        velocity * force_estimate
        - 2.0 * config.damping_ratio * velocity**2
        - config.quadratic_damping * np.abs(velocity) ** 3
        + stiffness_rate * potential_shape
    )
    saddle_rate_tau = np.full_like(selected_saddle, np.nan)
    saddle_rate_tau[:, 1:] = np.diff(selected_saddle, axis=1) / dt_tau
    depletion = config.omega_n_rad_s * (energy_rate_tau - saddle_rate_tau)

    positive_critical = fit.positive.growth_rate_s * np.maximum(
        fit.positive.vanishing_distance_rad
        - (dataset.angle_rad - fit.equilibrium_angle_rad),
        0.0,
    )
    negative_critical = fit.negative.growth_rate_s * np.maximum(
        fit.negative.vanishing_distance_rad
        + (dataset.angle_rad - fit.equilibrium_angle_rad),
        0.0,
    )
    positive_side = positive_critical - dataset.rate_rad_s <= (
        negative_critical + dataset.rate_rad_s
    )
    normal_x = np.where(
        positive_side,
        fit.positive.growth_rate_s / config.omega_n_rad_s,
        fit.negative.growth_rate_s / config.omega_n_rad_s,
    )
    normal_scale = np.sqrt(normal_x**2 + 1.0)
    n0, n1 = normal_x / normal_scale, 1.0 / normal_scale
    restoring_slope = (
        np.ones_like(x)
        if config.linear_restoring
        else 1.0 - 3.0 * x**2 + 5.0 * config.quintic_coefficient * x**4
    )
    j10 = -stiffness * restoring_slope
    j11 = -2.0 * config.damping_ratio - 2.0 * config.quadratic_damping * np.abs(velocity)
    rho = 2.0 * n0 * n1 * (1.0 + j10) / 2.0 + n1**2 * j11

    features = {
        "margin": np.stack((margin, margin_rate), axis=-1),
        "energy": np.stack((reserve, -depletion), axis=-1),
    }
    scores = {
        "S1_margin": -margin,
        "S2_margin_closure": closure,
        "S3_time_to_closure": -time_to_closure,
        "S4_energy_reserve": -reserve,
        "S4_energy_depletion": depletion,
        "S7_instantaneous_normal_strain": rho,
    }
    return features, scores


def statistic_rows(
    rollout: TangentRollout,
    *,
    setting: str,
    first_endpoint_s: float | None = None,
    last_endpoint_s: float | None = None,
    logistic_models: dict[str, FrozenLogistic] | None = None,
) -> StatisticRows:
    trajectories, times, labels = _window_rows(
        rollout.dataset,
        first_endpoint_s=first_endpoint_s,
        last_endpoint_s=last_endpoint_s,
    )
    sample_indices = np.searchsorted(rollout.dataset.time_s, times)
    raw_features, raw_scores = _raw_statistics(rollout, setting=setting)
    features = {
        name: values[trajectories, sample_indices] for name, values in raw_features.items()
    }
    scores = {
        name: values[trajectories, sample_indices] for name, values in raw_scores.items()
    }
    if logistic_models:
        scores["S5_margin_level_rate"] = logistic_models["margin"].predict(features["margin"])
        scores["S5_energy_level_rate"] = logistic_models["energy"].predict(features["energy"])
    return StatisticRows(trajectories, times, labels, features, scores)


def finite_time_score(
    rollout: TangentRollout,
    rows: StatisticRows,
    *,
    periods: int,
    escape_directed: bool,
) -> FloatArray:
    config = SimulationConfig.from_dict(rollout.dataset.config)
    dt_s = float(np.median(np.diff(rollout.dataset.time_s)))
    intervals = round(periods * config.natural_period_s / dt_s)
    duration_tau = periods * 2.0 * math.pi
    values = np.full(len(rows.labels), np.nan, dtype=np.float64)
    fit = fit_piecewise_linear_restoring(rollout.dataset.config)
    for row, (trajectory, time_s) in enumerate(
        zip(rows.trajectory_indices, rows.end_times_s, strict=True)
    ):
        start = int(np.searchsorted(rollout.dataset.time_s, time_s))
        end = start + intervals
        capsize = rollout.dataset.t_capsize_s[trajectory]
        if end >= len(rollout.dataset.time_s) or (
            np.isfinite(capsize) and capsize <= rollout.dataset.time_s[end]
        ):
            continue
        transition = np.eye(2)
        for local in rollout.transition_matrices[trajectory, start:end]:
            transition = local @ transition
        if escape_directed:
            angle = rollout.dataset.angle_rad[trajectory, start]
            rate = rollout.dataset.rate_rad_s[trajectory, start]
            displacement = angle - fit.equilibrium_angle_rad
            positive_margin = (
                fit.positive.growth_rate_s
                * max(fit.positive.vanishing_distance_rad - displacement, 0.0)
                - rate
            )
            negative_margin = (
                fit.negative.growth_rate_s
                * max(fit.negative.vanishing_distance_rad + displacement, 0.0)
                + rate
            )
            growth = (
                fit.positive.growth_rate_s
                if positive_margin <= negative_margin
                else fit.negative.growth_rate_s
            )
            normal = np.array([growth / config.omega_n_rad_s, 1.0])
            normal /= np.linalg.norm(normal)
            gain = np.linalg.norm(normal @ transition)
        else:
            gain = np.linalg.svd(transition, compute_uv=False)[0]
        values[row] = math.log(max(float(gain), np.finfo(float).tiny)) / duration_tau
    return values


def trajectory_scores(
    dataset: SimulationDataset, rows: StatisticRows, values: FloatArray
) -> list[TrajectoryScores]:
    output = []
    for trajectory in range(dataset.batch_size):
        selected = (rows.trajectory_indices == trajectory) & np.isfinite(values)
        times = rows.end_times_s[selected]
        capsize = float(dataset.t_capsize_s[trajectory])
        output.append(
            TrajectoryScores(
                times_s=times,
                scores=np.asarray(values[selected], dtype=np.float64),
                record_end_s=float(times[-1]) if len(times) else float(dataset.time_s[0]),
                t_capsize_s=capsize if np.isfinite(capsize) else None,
                record_start_s=float(times[0]) if len(times) else float(dataset.time_s[0]),
            )
        )
    return output
