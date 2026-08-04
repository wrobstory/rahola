"""Shared calibration-frozen plumbing for the U1 experiment family."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    EWS_HORIZON_PERIODS,
    TRAJECTORY_BOOTSTRAP_REPLICATES,
    TRAJECTORY_BOOTSTRAP_SEED,
    U1_HAZARD_KERNEL_PERIODS,
    U1_PARAMETRIC_BOOTSTRAP_SEED,
    U1_RELIABILITY_BINS,
    U1_TRACKING_MAX_LAG_PERIODS,
    SeedBlock,
)
from rahola_lab.evaluation import (
    TrajectoryScores,
    clopper_pearson_interval,
    trajectory_block_bootstrap,
)
from rahola_lab.experiments.detector_common import detector_risk_end_s
from rahola_lab.forecast import DangerMarginFit, fit_piecewise_linear_restoring
from rahola_lab.splittime import (
    GammaRatePrior,
    RateTrajectory,
    SplitTimeConfig,
    decluster_crossings,
    detect_crossings,
    estimate_rate_trajectory,
    exponential_rate_mle,
    roll_decorrelation_time,
)


@dataclass(frozen=True)
class ScoredRateTrajectory:
    seed: int
    capsized: bool
    t_capsize_s: float | None
    exposure_end_s: float
    rate: RateTrajectory

    @property
    def average_rate_per_hour(self) -> float:
        hours = self.exposure_end_s / 3_600.0
        return self.rate.integrated_count / hours if hours > 0.0 else 0.0

    @property
    def predicted_capsize_probability(self) -> float:
        return float(-np.expm1(-self.rate.integrated_count))


def campaign_family(name: str) -> str:
    family = name.split("_", 1)[0]
    if family not in {"softening", "parametric", "biased"}:
        raise ValueError(f"unknown U1 campaign family: {name}")
    return family


def campaign_path(data_root: Path, name: str) -> Path:
    return data_root / name


def load_split(data_root: Path, name: str, block: SeedBlock) -> SimulationDataset:
    return load_campaign_split(campaign_path(data_root, name), block)


def restoring_fit(dataset: SimulationDataset) -> DangerMarginFit:
    return fit_piecewise_linear_restoring(dataset.config)


def terminal_severities(
    dataset: SimulationDataset, *, fit: DangerMarginFit | None = None
) -> NDArray[np.float64]:
    """Pool terminal, declustered severities without reading outcome labels."""
    selected_fit = fit or restoring_fit(dataset)
    sample_interval_s = float(np.median(np.diff(dataset.time_s)))
    pooled: list[float] = []
    for angle, rate in zip(dataset.angle_rad, dataset.rate_rad_s, strict=True):
        finite = np.isfinite(angle) & np.isfinite(rate)
        stop = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(angle)
        if stop < 3:
            continue
        decorrelation = roll_decorrelation_time(angle[:stop], sample_interval_s)
        retained = decluster_crossings(
            detect_crossings(dataset.time_s, angle, rate, selected_fit), decorrelation
        )
        pooled.extend(event.severity_u for event in retained)
    values = np.asarray(pooled, dtype=np.float64)
    if not len(values):
        raise ValueError("campaign produced no terminal declustered crossings")
    return values


def adaptive_threshold_fit(
    dataset: SimulationDataset,
) -> tuple[DangerMarginFit, dict[str, float]]:
    """Choose one calibration-only threshold scale targeting 7--10/30 min."""
    base = restoring_fit(dataset)
    sample_interval_s = float(np.median(np.diff(dataset.time_s)))
    motion = []
    for angle, rate in zip(dataset.angle_rad, dataset.rate_rad_s, strict=True):
        finite = np.isfinite(angle) & np.isfinite(rate)
        stop = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(angle)
        if stop < 3:
            continue
        motion.append(
            (
                angle,
                rate,
                roll_decorrelation_time(angle[:stop], sample_interval_s),
            )
        )
    target = 8.5
    rows = []
    for fraction in np.linspace(0.0, 1.0, 21):
        sides = {}
        for name, side in (("positive", base.positive), ("negative", base.negative)):
            distance = side.threshold_distance_rad + fraction * (
                0.95 * side.vanishing_distance_rad - side.threshold_distance_rad
            )
            sides[name] = replace(
                side,
                threshold_angle_rad=base.equilibrium_angle_rad + side.direction * distance,
                threshold_distance_rad=distance,
            )
        candidate = replace(
            base,
            positive=sides["positive"],
            negative=sides["negative"],
        )
        counts = [
            len(
                decluster_crossings(
                    detect_crossings(dataset.time_s, angle, rate, candidate), decorrelation
                )
            )
            for angle, rate, decorrelation in motion
        ]
        per_30_minutes = float(np.mean(counts)) * 1_800.0 / float(dataset.time_s[-1])
        rows.append((abs(per_30_minutes - target), fraction, per_30_minutes, candidate))
    _, fraction, rate, selected = min(rows, key=lambda row: (row[0], row[1]))
    return selected, {
        "threshold_interpolation_fraction": float(fraction),
        "declustered_crossings_per_30_minutes": float(rate),
        "target_lower": 7.0,
        "target_upper": 10.0,
    }


def calibration_prior_means(
    datasets: dict[str, SimulationDataset], quantiles: tuple[float, ...]
) -> dict[str, dict[str, float]]:
    by_family: dict[str, list[NDArray[np.float64]]] = {
        "softening": [],
        "parametric": [],
        "biased": [],
    }
    for name, dataset in datasets.items():
        by_family[campaign_family(name)].append(terminal_severities(dataset))
    return {
        str(quantile): {
            family: exponential_rate_mle(np.concatenate(parts), quantile=quantile)
            for family, parts in by_family.items()
        }
        for quantile in quantiles
    }


def calibration_tail_priors(
    datasets: dict[str, SimulationDataset], quantiles: tuple[float, ...]
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit pooled family thresholds and exponential-rate prior means."""
    by_family: dict[str, list[NDArray[np.float64]]] = {
        "softening": [],
        "parametric": [],
        "biased": [],
    }
    for name, dataset in datasets.items():
        by_family[campaign_family(name)].append(terminal_severities(dataset))
    output: dict[str, dict[str, dict[str, float]]] = {}
    for quantile in quantiles:
        family_rows = {}
        for family, parts in by_family.items():
            values = np.concatenate(parts)
            threshold = min(
                float(np.quantile(values, quantile)),
                float(np.nextafter(1.0, -np.inf)),
            )
            family_rows[family] = {
                "mean_rate": exponential_rate_mle(values, quantile=quantile),
                "threshold_w": threshold,
                "exceedance_probability": float(np.mean(values > threshold)),
                "retained_crossings": len(values),
            }
        output[str(quantile)] = family_rows
    return output


def score_dataset(
    dataset: SimulationDataset,
    *,
    prior_mean: float,
    prior_strength: float,
    config: SplitTimeConfig,
    fit: DangerMarginFit | None = None,
    critical_rate_scales: list[dict[int, NDArray[np.float64]]] | None = None,
    prior_threshold_w: float | None = None,
    prior_exceedance_probability: float | None = None,
) -> list[ScoredRateTrajectory]:
    selected_fit = fit or restoring_fit(dataset)
    prior = GammaRatePrior.from_mean(
        prior_mean,
        prior_strength,
        threshold_w=prior_threshold_w,
        exceedance_probability=prior_exceedance_probability,
    )
    scored: list[ScoredRateTrajectory] = []
    for index in range(dataset.batch_size):
        trajectory_config = replace(
            config,
            bootstrap_seed=U1_PARAMETRIC_BOOTSTRAP_SEED + int(dataset.seeds[index]),
        )
        rate = estimate_rate_trajectory(
            dataset.time_s,
            dataset.angle_rad[index],
            dataset.rate_rad_s[index],
            selected_fit,
            prior=prior,
            config=trajectory_config,
            critical_rate_scales=(
                None if critical_rate_scales is None else critical_rate_scales[index]
            ),
        )
        capsize_time = float(dataset.t_capsize_s[index])
        capsize = capsize_time if np.isfinite(capsize_time) else None
        finite = np.isfinite(dataset.angle_rad[index]) & np.isfinite(dataset.rate_rad_s[index])
        stop = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(dataset.time_s)
        exposure_end = float(dataset.time_s[max(0, stop - 1)])
        scored.append(
            ScoredRateTrajectory(
                seed=int(dataset.seeds[index]),
                capsized=bool(dataset.capsized[index]),
                t_capsize_s=capsize,
                exposure_end_s=exposure_end,
                rate=rate,
            )
        )
    return scored


def campaign_count_summary(
    scores: list[ScoredRateTrajectory], *, absorbing_events: bool = False
) -> dict[str, object]:
    point_contributions = np.asarray(
        [item.rate.integrated_count for item in scores], dtype=np.float64
    )
    draw_contributions = np.stack([item.rate.integrated_count_draws for item in scores])
    if absorbing_events:
        point_contributions = -np.expm1(-point_contributions)
        draw_contributions = -np.expm1(-draw_contributions)
    predicted = float(np.sum(point_contributions))
    draw_sum = np.sum(draw_contributions, axis=0)
    lower, upper = (float(value) for value in np.quantile(draw_sum, [0.025, 0.975]))
    realized = sum(item.capsized for item in scores)
    emissions = sum(len(item.rate.emissions) for item in scores)
    flagged = sum(bool(emission.flags) for item in scores for emission in item.rate.emissions)
    side_positive = sum(
        emission.positive_crossings for item in scores for emission in item.rate.emissions[-1:]
    )
    side_negative = sum(
        emission.negative_crossings for item in scores for emission in item.rate.emissions[-1:]
    )
    return {
        "trajectory_count": len(scores),
        "realized_capsize_count": realized,
        "predicted_capsize_count": predicted,
        "predicted_count_interval": [lower, upper],
        "captures_realized_count": lower <= realized <= upper,
        "valid_emission_count": emissions,
        "flagged_emission_count": flagged,
        "terminal_side_split": {
            "positive": side_positive,
            "negative": side_negative,
        },
        "event_accounting": "absorbing_probability" if absorbing_events else "rate_integral",
    }


def reliability_edges(
    scores: list[ScoredRateTrajectory], bins: int = U1_RELIABILITY_BINS
) -> NDArray[np.float64]:
    values = np.asarray([item.average_rate_per_hour for item in scores], dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    if edges[-1] == edges[0]:
        width = max(1e-12, abs(edges[0]) * 1e-9)
        edges = np.linspace(edges[0] - width, edges[-1] + width, bins + 1)
    else:
        edges[0] = np.nextafter(edges[0], -np.inf)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def reliability_summary(
    scores: list[ScoredRateTrajectory], edges: NDArray[np.floating]
) -> dict[str, object]:
    edge_values = np.asarray(edges, dtype=np.float64)
    average_rates = np.asarray([item.average_rate_per_hour for item in scores], dtype=np.float64)
    assignments = np.clip(np.digitize(average_rates, edge_values[1:-1]), 0, len(edge_values) - 2)
    rows: list[dict[str, object]] = []
    weighted_error = 0.0
    for index in range(len(edge_values) - 1):
        selected = np.flatnonzero(assignments == index)
        if not len(selected):
            continue
        realized = sum(scores[int(item)].capsized for item in selected)
        observed = realized / len(selected)
        predicted = float(
            np.mean([scores[int(item)].predicted_capsize_probability for item in selected])
        )
        interval = clopper_pearson_interval(realized, len(selected))
        weighted_error += len(selected) * abs(predicted - observed)
        rows.append(
            {
                "bin": index,
                "count": len(selected),
                "mean_rate_per_hour": float(np.mean(average_rates[selected])),
                "predicted_capsize_fraction": predicted,
                "realized_capsize_fraction": observed,
                "realized_exact_interval": [interval.lower, interval.upper],
            }
        )
    return {
        "edges_rate_per_hour": edge_values.tolist(),
        "bins": rows,
        "weighted_mean_absolute_error": weighted_error / len(scores),
    }


def as_trajectory_scores(
    scores: list[ScoredRateTrajectory], *, natural_period_s: float
) -> list[TrajectoryScores]:
    horizon_s = EWS_HORIZON_PERIODS * natural_period_s
    record_start_s = 60.0 * natural_period_s
    output = []
    for item in scores:
        times = np.asarray([emission.time_s for emission in item.rate.emissions], dtype=np.float64)
        values = np.asarray(
            [emission.rate_per_hour for emission in item.rate.emissions], dtype=np.float64
        )
        output.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_end_s=detector_risk_end_s(
                    times,
                    t_capsize_s=item.t_capsize_s or np.nan,
                    raw_record_end_s=item.exposure_end_s,
                    horizon_s=horizon_s,
                    record_start_s=record_start_s,
                ),
                t_capsize_s=item.t_capsize_s,
                record_start_s=record_start_s,
            )
        )
    return output


def ensemble_rate_and_hazard(
    scores: list[ScoredRateTrajectory],
    grid_s: NDArray[np.floating],
    *,
    natural_period_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    grid = np.asarray(grid_s, dtype=np.float64)
    paths = np.full((len(scores), len(grid)), np.nan, dtype=np.float64)
    for row, item in enumerate(scores):
        active = grid <= item.exposure_end_s + 1e-9
        paths[row, active] = 0.0
        emissions = item.rate.emissions
        if emissions:
            times = np.asarray([emission.time_s for emission in emissions])
            values = np.asarray([emission.rate_per_hour for emission in emissions])
            positions = np.searchsorted(times, grid[active], side="right") - 1
            available = positions >= 0
            active_indices = np.flatnonzero(active)
            paths[row, active_indices[available]] = values[positions[available]]
    with np.errstate(invalid="ignore"):
        mean_rate = np.nanmean(paths, axis=0)
    mean_rate = np.nan_to_num(mean_rate, nan=0.0)

    bandwidth_s = U1_HAZARD_KERNEL_PERIODS * natural_period_s
    normalization = bandwidth_s * np.sqrt(2.0 * np.pi)
    capsize_times = np.asarray(
        [item.t_capsize_s for item in scores if item.t_capsize_s is not None],
        dtype=np.float64,
    )
    if len(capsize_times):
        density = (
            np.exp(-0.5 * ((grid[:, None] - capsize_times[None, :]) / bandwidth_s) ** 2).sum(axis=1)
            / normalization
        )
    else:
        density = np.zeros(len(grid), dtype=np.float64)
    at_risk = np.sum(
        grid[:, None]
        <= np.asarray([item.exposure_end_s for item in scores], dtype=np.float64)[None, :] + 1e-9,
        axis=1,
    )
    hazard = np.divide(
        density * 3_600.0,
        at_risk,
        out=np.zeros_like(density),
        where=at_risk > 0,
    )
    return mean_rate, hazard


def tracking_statistics(
    scores: list[ScoredRateTrajectory],
    grid_s: NDArray[np.floating],
    *,
    natural_period_s: float,
) -> NDArray[np.float64]:
    estimated, hazard = ensemble_rate_and_hazard(scores, grid_s, natural_period_s=natural_period_s)
    grid = np.asarray(grid_s, dtype=np.float64)
    cadence = float(np.median(np.diff(grid)))
    maximum = round(U1_TRACKING_MAX_LAG_PERIODS * natural_period_s / cadence)
    best_lag = 0
    best_correlation = -np.inf
    for lag in range(-maximum, maximum + 1):
        if lag >= 0:
            left = hazard[: len(hazard) - lag or None]
            right = estimated[lag:]
        else:
            left = hazard[-lag:]
            right = estimated[: len(estimated) + lag]
        if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    bias = float(np.mean(estimated - hazard))
    return np.asarray([best_lag * cadence, bias], dtype=np.float64)


def tracking_summary(
    scores: list[ScoredRateTrajectory],
    grid_s: NDArray[np.floating],
    *,
    natural_period_s: float,
) -> dict[str, object]:
    estimate = tracking_statistics(scores, grid_s, natural_period_s=natural_period_s)
    estimated_path, hazard_path = ensemble_rate_and_hazard(
        scores, grid_s, natural_period_s=natural_period_s
    )
    lag_estimable = bool(np.std(estimated_path) > 0.0 and np.std(hazard_path) > 0.0)
    interval = trajectory_block_bootstrap(
        scores,
        lambda sample: tracking_statistics(sample, grid_s, natural_period_s=natural_period_s),
        replicates=TRAJECTORY_BOOTSTRAP_REPLICATES,
        seed=TRAJECTORY_BOOTSTRAP_SEED,
    )
    return {
        "tracking_lag_s": float(estimate[0]),
        "tracking_lag_estimable": lag_estimable,
        "tracking_lag_trajectory_bootstrap_interval": [
            float(interval.lower[0]),
            float(interval.upper[0]),
        ],
        "bias_per_hour": float(estimate[1]),
        "bias_trajectory_bootstrap_interval": [
            float(interval.lower[1]),
            float(interval.upper[1]),
        ],
        "trajectory_bootstrap_replicates": interval.requested_replicates,
        "trajectory_bootstrap_seed": interval.seed,
    }
