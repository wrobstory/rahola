"""Shared fitting, scoring, and operating metrics for Prototype #2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from rahola.dataset import SimulationDataset
from rahola.windowing import binary_auc
from rahola_lab.constants import (
    CNN_GRID,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    EWS_SUBWINDOW_FRACTIONS,
    EXCLUSION_BUFFER_PERIODS,
    NEIGHBOR_RADIUS_GRID,
)
from rahola_lab.detectors import (
    DetectorWindowDataset,
    JaxTemporalCNN,
    classical_ews_scores,
    extract_detector_windows,
    galeazzi_roll_power_glrt,
    neighbor_count_scores,
)
from rahola_lab.evaluation import (
    EpisodeConfig,
    OperatingPoint,
    TrajectoryScores,
    estimate_decorrelation_time,
    operating_curve,
)
from rahola_lab.forecast import fit_piecewise_linear_restoring

DETECTOR_NAMES = ("cnn", "classical_ews", "galeazzi_glrt", "danger_margin", "neighbor_2009")


def common_natural_period_s(datasets: list[SimulationDataset]) -> float:
    """Return the shared configured natural period for an evaluation pool."""
    periods = {float(dataset.config["natural_period_s"]) for dataset in datasets}
    if len(periods) != 1:
        raise ValueError("evaluation datasets must share one natural period")
    return periods.pop()


@dataclass
class DetectorSuite:
    cnn: JaxTemporalCNN
    cnn_grid_index: int
    ews_statistic: str
    ews_fraction: float
    neighbor_radius: float
    selection_rows: list[dict[str, object]]


def _concatenate(parts: list[DetectorWindowDataset]) -> DetectorWindowDataset:
    if not parts:
        raise ValueError("at least one window dataset is required")
    return DetectorWindowDataset(
        features=np.concatenate([part.features for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        family_labels=np.concatenate([part.family_labels for part in parts]),
        trajectory_indices=np.concatenate([part.trajectory_indices for part in parts]),
        end_times_s=np.concatenate([part.end_times_s for part in parts]),
        raw_angle_rad=np.concatenate([part.raw_angle_rad for part in parts]),
        raw_rate_rad_s=np.concatenate([part.raw_rate_rad_s for part in parts]),
    )


def training_windows(
    datasets: list[SimulationDataset], *, max_windows_per_trajectory: int = 3
) -> DetectorWindowDataset:
    return _concatenate(
        [
            extract_detector_windows(
                dataset,
                stride_s=20.0,
                max_windows_per_trajectory=max_windows_per_trajectory,
            )
            for dataset in datasets
        ]
    )


def fit_detector_suite(
    training: DetectorWindowDataset, calibration: DetectorWindowDataset
) -> DetectorSuite:
    """Select only from the frozen train/calibration detector grids."""
    samples_per_period = training.features.shape[1] // 60
    selection: list[dict[str, object]] = []
    best_ews: tuple[float, str, float] | None = None
    for statistic in ("variance", "ac1"):
        for fraction in EWS_SUBWINDOW_FRACTIONS:
            scores = classical_ews_scores(
                calibration.features, statistic=statistic, subwindow_fraction=fraction
            )
            auc = binary_auc(calibration.labels, scores)
            selection.append(
                {
                    "method": "classical_ews",
                    "statistic": statistic,
                    "fraction": fraction,
                    "auc": auc,
                }
            )
            candidate = (auc, statistic, fraction)
            if best_ews is None or candidate[0] > best_ews[0]:
                best_ews = candidate
    best_neighbor: tuple[float, float] | None = None
    for radius in NEIGHBOR_RADIUS_GRID:
        scores = neighbor_count_scores(
            calibration.features, radius=radius, samples_per_period=samples_per_period
        )
        auc = binary_auc(calibration.labels, scores)
        selection.append({"method": "neighbor_2009", "radius": radius, "auc": auc})
        if best_neighbor is None or auc > best_neighbor[0]:
            best_neighbor = (auc, radius)
    best_cnn: tuple[float, int, JaxTemporalCNN] | None = None
    for index, config in enumerate(CNN_GRID):
        model = JaxTemporalCNN(
            channels=tuple(config["channels"]),
            kernel_size=int(config["kernel_size"]),
            family_head_weight=float(config["family_head_weight"]),
            epochs=4,
        )
        model.fit(training.features, training.labels, family_labels=training.family_labels)
        scores = model.predict_scores(calibration.features)
        auc = binary_auc(calibration.labels, scores)
        selection.append(
            {
                "method": "cnn",
                "grid_index": index,
                "config": config,
                "auc": auc,
                "parameters": model.parameter_count(),
            }
        )
        if best_cnn is None or auc > best_cnn[0]:
            best_cnn = (auc, index, model)
    assert best_ews is not None and best_neighbor is not None and best_cnn is not None
    return DetectorSuite(
        cnn=best_cnn[2],
        cnn_grid_index=best_cnn[1],
        ews_statistic=best_ews[1],
        ews_fraction=best_ews[2],
        neighbor_radius=best_neighbor[1],
        selection_rows=selection,
    )


def fit_frozen_suite(training: DetectorWindowDataset, selected: dict[str, object]) -> DetectorSuite:
    """Refit the deterministic D1-selected CNN without reopening the grid."""
    index = int(selected["cnn_grid_index"])
    config = CNN_GRID[index]
    model = JaxTemporalCNN(
        channels=tuple(config["channels"]),
        kernel_size=int(config["kernel_size"]),
        family_head_weight=float(config["family_head_weight"]),
        epochs=4,
    )
    model.fit(training.features, training.labels, family_labels=training.family_labels)
    return DetectorSuite(
        cnn=model,
        cnn_grid_index=index,
        ews_statistic=str(selected["ews_statistic"]),
        ews_fraction=float(selected["ews_fraction"]),
        neighbor_radius=float(selected["neighbor_radius"]),
        selection_rows=[],
    )


def score_dataset(
    dataset: SimulationDataset, suite: DetectorSuite
) -> dict[str, list[TrajectoryScores]]:
    """Score a campaign in bounded trajectory batches."""
    output = {name: [] for name in DETECTOR_NAMES}
    period = float(dataset.config["natural_period_s"])
    samples_per_period = round(period / float(np.median(np.diff(dataset.time_s))))
    danger = fit_piecewise_linear_restoring(dataset.config)
    for start in range(0, dataset.batch_size, 128):
        stop = min(start + 128, dataset.batch_size)
        chunk = SimulationDataset(
            time_s=dataset.time_s,
            angle_rad=dataset.angle_rad[start:stop],
            rate_rad_s=dataset.rate_rad_s[start:stop],
            seeds=dataset.seeds[start:stop],
            capsized=dataset.capsized[start:stop],
            t_capsize_s=dataset.t_capsize_s[start:stop],
            metadata=dataset.metadata[start:stop],
            config=dataset.config,
        )
        windows = extract_detector_windows(
            chunk, stride_s=10.0, allow_censored_for_inference=True
        )
        scores = {
            "cnn": suite.cnn.predict_scores(windows.features),
            "classical_ews": classical_ews_scores(
                windows.features,
                statistic=suite.ews_statistic,
                subwindow_fraction=suite.ews_fraction,
            ),
            "galeazzi_glrt": galeazzi_roll_power_glrt(
                windows.features, samples_per_period=samples_per_period
            ),
            "danger_margin": danger.danger_score(windows.raw_angle_rad, windows.raw_rate_rad_s),
            "neighbor_2009": neighbor_count_scores(
                windows.features,
                radius=suite.neighbor_radius,
                samples_per_period=samples_per_period,
            ),
        }
        for local in range(chunk.batch_size):
            selected = windows.trajectory_indices == local
            selected_times = windows.end_times_s[selected]
            capsize = float(chunk.t_capsize_s[local])
            record_end_s = detector_risk_end_s(
                selected_times,
                t_capsize_s=capsize,
                raw_record_end_s=float(chunk.time_s[-1]),
                horizon_s=EWS_HORIZON_PERIODS * period,
                record_start_s=60.0 * period,
            )
            capsize_value = capsize if np.isfinite(capsize) else None
            for name in DETECTOR_NAMES:
                selected_scores = scores[name][selected]
                output[name].append(
                    TrajectoryScores(
                        times_s=selected_times,
                        scores=selected_scores,
                        record_end_s=record_end_s,
                        t_capsize_s=capsize_value,
                        record_start_s=60.0 * period,
                    )
                )
    return output


def detector_risk_end_s(
    score_times_s: np.ndarray,
    *,
    t_capsize_s: float,
    raw_record_end_s: float,
    horizon_s: float,
    record_start_s: float,
) -> float:
    """End the evaluable risk interval without censoring the causal score stream.

    Every trajectory uses the same horizon-complete follow-up cutoff. Causal
    scores may continue beyond it for live inference, but neither late events nor
    late non-events enter supervised comparisons or false-alarm exposure.
    """
    times = np.asarray(score_times_s, dtype=np.float64)
    if not len(times):
        return float(record_start_s)
    outcome_end = min(float(times[-1]), float(raw_record_end_s - horizon_s))
    evaluable = times <= outcome_end + 1e-9
    return float(times[evaluable][-1]) if np.any(evaluable) else float(record_start_s)


def relative_fpr_reduction(candidate: float, baseline: float) -> float | None:
    """Return relative FPR improvement when the baseline supports that estimand."""
    if (
        not np.isfinite(candidate)
        or not np.isfinite(baseline)
        or candidate < 0.0
        or baseline < 0.0
    ):
        raise ValueError("FPR values must be finite and nonnegative")
    if baseline == 0.0:
        return None
    return 1.0 - candidate / baseline


def merge_scores(
    parts: list[dict[str, list[TrajectoryScores]]],
) -> dict[str, list[TrajectoryScores]]:
    return {
        name: list(chain.from_iterable(part[name] for part in parts)) for name in DETECTOR_NAMES
    }


def decorrelation_times(
    calibration_scores: dict[str, list[TrajectoryScores]], fallback_s: float = 10.0
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, trajectories in calibration_scores.items():
        estimates = []
        for trajectory in trajectories:
            if len(trajectory.scores) >= 4:
                dt = float(np.median(np.diff(trajectory.times_s)))
                estimates.append(estimate_decorrelation_time(trajectory.scores, dt))
        output[name] = float(np.median(estimates)) if estimates else fallback_s
    return output


def threshold_grids(
    calibration_scores: dict[str, list[TrajectoryScores]], count: int = 41
) -> dict[str, np.ndarray]:
    grids = {}
    for name, trajectories in calibration_scores.items():
        values = np.concatenate(
            [trajectory.scores for trajectory in trajectories if len(trajectory.scores)]
        )
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"{name} produced no finite calibration scores")
        grid = np.append(
            np.quantile(values, np.linspace(0.0, 1.0, count)),
            -np.finfo(np.float64).max,
        )
        if name == "neighbor_2009":
            # Story (2009, Ch. 3) flagged fewer than 50 neighbors. The score is
            # negative count, so -50 is the faithful binary point on the sweep.
            grid = np.append(grid, -50.0)
        grids[name] = np.unique(grid)
    return grids


def evaluate_suite(
    scores: dict[str, list[TrajectoryScores]],
    grids: dict[str, np.ndarray],
    decorrelation_s: dict[str, float],
) -> dict[str, tuple[OperatingPoint, ...]]:
    horizon_s = EWS_HORIZON_PERIODS * 4.0
    return {
        name: operating_curve(
            trajectories,
            EpisodeConfig(threshold=0.0, debounce_windows=3, refractory_windows=3),
            grids[name],
            horizon_s=horizon_s,
            decorrelation_time_s=decorrelation_s[name],
        )
        for name, trajectories in scores.items()
    }


def select_operating_points(
    calibration_scores: dict[str, list[TrajectoryScores]],
    grids: dict[str, np.ndarray],
    decorrelation_s: dict[str, float],
) -> dict[str, OperatingPoint]:
    """Choose every operational threshold from calibration outcomes only."""
    curves = evaluate_suite(calibration_scores, grids, decorrelation_s)
    return {name: matched_point(curve) for name, curve in curves.items()}


def evaluate_suite_at_thresholds(
    scores: dict[str, list[TrajectoryScores]],
    thresholds: dict[str, float],
    decorrelation_s: dict[str, float],
) -> dict[str, OperatingPoint]:
    """Evaluate one frozen threshold per detector without reopening a sweep."""
    curves = evaluate_suite(
        scores,
        {name: np.asarray([thresholds[name]], dtype=np.float64) for name in scores},
        decorrelation_s,
    )
    return {name: curve[0] for name, curve in curves.items()}


def point_payload(point: OperatingPoint) -> dict[str, object]:
    metrics = point.metrics
    return {
        "threshold": point.threshold,
        "sensitivity": metrics.sensitivity,
        "sensitivity_interval": [
            metrics.sensitivity_interval.lower,
            metrics.sensitivity_interval.upper,
        ],
        "false_episodes_per_hour": metrics.false_positives_per_hour,
        "false_episodes_per_hour_interval": [
            metrics.false_positives_per_hour_interval.lower,
            metrics.false_positives_per_hour_interval.upper,
        ],
        "lead_time_quantiles_s": list(point.lead_time_quantiles_s),
        "false_episode_count": metrics.false_episode_count,
        "capsize_count": metrics.capsize_count,
        "detected_capsize_count": metrics.detected_capsize_count,
        "exposure_hours": metrics.exposure_hours,
        "alarm_opportunity_count": metrics.alarm_opportunity_count,
    }


def matched_point(curve: tuple[OperatingPoint, ...]) -> OperatingPoint:
    eligible = [
        point for point in curve if point.metrics.sensitivity >= DETECTOR_MATCHED_SENSITIVITY
    ]
    if eligible:
        return min(eligible, key=lambda point: point.metrics.false_positives_per_hour)
    return max(curve, key=lambda point: point.metrics.sensitivity)


def window_auc(
    trajectories: list[TrajectoryScores],
    *,
    horizon_s: float = 200.0,
    buffer_s: float = EXCLUSION_BUFFER_PERIODS * 4.0,
) -> float:
    """Rank horizon-complete positive and unambiguous negative endpoints."""
    if horizon_s <= 0.0 or buffer_s < 0.0:
        raise ValueError("horizon must be positive and buffer must be nonnegative")
    labels = []
    values = []
    for trajectory in trajectories:
        times = np.asarray(trajectory.times_s, dtype=np.float64)
        scores = np.asarray(trajectory.scores, dtype=np.float64)
        selected = (
            (times >= trajectory.record_start_s)
            & (times <= trajectory.record_end_s)
            & np.isfinite(scores)
        )
        capsize = trajectory.t_capsize_s
        if capsize is None:
            labels.extend([0] * int(np.sum(selected)))
        else:
            lead = capsize - times
            selected &= ((lead > 0.0) & (lead <= horizon_s)) | (
                lead > horizon_s + buffer_s
            )
            labels.extend(((lead[selected] > 0.0) & (lead[selected] <= horizon_s)).tolist())
        values.extend(scores[selected].tolist())
    return binary_auc(np.asarray(labels, dtype=np.int8), np.asarray(values, dtype=np.float64))


def campaign_dir(data_root: Path, name: str) -> Path:
    return data_root / name
