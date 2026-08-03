"""E3: fixed split-CQR versus ACI through a sea-state step."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import (
    SplitCQRUpper,
    adaptive_conformal_bounds,
    normalized_alarm_scores,
)
from rahola_lab.constants import (
    ACI_EXPLOSION_FACTOR,
    ACI_EXPLOSION_FPR_PER_HOUR,
    ACI_GAMMA_GRID,
    SeedBlock,
)
from rahola_lab.evaluation import EpisodeConfig, TrajectoryScores, evaluate_alarms
from rahola_lab.experiments.common import (
    campaign_path,
    fit_forecasters,
    snapshot,
    subset_dataset,
    trajectory_forecasts,
    write_result,
)

ALPHA = 0.1
HORIZON_S = 60.0
TRANSITION_S = 300.0
MODEL_NAME = "lstm"


def _bounds(streams, scores, gamma=None):
    fixed_correction = SplitCQRUpper(scores).correction(ALPHA)
    output = []
    for stream in streams:
        raw = stream.raw_upper_rad[MODEL_NAME]
        if gamma is None:
            bound = raw + fixed_correction
            errors = stream.targets_rad > bound
        else:
            result = adaptive_conformal_bounds(
                scores, raw, stream.targets_rad, alpha=ALPHA, gamma=gamma
            )
            bound = result.upper_bounds
            errors = result.errors
        output.append((stream, bound, errors))
    return output


def _episode_metrics(bounded, escape_angle):
    trajectories = []
    for stream, bound, _ in bounded:
        times = stream.times_s if len(stream.times_s) else np.array([0.0])
        values = normalized_alarm_scores(bound, escape_angle) if len(bound) else np.array([-np.inf])
        trajectories.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_start_s=120.0,
                record_end_s=stream.record_end_s,
                t_capsize_s=stream.t_capsize_s,
            )
        )
    return evaluate_alarms(
        trajectories,
        EpisodeConfig(threshold=1.0, debounce_windows=3, refractory_windows=3),
        horizon_s=HORIZON_S,
    )


def _rolling_coverage(bounded, width_bins=6):
    errors_by_time = defaultdict(list)
    for stream, _, errors in bounded:
        for time_s, error in zip(stream.times_s, errors, strict=True):
            errors_by_time[float(time_s)].append(bool(error))
    times = np.array(sorted(errors_by_time), dtype=np.float64)
    misses = np.array([sum(errors_by_time[t]) for t in times], dtype=np.float64)
    counts = np.array([len(errors_by_time[t]) for t in times], dtype=np.float64)
    rolling_misses = np.convolve(misses, np.ones(width_bins), mode="full")[: len(times)]
    rolling_counts = np.convolve(counts, np.ones(width_bins), mode="full")[: len(times)]
    return times, 1.0 - rolling_misses / rolling_counts


def _recovery_time(times, coverage):
    acceptable = np.abs(coverage - (1.0 - ALPHA)) <= 0.03
    for index in range(len(times) - 2):
        if times[index] >= TRANSITION_S and np.all(acceptable[index : index + 3]):
            return float(times[index] - TRANSITION_S)
    return None


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    training = load_campaign_split(campaign_path(data_root, "softening", "stationary"), "train")
    models = fit_forecasters(training, HORIZON_S)
    step_path = data_root / "softening_step"
    calibration = load_campaign_split(step_path, SeedBlock.CALIBRATION)
    score_half = subset_dataset(calibration, 0, calibration.batch_size // 2)
    tuning_half = subset_dataset(calibration, calibration.batch_size // 2, calibration.batch_size)
    calibration_y, calibration_raw = snapshot(score_half, models, HORIZON_S, history_end_s=240.0)
    calibration_scores = calibration_y - calibration_raw[MODEL_NAME]
    tuning_streams = trajectory_forecasts(
        tuning_half, {MODEL_NAME: models[MODEL_NAME]}, HORIZON_S, stride_s=10.0
    )
    escape_angle = float(calibration.config["escape_angle_rad"])
    fixed_tuning = _bounds(tuning_streams, calibration_scores)
    fixed_tuning_metrics = _episode_metrics(fixed_tuning, escape_angle)
    gamma_rows = []
    for gamma in ACI_GAMMA_GRID:
        adaptive = _bounds(tuning_streams, calibration_scores, gamma)
        times, coverage = _rolling_coverage(adaptive)
        post = coverage[times >= TRANSITION_S]
        metrics = _episode_metrics(adaptive, escape_angle)
        recovery = _recovery_time(times, coverage)
        exploding = (
            metrics.false_positives_per_hour > ACI_EXPLOSION_FPR_PER_HOUR
            and metrics.false_positives_per_hour
            > ACI_EXPLOSION_FACTOR * fixed_tuning_metrics.false_positives_per_hour
        )
        gamma_rows.append(
            {
                "gamma": gamma,
                "post_transition_mean_absolute_coverage_delta_pp": 100.0
                * float(np.mean(np.abs(post - (1.0 - ALPHA)))),
                "false_positives_per_hour": metrics.false_positives_per_hour,
                "recovery_time_s": recovery,
                "exploding": exploding,
            }
        )
    selected = min(
        gamma_rows,
        key=lambda row: row["post_transition_mean_absolute_coverage_delta_pp"],
    )
    gamma = float(selected["gamma"])

    # Pseudo-prospective seal: this is the experiment's only test load.
    test = load_campaign_split(step_path, SeedBlock.TEST)
    test_streams = trajectory_forecasts(
        test, {MODEL_NAME: models[MODEL_NAME]}, HORIZON_S, stride_s=10.0
    )
    fixed = _bounds(test_streams, calibration_scores)
    adaptive = _bounds(test_streams, calibration_scores, gamma)
    fixed_times, fixed_coverage = _rolling_coverage(fixed)
    adaptive_times, adaptive_coverage = _rolling_coverage(adaptive)
    fixed_metrics = _episode_metrics(fixed, escape_angle)
    adaptive_metrics = _episode_metrics(adaptive, escape_angle)
    pre = fixed_times < TRANSITION_S
    post = fixed_times >= TRANSITION_S
    recovery = _recovery_time(adaptive_times, adaptive_coverage)
    exploding = (
        adaptive_metrics.false_positives_per_hour > ACI_EXPLOSION_FPR_PER_HOUR
        and adaptive_metrics.false_positives_per_hour
        > ACI_EXPLOSION_FACTOR * fixed_metrics.false_positives_per_hour
    )

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e3_transition.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.plot(fixed_times, fixed_coverage, label="fixed split-CQR")
    axis.plot(adaptive_times, adaptive_coverage, label=f"ACI, gamma={gamma:g}")
    axis.axvline(TRANSITION_S, color="black", linestyle=":", label="sea-state step")
    axis.axhline(1.0 - ALPHA, color="black", linestyle="--", linewidth=1)
    axis.fill_between(
        fixed_times,
        1.0 - ALPHA - 0.03,
        1.0 - ALPHA + 0.03,
        color="black",
        alpha=0.08,
    )
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("time in trajectory (s)")
    axis.set_ylabel("trailing 60 s empirical coverage")
    axis.set_title("E3 — coverage through a sea-state step")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "E3",
        "alpha": ALPHA,
        "selected_gamma": gamma,
        "gamma_calibration": gamma_rows,
        "fixed_pre_coverage": float(np.mean(fixed_coverage[pre])),
        "fixed_post_coverage": float(np.mean(fixed_coverage[post])),
        "aci_pre_coverage": float(np.mean(adaptive_coverage[pre])),
        "aci_post_coverage": float(np.mean(adaptive_coverage[post])),
        "aci_recovery_time_s": recovery,
        "fixed_fpr_per_hour": fixed_metrics.false_positives_per_hour,
        "aci_fpr_per_hour": adaptive_metrics.false_positives_per_hour,
        "kill_criterion": {
            "absolute_fpr_per_hour": ACI_EXPLOSION_FPR_PER_HOUR,
            "relative_to_fixed": ACI_EXPLOSION_FACTOR,
            "exploded": exploding,
            "coverage_within_3pp_after_recovery": recovery is not None,
            "passed": recovery is not None and not exploding,
        },
        "figure": str(figure_path),
    }
    write_result(output_root, "e3_transition", payload)
    return payload
