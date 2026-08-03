"""E2: guaranteed-alarm sensitivity versus false episodes per exposure hour."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import SplitCQRUpper, normalized_alarm_scores
from rahola_lab.constants import DANGER_SCORE_THRESHOLDS_RAD_S, SeedBlock
from rahola_lab.evaluation import EpisodeConfig, TrajectoryScores, evaluate_alarms
from rahola_lab.experiments.common import (
    FAMILIES,
    MODEL_NAMES,
    campaign_path,
    fit_forecasters,
    snapshot,
    trajectory_forecasts,
    write_result,
)
from rahola_lab.forecast import DangerMarginFit, fit_piecewise_linear_restoring

HORIZON_S = 60.0


def _trajectory_scores(streams, model_name: str, correction: float, escape_angle: float):
    converted: list[TrajectoryScores] = []
    for stream in streams:
        values = normalized_alarm_scores(
            stream.raw_upper_rad[model_name] + correction, escape_angle
        )
        times = stream.times_s
        if len(times) == 0:
            times = np.array([0.0])
            values = np.array([-np.inf])
        converted.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_end_s=stream.record_end_s,
                t_capsize_s=stream.t_capsize_s,
                record_start_s=120.0,
            )
        )
    return converted


def _physics_scores(streams, fit: DangerMarginFit):
    converted: list[TrajectoryScores] = []
    for stream in streams:
        values = fit.danger_score(stream.angle_rad, stream.rate_rad_s)
        times = stream.times_s
        if len(times) == 0:
            times = np.array([0.0])
            values = np.array([-np.inf])
        converted.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_end_s=stream.record_end_s,
                t_capsize_s=stream.t_capsize_s,
                record_start_s=120.0,
            )
        )
    return converted


def _interval_payload(interval):
    return [interval.lower, interval.upper]


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    alpha_grid = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    by_model_alpha: dict[tuple[str, float], list[TrajectoryScores]] = {
        (model, float(alpha)): [] for model in MODEL_NAMES for alpha in alpha_grid
    }
    physics_trajectories: list[TrajectoryScores] = []
    for family in FAMILIES:
        training = load_campaign_split(campaign_path(data_root, family, "stationary"), "train")
        models = fit_forecasters(training, HORIZON_S)
        evaluation_path = campaign_path(data_root, family, "evaluation")
        calibration = load_campaign_split(evaluation_path, SeedBlock.CALIBRATION)
        calibration_y, calibration_raw = snapshot(
            calibration, models, HORIZON_S, history_end_s=180.0
        )
        # Pseudo-prospective seal: this is the experiment's only test load.
        test = load_campaign_split(evaluation_path, SeedBlock.TEST)
        streams = trajectory_forecasts(test, models, HORIZON_S, stride_s=10.0)
        physics_fit = fit_piecewise_linear_restoring(test.config)
        physics_trajectories.extend(_physics_scores(streams, physics_fit))
        escape_angle = float(test.config["escape_angle_rad"])
        for model_name in MODEL_NAMES:
            conformal = SplitCQRUpper.calibrate(calibration_y, calibration_raw[model_name])
            for alpha in alpha_grid:
                by_model_alpha[(model_name, float(alpha))].extend(
                    _trajectory_scores(
                        streams,
                        model_name,
                        conformal.correction(float(alpha)),
                        escape_angle,
                    )
                )

    rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        for alpha in alpha_grid:
            metrics = evaluate_alarms(
                by_model_alpha[(model_name, float(alpha))],
                EpisodeConfig(threshold=1.0, debounce_windows=3, refractory_windows=3),
                horizon_s=HORIZON_S,
            )
            lead = (
                np.quantile(metrics.lead_times_s, [0.1, 0.5, 0.9])
                if len(metrics.lead_times_s)
                else np.full(3, np.nan)
            )
            rows.append(
                {
                    "model": model_name,
                    "alpha": float(alpha),
                    "sensitivity": metrics.sensitivity,
                    "sensitivity_interval": _interval_payload(metrics.sensitivity_interval),
                    "false_positives_per_hour": metrics.false_positives_per_hour,
                    "false_positives_per_hour_interval": _interval_payload(
                        metrics.false_positives_per_hour_interval
                    ),
                    "lead_time_q10_s": float(lead[0]),
                    "lead_time_median_s": float(lead[1]),
                    "lead_time_q90_s": float(lead[2]),
                    "capsizes": metrics.capsize_count,
                    "exposure_hours": metrics.exposure_hours,
                }
            )
    for threshold in DANGER_SCORE_THRESHOLDS_RAD_S:
        metrics = evaluate_alarms(
            physics_trajectories,
            EpisodeConfig(threshold=threshold, debounce_windows=3, refractory_windows=3),
            horizon_s=HORIZON_S,
        )
        lead = (
            np.quantile(metrics.lead_times_s, [0.1, 0.5, 0.9])
            if len(metrics.lead_times_s)
            else np.full(3, np.nan)
        )
        rows.append(
            {
                "model": "danger_margin",
                "threshold_rad_s": threshold,
                "sensitivity": metrics.sensitivity,
                "sensitivity_interval": _interval_payload(metrics.sensitivity_interval),
                "false_positives_per_hour": metrics.false_positives_per_hour,
                "false_positives_per_hour_interval": _interval_payload(
                    metrics.false_positives_per_hour_interval
                ),
                "lead_time_q10_s": float(lead[0]),
                "lead_time_median_s": float(lead[1]),
                "lead_time_q90_s": float(lead[2]),
                "capsizes": metrics.capsize_count,
                "exposure_hours": metrics.exposure_hours,
            }
        )

    headline: dict[str, object] = {}
    for model_name in (*MODEL_NAMES, "danger_margin"):
        eligible = [row for row in rows if row["model"] == model_name and row["sensitivity"] >= 0.9]
        if eligible:
            point = min(eligible, key=lambda row: row["false_positives_per_hour"])
            headline[model_name] = {
                "reached_90_percent_sensitivity": True,
                "fpr_per_hour": point["false_positives_per_hour"],
                "sensitivity": point["sensitivity"],
                "sensitivity_interval": point["sensitivity_interval"],
                "fpr_per_hour_interval": point["false_positives_per_hour_interval"],
                "control": point.get("alpha", point.get("threshold_rad_s")),
                "control_name": "alpha" if "alpha" in point else "threshold_rad_s",
                "median_lead_time_s": point["lead_time_median_s"],
            }
        else:
            point = max(
                (row for row in rows if row["model"] == model_name),
                key=lambda row: row["sensitivity"],
            )
            headline[model_name] = {
                "reached_90_percent_sensitivity": False,
                "maximum_sensitivity": point["sensitivity"],
                "fpr_per_hour_at_maximum": point["false_positives_per_hour"],
                "sensitivity_interval": point["sensitivity_interval"],
                "fpr_per_hour_interval": point["false_positives_per_hour_interval"],
                "control": point.get("alpha", point.get("threshold_rad_s")),
                "control_name": "alpha" if "alpha" in point else "threshold_rad_s",
            }

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e2_operating_curve.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    for model_name in (*MODEL_NAMES, "danger_margin"):
        series = [row for row in rows if row["model"] == model_name]
        axis.plot(
            [row["false_positives_per_hour"] for row in series],
            [row["sensitivity"] for row in series],
            marker="o",
            label=model_name,
        )
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("false alarm episodes / exposure hour")
    axis.set_ylabel("capsize sensitivity (60 s horizon)")
    axis.set_title("E2 — conformal alarm operating curve")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    payload: dict[str, object] = {
        "experiment": "E2",
        "horizon_s": HORIZON_S,
        "alpha_grid": alpha_grid.tolist(),
        "danger_score_thresholds_rad_s": list(DANGER_SCORE_THRESHOLDS_RAD_S),
        "episode_config": {"debounce_windows": 3, "refractory_windows": 3},
        "rows": rows,
        "at_90_percent_sensitivity": headline,
        "figure": str(figure_path),
    }
    write_result(output_root, "e2_operating_curve", payload)
    return payload
