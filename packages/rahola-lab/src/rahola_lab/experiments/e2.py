"""E2: guaranteed-alarm sensitivity versus false episodes per exposure hour."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import SplitCQRUpper, normalized_alarm_scores
from rahola_lab.constants import SeedBlock
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


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    alpha_grid = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    by_model_alpha: dict[tuple[str, float], list[TrajectoryScores]] = {
        (model, float(alpha)): [] for model in MODEL_NAMES for alpha in alpha_grid
    }
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
                    "false_positives_per_hour": metrics.false_positives_per_hour,
                    "lead_time_q10_s": float(lead[0]),
                    "lead_time_median_s": float(lead[1]),
                    "lead_time_q90_s": float(lead[2]),
                    "capsizes": metrics.capsize_count,
                    "exposure_hours": metrics.exposure_hours,
                }
            )

    headline: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        eligible = [row for row in rows if row["model"] == model_name and row["sensitivity"] >= 0.9]
        if eligible:
            point = min(eligible, key=lambda row: row["false_positives_per_hour"])
            headline[model_name] = {
                "reached_90_percent_sensitivity": True,
                "fpr_per_hour": point["false_positives_per_hour"],
                "sensitivity": point["sensitivity"],
                "alpha": point["alpha"],
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
                "alpha": point["alpha"],
            }

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e2_operating_curve.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    for model_name in MODEL_NAMES:
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
        "episode_config": {"debounce_windows": 3, "refractory_windows": 3},
        "rows": rows,
        "at_90_percent_sensitivity": headline,
        "figure": str(figure_path),
    }
    write_result(output_root, "e2_operating_curve", payload)
    return payload
