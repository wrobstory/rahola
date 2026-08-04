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
from rahola_lab.forecast import (
    DangerMarginFit,
    absolute_roll_escape_angle,
    fit_piecewise_linear_restoring,
)

HORIZON_S = 60.0


def _trajectory_scores(streams, model_name: str, correction: float, escape_angle: float):
    converted: list[TrajectoryScores] = []
    for stream in streams:
        values = normalized_alarm_scores(
            stream.raw_upper_rad[model_name] + correction, escape_angle
        )
        times = stream.times_s
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


def _metric_row(
    model_name: str,
    control_name: str,
    control: float,
    trajectories: list[TrajectoryScores],
) -> dict[str, object]:
    metrics = evaluate_alarms(
        trajectories,
        EpisodeConfig(
            threshold=1.0 if control_name == "alpha" else control,
            debounce_windows=3,
            refractory_windows=3,
        ),
        horizon_s=HORIZON_S,
    )
    lead = (
        np.quantile(metrics.lead_times_s, [0.1, 0.5, 0.9])
        if len(metrics.lead_times_s)
        else np.full(3, np.nan)
    )
    return {
        "model": model_name,
        control_name: control,
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


def _select_control(rows: list[dict[str, object]], model_name: str) -> dict[str, object]:
    candidates = [row for row in rows if row["model"] == model_name]
    eligible = [row for row in candidates if row["sensitivity"] >= 0.9]
    return (
        min(eligible, key=lambda row: row["false_positives_per_hour"])
        if eligible
        else max(candidates, key=lambda row: row["sensitivity"])
    )


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    alpha_grid = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    calibration_by_model_alpha: dict[tuple[str, float], list[TrajectoryScores]] = {
        (model, float(alpha)): [] for model in MODEL_NAMES for alpha in alpha_grid
    }
    calibration_physics_trajectories: list[TrajectoryScores] = []
    test_bundles = []
    for family in FAMILIES:
        training = load_campaign_split(campaign_path(data_root, family, "stationary"), "train")
        models = fit_forecasters(training, HORIZON_S)
        evaluation_path = campaign_path(data_root, family, "evaluation")
        calibration = load_campaign_split(evaluation_path, SeedBlock.CALIBRATION)
        calibration_y, calibration_raw = snapshot(
            calibration, models, HORIZON_S, history_end_s=180.0
        )
        calibration_streams = trajectory_forecasts(
            calibration, models, HORIZON_S, stride_s=10.0
        )
        # Pseudo-prospective seal: this is the experiment's only test load.
        test = load_campaign_split(evaluation_path, SeedBlock.TEST)
        streams = trajectory_forecasts(test, models, HORIZON_S, stride_s=10.0)
        physics_fit = fit_piecewise_linear_restoring(test.config)
        calibration_physics_trajectories.extend(
            _physics_scores(calibration_streams, physics_fit)
        )
        escape_angle = absolute_roll_escape_angle(test.config)
        corrections: dict[tuple[str, float], float] = {}
        for model_name in MODEL_NAMES:
            conformal = SplitCQRUpper.calibrate(calibration_y, calibration_raw[model_name])
            for alpha in alpha_grid:
                correction = conformal.correction(float(alpha))
                corrections[(model_name, float(alpha))] = correction
                calibration_by_model_alpha[(model_name, float(alpha))].extend(
                    _trajectory_scores(
                        calibration_streams,
                        model_name,
                        correction,
                        escape_angle,
                    )
                )
        test_bundles.append((streams, corrections, escape_angle, physics_fit))

    calibration_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        for alpha in alpha_grid:
            calibration_rows.append(
                _metric_row(
                    model_name,
                    "alpha",
                    float(alpha),
                    calibration_by_model_alpha[(model_name, float(alpha))],
                )
            )
    for threshold in DANGER_SCORE_THRESHOLDS_RAD_S:
        calibration_rows.append(
            _metric_row(
                "danger_margin",
                "threshold_rad_s",
                threshold,
                calibration_physics_trajectories,
            )
        )

    calibration_controls = {
        model_name: _select_control(calibration_rows, model_name)
        for model_name in (*MODEL_NAMES, "danger_margin")
    }
    selected_test: dict[str, list[TrajectoryScores]] = {
        model_name: [] for model_name in (*MODEL_NAMES, "danger_margin")
    }
    for streams, corrections, escape_angle, physics_fit in test_bundles:
        for model_name in MODEL_NAMES:
            alpha = float(calibration_controls[model_name]["alpha"])
            selected_test[model_name].extend(
                _trajectory_scores(
                    streams,
                    model_name,
                    corrections[(model_name, alpha)],
                    escape_angle,
                )
            )
        selected_test["danger_margin"].extend(_physics_scores(streams, physics_fit))

    headline: dict[str, object] = {}
    for model_name in (*MODEL_NAMES, "danger_margin"):
        calibration_point = calibration_controls[model_name]
        control_name = "alpha" if "alpha" in calibration_point else "threshold_rad_s"
        point = _metric_row(
            model_name,
            control_name,
            float(calibration_point[control_name]),
            selected_test[model_name],
        )
        if calibration_point["sensitivity"] >= 0.9:
            headline[model_name] = {
                "calibration_reached_90_percent_sensitivity": True,
                "test_reached_90_percent_sensitivity": point["sensitivity"] >= 0.9,
                "fpr_per_hour": point["false_positives_per_hour"],
                "sensitivity": point["sensitivity"],
                "sensitivity_interval": point["sensitivity_interval"],
                "fpr_per_hour_interval": point["false_positives_per_hour_interval"],
                "control": point[control_name],
                "control_name": control_name,
                "median_lead_time_s": point["lead_time_median_s"],
                "calibration_point": calibration_point,
            }
        else:
            headline[model_name] = {
                "calibration_reached_90_percent_sensitivity": False,
                "test_reached_90_percent_sensitivity": point["sensitivity"] >= 0.9,
                "maximum_sensitivity": point["sensitivity"],
                "fpr_per_hour_at_maximum": point["false_positives_per_hour"],
                "sensitivity_interval": point["sensitivity_interval"],
                "fpr_per_hour_interval": point["false_positives_per_hour_interval"],
                "control": point[control_name],
                "control_name": control_name,
                "calibration_point": calibration_point,
            }

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e2_operating_curve.png"
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    for model_name in (*MODEL_NAMES, "danger_margin"):
        series = [row for row in calibration_rows if row["model"] == model_name]
        axis.plot(
            [row["false_positives_per_hour"] for row in series],
            [row["sensitivity"] for row in series],
            marker="o",
            label=f"{model_name} calibration",
        )
        point = headline[model_name]
        axis.scatter(
            [point.get("fpr_per_hour", point.get("fpr_per_hour_at_maximum"))],
            [point.get("sensitivity", point.get("maximum_sensitivity"))],
            marker="x",
            s=60,
            color=axis.lines[-1].get_color(),
        )
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("false alarm episodes / exposure hour")
    axis.set_ylabel("capsize sensitivity (60 s horizon)")
    axis.set_title("E2 — calibration curves and frozen test points")
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
        "operating_point_policy": "Controls are selected on calibration and frozen for test.",
        "calibration_rows": calibration_rows,
        "test_at_calibration_selected_control": headline,
        "figure": str(figure_path),
    }
    write_result(output_root, "e2_operating_curve", payload)
    return payload
