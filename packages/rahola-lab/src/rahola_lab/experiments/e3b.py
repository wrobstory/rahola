"""E3b: DtACI and recent-score recalibration through the frozen sea-state step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import (
    dynamically_tuned_aci_bounds,
    sliding_recalibrated_aci_bounds,
)
from rahola_lab.constants import (
    ACI_EXPLOSION_FACTOR,
    ACI_EXPLOSION_FPR_PER_HOUR,
    ACI_GAMMA_GRID,
    DTACI_GAMMA_EXPERTS,
    SLIDING_RECALIBRATION_WINDOWS,
    SeedBlock,
)
from rahola_lab.experiments.common import (
    campaign_path,
    fit_forecasters,
    snapshot,
    subset_dataset,
    trajectory_forecasts,
    write_result,
)
from rahola_lab.experiments.e3 import (
    ALPHA,
    HORIZON_S,
    MODEL_NAME,
    TRANSITION_S,
    _bounds,
    _coverage_summary,
    _episode_metrics,
    _recovery_time,
    _rolling_coverage,
)


def _dtaci_bounds(streams, scores):
    output = []
    for stream in streams:
        result = dynamically_tuned_aci_bounds(
            scores,
            stream.raw_upper_rad[MODEL_NAME],
            stream.targets_rad,
            alpha=ALPHA,
            gamma_experts=DTACI_GAMMA_EXPERTS,
        )
        output.append((stream, result.upper_bounds, result.errors))
    return output


def _sliding_bounds(streams, scores, gamma, window_size):
    output = []
    for stream in streams:
        result = sliding_recalibrated_aci_bounds(
            scores,
            stream.raw_upper_rad[MODEL_NAME],
            stream.targets_rad,
            alpha=ALPHA,
            gamma=gamma,
            window_size=window_size,
        )
        output.append((stream, result.upper_bounds, result.errors))
    return output


def _method_summary(bounded, fixed_fpr, escape_angle):
    times, coverage = _rolling_coverage(bounded)
    metrics = _episode_metrics(bounded, escape_angle)
    recovery = _recovery_time(times, coverage)
    exploded = (
        metrics.false_positives_per_hour > ACI_EXPLOSION_FPR_PER_HOUR
        and metrics.false_positives_per_hour > ACI_EXPLOSION_FACTOR * fixed_fpr
    )
    return {
        "times": times,
        "coverage": coverage,
        "metrics": metrics,
        "recovery_time_s": recovery,
        "exploded": exploded,
        "passed": recovery is not None and not exploded,
    }


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    training = load_campaign_split(campaign_path(data_root, "softening", "stationary"), "train")
    models = fit_forecasters(training, HORIZON_S)
    step_path = data_root / "softening_step"
    calibration = load_campaign_split(step_path, SeedBlock.CALIBRATION)
    score_half = subset_dataset(calibration, 0, calibration.batch_size // 2)
    tuning_half = subset_dataset(calibration, calibration.batch_size // 2, calibration.batch_size)
    calibration_y, calibration_raw = snapshot(score_half, models, HORIZON_S, history_end_s=240.0)
    scores = calibration_y - calibration_raw[MODEL_NAME]
    tuning_streams = trajectory_forecasts(
        tuning_half, {MODEL_NAME: models[MODEL_NAME]}, HORIZON_S, stride_s=10.0
    )
    escape_angle = float(calibration.config["escape_angle_rad"])
    fixed_tuning = _bounds(tuning_streams, scores)
    fixed_tuning_fpr = _episode_metrics(fixed_tuning, escape_angle).false_positives_per_hour

    sliding_candidates = []
    for gamma in ACI_GAMMA_GRID:
        for window_size in SLIDING_RECALIBRATION_WINDOWS:
            bounded = _sliding_bounds(tuning_streams, scores, gamma, window_size)
            summary = _method_summary(bounded, fixed_tuning_fpr, escape_angle)
            post = summary["coverage"][summary["times"] >= TRANSITION_S]
            sliding_candidates.append(
                {
                    "gamma": gamma,
                    "window_size": window_size,
                    "post_transition_mean_absolute_coverage_delta_pp": 100.0
                    * float(abs(post - (1.0 - ALPHA)).mean()),
                    "false_positives_per_hour": summary["metrics"].false_positives_per_hour,
                    "recovery_time_s": summary["recovery_time_s"],
                    "exploded": summary["exploded"],
                    "passed": summary["passed"],
                }
            )
    selected = min(
        sliding_candidates,
        key=lambda row: (
            not row["passed"],
            row["exploded"],
            row["post_transition_mean_absolute_coverage_delta_pp"],
        ),
    )

    # E3b's only test load; all adapter choices above used calibration trajectories.
    test = load_campaign_split(step_path, SeedBlock.TEST)
    streams = trajectory_forecasts(test, {MODEL_NAME: models[MODEL_NAME]}, HORIZON_S, stride_s=10.0)
    fixed = _bounds(streams, scores)
    fixed_summary = _method_summary(fixed, 0.0, escape_angle)
    fixed_fpr = fixed_summary["metrics"].false_positives_per_hour
    dtaci = _dtaci_bounds(streams, scores)
    dtaci_summary = _method_summary(dtaci, fixed_fpr, escape_angle)
    sliding = _sliding_bounds(
        streams, scores, float(selected["gamma"]), int(selected["window_size"])
    )
    sliding_summary = _method_summary(sliding, fixed_fpr, escape_angle)

    def public_summary(bounded, summary):
        metrics = summary["metrics"]
        return {
            "post_transition": _coverage_summary(bounded, TRANSITION_S, None),
            "recovery_time_s": summary["recovery_time_s"],
            "false_positives_per_hour": metrics.false_positives_per_hour,
            "false_positives_per_hour_interval": [
                metrics.false_positives_per_hour_interval.lower,
                metrics.false_positives_per_hour_interval.upper,
            ],
            "exploded": summary["exploded"],
            "passed": summary["passed"],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e3b_adapters.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.plot(fixed_summary["times"], fixed_summary["coverage"], label="fixed split-CQR")
    axis.plot(dtaci_summary["times"], dtaci_summary["coverage"], label="DtACI")
    axis.plot(
        sliding_summary["times"],
        sliding_summary["coverage"],
        label=f"sliding ACI, gamma={selected['gamma']:g}, W={selected['window_size']}",
    )
    axis.axvline(TRANSITION_S, color="black", linestyle=":", label="sea-state step")
    axis.axhline(1.0 - ALPHA, color="black", linestyle="--", linewidth=1)
    axis.fill_between(
        fixed_summary["times"],
        1.0 - ALPHA - 0.03,
        1.0 - ALPHA + 0.03,
        color="black",
        alpha=0.08,
    )
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("time in trajectory (s)")
    axis.set_ylabel("trailing 60 s empirical coverage")
    axis.set_title("E3b — online adapters through a sea-state step")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "E3b",
        "alpha": ALPHA,
        "frozen_kill_thresholds": {
            "absolute_fpr_per_hour": ACI_EXPLOSION_FPR_PER_HOUR,
            "relative_to_fixed": ACI_EXPLOSION_FACTOR,
            "rolling_coverage_tolerance_pp": 3.0,
        },
        "dtaci_gamma_experts": list(DTACI_GAMMA_EXPERTS),
        "sliding_candidate_grid": sliding_candidates,
        "selected_sliding": {
            "gamma": selected["gamma"],
            "window_size": selected["window_size"],
        },
        "fixed": public_summary(fixed, fixed_summary),
        "dtaci": public_summary(dtaci, dtaci_summary),
        "sliding_aci": public_summary(sliding, sliding_summary),
        "figure": str(figure_path),
    }
    write_result(output_root, "e3b_adapters", payload)
    return payload
