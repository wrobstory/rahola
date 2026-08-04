"""D5 v0.2: fully established post-step within-regime discrimination."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    D5_V02_AUC_LIMIT,
    D5_V02_FIRST_ENDPOINT_S,
    D5_V02_LAST_ENDPOINT_S,
    D5_V02_PREREGISTERED_PREDICTION,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    SeedBlock,
)
from rahola_lab.detectors import NormalizationMode
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    bootstrap_window_auc,
    decorrelation_times,
    evaluate_suite,
    evaluate_suite_at_thresholds,
    fit_frozen_suite,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
)
from rahola_lab.experiments.v02_common import (
    load_campaign_split_v02,
    load_frozen_v02_result,
    point_payload_without_dependent_intervals,
)


def fully_post_step(
    scores: dict[str, list[TrajectoryScores]],
) -> dict[str, list[TrajectoryScores]]:
    """Keep endpoints whose 60-period history starts after the transition."""
    output = {}
    for name, trajectories in scores.items():
        output[name] = []
        for trajectory in trajectories:
            selected = (
                (trajectory.times_s >= D5_V02_FIRST_ENDPOINT_S)
                & (trajectory.times_s <= D5_V02_LAST_ENDPOINT_S)
            )
            output[name].append(
                TrajectoryScores(
                    times_s=trajectory.times_s[selected],
                    scores=trajectory.scores[selected],
                    record_end_s=min(
                        trajectory.record_end_s, D5_V02_LAST_ENDPOINT_S
                    ),
                    t_capsize_s=trajectory.t_capsize_s,
                    record_start_s=D5_V02_FIRST_ENDPOINT_S,
                )
            )
    return output


def _clock_scores(
    template: list[TrajectoryScores],
) -> list[TrajectoryScores]:
    return [
        TrajectoryScores(
            times_s=trajectory.times_s,
            scores=trajectory.times_s.copy(),
            record_end_s=trajectory.record_end_s,
            t_capsize_s=trajectory.t_capsize_s,
            record_start_s=trajectory.record_start_s,
        )
        for trajectory in template
    ]


def run(
    historical_root: Path, versioned_root: Path, output_root: Path
) -> dict[str, object]:
    d1 = load_frozen_v02_result(output_root / "d1_operating_curves_v02.json")
    training_names = [
        f"{family}_{role}"
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    training_data = [
        load_campaign_split_v02(
            historical_root, versioned_root, name, SeedBlock.TRAIN
        )
        for name in training_names
    ]
    primary_training = training_windows(
        training_data,
        normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    primary_suite = fit_frozen_suite(
        primary_training,
        d1["selected"],
        cnn_normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    cumulative_training = training_windows(
        training_data,
        normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
    )
    cumulative_suite = fit_frozen_suite(
        cumulative_training,
        d1["selected"],
        cnn_normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
    )
    del primary_training, cumulative_training, training_data
    gc.collect()

    step_path = versioned_root / "softening_step_v02"
    calibration_dataset = load_campaign_split(
        step_path, SeedBlock.CALIBRATION
    )
    test_dataset = load_campaign_split(step_path, SeedBlock.TEST)
    calibration = fully_post_step(score_dataset(calibration_dataset, primary_suite))
    test = fully_post_step(score_dataset(test_dataset, primary_suite))
    cumulative_calibration = fully_post_step(
        {"cnn_cumulative_online": score_dataset(calibration_dataset, cumulative_suite)["cnn"]}
    )
    cumulative_test = fully_post_step(
        {"cnn_cumulative_online": score_dataset(test_dataset, cumulative_suite)["cnn"]}
    )
    calibration.update(cumulative_calibration)
    test.update(cumulative_test)
    calibration["protocol_clock_only"] = _clock_scores(calibration["cnn"])
    test["protocol_clock_only"] = _clock_scores(test["cnn"])

    for trajectories in test.values():
        endpoints = np.concatenate(
            [trajectory.times_s for trajectory in trajectories if len(trajectory.times_s)]
        )
        if np.min(endpoints) < D5_V02_FIRST_ENDPOINT_S or np.max(
            endpoints
        ) > D5_V02_LAST_ENDPOINT_S:
            raise ValueError("D5_v02 endpoint geometry violated")

    grids = threshold_grids(calibration)
    decorrelation = decorrelation_times(calibration)
    calibration_curves = evaluate_suite(calibration, grids, decorrelation)
    calibration_points = select_operating_points(calibration, grids, decorrelation)
    thresholds = {name: point.threshold for name, point in calibration_points.items()}
    test_points = evaluate_suite_at_thresholds(test, thresholds, decorrelation)
    horizon_s = EWS_HORIZON_PERIODS * 4.0
    methods = {}
    for name, point in test_points.items():
        auc_payload = bootstrap_window_auc(test[name])
        auc = float(auc_payload["auc"])
        maximum_sensitivity = max(
            candidate.metrics.sensitivity for candidate in calibration_curves[name]
        )
        methods[name] = (
            bootstrap_point_payload(
                point,
                test[name],
                horizon_s=horizon_s,
                decorrelation_s=decorrelation[name],
            )
            | auc_payload
            | {
                "orientation_independent_auc": max(auc, 1.0 - auc),
                "calibration_operating_point": point_payload_without_dependent_intervals(
                    calibration_points[name]
                ),
                "target_sensitivity": DETECTOR_MATCHED_SENSITIVITY,
                "target_attainable": maximum_sensitivity
                >= DETECTOR_MATCHED_SENSITIVITY,
                "maximum_sensitivity": maximum_sensitivity,
            }
        )
    motion_names = [name for name in methods if name != "protocol_clock_only"]
    prediction_holds = all(
        float(methods[name]["orientation_independent_auc"]) < D5_V02_AUC_LIMIT
        for name in motion_names
    )
    payload: dict[str, object] = {
        "experiment": "D5_v02",
        "prediction_preregistered_verbatim": D5_V02_PREREGISTERED_PREDICTION,
        "prediction_evaluation_uses_orientation_independent_auc": True,
        "prediction_holds": prediction_holds,
        "regime": {
            "transition_s": 300.0,
            "history_periods": 60.0,
            "horizon_periods": 50.0,
            "first_endpoint_s": D5_V02_FIRST_ENDPOINT_S,
            "last_endpoint_s": D5_V02_LAST_ENDPOINT_S,
        },
        "normalization_policy": {
            "physics_adjacent_primary": "physical",
            "cnn_primary": "fixed_window_causal",
            "cnn_secondary": "cumulative_online normalization-plus-detector system",
        },
        "operating_point_policy": "Calibration selects thresholds; test policies are frozen.",
        "upstream_d1_artifact_sha256": d1["_artifact_sha256"],
        "methods": methods,
    }
    write_result(output_root, "d5_within_regime_v02", payload)
    return payload
