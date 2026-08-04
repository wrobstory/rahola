"""D5: within-regime discrimination after the sea-state step."""

from __future__ import annotations

import gc
from pathlib import Path

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import DETECTOR_MATCHED_SENSITIVITY, SeedBlock
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite,
    evaluate_suite_at_thresholds,
    fit_frozen_suite,
    point_payload,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
    window_auc,
)

TRANSITION_S = 300.0


def _post(scores: dict[str, list[TrajectoryScores]]) -> dict[str, list[TrajectoryScores]]:
    output = {}
    for name, trajectories in scores.items():
        output[name] = []
        for trajectory in trajectories:
            selected = trajectory.times_s >= TRANSITION_S
            times = trajectory.times_s[selected]
            values = trajectory.scores[selected]
            output[name].append(
                TrajectoryScores(
                    times_s=times,
                    scores=values,
                    record_end_s=trajectory.record_end_s,
                    t_capsize_s=trajectory.t_capsize_s,
                    record_start_s=TRANSITION_S,
                )
            )
    return output


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = load_result(output_root, "d1_operating_curves")
    training_data = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    training = training_windows(training_data, max_windows_per_trajectory=3)
    suite = fit_frozen_suite(training, d1["selected"])
    del training, training_data
    gc.collect()
    step_path = campaign_dir(data_root, "softening_step")
    calibration = _post(score_dataset(load_campaign_split(step_path, SeedBlock.CALIBRATION), suite))
    test = _post(score_dataset(load_campaign_split(step_path, SeedBlock.TEST), suite))
    grids = threshold_grids(calibration)
    decorrelation = decorrelation_times(calibration)
    calibration_curves = evaluate_suite(calibration, grids, decorrelation)
    calibration_points = select_operating_points(calibration, grids, decorrelation)
    thresholds = {name: point.threshold for name, point in calibration_points.items()}
    test_points = evaluate_suite_at_thresholds(test, thresholds, decorrelation)
    methods = {}
    for name, point in test_points.items():
        maximum_sensitivity = max(
            candidate.metrics.sensitivity for candidate in calibration_curves[name]
        )
        methods[name] = {
            **point_payload(point),
            "calibration_operating_point": point_payload(calibration_points[name]),
            "within_regime_auc": window_auc(test[name]),
            "target_sensitivity": DETECTOR_MATCHED_SENSITIVITY,
            "target_attainable": maximum_sensitivity >= DETECTOR_MATCHED_SENSITIVITY,
            "maximum_sensitivity": maximum_sensitivity,
        }
    payload: dict[str, object] = {
        "experiment": "D5",
        "regime": "softening_step times >=300 s",
        "operating_point_policy": "Thresholds are selected on calibration and frozen for test.",
        "methods": methods,
    }
    write_result(
        output_root,
        "d5_within_regime",
        payload,
        upstream_results={"d1_operating_curves": d1},
    )
    return payload
