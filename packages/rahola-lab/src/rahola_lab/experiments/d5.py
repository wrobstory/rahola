"""D5: within-regime discrimination after the sea-state step."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import DETECTOR_MATCHED_SENSITIVITY, SeedBlock
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite,
    fit_frozen_suite,
    matched_point,
    point_payload,
    score_dataset,
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
            if not len(times):
                times = np.array([TRANSITION_S])
                values = np.array([-np.inf])
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
    d1 = json.loads((output_root / "d1_operating_curves.json").read_text(encoding="utf-8"))
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
    curves = evaluate_suite(test, grids, decorrelation)
    methods = {}
    for name, curve in curves.items():
        maximum_sensitivity = max(point.metrics.sensitivity for point in curve)
        methods[name] = {
            **point_payload(matched_point(curve)),
            "within_regime_auc": window_auc(test[name]),
            "target_sensitivity": DETECTOR_MATCHED_SENSITIVITY,
            "target_attainable": maximum_sensitivity >= DETECTOR_MATCHED_SENSITIVITY,
            "maximum_sensitivity": maximum_sensitivity,
        }
    payload: dict[str, object] = {
        "experiment": "D5",
        "regime": "softening_step times >=300 s",
        "methods": methods,
    }
    write_result(output_root, "d5_within_regime", payload)
    return payload
