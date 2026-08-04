"""D2: leave-one-family-out detector generalization."""

from __future__ import annotations

import gc
from pathlib import Path

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    D2_MATERIAL_FPR_REDUCTION,
    DETECTOR_MATCHED_SENSITIVITY,
    SeedBlock,
)
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite_at_thresholds,
    fit_detector_suite,
    merge_scores,
    point_payload,
    relative_fpr_reduction,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
)


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    rotations = []
    for held_out in FAMILIES:
        included = [family for family in FAMILIES if family != held_out]
        training_data = [
            load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
            for family in included
            for role in ("stationary", "ramp")
        ]
        calibration_data = [
            load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.CALIBRATION)
            for family in included
            for role in ("stationary", "ramp")
        ]
        training = training_windows(training_data, max_windows_per_trajectory=3)
        calibration = training_windows(calibration_data, max_windows_per_trajectory=3)
        suite = fit_detector_suite(training, calibration)
        calibration_scores = merge_scores(
            [score_dataset(dataset, suite) for dataset in calibration_data]
        )
        grids = threshold_grids(calibration_scores)
        decorrelation = decorrelation_times(calibration_scores)
        calibration_points = select_operating_points(calibration_scores, grids, decorrelation)
        frozen_thresholds = {
            name: point.threshold for name, point in calibration_points.items()
        }
        del training, calibration, training_data, calibration_data
        gc.collect()

        test_data = [
            load_campaign_split(campaign_dir(data_root, f"{held_out}_evaluation"), SeedBlock.TEST),
            load_campaign_split(campaign_dir(data_root, f"{held_out}_ramp"), SeedBlock.TEST),
        ]
        scores = merge_scores([score_dataset(dataset, suite) for dataset in test_data])
        test_points = evaluate_suite_at_thresholds(scores, frozen_thresholds, decorrelation)
        headline = {name: point_payload(point) for name, point in test_points.items()}
        cnn_fpr = headline["cnn"]["false_episodes_per_hour"]
        b1_fpr = headline["classical_ews"]["false_episodes_per_hour"]
        fpr_reduction = relative_fpr_reduction(cnn_fpr, b1_fpr)
        materially_above_b1 = (
            headline["cnn"]["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
            and headline["classical_ews"]["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
            and fpr_reduction is not None
            and fpr_reduction >= D2_MATERIAL_FPR_REDUCTION
        )
        rotations.append(
            {
                "held_out_family": held_out,
                "cnn_grid_index": suite.cnn_grid_index,
                "calibration_operating_points": {
                    name: point_payload(point) for name, point in calibration_points.items()
                },
                "headline_at_calibration_selected_threshold": headline,
                "cnn_relative_fpr_reduction_vs_b1": fpr_reduction,
                "cnn_materially_above_b1": materially_above_b1,
            }
        )
        del test_data, scores, test_points, suite
        gc.collect()
    payload: dict[str, object] = {
        "experiment": "D2",
        "operating_point_policy": "Thresholds are selected on calibration and frozen for test.",
        "material_fpr_reduction": D2_MATERIAL_FPR_REDUCTION,
        "rotations": rotations,
        "cnn_survives_all_families": all(row["cnn_materially_above_b1"] for row in rotations),
    }
    write_result(output_root, "d2_family_generalization", payload)
    return payload
