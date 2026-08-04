"""D2 v0.2: leave-one-family-out transfer under frozen preprocessing modes."""

from __future__ import annotations

import gc
from pathlib import Path

from rahola_lab.constants import (
    D2_MATERIAL_FPR_REDUCTION,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    SeedBlock,
)
from rahola_lab.detectors import NormalizationMode
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    decorrelation_times,
    evaluate_suite_at_thresholds,
    fit_detector_suite,
    fit_frozen_suite,
    merge_scores,
    relative_fpr_reduction,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
)
from rahola_lab.experiments.v02_common import (
    campaign_strata,
    load_campaign_split_v02,
    point_payload_without_dependent_intervals,
)


def _load(
    historical_root: Path,
    versioned_root: Path,
    names: list[str],
    block: SeedBlock,
):
    return [
        load_campaign_split_v02(historical_root, versioned_root, name, block)
        for name in names
    ]


def run(
    historical_root: Path, versioned_root: Path, output_root: Path
) -> dict[str, object]:
    rotations = []
    for held_out in FAMILIES:
        included = [family for family in FAMILIES if family != held_out]
        source_names = [
            f"{family}_{role}"
            for family in included
            for role in ("stationary", "ramp")
        ]
        training_data = _load(
            historical_root, versioned_root, source_names, SeedBlock.TRAIN
        )
        calibration_data = _load(
            historical_root, versioned_root, source_names, SeedBlock.CALIBRATION
        )
        primary_training = training_windows(
            training_data,
            normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        primary_calibration = training_windows(
            calibration_data,
            normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        physics_calibration = training_windows(
            calibration_data,
            normalization_mode=NormalizationMode.PHYSICAL,
        )
        suite = fit_detector_suite(
            primary_training,
            primary_calibration,
            physics_calibration=physics_calibration,
            cnn_normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        selected = {
            "cnn_grid_index": suite.cnn_grid_index,
            "ews_statistic": suite.ews_statistic,
            "ews_fraction": suite.ews_fraction,
            "neighbor_radius": suite.neighbor_radius,
        }
        calibration_scores = merge_scores(
            [score_dataset(dataset, suite) for dataset in calibration_data]
        )
        grids = threshold_grids(calibration_scores)
        decorrelation = decorrelation_times(calibration_scores)
        calibration_points = select_operating_points(
            calibration_scores, grids, decorrelation
        )

        cumulative_training = training_windows(
            training_data,
            normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
        )
        cumulative_suite = fit_frozen_suite(
            cumulative_training,
            selected,
            cnn_normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
        )
        cumulative_calibration = merge_scores(
            [score_dataset(dataset, cumulative_suite) for dataset in calibration_data]
        )["cnn"]
        cumulative_grid = threshold_grids({"cnn": cumulative_calibration})
        cumulative_decorrelation = decorrelation_times(
            {"cnn": cumulative_calibration}
        )
        cumulative_calibration_point = select_operating_points(
            {"cnn": cumulative_calibration},
            cumulative_grid,
            cumulative_decorrelation,
        )["cnn"]
        del (
            primary_training,
            primary_calibration,
            physics_calibration,
            cumulative_training,
        )
        gc.collect()

        test_names = [f"{held_out}_evaluation", f"{held_out}_ramp"]
        test_data = _load(
            historical_root, versioned_root, test_names, SeedBlock.TEST
        )
        strata = campaign_strata(test_names, test_data)
        scores = merge_scores([score_dataset(dataset, suite) for dataset in test_data])
        thresholds = {name: point.threshold for name, point in calibration_points.items()}
        test_points = evaluate_suite_at_thresholds(scores, thresholds, decorrelation)
        horizon_s = EWS_HORIZON_PERIODS * 4.0
        headline = {
            name: bootstrap_point_payload(
                point,
                scores[name],
                horizon_s=horizon_s,
                decorrelation_s=decorrelation[name],
                campaign_strata=strata,
            )
            for name, point in test_points.items()
        }
        cumulative_test = merge_scores(
            [score_dataset(dataset, cumulative_suite) for dataset in test_data]
        )["cnn"]
        cumulative_point = evaluate_suite_at_thresholds(
            {"cnn": cumulative_test},
            {"cnn": cumulative_calibration_point.threshold},
            cumulative_decorrelation,
        )["cnn"]
        headline["cnn_cumulative_online"] = bootstrap_point_payload(
            cumulative_point,
            cumulative_test,
            horizon_s=horizon_s,
            decorrelation_s=cumulative_decorrelation["cnn"],
            campaign_strata=strata,
        )
        cnn_fpr = float(headline["cnn"]["false_episodes_per_hour"])
        baseline_fpr = float(
            headline["classical_ews"]["false_episodes_per_hour"]
        )
        reduction = relative_fpr_reduction(cnn_fpr, baseline_fpr)
        materially_above = (
            float(headline["cnn"]["sensitivity"])
            >= DETECTOR_MATCHED_SENSITIVITY
            and float(headline["classical_ews"]["sensitivity"])
            >= DETECTOR_MATCHED_SENSITIVITY
            and reduction is not None
            and reduction >= D2_MATERIAL_FPR_REDUCTION
        )
        rotations.append(
            {
                "held_out_family": held_out,
                "selected": selected,
                "normalization_policy": {
                    "physics_adjacent_primary": "physical",
                    "cnn_primary": "fixed_window_causal",
                    "cnn_secondary": "cumulative_online",
                },
                "calibration_operating_points": {
                    name: point_payload_without_dependent_intervals(point)
                    for name, point in calibration_points.items()
                }
                | {
                    "cnn_cumulative_online": point_payload_without_dependent_intervals(
                        cumulative_calibration_point
                    )
                },
                "headline_at_calibration_selected_threshold": headline,
                "cnn_relative_fpr_reduction_vs_b1": reduction,
                "cnn_materially_above_b1": materially_above,
            }
        )
        del training_data, calibration_data, test_data, scores, suite, cumulative_suite
        gc.collect()
    payload: dict[str, object] = {
        "experiment": "D2_v02",
        "operating_point_policy": "Calibration-selected thresholds are frozen for test.",
        "material_fpr_reduction": D2_MATERIAL_FPR_REDUCTION,
        "rotations": rotations,
        "cnn_survives_all_families": all(
            row["cnn_materially_above_b1"] for row in rotations
        ),
    }
    write_result(output_root, "d2_family_generalization_v02", payload)
    return payload
