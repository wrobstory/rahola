"""D1 v0.2: frozen normalization policy and trajectory-bootstrap uncertainty."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.constants import EWS_HORIZON_PERIODS, SeedBlock
from rahola_lab.detectors import NormalizationMode
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    decorrelation_times,
    evaluate_suite,
    evaluate_suite_at_thresholds,
    fit_detector_suite,
    fit_frozen_suite,
    merge_scores,
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


def _campaign_names(role: str) -> list[str]:
    return [f"{family}_{source}" for family in FAMILIES for source in (role, "ramp")]


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
    training_names = _campaign_names("stationary")
    training_data = _load(
        historical_root, versioned_root, training_names, SeedBlock.TRAIN
    )
    calibration_data = _load(
        historical_root, versioned_root, training_names, SeedBlock.CALIBRATION
    )
    primary_training = training_windows(
        training_data,
        max_windows_per_trajectory=3,
        normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    primary_calibration = training_windows(
        calibration_data,
        max_windows_per_trajectory=3,
        normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    physical_calibration = training_windows(
        calibration_data,
        max_windows_per_trajectory=3,
        normalization_mode=NormalizationMode.PHYSICAL,
    )
    suite = fit_detector_suite(
        primary_training,
        primary_calibration,
        physics_calibration=physical_calibration,
        cnn_normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    selected = {
        "cnn_grid_index": suite.cnn_grid_index,
        "cnn_parameter_count": suite.cnn.parameter_count(),
        "ews_statistic": suite.ews_statistic,
        "ews_fraction": suite.ews_fraction,
        "neighbor_radius": suite.neighbor_radius,
    }
    calibration_scores = merge_scores(
        [score_dataset(dataset, suite) for dataset in calibration_data]
    )
    grids = threshold_grids(calibration_scores)
    decorrelation = decorrelation_times(calibration_scores)
    calibration_curves = evaluate_suite(calibration_scores, grids, decorrelation)
    calibration_points = select_operating_points(
        calibration_scores, grids, decorrelation
    )

    cumulative_training = training_windows(
        training_data,
        max_windows_per_trajectory=3,
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
    cumulative_calibration_curve = evaluate_suite(
        {"cnn": cumulative_calibration},
        cumulative_grid,
        cumulative_decorrelation,
    )["cnn"]
    cumulative_calibration_point = select_operating_points(
        {"cnn": cumulative_calibration},
        cumulative_grid,
        cumulative_decorrelation,
    )["cnn"]
    del (
        primary_training,
        primary_calibration,
        physical_calibration,
        cumulative_training,
    )
    gc.collect()

    evaluation_names = _campaign_names("evaluation")
    evaluation_data = _load(
        historical_root, versioned_root, evaluation_names, SeedBlock.TEST
    )
    strata = campaign_strata(evaluation_names, evaluation_data)
    scores = merge_scores([score_dataset(dataset, suite) for dataset in evaluation_data])
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
        [score_dataset(dataset, cumulative_suite) for dataset in evaluation_data]
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

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "d1_operating_curves_v02.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    curves = calibration_curves | {"cnn_cumulative_online": cumulative_calibration_curve}
    for name, curve in curves.items():
        axis.plot(
            [point.metrics.false_positives_per_hour for point in curve],
            [point.metrics.sensitivity for point in curve],
            marker=".",
            label=name,
        )
    axis.set_xlabel("declustered false episodes per exposure hour")
    axis.set_ylabel("capsize sensitivity")
    axis.set_ylim(0.0, 1.01)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "D1_v02",
        "selection": suite.selection_rows,
        "selected": selected,
        "normalization_policy": {
            "physics_adjacent_primary": "physical",
            "cnn_primary": "fixed_window_causal",
            "cnn_secondary": "cumulative_online",
            "cumulative_estimand": "normalization-plus-detector system with full-history state",
        },
        "decorrelation_time_s": decorrelation
        | {"cnn_cumulative_online": cumulative_decorrelation["cnn"]},
        "operating_point_policy": (
            "Calibration selects each threshold; test scoring freezes that single policy."
        ),
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
        "figure": str(figure_path),
    }
    write_result(output_root, "d1_operating_curves_v02", payload)
    return payload
