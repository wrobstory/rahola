"""D1: within-distribution detector operating curves."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import SeedBlock
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite,
    fit_detector_suite,
    matched_point,
    merge_scores,
    point_payload,
    score_dataset,
    threshold_grids,
    training_windows,
)


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    training_data = []
    calibration_data = []
    for family in FAMILIES:
        training_data.extend(
            [
                load_campaign_split(
                    campaign_dir(data_root, f"{family}_stationary"), SeedBlock.TRAIN
                ),
                load_campaign_split(campaign_dir(data_root, f"{family}_ramp"), SeedBlock.TRAIN),
            ]
        )
        calibration_data.extend(
            [
                load_campaign_split(
                    campaign_dir(data_root, f"{family}_stationary"), SeedBlock.CALIBRATION
                ),
                load_campaign_split(
                    campaign_dir(data_root, f"{family}_ramp"), SeedBlock.CALIBRATION
                ),
            ]
        )
    training = training_windows(training_data, max_windows_per_trajectory=3)
    calibration = training_windows(calibration_data, max_windows_per_trajectory=3)
    suite = fit_detector_suite(training, calibration)
    calibration_scores = merge_scores(
        [score_dataset(dataset, suite) for dataset in calibration_data]
    )
    grids = threshold_grids(calibration_scores)
    decorrelation = decorrelation_times(calibration_scores)
    del training, calibration, training_data, calibration_data
    gc.collect()

    evaluation_data = []
    for family in FAMILIES:
        evaluation_data.extend(
            [
                load_campaign_split(
                    campaign_dir(data_root, f"{family}_evaluation"), SeedBlock.TEST
                ),
                load_campaign_split(campaign_dir(data_root, f"{family}_ramp"), SeedBlock.TEST),
            ]
        )
    scores = merge_scores([score_dataset(dataset, suite) for dataset in evaluation_data])
    curves = evaluate_suite(scores, grids, decorrelation)
    headline = {name: point_payload(matched_point(curve)) for name, curve in curves.items()}

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "d1_operating_curves.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
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
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    payload: dict[str, object] = {
        "experiment": "D1",
        "selection": suite.selection_rows,
        "selected": {
            "cnn_grid_index": suite.cnn_grid_index,
            "cnn_parameter_count": suite.cnn.parameter_count(),
            "ews_statistic": suite.ews_statistic,
            "ews_fraction": suite.ews_fraction,
            "neighbor_radius": suite.neighbor_radius,
        },
        "decorrelation_time_s": decorrelation,
        "headline_at_90_percent_sensitivity": headline,
        "curves": {
            name: [point_payload(point) for point in curve] for name, curve in curves.items()
        },
        "figure": str(figure_path),
    }
    write_result(output_root, "d1_operating_curves", payload)
    return payload
