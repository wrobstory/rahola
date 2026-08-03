"""Deliberately acausal normalization diagnostic for the 2009 neighbor detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rahola.windowing import binary_auc
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import NEIGHBOR_RADIUS_GRID, SeedBlock
from rahola_lab.detectors import (
    DetectorWindowDataset,
    acausal_whole_record_features,
    extract_detector_windows,
    neighbor_count_scores,
)
from rahola_lab.experiments.common import FAMILIES, subset_dataset, write_result
from rahola_lab.experiments.detector_common import campaign_dir


def _windows(parts: list[DetectorWindowDataset]) -> DetectorWindowDataset:
    return DetectorWindowDataset(
        **{
            field: np.concatenate([getattr(part, field) for part in parts])
            for field in DetectorWindowDataset.__dataclass_fields__
        }
    )


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    """Select on one calibration slice and report once on a disjoint slice."""
    selection_parts = []
    heldout_parts = []
    period_s = 4.0
    for family in FAMILIES:
        for role in ("evaluation", "ramp"):
            dataset = load_campaign_split(
                campaign_dir(data_root, f"{family}_{role}"), SeedBlock.CALIBRATION, limit=128
            )
            period_s = float(dataset.config["natural_period_s"])
            for start, target in ((0, selection_parts), (64, heldout_parts)):
                half = subset_dataset(dataset, start, start + 64)
                windows = extract_detector_windows(
                    half, stride_s=20.0, max_windows_per_trajectory=8
                )
                target.append(
                    DetectorWindowDataset(
                        features=acausal_whole_record_features(half, windows),
                        labels=windows.labels,
                        family_labels=windows.family_labels,
                        trajectory_indices=windows.trajectory_indices,
                        end_times_s=windows.end_times_s,
                        raw_angle_rad=windows.raw_angle_rad,
                        raw_rate_rad_s=windows.raw_rate_rad_s,
                    )
                )
    selection = _windows(selection_parts)
    heldout = _windows(heldout_parts)
    samples_per_period = round(period_s / 0.5)
    choices = []
    for radius in NEIGHBOR_RADIUS_GRID:
        score = neighbor_count_scores(
            selection.features, radius=radius, samples_per_period=samples_per_period
        )
        choices.append((binary_auc(selection.labels, score), radius))
    _, radius = max(choices)
    heldout_score = neighbor_count_scores(
        heldout.features, radius=radius, samples_per_period=samples_per_period
    )
    payload: dict[str, object] = {
        "experiment": "Acausal neighbor appendix",
        "warning": "Deliberately acausal; not an operational detector result.",
        "normalization": "whole finite trajectory mean and standard deviation",
        "selection_windows": len(selection.labels),
        "heldout_windows": len(heldout.labels),
        "selected_radius": radius,
        "heldout_auc": binary_auc(heldout.labels, heldout_score),
        "calibration_split": "first 64 trajectories select; next 64 report, per campaign",
    }
    write_result(output_root, "p3_acausal_neighbor", payload)
    return payload
