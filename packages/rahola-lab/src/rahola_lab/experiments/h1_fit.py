"""Fit and freeze H1's offline conditionals, intercepts, and comparators."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rahola_lab.campaigns import verify_h1_test_slices
from rahola_lab.constants import H1_EXPECTED_CAPSIZE_FLOOR, SeedBlock
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.h1_common import (
    H1Score,
    TrajectoryObservation,
    fit_conditional_model,
    fit_intercept,
    fit_rate_map,
    observe_dataset,
    score_hybrid,
)
from rahola_lab.experiments.u1_common import load_split


def _reliability_edges(scores: list[H1Score], bins: int = 5) -> list[float]:
    values = np.asarray([score.average_rate_per_hour for score in scores], dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    if edges[-1] == edges[0]:
        width = max(1e-12, abs(edges[0]) * 1e-9)
        edges = np.linspace(edges[0] - width, edges[-1] + width, bins + 1)
    else:
        edges[0] = np.nextafter(edges[0], -np.inf)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges.tolist()


def _partition_summary(rows: list[TrajectoryObservation]) -> dict[str, object]:
    capsizes = sum(row.capsized for row in rows)
    heralded = sum(row.heralded for row in rows)
    unheralded = sum(row.unheralded for row in rows)
    return {
        "trajectory_count": len(rows),
        "capsizes": capsizes,
        "heralded_capsizes": heralded,
        "unheralded_capsizes": unheralded,
        "unheralded_fraction_of_capsizes": unheralded / capsizes if capsizes else 0.0,
        "unheralded_sampling_gap_signatures": sum(
            row.sampling_gap_signature for row in rows if row.unheralded
        ),
    }


def run(
    data_root: Path,
    versioned_root: Path,
    u1r2_root: Path,
    output_root: Path,
) -> dict[str, object]:
    primary_datasets = {}
    primary_observations: dict[str, list[TrajectoryObservation]] = {
        family: [] for family in FAMILIES
    }
    pooled_observations: dict[str, list[TrajectoryObservation]] = {
        family: [] for family in FAMILIES
    }
    for family in FAMILIES:
        stationary = f"{family}_stationary"
        for block in (SeedBlock.TRAIN, SeedBlock.CALIBRATION):
            dataset = load_split(data_root, stationary, block)
            primary_datasets[(stationary, str(block))] = dataset
            rows = observe_dataset(dataset, stationary)
            primary_observations[family].extend(rows)
            pooled_observations[family].extend(rows)
        evaluation = f"{family}_evaluation"
        dataset = load_split(data_root, evaluation, SeedBlock.CALIBRATION)
        pooled_observations[family].extend(observe_dataset(dataset, evaluation))

    conditional_models = {
        "primary": {
            family: fit_conditional_model(primary_observations[family], rms_terciles=False)
            for family in FAMILIES
        },
        "secondary": {
            family: fit_conditional_model(pooled_observations[family], rms_terciles=False)
            for family in FAMILIES
        },
        "tertiary": {
            family: fit_conditional_model(pooled_observations[family], rms_terciles=True)
            for family in FAMILIES
        },
    }
    intercepts = {
        "primary": {family: fit_intercept(primary_observations[family]) for family in FAMILIES},
        "secondary": {family: fit_intercept(pooled_observations[family]) for family in FAMILIES},
    }
    intercepts["tertiary"] = intercepts["secondary"]

    comparator_maps = {"crossing_rate_only": {}, "rolling_variance": {}}
    for family in FAMILIES:
        rows = primary_observations[family]
        target = np.asarray(
            [float(row.capsized) / row.exposure_hours for row in rows], dtype=np.float64
        )
        comparator_maps["crossing_rate_only"][family] = fit_rate_map(
            np.asarray([row.crossing_rate_per_hour for row in rows]), target
        )
        comparator_maps["rolling_variance"][family] = fit_rate_map(
            np.asarray([row.rolling_variance for row in rows]), target
        )

    fit_scores = {"hybrid": [], "crossing_rate_only": [], "rolling_variance": []}
    for family in FAMILIES:
        for (campaign, _), dataset in primary_datasets.items():
            if not campaign.startswith(f"{family}_"):
                continue
            fit_scores["hybrid"].extend(
                score_hybrid(
                    dataset,
                    conditional_models["primary"][family],
                    intercepts["primary"][family],
                )
            )
        for row in primary_observations[family]:
            crossing_rate = comparator_maps["crossing_rate_only"][family].predict(
                row.crossing_rate_per_hour
            )
            variance_rate = comparator_maps["rolling_variance"][family].predict(
                row.rolling_variance
            )
            for method, rate in (
                ("crossing_rate_only", crossing_rate),
                ("rolling_variance", variance_rate),
            ):
                fit_scores[method].append(
                    H1Score(
                        seed=row.seed,
                        capsized=row.capsized,
                        exposure_hours=row.exposure_hours,
                        integrated_count=rate * row.exposure_hours,
                        integrated_count_draws=np.empty(0, dtype=np.float64),
                    )
                )

    slice_report = verify_h1_test_slices((data_root, versioned_root, u1r2_root))
    below_floor = {
        name: value
        for name, value in slice_report["expected_capsizes"].items()
        if value < H1_EXPECTED_CAPSIZE_FLOOR
    }
    if below_floor:
        raise AssertionError(f"H1 slices miss the expected-capsize floor: {below_floor}")

    primary_unheralded = [
        row for family in FAMILIES for row in primary_observations[family] if row.unheralded
    ]
    if len(primary_unheralded) != 18:
        raise AssertionError(
            f"expected 18 primary-fit observability gaps, found {len(primary_unheralded)}"
        )
    if not all(row.sampling_gap_signature for row in primary_unheralded):
        raise AssertionError("an H1 observability gap lacks the predeclared sampling signature")

    payload: dict[str, object] = {
        "experiment": "H1 offline fit and preflight",
        "test_data_accessed": False,
        "fit_hierarchy": {
            "primary": "stationary train+calibration only",
            "secondary": "stationary train+calibration plus rare-event calibration",
            "tertiary": ("secondary data plus causal trailing-30-minute roll-RMS terciles"),
        },
        "terminal_partition": {
            "rule": (
                "A heralded capsize occurs before the next retained crossing on either side "
                "and within one decorrelation time of the cluster's last raw crossing; every "
                "other capsize is unheralded. The channels are exhaustive and exclusive."
            ),
            "primary_by_family": {
                family: _partition_summary(primary_observations[family]) for family in FAMILIES
            },
            "pooled_by_family": {
                family: _partition_summary(pooled_observations[family]) for family in FAMILIES
            },
            "primary_unheralded_seeds": [row.seed for row in primary_unheralded],
            "sampling_gap_signature_verified": True,
        },
        "models": {
            variant: {family: model.to_payload() for family, model in family_models.items()}
            for variant, family_models in conditional_models.items()
        },
        "intercepts": {
            variant: {
                family: intercept.to_payload() for family, intercept in family_intercepts.items()
            }
            for variant, family_intercepts in intercepts.items()
        },
        "comparators": {
            method: {family: model.to_payload() for family, model in maps.items()}
            for method, maps in comparator_maps.items()
        },
        "reliability_edges_rate_per_hour": {
            method: _reliability_edges(scores) for method, scores in fit_scores.items()
        },
        "fresh_test_predeclaration": slice_report,
        "expected_capsize_floor_per_campaign": H1_EXPECTED_CAPSIZE_FLOOR,
    }
    write_result(output_root, "h1_offline_fit_h1", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--versioned-root", type=Path, default=Path("data/reference_v02"))
    parser.add_argument("--u1r2-root", type=Path, default=Path("data/u1r2"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(
        arguments.data_root,
        arguments.versioned_root,
        arguments.u1r2_root,
        arguments.output_root,
    )
    print(
        "H1 offline fit calibration-only "
        f"gaps={len(payload['terminal_partition']['primary_unheralded_seeds'])}"
    )


if __name__ == "__main__":
    main()
