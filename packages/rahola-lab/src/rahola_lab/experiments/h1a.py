"""H1a: one-shot evaluation of the offline-conditional hybrid estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.campaigns import H1_TEST_SLICES, h1_name, load_campaign_split
from rahola_lab.constants import (
    H1_CALIBRATION_FAILURE,
    H1_CALIBRATION_SUCCESS,
    H1_CAMPAIGN_CAPTURE_TARGET,
    H1_TRANSFER_FAILURE,
    H1_TRANSFER_SUCCESS,
    H1_VALUE_ADDED_FAILURE,
    H1_VALUE_ADDED_SUCCESS,
    TRAJECTORY_BOOTSTRAP_REPLICATES,
    TRAJECTORY_BOOTSTRAP_SEED,
    SeedBlock,
)
from rahola_lab.evaluation import clopper_pearson_interval
from rahola_lab.experiments.common import FAMILIES, _artifact_digest, write_result
from rahola_lab.experiments.h1_common import (
    H1Score,
    conditional_model_from_payload,
    intercept_from_payload,
    observe_dataset,
    rate_map_from_payload,
    score_hybrid,
    score_rate_map,
)


def _load_frozen_fit(output_root: Path) -> dict[str, object]:
    path = output_root / "h1_offline_fit_h1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("_artifact_sha256") != _artifact_digest(payload):
        raise ValueError("H1 offline-fit artifact digest mismatch")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("H1 offline fit does not attest calibration-only access")
    return payload


def _campaign_summary(scores: list[H1Score], *, parametric: bool) -> dict[str, object]:
    contributions = np.asarray(
        [score.predicted_capsize_probability for score in scores], dtype=np.float64
    )
    predicted = float(np.sum(contributions))
    if parametric:
        draws = np.sum(
            -np.expm1(-np.stack([score.integrated_count_draws for score in scores])),
            axis=0,
        )
    else:
        rng = np.random.default_rng(TRAJECTORY_BOOTSTRAP_SEED)
        draws = np.sum(
            contributions[
                rng.integers(
                    0,
                    len(contributions),
                    size=(TRAJECTORY_BOOTSTRAP_REPLICATES, len(contributions)),
                )
            ],
            axis=1,
        )
    lower, upper = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    realized = sum(score.capsized for score in scores)
    return {
        "trajectory_count": len(scores),
        "predicted_capsize_count": predicted,
        "predicted_count_interval": [lower, upper],
        "realized_capsize_count": realized,
        "captures_realized_count": lower <= realized <= upper,
        "event_accounting": "absorbing_probability",
        "interval_method": (
            "Poisson/Wilson/intercept parametric draws" if parametric else "trajectory bootstrap"
        ),
    }


def _reliability(scores: list[H1Score], edges: list[float]) -> dict[str, object]:
    edge_values = np.asarray(edges, dtype=np.float64)
    rates = np.asarray([score.average_rate_per_hour for score in scores])
    assignments = np.clip(np.digitize(rates, edge_values[1:-1]), 0, len(edge_values) - 2)
    bins = []
    error = 0.0
    for index in range(len(edge_values) - 1):
        selected = np.flatnonzero(assignments == index)
        if not len(selected):
            continue
        realized = sum(scores[int(item)].capsized for item in selected)
        observed = realized / len(selected)
        predicted = float(
            np.mean([scores[int(item)].predicted_capsize_probability for item in selected])
        )
        interval = clopper_pearson_interval(realized, len(selected))
        error += len(selected) * abs(predicted - observed)
        bins.append(
            {
                "bin": index,
                "count": len(selected),
                "mean_rate_per_hour": float(np.mean(rates[selected])),
                "predicted_capsize_fraction": predicted,
                "realized_capsize_fraction": observed,
                "realized_exact_interval": [interval.lower, interval.upper],
            }
        )
    return {
        "edges_rate_per_hour": edges,
        "bins": bins,
        "weighted_mean_absolute_error": error / len(scores),
    }


def _capture_breakdown(campaigns: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "pooled": sum(bool(row["captures_realized_count"]) for row in campaigns.values()),
        "by_family": {
            family: sum(
                bool(row["captures_realized_count"])
                for name, row in campaigns.items()
                if name.startswith(f"{family}_")
            )
            for family in FAMILIES
        },
        "by_severity": {
            role: sum(
                bool(row["captures_realized_count"])
                for name, row in campaigns.items()
                if name.endswith(f"_{role}")
            )
            for role in ("stationary", "evaluation")
        },
    }


def run(fresh_root: Path, output_root: Path) -> dict[str, object]:
    frozen = _load_frozen_fit(output_root)
    variants = ("primary", "secondary", "tertiary")
    models = {
        variant: {
            family: conditional_model_from_payload(frozen["models"][variant][family])
            for family in FAMILIES
        }
        for variant in variants
    }
    intercepts = {
        variant: {
            family: intercept_from_payload(frozen["intercepts"][variant][family])
            for family in FAMILIES
        }
        for variant in variants
    }
    comparator_maps = {
        method: {
            family: rate_map_from_payload(frozen["comparators"][method][family])
            for family in FAMILIES
        }
        for method in ("crossing_rate_only", "rolling_variance")
    }

    datasets = {
        name: load_campaign_split(fresh_root / h1_name(name), SeedBlock.TEST)
        for name in H1_TEST_SLICES
    }
    observations = {name: observe_dataset(dataset, name) for name, dataset in datasets.items()}
    variant_scores: dict[str, dict[str, list[H1Score]]] = {variant: {} for variant in variants}
    for variant in variants:
        for name, dataset in datasets.items():
            family = name.split("_", maxsplit=1)[0]
            variant_scores[variant][name] = score_hybrid(
                dataset, models[variant][family], intercepts[variant][family]
            )

    method_scores: dict[str, dict[str, list[H1Score]]] = {
        "hybrid": variant_scores["primary"],
        "crossing_rate_only": {},
        "rolling_variance": {},
    }
    for name, rows in observations.items():
        family = name.split("_", maxsplit=1)[0]
        method_scores["crossing_rate_only"][name] = score_rate_map(
            rows,
            comparator_maps["crossing_rate_only"][family],
            predictor="crossing_rate_per_hour",
        )
        method_scores["rolling_variance"][name] = score_rate_map(
            rows,
            comparator_maps["rolling_variance"][family],
            predictor="rolling_variance",
        )

    campaign_results = {
        method: {
            name: _campaign_summary(scores, parametric=method == "hybrid")
            for name, scores in campaigns.items()
        }
        for method, campaigns in method_scores.items()
    }
    sensitivity_results = {
        variant: {
            name: _campaign_summary(scores, parametric=True)
            for name, scores in variant_scores[variant].items()
        }
        for variant in ("secondary", "tertiary")
    }
    capture_results = {
        method: _capture_breakdown(campaigns) for method, campaigns in campaign_results.items()
    }
    reliability = {}
    for method, campaigns in method_scores.items():
        pooled = [score for scores in campaigns.values() for score in scores]
        reliability[method] = _reliability(
            pooled, frozen["reliability_edges_rate_per_hour"][method]
        )

    hybrid_captures = int(capture_results["hybrid"]["pooled"])
    crossing_captures = int(capture_results["crossing_rate_only"]["pooled"])
    hybrid_error = float(reliability["hybrid"]["weighted_mean_absolute_error"])
    crossing_error = float(reliability["crossing_rate_only"]["weighted_mean_absolute_error"])
    calibration_met = hybrid_captures >= H1_CAMPAIGN_CAPTURE_TARGET
    value_added = hybrid_captures > crossing_captures and hybrid_error < crossing_error
    transfer_met = int(capture_results["hybrid"]["by_severity"]["evaluation"]) == 3

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "h1_reliability_h1.png"
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharex=True, sharey=True)
    for axis, method in zip(axes, method_scores, strict=True):
        rows = reliability[method]["bins"]
        predicted = [row["predicted_capsize_fraction"] for row in rows]
        observed = [row["realized_capsize_fraction"] for row in rows]
        lower = [row["realized_exact_interval"][0] for row in rows]
        upper = [row["realized_exact_interval"][1] for row in rows]
        axis.errorbar(
            predicted,
            observed,
            yerr=[
                [value - bound for value, bound in zip(observed, lower, strict=True)],
                [bound - value for value, bound in zip(observed, upper, strict=True)],
            ],
            fmt="o",
            capsize=3,
        )
        limit = max([0.01, *predicted, *observed])
        axis.plot([0.0, limit], [0.0, limit], color="black", linestyle="--")
        axis.set_title(method.replace("_", " "))
        axis.set_xlabel("predicted capsize fraction")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("realized capsize fraction")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    test_partition = {
        family: {
            "capsizes": sum(
                row.capsized
                for name, rows in observations.items()
                if name.startswith(f"{family}_")
                for row in rows
            ),
            "heralded_capsizes": sum(
                row.heralded
                for name, rows in observations.items()
                if name.startswith(f"{family}_")
                for row in rows
            ),
            "unheralded_capsizes": sum(
                row.unheralded
                for name, rows in observations.items()
                if name.startswith(f"{family}_")
                for row in rows
            ),
        }
        for family in FAMILIES
    }
    for row in test_partition.values():
        row["unheralded_fraction_of_capsizes"] = (
            row["unheralded_capsizes"] / row["capsizes"] if row["capsizes"] else 0.0
        )

    payload: dict[str, object] = {
        "experiment": "H1a",
        "frozen_offline_fit_artifact_sha256": frozen["_artifact_sha256"],
        "methods": campaign_results,
        "capture_breakdown": capture_results,
        "reliability": reliability,
        "hybrid_sensitivities": {
            variant: {
                "campaigns": rows,
                "capture_breakdown": _capture_breakdown(rows),
            }
            for variant, rows in sensitivity_results.items()
        },
        "test_terminal_partition_by_family": test_partition,
        "verdicts": {
            "calibration": H1_CALIBRATION_SUCCESS if calibration_met else H1_CALIBRATION_FAILURE,
            "calibration_criterion_met": calibration_met,
            "value_added": H1_VALUE_ADDED_SUCCESS if value_added else H1_VALUE_ADDED_FAILURE,
            "value_added_criterion_met": value_added,
            "severity_transfer": H1_TRANSFER_SUCCESS if transfer_met else H1_TRANSFER_FAILURE,
            "severity_transfer_criterion_met": transfer_met,
        },
        "figure": figure_path.name,
    }
    write_result(output_root, "h1a_h1", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", type=Path, default=Path("data/h1"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.fresh_root, arguments.output_root)
    print(
        f"H1a hybrid captures={payload['capture_breakdown']['hybrid']['pooled']}/6 "
        f"verdict={payload['verdicts']['value_added']}"
    )


if __name__ == "__main__":
    main()
