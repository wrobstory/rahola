"""U1a-r2: fresh campaign-count capture and reliability evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.constants import (
    U1_CAMPAIGN_CAPTURE_TARGET,
    U1R2_FAMILY_SCOPE_CLAIM,
    U1R2_PRIOR_STRENGTH,
    U1R2_TAIL_QUANTILE,
)
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.u1_common import (
    adaptive_threshold_fit,
    campaign_count_summary,
    campaign_family,
    reliability_edges,
    reliability_summary,
    score_dataset,
    terminal_severities,
)
from rahola_lab.experiments.u1r2_common import (
    calibration_datasets,
    frozen_tail_priors,
    load_fresh_test,
    score_selected,
    u1a_campaigns,
)
from rahola_lab.splittime import SplitTimeConfig, exponential_rate_mle


def run(data_root: Path, fresh_root: Path, output_root: Path) -> dict[str, object]:
    names = u1a_campaigns()
    calibration = calibration_datasets(data_root)
    priors = frozen_tail_priors(calibration)
    calibration_scores = {
        name: score_selected(dataset, name, priors) for name, dataset in calibration.items()
    }
    frozen_edges = reliability_edges([item for name in names for item in calibration_scores[name]])

    adaptive = {name: adaptive_threshold_fit(dataset) for name, dataset in calibration.items()}
    adaptive_priors = {}
    for name, dataset in calibration.items():
        values = terminal_severities(dataset, fit=adaptive[name][0])
        threshold = min(
            float(np.quantile(values, U1R2_TAIL_QUANTILE)),
            float(np.nextafter(1.0, -np.inf)),
        )
        adaptive_priors[name] = {
            "mean_rate": exponential_rate_mle(values, quantile=U1R2_TAIL_QUANTILE),
            "threshold_w": threshold,
            "exceedance_probability": float(np.mean(values > threshold)),
        }

    test_scores = {}
    summaries = {}
    adaptive_summaries = {}
    for name in names:
        dataset = load_fresh_test(fresh_root, name)
        scores = score_selected(dataset, name, priors)
        test_scores[name] = scores
        summaries[name] = campaign_count_summary(scores, absorbing_events=True)
        prior = adaptive_priors[name]
        adaptive_scores = score_dataset(
            dataset,
            prior_mean=prior["mean_rate"],
            prior_strength=U1R2_PRIOR_STRENGTH,
            prior_threshold_w=prior["threshold_w"],
            prior_exceedance_probability=prior["exceedance_probability"],
            config=SplitTimeConfig(
                tail_quantile=U1R2_TAIL_QUANTILE,
                trailing_window_s=None,
                emission_policy="prior_from_start",
            ),
            fit=adaptive[name][0],
        )
        adaptive_summaries[name] = (
            campaign_count_summary(adaptive_scores, absorbing_events=True) | adaptive[name][1]
        )

    pooled = [item for name in names for item in test_scores[name]]
    reliability = reliability_summary(pooled, frozen_edges)
    capture_count = sum(bool(summary["captures_realized_count"]) for summary in summaries.values())
    family_captures = {
        family: sum(
            bool(summaries[name]["captures_realized_count"])
            for name in names
            if campaign_family(name) == family
        )
        for family in FAMILIES
    }

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "u1a_reliability_u1r2.png"
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    observed = [row["realized_capsize_fraction"] for row in reliability["bins"]]
    predicted = [row["predicted_capsize_fraction"] for row in reliability["bins"]]
    lower = [row["realized_exact_interval"][0] for row in reliability["bins"]]
    upper = [row["realized_exact_interval"][1] for row in reliability["bins"]]
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
    axis.set_xlabel("predicted capsize fraction")
    axis.set_ylabel("realized capsize fraction")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "U1a-r2",
        "selected_controls": {
            "tail_quantile": U1R2_TAIL_QUANTILE,
            "prior_strength": U1R2_PRIOR_STRENGTH,
            "trailing_window_s": None,
            "emission_policy": "prior_from_start",
            "event_accounting": "absorbing_probability",
            "critical_rate": "unforced_eq13",
        },
        "calibration_tail_priors": priors,
        "test_campaigns": summaries,
        "adaptive_threshold_sensitivity": adaptive_summaries,
        "campaign_capture_count": capture_count,
        "family_capture_counts": family_captures,
        "capture_target": U1_CAMPAIGN_CAPTURE_TARGET,
        "criterion_met": capture_count >= U1_CAMPAIGN_CAPTURE_TARGET,
        "family_scope_claim": U1R2_FAMILY_SCOPE_CLAIM,
        "family_scope_observed": {family: family_captures[family] == 2 for family in FAMILIES},
        "reliability": reliability,
        "figure": figure_path.name,
        "prior_dominated": {
            name: {
                "emissions": sum(len(item.rate.emissions) for item in scores),
                "flagged_emissions": sum(
                    "prior_dominated" in emission.flags
                    for item in scores
                    for emission in item.rate.emissions
                ),
            }
            for name, scores in test_scores.items()
        },
    }
    write_result(output_root, "u1a_u1r2", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--fresh-root", type=Path, default=Path("data/u1r2"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.data_root, arguments.fresh_root, arguments.output_root)
    print(
        f"U1a-r2 captures={payload['campaign_capture_count']}/6 "
        f"families={payload['family_capture_counts']}"
    )


if __name__ == "__main__":
    main()
