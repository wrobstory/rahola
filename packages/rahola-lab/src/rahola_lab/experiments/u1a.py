"""U1a: campaign-level confidence-interval capture and reliability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto, norm

from rahola_lab.constants import (
    U1_CAMPAIGN_CAPTURE_TARGET,
    U1_PRIOR_STRENGTHS,
    U1_TAIL_QUANTILES,
    U1_TRAILING_WINDOWS_S,
    SeedBlock,
)
from rahola_lab.evaluation import clopper_pearson_interval
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.u1_common import (
    adaptive_threshold_fit,
    calibration_prior_means,
    campaign_count_summary,
    campaign_family,
    load_split,
    reliability_edges,
    reliability_summary,
    score_dataset,
    terminal_severities,
)
from rahola_lab.splittime import (
    GammaRatePrior,
    SplitTimeConfig,
    estimate_exponential_tail,
    exponential_rate_mle,
)


def _campaigns() -> list[str]:
    return [
        f"{family}_{role}"
        for family in FAMILIES
        for role in ("stationary", "evaluation")
    ]


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    names = _campaigns()
    calibration_data = {
        name: load_split(data_root, name, SeedBlock.CALIBRATION) for name in names
    }
    prior_means = calibration_prior_means(calibration_data, U1_TAIL_QUANTILES)
    selection_rows: list[dict[str, object]] = []
    candidate_index = 0
    for quantile in U1_TAIL_QUANTILES:
        for strength in U1_PRIOR_STRENGTHS:
            base_config = SplitTimeConfig(
                tail_quantile=quantile,
                trailing_window_s=None,
            )
            scored = {
                name: score_dataset(
                    dataset,
                    prior_mean=prior_means[str(quantile)][campaign_family(name)],
                    prior_strength=strength,
                    config=base_config,
                )
                for name, dataset in calibration_data.items()
            }
            pooled = [item for name in names for item in scored[name]]
            edges = reliability_edges(pooled)
            reliability = reliability_summary(pooled, edges)
            summaries = {
                name: campaign_count_summary(scores) for name, scores in scored.items()
            }
            captures = sum(
                bool(summary["captures_realized_count"])
                for summary in summaries.values()
            )
            for window_s in U1_TRAILING_WINDOWS_S:
                selection_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "tail_quantile": quantile,
                        "prior_strength": strength,
                        "trailing_window_s": window_s,
                        "equivalent_due_to_600_s_record_length": window_s is not None,
                        "campaign_captures": captures,
                        "reliability_weighted_mean_absolute_error": reliability[
                            "weighted_mean_absolute_error"
                        ],
                        "campaigns": summaries,
                    }
                )
                candidate_index += 1
    selected = min(
        selection_rows,
        key=lambda row: (
            -int(row["campaign_captures"]),
            float(row["reliability_weighted_mean_absolute_error"]),
            int(row["candidate_index"]),
        ),
    )
    selected_controls = {
        "tail_quantile": selected["tail_quantile"],
        "prior_strength": selected["prior_strength"],
        "trailing_window_s": selected["trailing_window_s"],
    }

    adaptive = {
        name: adaptive_threshold_fit(dataset) for name, dataset in calibration_data.items()
    }
    adaptive_prior_means = {
        name: exponential_rate_mle(
            terminal_severities(dataset, fit=adaptive[name][0]),
            quantile=float(selected_controls["tail_quantile"]),
        )
        for name, dataset in calibration_data.items()
    }
    test_scores = {}
    test_summaries = {}
    adaptive_summaries = {}
    for name in names:
        dataset = load_split(data_root, name, SeedBlock.TEST)
        scores = score_dataset(
            dataset,
            prior_mean=prior_means[str(selected_controls["tail_quantile"])][
                campaign_family(name)
            ],
            prior_strength=float(selected_controls["prior_strength"]),
            config=SplitTimeConfig(
                tail_quantile=float(selected_controls["tail_quantile"]),
                trailing_window_s=selected_controls["trailing_window_s"],
            ),
        )
        test_scores[name] = scores
        test_summaries[name] = campaign_count_summary(scores)
        adaptive_scores = score_dataset(
            dataset,
            prior_mean=adaptive_prior_means[name],
            prior_strength=float(selected_controls["prior_strength"]),
            config=SplitTimeConfig(
                tail_quantile=float(selected_controls["tail_quantile"]),
                trailing_window_s=selected_controls["trailing_window_s"],
            ),
            fit=adaptive[name][0],
        )
        adaptive_summaries[name] = campaign_count_summary(adaptive_scores) | adaptive[name][1]
    pooled_test = [item for name in names for item in test_scores[name]]
    selected_calibration_by_name = {}
    for name, dataset in calibration_data.items():
        selected_calibration_by_name[name] = score_dataset(
            dataset,
            prior_mean=prior_means[str(selected_controls["tail_quantile"])][
                campaign_family(name)
            ],
            prior_strength=float(selected_controls["prior_strength"]),
            config=SplitTimeConfig(
                tail_quantile=float(selected_controls["tail_quantile"]),
                trailing_window_s=selected_controls["trailing_window_s"],
            ),
        )
    selected_calibration_scores = [
        item for name in names for item in selected_calibration_by_name[name]
    ]
    frozen_edges = reliability_edges(selected_calibration_scores)
    reliability = reliability_summary(pooled_test, frozen_edges)
    capture_count = sum(
        bool(summary["captures_realized_count"])
        for summary in test_summaries.values()
    )
    pooled_terminal = np.concatenate(
        [terminal_severities(dataset) for dataset in calibration_data.values()]
    )
    diagnostic_w = float(
        min(
            np.quantile(pooled_terminal, float(selected_controls["tail_quantile"])),
            np.nextafter(1.0, -np.inf),
        )
    )
    diagnostic_exceedances = pooled_terminal[pooled_terminal > diagnostic_w] - diagnostic_w
    gpd_shape, _, gpd_scale = genpareto.fit(diagnostic_exceedances, floc=0.0)
    comparison_name = "softening_stationary"
    comparison_values = terminal_severities(calibration_data[comparison_name])
    comparison_prior = GammaRatePrior.from_mean(
        float(prior_means[str(selected_controls["tail_quantile"])]["softening"]),
        float(selected_controls["prior_strength"]),
    )
    comparison_tail = estimate_exponential_tail(
        comparison_values,
        quantile=float(selected_controls["tail_quantile"]),
        prior=comparison_prior,
    )
    comparison_dataset = calibration_data[comparison_name]
    dt = float(np.median(np.diff(comparison_dataset.time_s)))
    exposure_s = float(
        sum(item.exposure_end_s for item in selected_calibration_by_name[comparison_name])
    )
    opportunities = round(exposure_s / dt)
    component_confidence = float(np.sqrt(0.95))
    upcross_interval = clopper_pearson_interval(
        comparison_tail.exceedance_count,
        opportunities,
        confidence_level=component_confidence,
    )
    exceedances = (
        comparison_values[comparison_values > comparison_tail.threshold_w]
        - comparison_tail.threshold_w
    )
    beta_hat = float(np.mean(exceedances))
    beta_se = float(np.std(exceedances, ddof=1) / np.sqrt(len(exceedances)))
    beta_confidence = float(np.sqrt(component_confidence))
    beta_z = float(norm.ppf((1.0 + beta_confidence) / 2.0))
    beta_low = max(np.finfo(np.float64).tiny, beta_hat - beta_z * beta_se)
    beta_high = beta_hat + beta_z * beta_se
    distance = 1.0 - comparison_tail.threshold_w
    paper_interval_rate = [
        upcross_interval.lower / dt * np.exp(-distance / beta_low) * 3_600.0,
        upcross_interval.upper / dt * np.exp(-distance / beta_high) * 3_600.0,
    ]
    paper_interval_count = [
        value * exposure_s / 3_600.0 for value in paper_interval_rate
    ]
    parametric_comparison = campaign_count_summary(
        selected_calibration_by_name[comparison_name]
    )["predicted_count_interval"]

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "u1a_reliability_u1.png"
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
    axis.plot([0.0, limit], [0.0, limit], color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("predicted capsize fraction")
    axis.set_ylabel("realized capsize fraction")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "U1a",
        "selected_controls": selected_controls,
        "calibration_prior_mean_rates": prior_means,
        "calibration_selection": selection_rows,
        "test_campaigns": test_summaries,
        "adaptive_threshold_sensitivity": adaptive_summaries,
        "campaign_capture_count": capture_count,
        "capture_target": U1_CAMPAIGN_CAPTURE_TARGET,
        "criterion_met": capture_count >= U1_CAMPAIGN_CAPTURE_TARGET,
        "reliability": reliability,
        "figure": figure_path.name,
        "gpd_diagnostic": {
            "label": "diagnostic only; never used in lambda_hat",
            "threshold_w": diagnostic_w,
            "shape": float(gpd_shape),
            "scale": float(gpd_scale),
            "exceedance_count": len(diagnostic_exceedances),
        },
        "component_interval_methods_note": {
            "campaign": comparison_name,
            "paper_sqrt_confidence_predicted_count_interval": paper_interval_count,
            "parametric_bootstrap_predicted_count_interval": parametric_comparison,
            "interval_width_difference_paper_minus_bootstrap": (
                paper_interval_count[1]
                - paper_interval_count[0]
                - float(parametric_comparison[1])
                + float(parametric_comparison[0])
            ),
            "note": (
                "The paper interval composes binomial upcrossing and exponential-scale "
                "boundaries at square-root confidence; U1's primary interval resamples "
                "the composed rate."
            ),
        },
        "interval_method": {
            "per_emission": "parametric bootstrap",
            "campaign_count": "sum of independent per-trajectory draw vectors",
        },
        "paper_departures": [
            {
                "section": "3.2-3.5",
                "departure": (
                    "uses the ROM closed-form critical rate; omits future-wave motion "
                    "perturbation"
                ),
            },
            {
                "section": "4.7",
                "departure": "uses calibration-frozen empirical quantiles and Gamma shrinkage",
            },
            {
                "section": "5.2",
                "departure": (
                    "uses a parametric bootstrap instead of component-boundary composition"
                ),
            },
        ],
    }
    write_result(output_root, "u1a_u1", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.data_root, arguments.output_root)
    print(
        f"U1a captures={payload['campaign_capture_count']}/6 "
        f"reliability_mae={payload['reliability']['weighted_mean_absolute_error']:.6f}"
    )


if __name__ == "__main__":
    main()
