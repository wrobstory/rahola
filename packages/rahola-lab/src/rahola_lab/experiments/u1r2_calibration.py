"""U1-r2 Phase B: calibration-only control selection under repaired emission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rahola_lab.constants import U1_PRIOR_STRENGTHS, U1_TAIL_QUANTILES, SeedBlock
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.u1_common import (
    calibration_tail_priors,
    campaign_count_summary,
    campaign_family,
    load_split,
    reliability_edges,
    reliability_summary,
    score_dataset,
)
from rahola_lab.splittime import SplitTimeConfig


def _campaigns() -> list[str]:
    return [f"{family}_{role}" for family in FAMILIES for role in ("stationary", "evaluation")]


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    names = _campaigns()
    datasets = {name: load_split(data_root, name, SeedBlock.CALIBRATION) for name in names}
    priors = calibration_tail_priors(datasets, U1_TAIL_QUANTILES)
    rows: list[dict[str, object]] = []
    candidate_index = 0
    for quantile in U1_TAIL_QUANTILES:
        for strength in U1_PRIOR_STRENGTHS:
            scored = {}
            for name, dataset in datasets.items():
                prior = priors[str(quantile)][campaign_family(name)]
                scored[name] = score_dataset(
                    dataset,
                    prior_mean=prior["mean_rate"],
                    prior_strength=strength,
                    prior_threshold_w=prior["threshold_w"],
                    prior_exceedance_probability=prior["exceedance_probability"],
                    config=SplitTimeConfig(
                        tail_quantile=quantile,
                        trailing_window_s=None,
                        emission_policy="prior_from_start",
                    ),
                )
            summaries = {
                name: campaign_count_summary(scores, absorbing_events=True)
                for name, scores in scored.items()
            }
            pooled = [item for name in names for item in scored[name]]
            edges = reliability_edges(pooled)
            reliability = reliability_summary(pooled, edges)
            captures = sum(
                bool(summary["captures_realized_count"]) for summary in summaries.values()
            )
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "tail_quantile": quantile,
                    "prior_strength": strength,
                    "trailing_window_s": None,
                    "campaign_captures": captures,
                    "family_captures": {
                        family: sum(
                            bool(summaries[name]["captures_realized_count"])
                            for name in names
                            if campaign_family(name) == family
                        )
                        for family in FAMILIES
                    },
                    "reliability_weighted_mean_absolute_error": reliability[
                        "weighted_mean_absolute_error"
                    ],
                    "campaigns": summaries,
                }
            )
            candidate_index += 1
    selected = min(
        rows,
        key=lambda row: (
            -int(row["campaign_captures"]),
            float(row["reliability_weighted_mean_absolute_error"]),
            int(row["candidate_index"]),
        ),
    )
    phase_a = json.loads((output_root / "u1_phase_a_u1r2.json").read_text(encoding="utf-8"))
    attribution = phase_a["softening_overshoot_attribution"]
    table = attribution["attribution_table"]
    payload: dict[str, object] = {
        "experiment": "U1-r2 calibration selection",
        "information_boundary": "calibration blocks only",
        "phase_a_artifact_sha256": phase_a["_artifact_sha256"],
        "decisions": {
            "emission_policy": "prior_from_start; no three-exceedance validity gate",
            "campaign_event_accounting": "sum of per-trajectory absorbing probabilities",
            "event_accounting_justification": {
                "critical_crossing_prediction": table[0]["predicted_count"],
                "absorbing_event_prediction": table[1]["predicted_count"],
                "absolute_shift": table[1]["magnitude_vs_baseline"],
                "relative_shift": table[1]["magnitude_vs_baseline"] / table[0]["predicted_count"],
            },
            "critical_rate": "unforced Eq. 13 primary",
            "critical_rate_justification": attribution["forced_correction"],
            "trailing_window_s": None,
        },
        "tail_priors": priors,
        "selection_rows": rows,
        "selected_controls": {
            "tail_quantile": selected["tail_quantile"],
            "prior_strength": selected["prior_strength"],
            "trailing_window_s": None,
            "campaign_captures": selected["campaign_captures"],
            "family_captures": selected["family_captures"],
            "reliability_weighted_mean_absolute_error": selected[
                "reliability_weighted_mean_absolute_error"
            ],
        },
        "family_scope_claim": (
            "The decomposition yields calibrated counts on softening-type campaigns; "
            "on parametric and biased campaigns its prior-dominated estimate is expected "
            "to remain uninformative, and that scope limit — if observed — is the finding."
        ),
    }
    write_result(output_root, "u1_calibration_selection_u1r2", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.data_root, arguments.output_root)
    selected = payload["selected_controls"]
    print(
        f"U1-r2 calibration q={selected['tail_quantile']} "
        f"a0={selected['prior_strength']} captures={selected['campaign_captures']}/6 "
        f"reliability_mae={selected['reliability_weighted_mean_absolute_error']:.6f}"
    )


if __name__ == "__main__":
    main()
