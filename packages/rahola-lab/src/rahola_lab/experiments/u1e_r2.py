"""U1e-r2: fresh causal roll-period stiffness fusion on ramp campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rahola_lab.constants import SeedBlock
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import (
    campaign_count_summary,
    load_split,
    reliability_edges,
    reliability_summary,
    tracking_summary,
)
from rahola_lab.experiments.u1e import _critical_rate_scales
from rahola_lab.experiments.u1r2_common import load_fresh_test, score_selected


def run(data_root: Path, fresh_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1r2")
    priors = u1a["calibration_tail_priors"]
    calibration_fixed = []
    calibration_adaptive = []
    test_fixed = []
    test_adaptive = []
    campaigns = {}
    for family in FAMILIES:
        name = f"{family}_ramp"
        calibration_dataset = load_split(data_root, name, SeedBlock.CALIBRATION)
        test_dataset = load_fresh_test(fresh_root, name)
        fixed_calibration = score_selected(calibration_dataset, name, priors)
        adaptive_calibration = score_selected(
            calibration_dataset,
            name,
            priors,
            critical_rate_scales=_critical_rate_scales(calibration_dataset, trailing_window_s=None),
        )
        fixed_test = score_selected(test_dataset, name, priors)
        adaptive_test = score_selected(
            test_dataset,
            name,
            priors,
            critical_rate_scales=_critical_rate_scales(test_dataset, trailing_window_s=None),
        )
        calibration_fixed.extend(fixed_calibration)
        calibration_adaptive.extend(adaptive_calibration)
        test_fixed.extend(fixed_test)
        test_adaptive.extend(adaptive_test)
        grid = np.arange(10.0, float(test_dataset.time_s[-1]) + 1e-9, 10.0)
        fixed_tracking = tracking_summary(
            fixed_test,
            grid,
            natural_period_s=float(test_dataset.config["natural_period_s"]),
        )
        adaptive_tracking = tracking_summary(
            adaptive_test,
            grid,
            natural_period_s=float(test_dataset.config["natural_period_s"]),
        )
        campaigns[name] = {
            "fixed": campaign_count_summary(fixed_test, absorbing_events=True) | fixed_tracking,
            "adaptive": campaign_count_summary(adaptive_test, absorbing_events=True)
            | adaptive_tracking,
            "tracking_lag_delta_s": (
                float(adaptive_tracking["tracking_lag_s"]) - float(fixed_tracking["tracking_lag_s"])
            ),
            "bias_delta_per_hour": (
                float(adaptive_tracking["bias_per_hour"]) - float(fixed_tracking["bias_per_hour"])
            ),
        }
    fixed_reliability = reliability_summary(test_fixed, reliability_edges(calibration_fixed))
    adaptive_reliability = reliability_summary(
        test_adaptive, reliability_edges(calibration_adaptive)
    )
    fixed_captures = sum(
        bool(row["fixed"]["captures_realized_count"]) for row in campaigns.values()
    )
    adaptive_captures = sum(
        bool(row["adaptive"]["captures_realized_count"]) for row in campaigns.values()
    )
    payload: dict[str, object] = {
        "experiment": "U1e-r2",
        "variant": (
            "mean causal equilibrium-upcrossing period; kappa_hat=(T_n/T_hat)^2; "
            "piecewise critical growth rate recomputed at each detected crossing"
        ),
        "campaigns": campaigns,
        "ramp_test": {
            "fixed_capture_count": fixed_captures,
            "adaptive_capture_count": adaptive_captures,
            "capture_count_delta": adaptive_captures - fixed_captures,
            "fixed_reliability_weighted_mean_absolute_error": fixed_reliability[
                "weighted_mean_absolute_error"
            ],
            "adaptive_reliability_weighted_mean_absolute_error": adaptive_reliability[
                "weighted_mean_absolute_error"
            ],
            "reliability_error_delta": (
                float(adaptive_reliability["weighted_mean_absolute_error"])
                - float(fixed_reliability["weighted_mean_absolute_error"])
            ),
        },
    }
    write_result(output_root, "u1e_u1r2", payload, upstream_results={"u1a_u1r2": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--fresh-root", type=Path, default=Path("data/u1r2"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.fresh_root, arguments.output_root)


if __name__ == "__main__":
    main()
