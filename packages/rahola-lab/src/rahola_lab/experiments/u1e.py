"""U1e: causal roll-period stiffness fusion on the three ramp campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rahola.dataset import SimulationDataset
from rahola_lab.constants import SeedBlock
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import (
    campaign_count_summary,
    campaign_family,
    load_split,
    reliability_edges,
    reliability_summary,
    restoring_fit,
    score_dataset,
    tracking_summary,
)
from rahola_lab.splittime import SplitTimeConfig


def _critical_rate_scales(
    dataset: SimulationDataset, *, trailing_window_s: float | None
) -> list[dict[int, np.ndarray]]:
    fit = restoring_fit(dataset)
    reference_period = float(dataset.config["natural_period_s"])
    output = []
    for angle in dataset.angle_rad:
        finite = np.isfinite(angle)
        stop = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(angle)
        crossings = np.flatnonzero(
            (angle[: stop - 1] < fit.equilibrium_angle_rad)
            & (angle[1:stop] >= fit.equilibrium_angle_rad)
        )
        crossing_times = dataset.time_s[crossings + 1]
        kappa = np.ones(len(dataset.time_s), dtype=np.float64)
        for index in range(stop):
            current_time = float(dataset.time_s[index])
            start_time = (
                float(dataset.time_s[0])
                if trailing_window_s is None
                else max(float(dataset.time_s[0]), current_time - trailing_window_s)
            )
            selected = crossing_times[
                (crossing_times >= start_time) & (crossing_times <= current_time)
            ]
            if len(selected) >= 2:
                estimated_period = float(np.mean(np.diff(selected)))
                if estimated_period > 0.0:
                    kappa[index] = (reference_period / estimated_period) ** 2
        kappa[stop:] = kappa[max(0, stop - 1)]

        side_scales = {}
        for side, side_fit in ((1, fit.positive), (-1, fit.negative)):
            growth = fit.damping_coefficient_s + np.sqrt(
                side_fit.repeller_slope * fit.central_omega_rad_s**2 * kappa
                + fit.damping_coefficient_s**2
            )
            side_scales[side] = growth / side_fit.growth_rate_s
        output.append(side_scales)
    return output


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1")
    selected = u1a["selected_controls"]
    quantile = float(selected["tail_quantile"])
    strength = float(selected["prior_strength"])
    prior_means = u1a["calibration_prior_mean_rates"]
    config = SplitTimeConfig(
        tail_quantile=quantile,
        trailing_window_s=selected["trailing_window_s"],
    )
    calibration_fixed = []
    calibration_adaptive = []
    test_fixed = []
    test_adaptive = []
    campaigns = {}
    for family in FAMILIES:
        name = f"{family}_ramp"
        calibration_dataset = load_split(data_root, name, SeedBlock.CALIBRATION)
        test_dataset = load_split(data_root, name, SeedBlock.TEST)
        prior_mean = float(prior_means[str(quantile)][campaign_family(name)])
        fixed_calibration = score_dataset(
            calibration_dataset,
            prior_mean=prior_mean,
            prior_strength=strength,
            config=config,
        )
        adaptive_calibration = score_dataset(
            calibration_dataset,
            prior_mean=prior_mean,
            prior_strength=strength,
            config=config,
            critical_rate_scales=_critical_rate_scales(
                calibration_dataset,
                trailing_window_s=selected["trailing_window_s"],
            ),
        )
        fixed_test = score_dataset(
            test_dataset,
            prior_mean=prior_mean,
            prior_strength=strength,
            config=config,
        )
        adaptive_test = score_dataset(
            test_dataset,
            prior_mean=prior_mean,
            prior_strength=strength,
            config=config,
            critical_rate_scales=_critical_rate_scales(
                test_dataset,
                trailing_window_s=selected["trailing_window_s"],
            ),
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
            "fixed": campaign_count_summary(fixed_test) | fixed_tracking,
            "adaptive": campaign_count_summary(adaptive_test) | adaptive_tracking,
            "tracking_lag_delta_s": (
                float(adaptive_tracking["tracking_lag_s"])
                - float(fixed_tracking["tracking_lag_s"])
            ),
        }
    fixed_reliability = reliability_summary(
        test_fixed, reliability_edges(calibration_fixed)
    )
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
        "experiment": "U1e",
        "variant": (
            "mean causal equilibrium-upcrossing period; kappa_hat=(T_n/T_hat)^2; "
            "piecewise critical growth rate recomputed at each detected crossing"
        ),
        "campaigns": campaigns,
        "ramp_calibration": {
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
    write_result(output_root, "u1e_u1", payload, upstream_results={"u1a_u1": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.output_root)


if __name__ == "__main__":
    main()
