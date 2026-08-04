"""U1b: nonstationary ramp tracking and sea-state step response."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.constants import (
    U1_SETTLING_RELATIVE_TOLERANCE,
    U1_TRAILING_WINDOWS_S,
    SeedBlock,
)
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import (
    campaign_family,
    ensemble_rate_and_hazard,
    load_split,
    score_dataset,
    tracking_summary,
)
from rahola_lab.splittime import SplitTimeConfig


def _window_label(window_s: float | None) -> str:
    return "full_history" if window_s is None else f"{int(window_s)}_s"


def _settling_time(
    estimated: np.ndarray,
    hazard: np.ndarray,
    grid_s: np.ndarray,
    *,
    transition_s: float,
) -> float | None:
    selected = grid_s >= transition_s
    tolerance = U1_SETTLING_RELATIVE_TOLERANCE * np.maximum(np.abs(hazard), 1e-9)
    within = np.abs(estimated - hazard) <= tolerance
    indices = np.flatnonzero(selected)
    for index in indices:
        if np.all(within[index:]):
            return float(grid_s[index] - transition_s)
    return None


def run(
    data_root: Path,
    versioned_root: Path,
    output_root: Path,
) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1")
    selected = u1a["selected_controls"]
    prior_means = u1a["calibration_prior_mean_rates"]
    quantile = float(selected["tail_quantile"])
    strength = float(selected["prior_strength"])

    ramp_results: dict[str, object] = {}
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=False)
    for axis, family in zip(axes.flat[:3], FAMILIES, strict=True):
        name = f"{family}_ramp"
        dataset = load_split(data_root, name, SeedBlock.TEST)
        grid = np.arange(10.0, float(dataset.time_s[-1]) + 1e-9, 10.0)
        rows = {}
        for window_s in U1_TRAILING_WINDOWS_S:
            scores = score_dataset(
                dataset,
                prior_mean=float(prior_means[str(quantile)][campaign_family(name)]),
                prior_strength=strength,
                config=SplitTimeConfig(
                    tail_quantile=quantile,
                    trailing_window_s=window_s,
                ),
            )
            rows[_window_label(window_s)] = tracking_summary(
                scores,
                grid,
                natural_period_s=float(dataset.config["natural_period_s"]),
            )
            estimated, hazard = ensemble_rate_and_hazard(
                scores,
                grid,
                natural_period_s=float(dataset.config["natural_period_s"]),
            )
            axis.plot(grid, estimated, label=f"lambda {_window_label(window_s)}")
        axis.plot(grid, hazard, color="black", linewidth=2, label="empirical hazard")
        axis.set_title(name)
        axis.set_ylabel("events per hour")
        axis.grid(alpha=0.2)
        ramp_results[name] = rows

    step_results = {}
    step_specs = [
        ("softening_step", data_root),
        ("softening_step_v02", versioned_root),
    ]
    step_axis = axes.flat[3]
    for name, root in step_specs:
        dataset = load_split(root, name, SeedBlock.TEST)
        grid = np.arange(10.0, float(dataset.time_s[-1]) + 1e-9, 10.0)
        scores = score_dataset(
            dataset,
            prior_mean=float(prior_means[str(quantile)]["softening"]),
            prior_strength=strength,
            config=SplitTimeConfig(
                tail_quantile=quantile,
                trailing_window_s=selected["trailing_window_s"],
            ),
        )
        estimated, hazard = ensemble_rate_and_hazard(
            scores,
            grid,
            natural_period_s=float(dataset.config["natural_period_s"]),
        )
        transition_s = float(dataset.config["protocol"]["steps"][0]["time_s"])
        step_results[name] = tracking_summary(
            scores,
            grid,
            natural_period_s=float(dataset.config["natural_period_s"]),
        ) | {
            "transition_s": transition_s,
            "settling_time_s": _settling_time(
                estimated,
                hazard,
                grid,
                transition_s=transition_s,
            ),
            "fully_post_step_segment_reported": name == "softening_step_v02",
        }
        step_axis.plot(grid, estimated, label=f"{name} lambda")
        step_axis.plot(grid, hazard, linestyle="--", label=f"{name} hazard")
    step_axis.set_title("sea-state steps")
    step_axis.set_xlabel("time (s)")
    step_axis.set_ylabel("events per hour")
    step_axis.grid(alpha=0.2)
    for axis in axes.flat:
        axis.legend(fontsize=7)
    figure.tight_layout()
    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "u1b_tracking_u1.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "U1b",
        "selected_controls": selected,
        "ramps": ramp_results,
        "steps": step_results,
        "geometry_caveat": (
            "With 600-second records and a 300-second step, no fully post-step trailing "
            "window of 30 minutes exists; U1b reports transient tracking only and makes "
            "no established-regime claim."
        ),
        "figure": figure_path.name,
    }
    write_result(output_root, "u1b_u1", payload, upstream_results={"u1a_u1": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--versioned-root", type=Path, default=Path("data/reference_v02"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.versioned_root, arguments.output_root)


if __name__ == "__main__":
    main()
