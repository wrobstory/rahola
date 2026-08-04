"""U1b-r2: fresh ramp tracking and sea-state step response."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import (
    campaign_count_summary,
    ensemble_rate_and_hazard,
    tracking_summary,
)
from rahola_lab.experiments.u1b import _settling_time
from rahola_lab.experiments.u1r2_common import load_fresh_test, score_selected


def run(fresh_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1r2")
    priors = u1a["calibration_tail_priors"]

    ramp_results: dict[str, object] = {}
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=False)
    for axis, family in zip(axes.flat[:3], FAMILIES, strict=True):
        name = f"{family}_ramp"
        dataset = load_fresh_test(fresh_root, name)
        scores = score_selected(dataset, name, priors)
        grid = np.arange(10.0, float(dataset.time_s[-1]) + 1e-9, 10.0)
        tracking = tracking_summary(
            scores,
            grid,
            natural_period_s=float(dataset.config["natural_period_s"]),
        )
        ramp_results[name] = campaign_count_summary(scores, absorbing_events=True) | tracking
        estimated, hazard = ensemble_rate_and_hazard(
            scores,
            grid,
            natural_period_s=float(dataset.config["natural_period_s"]),
        )
        axis.plot(grid, estimated, label="lambda full_history")
        axis.plot(grid, hazard, color="black", linewidth=2, label="empirical hazard")
        axis.set_title(name)
        axis.set_ylabel("events per hour")
        axis.grid(alpha=0.2)

    step_results = {}
    step_axis = axes.flat[3]
    for name in ("softening_step", "softening_step_v02"):
        dataset = load_fresh_test(fresh_root, name)
        scores = score_selected(dataset, name, priors)
        grid = np.arange(10.0, float(dataset.time_s[-1]) + 1e-9, 10.0)
        estimated, hazard = ensemble_rate_and_hazard(
            scores,
            grid,
            natural_period_s=float(dataset.config["natural_period_s"]),
        )
        transition_s = float(dataset.config["protocol"]["steps"][0]["time_s"])
        step_results[name] = (
            campaign_count_summary(scores, absorbing_events=True)
            | tracking_summary(
                scores,
                grid,
                natural_period_s=float(dataset.config["natural_period_s"]),
            )
            | {
                "transition_s": transition_s,
                "settling_time_s": _settling_time(
                    estimated,
                    hazard,
                    grid,
                    transition_s=transition_s,
                ),
                "fully_post_step_segment_reported": name == "softening_step_v02",
            }
        )
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
    figure_path = output_root / "u1b_tracking_u1r2.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "U1b-r2",
        "selected_controls": u1a["selected_controls"],
        "ramps": ramp_results,
        "steps": step_results,
        "geometry_caveat": (
            "The 600-second step campaign supports transient tracking only; the "
            "900-second v0.2 geometry also reports its frozen fully post-step segment."
        ),
        "figure": figure_path.name,
    }
    write_result(output_root, "u1b_u1r2", payload, upstream_results={"u1a_u1r2": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", type=Path, default=Path("data/u1r2"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.fresh_root, arguments.output_root)


if __name__ == "__main__":
    main()
