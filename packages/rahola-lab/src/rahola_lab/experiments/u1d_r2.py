"""U1d-r2: fresh component baselines and frozen kill criterion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import (
    TRAJECTORY_BOOTSTRAP_REPLICATES,
    TRAJECTORY_BOOTSTRAP_SEED,
    U1R2_KILL_CRITERION,
    SeedBlock,
)
from rahola_lab.evaluation import trajectory_block_bootstrap
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import load_split
from rahola_lab.experiments.u1d import (
    BaselineRow,
    _isotonic_fit,
    _reliability,
    _trajectory_metrics,
)
from rahola_lab.experiments.u1r2_common import load_fresh_test, score_selected


def _method_summary(
    calibration_rows: list[BaselineRow],
    calibration_rates: NDArray[np.floating],
    test_rows: list[BaselineRow],
    test_rates: NDArray[np.floating],
) -> dict[str, object]:
    edges = np.quantile(calibration_rates, np.linspace(0.0, 1.0, 6))
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    reliability = _reliability(test_rows, test_rates, edges)
    reliability_interval = trajectory_block_bootstrap(
        list(range(len(test_rows))),
        lambda indices: _reliability(
            [test_rows[index] for index in indices],
            np.asarray([test_rates[index] for index in indices]),
            edges,
        )["weighted_mean_absolute_error"],
        replicates=TRAJECTORY_BOOTSTRAP_REPLICATES,
        seed=TRAJECTORY_BOOTSTRAP_SEED,
    )
    campaigns = {}
    for campaign in sorted({row.campaign for row in test_rows}):
        selected = [index for index, row in enumerate(test_rows) if row.campaign == campaign]
        contributions = np.asarray(
            [-np.expm1(-test_rates[index] * test_rows[index].exposure_hours) for index in selected]
        )
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
        lower, upper = np.quantile(draws, [0.025, 0.975])
        realized = sum(test_rows[index].capsized for index in selected)
        campaigns[campaign] = {
            "predicted_count": float(np.sum(contributions)),
            "predicted_count_trajectory_bootstrap_interval": [
                float(lower),
                float(upper),
            ],
            "realized_count": realized,
            "captures_realized_count": lower <= realized <= upper,
            "event_accounting": "absorbing_probability",
        }
    return {
        "campaigns": campaigns,
        "campaign_capture_count": sum(
            bool(row["captures_realized_count"]) for row in campaigns.values()
        ),
        "reliability": reliability
        | {
            "weighted_mean_absolute_error_trajectory_bootstrap_interval": [
                float(reliability_interval.lower[0]),
                float(reliability_interval.upper[0]),
            ],
            "edges_rate_per_hour": edges.tolist(),
        },
    }


def run(data_root: Path, fresh_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1r2")
    priors = u1a["calibration_tail_priors"]
    names = [f"{family}_{role}" for family in FAMILIES for role in ("stationary", "evaluation")]
    calibration_rows: list[BaselineRow] = []
    test_rows: list[BaselineRow] = []
    for name in names:
        calibration_dataset = load_split(data_root, name, SeedBlock.CALIBRATION)
        calibration_rows.extend(
            _trajectory_metrics(
                calibration_dataset,
                name,
                score_selected(calibration_dataset, name, priors),
                trailing_window_s=None,
            )
        )
        test_dataset = load_fresh_test(fresh_root, name)
        test_rows.extend(
            _trajectory_metrics(
                test_dataset,
                name,
                score_selected(test_dataset, name, priors),
                trailing_window_s=None,
            )
        )

    exposure = np.asarray([row.exposure_hours for row in calibration_rows])
    target_rate = np.asarray(
        [
            float(row.capsized) / hours if hours > 0.0 else 0.0
            for row, hours in zip(calibration_rows, exposure, strict=True)
        ]
    )
    variance_model = _isotonic_fit(
        np.asarray([row.rolling_variance for row in calibration_rows]), target_rate
    )
    upcross_model = _isotonic_fit(
        np.asarray([row.upcrossing_rate_per_hour for row in calibration_rows]),
        target_rate,
    )
    calibration_campaign_rate = {
        campaign: float(
            np.mean(
                [
                    row.upcrossing_rate_per_hour
                    for row in calibration_rows
                    if row.campaign == campaign
                ]
            )
        )
        for campaign in names
    }
    method_rates = {
        "rolling_variance": (
            variance_model(np.asarray([row.rolling_variance for row in calibration_rows])),
            variance_model(np.asarray([row.rolling_variance for row in test_rows])),
        ),
        "upcrossing_rate_alone": (
            upcross_model(np.asarray([row.upcrossing_rate_per_hour for row in calibration_rows])),
            upcross_model(np.asarray([row.upcrossing_rate_per_hour for row in test_rows])),
        ),
        "tail_factor_alone": (
            np.asarray(
                [
                    calibration_campaign_rate[row.campaign] * row.tail_factor
                    for row in calibration_rows
                ]
            ),
            np.asarray(
                [calibration_campaign_rate[row.campaign] * row.tail_factor for row in test_rows]
            ),
        ),
    }
    methods = {
        name: _method_summary(calibration_rows, calibration, test_rows, test)
        for name, (calibration, test) in method_rates.items()
    }
    full_captures = int(u1a["campaign_capture_count"])
    full_error = float(u1a["reliability"]["weighted_mean_absolute_error"])
    beats_variance = (
        full_captures > methods["rolling_variance"]["campaign_capture_count"]
        and full_error < methods["rolling_variance"]["reliability"]["weighted_mean_absolute_error"]
    )
    beats_upcrossing = (
        full_captures > methods["upcrossing_rate_alone"]["campaign_capture_count"]
        and full_error
        < methods["upcrossing_rate_alone"]["reliability"]["weighted_mean_absolute_error"]
    )
    payload: dict[str, object] = {
        "experiment": "U1d-r2",
        "full_decomposition": {
            "campaign_capture_count": full_captures,
            "reliability_weighted_mean_absolute_error": full_error,
        },
        "baselines": methods,
        "beats_rolling_variance_on_both_metrics": beats_variance,
        "beats_upcrossing_rate_on_both_metrics": beats_upcrossing,
        "kill_fired": not (beats_variance and beats_upcrossing),
        "kill_criterion": U1R2_KILL_CRITERION,
    }
    write_result(output_root, "u1d_u1r2", payload, upstream_results={"u1a_u1r2": u1a})
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
