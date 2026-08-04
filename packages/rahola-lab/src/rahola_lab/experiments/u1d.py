"""U1d: online-component baselines and the predeclared kill criterion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset
from rahola_lab.constants import (
    TRAJECTORY_BOOTSTRAP_REPLICATES,
    TRAJECTORY_BOOTSTRAP_SEED,
    SeedBlock,
)
from rahola_lab.evaluation import clopper_pearson_interval, trajectory_block_bootstrap
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.u1_common import (
    ScoredRateTrajectory,
    campaign_family,
    load_split,
    restoring_fit,
    score_dataset,
)
from rahola_lab.splittime import (
    SplitTimeConfig,
    decluster_crossings,
    detect_crossings,
    roll_decorrelation_time,
)


@dataclass(frozen=True)
class BaselineRow:
    campaign: str
    capsized: bool
    exposure_hours: float
    rolling_variance: float
    upcrossing_rate_per_hour: float
    tail_factor: float


def _trajectory_metrics(
    dataset: SimulationDataset,
    campaign: str,
    scores: list[ScoredRateTrajectory],
    *,
    trailing_window_s: float | None,
) -> list[BaselineRow]:
    fit = restoring_fit(dataset)
    dt = float(np.median(np.diff(dataset.time_s)))
    rows = []
    for index, score in enumerate(scores):
        angle = dataset.angle_rad[index]
        rate = dataset.rate_rad_s[index]
        finite = np.isfinite(angle) & np.isfinite(rate)
        stop = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(angle)
        end_s = float(dataset.time_s[max(0, stop - 1)])
        starts = np.arange(10.0, end_s + 1e-9, 10.0)
        variances = []
        for end in starts:
            start_s = 0.0 if trailing_window_s is None else max(0.0, end - trailing_window_s)
            left = int(np.searchsorted(dataset.time_s[:stop], start_s, side="left"))
            right = int(np.searchsorted(dataset.time_s[:stop], end, side="right"))
            if right - left >= 2:
                variances.append(float(np.var(angle[left:right], ddof=1)))
        decorrelation = roll_decorrelation_time(angle[:stop], dt) if stop >= 3 else dt
        retained = decluster_crossings(
            detect_crossings(dataset.time_s, angle, rate, fit), decorrelation
        )
        exposure_hours = end_s / 3_600.0
        terminal = score.rate.emissions[-1] if score.rate.emissions else None
        rows.append(
            BaselineRow(
                campaign=campaign,
                capsized=score.capsized,
                exposure_hours=exposure_hours,
                rolling_variance=float(np.mean(variances)) if variances else 0.0,
                upcrossing_rate_per_hour=(
                    len(retained) / exposure_hours if exposure_hours > 0.0 else 0.0
                ),
                tail_factor=0.0 if terminal is None else terminal.critical_probability,
            )
        )
    return rows


def _isotonic_fit(x: NDArray[np.floating], y: NDArray[np.floating]):
    order = np.argsort(x, kind="stable")
    xs = np.asarray(x, dtype=np.float64)[order]
    ys = np.asarray(y, dtype=np.float64)[order]
    blocks: list[list[float]] = []
    for value, target in zip(xs, ys, strict=True):
        blocks.append([float(value), float(value), float(target), 1.0])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[3] + right[3]
            blocks.append(
                [left[0], right[1], (left[2] * left[3] + right[2] * right[3]) / weight, weight]
            )
    knots = np.asarray([block[1] for block in blocks], dtype=np.float64)
    values = np.asarray([block[2] for block in blocks], dtype=np.float64)

    def predict(query: NDArray[np.floating]) -> NDArray[np.float64]:
        indices = np.searchsorted(knots, np.asarray(query, dtype=np.float64), side="left")
        return values[np.clip(indices, 0, len(values) - 1)]

    return predict


def _reliability(
    rows: list[BaselineRow], rates: NDArray[np.floating], edges: NDArray[np.floating]
) -> dict[str, object]:
    values = np.asarray(rates, dtype=np.float64)
    assignments = np.clip(np.digitize(values, np.asarray(edges)[1:-1]), 0, len(edges) - 2)
    bins = []
    error = 0.0
    for index in range(len(edges) - 1):
        selected = np.flatnonzero(assignments == index)
        if not len(selected):
            continue
        realized_count = sum(rows[int(item)].capsized for item in selected)
        realized = realized_count / len(selected)
        predicted = float(
            np.mean(
                [
                    -np.expm1(-values[item] * rows[int(item)].exposure_hours)
                    for item in selected
                ]
            )
        )
        interval = clopper_pearson_interval(realized_count, len(selected))
        error += len(selected) * abs(predicted - realized)
        bins.append(
            {
                "count": len(selected),
                "predicted_capsize_fraction": predicted,
                "realized_capsize_fraction": realized,
                "realized_exact_interval": [interval.lower, interval.upper],
            }
        )
    return {"bins": bins, "weighted_mean_absolute_error": error / len(rows)}


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
            [test_rates[index] * test_rows[index].exposure_hours for index in selected]
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
            "predicted_count_trajectory_bootstrap_interval": [float(lower), float(upper)],
            "realized_count": realized,
            "captures_realized_count": lower <= realized <= upper,
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


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1")
    selected = u1a["selected_controls"]
    prior_means = u1a["calibration_prior_mean_rates"]
    quantile = float(selected["tail_quantile"])
    strength = float(selected["prior_strength"])
    names = [
        f"{family}_{role}"
        for family in FAMILIES
        for role in ("stationary", "evaluation")
    ]
    calibration_rows: list[BaselineRow] = []
    test_rows: list[BaselineRow] = []
    for name in names:
        for block, destination in (
            (SeedBlock.CALIBRATION, calibration_rows),
            (SeedBlock.TEST, test_rows),
        ):
            dataset = load_split(data_root, name, block)
            scores = score_dataset(
                dataset,
                prior_mean=float(prior_means[str(quantile)][campaign_family(name)]),
                prior_strength=strength,
                config=SplitTimeConfig(
                    tail_quantile=quantile,
                    trailing_window_s=selected["trailing_window_s"],
                ),
            )
            destination.extend(
                _trajectory_metrics(
                    dataset,
                    name,
                    scores,
                    trailing_window_s=selected["trailing_window_s"],
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
        np.asarray([row.upcrossing_rate_per_hour for row in calibration_rows]), target_rate
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
        and full_error
        < methods["rolling_variance"]["reliability"]["weighted_mean_absolute_error"]
    )
    beats_upcrossing = (
        full_captures > methods["upcrossing_rate_alone"]["campaign_capture_count"]
        and full_error
        < methods["upcrossing_rate_alone"]["reliability"]["weighted_mean_absolute_error"]
    )
    kill_fired = not (beats_variance and beats_upcrossing)
    payload: dict[str, object] = {
        "experiment": "U1d",
        "full_decomposition": {
            "campaign_capture_count": full_captures,
            "reliability_weighted_mean_absolute_error": full_error,
        },
        "baselines": methods,
        "kill_fired": kill_fired,
        "kill_criterion": (
            "If the full decomposition does not outperform both rolling variance and "
            "declustered upcrossing rate alone on campaign-level CI captures and on the "
            "bin-count-weighted mean absolute reliability error, the split-time decomposition "
            "adds nothing online beyond its components — report that negative as the result "
            "and stop tuning."
        ),
    }
    write_result(output_root, "u1d_u1", payload, upstream_results={"u1a_u1": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.output_root)


if __name__ == "__main__":
    main()
