"""E1: stationary finite-sample coverage validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import SplitCQRUpper
from rahola_lab.constants import FORECAST_HORIZONS_S, SeedBlock
from rahola_lab.evaluation import clopper_pearson_interval
from rahola_lab.experiments.common import (
    FAMILIES,
    MODEL_NAMES,
    campaign_path,
    fit_forecasters,
    snapshot,
    write_result,
)


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    alphas = np.array([0.02, 0.05, 0.1, 0.2], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        path = campaign_path(data_root, family, "stationary")
        train = load_campaign_split(path, SeedBlock.TRAIN)
        calibration = load_campaign_split(path, SeedBlock.CALIBRATION)
        # Pseudo-prospective seal: this is the experiment's only test load.
        test = load_campaign_split(path, SeedBlock.TEST)
        for horizon_s in FORECAST_HORIZONS_S:
            models = fit_forecasters(train, horizon_s)
            calibration_y, calibration_raw = snapshot(
                calibration, models, horizon_s, history_end_s=180.0
            )
            test_y, test_raw = snapshot(test, models, horizon_s, history_end_s=180.0)
            for model_name in MODEL_NAMES:
                conformal = SplitCQRUpper.calibrate(calibration_y, calibration_raw[model_name])
                for alpha in alphas:
                    bound = conformal.upper_bound(test_raw[model_name], float(alpha))
                    misses = int(np.sum(test_y > bound))
                    rate = misses / len(test_y)
                    interval = clopper_pearson_interval(len(test_y) - misses, len(test_y))
                    rows.append(
                        {
                            "family": family,
                            "horizon_s": horizon_s,
                            "model": model_name,
                            "alpha": float(alpha),
                            "exceedance_rate": rate,
                            "coverage": 1.0 - rate,
                            "coverage_interval": [interval.lower, interval.upper],
                            "n": len(test_y),
                            "coverage_delta_pp": 100.0 * ((1.0 - rate) - (1.0 - alpha)),
                            "coverage_delta_interval_pp": [
                                100.0 * (interval.lower - (1.0 - alpha)),
                                100.0 * (interval.upper - (1.0 - alpha)),
                            ],
                        }
                    )

    figure_path = output_root / "e1_coverage.png"
    output_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for axis, model_name in zip(axes, MODEL_NAMES, strict=True):
        selected = [row for row in rows if row["model"] == model_name]
        for family in FAMILIES:
            for horizon_s in FORECAST_HORIZONS_S:
                series = [
                    row
                    for row in selected
                    if row["family"] == family and row["horizon_s"] == horizon_s
                ]
                axis.plot(
                    [row["alpha"] for row in series],
                    [row["exceedance_rate"] for row in series],
                    marker="o",
                    alpha=0.45,
                    linewidth=1,
                    label=f"{family}, {horizon_s:.0f}s",
                )
        n = int(np.median([row["n"] for row in selected]))
        low = binom.ppf(0.025, n, alphas) / n
        high = binom.ppf(0.975, n, alphas) / n
        axis.fill_between(alphas, low, high, color="black", alpha=0.1, label="95% binomial")
        axis.plot(alphas, alphas, "k--", linewidth=1)
        axis.set_title(model_name)
        axis.set_xlabel("nominal alpha")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("empirical exceedance rate")
    axes[-1].legend(fontsize=6, loc="upper left")
    figure.suptitle("E1 — stationary split-CQR coverage")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    deltas = np.asarray([row["coverage_delta_pp"] for row in rows], dtype=np.float64)
    worst = rows[int(np.argmax(np.abs(deltas)))]
    payload: dict[str, object] = {
        "experiment": "E1",
        "alphas": alphas.tolist(),
        "rows": rows,
        "mean_absolute_coverage_delta_pp": float(np.mean(np.abs(deltas))),
        "max_absolute_coverage_delta_pp": float(np.max(np.abs(deltas))),
        "worst_cell": worst,
        "figure": str(figure_path),
    }
    write_result(output_root, "e1_coverage", payload)
    return payload
