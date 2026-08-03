"""E4: cross-sea-state LSTM calibration stress test."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.conformal import SplitCQRUpper, adaptive_conformal_bounds
from rahola_lab.constants import SeedBlock
from rahola_lab.experiments.common import (
    campaign_path,
    fit_forecasters,
    snapshot,
    trajectory_forecasts,
    write_result,
)

ALPHA = 0.05
GAMMA = 0.05
HORIZON_S = 60.0
MODEL_NAME = "lstm"


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    training = load_campaign_split(campaign_path(data_root, "softening", "stationary"), "train")
    models = fit_forecasters(training, HORIZON_S)
    model = models[MODEL_NAME]
    deployment_path = data_root / "softening_step"
    calibration = load_campaign_split(deployment_path, SeedBlock.CALIBRATION)
    calibration_y, calibration_raw = snapshot(
        calibration, {MODEL_NAME: model}, HORIZON_S, history_end_s=330.0
    )
    conformal = SplitCQRUpper.calibrate(calibration_y, calibration_raw[MODEL_NAME])

    # Pseudo-prospective seal: this is the experiment's only test load.
    test = load_campaign_split(deployment_path, SeedBlock.TEST)
    test_y, test_raw = snapshot(test, {MODEL_NAME: model}, HORIZON_S, history_end_s=330.0)
    raw_coverage = float(np.mean(test_y <= test_raw[MODEL_NAME]))
    cqr_bound = conformal.upper_bound(test_raw[MODEL_NAME], ALPHA)
    cqr_coverage = float(np.mean(test_y <= cqr_bound))

    streams = trajectory_forecasts(
        test,
        {MODEL_NAME: model},
        HORIZON_S,
        stride_s=10.0,
        first_history_end_s=300.0,
    )
    aci_errors: list[np.ndarray] = []
    fixed_errors: list[np.ndarray] = []
    raw_errors: list[np.ndarray] = []
    for stream in streams:
        raw = stream.raw_upper_rad[MODEL_NAME]
        raw_errors.append(stream.targets_rad > raw)
        fixed_errors.append(stream.targets_rad > conformal.upper_bound(raw, ALPHA))
        adaptive = adaptive_conformal_bounds(
            conformal.scores,
            raw,
            stream.targets_rad,
            alpha=ALPHA,
            gamma=GAMMA,
        )
        aci_errors.append(adaptive.errors)
    dense_raw_coverage = 1.0 - float(np.mean(np.concatenate(raw_errors)))
    dense_cqr_coverage = 1.0 - float(np.mean(np.concatenate(fixed_errors)))
    aci_coverage = 1.0 - float(np.mean(np.concatenate(aci_errors)))

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "e4_stress_test.png"
    labels = ["raw LSTM\n(snapshot)", "split CQR\n(snapshot)", "ACI\n(post-step stream)"]
    values = [raw_coverage, cqr_coverage, aci_coverage]
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    bars = axis.bar(labels, values, color=["#c95f4b", "#4c78a8", "#59a14f"])
    axis.axhline(1.0 - ALPHA, color="black", linestyle="--", label="nominal 95%")
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("empirical upper-bound coverage")
    axis.set_title("E4 — train at Hs=4 m, deploy after Hs=5 m step")
    axis.bar_label(bars, fmt="%.3f")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    payload: dict[str, object] = {
        "experiment": "E4",
        "alpha": ALPHA,
        "aci_gamma": GAMMA,
        "raw_lstm_snapshot_coverage": raw_coverage,
        "split_cqr_snapshot_coverage": cqr_coverage,
        "raw_lstm_dense_post_coverage": dense_raw_coverage,
        "split_cqr_dense_post_coverage": dense_cqr_coverage,
        "aci_dense_post_coverage": aci_coverage,
        "raw_coverage_gap_pp": 100.0 * ((1.0 - ALPHA) - raw_coverage),
        "cqr_coverage_delta_pp": 100.0 * (cqr_coverage - (1.0 - ALPHA)),
        "figure": str(figure_path),
    }
    write_result(output_root, "e4_stress_test", payload)
    return payload
