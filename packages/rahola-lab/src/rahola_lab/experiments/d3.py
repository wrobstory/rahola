"""D3: detector skill versus forcing bandwidth."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    D2_MATERIAL_FPR_REDUCTION,
    D3_BROADBAND_VERDICT,
    D3_MATERIAL_AUC_MARGIN,
    D3_SURVIVAL_VERDICT,
    SeedBlock,
)
from rahola_lab.experiments.common import write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite,
    fit_detector_suite,
    matched_point,
    point_payload,
    score_dataset,
    threshold_grids,
    training_windows,
    window_auc,
)

GAMMAS = (1.0, 3.3, 7.0, 15.0, 30.0)


def _name(gamma: float) -> str:
    return f"softening_bandwidth_gamma_{str(gamma).replace('.', '_').removesuffix('_0')}"


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = json.loads((output_root / "d1_operating_curves.json").read_text(encoding="utf-8"))
    fixed_ews = d1["selected"]["ews_statistic"], float(d1["selected"]["ews_fraction"])
    fixed_radius = float(d1["selected"]["neighbor_radius"])
    datasets = {
        gamma: {
            block: load_campaign_split(campaign_dir(data_root, _name(gamma)), block)
            for block in (SeedBlock.TRAIN, SeedBlock.CALIBRATION, SeedBlock.TEST)
        }
        for gamma in GAMMAS
    }
    pooled_training = training_windows(
        [datasets[gamma][SeedBlock.TRAIN] for gamma in GAMMAS], max_windows_per_trajectory=3
    )
    pooled_calibration = training_windows(
        [datasets[gamma][SeedBlock.CALIBRATION] for gamma in GAMMAS],
        max_windows_per_trajectory=3,
    )
    cross_suite = fit_detector_suite(pooled_training, pooled_calibration)
    cross_suite.ews_statistic, cross_suite.ews_fraction = fixed_ews
    cross_suite.neighbor_radius = fixed_radius
    del pooled_training, pooled_calibration
    gc.collect()

    rows = []
    for gamma in GAMMAS:
        training = training_windows(
            [datasets[gamma][SeedBlock.TRAIN]], max_windows_per_trajectory=3
        )
        calibration = training_windows(
            [datasets[gamma][SeedBlock.CALIBRATION]], max_windows_per_trajectory=3
        )
        suite = fit_detector_suite(training, calibration)
        suite.ews_statistic, suite.ews_fraction = fixed_ews
        suite.neighbor_radius = fixed_radius
        calibration_scores = score_dataset(datasets[gamma][SeedBlock.CALIBRATION], suite)
        grids = threshold_grids(calibration_scores)
        decorrelation = decorrelation_times(calibration_scores)
        test_scores = score_dataset(datasets[gamma][SeedBlock.TEST], suite)
        curves = evaluate_suite(test_scores, grids, decorrelation)
        methods = {
            name: {
                **point_payload(matched_point(curve)),
                "auc": window_auc(test_scores[name]),
            }
            for name, curve in curves.items()
        }

        cross_calibration = score_dataset(datasets[gamma][SeedBlock.CALIBRATION], cross_suite)
        cross_test = score_dataset(datasets[gamma][SeedBlock.TEST], cross_suite)
        cross_grids = threshold_grids(cross_calibration)
        cross_decorrelation = decorrelation_times(cross_calibration)
        cross_curve = evaluate_suite(cross_test, cross_grids, cross_decorrelation)["cnn"]
        methods["cnn_cross_gamma"] = {
            **point_payload(matched_point(cross_curve)),
            "auc": window_auc(cross_test["cnn"]),
        }
        rows.append({"gamma": gamma, "methods": methods})
        del training, calibration, suite, test_scores, curves, cross_calibration, cross_test
        gc.collect()

    broadband = rows[0]["methods"]
    b1 = broadband["classical_ews"]
    survives = any(
        broadband[name]["false_episodes_per_hour"]
        <= (1.0 - D2_MATERIAL_FPR_REDUCTION) * b1["false_episodes_per_hour"]
        and broadband[name]["auc"] >= b1["auc"] + D3_MATERIAL_AUC_MARGIN
        for name in ("cnn", "cnn_cross_gamma")
    )
    verdict = D3_SURVIVAL_VERDICT if survives else D3_BROADBAND_VERDICT

    figure_path = output_root / "d3_bandwidth_skill.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in rows[0]["methods"]:
        axis.plot(
            GAMMAS,
            [row["methods"][name]["false_episodes_per_hour"] for row in rows],
            marker="o",
            label=name,
        )
    axis.set_xscale("log")
    axis.set_xlabel("JONSWAP peak enhancement gamma (larger is narrower band)")
    axis.set_ylabel("false episodes/h at >=90% sensitivity")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    payload: dict[str, object] = {
        "experiment": "D3",
        "predeclared_broadband_verdict": D3_BROADBAND_VERDICT,
        "predeclared_survival_verdict": D3_SURVIVAL_VERDICT,
        "applied_verdict": verdict,
        "survives_broadband": survives,
        "rows": rows,
        "figure": str(figure_path),
    }
    write_result(output_root, "d3_bandwidth_skill", payload)
    return payload
