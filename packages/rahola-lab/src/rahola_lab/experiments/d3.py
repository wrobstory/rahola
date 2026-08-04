"""D3: detector skill versus forcing bandwidth."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    D2_MATERIAL_FPR_REDUCTION,
    D3_BROADBAND_VERDICT,
    D3_INCONCLUSIVE_VERDICT,
    D3_MATERIAL_AUC_MARGIN,
    D3_SURVIVAL_VERDICT,
    DETECTOR_MATCHED_SENSITIVITY,
    SeedBlock,
)
from rahola_lab.experiments.common import load_result, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    decorrelation_times,
    evaluate_suite_at_thresholds,
    fit_detector_suite,
    point_payload,
    relative_fpr_reduction,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
    window_auc,
)


def _all_motion_skill_collapses(methods: dict[str, dict[str, object]]) -> bool:
    baseline_auc = float(methods["classical_ews"]["auc"])
    return all(
        float(payload["auc"]) <= baseline_auc + D3_MATERIAL_AUC_MARGIN
        for name, payload in methods.items()
        if name != "classical_ews"
    )

GAMMAS = (1.0, 3.3, 7.0, 15.0, 30.0)


def _name(gamma: float) -> str:
    return f"softening_bandwidth_gamma_{str(gamma).replace('.', '_').removesuffix('_0')}"


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = load_result(output_root, "d1_operating_curves")
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
        calibration_points = select_operating_points(calibration_scores, grids, decorrelation)
        thresholds = {name: point.threshold for name, point in calibration_points.items()}
        test_scores = score_dataset(datasets[gamma][SeedBlock.TEST], suite)
        test_points = evaluate_suite_at_thresholds(test_scores, thresholds, decorrelation)
        methods = {
            name: {
                **point_payload(point),
                "auc": window_auc(test_scores[name]),
            }
            for name, point in test_points.items()
        }

        cross_calibration = score_dataset(datasets[gamma][SeedBlock.CALIBRATION], cross_suite)
        cross_test = score_dataset(datasets[gamma][SeedBlock.TEST], cross_suite)
        cross_grids = threshold_grids(cross_calibration)
        cross_decorrelation = decorrelation_times(cross_calibration)
        cross_calibration_point = select_operating_points(
            cross_calibration, cross_grids, cross_decorrelation
        )["cnn"]
        cross_point = evaluate_suite_at_thresholds(
            cross_test,
            {name: cross_calibration_point.threshold for name in cross_test},
            cross_decorrelation,
        )["cnn"]
        methods["cnn_cross_gamma"] = {
            **point_payload(cross_point),
            "auc": window_auc(cross_test["cnn"]),
        }
        rows.append(
            {
                "gamma": gamma,
                "methods": methods,
                "calibration_operating_points": {
                    name: point_payload(point) for name, point in calibration_points.items()
                }
                | {"cnn_cross_gamma": point_payload(cross_calibration_point)},
            }
        )
        del training, calibration, suite, test_scores, test_points, cross_calibration, cross_test
        gc.collect()

    broadband = rows[0]["methods"]
    b1 = broadband["classical_ews"]
    def materially_better(name: str) -> bool:
        reduction = relative_fpr_reduction(
            broadband[name]["false_episodes_per_hour"],
            b1["false_episodes_per_hour"],
        )
        return (
            broadband[name]["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
            and b1["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
            and reduction is not None
            and reduction >= D2_MATERIAL_FPR_REDUCTION
            and broadband[name]["auc"] >= b1["auc"] + D3_MATERIAL_AUC_MARGIN
        )

    survives = any(materially_better(name) for name in ("cnn", "cnn_cross_gamma"))
    collapses = _all_motion_skill_collapses(broadband)
    if survives:
        verdict = D3_SURVIVAL_VERDICT
        verdict_branch = "survival"
    elif collapses:
        verdict = D3_BROADBAND_VERDICT
        verdict_branch = "collapse"
    else:
        verdict = D3_INCONCLUSIVE_VERDICT
        verdict_branch = "inconclusive"

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
    axis.set_ylabel("false episodes/h at calibration-selected threshold")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    payload: dict[str, object] = {
        "experiment": "D3",
        "operating_point_policy": "Thresholds are selected on calibration and frozen for test.",
        "predeclared_broadband_verdict": D3_BROADBAND_VERDICT,
        "predeclared_survival_verdict": D3_SURVIVAL_VERDICT,
        "inconclusive_verdict": D3_INCONCLUSIVE_VERDICT,
        "applied_verdict": verdict,
        "verdict_branch": verdict_branch,
        "survives_broadband": survives,
        "collapses_at_broadband": collapses,
        "rows": rows,
        "figure": str(figure_path),
    }
    write_result(
        output_root,
        "d3_bandwidth_skill",
        payload,
        upstream_results={"d1_operating_curves": d1},
    )
    return payload
