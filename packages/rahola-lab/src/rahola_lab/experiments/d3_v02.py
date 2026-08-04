"""D3 v0.2: forcing-bandwidth skill with frozen preprocessing and data routing."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt

from rahola_lab.constants import (
    D2_MATERIAL_FPR_REDUCTION,
    D3_BROADBAND_VERDICT,
    D3_INCONCLUSIVE_VERDICT,
    D3_MATERIAL_AUC_MARGIN,
    D3_SURVIVAL_VERDICT,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    SeedBlock,
)
from rahola_lab.detectors import NormalizationMode
from rahola_lab.experiments.common import write_result
from rahola_lab.experiments.d3 import _all_motion_skill_collapses
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    bootstrap_window_auc,
    decorrelation_times,
    evaluate_suite_at_thresholds,
    fit_detector_suite,
    fit_frozen_suite,
    relative_fpr_reduction,
    score_dataset,
    select_operating_points,
    threshold_grids,
    training_windows,
)
from rahola_lab.experiments.v02_common import (
    load_campaign_split_v02,
    load_frozen_v02_result,
    point_payload_without_dependent_intervals,
)

GAMMAS = (1.0, 3.3, 7.0, 15.0, 30.0)


def _name(gamma: float) -> str:
    return f"softening_bandwidth_gamma_{str(gamma).replace('.', '_').removesuffix('_0')}"


def _selected(suite) -> dict[str, object]:
    return {
        "cnn_grid_index": suite.cnn_grid_index,
        "ews_statistic": suite.ews_statistic,
        "ews_fraction": suite.ews_fraction,
        "neighbor_radius": suite.neighbor_radius,
    }


def run(
    historical_root: Path, versioned_root: Path, output_root: Path
) -> dict[str, object]:
    d1 = load_frozen_v02_result(output_root / "d1_operating_curves_v02.json")
    fixed_ews = (
        str(d1["selected"]["ews_statistic"]),
        float(d1["selected"]["ews_fraction"]),
    )
    fixed_radius = float(d1["selected"]["neighbor_radius"])
    datasets = {
        gamma: {
            block: load_campaign_split_v02(
                historical_root, versioned_root, _name(gamma), block
            )
            for block in (SeedBlock.TRAIN, SeedBlock.CALIBRATION, SeedBlock.TEST)
        }
        for gamma in GAMMAS
    }
    pooled_training = training_windows(
        [datasets[gamma][SeedBlock.TRAIN] for gamma in GAMMAS],
        normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    pooled_calibration = training_windows(
        [datasets[gamma][SeedBlock.CALIBRATION] for gamma in GAMMAS],
        normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    pooled_physical = training_windows(
        [datasets[gamma][SeedBlock.CALIBRATION] for gamma in GAMMAS],
        normalization_mode=NormalizationMode.PHYSICAL,
    )
    cross_suite = fit_detector_suite(
        pooled_training,
        pooled_calibration,
        physics_calibration=pooled_physical,
        cnn_normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
    )
    cross_suite.ews_statistic, cross_suite.ews_fraction = fixed_ews
    cross_suite.neighbor_radius = fixed_radius
    del pooled_training, pooled_calibration, pooled_physical
    gc.collect()

    rows = []
    for gamma in GAMMAS:
        training_dataset = datasets[gamma][SeedBlock.TRAIN]
        calibration_dataset = datasets[gamma][SeedBlock.CALIBRATION]
        test_dataset = datasets[gamma][SeedBlock.TEST]
        primary_training = training_windows(
            [training_dataset],
            normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        primary_calibration = training_windows(
            [calibration_dataset],
            normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        physical_calibration = training_windows(
            [calibration_dataset], normalization_mode=NormalizationMode.PHYSICAL
        )
        suite = fit_detector_suite(
            primary_training,
            primary_calibration,
            physics_calibration=physical_calibration,
            cnn_normalization_mode=NormalizationMode.FIXED_WINDOW_CAUSAL,
        )
        suite.ews_statistic, suite.ews_fraction = fixed_ews
        suite.neighbor_radius = fixed_radius
        calibration_scores = score_dataset(calibration_dataset, suite)
        grids = threshold_grids(calibration_scores)
        decorrelation = decorrelation_times(calibration_scores)
        calibration_points = select_operating_points(
            calibration_scores, grids, decorrelation
        )
        thresholds = {name: point.threshold for name, point in calibration_points.items()}
        test_scores = score_dataset(test_dataset, suite)
        test_points = evaluate_suite_at_thresholds(
            test_scores, thresholds, decorrelation
        )
        horizon_s = EWS_HORIZON_PERIODS * 4.0
        methods = {
            name: bootstrap_point_payload(
                point,
                test_scores[name],
                horizon_s=horizon_s,
                decorrelation_s=decorrelation[name],
            )
            | bootstrap_window_auc(test_scores[name])
            for name, point in test_points.items()
        }

        cumulative_training = training_windows(
            [training_dataset],
            normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
        )
        cumulative_suite = fit_frozen_suite(
            cumulative_training,
            _selected(suite),
            cnn_normalization_mode=NormalizationMode.CUMULATIVE_ONLINE,
        )
        cumulative_calibration = score_dataset(
            calibration_dataset, cumulative_suite
        )["cnn"]
        cumulative_grid = threshold_grids({"cnn": cumulative_calibration})
        cumulative_decorrelation = decorrelation_times(
            {"cnn": cumulative_calibration}
        )
        cumulative_calibration_point = select_operating_points(
            {"cnn": cumulative_calibration},
            cumulative_grid,
            cumulative_decorrelation,
        )["cnn"]
        cumulative_test = score_dataset(test_dataset, cumulative_suite)["cnn"]
        cumulative_point = evaluate_suite_at_thresholds(
            {"cnn": cumulative_test},
            {"cnn": cumulative_calibration_point.threshold},
            cumulative_decorrelation,
        )["cnn"]
        methods["cnn_cumulative_online"] = bootstrap_point_payload(
            cumulative_point,
            cumulative_test,
            horizon_s=horizon_s,
            decorrelation_s=cumulative_decorrelation["cnn"],
        ) | bootstrap_window_auc(cumulative_test)

        cross_calibration = score_dataset(calibration_dataset, cross_suite)
        cross_test = score_dataset(test_dataset, cross_suite)
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
        methods["cnn_cross_gamma"] = bootstrap_point_payload(
            cross_point,
            cross_test["cnn"],
            horizon_s=horizon_s,
            decorrelation_s=cross_decorrelation["cnn"],
        ) | bootstrap_window_auc(cross_test["cnn"])
        rows.append(
            {
                "gamma": gamma,
                "data_source": (
                    "regenerated_v02" if gamma in {7.0, 15.0, 30.0} else "historical_invariant"
                ),
                "methods": methods,
                "calibration_operating_points": {
                    name: point_payload_without_dependent_intervals(point)
                    for name, point in calibration_points.items()
                }
                | {
                    "cnn_cumulative_online": point_payload_without_dependent_intervals(
                        cumulative_calibration_point
                    ),
                    "cnn_cross_gamma": point_payload_without_dependent_intervals(
                        cross_calibration_point
                    ),
                },
            }
        )
        del (
            primary_training,
            primary_calibration,
            physical_calibration,
            cumulative_training,
            suite,
            cumulative_suite,
            calibration_scores,
            test_scores,
            cross_calibration,
            cross_test,
        )
        gc.collect()

    broadband = rows[0]["methods"]
    baseline = broadband["classical_ews"]

    def materially_better(name: str) -> bool:
        reduction = relative_fpr_reduction(
            float(broadband[name]["false_episodes_per_hour"]),
            float(baseline["false_episodes_per_hour"]),
        )
        return (
            float(broadband[name]["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
            and float(baseline["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
            and reduction is not None
            and reduction >= D2_MATERIAL_FPR_REDUCTION
            and float(broadband[name]["auc"])
            >= float(baseline["auc"]) + D3_MATERIAL_AUC_MARGIN
        )

    survives = any(materially_better(name) for name in ("cnn", "cnn_cross_gamma"))
    collapses = _all_motion_skill_collapses(broadband)
    if survives:
        verdict, branch = D3_SURVIVAL_VERDICT, "survival"
    elif collapses:
        verdict, branch = D3_BROADBAND_VERDICT, "collapse"
    else:
        verdict, branch = D3_INCONCLUSIVE_VERDICT, "inconclusive"

    figure_path = output_root / "d3_bandwidth_skill_v02.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in rows[0]["methods"]:
        axis.plot(
            GAMMAS,
            [row["methods"][name]["auc"] for row in rows],
            marker="o",
            label=name,
        )
    axis.set_xscale("log")
    axis.set_xlabel("JONSWAP peak enhancement gamma (larger is narrower band)")
    axis.set_ylabel("window AUC")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    payload: dict[str, object] = {
        "experiment": "D3_v02",
        "normalization_policy": {
            "physics_adjacent_primary": "physical",
            "cnn_primary": "fixed_window_causal",
            "cnn_secondary": "cumulative_online normalization-plus-detector system",
        },
        "predeclared_broadband_verdict": D3_BROADBAND_VERDICT,
        "predeclared_survival_verdict": D3_SURVIVAL_VERDICT,
        "inconclusive_verdict": D3_INCONCLUSIVE_VERDICT,
        "applied_verdict": verdict,
        "verdict_branch": branch,
        "survives_broadband": survives,
        "collapses_at_broadband": collapses,
        "upstream_d1_artifact_sha256": d1["_artifact_sha256"],
        "rows": rows,
        "figure": str(figure_path),
    }
    write_result(output_root, "d3_bandwidth_skill_v02", payload)
    return payload
