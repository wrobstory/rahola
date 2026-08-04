"""U1c-r2: fresh relation to the frozen detector framing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rahola_lab.constants import (
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    U1_D5_LEAKAGE_AUC,
    SeedBlock,
)
from rahola_lab.evaluation import (
    EpisodeConfig,
    OperatingPoint,
    TrajectoryScores,
    operating_curve,
)
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.detector_common import (
    decorrelation_times,
    matched_point,
    threshold_grids,
)
from rahola_lab.experiments.u1_common import as_trajectory_scores, load_split
from rahola_lab.experiments.u1c import (
    _auc_or_unevaluable,
    _point_with_bootstrap_or_unevaluable,
    _post_step,
)
from rahola_lab.experiments.u1r2_common import load_fresh_test, score_selected


def _score_calibration(
    data_root: Path,
    names: list[str],
    priors: dict[str, dict[str, float]],
) -> tuple[list[TrajectoryScores], float]:
    output: list[TrajectoryScores] = []
    natural_period_s: float | None = None
    for name in names:
        dataset = load_split(data_root, name, SeedBlock.CALIBRATION)
        period = float(dataset.config["natural_period_s"])
        if natural_period_s is None:
            natural_period_s = period
        elif period != natural_period_s:
            raise ValueError("U1c-r2 campaigns must share one natural period")
        output.extend(
            as_trajectory_scores(score_selected(dataset, name, priors), natural_period_s=period)
        )
    if natural_period_s is None:
        raise ValueError("U1c-r2 requires calibration campaigns")
    return output, natural_period_s


def _score_fresh(
    fresh_root: Path,
    names: list[str],
    priors: dict[str, dict[str, float]],
) -> tuple[list[TrajectoryScores], list[str], float]:
    output: list[TrajectoryScores] = []
    strata: list[str] = []
    natural_period_s: float | None = None
    for name in names:
        dataset = load_fresh_test(fresh_root, name)
        period = float(dataset.config["natural_period_s"])
        if natural_period_s is None:
            natural_period_s = period
        elif period != natural_period_s:
            raise ValueError("U1c-r2 campaigns must share one natural period")
        converted = as_trajectory_scores(
            score_selected(dataset, name, priors), natural_period_s=period
        )
        output.extend(converted)
        strata.extend([name] * len(converted))
    if natural_period_s is None:
        raise ValueError("U1c-r2 requires fresh campaigns")
    return output, strata, natural_period_s


def calibration_operating_point(
    data_root: Path,
    priors: dict[str, dict[str, float]],
) -> tuple[OperatingPoint, list[TrajectoryScores], float, float]:
    names = [f"{family}_{role}" for family in FAMILIES for role in ("stationary", "ramp")]
    scores, period_s = _score_calibration(data_root, names, priors)
    grid = threshold_grids({"lambda_hat": scores})["lambda_hat"]
    decorrelation = decorrelation_times({"lambda_hat": scores})["lambda_hat"]
    horizon_s = EWS_HORIZON_PERIODS * period_s
    curve = operating_curve(
        scores,
        EpisodeConfig(threshold=0.0),
        grid,
        horizon_s=horizon_s,
        decorrelation_time_s=decorrelation,
    )
    return matched_point(curve), scores, horizon_s, decorrelation


def run(
    data_root: Path,
    versioned_root: Path,
    fresh_root: Path,
    output_root: Path,
) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1r2")
    priors = u1a["calibration_tail_priors"]
    selected_point, _, horizon_s, decorrelation = calibration_operating_point(data_root, priors)

    d1_names = [f"{family}_{role}" for family in FAMILIES for role in ("evaluation", "ramp")]
    d1, d1_strata, _ = _score_fresh(fresh_root, d1_names, priors)
    d1_point = operating_curve(
        d1,
        EpisodeConfig(threshold=selected_point.threshold),
        np.asarray([selected_point.threshold]),
        horizon_s=horizon_s,
        decorrelation_time_s=decorrelation,
    )[0]
    d1_payload = _point_with_bootstrap_or_unevaluable(
        d1_point,
        d1,
        horizon_s=horizon_s,
        decorrelation_s=decorrelation,
        campaign_strata=d1_strata,
    ) | _auc_or_unevaluable(d1, campaign_strata=d1_strata)

    step_name = "softening_step_v02"
    calibration_dataset = load_split(versioned_root, step_name, SeedBlock.CALIBRATION)
    d5_calibration = _post_step(
        as_trajectory_scores(
            score_selected(calibration_dataset, step_name, priors),
            natural_period_s=float(calibration_dataset.config["natural_period_s"]),
        )
    )
    test_dataset = load_fresh_test(fresh_root, step_name)
    d5 = _post_step(
        as_trajectory_scores(
            score_selected(test_dataset, step_name, priors),
            natural_period_s=float(test_dataset.config["natural_period_s"]),
        )
    )
    d5_auc = _auc_or_unevaluable(d5)
    orientation_auc = (
        None if d5_auc["auc"] is None else max(float(d5_auc["auc"]), 1.0 - float(d5_auc["auc"]))
    )
    payload: dict[str, object] = {
        "experiment": "U1c-r2",
        "selected_controls": u1a["selected_controls"],
        "calibration_threshold": selected_point.threshold,
        "target_sensitivity": DETECTOR_MATCHED_SENSITIVITY,
        "d1": d1_payload,
        "d5": d5_auc
        | {
            "orientation_independent_auc": orientation_auc,
            "leakage_trigger": U1_D5_LEAKAGE_AUC,
            "leakage_audit_triggered": (
                None if orientation_auc is None else orientation_auc > U1_D5_LEAKAGE_AUC
            ),
            "calibration_trajectory_count": len(d5_calibration),
        },
    }
    write_result(output_root, "u1c_u1r2", payload, upstream_results={"u1a_u1r2": u1a})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--versioned-root", type=Path, default=Path("data/reference_v02"))
    parser.add_argument("--fresh-root", type=Path, default=Path("data/u1r2"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(
        arguments.data_root,
        arguments.versioned_root,
        arguments.fresh_root,
        arguments.output_root,
    )


if __name__ == "__main__":
    main()
