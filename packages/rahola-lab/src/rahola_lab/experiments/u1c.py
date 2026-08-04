"""U1c: relation of the online rate to the frozen detector framing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rahola_lab.constants import (
    D5_V02_FIRST_ENDPOINT_S,
    D5_V02_LAST_ENDPOINT_S,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    U1_D5_LEAKAGE_AUC,
    SeedBlock,
)
from rahola_lab.evaluation import EpisodeConfig, TrajectoryScores, operating_curve
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    bootstrap_window_auc,
    decorrelation_times,
    matched_point,
    point_payload,
    threshold_grids,
)
from rahola_lab.experiments.u1_common import (
    as_trajectory_scores,
    campaign_family,
    load_split,
    score_dataset,
)
from rahola_lab.splittime import SplitTimeConfig


def _score_names(
    data_root: Path,
    names: list[str],
    block: SeedBlock,
    *,
    quantile: float,
    strength: float,
    window_s: float | None,
    prior_means: dict[str, dict[str, float]],
) -> tuple[list[TrajectoryScores], list[str], float]:
    output: list[TrajectoryScores] = []
    strata: list[str] = []
    natural_period_s: float | None = None
    for name in names:
        dataset = load_split(data_root, name, block)
        dataset_period = float(dataset.config["natural_period_s"])
        if natural_period_s is None:
            natural_period_s = dataset_period
        elif dataset_period != natural_period_s:
            raise ValueError("U1c campaigns must share one configured natural period")
        scores = score_dataset(
            dataset,
            prior_mean=float(prior_means[str(quantile)][campaign_family(name)]),
            prior_strength=strength,
            config=SplitTimeConfig(
                tail_quantile=quantile,
                trailing_window_s=window_s,
            ),
        )
        converted = as_trajectory_scores(
            scores,
            natural_period_s=float(dataset.config["natural_period_s"]),
        )
        output.extend(converted)
        strata.extend([name] * len(converted))
    if natural_period_s is None:
        raise ValueError("U1c requires at least one campaign")
    return output, strata, natural_period_s


def _post_step(scores: list[TrajectoryScores]) -> list[TrajectoryScores]:
    output = []
    for trajectory in scores:
        selected = (
            (trajectory.times_s >= D5_V02_FIRST_ENDPOINT_S)
            & (trajectory.times_s <= D5_V02_LAST_ENDPOINT_S)
        )
        output.append(
            TrajectoryScores(
                times_s=trajectory.times_s[selected],
                scores=trajectory.scores[selected],
                record_end_s=min(trajectory.record_end_s, D5_V02_LAST_ENDPOINT_S),
                t_capsize_s=trajectory.t_capsize_s,
                record_start_s=D5_V02_FIRST_ENDPOINT_S,
            )
        )
    return output


def _point_with_bootstrap_or_unevaluable(
    point,
    scores: list[TrajectoryScores],
    *,
    horizon_s: float,
    decorrelation_s: float,
    campaign_strata: list[str] | None = None,
) -> dict[str, object]:
    try:
        return bootstrap_point_payload(
            point,
            scores,
            horizon_s=horizon_s,
            decorrelation_s=decorrelation_s,
            campaign_strata=campaign_strata,
        )
    except ValueError as error:
        if "no finite replicates" not in str(error):
            raise
        return point_payload(point) | {
            "trajectory_bootstrap_status": "unevaluable: no finite replicates"
        }


def _auc_or_unevaluable(
    scores: list[TrajectoryScores], *, campaign_strata: list[str] | None = None
) -> dict[str, object]:
    try:
        return bootstrap_window_auc(scores, campaign_strata=campaign_strata)
    except ValueError as error:
        return {"auc": None, "auc_status": f"unevaluable: {error}"}


def run(data_root: Path, versioned_root: Path, output_root: Path) -> dict[str, object]:
    u1a = load_result(output_root, "u1a_u1")
    selected = u1a["selected_controls"]
    prior_means = u1a["calibration_prior_mean_rates"]
    quantile = float(selected["tail_quantile"])
    strength = float(selected["prior_strength"])
    window_s = selected["trailing_window_s"]
    calibration_names = [
        f"{family}_{role}"
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    d1_names = [
        f"{family}_{role}"
        for family in FAMILIES
        for role in ("evaluation", "ramp")
    ]
    calibration, _, period_s = _score_names(
        data_root,
        calibration_names,
        SeedBlock.CALIBRATION,
        quantile=quantile,
        strength=strength,
        window_s=window_s,
        prior_means=prior_means,
    )
    grid = threshold_grids({"lambda_hat": calibration})["lambda_hat"]
    decorrelation = decorrelation_times({"lambda_hat": calibration})["lambda_hat"]
    horizon_s = EWS_HORIZON_PERIODS * period_s
    calibration_curve = operating_curve(
        calibration,
        EpisodeConfig(threshold=0.0),
        grid,
        horizon_s=horizon_s,
        decorrelation_time_s=decorrelation,
    )
    selected_point = matched_point(calibration_curve)

    d1, d1_strata, test_period_s = _score_names(
        data_root,
        d1_names,
        SeedBlock.TEST,
        quantile=quantile,
        strength=strength,
        window_s=window_s,
        prior_means=prior_means,
    )
    if test_period_s != period_s:
        raise ValueError("U1c calibration and test periods differ")
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
    test_dataset = load_split(versioned_root, step_name, SeedBlock.TEST)
    d5_calibration = _post_step(
        as_trajectory_scores(
            score_dataset(
                calibration_dataset,
                prior_mean=float(prior_means[str(quantile)]["softening"]),
                prior_strength=strength,
                config=SplitTimeConfig(
                    tail_quantile=quantile,
                    trailing_window_s=window_s,
                ),
            ),
            natural_period_s=float(calibration_dataset.config["natural_period_s"]),
        )
    )
    d5 = _post_step(
        as_trajectory_scores(
            score_dataset(
                test_dataset,
                prior_mean=float(prior_means[str(quantile)]["softening"]),
                prior_strength=strength,
                config=SplitTimeConfig(
                    tail_quantile=quantile,
                    trailing_window_s=window_s,
                ),
            ),
            natural_period_s=float(test_dataset.config["natural_period_s"]),
        )
    )
    d5_auc = _auc_or_unevaluable(d5)
    orientation_auc = (
        None
        if d5_auc["auc"] is None
        else max(float(d5_auc["auc"]), 1.0 - float(d5_auc["auc"]))
    )
    payload: dict[str, object] = {
        "experiment": "U1c",
        "selected_controls": selected,
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
    write_result(output_root, "u1c_u1", payload, upstream_results={"u1a_u1": u1a})
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
