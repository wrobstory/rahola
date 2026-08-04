"""B2 Chronos foundation-model transfer probe."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    B2_ADAPTATION_CAPSIZES,
    B2_D5_LEAKAGE_AUC,
    B2_FINETUNE_MAX_WINDOWS,
    B2_FPR_IMPROVEMENT,
    B2_TRAJECTORIES_PER_CAMPAIGN,
    CHRONOS_CHECKPOINT,
    CHRONOS_LICENSE,
    CHRONOS_REVISION,
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    SeedBlock,
)
from rahola_lab.detectors import ChronosClassifier, DetectorWindowDataset, extract_detector_windows
from rahola_lab.evaluation import (
    EpisodeConfig,
    TrajectoryScores,
    estimate_decorrelation_time,
    operating_curve,
)
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.d5 import TRANSITION_S
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    detector_risk_end_s,
    fit_detector_suite,
    fit_frozen_suite,
    matched_point,
    point_payload,
    relative_fpr_reduction,
    score_dataset,
    training_windows,
    window_auc,
)


def _concat(parts: list[DetectorWindowDataset]) -> DetectorWindowDataset:
    return DetectorWindowDataset(
        **{
            field: np.concatenate([getattr(part, field) for part in parts])
            for field in DetectorWindowDataset.__dataclass_fields__
        }
    )


def _limited(
    data_root: Path,
    name: str,
    block: SeedBlock,
    *,
    limit: int = B2_TRAJECTORIES_PER_CAMPAIGN,
) -> SimulationDataset:
    return load_campaign_split(campaign_dir(data_root, name), block, limit=limit)


def _training_data(data_root: Path, families: list[str]) -> list[SimulationDataset]:
    return [
        _limited(data_root, f"{family}_{role}", SeedBlock.TRAIN)
        for family in families
        for role in ("stationary", "ramp")
    ]


def _training_windows(data: list[SimulationDataset]) -> DetectorWindowDataset:
    return _concat(
        [
            extract_detector_windows(dataset, stride_s=20.0, max_windows_per_trajectory=3)
            for dataset in data
        ]
    )


def _score_foundation(
    dataset: SimulationDataset,
    model: ChronosClassifier,
    *,
    max_windows_per_trajectory: int | None = None,
) -> list[TrajectoryScores]:
    windows = extract_detector_windows(
        dataset,
        stride_s=10.0,
        max_windows_per_trajectory=max_windows_per_trajectory,
        allow_censored_for_inference=True,
    )
    scores = model.predict_scores(windows.features)
    period = float(dataset.config["natural_period_s"])
    output = []
    for trajectory in range(dataset.batch_size):
        selected = windows.trajectory_indices == trajectory
        times = windows.end_times_s[selected]
        values = scores[selected]
        capsize = float(dataset.t_capsize_s[trajectory])
        record_end_s = detector_risk_end_s(
            times,
            t_capsize_s=capsize,
            raw_record_end_s=float(dataset.time_s[-1]),
            horizon_s=EWS_HORIZON_PERIODS * period,
            record_start_s=60.0 * period,
        )
        output.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_end_s=record_end_s,
                t_capsize_s=capsize if np.isfinite(capsize) else None,
                record_start_s=60.0 * period,
            )
        )
    return output


def _select_operating_policy(calibration: list[TrajectoryScores]) -> dict[str, object]:
    """Freeze a Chronos operating point using calibration trajectories only."""
    values = np.concatenate([item.scores for item in calibration])
    values = values[np.isfinite(values)]
    thresholds = np.unique(
        np.append(
            np.quantile(values, np.linspace(0.0, 1.0, 41)),
            -np.finfo(np.float64).max,
        )
    )
    estimates = []
    for trajectory in calibration:
        if len(trajectory.scores) >= 4:
            dt_s = float(np.median(np.diff(trajectory.times_s)))
            estimates.append(estimate_decorrelation_time(trajectory.scores, dt_s))
    decorrelation_s = float(np.median(estimates)) if estimates else 10.0
    calibration_curve = operating_curve(
        calibration,
        EpisodeConfig(threshold=0.0, debounce_windows=3, refractory_windows=3),
        thresholds,
        horizon_s=EWS_HORIZON_PERIODS * 4.0,
        decorrelation_time_s=decorrelation_s,
    )
    calibration_point = matched_point(calibration_curve)
    return {
        "threshold": calibration_point.threshold,
        "calibration_operating_point": point_payload(calibration_point),
        "decorrelation_time_s": decorrelation_s,
    }


def _evaluate_at_policy(
    test: list[TrajectoryScores], policy: dict[str, object]
) -> dict[str, object]:
    """Evaluate a previously frozen Chronos policy on one outcome block."""
    threshold = float(policy["threshold"])
    decorrelation_s = float(policy["decorrelation_time_s"])
    test_point = operating_curve(
        test,
        EpisodeConfig(threshold=0.0, debounce_windows=3, refractory_windows=3),
        np.asarray([threshold], dtype=np.float64),
        horizon_s=EWS_HORIZON_PERIODS * 4.0,
        decorrelation_time_s=decorrelation_s,
    )[0]
    return point_payload(test_point) | {
        "calibration_operating_point": policy["calibration_operating_point"],
        "decorrelation_time_s": decorrelation_s,
    }


def _evaluate_scores(
    calibration: list[TrajectoryScores], test: list[TrajectoryScores]
) -> dict[str, object]:
    return _evaluate_at_policy(test, _select_operating_policy(calibration))


def _foundation_rotation(
    data_root: Path,
    included: list[str],
    held_out: str,
    mode: str,
) -> dict[str, object]:
    training = _training_windows(_training_data(data_root, included))
    model = ChronosClassifier(mode=mode).fit(training.features, training.labels)
    calibration_data = [
        _limited(data_root, f"{family}_{role}", SeedBlock.CALIBRATION)
        for family in included
        for role in ("stationary", "ramp")
    ]
    test_data = [
        _limited(data_root, f"{held_out}_{role}", SeedBlock.TEST) for role in ("evaluation", "ramp")
    ]
    calibration = [
        item
        for dataset in calibration_data
        for item in _score_foundation(dataset, model)
    ]
    test = [item for dataset in test_data for item in _score_foundation(dataset, model)]
    return _evaluate_scores(calibration, test) | {
        "training_sample_counts": model.training_sample_counts_
    }


def _cnn_rotation(
    data_root: Path,
    included: list[str],
    held_out: str,
    *,
    selected: dict[str, object] | None = None,
    adaptation: SimulationDataset | None = None,
) -> dict[str, object]:
    training_data = _training_data(data_root, included)
    if adaptation is not None:
        training_data.append(adaptation)
    calibration_data = [
        _limited(data_root, f"{family}_{role}", SeedBlock.CALIBRATION)
        for family in included
        for role in ("stationary", "ramp")
    ]
    training = training_windows(training_data, max_windows_per_trajectory=3)
    calibration = training_windows(calibration_data, max_windows_per_trajectory=3)
    suite = (
        fit_frozen_suite(training, selected)
        if selected is not None
        else fit_detector_suite(training, calibration)
    )
    calibration_scores = [
        item for dataset in calibration_data for item in score_dataset(dataset, suite)["cnn"]
    ]
    test_data = [
        _limited(data_root, f"{held_out}_{role}", SeedBlock.TEST) for role in ("evaluation", "ramp")
    ]
    test_scores = [item for dataset in test_data for item in score_dataset(dataset, suite)["cnn"]]
    return _evaluate_scores(calibration_scores, test_scores)


def _indexed(dataset: SimulationDataset, indices: np.ndarray) -> SimulationDataset:
    return SimulationDataset(
        time_s=dataset.time_s,
        angle_rad=dataset.angle_rad[indices],
        rate_rad_s=dataset.rate_rad_s[indices],
        seeds=dataset.seeds[indices],
        capsized=dataset.capsized[indices],
        t_capsize_s=dataset.t_capsize_s[indices],
        metadata=tuple(dataset.metadata[index] for index in indices),
        config=dataset.config,
    )


def _adaptation_data(data_root: Path, held_out: str) -> tuple[SimulationDataset, int, int]:
    dataset = load_campaign_split(
        campaign_dir(data_root, f"{held_out}_stationary"), SeedBlock.TRAIN
    )
    normal = np.flatnonzero(~dataset.capsized)
    capsize = np.flatnonzero(dataset.capsized)[:B2_ADAPTATION_CAPSIZES]
    if len(capsize) != B2_ADAPTATION_CAPSIZES:
        raise ValueError(f"{held_out} training split has fewer than 20 capsizes")
    return _indexed(dataset, np.concatenate((normal, capsize))), len(normal), len(capsize)


def _post_transition(scores: list[TrajectoryScores]) -> list[TrajectoryScores]:
    output = []
    for trajectory in scores:
        selected = trajectory.times_s >= TRANSITION_S
        times = trajectory.times_s[selected]
        values = trajectory.scores[selected]
        output.append(
            TrajectoryScores(
                times_s=times,
                scores=values,
                record_end_s=trajectory.record_end_s,
                t_capsize_s=trajectory.t_capsize_s,
                record_start_s=TRANSITION_S,
            )
        )
    return output


def _d5(data_root: Path, mode: str) -> float:
    training = _training_windows(_training_data(data_root, list(FAMILIES)))
    model = ChronosClassifier(mode=mode, seed=72_001).fit(training.features, training.labels)
    step = _limited(data_root, "softening_step", SeedBlock.TEST)
    return window_auc(_post_transition(_score_foundation(step, model)))


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = load_result(output_root, "d1_operating_curves")
    b1 = load_result(output_root, "p3_b1_graybox")
    b1_by_family = {row["held_out_family"]: row for row in b1["d2_rotations"]}
    rotations = []
    qualifying_rotations = 0
    for held_out in FAMILIES:
        included = [family for family in FAMILIES if family != held_out]
        cnn = _cnn_rotation(data_root, included, held_out)
        adaptation, normal_count, capsize_count = _adaptation_data(data_root, held_out)
        adapted = _cnn_rotation(
            data_root,
            included,
            held_out,
            selected=d1["selected"],
            adaptation=adaptation,
        )
        modes = {}
        rotation_qualifies = False
        for mode in ("frozen", "finetune"):
            result = _foundation_rotation(data_root, included, held_out, mode)
            reduction = relative_fpr_reduction(
                float(result["false_episodes_per_hour"]),
                float(cnn["false_episodes_per_hour"]),
            )
            result["relative_fpr_reduction_vs_cnn"] = reduction
            result["earns_10_percent_improvement"] = (
                float(result["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
                and float(cnn["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
                and reduction is not None
                and reduction >= B2_FPR_IMPROVEMENT
            )
            rotation_qualifies |= bool(result["earns_10_percent_improvement"])
            modes[mode] = result
            gc.collect()
        qualifying_rotations += int(rotation_qualifies)
        rotations.append(
            {
                "held_out_family": held_out,
                "chronos": modes,
                "from_scratch_cnn": cnn,
                "twenty_capsize_cnn": adapted
                | {
                    "target_normal_trajectories": normal_count,
                    "target_capsize_trajectories": capsize_count,
                },
                "graybox_full_d2": b1_by_family[held_out]["graybox"],
                "rotation_qualifies": rotation_qualifies,
            }
        )
        gc.collect()
    d5 = {mode: _d5(data_root, mode) for mode in ("frozen", "finetune")}
    leakage = {
        mode: max(auc, 1.0 - auc) > B2_D5_LEAKAGE_AUC for mode, auc in d5.items()
    }
    kill_fired = qualifying_rotations == 0
    payload: dict[str, object] = {
        "experiment": "B2 Chronos transfer probe",
        "operating_point_policy": "Thresholds are selected on calibration and frozen for test.",
        "checkpoint": CHRONOS_CHECKPOINT,
        "revision": CHRONOS_REVISION,
        "license": CHRONOS_LICENSE,
        "chronos_package_version": "2.3.1",
        "modes": ["frozen encoder embedding + linear head", "full encoder fine-tune + head"],
        "compute_subsample": {
            "trajectories_per_campaign_split": B2_TRAJECTORIES_PER_CAMPAIGN,
            "finetune_max_windows": B2_FINETUNE_MAX_WINDOWS,
            "reason": "CPU-only 8M-parameter transformer probe; all comparators share test subset",
        },
        "rotations": rotations,
        "d5_within_regime_auc": d5,
        "d5_orientation_independent_auc": {
            mode: max(auc, 1.0 - auc) for mode, auc in d5.items()
        },
        "d5_leakage_audit_triggered": leakage,
        "kill": {
            "fired": kill_fired,
            "verbatim": (
                "Kill for the probe: no rotation shows >=10% FPR/h improvement over the "
                "from-scratch CNN."
            ),
            "qualifying_rotations": qualifying_rotations,
        },
        "survives_kill": not kill_fired,
    }
    write_result(
        output_root,
        "p3_b2_chronos",
        payload,
        upstream_results={
            "d1_operating_curves": d1,
            "p3_b1_graybox": b1,
        },
    )
    return payload
