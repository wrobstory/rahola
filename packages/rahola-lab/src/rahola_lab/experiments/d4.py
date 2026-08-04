"""D4: evaluator-only wave-group stratification of frozen detectors."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np

from rahola.config import SimulationConfig
from rahola.dataset import SimulationDataset
from rahola.simulate import _forcing_for_seed
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    EWS_HORIZON_PERIODS,
    WAVE_GROUP_HEIGHT_HS_FRACTION,
    WAVE_GROUP_MIN_PERIODS,
    SeedBlock,
)
from rahola_lab.detectors import NormalizationMode, extract_detector_windows
from rahola_lab.evaluation import (
    AlarmMetrics,
    EpisodeConfig,
    TrajectoryScores,
    WaveGroup,
    alarm_episodes,
    bootstrap_alarm_metrics,
    clopper_pearson_interval,
    decluster_episodes,
    evaluate_alarms,
    identify_wave_groups,
    intervals_overlap,
)
from rahola_lab.experiments.common import FAMILIES, load_result, write_result
from rahola_lab.experiments.detector_common import (
    DETECTOR_NAMES,
    campaign_dir,
    common_natural_period_s,
    detector_risk_end_s,
    fit_frozen_suite,
    merge_scores,
    score_dataset,
    training_windows,
)
from rahola_lab.experiments.v02_common import load_frozen_v02_result
from rahola_lab.forecast import fit_piecewise_linear_restoring


def _groups_for_dataset(dataset: SimulationDataset) -> list[tuple[WaveGroup, ...]]:
    config = SimulationConfig.from_dict(dataset.config)
    n_steps = math.ceil(config.duration_s / config.integration_dt_s)
    n_half_steps = 2 * n_steps
    times_s = np.arange(n_half_steps + 1, dtype=np.float64) * (0.5 * config.integration_dt_s)
    sea = config.forcing.sea_state
    groups = []
    for seed in dataset.seeds:
        _, elevation_m = _forcing_for_seed(config, int(seed), n_half_steps, channel=0)
        groups.append(
            identify_wave_groups(
                times_s,
                elevation_m,
                significant_height_m=sea.hs_m,
                peak_period_s=sea.tp_s,
                height_fraction=WAVE_GROUP_HEIGHT_HS_FRACTION,
                minimum_periods=WAVE_GROUP_MIN_PERIODS,
            )
        )
    return groups


def _precedes_capsize(
    groups: tuple[WaveGroup, ...], capsize_s: float | None, horizon_s: float
) -> bool:
    if capsize_s is None:
        return False
    return any(
        group.start_s < capsize_s
        and intervals_overlap(group.start_s, group.end_s, capsize_s - horizon_s, capsize_s)
        for group in groups
    )


def _sensitivity_payload(metrics: AlarmMetrics) -> dict[str, object]:
    lead_times = metrics.lead_times_s
    quantiles = (
        [float(value) for value in np.quantile(lead_times, [0.1, 0.5, 0.9])]
        if len(lead_times)
        else [float("nan")] * 3
    )
    return {
        "capsize_count": metrics.capsize_count,
        "detected_capsize_count": metrics.detected_capsize_count,
        "sensitivity": metrics.sensitivity,
        "sensitivity_interval": [
            metrics.sensitivity_interval.lower,
            metrics.sensitivity_interval.upper,
        ],
        "lead_time_quantiles_s": quantiles,
    }


def _false_group_fraction(
    trajectories: list[TrajectoryScores],
    groups_by_trajectory: list[tuple[WaveGroup, ...]],
    *,
    threshold: float,
    decorrelation_s: float,
    horizon_s: float,
) -> dict[str, object]:
    false_count = 0
    group_coincident = 0
    config = EpisodeConfig(threshold=threshold, debounce_windows=3, refractory_windows=3)
    for trajectory, groups in zip(trajectories, groups_by_trajectory, strict=True):
        times = np.asarray(trajectory.times_s, dtype=np.float64)
        within_record = (times >= trajectory.record_start_s) & (times <= trajectory.record_end_s)
        record_times = times[within_record]
        observation_start = trajectory.record_start_s
        if len(record_times) >= config.debounce_windows:
            observation_start = max(
                observation_start,
                float(record_times[config.debounce_windows - 1]),
            )
        else:
            observation_start = float("inf")
        capsize_s = trajectory.t_capsize_s
        observable_capsize = (
            capsize_s
            if capsize_s is not None
            and capsize_s > observation_start
            and np.any(within_record & (times < capsize_s) & (times >= capsize_s - horizon_s))
            else None
        )
        event_end = min(
            trajectory.record_end_s,
            observable_capsize or trajectory.record_end_s,
        )
        risk = within_record & (times <= event_end)
        raw_episodes = alarm_episodes(times[risk], trajectory.scores[risk], config)
        episodes = decluster_episodes(raw_episodes, decorrelation_s)
        associated = set()
        if observable_capsize is not None:
            associated = {
                episode
                for episode in episodes
                if episode.start_s < observable_capsize
                and episode.end_s >= observable_capsize - horizon_s
            }
        noncapsizing_groups = [
            group
            for group in groups
            if observable_capsize is None
            or not intervals_overlap(
                group.start_s,
                group.end_s,
                observable_capsize - horizon_s,
                observable_capsize,
            )
        ]
        for episode in episodes:
            if episode in associated:
                continue
            false_count += 1
            constituent_episodes = (
                raw_episode
                for raw_episode in raw_episodes
                if raw_episode.start_s >= episode.start_s and raw_episode.end_s <= episode.end_s
            )
            if any(
                intervals_overlap(
                    raw_episode.start_s,
                    raw_episode.end_s,
                    group.start_s,
                    group.end_s,
                )
                for raw_episode in constituent_episodes
                for group in noncapsizing_groups
            ):
                group_coincident += 1
    interval = clopper_pearson_interval(group_coincident, false_count)
    return {
        "coincident_false_episodes": group_coincident,
        "false_episode_count": false_count,
        "fraction": group_coincident / false_count if false_count else float("nan"),
        "episode_level_binomial_interval_descriptive_only": [
            interval.lower,
            interval.upper,
        ],
    }


def _danger_scores(dataset: SimulationDataset) -> list[TrajectoryScores]:
    """Score the corrected two-sided known-configuration comparator."""
    period = float(dataset.config["natural_period_s"])
    danger = fit_piecewise_linear_restoring(dataset.config)
    output: list[TrajectoryScores] = []
    for start in range(0, dataset.batch_size, 128):
        stop = min(start + 128, dataset.batch_size)
        chunk = SimulationDataset(
            time_s=dataset.time_s,
            angle_rad=dataset.angle_rad[start:stop],
            rate_rad_s=dataset.rate_rad_s[start:stop],
            seeds=dataset.seeds[start:stop],
            capsized=dataset.capsized[start:stop],
            t_capsize_s=dataset.t_capsize_s[start:stop],
            metadata=dataset.metadata[start:stop],
            config=dataset.config,
        )
        windows = extract_detector_windows(
            chunk,
            stride_s=10.0,
            allow_censored_for_inference=True,
            normalization_mode=NormalizationMode.PHYSICAL,
        )
        values = danger.danger_score(windows.raw_angle_rad, windows.raw_rate_rad_s)
        for local in range(chunk.batch_size):
            selected = windows.trajectory_indices == local
            times = windows.end_times_s[selected]
            capsize = float(chunk.t_capsize_s[local])
            output.append(
                TrajectoryScores(
                    times_s=times,
                    scores=values[selected],
                    record_end_s=detector_risk_end_s(
                        times,
                        t_capsize_s=capsize,
                        raw_record_end_s=float(chunk.time_s[-1]),
                        horizon_s=EWS_HORIZON_PERIODS * period,
                        record_start_s=60.0 * period,
                    ),
                    t_capsize_s=capsize if np.isfinite(capsize) else None,
                    record_start_s=60.0 * period,
                )
            )
    return output


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = load_result(output_root, "d1_operating_curves")
    training_data = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    suite = fit_frozen_suite(
        training_windows(training_data, max_windows_per_trajectory=3), d1["selected"]
    )
    del training_data
    gc.collect()

    evaluation_data = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TEST)
        for family in FAMILIES
        for role in ("evaluation", "ramp")
    ]
    scores = merge_scores([score_dataset(dataset, suite) for dataset in evaluation_data])
    groups_by_trajectory = [
        groups for dataset in evaluation_data for groups in _groups_for_dataset(dataset)
    ]
    horizon_s = EWS_HORIZON_PERIODS * common_natural_period_s(evaluation_data)
    preceded = np.asarray(
        [
            _precedes_capsize(groups, trajectory.t_capsize_s, horizon_s)
            for groups, trajectory in zip(
                groups_by_trajectory, scores[DETECTOR_NAMES[0]], strict=True
            )
        ],
        dtype=bool,
    )
    capsize = np.asarray(
        [trajectory.t_capsize_s is not None for trajectory in scores[DETECTOR_NAMES[0]]],
        dtype=bool,
    )
    capsize_group_count = int(np.sum(capsize & preceded))
    capsize_count = int(np.sum(capsize))
    group_interval = clopper_pearson_interval(capsize_group_count, capsize_count)

    methods = {}
    for name in DETECTOR_NAMES:
        threshold = float(d1["headline_at_calibration_selected_threshold"][name]["threshold"])
        decorrelation_s = float(d1["decorrelation_time_s"][name])
        with_group = [
            trajectory
            for trajectory, has_group, has_capsize in zip(
                scores[name], preceded, capsize, strict=True
            )
            if has_group and has_capsize
        ]
        without_group = [
            trajectory
            for trajectory, has_group, has_capsize in zip(
                scores[name], preceded, capsize, strict=True
            )
            if not has_group and has_capsize
        ]
        episode_config = EpisodeConfig(
            threshold=threshold, debounce_windows=3, refractory_windows=3
        )
        methods[name] = {
            "preceded_by_group": _sensitivity_payload(
                evaluate_alarms(
                    with_group,
                    episode_config,
                    horizon_s=horizon_s,
                    decorrelation_time_s=decorrelation_s,
                )
            ),
            "not_preceded_by_group": _sensitivity_payload(
                evaluate_alarms(
                    without_group,
                    episode_config,
                    horizon_s=horizon_s,
                    decorrelation_time_s=decorrelation_s,
                )
            ),
            "false_episode_group_coincidence": _false_group_fraction(
                scores[name],
                groups_by_trajectory,
                threshold=threshold,
                decorrelation_s=decorrelation_s,
                horizon_s=horizon_s,
            ),
        }
    payload: dict[str, object] = {
        "experiment": "D4",
        "definition": {
            "instantaneous_wave_height": "2 * abs(Hilbert(elevation))",
            "height_threshold_hs_fraction": WAVE_GROUP_HEIGHT_HS_FRACTION,
            "minimum_duration_peak_periods": WAVE_GROUP_MIN_PERIODS,
            "preceding_horizon_s": horizon_s,
            "evaluator_only": True,
        },
        "capsizes_preceded_by_group": {
            "population": "all capsizes in the six source campaign records",
            "count": capsize_group_count,
            "capsize_count": capsize_count,
            "fraction": capsize_group_count / capsize_count,
            "interval": [group_interval.lower, group_interval.upper],
        },
        "methods": methods,
    }
    write_result(
        output_root,
        "d4_wave_groups",
        payload,
        upstream_results={"d1_operating_curves": d1},
    )
    return payload


def run_v02_danger(data_root: Path, output_root: Path) -> dict[str, object]:
    """Regenerate only the D4 row affected by the two-sided danger repair."""
    d1 = load_frozen_v02_result(output_root / "d1_operating_curves_v02.json")
    names = [f"{family}_{role}" for family in FAMILIES for role in ("evaluation", "ramp")]
    datasets = [
        load_campaign_split(campaign_dir(data_root, name), SeedBlock.TEST) for name in names
    ]
    scores = [trajectory for dataset in datasets for trajectory in _danger_scores(dataset)]
    strata = [
        name
        for name, dataset in zip(names, datasets, strict=True)
        for _ in range(dataset.batch_size)
    ]
    groups = [group for dataset in datasets for group in _groups_for_dataset(dataset)]
    horizon_s = EWS_HORIZON_PERIODS * common_natural_period_s(datasets)
    capsize = np.asarray([trajectory.t_capsize_s is not None for trajectory in scores])
    preceded = np.asarray(
        [
            _precedes_capsize(group, trajectory.t_capsize_s, horizon_s)
            for group, trajectory in zip(groups, scores, strict=True)
        ]
    )
    threshold = float(
        d1["headline_at_calibration_selected_threshold"]["danger_margin"]["threshold"]
    )
    decorrelation_s = float(d1["decorrelation_time_s"]["danger_margin"])
    config = EpisodeConfig(threshold=threshold, debounce_windows=3, refractory_windows=3)

    def subset_payload(selected: np.ndarray) -> dict[str, object]:
        selected_scores = [score for score, keep in zip(scores, selected, strict=True) if keep]
        selected_strata = [stratum for stratum, keep in zip(strata, selected, strict=True) if keep]
        point = _sensitivity_payload(
            evaluate_alarms(
                selected_scores,
                config,
                horizon_s=horizon_s,
                decorrelation_time_s=decorrelation_s,
            )
        )
        point["sensitivity_exact_capsize_event_interval"] = point.pop("sensitivity_interval")
        interval = bootstrap_alarm_metrics(
            selected_scores,
            config,
            horizon_s=horizon_s,
            decorrelation_time_s=decorrelation_s,
            campaign_strata=selected_strata,
        )
        point["sensitivity_trajectory_bootstrap_interval"] = [
            interval.sensitivity.lower,
            interval.sensitivity.upper,
        ]
        return point

    group_count = int(np.sum(capsize & preceded))
    capsize_count = int(np.sum(capsize))
    group_interval = clopper_pearson_interval(group_count, capsize_count)
    payload: dict[str, object] = {
        "experiment": "D4_v02 selective danger-margin regeneration",
        "supersedes": "methods.danger_margin in d4_wave_groups.json only",
        "interval_conditioning": "conditional on the calibration-frozen alarm policy",
        "definition": {
            "preceding_horizon_s": horizon_s,
            "evaluator_only": True,
        },
        "capsizes_preceded_by_group": {
            "count": group_count,
            "capsize_count": capsize_count,
            "fraction": group_count / capsize_count,
            "exact_capsize_event_interval": [group_interval.lower, group_interval.upper],
        },
        "methods": {
            "danger_margin": {
                "preceded_by_group": subset_payload(capsize & preceded),
                "not_preceded_by_group": subset_payload(capsize & ~preceded),
                "false_episode_group_coincidence": _false_group_fraction(
                    scores,
                    groups,
                    threshold=threshold,
                    decorrelation_s=decorrelation_s,
                    horizon_s=horizon_s,
                ),
            }
        },
    }
    write_result(output_root, "d4_wave_groups_v02", payload)
    return payload
