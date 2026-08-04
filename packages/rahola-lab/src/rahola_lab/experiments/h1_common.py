"""Frozen H1 terminal labels, offline models, and hybrid scoring."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from rahola.dataset import SimulationDataset
from rahola_lab.constants import (
    H1_CONDITIONAL_BINS,
    H1_RMS_TERCILES,
    H1_RMS_WINDOW_S,
    U1_DECORRELATION_SIGNIFICANCE,
    U1_EMISSION_CADENCE_S,
    U1_INTERVAL_CADENCE_S,
    U1_PARAMETRIC_BOOTSTRAP_DRAWS,
    U1_PARAMETRIC_BOOTSTRAP_SEED,
)
from rahola_lab.experiments.u1_common import restoring_fit
from rahola_lab.splittime import (
    Crossing,
    decluster_crossings,
    detect_crossings,
    roll_decorrelation_time,
)


@dataclass(frozen=True)
class CrossingCluster:
    """One chainwise cluster, retaining its maximum and final raw crossing."""

    crossings: tuple[Crossing, ...]
    retained: Crossing

    @property
    def last(self) -> Crossing:
        return self.crossings[-1]


@dataclass(frozen=True)
class TerminalPartition:
    terminal_labels: tuple[bool, ...]
    heralded: bool
    unheralded: bool


@dataclass(frozen=True)
class ClusterObservation:
    severity_u: float
    rms_rad: float
    terminal: bool


@dataclass(frozen=True)
class TrajectoryObservation:
    campaign: str
    family: str
    role: str
    seed: int
    capsized: bool
    heralded: bool
    unheralded: bool
    exposure_hours: float
    crossing_rate_per_hour: float
    rolling_variance: float
    clusters: tuple[ClusterObservation, ...]
    sampling_gap_signature: bool


@dataclass(frozen=True)
class IsotonicConditional:
    edges: NDArray[np.float64]
    point: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    successes: NDArray[np.int64]
    trials: NDArray[np.int64]

    def bin_index(self, value: float) -> int:
        return int(np.clip(np.digitize(value, self.edges[1:-1]), 0, len(self.point) - 1))

    def predict(self, value: float) -> tuple[float, float, float, int]:
        index = self.bin_index(value)
        return (
            float(self.point[index]),
            float(self.lower[index]),
            float(self.upper[index]),
            index,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "edges": self.edges.tolist(),
            "point": self.point.tolist(),
            "wilson_lower": self.lower.tolist(),
            "wilson_upper": self.upper.tolist(),
            "successes": self.successes.tolist(),
            "trials": self.trials.tolist(),
        }


@dataclass(frozen=True)
class ConditionalModel:
    curves: dict[int, IsotonicConditional]
    rms_edges: NDArray[np.float64] | None = None

    def stratum(self, rms_rad: float) -> int:
        if self.rms_edges is None:
            return 0
        return int(
            np.clip(
                np.digitize(rms_rad, self.rms_edges[1:-1]),
                0,
                len(self.rms_edges) - 2,
            )
        )

    def predict(self, severity_u: float, rms_rad: float) -> tuple[float, float, float, int, int]:
        stratum = self.stratum(rms_rad)
        point, lower, upper, bin_index = self.curves[stratum].predict(severity_u)
        return point, lower, upper, stratum, bin_index

    def to_payload(self) -> dict[str, object]:
        return {
            "rms_edges": None if self.rms_edges is None else self.rms_edges.tolist(),
            "curves": {str(key): curve.to_payload() for key, curve in self.curves.items()},
        }


@dataclass(frozen=True)
class OfflineIntercept:
    rate_per_hour: float
    lower_per_hour: float
    upper_per_hour: float
    unheralded_capsizes: int
    exposure_hours: float

    def to_payload(self) -> dict[str, object]:
        return {
            "rate_per_hour": self.rate_per_hour,
            "poisson_interval_per_hour": [self.lower_per_hour, self.upper_per_hour],
            "unheralded_capsizes": self.unheralded_capsizes,
            "exposure_hours": self.exposure_hours,
        }


@dataclass(frozen=True)
class IsotonicRateMap:
    knots: NDArray[np.float64]
    values: NDArray[np.float64]

    def predict(self, value: float) -> float:
        index = int(np.searchsorted(self.knots, value, side="left"))
        return float(self.values[np.clip(index, 0, len(self.values) - 1)])

    def to_payload(self) -> dict[str, object]:
        return {"knots": self.knots.tolist(), "rate_per_hour": self.values.tolist()}


@dataclass(frozen=True)
class H1Score:
    seed: int
    capsized: bool
    exposure_hours: float
    integrated_count: float
    integrated_count_draws: NDArray[np.float64]

    @property
    def predicted_capsize_probability(self) -> float:
        return float(-np.expm1(-self.integrated_count))

    @property
    def average_rate_per_hour(self) -> float:
        if self.exposure_hours <= 0.0:
            return 0.0
        return self.integrated_count / self.exposure_hours


def conditional_model_from_payload(payload: dict[str, object]) -> ConditionalModel:
    curves = {}
    for key, raw in payload["curves"].items():
        curves[int(key)] = IsotonicConditional(
            edges=np.asarray(raw["edges"], dtype=np.float64),
            point=np.asarray(raw["point"], dtype=np.float64),
            lower=np.asarray(raw["wilson_lower"], dtype=np.float64),
            upper=np.asarray(raw["wilson_upper"], dtype=np.float64),
            successes=np.asarray(raw["successes"], dtype=np.int64),
            trials=np.asarray(raw["trials"], dtype=np.int64),
        )
    rms_edges = payload["rms_edges"]
    return ConditionalModel(
        curves=curves,
        rms_edges=None if rms_edges is None else np.asarray(rms_edges, dtype=np.float64),
    )


def intercept_from_payload(payload: dict[str, object]) -> OfflineIntercept:
    lower, upper = payload["poisson_interval_per_hour"]
    return OfflineIntercept(
        rate_per_hour=float(payload["rate_per_hour"]),
        lower_per_hour=float(lower),
        upper_per_hour=float(upper),
        unheralded_capsizes=int(payload["unheralded_capsizes"]),
        exposure_hours=float(payload["exposure_hours"]),
    )


def rate_map_from_payload(payload: dict[str, object]) -> IsotonicRateMap:
    return IsotonicRateMap(
        knots=np.asarray(payload["knots"], dtype=np.float64),
        values=np.asarray(payload["rate_per_hour"], dtype=np.float64),
    )


def cluster_crossings(
    crossings: tuple[Crossing, ...] | list[Crossing], decorrelation_time_s: float
) -> tuple[CrossingCluster, ...]:
    """Build the same chainwise clusters used by ``decluster_crossings``."""
    if not np.isfinite(decorrelation_time_s) or decorrelation_time_s < 0.0:
        raise ValueError("decorrelation time must be finite and nonnegative")
    events = tuple(crossings)
    if any(right.time_s < left.time_s for left, right in pairwise(events)):
        raise ValueError("crossings must be ordered by time")
    if not events:
        return ()
    groups: list[list[Crossing]] = [[events[0]]]
    for event in events[1:]:
        if event.time_s - groups[-1][-1].time_s <= decorrelation_time_s:
            groups[-1].append(event)
        else:
            groups.append([event])
    return tuple(
        CrossingCluster(
            crossings=tuple(group),
            retained=max(group, key=lambda event: event.severity_u),
        )
        for group in groups
    )


def terminal_partition(
    clusters: tuple[CrossingCluster, ...],
    *,
    capsized: bool,
    t_capsize_s: float | None,
    decorrelation_time_s: float,
) -> TerminalPartition:
    """Partition each capsize into one heralded or unheralded channel."""
    labels = [False] * len(clusters)
    if not capsized:
        return TerminalPartition(tuple(labels), False, False)
    if t_capsize_s is None or not np.isfinite(t_capsize_s):
        raise ValueError("capsizing trajectories require a finite capsize time")
    candidates = []
    for index, cluster in enumerate(clusters):
        next_retained_time = (
            clusters[index + 1].retained.time_s if index + 1 < len(clusters) else np.inf
        )
        after_cluster = t_capsize_s >= cluster.last.time_s
        before_next_retained = t_capsize_s < next_retained_time
        within_influence = t_capsize_s - cluster.last.time_s <= decorrelation_time_s
        if after_cluster and before_next_retained and within_influence:
            candidates.append(index)
    if len(candidates) > 1:
        raise AssertionError("a capsize cannot have more than one terminal cluster")
    if candidates:
        labels[candidates[0]] = True
    heralded = len(candidates) == 1
    unheralded = not heralded
    if heralded == unheralded:
        raise AssertionError("capsize partition must be exhaustive and exclusive")
    return TerminalPartition(tuple(labels), heralded, unheralded)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _pava(values: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            blocks.append(
                [
                    left[0],
                    right[1],
                    (left[2] * left[3] + right[2] * right[3]) / total,
                    total,
                ]
            )
    output = np.empty(len(values), dtype=np.float64)
    for start, stop, value, _ in blocks:
        output[int(start) : int(stop) + 1] = value
    return output


def fit_conditional(
    severities_u: NDArray[np.floating],
    terminal: NDArray[np.bool_],
    *,
    bins: int = H1_CONDITIONAL_BINS,
) -> IsotonicConditional:
    values = np.asarray(severities_u, dtype=np.float64)
    labels = np.asarray(terminal, dtype=np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or not len(values):
        raise ValueError("conditional fit requires matching nonempty vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("conditional severities must be finite")
    internal = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], internal, [np.inf])).astype(np.float64)
    assignments = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    trials = np.bincount(assignments, minlength=len(edges) - 1).astype(np.int64)
    successes = np.bincount(
        assignments, weights=labels.astype(np.int64), minlength=len(edges) - 1
    ).astype(np.int64)
    empirical = successes / trials
    point = _pava(empirical.astype(np.float64), trials.astype(np.float64))
    raw_intervals = np.asarray(
        [wilson_interval(int(k), int(n)) for k, n in zip(successes, trials, strict=True)]
    )
    lower = np.minimum(_pava(raw_intervals[:, 0], trials.astype(np.float64)), point)
    upper = np.maximum(_pava(raw_intervals[:, 1], trials.astype(np.float64)), point)
    return IsotonicConditional(edges, point, lower, upper, successes, trials)


def fit_conditional_model(
    observations: list[TrajectoryObservation], *, rms_terciles: bool
) -> ConditionalModel:
    clusters = [cluster for row in observations for cluster in row.clusters]
    if not clusters:
        raise ValueError("conditional model requires retained clusters")
    rms = np.asarray([cluster.rms_rad for cluster in clusters], dtype=np.float64)
    rms_edges = None
    assignments = np.zeros(len(clusters), dtype=np.int64)
    if rms_terciles:
        internal = np.unique(np.quantile(rms, np.linspace(0.0, 1.0, H1_RMS_TERCILES + 1)[1:-1]))
        if len(internal) != H1_RMS_TERCILES - 1:
            raise ValueError("RMS covariate does not support three distinct terciles")
        rms_edges = np.concatenate(([-np.inf], internal, [np.inf])).astype(np.float64)
        assignments = np.digitize(rms, rms_edges[1:-1]).astype(np.int64)
    curves = {}
    for stratum in sorted(set(assignments.tolist())):
        selected = assignments == stratum
        curves[stratum] = fit_conditional(
            np.asarray([clusters[index].severity_u for index in np.flatnonzero(selected)]),
            np.asarray(
                [clusters[index].terminal for index in np.flatnonzero(selected)],
                dtype=np.bool_,
            ),
        )
    return ConditionalModel(curves=curves, rms_edges=rms_edges)


def fit_intercept(observations: list[TrajectoryObservation]) -> OfflineIntercept:
    events = sum(row.unheralded for row in observations)
    exposure = sum(row.exposure_hours for row in observations)
    if exposure <= 0.0:
        raise ValueError("intercept fit requires positive exposure")
    lower_count = 0.0 if events == 0 else 0.5 * chi2.ppf(0.025, 2 * events)
    upper_count = 0.5 * chi2.ppf(0.975, 2 * (events + 1))
    return OfflineIntercept(
        rate_per_hour=events / exposure,
        lower_per_hour=float(lower_count / exposure),
        upper_per_hour=float(upper_count / exposure),
        unheralded_capsizes=events,
        exposure_hours=exposure,
    )


def fit_rate_map(
    predictor: NDArray[np.floating], target_rate: NDArray[np.floating]
) -> IsotonicRateMap:
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target_rate, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("rate-map fit requires matching nonempty vectors")
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order]
    blocks: list[list[float]] = []
    for value, target in zip(xs, ys, strict=True):
        blocks.append([float(value), float(target), 1.0])
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[2] + right[2]
            blocks.append([right[0], (left[1] * left[2] + right[1] * right[2]) / weight, weight])
    return IsotonicRateMap(
        knots=np.asarray([block[0] for block in blocks], dtype=np.float64),
        values=np.asarray([block[1] for block in blocks], dtype=np.float64),
    )


def _finite_stop(angle: NDArray[np.float64], rate: NDArray[np.float64]) -> int:
    finite = np.isfinite(angle) & np.isfinite(rate)
    invalid = np.flatnonzero(~finite)
    stop = int(invalid[0]) if len(invalid) else len(angle)
    if np.any(finite[stop:]):
        raise ValueError("non-finite motion samples must end the stream")
    return stop


def _causal_rms(angle: NDArray[np.float64], index: int, time_s: NDArray[np.float64]) -> float:
    start_s = max(float(time_s[0]), float(time_s[index]) - H1_RMS_WINDOW_S)
    start = int(np.searchsorted(time_s[: index + 1], start_s, side="left"))
    return float(np.sqrt(np.mean(np.square(angle[start : index + 1]))))


def _mean_causal_variance(
    time_s: NDArray[np.float64], angle: NDArray[np.float64], stop: int
) -> float:
    end_times = np.arange(10.0, float(time_s[stop - 1]) + 1e-9, 10.0)
    if not len(end_times):
        return 0.0
    values = angle[:stop]
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(np.square(values))))
    variances = []
    for end_s in end_times:
        count = int(np.searchsorted(time_s[:stop], end_s, side="right"))
        if count < 2:
            continue
        total = cumulative[count]
        total_sq = cumulative_sq[count]
        variances.append((total_sq - total**2 / count) / (count - 1))
    return float(np.mean(variances)) if variances else 0.0


def observe_dataset(dataset: SimulationDataset, campaign: str) -> list[TrajectoryObservation]:
    family, role = campaign.split("_", maxsplit=1)
    fit = restoring_fit(dataset)
    dt = float(np.median(np.diff(dataset.time_s)))
    output = []
    for index, (angle, rate) in enumerate(zip(dataset.angle_rad, dataset.rate_rad_s, strict=True)):
        stop = _finite_stop(angle, rate)
        exposure_hours = max(float(dataset.time_s[stop - 1] - dataset.time_s[0]), dt) / 3_600.0
        raw = detect_crossings(dataset.time_s, angle, rate, fit)
        decorrelation = (
            dt
            if stop < 3
            else roll_decorrelation_time(
                angle[:stop], dt, significance_level=U1_DECORRELATION_SIGNIFICANCE
            )
        )
        clusters = cluster_crossings(raw, decorrelation)
        capsize_time = float(dataset.t_capsize_s[index])
        if not np.isfinite(capsize_time):
            capsize_time = None
        partition = terminal_partition(
            clusters,
            capsized=bool(dataset.capsized[index]),
            t_capsize_s=capsize_time,
            decorrelation_time_s=decorrelation,
        )
        cluster_rows = tuple(
            ClusterObservation(
                severity_u=cluster.retained.severity_u,
                rms_rad=_causal_rms(angle, cluster.retained.detection_index, dataset.time_s),
                terminal=partition.terminal_labels[cluster_index],
            )
            for cluster_index, cluster in enumerate(clusters)
        )
        signature = False
        if partition.unheralded and capsize_time is not None:
            last_angle = float(angle[stop - 1])
            side = 1 if last_angle >= fit.equilibrium_angle_rad else -1
            threshold = (
                fit.positive.threshold_angle_rad if side == 1 else fit.negative.threshold_angle_rad
            )
            inside = last_angle < threshold if side == 1 else last_angle > threshold
            signature = inside and capsize_time - float(dataset.time_s[stop - 1]) <= dt + 1e-12
        output.append(
            TrajectoryObservation(
                campaign=campaign,
                family=family,
                role=role,
                seed=int(dataset.seeds[index]),
                capsized=bool(dataset.capsized[index]),
                heralded=partition.heralded,
                unheralded=partition.unheralded,
                exposure_hours=exposure_hours,
                crossing_rate_per_hour=len(clusters) / exposure_hours,
                rolling_variance=_mean_causal_variance(dataset.time_s, angle, stop),
                clusters=cluster_rows,
                sampling_gap_signature=signature,
            )
        )
    capsizes = sum(row.capsized for row in output)
    if capsizes != sum(row.heralded for row in output) + sum(row.unheralded for row in output):
        raise AssertionError("heralded/unheralded partition is not exhaustive")
    return output


def _split_normal_draws(
    point: float,
    lower: float,
    upper: float,
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    scale = np.where(
        z < 0.0, (point - lower) / 1.959963984540054, (upper - point) / 1.959963984540054
    )
    return np.clip(point + z * scale, 0.0, None)


def score_hybrid(
    dataset: SimulationDataset,
    model: ConditionalModel,
    intercept: OfflineIntercept,
) -> list[H1Score]:
    fit = restoring_fit(dataset)
    dt = float(np.median(np.diff(dataset.time_s)))
    output = []
    for row, (angle, rate) in enumerate(zip(dataset.angle_rad, dataset.rate_rad_s, strict=True)):
        stop = _finite_stop(angle, rate)
        finite_end_s = float(dataset.time_s[stop - 1])
        exposure_hours = max(finite_end_s - float(dataset.time_s[0]), dt) / 3_600.0
        all_crossings = detect_crossings(dataset.time_s, angle, rate, fit)
        rms_by_index = {
            event.detection_index: _causal_rms(angle, event.detection_index, dataset.time_s)
            for event in all_crossings
        }
        emission_times = np.arange(
            float(dataset.time_s[0]),
            finite_end_s + 0.5 * U1_EMISSION_CADENCE_S,
            U1_EMISSION_CADENCE_S,
        )
        rng = np.random.default_rng(U1_PARAMETRIC_BOOTSTRAP_SEED + int(dataset.seeds[row]))
        point_rates = []
        draw_rates = []
        current_draws = np.full(
            U1_PARAMETRIC_BOOTSTRAP_DRAWS, intercept.rate_per_hour, dtype=np.float64
        )
        last_interval_time = -np.inf
        for emission_time in emission_times:
            end = int(np.searchsorted(dataset.time_s[:stop], emission_time, side="right"))
            decorrelation = (
                dt
                if end < 3
                else roll_decorrelation_time(
                    angle[:end], dt, significance_level=U1_DECORRELATION_SIGNIFICANCE
                )
            )
            observed = tuple(event for event in all_crossings if event.time_s <= emission_time)
            retained = decluster_crossings(observed, decorrelation)
            exposure_s = max(emission_time - float(dataset.time_s[0]), dt)
            grouped: dict[tuple[int, int], list[float]] = {}
            parameters: dict[tuple[int, int], tuple[float, float, float]] = {}
            for event in retained:
                prediction = model.predict(event.severity_u, rms_by_index[event.detection_index])
                point, lower, upper, stratum, bin_index = prediction
                key = (stratum, bin_index)
                grouped.setdefault(key, []).append(point)
                parameters[key] = (point, lower, upper)
            weighted_crossings = sum(sum(values) for values in grouped.values())
            point_rate = weighted_crossings / exposure_s * 3_600.0 + intercept.rate_per_hour
            if (
                emission_time - last_interval_time >= U1_INTERVAL_CADENCE_S - 1e-9
                or last_interval_time == -np.inf
            ):
                current_draws = _split_normal_draws(
                    intercept.rate_per_hour,
                    intercept.lower_per_hour,
                    intercept.upper_per_hour,
                    rng.standard_normal(U1_PARAMETRIC_BOOTSTRAP_DRAWS),
                )
                for key, values in grouped.items():
                    point, lower, upper = parameters[key]
                    conditional_draws = _split_normal_draws(
                        point,
                        lower,
                        upper,
                        rng.standard_normal(U1_PARAMETRIC_BOOTSTRAP_DRAWS),
                    )
                    count_draws = rng.poisson(len(values), size=U1_PARAMETRIC_BOOTSTRAP_DRAWS)
                    current_draws += count_draws * conditional_draws / exposure_s * 3_600.0
                last_interval_time = emission_time
            point_rates.append(float(point_rate))
            draw_rates.append(current_draws.copy())
        integrated = 0.0
        integrated_draws = np.zeros(U1_PARAMETRIC_BOOTSTRAP_DRAWS, dtype=np.float64)
        for index, emission_time in enumerate(emission_times):
            next_time = (
                emission_times[index + 1] if index + 1 < len(emission_times) else finite_end_s
            )
            duration_hours = max(0.0, next_time - emission_time) / 3_600.0
            integrated += point_rates[index] * duration_hours
            integrated_draws += draw_rates[index] * duration_hours
        output.append(
            H1Score(
                seed=int(dataset.seeds[row]),
                capsized=bool(dataset.capsized[row]),
                exposure_hours=exposure_hours,
                integrated_count=float(integrated),
                integrated_count_draws=integrated_draws,
            )
        )
    return output


def score_rate_map(
    observations: list[TrajectoryObservation], model: IsotonicRateMap, *, predictor: str
) -> list[H1Score]:
    output = []
    for row in observations:
        value = float(getattr(row, predictor))
        rate = model.predict(value)
        output.append(
            H1Score(
                seed=row.seed,
                capsized=row.capsized,
                exposure_hours=row.exposure_hours,
                integrated_count=rate * row.exposure_hours,
                integrated_count_draws=np.empty(0, dtype=np.float64),
            )
        )
    return output
