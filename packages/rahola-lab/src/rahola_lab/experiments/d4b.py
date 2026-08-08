"""D4b: critical wave groups with naturally attained entry states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import hilbert
from scipy.special import expit
from scipy.stats import ks_2samp, nbinom, rankdata

from rahola.config import SeaState
from rahola.dynamics import integrate_rk4_batch
from rahola.spectrum import jonswap_spectrum
from rahola_lab.campaigns import load_campaign_definition
from rahola_lab.experiments.common import load_result, write_result
from rahola_lab.experiments.h1_common import _pava

FloatArray = NDArray[np.float64]

PREREGISTRATION_PATH = "results/d4b_preregistration_d4b.json"
D4B_TEST_RANGES = ((202_500, 204_000), (204_000, 204_200))


@dataclass(frozen=True)
class ExtendedSea:
    """One nonrepeating excerpt from a longer periodic spectral realization."""

    time_s: FloatArray
    elevation_m: FloatArray
    slope_rad: FloatArray
    fft_period_s: float


@dataclass(frozen=True)
class DetectedGroup:
    """One declustered envelope group and its frozen shape parameters."""

    source_seed: int
    start_index: int
    stop_index: int
    center_index: int
    carrier_period_s: float
    central_height_m: float
    cycle_count: float
    envelope_shape: tuple[float, ...]


@dataclass(frozen=True)
class CompositeRecord:
    """An irregular prelude with one tapered target group and irregular tail."""

    time_s: FloatArray
    elevation_m: FloatArray
    slope_rad: FloatArray
    blend_start_index: int
    target_start_index: int
    target_stop_index: int
    plateau_start_index: int
    plateau_stop_index: int


@dataclass(frozen=True)
class LogisticFit:
    mean: FloatArray
    scale: FloatArray
    coefficients: FloatArray

    def predict(self, features: NDArray[np.floating]) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - self.mean) / self.scale
        return expit(self.coefficients[0] + standardized @ self.coefficients[1:])


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _preregistration() -> dict[str, object]:
    return json.loads((_repository_root() / PREREGISTRATION_PATH).read_text())


def _governing_inputs() -> dict[str, str]:
    path = _repository_root() / PREREGISTRATION_PATH
    return {PREREGISTRATION_PATH: hashlib.sha256(path.read_bytes()).hexdigest()}


def synthesize_extended_jonswap(
    sea_state: SeaState,
    duration_s: float,
    dt_s: float,
    seed: int,
    *,
    period_factor: int = 8,
    max_frequency_rad_s: float,
    gravity_m_s2: float = 9.80665,
) -> ExtendedSea:
    """Return the requested prefix of a regular ``period_factor``-long field."""
    intervals = round(duration_s / dt_s)
    if intervals < 2 or period_factor < 8:
        raise ValueError("extended seas require at least two samples and period_factor >= 8")
    fft_n = intervals * period_factor
    omega = 2.0 * np.pi * np.fft.rfftfreq(fft_n, d=dt_s)
    active = omega < max_frequency_rad_s * (1.0 - 1e-12)
    active_omega = omega[active]
    spectrum = jonswap_spectrum(active_omega, sea_state, gravity_m_s2)
    amplitudes = np.sqrt(2.0 * spectrum * (2.0 * np.pi / (fft_n * dt_s)))
    phases = np.random.default_rng(np.uint64(seed)).uniform(0.0, 2.0 * np.pi, len(active_omega))
    phases[0] = 0.0
    amplitudes[0] = 0.0
    coefficients = np.zeros(len(omega), dtype=np.complex128)
    coefficients[active] = 0.5 * fft_n * amplitudes * np.exp(1j * phases)
    coefficients[-1] = 0.0
    elevation = np.fft.irfft(coefficients, n=fft_n)[: intervals + 1]
    wave_number = omega**2 / gravity_m_s2
    slope = np.fft.irfft(-1j * coefficients * wave_number, n=fft_n)[: intervals + 1]
    return ExtendedSea(
        time_s=np.arange(intervals + 1, dtype=np.float64) * dt_s,
        elevation_m=np.asarray(elevation, dtype=np.float64),
        slope_rad=np.asarray(slope, dtype=np.float64),
        fft_period_s=fft_n * dt_s,
    )


def _merged_runs(selected: NDArray[np.bool_], maximum_gap: int) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(selected.astype(np.int8), (1, 1)))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))
    merged: list[tuple[int, int]] = []
    for start, stop in runs:
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], int(stop))
        else:
            merged.append((int(start), int(stop)))
    return merged


def _carrier_period(time: FloatArray, phase: FloatArray) -> float:
    direction = 1.0 if np.median(np.diff(phase)) >= 0.0 else -1.0
    oriented = direction * phase
    cycle = np.floor(oriented / (2.0 * np.pi)).astype(np.int64)
    indices = np.flatnonzero((np.diff(cycle) > 0) & (np.diff(oriented) > 0.0))
    crossings = []
    for index in indices:
        target = (cycle[index] + 1) * 2.0 * np.pi
        fraction = (target - oriented[index]) / (oriented[index + 1] - oriented[index])
        crossings.append(float(time[index] + fraction * (time[index + 1] - time[index])))
    if len(crossings) >= 2:
        return float(np.median(np.diff(crossings)))
    phase_span = oriented[-1] - oriented[0]
    if phase_span <= 0.0:
        raise ValueError("group phase does not advance")
    return float((time[-1] - time[0]) * 2.0 * np.pi / phase_span)


def detect_groups(
    time_s: NDArray[np.floating],
    elevation_m: NDArray[np.floating],
    *,
    source_seed: int,
    significant_height_m: float,
    peak_period_s: float,
    threshold_fraction: float = 0.75,
    minimum_periods: float = 1.5,
    envelope_ordinates: int = 9,
) -> tuple[DetectedGroup, ...]:
    """Detect, merge, and parameterize sustained Hilbert-envelope groups."""
    time = np.asarray(time_s, dtype=np.float64)
    elevation = np.asarray(elevation_m, dtype=np.float64)
    if time.ndim != 1 or elevation.shape != time.shape or len(time) < 3:
        raise ValueError("time and elevation must be matching nonempty vectors")
    steps = np.diff(time)
    if not np.all(np.isfinite(elevation)) or not np.all(steps > 0.0):
        raise ValueError("elevation must be finite and time strictly increasing")
    if min(significant_height_m, peak_period_s, threshold_fraction, minimum_periods) <= 0.0:
        raise ValueError("group controls must be positive")
    dt_s = float(np.median(steps))
    analytic = hilbert(elevation)
    height = 2.0 * np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    runs = _merged_runs(
        height >= threshold_fraction * significant_height_m,
        round(peak_period_s / dt_s),
    )
    output = []
    for start, stop in runs:
        duration_s = (stop - start) * dt_s
        if duration_s < minimum_periods * peak_period_s:
            continue
        carrier_period_s = _carrier_period(time[start:stop], phase[start:stop])
        cycles = duration_s / carrier_period_s
        if cycles <= 0.5:
            continue
        local_height = height[start:stop]
        center = start + int(np.argmax(local_height))
        ordinates = np.interp(
            np.linspace(0.0, 1.0, envelope_ordinates),
            np.linspace(0.0, 1.0, len(local_height)),
            local_height / np.max(local_height),
        )
        output.append(
            DetectedGroup(
                source_seed=source_seed,
                start_index=start,
                stop_index=stop,
                center_index=center,
                carrier_period_s=carrier_period_s,
                central_height_m=float(np.max(local_height)),
                cycle_count=cycles,
                envelope_shape=tuple(float(value) for value in ordinates),
            )
        )
    return tuple(output)


def _feature_matrix(groups: list[DetectedGroup]) -> FloatArray:
    return np.asarray(
        [
            [
                group.carrier_period_s,
                group.central_height_m,
                group.cycle_count,
                *group.envelope_shape,
            ]
            for group in groups
        ],
        dtype=np.float64,
    )


def cluster_groups(
    groups: list[DetectedGroup], count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray, FloatArray]:
    """Cluster by deterministic farthest-first medoids after robust scaling."""
    if count < 1 or len(groups) < count:
        raise ValueError("cluster count must not exceed the number of groups")
    features = _feature_matrix(groups)
    center = np.median(features, axis=0)
    scale = np.median(np.abs(features - center), axis=0)
    fallback = np.std(features, axis=0)
    scale = np.where(scale > 0.0, scale, np.where(fallback > 0.0, fallback, 1.0))
    standardized = (features - center) / scale
    medoids = [int(np.argmax(np.sum(np.square(standardized), axis=1)))]
    while len(medoids) < count:
        distances = np.min(
            np.sum(
                np.square(standardized[:, None, :] - standardized[np.asarray(medoids)][None, :, :]),
                axis=2,
            ),
            axis=1,
        )
        distances[np.asarray(medoids)] = -1.0
        medoids.append(int(np.argmax(distances)))
    medoids_array = np.asarray(
        sorted(
            medoids,
            key=lambda index: (
                groups[index].central_height_m,
                groups[index].carrier_period_s,
                groups[index].source_seed,
                groups[index].start_index,
            ),
        ),
        dtype=np.int64,
    )
    assignments = np.argmin(
        np.sum(
            np.square(standardized[:, None, :] - standardized[medoids_array][None, :, :]),
            axis=2,
        ),
        axis=1,
    ).astype(np.int64)
    return assignments, medoids_array, center, scale


def _predictive_count_interval(count: int, exposure_hours: float) -> list[int]:
    shape = count + 0.5
    probability = exposure_hours / (exposure_hours + 1.0)
    values = nbinom.ppf((0.025, 0.975), shape, probability)
    return [int(values[0]), int(values[1])]


def _group_payload(group: DetectedGroup) -> dict[str, object]:
    return {
        "source_seed": group.source_seed,
        "start_index": group.start_index,
        "stop_index": group.stop_index,
        "center_index": group.center_index,
        "carrier_period_s": group.carrier_period_s,
        "central_height_m": group.central_height_m,
        "cycle_count": group.cycle_count,
        "envelope_shape": list(group.envelope_shape),
    }


def embed_group(
    prelude: ExtendedSea,
    target_elevation_m: NDArray[np.floating],
    target_slope_rad: NDArray[np.floating],
    *,
    arrival_s: float,
    blend_half_width_s: float,
    height_scale: float = 1.0,
) -> CompositeRecord:
    """Replace one target-sized interval through a two-sided raised-cosine crossfade."""
    elevation = np.asarray(target_elevation_m, dtype=np.float64)
    slope = np.asarray(target_slope_rad, dtype=np.float64)
    if elevation.ndim != 1 or slope.shape != elevation.shape or len(elevation) < 3:
        raise ValueError("target elevation and slope must be matching vectors")
    if not np.all(np.isfinite(elevation)) or not np.all(np.isfinite(slope)):
        raise ValueError("target waveforms must be finite")
    if not np.isfinite(height_scale) or height_scale <= 0.0:
        raise ValueError("height_scale must be positive and finite")
    dt_s = float(np.median(np.diff(prelude.time_s)))
    target_start = round(arrival_s / dt_s)
    target_stop = target_start + len(elevation)
    if target_start < 0 or target_stop > len(prelude.time_s):
        raise ValueError("target window must fit inside the prelude record")
    blend_samples = round(blend_half_width_s / dt_s)
    if blend_samples < 1 or 2 * blend_samples >= len(elevation):
        raise ValueError("blend windows must leave a nonempty target plateau")
    weights = np.ones(len(elevation), dtype=np.float64)
    phase = np.linspace(0.0, np.pi, blend_samples + 1)
    weights[: blend_samples + 1] = 0.5 * (1.0 - np.cos(phase))
    weights[-(blend_samples + 1) :] = 0.5 * (1.0 + np.cos(phase))
    composite_elevation = prelude.elevation_m.copy()
    composite_slope = prelude.slope_rad.copy()
    selected = slice(target_start, target_stop)
    composite_elevation[selected] = (
        (1.0 - weights) * composite_elevation[selected] + weights * height_scale * elevation
    )
    composite_slope[selected] = (
        (1.0 - weights) * composite_slope[selected] + weights * height_scale * slope
    )
    return CompositeRecord(
        time_s=prelude.time_s,
        elevation_m=composite_elevation,
        slope_rad=composite_slope,
        blend_start_index=target_start,
        target_start_index=target_start,
        target_stop_index=target_stop,
        plateau_start_index=target_start + blend_samples,
        plateau_stop_index=target_stop - blend_samples,
    )


def spectral_distortion(
    original_elevation_m: NDArray[np.floating],
    composite_elevation_m: NDArray[np.floating],
    dt_s: float,
    peak_period_s: float,
) -> float:
    """Return added high-frequency energy as a fraction of composite variance."""
    original = np.asarray(original_elevation_m, dtype=np.float64)
    composite = np.asarray(composite_elevation_m, dtype=np.float64)
    if original.shape != composite.shape or original.ndim != 1:
        raise ValueError("spectral records must be matching vectors")
    frequencies = np.fft.rfftfreq(len(original), d=dt_s)
    high = frequencies > 2.5 / peak_period_s
    original_power = np.square(np.abs(np.fft.rfft(original - np.mean(original))))
    composite_power = np.square(np.abs(np.fft.rfft(composite - np.mean(composite))))
    increase = max(0.0, float(np.sum(composite_power[high] - original_power[high])))
    total = float(np.sum(composite_power))
    return increase / total if total > 0.0 else 0.0


def _nearest_center_group(
    composite: CompositeRecord,
    sea_state: SeaState,
    source_seed: int,
    target_center_index: int,
) -> DetectedGroup | None:
    selected = slice(composite.target_start_index, composite.target_stop_index)
    local_time = composite.time_s[selected] - composite.time_s[composite.target_start_index]
    groups = detect_groups(
        local_time,
        composite.elevation_m[selected],
        source_seed=source_seed,
        significant_height_m=sea_state.hs_m,
        peak_period_s=sea_state.tp_s,
    )
    if not groups:
        return None
    return min(groups, key=lambda group: abs(group.center_index - target_center_index))


def run_c2(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c2_embedding"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    library = load_result(output_root, "d4b_group_library_d4b")
    dt_s = float(library["construction"]["dt_s"])
    blend_samples = round(float(controls["blend_half_width_s"]) / dt_s)
    targets = [
        (
            np.asarray(row["waveform_elevation_m"], dtype=np.float64),
            np.asarray(row["waveform_slope_rad"], dtype=np.float64),
            int(row["waveform_group_start_index"]),
            int(row["waveform_group_center_index"]),
        )
        for row in library["classes"]
    ]
    required_duration_s = (
        float(controls["group_arrival_s"])
        + max((len(elevation) - 1) * dt_s for elevation, _, _, _ in targets)
        + float(controls["tail_s"])
    )
    duration_s = float(np.ceil(required_duration_s / (2.0 * dt_s)) * (2.0 * dt_s))
    target_parameters = []
    for class_index, (elevation, _, _, group_center) in enumerate(targets):
        time = np.arange(len(elevation), dtype=np.float64) * dt_s
        groups = detect_groups(
            time,
            elevation,
            source_seed=class_index,
            significant_height_m=sea_state.hs_m,
            peak_period_s=sea_state.tp_s,
        )
        if not groups:
            raise ValueError(f"class {class_index} medoid waveform has no retained group")
        target_parameters.append(
            min(groups, key=lambda group: abs(group.center_index - group_center))
        )
    rows = []
    for seed in range(180_000, 180_200):
        prelude = synthesize_extended_jonswap(
            sea_state,
            duration_s,
            dt_s,
            seed,
            period_factor=int(reference["extended_period_factor"]),
            max_frequency_rad_s=40.0 * 2.0 * np.pi / 4.0,
        )
        for class_index, (elevation, slope, _, group_center) in enumerate(targets):
            composite = embed_group(
                prelude,
                elevation,
                slope,
                arrival_s=float(controls["group_arrival_s"]),
                blend_half_width_s=float(controls["blend_half_width_s"]),
            )
            observed = _nearest_center_group(composite, sea_state, seed, group_center)
            target = target_parameters[class_index]
            rows.append(
                {
                    "seed": seed,
                    "class": class_index,
                    "prefix_byte_exact": bool(
                        np.array_equal(
                            composite.elevation_m[: composite.blend_start_index],
                            prelude.elevation_m[: composite.blend_start_index],
                        )
                    ),
                    "target_plateau_byte_exact": bool(
                        np.array_equal(
                            composite.elevation_m[
                                composite.plateau_start_index : composite.plateau_stop_index
                            ],
                            elevation[blend_samples:-blend_samples],
                        )
                    ),
                    "spectral_distortion": spectral_distortion(
                        prelude.elevation_m,
                        composite.elevation_m,
                        dt_s,
                        sea_state.tp_s,
                    ),
                    "carrier_period_relative_error": None
                    if observed is None
                    else abs(observed.carrier_period_s / target.carrier_period_s - 1.0),
                    "central_height_relative_error": None
                    if observed is None
                    else abs(observed.central_height_m / target.central_height_m - 1.0),
                }
            )
    distortions = np.asarray([row["spectral_distortion"] for row in rows], dtype=np.float64)
    period_errors = np.asarray(
        [row["carrier_period_relative_error"] for row in rows], dtype=np.float64
    )
    height_errors = np.asarray(
        [row["central_height_relative_error"] for row in rows], dtype=np.float64
    )
    worst_rows = sorted(
        rows,
        key=lambda row: max(
            float(row["carrier_period_relative_error"] or 0.0),
            float(row["central_height_relative_error"] or 0.0),
        ),
        reverse=True,
    )[:12]
    limit = float(controls["spectral_distortion_limit"])
    payload: dict[str, object] = {
        "experiment": "D4b C2 natural-initial-condition embedding",
        "_governing_inputs": _governing_inputs(),
        "configuration": {
            "duration_s": duration_s,
            "dt_s": dt_s,
            "calibration_seed_range": [180_000, 180_200],
            "extended_period_factor": int(reference["extended_period_factor"]),
            "arrival_s": float(controls["group_arrival_s"]),
            "blend_half_width_s": float(controls["blend_half_width_s"]),
        },
        "records": len(rows),
        "all_prefixes_byte_exact": all(row["prefix_byte_exact"] for row in rows),
        "all_target_plateaus_byte_exact": all(
            row["target_plateau_byte_exact"] for row in rows
        ),
        "missing_center_groups": sum(row["carrier_period_relative_error"] is None for row in rows),
        "worst_parameter_rows": worst_rows,
        "spectral_distortion": {
            "maximum": float(np.max(distortions)),
            "quantiles": np.quantile(distortions, [0.5, 0.9, 0.99]).tolist(),
            "limit": limit,
        },
        "carrier_period_relative_error": {
            "maximum": float(np.nanmax(period_errors)),
            "quantiles": np.nanquantile(period_errors, [0.5, 0.9, 0.99]).tolist(),
        },
        "central_height_relative_error": {
            "maximum": float(np.nanmax(height_errors)),
            "quantiles": np.nanquantile(height_errors, [0.5, 0.9, 0.99]).tolist(),
        },
        "passes_embedding_checks": bool(
            all(row["prefix_byte_exact"] for row in rows)
            and all(row["target_plateau_byte_exact"] for row in rows)
            and np.all(np.isfinite(period_errors))
            and np.all(np.isfinite(height_errors))
            and np.quantile(period_errors, 0.99) <= 0.05
            and np.quantile(height_errors, 0.99) <= 0.10
            and np.max(distortions) <= limit
        ),
    }
    write_result(
        output_root,
        "d4b_embedding_d4b",
        payload,
        upstream_results={"d4b_group_library_d4b": library},
    )
    return payload


def _integrate_slopes(
    slopes: FloatArray, config: object
) -> tuple[FloatArray, NDArray[np.int32]]:
    rows = np.asarray(slopes, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 3 or rows.shape[1] % 2 == 0:
        raise ValueError("slope records must be an odd-length matrix")
    forcing = config.forcing.effective_wave_slope * rows / config.escape_angle_rad
    zeros = np.zeros_like(forcing)
    ones = np.ones_like(forcing)
    initial = np.zeros((len(rows), 2), dtype=np.float64)
    states, cap_steps = integrate_rk4_batch(
        jax.device_put(forcing),
        jax.device_put(zeros),
        jax.device_put(ones),
        config.omega_n_rad_s * config.integration_dt_s,
        jax.device_put(initial),
        config.damping_ratio,
        config.quadratic_damping,
        config.bias_moment,
        config.quintic_coefficient,
        1.0,
        config.negative_escape_rad / config.escape_angle_rad,
        family_code=0,
        linear_restoring=config.linear_restoring,
    )
    return np.asarray(states, dtype=np.float64), np.asarray(cap_steps, dtype=np.int32)


def _entry_rows(
    states: FloatArray,
    cap_steps: NDArray[np.int32],
    *,
    seeds: range,
    class_index: int | None,
    arrival_s: float,
    config: object,
) -> list[dict[str, object]]:
    entry_step = round(arrival_s / config.integration_dt_s)
    if entry_step >= states.shape[1]:
        raise ValueError("entry time falls outside integrated states")
    x = states[:, entry_step, 0]
    velocity = states[:, entry_step, 1]
    energy = 0.5 * np.square(velocity) + 0.5 * np.square(x) - 0.25 * np.power(x, 4)
    reserve = 0.25 - energy
    return [
        {
            "seed": seed,
            "class": class_index,
            "valid_entry": bool(cap_step < 0 or cap_step >= entry_step),
            "roll_rad": float(x[index] * config.escape_angle_rad),
            "roll_rate_rad_s": float(
                velocity[index] * config.escape_angle_rad * config.omega_n_rad_s
            ),
            "danger_margin_rad": float(config.escape_angle_rad * (1.0 - abs(x[index]))),
            "energy_reserve": float(reserve[index]),
            "pre_entry_capsize": bool(0 <= cap_step < entry_step),
        }
        for index, (seed, cap_step) in enumerate(zip(seeds, cap_steps, strict=True))
    ]


def run_c3(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c2_embedding"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    library = load_result(output_root, "d4b_group_library_d4b")
    embedding = load_result(output_root, "d4b_embedding_d4b")
    if not embedding["passes_embedding_checks"]:
        raise ValueError("C3 requires a passing C2 embedding gate")
    config_path = (
        _repository_root()
        / "packages/rahola-lab/src/rahola_lab/campaigns/configs/softening_evaluation.yaml"
    )
    config = load_campaign_definition(config_path).simulation
    dt_s = 0.5 * config.integration_dt_s
    targets = [
        (
            np.asarray(row["waveform_elevation_m"], dtype=np.float64),
            np.asarray(row["waveform_slope_rad"], dtype=np.float64),
            int(row["waveform_group_start_index"]),
        )
        for row in library["classes"]
    ]
    duration_s = float(embedding["configuration"]["duration_s"])
    seeds = range(180_000, 180_200)
    preludes = [
        synthesize_extended_jonswap(
            sea_state,
            duration_s,
            dt_s,
            seed,
            period_factor=int(reference["extended_period_factor"]),
            max_frequency_rad_s=40.0 * config.omega_n_rad_s,
        )
        for seed in seeds
    ]
    arrival_s = float(controls["group_arrival_s"])
    rows = []
    for class_index, (elevation, slope, _) in enumerate(targets):
        composite_slopes = np.stack(
            [
                embed_group(
                    prelude,
                    elevation,
                    slope,
                    arrival_s=arrival_s,
                    blend_half_width_s=float(controls["blend_half_width_s"]),
                ).slope_rad
                for prelude in preludes
            ]
        )
        states, cap_steps = _integrate_slopes(composite_slopes, config)
        rows.extend(
            _entry_rows(
                states,
                cap_steps,
                seeds=seeds,
                class_index=class_index,
                arrival_s=arrival_s,
                config=config,
            )
        )
    unconditional_states, unconditional_cap_steps = _integrate_slopes(
        np.stack([prelude.slope_rad for prelude in preludes]), config
    )
    unconditional_rows = _entry_rows(
        unconditional_states,
        unconditional_cap_steps,
        seeds=seeds,
        class_index=None,
        arrival_s=arrival_s,
        config=config,
    )
    valid = [row for row in rows if row["valid_entry"]]
    valid_unconditional = [row for row in unconditional_rows if row["valid_entry"]]
    reserves = np.asarray([row["energy_reserve"] for row in valid], dtype=np.float64)
    unconditional_reserves = np.asarray(
        [row["energy_reserve"] for row in valid_unconditional], dtype=np.float64
    )
    edges = np.quantile(reserves, [0.25, 0.5, 0.75])
    if len(np.unique(edges)) != 3:
        raise ValueError("energy reserve does not support four distinct strata")
    for row in rows:
        row["stratum"] = (
            int(np.digitize(float(row["energy_reserve"]), edges)) if row["valid_entry"] else None
        )
    for row in unconditional_rows:
        row["stratum"] = (
            int(np.digitize(float(row["energy_reserve"]), edges)) if row["valid_entry"] else None
        )
    comparison = ks_2samp(reserves, unconditional_reserves)
    payload: dict[str, object] = {
        "experiment": "D4b C3 entry-state strata",
        "_governing_inputs": _governing_inputs(),
        "configuration": {
            "arrival_s": arrival_s,
            "calibration_seed_range": [180_000, 180_200],
            "energy_reserve_definition": prereg["c3_entry_strata"]["energy_reserve"],
        },
        "energy_reserve_internal_edges": edges.tolist(),
        "embedded_valid_entries": len(valid),
        "embedded_pre_entry_capsizes": len(rows) - len(valid),
        "unconditional_valid_entries": len(valid_unconditional),
        "unconditional_pre_entry_capsizes": len(unconditional_rows) - len(valid_unconditional),
        "representativeness": {
            "embedded_mean_energy_reserve": float(np.mean(reserves)),
            "unconditional_mean_energy_reserve": float(np.mean(unconditional_reserves)),
            "two_sample_ks_statistic_descriptive": float(comparison.statistic),
            "two_sample_ks_pvalue_descriptive": float(comparison.pvalue),
            "embedded_stratum_fractions": (
                np.bincount(np.digitize(reserves, edges), minlength=4) / len(reserves)
            ).tolist(),
            "unconditional_stratum_fractions": (
                np.bincount(np.digitize(unconditional_reserves, edges), minlength=4)
                / len(unconditional_reserves)
            ).tolist(),
        },
        "entries": rows,
        "unconditional_entries": unconditional_rows,
    }
    write_result(
        output_root,
        "d4b_entry_strata_d4b",
        payload,
        upstream_results={
            "d4b_group_library_d4b": library,
            "d4b_embedding_d4b": embedding,
        },
    )
    return payload


def bisect_threshold(
    oracle: Callable[[float], bool],
    lower: float,
    upper: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Return the smallest bracketed true threshold to absolute tolerance."""
    if not lower < upper or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("bisection requires an ordered bracket and positive controls")
    if oracle(lower) or not oracle(upper):
        raise ValueError("bisection oracle must be false at lower and true at upper")
    left, right = lower, upper
    for _ in range(max_iterations):
        if right - left <= tolerance:
            break
        middle = 0.5 * (left + right)
        if oracle(middle):
            right = middle
        else:
            left = middle
    return right


def _group_outcomes(
    cap_steps: NDArray[np.int32],
    *,
    arrival_s: float,
    target_duration_s: float,
    integration_dt_s: float,
) -> NDArray[np.bool_]:
    start = round(arrival_s / integration_dt_s)
    stop = round((arrival_s + target_duration_s) / integration_dt_s)
    return (cap_steps >= start) & (cap_steps <= stop)


def _simulate_scaled_targets(
    preludes: list[ExtendedSea],
    target_elevation: FloatArray,
    target_slope: FloatArray,
    scales: FloatArray,
    *,
    arrival_s: float,
    blend_half_width_s: float,
    config: object,
) -> NDArray[np.bool_]:
    if scales.shape != (len(preludes),):
        raise ValueError("one target scale is required per prelude")
    slopes = np.stack(
        [
            embed_group(
                prelude,
                target_elevation,
                target_slope,
                arrival_s=arrival_s,
                blend_half_width_s=blend_half_width_s,
                height_scale=float(scale),
            ).slope_rad
            for prelude, scale in zip(preludes, scales, strict=True)
        ]
    )
    _, cap_steps = _integrate_slopes(slopes, config)
    target_duration_s = (len(target_elevation) - 1) * 0.5 * config.integration_dt_s
    return _group_outcomes(
        cap_steps,
        arrival_s=arrival_s,
        target_duration_s=target_duration_s,
        integration_dt_s=config.integration_dt_s,
    )


def _response_curve(
    outcomes: NDArray[np.bool_],
    selected: NDArray[np.bool_],
    *,
    replicates: int,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rows = outcomes[:, selected]
    if rows.shape[1] == 0:
        raise ValueError("every response stratum requires calibration preludes")
    weights = np.full(rows.shape[0], rows.shape[1], dtype=np.float64)
    point = _pava(np.mean(rows, axis=1), weights)
    samples = np.empty((replicates, rows.shape[0]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, rows.shape[1], rows.shape[1])
        samples[replicate] = _pava(np.mean(rows[:, indices], axis=1), weights)
    return point, np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def _critical_heights(
    preludes: list[ExtendedSea],
    target_elevation: FloatArray,
    target_slope: FloatArray,
    native_height_m: float,
    valid: NDArray[np.bool_],
    *,
    upper_height_m: float,
    tolerance_m: float,
    max_iterations: int,
    arrival_s: float,
    blend_half_width_s: float,
    config: object,
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    count = len(preludes)
    lower = np.zeros(count, dtype=np.float64)
    upper = np.full(count, upper_height_m, dtype=np.float64)
    low_outcome = _simulate_scaled_targets(
        preludes,
        target_elevation,
        target_slope,
        np.full(count, 1e-9),
        arrival_s=arrival_s,
        blend_half_width_s=blend_half_width_s,
        config=config,
    )
    high_outcome = _simulate_scaled_targets(
        preludes,
        target_elevation,
        target_slope,
        upper / native_height_m,
        arrival_s=arrival_s,
        blend_half_width_s=blend_half_width_s,
        config=config,
    )
    bracketed = valid & ~low_outcome & high_outcome
    active = bracketed.copy()
    for _ in range(max_iterations):
        active &= upper - lower > tolerance_m
        if not np.any(active):
            break
        middle = 0.5 * (lower + upper)
        outcome = _simulate_scaled_targets(
            preludes,
            target_elevation,
            target_slope,
            np.maximum(middle / native_height_m, 1e-9),
            arrival_s=arrival_s,
            blend_half_width_s=blend_half_width_s,
            config=config,
        )
        upper = np.where(active & outcome, middle, upper)
        lower = np.where(active & ~outcome, middle, lower)
    critical = np.where(bracketed, upper, np.where(valid & low_outcome, 0.0, np.nan))
    return critical, valid & low_outcome, valid & ~high_outcome


def run_c4(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c2_embedding"]
    response_controls = prereg["c4_response"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    library = load_result(output_root, "d4b_group_library_d4b")
    embedding = load_result(output_root, "d4b_embedding_d4b")
    strata = load_result(output_root, "d4b_entry_strata_d4b")
    config_path = (
        _repository_root()
        / "packages/rahola-lab/src/rahola_lab/campaigns/configs/softening_evaluation.yaml"
    )
    config = load_campaign_definition(config_path).simulation
    dt_s = 0.5 * config.integration_dt_s
    duration_s = float(embedding["configuration"]["duration_s"])
    seeds = range(180_000, 180_200)
    preludes = [
        synthesize_extended_jonswap(
            sea_state,
            duration_s,
            dt_s,
            seed,
            period_factor=int(reference["extended_period_factor"]),
            max_frequency_rad_s=40.0 * config.omega_n_rad_s,
        )
        for seed in seeds
    ]
    multipliers = np.asarray(response_controls["height_multipliers"], dtype=np.float64)
    arrival_s = float(controls["group_arrival_s"])
    blend_half_width_s = float(controls["blend_half_width_s"])
    replicates = int(response_controls["bootstrap_replicates"])
    response_maps = []
    critical_rows = []
    monotonicity_violations = 0
    for class_index, class_row in enumerate(library["classes"]):
        elevation = np.asarray(class_row["waveform_elevation_m"], dtype=np.float64)
        slope = np.asarray(class_row["waveform_slope_rad"], dtype=np.float64)
        native_height = float(class_row["medoid"]["central_height_m"])
        grid_outcomes = np.stack(
            [
                _simulate_scaled_targets(
                    preludes,
                    elevation,
                    slope,
                    np.full(len(preludes), multiplier),
                    arrival_s=arrival_s,
                    blend_half_width_s=blend_half_width_s,
                    config=config,
                )
                for multiplier in multipliers
            ]
        )
        entry_rows = sorted(
            (row for row in strata["entries"] if row["class"] == class_index),
            key=lambda row: row["seed"],
        )
        valid = np.asarray([row["valid_entry"] for row in entry_rows], dtype=np.bool_)
        assignments = np.asarray(
            [-1 if row["stratum"] is None else row["stratum"] for row in entry_rows],
            dtype=np.int64,
        )
        monotonicity_violations += int(
            np.sum(np.any(np.diff(grid_outcomes.astype(np.int8), axis=0) < 0, axis=0) & valid)
        )
        for stratum in range(4):
            selected = valid & (assignments == stratum)
            point, lower, upper = _response_curve(
                grid_outcomes,
                selected,
                replicates=replicates,
                seed=20_260_808 + 4 * class_index + stratum,
            )
            response_maps.append(
                {
                    "class": class_index,
                    "stratum": stratum,
                    "preludes": int(np.sum(selected)),
                    "height_m": (native_height * multipliers).tolist(),
                    "capsizes": np.sum(grid_outcomes[:, selected], axis=1).tolist(),
                    "probability": point.tolist(),
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                }
            )
        critical, left_censored, right_censored = _critical_heights(
            preludes,
            elevation,
            slope,
            native_height,
            valid,
            upper_height_m=float(response_controls["upper_bracket_hs_fraction"])
            * sea_state.hs_m,
            tolerance_m=float(response_controls["bisection_absolute_tolerance_m"]),
            max_iterations=int(response_controls["bisection_max_iterations"]),
            arrival_s=arrival_s,
            blend_half_width_s=blend_half_width_s,
            config=config,
        )
        critical_rows.extend(
            {
                "seed": seed,
                "class": class_index,
                "stratum": None if assignments[index] < 0 else int(assignments[index]),
                "critical_height_m": None
                if not np.isfinite(critical[index])
                else float(critical[index]),
                "left_censored": bool(left_censored[index]),
                "right_censored": bool(right_censored[index]),
                "valid_entry": bool(valid[index]),
            }
            for index, seed in enumerate(seeds)
        )
    payload: dict[str, object] = {
        "experiment": "D4b C4 monotone response maps",
        "_governing_inputs": _governing_inputs(),
        "configuration": {
            "height_multipliers": multipliers.tolist(),
            "bisection_absolute_tolerance_m": float(
                response_controls["bisection_absolute_tolerance_m"]
            ),
            "bisection_max_iterations": int(response_controls["bisection_max_iterations"]),
            "bootstrap_replicates": replicates,
        },
        "grid_monotonicity_violating_preludes": monotonicity_violations,
        "response_maps": response_maps,
        "critical_heights": critical_rows,
    }
    write_result(
        output_root,
        "d4b_response_maps_d4b",
        payload,
        upstream_results={
            "d4b_group_library_d4b": library,
            "d4b_embedding_d4b": embedding,
            "d4b_entry_strata_d4b": strata,
        },
    )
    return payload


def _response_samples(response: dict[str, object]) -> dict[tuple[int, int], FloatArray]:
    samples: dict[tuple[int, int], list[float]] = {}
    for row in response["critical_heights"]:
        if not row["valid_entry"]:
            continue
        key = (int(row["class"]), int(row["stratum"]))
        value = row["critical_height_m"]
        samples.setdefault(key, []).append(np.inf if value is None else float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples.items()}


def _entry_weights(strata: NDArray[np.int64]) -> FloatArray:
    counts = np.bincount(strata, minlength=4).astype(np.float64)
    return counts / np.sum(counts)


def _composed_rate(
    classes: NDArray[np.int64],
    heights: FloatArray,
    group_weights: FloatArray,
    response: dict[tuple[int, int], FloatArray],
    entry_weights: FloatArray,
    exposure_hours: float,
) -> float:
    total = 0.0
    for class_index in range(6):
        selected = classes == class_index
        if not np.any(selected):
            continue
        probability = np.zeros(np.sum(selected), dtype=np.float64)
        selected_heights = heights[selected]
        for stratum in range(4):
            critical = np.sort(response[(class_index, stratum)])
            probability += entry_weights[stratum] * (
                np.searchsorted(critical, selected_heights, side="right") / len(critical)
            )
        total += float(np.dot(group_weights[selected], probability))
    return total / exposure_hours


def _rate_uncertainty(
    library: dict[str, object],
    strata: dict[str, object],
    response: dict[str, object],
    *,
    replicates: int,
    seed: int,
    count_exposure_hours: float,
) -> dict[str, object]:
    groups = library["groups"]
    classes = np.asarray([row["class"] for row in groups], dtype=np.int64)
    heights = np.asarray([row["central_height_m"] for row in groups], dtype=np.float64)
    source_seeds = np.asarray([row["source_seed"] for row in groups], dtype=np.int64)
    unique_sources = np.unique(source_seeds)
    fixed_group_weights = np.ones(len(groups), dtype=np.float64)
    fixed_response = _response_samples(response)
    entry_strata = np.asarray(
        [row["stratum"] for row in strata["unconditional_entries"] if row["valid_entry"]],
        dtype=np.int64,
    )
    fixed_entry_weights = _entry_weights(entry_strata)
    exposure_hours = float(library["exposure_hours"])
    point_rate = _composed_rate(
        classes,
        heights,
        fixed_group_weights,
        fixed_response,
        fixed_entry_weights,
        exposure_hours,
    )
    joint = np.empty(replicates, dtype=np.float64)
    group_only = np.empty(replicates, dtype=np.float64)
    entry_only = np.empty(replicates, dtype=np.float64)
    response_only = np.empty(replicates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled_sources = rng.choice(unique_sources, size=len(unique_sources), replace=True)
        source_counts = {
            source: int(np.sum(sampled_sources == source)) for source in unique_sources
        }
        sampled_group_weights = np.asarray(
            [source_counts[source] for source in source_seeds], dtype=np.float64
        )
        sampled_entry = _entry_weights(
            rng.choice(entry_strata, size=len(entry_strata), replace=True)
        )
        sampled_response = {
            key: rng.choice(values, size=len(values), replace=True)
            for key, values in fixed_response.items()
        }
        group_only[replicate] = _composed_rate(
            classes,
            heights,
            sampled_group_weights,
            fixed_response,
            fixed_entry_weights,
            exposure_hours,
        )
        entry_only[replicate] = _composed_rate(
            classes,
            heights,
            fixed_group_weights,
            fixed_response,
            sampled_entry,
            exposure_hours,
        )
        response_only[replicate] = _composed_rate(
            classes,
            heights,
            fixed_group_weights,
            sampled_response,
            fixed_entry_weights,
            exposure_hours,
        )
        joint[replicate] = _composed_rate(
            classes,
            heights,
            sampled_group_weights,
            sampled_response,
            sampled_entry,
            exposure_hours,
        )
    predictive_counts = rng.poisson(joint * count_exposure_hours)
    return {
        "point_rate_per_hour": point_rate,
        "joint_rate_draws_per_hour": joint.tolist(),
        "group_rate_only_draws_per_hour": group_only.tolist(),
        "entry_distribution_only_draws_per_hour": entry_only.tolist(),
        "response_map_only_draws_per_hour": response_only.tolist(),
        "predictive_count_draws": predictive_counts.tolist(),
        "rate_interval_per_hour": np.quantile(joint, [0.025, 0.975]).tolist(),
        "predictive_count_interval": [
            int(np.quantile(predictive_counts, 0.025, method="inverted_cdf")),
            int(np.quantile(predictive_counts, 0.975, method="inverted_cdf")),
        ],
    }


def run_c5(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c5_rate_validation"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    library = load_result(output_root, "d4b_group_library_d4b")
    strata = load_result(output_root, "d4b_entry_strata_d4b")
    response = load_result(output_root, "d4b_response_maps_d4b")
    config_path = (
        _repository_root()
        / "packages/rahola-lab/src/rahola_lab/campaigns/configs/softening_evaluation.yaml"
    )
    config = load_campaign_definition(config_path).simulation
    duration_s = float(controls["unconditional_duration_s"])
    dt_s = 0.5 * config.integration_dt_s
    test_start, test_stop = D4B_TEST_RANGES[0]
    test_seeds = range(test_start, test_stop)
    direct_rows = []
    for chunk_start in range(test_start, test_stop, 250):
        chunk_seeds = range(chunk_start, min(chunk_start + 250, test_stop))
        records = [
            synthesize_extended_jonswap(
                sea_state,
                duration_s,
                dt_s,
                seed,
                period_factor=int(reference["extended_period_factor"]),
                max_frequency_rad_s=40.0 * config.omega_n_rad_s,
            )
            for seed in chunk_seeds
        ]
        _, cap_steps = _integrate_slopes(np.stack([record.slope_rad for record in records]), config)
        direct_rows.extend(
            {"seed": seed, "capsized": bool(cap_step >= 0)}
            for seed, cap_step in zip(chunk_seeds, cap_steps, strict=True)
        )
    realized_count = sum(row["capsized"] for row in direct_rows)
    exposure_hours = len(test_seeds) * duration_s / 3600.0
    uncertainty = _rate_uncertainty(
        library,
        strata,
        response,
        replicates=int(prereg["c7_uncertainty"]["nested_bootstrap_replicates"]),
        seed=int(prereg["c7_uncertainty"]["nested_bootstrap_seed"]),
        count_exposure_hours=exposure_hours,
    )
    interval = uncertainty["predictive_count_interval"]
    captured = bool(interval[0] <= realized_count <= interval[1])
    verdict = (
        "The encounter-conditioned predictive interval captured the realized matched "
        "unconditional capsize count."
        if captured
        else "The encounter-conditioned predictive interval did not capture the realized matched "
        "unconditional capsize count; the oracle-group decomposition fails its predeclared C5 gate."
    )
    payload: dict[str, object] = {
        "experiment": "D4b C5 encounter-conditioned rate validation",
        "_governing_inputs": _governing_inputs(),
        "configuration": {
            "test_seed_range": [test_start, test_stop],
            "trajectories": len(test_seeds),
            "duration_s": duration_s,
            "exposure_hours": exposure_hours,
            "expected_event_floor": float(controls["expected_event_floor"]),
            "extended_period_factor": int(reference["extended_period_factor"]),
        },
        "encounter_conditioned": uncertainty,
        "direct_count": realized_count,
        "direct_rate_per_hour": realized_count / exposure_hours,
        "captured": captured,
        "verdict_verbatim": verdict,
        "direct_trials": direct_rows,
    }
    write_result(
        output_root,
        "d4b_rate_validation_d4b",
        payload,
        upstream_results={
            "d4b_group_library_d4b": library,
            "d4b_entry_strata_d4b": strata,
            "d4b_response_maps_d4b": response,
        },
    )
    return payload


def _fit_logistic(features: FloatArray, labels: NDArray[np.bool_], penalty: float) -> LogisticFit:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (values - mean) / scale
    design = np.column_stack((np.ones(len(values)), standardized))

    def objective(coefficients: FloatArray) -> tuple[float, FloatArray]:
        logits = design @ coefficients
        value = np.sum(np.logaddexp(0.0, logits) - target * logits)
        value += 0.5 * penalty * float(np.dot(coefficients[1:], coefficients[1:]))
        gradient = design.T @ (expit(logits) - target)
        gradient[1:] += penalty * coefficients[1:]
        return float(value), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"logistic fit did not converge: {result.message}")
    return LogisticFit(mean, scale, np.asarray(result.x, dtype=np.float64))


def _auc(labels: NDArray[np.bool_], scores: FloatArray) -> float:
    target = np.asarray(labels, dtype=np.bool_)
    positives = int(np.sum(target))
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = np.sum(ranks[target]) - positives * (positives + 1) / 2.0
    return float(rank_sum / (positives * negatives))


def _reliability_edges(scores: FloatArray) -> FloatArray:
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 6)))
    if len(edges) < 3:
        minimum, maximum = float(np.min(scores)), float(np.max(scores))
        if minimum == maximum:
            return np.asarray([-np.inf, np.inf], dtype=np.float64)
        edges = np.linspace(minimum, maximum, 6)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _reliability(
    labels: NDArray[np.bool_], scores: FloatArray, edges: FloatArray
) -> dict[str, object]:
    assignments = np.digitize(scores, edges[1:-1])
    rows = []
    absolute_error = 0.0
    for index in range(len(edges) - 1):
        selected = assignments == index
        if not np.any(selected):
            continue
        predicted = float(np.mean(scores[selected]))
        observed = float(np.mean(labels[selected]))
        count = int(np.sum(selected))
        absolute_error += count * abs(predicted - observed)
        rows.append(
            {
                "bin": index,
                "count": count,
                "predicted": predicted,
                "observed": observed,
            }
        )
    return {"weighted_mean_absolute_error": absolute_error / len(scores), "bins": rows}


def _group_features(class_row: dict[str, object], height_m: float) -> list[float]:
    medoid = class_row["medoid"]
    return [
        float(medoid["carrier_period_s"]),
        height_m,
        float(medoid["cycle_count"]),
        *[float(value) for value in medoid["envelope_shape"]],
    ]


def _entry_features(row: dict[str, object]) -> list[float]:
    return [
        float(row["roll_rad"]),
        float(row["roll_rate_rad_s"]),
        float(row["danger_margin_rad"]),
        float(row["energy_reserve"]),
    ]


def _calibration_trials(
    library: dict[str, object],
    strata: dict[str, object],
    response: dict[str, object],
    multipliers: FloatArray,
) -> tuple[FloatArray, NDArray[np.bool_], list[dict[str, object]]]:
    entries = {
        (int(row["class"]), int(row["seed"])): row
        for row in strata["entries"]
        if row["valid_entry"]
    }
    features = []
    labels = []
    metadata = []
    for critical in response["critical_heights"]:
        if not critical["valid_entry"]:
            continue
        class_index = int(critical["class"])
        seed = int(critical["seed"])
        class_row = library["classes"][class_index]
        native_height = float(class_row["medoid"]["central_height_m"])
        threshold = np.inf if critical["critical_height_m"] is None else float(
            critical["critical_height_m"]
        )
        for multiplier in multipliers:
            height = native_height * float(multiplier)
            features.append(
                _entry_features(entries[(class_index, seed)])
                + _group_features(class_row, height)
            )
            labels.append(height >= threshold)
            metadata.append({"seed": seed, "class": class_index, "height_m": height})
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.bool_),
        metadata,
    )


def _fit_payload(fit: LogisticFit, columns: list[int], edges: FloatArray) -> dict[str, object]:
    return {
        "columns": columns,
        "mean": fit.mean.tolist(),
        "scale": fit.scale.tolist(),
        "coefficients": fit.coefficients.tolist(),
        "reliability_edges": edges.tolist(),
    }


def _fit_from_payload(payload: dict[str, object]) -> LogisticFit:
    return LogisticFit(
        np.asarray(payload["mean"], dtype=np.float64),
        np.asarray(payload["scale"], dtype=np.float64),
        np.asarray(payload["coefficients"], dtype=np.float64),
    )


def run_c6_fit(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    library = load_result(output_root, "d4b_group_library_d4b")
    strata = load_result(output_root, "d4b_entry_strata_d4b")
    response = load_result(output_root, "d4b_response_maps_d4b")
    multipliers = np.asarray(prereg["c4_response"]["height_multipliers"], dtype=np.float64)
    features, labels, _ = _calibration_trials(library, strata, response, multipliers)
    column_sets = {
        "entry_only": list(range(4)),
        "group_only": list(range(4, features.shape[1])),
        "both": list(range(features.shape[1])),
    }
    models = {}
    for name, columns in column_sets.items():
        fit = _fit_logistic(features[:, columns], labels, penalty=1e-4)
        scores = fit.predict(features[:, columns])
        models[name] = _fit_payload(fit, columns, _reliability_edges(scores)) | {
            "calibration_auc": _auc(labels, scores),
            "calibration_brier": float(np.mean(np.square(scores - labels))),
        }
    payload: dict[str, object] = {
        "experiment": "D4b C6 calibration-only observability fit",
        "_governing_inputs": _governing_inputs(),
        "calibration_trials": len(labels),
        "calibration_capsizes": int(np.sum(labels)),
        "models": models,
    }
    write_result(
        output_root,
        "d4b_observability_fit_d4b",
        payload,
        upstream_results={
            "d4b_group_library_d4b": library,
            "d4b_entry_strata_d4b": strata,
            "d4b_response_maps_d4b": response,
        },
    )
    return payload


def _metric_vector(labels: NDArray[np.bool_], scores: FloatArray, edges: FloatArray) -> FloatArray:
    return np.asarray(
        [
            _auc(labels, scores),
            np.mean(np.square(scores - labels)),
            _reliability(labels, scores, edges)["weighted_mean_absolute_error"],
        ],
        dtype=np.float64,
    )


def run_c6(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c2_embedding"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    library = load_result(output_root, "d4b_group_library_d4b")
    embedding = load_result(output_root, "d4b_embedding_d4b")
    fit_artifact = load_result(output_root, "d4b_observability_fit_d4b")
    config_path = (
        _repository_root()
        / "packages/rahola-lab/src/rahola_lab/campaigns/configs/softening_evaluation.yaml"
    )
    config = load_campaign_definition(config_path).simulation
    dt_s = 0.5 * config.integration_dt_s
    duration_s = float(embedding["configuration"]["duration_s"])
    test_start, test_stop = D4B_TEST_RANGES[1]
    seeds = range(test_start, test_stop)
    preludes = [
        synthesize_extended_jonswap(
            sea_state,
            duration_s,
            dt_s,
            seed,
            period_factor=int(reference["extended_period_factor"]),
            max_frequency_rad_s=40.0 * config.omega_n_rad_s,
        )
        for seed in seeds
    ]
    states, cap_steps = _integrate_slopes(np.stack([row.slope_rad for row in preludes]), config)
    entry_rows = _entry_rows(
        states,
        cap_steps,
        seeds=seeds,
        class_index=None,
        arrival_s=float(controls["group_arrival_s"]),
        config=config,
    )
    valid = np.asarray([row["valid_entry"] for row in entry_rows], dtype=np.bool_)
    multipliers = np.asarray(prereg["c4_response"]["height_multipliers"], dtype=np.float64)
    trial_features = []
    trial_labels = []
    trial_metadata = []
    for class_index, class_row in enumerate(library["classes"]):
        elevation = np.asarray(class_row["waveform_elevation_m"], dtype=np.float64)
        slope = np.asarray(class_row["waveform_slope_rad"], dtype=np.float64)
        native_height = float(class_row["medoid"]["central_height_m"])
        for multiplier in multipliers:
            outcomes = _simulate_scaled_targets(
                preludes,
                elevation,
                slope,
                np.full(len(preludes), multiplier),
                arrival_s=float(controls["group_arrival_s"]),
                blend_half_width_s=float(controls["blend_half_width_s"]),
                config=config,
            )
            height = native_height * float(multiplier)
            for index in np.flatnonzero(valid):
                trial_features.append(
                    _entry_features(entry_rows[int(index)]) + _group_features(class_row, height)
                )
                trial_labels.append(bool(outcomes[index]))
                trial_metadata.append(
                    {"seed": int(seeds[index]), "class": class_index, "height_m": height}
                )
    features = np.asarray(trial_features, dtype=np.float64)
    labels = np.asarray(trial_labels, dtype=np.bool_)
    model_results = {}
    trial_predictions: dict[str, FloatArray] = {}
    rng = np.random.default_rng(20_260_808)
    unique_seeds = np.asarray(sorted({row["seed"] for row in trial_metadata}), dtype=np.int64)
    row_seeds = np.asarray([row["seed"] for row in trial_metadata], dtype=np.int64)
    for name, model_payload in fit_artifact["models"].items():
        columns = np.asarray(model_payload["columns"], dtype=np.int64)
        fit = _fit_from_payload(model_payload)
        scores = fit.predict(features[:, columns])
        trial_predictions[name] = scores
        edges = np.asarray(model_payload["reliability_edges"], dtype=np.float64)
        estimate = _metric_vector(labels, scores, edges)
        samples = np.empty((1_000, 3), dtype=np.float64)
        for replicate in range(1_000):
            sampled_seeds = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
            indices = np.concatenate([np.flatnonzero(row_seeds == seed) for seed in sampled_seeds])
            samples[replicate] = _metric_vector(labels[indices], scores[indices], edges)
        model_results[name] = {
            "auc": float(estimate[0]),
            "auc_interval": np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
            "brier": float(estimate[1]),
            "brier_interval": np.quantile(samples[:, 1], [0.025, 0.975]).tolist(),
            "reliability": _reliability(labels, scores, edges),
            "reliability_error_interval": np.quantile(
                samples[:, 2], [0.025, 0.975]
            ).tolist(),
        }
    both_auc = float(model_results["both"]["auc"])
    entry_auc = float(model_results["entry_only"]["auc"])
    group_auc = float(model_results["group_only"]["auc"])
    substantial = float(prereg["c6_observability"]["substantial_auc_margin"])
    prediction_i = both_auc >= max(entry_auc, group_auc) + substantial
    prediction_ii = group_auc > entry_auc
    sharp = both_auc >= 0.80
    headline = (
        "The encounter channel is confirmed valuable and quantifies the Upwave sensing target."
        if sharp
        else "Motion-plus-encounter observability is jointly insufficient in this model and the "
        "program's negative answer is complete."
    )
    trials = [
        metadata
        | {"capsized": bool(labels[index])}
        | {name: float(scores[index]) for name, scores in trial_predictions.items()}
        for index, metadata in enumerate(trial_metadata)
    ]
    payload: dict[str, object] = {
        "experiment": "D4b C6 observability decomposition",
        "_governing_inputs": _governing_inputs(),
        "configuration": {
            "test_seed_range": [test_start, test_stop],
            "valid_test_preludes": int(np.sum(valid)),
            "pre_entry_capsizes": int(np.sum(~valid)),
            "test_trials": len(labels),
            "test_capsizes": int(np.sum(labels)),
        },
        "models": model_results,
        "predictions": {
            "i_both_substantially_exceeds_either_alone": prediction_i,
            "ii_group_only_exceeds_entry_only": prediction_ii,
            "iii_both_auc_at_least_0_80": sharp,
        },
        "headline_verdict_verbatim": headline,
        "trials": trials,
    }
    write_result(
        output_root,
        "d4b_observability_d4b",
        payload,
        upstream_results={
            "d4b_group_library_d4b": library,
            "d4b_embedding_d4b": embedding,
            "d4b_observability_fit_d4b": fit_artifact,
        },
    )
    return payload


def run_c7(output_root: Path) -> dict[str, object]:
    rate = load_result(output_root, "d4b_rate_validation_d4b")
    uncertainty = rate["encounter_conditioned"]
    draws = {
        "group_rates": np.asarray(
            uncertainty["group_rate_only_draws_per_hour"], dtype=np.float64
        ),
        "entry_distribution": np.asarray(
            uncertainty["entry_distribution_only_draws_per_hour"], dtype=np.float64
        ),
        "response_map": np.asarray(
            uncertainty["response_map_only_draws_per_hour"], dtype=np.float64
        ),
    }
    component_variance = {name: float(np.var(values, ddof=1)) for name, values in draws.items()}
    component_total = sum(component_variance.values())
    joint = np.asarray(uncertainty["joint_rate_draws_per_hour"], dtype=np.float64)
    payload: dict[str, object] = {
        "experiment": "D4b C7 nested uncertainty propagation",
        "_governing_inputs": _governing_inputs(),
        "point_rate_per_hour": float(uncertainty["point_rate_per_hour"]),
        "joint_rate_interval_per_hour": uncertainty["rate_interval_per_hour"],
        "joint_rate_variance": float(np.var(joint, ddof=1)),
        "predictive_count_interval": uncertainty["predictive_count_interval"],
        "component_variance": component_variance,
        "component_fraction_of_one_source_sum": {
            name: (value / component_total if component_total > 0.0 else 0.0)
            for name, value in component_variance.items()
        },
        "interpretation": (
            "One-source-at-a-time variances need not sum to the joint variance because the "
            "composition is nonlinear; fractions are normalized only across the three isolated "
            "source variances."
        ),
    }
    write_result(
        output_root,
        "d4b_uncertainty_d4b",
        payload,
        upstream_results={"d4b_rate_validation_d4b": rate},
    )
    return payload


def run_c1(output_root: Path) -> dict[str, object]:
    prereg = _preregistration()
    controls = prereg["c1_group_library"]
    reference = prereg["reference_configuration"]
    sea_state = SeaState(**reference["sea_state"])
    dt_s = 0.05
    duration_s = float(controls["library_record_duration_s"])
    seeds = range(90_000, 90_000 + int(controls["library_record_count"]))
    records: dict[int, ExtendedSea] = {}
    groups: list[DetectedGroup] = []
    for seed in seeds:
        record = synthesize_extended_jonswap(
            sea_state,
            duration_s,
            dt_s,
            seed,
            period_factor=int(reference["extended_period_factor"]),
            max_frequency_rad_s=40.0 * 2.0 * np.pi / 4.0,
        )
        records[seed] = record
        groups.extend(
            detect_groups(
                record.time_s,
                record.elevation_m,
                source_seed=seed,
                significant_height_m=sea_state.hs_m,
                peak_period_s=sea_state.tp_s,
                threshold_fraction=float(controls["threshold_hs_fraction"]),
                minimum_periods=float(controls["minimum_run_peak_periods"]),
                envelope_ordinates=int(controls["envelope_shape_ordinates"]),
            )
        )
    margin_samples = round(float(prereg["c2_embedding"]["blend_half_width_s"]) / dt_s)
    groups = [
        group
        for group in groups
        if group.start_index >= margin_samples
        and group.stop_index + margin_samples < len(records[group.source_seed].time_s)
    ]
    assignments, medoids, center, scale = cluster_groups(groups, int(controls["cluster_count"]))
    exposure_hours = len(records) * duration_s / 3600.0
    classes = []
    for class_index, medoid_index in enumerate(medoids):
        selected = np.flatnonzero(assignments == class_index)
        medoid = groups[int(medoid_index)]
        record = records[medoid.source_seed]
        start = medoid.start_index - margin_samples
        stop = medoid.stop_index + margin_samples + 1
        count = len(selected)
        classes.append(
            {
                "class": class_index,
                "count": count,
                "rate_per_hour": count / exposure_hours,
                "one_hour_predictive_count_interval": _predictive_count_interval(
                    count, exposure_hours
                ),
                "medoid": _group_payload(medoid),
                "waveform_dt_s": dt_s,
                "waveform_group_start_index": margin_samples,
                "waveform_group_stop_index": margin_samples
                + medoid.stop_index
                - medoid.start_index,
                "waveform_group_center_index": margin_samples
                + medoid.center_index
                - medoid.start_index,
                "waveform_elevation_m": record.elevation_m[start:stop].tolist(),
                "waveform_slope_rad": record.slope_rad[start:stop].tolist(),
            }
        )
    payload: dict[str, object] = {
        "experiment": "D4b C1 group library and occurrence rates",
        "_governing_inputs": _governing_inputs(),
        "construction": {
            "extended_period_factor": int(reference["extended_period_factor"]),
            "record_duration_s": duration_s,
            "fft_period_s": duration_s * int(reference["extended_period_factor"]),
            "dt_s": dt_s,
            "seeds": list(seeds),
            "sea_state": reference["sea_state"],
        },
        "exposure_hours": exposure_hours,
        "detected_group_count": len(groups),
        "robust_feature_center": center.tolist(),
        "robust_feature_scale": scale.tolist(),
        "classes": classes,
        "groups": [
            _group_payload(group) | {"class": int(class_index)}
            for group, class_index in zip(groups, assignments, strict=True)
        ],
    }
    write_result(output_root, "d4b_group_library_d4b", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rahola_lab.experiments.d4b")
    parser.add_argument(
        "phase", choices=("c1", "c2", "c3", "c4", "c5", "c6-fit", "c6", "c7")
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "c1":
        run_c1(args.out)
    elif args.phase == "c2":
        run_c2(args.out)
    elif args.phase == "c3":
        run_c3(args.out)
    elif args.phase == "c4":
        run_c4(args.out)
    elif args.phase == "c5":
        run_c5(args.out)
    elif args.phase == "c6-fit":
        run_c6_fit(args.out)
    elif args.phase == "c6":
        run_c6(args.out)
    elif args.phase == "c7":
        run_c7(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
