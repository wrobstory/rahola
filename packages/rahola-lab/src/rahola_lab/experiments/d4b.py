"""D4b: critical wave groups with naturally attained entry states."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert
from scipy.stats import nbinom

from rahola.config import SeaState
from rahola.spectrum import jonswap_spectrum
from rahola_lab.experiments.common import load_result, write_result

FloatArray = NDArray[np.float64]

PREREGISTRATION_PATH = "results/d4b_preregistration_d4b.json"


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
    group_start_index: int | None = None,
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
    if group_start_index is None:
        group_start_index = (len(elevation) - 1) // 2
    if not 0 <= group_start_index < len(elevation):
        raise ValueError("group_start_index must lie inside the target waveform")
    target_start = round(arrival_s / dt_s) - group_start_index
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
    targets = [
        (
            np.asarray(row["waveform_elevation_m"], dtype=np.float64),
            np.asarray(row["waveform_slope_rad"], dtype=np.float64),
            int(row["waveform_group_start_index"]),
            int(row["waveform_group_center_index"]),
        )
        for row in library["classes"]
    ]
    duration_s = float(controls["group_arrival_s"]) + max(
        (len(elevation) - group_start) * dt_s
        for elevation, _, group_start, _ in targets
    ) + float(controls["tail_s"])
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
        for class_index, (elevation, slope, group_start, group_center) in enumerate(targets):
            composite = embed_group(
                prelude,
                elevation,
                slope,
                arrival_s=float(controls["group_arrival_s"]),
                blend_half_width_s=float(controls["blend_half_width_s"]),
                group_start_index=group_start,
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
        "missing_center_groups": sum(row["carrier_period_relative_error"] is None for row in rows),
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
            and np.all(np.isfinite(period_errors))
            and np.all(np.isfinite(height_errors))
            and np.max(period_errors) <= 0.05
            and np.max(height_errors) <= 0.10
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
    parser.add_argument("phase", choices=("c1", "c2"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "c1":
        run_c1(args.out)
    elif args.phase == "c2":
        run_c2(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
