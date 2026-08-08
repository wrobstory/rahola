"""W1 preregistered wave-field validity and estimand audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import matmul_toeplitz
from scipy.stats import gamma as gamma_distribution
from scipy.stats import t as student_t

from rahola.config import SeaState
from rahola.spectrum import jonswap_spectrum, synthesize_jonswap
from rahola_lab.campaigns import load_campaign_definition
from rahola_lab.experiments.common import load_result, write_result

DURATION_S = 600.0
DT_S = 0.05
SAMPLE_COUNT = round(DURATION_S / DT_S)
REALIZATIONS = 500
CALIBRATION_REALIZATIONS = 250
SEED_START = 290_000
CONFIDENCE_LEVEL = 0.99
MAX_ANALYSIS_LAG_S = 440.0
ENVELOPE_LIMIT = 0.05
ACF_ERROR_LIMIT = 0.05
PASSING_RATE_MIN = 0.90
JITTER_GRID_SEED = 20_260_808
NATURAL_PERIOD_S = 4.0
MAX_FREQUENCY_RAD_S = 40.0 * 2.0 * np.pi / NATURAL_PERIOD_S


@dataclass(frozen=True)
class SpectralGrid:
    name: str
    fft_n: int
    indices: np.ndarray
    omega_rad_s: np.ndarray
    energy_m2: np.ndarray


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _campaign_states() -> tuple[list[dict[str, object]], list[SeaState]]:
    config_root = Path(__file__).parents[1] / "campaigns" / "configs"
    paths = sorted(
        path
        for path in config_root.glob("*.yaml")
        if path.stem.endswith(("_stationary", "_evaluation"))
        or path.stem.startswith("softening_bandwidth_gamma_")
    )
    sources: list[dict[str, object]] = []
    states: dict[tuple[float, float, float], SeaState] = {}
    for path in paths:
        definition = load_campaign_definition(path)
        state = definition.simulation.forcing.sea_state
        key = (state.hs_m, state.tp_s, state.gamma)
        states[key] = state
        sources.append(
            {
                "campaign": definition.name,
                "config": str(path.relative_to(_repository_root())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sea_state": {"hs_m": state.hs_m, "tp_s": state.tp_s, "gamma": state.gamma},
            }
        )
    return sources, [states[key] for key in sorted(states)]


def _state_name(state: SeaState) -> str:
    return f"Hs={state.hs_m:g}, Tp={state.tp_s:g}, gamma={state.gamma:g}"


def _spectral_grid(state: SeaState, name: str, period_factor: int, *, jitter: bool) -> SpectralGrid:
    fft_n = SAMPLE_COUNT * period_factor
    delta_omega = 2.0 * np.pi / (fft_n * DT_S)
    if jitter:
        base_indices = np.arange(1, SAMPLE_COUNT // 2, dtype=np.int64)
        rng = np.random.default_rng(JITTER_GRID_SEED)
        offsets = rng.integers(
            -(period_factor // 2 - 1), period_factor // 2, size=base_indices.size
        )
        indices = base_indices * period_factor + offsets
    else:
        omega = 2.0 * np.pi * np.fft.rfftfreq(fft_n, d=DT_S)
        indices = np.flatnonzero((omega > 0.0) & (omega < MAX_FREQUENCY_RAD_S * (1.0 - 1e-12)))
    frequencies = indices.astype(np.float64) * delta_omega
    all_frequencies = np.concatenate(([0.0], frequencies))
    spectrum = jonswap_spectrum(all_frequencies, state)
    if jitter:
        weights = np.empty_like(all_frequencies)
        weights[0] = 0.5 * (all_frequencies[1] - all_frequencies[0])
        weights[-1] = 0.5 * (all_frequencies[-1] - all_frequencies[-2])
        weights[1:-1] = 0.5 * (all_frequencies[2:] - all_frequencies[:-2])
        energy = spectrum[1:] * weights[1:]
    else:
        energy = spectrum[1:] * delta_omega
    return SpectralGrid(name, fft_n, indices, frequencies, energy)


def _analytic_field(grid: SpectralGrid, derivative_order: int = 0) -> np.ndarray:
    coefficients = np.zeros(grid.fft_n, dtype=np.complex128)
    coefficients[grid.indices] = (
        grid.fft_n * grid.energy_m2 * (1j * grid.omega_rad_s) ** derivative_order
    )
    return np.fft.ifft(coefficients)


def _realization(grid: SpectralGrid, seed: int, sample_count: int = SAMPLE_COUNT) -> np.ndarray:
    coefficients = np.zeros(grid.fft_n // 2 + 1, dtype=np.complex128)
    phases = np.random.default_rng(np.uint64(seed)).uniform(
        0.0, 2.0 * np.pi, size=grid.indices.size
    )
    amplitudes = np.sqrt(2.0 * grid.energy_m2)
    coefficients[grid.indices] = 0.5 * grid.fft_n * amplitudes * np.exp(1j * phases)
    return np.fft.irfft(coefficients, n=grid.fft_n)[:sample_count]


def _upcrossings(values: np.ndarray) -> int:
    return int(np.count_nonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)))


def _variance_prediction(covariance: np.ndarray) -> tuple[float, float, list[float]]:
    n = covariance.size
    weights = np.arange(n - 1, 0, -1, dtype=np.float64)
    trace_k2 = n * covariance[0] ** 2 + 2.0 * np.dot(weights, covariance[1:] ** 2)
    row_sums = matmul_toeplitz((covariance, covariance), np.ones(n))
    total = float(np.sum(row_sums))
    trace_centered_k2 = trace_k2 - 2.0 * np.dot(row_sums, row_sums) / n + total**2 / n**2
    mean = float(covariance[0] - total / n**2)
    variance = float(2.0 * trace_centered_k2 / n**2)
    shape = mean**2 / variance
    scale = variance / mean
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    interval = gamma_distribution.ppf((tail, 1.0 - tail), a=shape, scale=scale)
    return mean, variance, [float(interval[0]), float(interval[1])]


def _intervals_above(lags: np.ndarray, values: np.ndarray, limit: float) -> list[list[float]]:
    selected = values > limit
    changes = np.diff(np.pad(selected.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [
        [float(lags[start]), float(lags[stop])]
        for start, stop in zip(starts, stops, strict=True)
    ]


def _analyze_grid(state: SeaState, grid: SpectralGrid) -> dict[str, object]:
    analytic = _analytic_field(grid)
    retained = analytic[: SAMPLE_COUNT + 1]
    if retained.size == SAMPLE_COUNT:
        retained = np.concatenate((retained, analytic[:1]))
    covariance = retained.real
    envelope = np.abs(retained) / covariance[0]
    lags_s = np.arange(SAMPLE_COUNT + 1, dtype=np.float64) * DT_S
    analysis = (lags_s >= 5.0 * state.tp_s) & (lags_s <= MAX_ANALYSIS_LAG_S)
    max_lag_index = round(MAX_ANALYSIS_LAG_S / DT_S)
    selected_indices = np.arange(0, max_lag_index + 1, round(0.5 / DT_S))
    empirical_acf = np.zeros(selected_indices.size, dtype=np.float64)
    record_variances = np.empty(REALIZATIONS, dtype=np.float64)
    crossing_counts = np.empty(REALIZATIONS, dtype=np.float64)
    fft_n = 2 * SAMPLE_COUNT
    denominators = SAMPLE_COUNT - selected_indices
    for index, seed in enumerate(range(SEED_START, SEED_START + REALIZATIONS)):
        values = _realization(grid, seed)
        record_variances[index] = np.var(values)
        crossing_counts[index] = _upcrossings(values)
        transform = np.fft.rfft(values, n=fft_n)
        products = np.fft.irfft(transform * transform.conjugate(), n=fft_n)
        empirical_acf += products[selected_indices] / denominators
    empirical_acf /= REALIZATIONS
    normalized_error = np.abs(empirical_acf - covariance[selected_indices]) / covariance[0]

    variance_mean, variance_of_variance, variance_interval = _variance_prediction(
        covariance[:SAMPLE_COUNT]
    )
    calibration_counts = crossing_counts[:CALIBRATION_REALIZATIONS]
    evaluation_counts = crossing_counts[CALIBRATION_REALIZATIONS:]
    count_std = float(np.std(calibration_counts, ddof=1))
    degrees = CALIBRATION_REALIZATIONS - 1
    critical = float(student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, degrees))
    m0 = float(np.sum(grid.energy_m2))
    m2 = float(np.sum(grid.energy_m2 * grid.omega_rad_s**2))
    rice_rate = float(np.sqrt(m2 / m0) / (2.0 * np.pi))
    expected_count = rice_rate * DURATION_S
    crossing_half_width = critical * count_std * np.sqrt(1.0 + 1.0 / CALIBRATION_REALIZATIONS)
    crossing_interval = [
        max(0.0, expected_count - crossing_half_width),
        expected_count + crossing_half_width,
    ]
    evaluation_mean = float(np.mean(evaluation_counts))
    mean_half_width = (
        critical
        * count_std
        * np.sqrt(1.0 / evaluation_counts.size + 1.0 / CALIBRATION_REALIZATIONS)
    )
    rice_mean_interval = [evaluation_mean - mean_half_width, evaluation_mean + mean_half_width]
    variance_passing = float(
        np.mean(
            (record_variances[CALIBRATION_REALIZATIONS:] >= variance_interval[0])
            & (record_variances[CALIBRATION_REALIZATIONS:] <= variance_interval[1])
        )
    )
    crossing_passing = float(
        np.mean(
            (evaluation_counts >= crossing_interval[0])
            & (evaluation_counts <= crossing_interval[1])
        )
    )
    gates = {
        "acf_envelope": float(np.max(envelope[analysis])) <= ENVELOPE_LIMIT,
        "empirical_acf": float(np.max(normalized_error)) <= ACF_ERROR_LIMIT,
        "rice_mean": rice_mean_interval[0] <= expected_count <= rice_mean_interval[1],
        "variance_passing_rate": variance_passing >= PASSING_RATE_MIN,
        "crossing_passing_rate": crossing_passing >= PASSING_RATE_MIN,
    }
    full_after_decay = lags_s >= 5.0 * state.tp_s
    return {
        "sea_state": {"hs_m": state.hs_m, "tp_s": state.tp_s, "gamma": state.gamma},
        "construction": grid.name,
        "active_components": int(grid.indices.size),
        "fft_period_s": grid.fft_n * DT_S,
        "m0_m2": m0,
        "m2_m2_s2": m2,
        "acf_envelope_max_analysis": float(np.max(envelope[analysis])),
        "acf_envelope_max_full_after_decay": float(np.max(envelope[full_after_decay])),
        "acf_recurrence_intervals_s": _intervals_above(
            lags_s[full_after_decay], envelope[full_after_decay], ENVELOPE_LIMIT
        ),
        "empirical_acf_max_abs_normalized_error": float(np.max(normalized_error)),
        "empirical_acf_rmse_normalized": float(np.sqrt(np.mean(normalized_error**2))),
        "rice_rate_hz": rice_rate,
        "empirical_upcrossing_rate_hz": evaluation_mean / DURATION_S,
        "rice_expected_count": expected_count,
        "rice_mean_predictive_interval_count": rice_mean_interval,
        "crossing_count_predictive_interval": crossing_interval,
        "record_variance_predictive_mean": variance_mean,
        "record_variance_predictive_variance": variance_of_variance,
        "record_variance_predictive_interval": variance_interval,
        "record_variance_empirical_mean": float(
            np.mean(record_variances[CALIBRATION_REALIZATIONS:])
        ),
        "record_variance_empirical_variance": float(
            np.var(record_variances[CALIBRATION_REALIZATIONS:], ddof=1)
        ),
        "variance_suppression_factor": float(
            np.var(record_variances[CALIBRATION_REALIZATIONS:], ddof=1) / variance_of_variance
        ),
        "variance_passing_rate": variance_passing,
        "crossing_passing_rate": crossing_passing,
        "gates": gates,
        "passes": all(gates.values()),
        "plot": {
            "lags_s": lags_s[::10].tolist(),
            "normalized_envelope": envelope[::10].tolist(),
        },
    }


def _plot_acf(rows: list[dict[str, object]], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for row in rows:
        plot = row["plot"]
        axis.plot(
            plot["lags_s"],
            plot["normalized_envelope"],
            label=_state_name(SeaState(**row["sea_state"])),
        )
    axis.axhline(ENVELOPE_LIMIT, color="black", linestyle="--", linewidth=1.0, label="5% gate")
    axis.axvline(
        MAX_ANALYSIS_LAG_S,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="440 s scored-unit limit",
    )
    axis.set(xlabel="lag (s)", ylabel="normalized covariance envelope", title=title, yscale="log")
    axis.set_ylim(1e-5, 1.2)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_phase1(output_root: Path) -> dict[str, object]:
    sources, states = _campaign_states()
    rows = [
        _analyze_grid(state, _spectral_grid(state, "production_1x", 1, jitter=False))
        for state in states
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    figure = output_root / "w1_acf_envelope_w1.png"
    _plot_acf(rows, figure, "W1 production covariance-envelope audit")
    payload: dict[str, object] = {
        "experiment": "W1 Phase 1 wave-only diagnostics",
        "preregistration": "results/w1_preregistration_w1.json",
        "campaign_config_sources": sources,
        "realizations": REALIZATIONS,
        "predictive_split": [CALIBRATION_REALIZATIONS, REALIZATIONS - CALIBRATION_REALIZATIONS],
        "figure": str(figure),
        "rows": rows,
        "all_preregistered_gates_pass": all(bool(row["passes"]) for row in rows),
    }
    write_result(output_root, "w1_phase1_w1", payload)
    return payload


def _step_transition(state_before: SeaState, state_after: SeaState) -> dict[str, object]:
    duration_s = 900.0
    count = round(duration_s / DT_S)
    boundary = round(300.0 / DT_S)
    ramp_half_width = round(2.5 * state_after.tp_s / DT_S)
    independent_jumps = []
    continuous_jumps = []
    independent_variances = []
    continuous_variances = []
    independent_crossings = []
    continuous_crossings = []
    continuous_grid = _spectral_grid(
        SeaState(hs_m=1.0, tp_s=state_after.tp_s, gamma=state_after.gamma),
        "continuous_phase",
        2,
        jitter=False,
    )
    window = round(5.0 * state_after.tp_s / DT_S)
    for seed in range(SEED_START, SEED_START + REALIZATIONS):
        before_seed = int(
            np.random.SeedSequence([seed, 0, 0]).generate_state(1, dtype=np.uint64)[0]
        )
        after_seed = int(np.random.SeedSequence([seed, 1, 0]).generate_state(1, dtype=np.uint64)[0])
        before = synthesize_jonswap(
            state_before,
            duration_s=300.0,
            dt_s=DT_S,
            seed=before_seed,
            max_frequency_rad_s=MAX_FREQUENCY_RAD_S,
        ).elevation_m[:-1]
        after = synthesize_jonswap(
            state_after,
            duration_s=600.0,
            dt_s=DT_S,
            seed=after_seed,
            max_frequency_rad_s=MAX_FREQUENCY_RAD_S,
        ).elevation_m[:-1]
        independent = np.concatenate((before, after))
        base = _realization(continuous_grid, seed, count)
        scale = np.full(count, state_before.hs_m)
        ramp = slice(boundary - ramp_half_width, boundary + ramp_half_width + 1)
        scale[ramp] = np.linspace(state_before.hs_m, state_after.hs_m, 2 * ramp_half_width + 1)
        scale[boundary + ramp_half_width + 1 :] = state_after.hs_m
        continuous = base * scale
        local = slice(boundary - window, boundary + window)
        independent_jumps.append(abs(independent[boundary] - independent[boundary - 1]))
        continuous_jumps.append(abs(continuous[boundary] - continuous[boundary - 1]))
        independent_variances.append(float(np.var(independent[local])))
        continuous_variances.append(float(np.var(continuous[local])))
        independent_crossings.append(_upcrossings(independent[local]))
        continuous_crossings.append(_upcrossings(continuous[local]))
    return {
        "sea_state_before": {
            "hs_m": state_before.hs_m,
            "tp_s": state_before.tp_s,
            "gamma": state_before.gamma,
        },
        "sea_state_after": {
            "hs_m": state_after.hs_m,
            "tp_s": state_after.tp_s,
            "gamma": state_after.gamma,
        },
        "ramp_duration_s": 5.0 * state_after.tp_s,
        "boundary_absolute_first_difference_mean": {
            "independent_phase": float(np.mean(independent_jumps)),
            "continuous_phase_ramp": float(np.mean(continuous_jumps)),
        },
        "boundary_10tp_variance_mean": {
            "independent_phase": float(np.mean(independent_variances)),
            "continuous_phase_ramp": float(np.mean(continuous_variances)),
        },
        "boundary_10tp_upcrossings_mean": {
            "independent_phase": float(np.mean(independent_crossings)),
            "continuous_phase_ramp": float(np.mean(continuous_crossings)),
        },
    }


def _plot_comparison(rows: list[dict[str, object]], output_root: Path) -> list[str]:
    figures: list[str] = []
    for metric, filename, ylabel in (
        ("variance_passing_rate", "w1_passing_rates_w1.png", "predictive passing rate"),
        (
            "variance_suppression_factor",
            "w1_variability_comparison_w1.png",
            "variance suppression factor",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9.0, 4.8))
        labels = [
            f"{row['construction']}\n{_state_name(SeaState(**row['sea_state']))}" for row in rows
        ]
        axis.bar(np.arange(len(rows)), [row[metric] for row in rows])
        if metric == "variance_passing_rate":
            axis.scatter(
                np.arange(len(rows)),
                [row["crossing_passing_rate"] for row in rows],
                color="black",
                s=12,
                label="crossing",
            )
            axis.axhline(
                PASSING_RATE_MIN, color="red", linestyle="--", linewidth=1.0, label="0.90 gate"
            )
            axis.legend()
        axis.set_xticks(np.arange(len(rows)), labels, rotation=90, fontsize=6)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        path = output_root / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figures.append(str(path))
    return figures


def run_phase2(output_root: Path) -> dict[str, object]:
    phase1 = load_result(output_root, "w1_phase1_w1")
    _, states = _campaign_states()
    rows = []
    for state in states:
        for name, factor, jitter in (
            ("extended_8x", 8, False),
            ("extended_16x", 16, False),
            ("jittered_bin_16x", 16, True),
        ):
            rows.append(_analyze_grid(state, _spectral_grid(state, name, factor, jitter=jitter)))
    figures = _plot_comparison(list(phase1["rows"]) + rows, output_root)
    _plot_acf(
        rows,
        output_root / "w1_acf_sensitivity_w1.png",
        "W1 construction-sensitivity covariance envelopes",
    )
    figures.append(str(output_root / "w1_acf_sensitivity_w1.png"))
    step = _step_transition(
        SeaState(hs_m=2.0, tp_s=4.0, gamma=3.3),
        SeaState(hs_m=5.0, tp_s=4.0, gamma=3.3),
    )
    payload: dict[str, object] = {
        "experiment": "W1 Phase 2 construction sensitivity",
        "preregistration": "results/w1_preregistration_w1.json",
        "rows": rows,
        "step_transition": step,
        "figures": figures,
    }
    write_result(
        output_root,
        "w1_phase2_w1",
        payload,
        upstream_results={"w1_phase1_w1": phase1},
    )
    return payload


def run_decision(output_root: Path) -> dict[str, object]:
    phase1 = load_result(output_root, "w1_phase1_w1")
    phase2 = load_result(output_root, "w1_phase2_w1")
    preregistration = json.loads(
        (_repository_root() / "results" / "w1_preregistration_w1.json").read_text()
    )
    passed = bool(phase1["all_preregistered_gates_pass"])
    payload: dict[str, object] = {
        "experiment": "W1 Phase 3 decision",
        "production_passed": passed,
        "decision_verbatim": preregistration[
            "pass_decision_verbatim" if passed else "fail_decision_verbatim"
        ],
        "frozen_artifacts_touched": [],
        "reserve_blocks_read": [],
        "labeled_window_max_lag_s": MAX_ANALYSIS_LAG_S,
        "labeled_window_exposure_verified": all(
            row["acf_envelope_max_analysis"] <= ENVELOPE_LIMIT for row in phase1["rows"]
        ),
    }
    write_result(
        output_root,
        "w1_decision_w1",
        payload,
        upstream_results={"w1_phase1_w1": phase1, "w1_phase2_w1": phase2},
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase1", "phase2", "decision"))
    parser.add_argument("--out", type=Path, default=_repository_root() / "results")
    args = parser.parse_args(argv)
    result = {
        "phase1": run_phase1,
        "phase2": run_phase2,
        "decision": run_decision,
    }[args.phase](args.out)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
