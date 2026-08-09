"""Serialize attained validation-gate residuals for the submission record.

Mirrors the acceptance computations in tests/test_spectrum.py and
tests/test_validation.py exactly, and writes the attained values next to
their thresholds so the manuscript's validation claims are bound to a
versioned artifact rather than to a transient pytest run.

Usage: PYTHONPATH=src uv run python examples/validation_acceptance.py
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from rahola.config import ForcingConfig, SeaState, SimulationConfig
from rahola.simulate import simulate_batch
from rahola.spectrum import jonswap_spectrum, synthesize_jonswap
from rahola.validation import (
    damped_mathieu_threshold,
    find_harmonic_capsize_boundary,
    harmonic_capsize_fraction,
    linear_transfer_function,
    mathieu_growth_rate,
    melnikov_heteroclinic_threshold,
)


def hs_gate() -> dict:
    sea_state = SeaState(hs_m=4.0, tp_s=10.0, gamma=3.3)
    realization = synthesize_jonswap(sea_state, 4096.0, 0.25, seed=814)
    recovered = 4.0 * float(np.std(realization.elevation_m))
    rel_error = abs(recovered - sea_state.hs_m) / sea_state.hs_m
    return {
        "target_hs_m": sea_state.hs_m,
        "recovered_hs_m": recovered,
        "attained_rel_error": rel_error,
        "threshold_rel_error": 0.02,
        "passed": rel_error < 0.02,
    }


def linear_variance_gate() -> dict:
    config = SimulationConfig(
        duration_s=2048.0,
        natural_period_s=8.0,
        escape_angle_rad=10.0,
        damping_ratio=0.08,
        quadratic_damping=0.0,
        output_rate_hz=5.0,
        forcing=ForcingConfig(
            sea_state=SeaState(hs_m=2.0, tp_s=8.0, gamma=1.0),
            effective_wave_slope=0.05,
        ),
        linear_restoring=True,
    )
    dataset = simulate_batch(config, range(8))
    simulated = float(
        np.mean(np.var(dataset.angle_rad[:, len(dataset.time_s) // 4 :], axis=1))
    )
    omega = np.linspace(1e-4, math.pi / (0.5 * config.integration_dt_s), 100_000)
    elevation = jonswap_spectrum(omega, config.forcing.sea_state, config.forcing.gravity_m_s2)
    slope = (omega**2 / config.forcing.gravity_m_s2) ** 2 * elevation
    moment = config.omega_n_rad_s**4 * config.forcing.effective_wave_slope**2 * slope
    transfer = linear_transfer_function(omega, config.omega_n_rad_s, config.damping_ratio)
    analytic = float(np.trapezoid(abs(transfer) ** 2 * moment, omega))
    rel_error = abs(simulated - analytic) / analytic
    return {
        "simulated_variance": simulated,
        "analytic_variance": analytic,
        "attained_rel_error": rel_error,
        "threshold_rel_error": 0.06,
        "passed": rel_error < 0.06,
    }


def mathieu_gate() -> dict:
    zeta = 0.02
    prediction = damped_mathieu_threshold(zeta)
    grid = np.linspace(0.85 * prediction, 1.15 * prediction, 13)
    rates = np.asarray([mathieu_growth_rate(value, zeta) for value in grid])
    crossing = float(grid[np.flatnonzero(rates > 0)[0]])
    rel_error = abs(crossing - prediction) / prediction
    return {
        "predicted_threshold": prediction,
        "simulated_crossing": crossing,
        "attained_rel_error": rel_error,
        "threshold_rel_error": 0.10,
        "passed": rel_error < 0.10,
    }


def melnikov_gates() -> dict:
    below: list[dict] = []
    for damping in (0.015, 0.04):
        for frequency in (0.7, 1.0, 1.3):
            melnikov = melnikov_heteroclinic_threshold(damping, frequency)
            fraction = harmonic_capsize_fraction(0.95 * melnikov, frequency, damping)
            below.append(
                {
                    "damping_ratio": damping,
                    "frequency_ratio": frequency,
                    "capsize_fraction_at_0p95_threshold": fraction,
                }
            )
    frequencies = np.array([0.7, 1.0, 1.3])
    damping = 0.015
    predictions = np.array(
        [melnikov_heteroclinic_threshold(damping, f) for f in frequencies]
    )
    simulated = np.array(
        [find_harmonic_capsize_boundary(f, damping) for f in frequencies]
    )
    shape_correlation = float(np.corrcoef(predictions, simulated)[0, 1])
    return {
        "no_capsize_below_threshold": below,
        "boundary_margins": [
            {
                "frequency_ratio": float(f),
                "melnikov_threshold": float(p),
                "simulated_boundary": float(s),
                "boundary_at_or_above": bool(s >= p),
            }
            for f, p, s in zip(frequencies, predictions, simulated)
        ],
        "shape_correlation": shape_correlation,
        "shape_correlation_threshold": 0.8,
        "passed": all(row["capsize_fraction_at_0p95_threshold"] == 0.0 for row in below)
        and bool(np.all(simulated >= predictions))
        and shape_correlation > 0.8,
    }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    record = {
        "commit": commit,
        "command": "PYTHONPATH=src uv run python examples/validation_acceptance.py",
        "gates": {
            "significant_wave_height": hs_gate(),
            "linear_limit_response_variance": linear_variance_gate(),
            "mathieu_principal_boundary": mathieu_gate(),
            "melnikov_necessary_condition": melnikov_gates(),
        },
    }
    record["all_passed"] = all(g["passed"] for g in record["gates"].values())
    out = Path(__file__).resolve().parents[1] / "results" / "validation_acceptance.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out} all_passed={record['all_passed']}")


if __name__ == "__main__":
    main()
