from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rahola_lab.experiments import w1
from rahola_lab.experiments.common import load_result

import rahola.spectrum as production_spectrum
from rahola.config import SeaState


def test_reference_spectrum_is_independent_of_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SeaState(hs_m=2.0, tp_s=4.0, gamma=3.3)
    grid = w1._spectral_grid(state, "production_1x", 1, jitter=False)
    values = w1._production_realization(state, 17)
    assert np.var(values) == pytest.approx(np.sum(grid.energy_m2), rel=2e-12)
    original = production_spectrum.jonswap_spectrum
    monkeypatch.setattr(
        production_spectrum,
        "jonswap_spectrum",
        lambda *args, **kwargs: 2.0 * original(*args, **kwargs),
    )
    mutated_grid = w1._spectral_grid(state, "production_1x", 1, jitter=False)
    np.testing.assert_array_equal(mutated_grid.energy_m2, grid.energy_m2)
    assert np.var(w1._production_realization(state, 17)) == pytest.approx(
        2.0 * np.sum(grid.energy_m2), rel=2e-12
    )


def test_sampled_crossing_rate_and_degenerate_variance_prediction_are_finite() -> None:
    frequency_hz = 0.5
    covariance = np.array([1.0, np.cos(2.0 * np.pi * frequency_hz * w1.DT_S)])
    assert w1._sampled_upcrossing_rate(covariance) == pytest.approx(frequency_hz)
    assert w1._variance_prediction(np.ones(8)) == (0.0, 0.0, [0.0, 0.0])
    for invalid in ([], [1.0], [1.0, np.nan], [0.0, 0.0], [1.0, 2.0]):
        with pytest.raises(ValueError, match="covariance"):
            w1._sampled_upcrossing_rate(np.asarray(invalid))


def test_variance_prediction_matches_dense_quadratic_form() -> None:
    n = 8
    delta = 1e-8
    covariance = np.array([1.0] + [1.0 - delta] * (n - 1))
    indices = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    covariance_matrix = covariance[indices]
    centering = np.eye(n) - np.ones((n, n)) / n
    expected_mean = np.trace(centering @ covariance_matrix) / n
    expected_variance = (
        2.0 * np.trace(centering @ covariance_matrix @ centering @ covariance_matrix) / n**2
    )
    mean, variance, interval = w1._variance_prediction(covariance)
    assert mean == pytest.approx(expected_mean, rel=1e-8)
    assert variance == pytest.approx(expected_variance, rel=1e-8)
    assert interval[0] < interval[1]


def test_w1_seed_streams_are_distinct_and_failed_gates_fail_the_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    assert w1._derived_seed(123, 0) != w1._derived_seed(123, 1)
    monkeypatch.setattr(
        w1,
        "run_phase1",
        lambda _: {"all_preregistered_gates_pass": False},
    )
    assert w1.main(["phase1", "--out", str(tmp_path)]) == 1
    monkeypatch.setattr(
        w1,
        "run_phase2",
        lambda _: {"step_transition": {"passes_postregistration_audit_checks": False}},
    )
    assert w1.main(["phase2", "--out", str(tmp_path)]) == 1
    monkeypatch.setattr(w1, "run_decision", lambda _: {"production_passed": False})
    assert w1.main(["decision", "--out", str(tmp_path)]) == 1


def test_committed_w1_graph_is_exactly_producer_generated(tmp_path: Path) -> None:
    committed_root = w1._repository_root() / "results"
    preregistration = json.loads((committed_root / "w1_preregistration_w1.json").read_text())
    assert preregistration["tolerances"]["confidence_level"] == w1.CONFIDENCE_LEVEL
    phase1 = w1.run_phase1(tmp_path)
    phase2 = w1.run_phase2(tmp_path)
    decision = w1.run_decision(tmp_path)
    assert phase1["all_corrected_diagnostic_gates_pass"]
    assert all(row["passes_corrected_diagnostic_gates"] for row in phase2["rows"])
    assert phase2["step_transition"]["passes_postregistration_audit_checks"]
    assert decision["corrected_diagnostics_passed"]
    assert not decision["production_passed"]
    assert all("plot" not in row for row in phase1["rows"] + phase2["rows"])
    for name in ("w1_phase1_w1.json", "w1_phase2_w1.json", "w1_decision_w1.json"):
        assert (tmp_path / name).read_bytes() == (committed_root / name).read_bytes()
    assert load_result(tmp_path, "w1_decision_w1") == load_result(committed_root, "w1_decision_w1")
