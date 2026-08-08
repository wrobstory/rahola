from __future__ import annotations

import json

import numpy as np
import pytest
from rahola_lab.experiments import w1
from rahola_lab.experiments.common import load_result

from rahola.config import SeaState


def test_reference_spectrum_matches_production_realization_energy() -> None:
    state = SeaState(hs_m=2.0, tp_s=4.0, gamma=3.3)
    grid = w1._spectral_grid(state, "production_1x", 1, jitter=False)
    values = w1._production_realization(state, 17)
    assert np.var(values) == pytest.approx(np.sum(grid.energy_m2), rel=2e-12)


def test_sampled_crossing_rate_and_degenerate_variance_prediction_are_finite() -> None:
    frequency_hz = 0.5
    covariance = np.array([1.0, np.cos(2.0 * np.pi * frequency_hz * w1.DT_S)])
    assert w1._sampled_upcrossing_rate(covariance) == pytest.approx(frequency_hz)
    assert w1._variance_prediction(np.ones(8)) == (0.0, 0.0, [0.0, 0.0])
    near_rank_one = np.array([1.0] + [1.0 - 1e-15] * 7)
    mean, variance, interval = w1._variance_prediction(near_rank_one)
    assert np.all(np.isfinite([mean, variance, *interval]))


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


def test_committed_w1_decision_is_exactly_producer_generated() -> None:
    output_root = w1._repository_root() / "results"
    phase1 = load_result(output_root, "w1_phase1_w1")
    phase2 = load_result(output_root, "w1_phase2_w1")
    preregistration = json.loads((output_root / "w1_preregistration_w1.json").read_text())
    assert preregistration["tolerances"]["confidence_level"] == w1.CONFIDENCE_LEVEL
    committed = load_result(output_root, "w1_decision_w1")
    committed.pop("_artifact_sha256")
    committed.pop("_provenance")
    assert committed == w1._decision_payload(phase1, phase2, preregistration)
    assert all("plot" not in row for row in phase1["rows"] + phase2["rows"])
