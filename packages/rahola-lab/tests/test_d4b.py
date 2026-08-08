from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from rahola_lab.experiments.common import load_result

from rahola.config import SeaState

_SPEC = importlib.util.spec_from_file_location(
    "rahola_d4b_producer", Path(__file__).resolve().parents[3] / "d4b.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_D4B = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _D4B
_SPEC.loader.exec_module(_D4B)

D4B_TEST_RANGES = _D4B.D4B_TEST_RANGES
ExtendedSea = _D4B.ExtendedSea
_auc = _D4B._auc
_fit_logistic = _D4B._fit_logistic
_reliability_edges = _D4B._reliability_edges
bisect_threshold = _D4B.bisect_threshold
cluster_groups = _D4B.cluster_groups
detect_groups = _D4B.detect_groups
embed_group = _D4B.embed_group
synthesize_extended_jonswap = _D4B.synthesize_extended_jonswap


def test_committed_d4b_artifact_graph_is_current() -> None:
    results = Path(__file__).resolve().parents[3] / "results"
    load_result(results, "d4b_uncertainty_d4b")
    load_result(results, "d4b_observability_d4b")


def test_extended_synthesis_is_step_halving_invariant() -> None:
    state = SeaState(hs_m=4.0, tp_s=4.0, gamma=3.3)
    common = {
        "sea_state": state,
        "duration_s": 64.0,
        "seed": 180_000,
        "period_factor": 8,
        "max_frequency_rad_s": 20.0 * np.pi,
    }
    coarse = synthesize_extended_jonswap(dt_s=0.05, **common)
    fine = synthesize_extended_jonswap(dt_s=0.025, **common)

    assert coarse.fft_period_s == fine.fft_period_s == 512.0
    np.testing.assert_allclose(coarse.elevation_m, fine.elevation_m[::2], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(coarse.slope_rad, fine.slope_rad[::2], rtol=0.0, atol=2e-14)


def test_hand_placed_wave_groups_recover_count_and_parameters() -> None:
    dt_s = 0.05
    time_s = np.arange(0.0, 140.0 + dt_s, dt_s)
    carrier_period_s = 4.0
    envelope = (
        0.01
        + 2.0 * np.exp(-0.5 * np.square((time_s - 30.0) / 5.0))
        + 2.5 * np.exp(-0.5 * np.square((time_s - 100.0) / 5.0))
    )
    elevation = envelope * np.cos(2.0 * np.pi * time_s / carrier_period_s)

    groups = detect_groups(
        time_s,
        elevation,
        source_seed=7,
        significant_height_m=4.0,
        peak_period_s=carrier_period_s,
    )

    assert len(groups) == 2
    assert [group.carrier_period_s for group in groups] == pytest.approx(
        [carrier_period_s, carrier_period_s], rel=1e-5
    )
    assert [group.central_height_m for group in groups] == pytest.approx([4.02, 5.02], rel=1e-5)
    assignments, medoids, _, _ = cluster_groups(list(groups), 2)
    assert sorted(assignments.tolist()) == [0, 1]
    assert sorted(medoids.tolist()) == [0, 1]


def test_embedding_preserves_prefix_and_target_parameters() -> None:
    dt_s = 0.05
    time_s = np.arange(0.0, 100.0 + dt_s, dt_s)
    rng = np.random.default_rng(8)
    original = rng.normal(scale=0.05, size=len(time_s))
    prelude = ExtendedSea(time_s, original, np.zeros_like(original), 800.0)
    target_time = np.arange(0.0, 24.0 + dt_s, dt_s)
    envelope = 0.01 + 2.0 * np.exp(-0.5 * np.square((target_time - 12.0) / 5.0))
    target = envelope * np.cos(2.0 * np.pi * target_time / 4.0)
    target_groups = detect_groups(
        target_time,
        target,
        source_seed=1,
        significant_height_m=4.0,
        peak_period_s=4.0,
    )

    composite = embed_group(
        prelude,
        target,
        np.zeros_like(target),
        arrival_s=50.0,
        blend_half_width_s=4.0,
    )

    np.testing.assert_array_equal(
        composite.elevation_m[: composite.blend_start_index],
        prelude.elevation_m[: composite.blend_start_index],
    )
    blend_samples = round(4.0 / dt_s)
    np.testing.assert_array_equal(
        composite.elevation_m[composite.plateau_start_index : composite.plateau_stop_index],
        target[blend_samples:-blend_samples],
    )
    embedded_window = composite.elevation_m[
        composite.target_start_index : composite.target_stop_index
    ]
    embedded_groups = detect_groups(
        target_time,
        embedded_window,
        source_seed=2,
        significant_height_m=4.0,
        peak_period_s=4.0,
    )
    assert len(target_groups) == len(embedded_groups) == 1
    assert embedded_groups[0].carrier_period_s == pytest.approx(
        target_groups[0].carrier_period_s, rel=0.01
    )
    assert embedded_groups[0].central_height_m == pytest.approx(
        target_groups[0].central_height_m, rel=0.02
    )


def test_bisection_recovers_known_critical_height() -> None:
    threshold = bisect_threshold(
        lambda height: height >= 3.125,
        0.0,
        10.0,
        tolerance=1e-6,
        max_iterations=32,
    )
    assert threshold == pytest.approx(3.125, abs=1e-6)
    with pytest.raises(ValueError, match="false at lower and true at upper"):
        bisect_threshold(lambda _: False, 0.0, 1.0, tolerance=0.01, max_iterations=8)


def test_d4b_test_ranges_are_fresh_ordinary_seeds() -> None:
    ledgered = (
        (200_000, 201_000),
        (201_000, 202_500),
        (205_000, 206_000),
        (206_000, 209_200),
        (210_000, 211_000),
        (211_000, 220_000),
        (221_000, 222_700),
        (225_000, 226_000),
        (230_000, 231_000),
        (235_000, 238_000),
        (238_000, 241_000),
        (241_000, 244_000),
        (244_000, 247_000),
        (250_000, 255_000),
        (260_000, 265_000),
        (268_000, 269_000),
        (270_000, 275_000),
        (276_000, 277_000),
        (277_000, 299_900),
    )
    for start, stop in D4B_TEST_RANGES:
        assert 200_000 <= start < stop <= 300_000
        assert all(stop <= used_start or start >= used_stop for used_start, used_stop in ledgered)


def test_penalized_logistic_fit_is_finite_and_orders_known_signal() -> None:
    features = np.arange(-4.0, 5.0, dtype=np.float64)[:, None]
    labels = features[:, 0] > 0.0
    fit = _fit_logistic(features, labels, penalty=1e-4)
    scores = fit.predict(features)
    assert np.all(np.isfinite(fit.coefficients))
    assert np.all(np.diff(scores) > 0.0)
    assert _auc(labels, scores) == pytest.approx(1.0)
    assert np.all(np.isfinite(_reliability_edges(scores)))
