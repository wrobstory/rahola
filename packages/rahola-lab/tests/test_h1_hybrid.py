from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rahola_lab.campaigns import H1_TEST_SLICES, verify_h1_test_slices
from rahola_lab.experiments.h1_common import (
    cluster_crossings,
    fit_conditional,
    terminal_partition,
    wilson_interval,
)
from rahola_lab.splittime import Crossing


def _crossing(time_s: float, severity: float, index: int) -> Crossing:
    return Crossing(
        time_s=time_s,
        detection_index=index,
        side=1,
        outward_rate_rad_s=severity,
        critical_rate_rad_s=1.0,
        severity_u=severity,
    )


def _manifest(root: Path, name: str, *, offset: int, count: int) -> None:
    campaign = root / name
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(
        json.dumps({"splits": {"test": {"offset": offset, "count": count}}}),
        encoding="utf-8",
    )


def test_terminal_partition_uses_cluster_last_crossing_not_retained_maximum() -> None:
    clusters = cluster_crossings(
        (_crossing(1.0, 1.5, 1), _crossing(2.0, 0.5, 2)),
        decorrelation_time_s=2.0,
    )
    assert clusters[0].retained.time_s == 1.0
    assert clusters[0].last.time_s == 2.0
    partition = terminal_partition(
        clusters,
        capsized=True,
        t_capsize_s=3.0,
        decorrelation_time_s=2.0,
    )
    assert partition.terminal_labels == (True,)
    assert partition.heralded and not partition.unheralded


def test_terminal_partition_is_exhaustive_and_leaves_nonterminal_clusters_unlabeled() -> None:
    clusters = cluster_crossings(
        (_crossing(1.0, 0.5, 1), _crossing(5.0, 0.8, 5)),
        decorrelation_time_s=1.0,
    )
    heralded = terminal_partition(
        clusters,
        capsized=True,
        t_capsize_s=5.5,
        decorrelation_time_s=1.0,
    )
    assert heralded.terminal_labels == (False, True)
    unheralded = terminal_partition(
        clusters,
        capsized=True,
        t_capsize_s=8.0,
        decorrelation_time_s=1.0,
    )
    assert unheralded.terminal_labels == (False, False)
    assert unheralded.unheralded and not unheralded.heralded
    ordinary = terminal_partition(
        clusters,
        capsized=False,
        t_capsize_s=None,
        decorrelation_time_s=1.0,
    )
    assert ordinary.terminal_labels == (False, False)
    assert not ordinary.heralded and not ordinary.unheralded


def test_isotonic_conditional_recovers_monotone_synthetic_relation() -> None:
    rng = np.random.default_rng(20_260_804)
    severity = rng.uniform(0.0, 1.0, 20_000)
    truth = 0.02 + 0.78 * severity
    terminal = rng.random(len(severity)) < truth
    model = fit_conditional(severity, terminal)
    estimates = np.asarray([model.predict(value)[0] for value in np.linspace(0.05, 0.95, 10)])
    expected = 0.02 + 0.78 * np.linspace(0.05, 0.95, 10)
    assert np.all(np.diff(model.point) >= 0.0)
    assert np.mean(np.abs(estimates - expected)) < 0.05
    assert np.all(model.lower <= model.point)
    assert np.all(model.point <= model.upper)


def test_wilson_interval_has_nominal_simulated_binomial_coverage() -> None:
    rng = np.random.default_rng(31_415)
    probability = 0.2
    draws = rng.binomial(100, probability, size=10_000)
    coverage = np.mean(
        [
            lower <= probability <= upper
            for lower, upper in (wilson_interval(int(value), 100) for value in draws)
        ]
    )
    assert 0.93 <= coverage <= 0.98


def test_h1_slices_are_pairwise_disjoint_and_reject_overlap(tmp_path: Path) -> None:
    intervals = [(offset, offset + count) for count, offset in H1_TEST_SLICES.values()]
    assert all(
        left[1] <= right[0] or right[1] <= left[0]
        for index, left in enumerate(intervals)
        for right in intervals[index + 1 :]
    )
    _manifest(tmp_path, "historical", offset=0, count=1_000)
    report = verify_h1_test_slices((tmp_path,))
    assert report["disjoint_from_existing_manifests"]
    assert report["declared_count"] == 7_900
    _manifest(tmp_path, "conflict", offset=92_250, count=10)
    with pytest.raises(ValueError, match="overlaps"):
        verify_h1_test_slices((tmp_path,))
