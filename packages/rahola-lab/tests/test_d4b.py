from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.experiments.d4b import (
    ExtendedSea,
    cluster_groups,
    detect_groups,
    embed_group,
)


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
        group_start_index=target_groups[0].start_index,
    )

    np.testing.assert_array_equal(
        composite.elevation_m[: composite.blend_start_index],
        prelude.elevation_m[: composite.blend_start_index],
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
