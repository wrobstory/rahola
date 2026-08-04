from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rahola_lab.campaigns import U1R2_TEST_SLICES, verify_u1r2_test_slices
from rahola_lab.evaluation import EpisodeConfig, TrajectoryScores, operating_curve
from rahola_lab.experiments.u1c import _point_with_bootstrap_or_unevaluable


def _manifest(root: Path, name: str, *, offset: int, count: int) -> None:
    campaign = root / name
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "test": {
                        "offset": offset,
                        "count": count,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_u1r2_slices_are_pairwise_disjoint_and_reject_existing_overlap(
    tmp_path: Path,
) -> None:
    intervals = [(offset, offset + count) for count, offset in U1R2_TEST_SLICES.values()]
    assert all(
        left[1] <= right[0] or right[1] <= left[0]
        for index, left in enumerate(intervals)
        for right in intervals[index + 1 :]
    )
    _manifest(tmp_path, "historical", offset=0, count=1_000)
    assert verify_u1r2_test_slices((tmp_path,))["disjoint_from_existing_manifests"]
    _manifest(tmp_path, "conflict", offset=11_500, count=100)
    with pytest.raises(ValueError, match="overlaps"):
        verify_u1r2_test_slices((tmp_path,))


def test_sparse_u1c_stream_is_durably_unevaluable() -> None:
    trajectories = [
        TrajectoryScores(
            times_s=np.empty(0),
            scores=np.empty(0),
            record_end_s=100.0,
            record_start_s=10.0,
        )
    ]
    point = operating_curve(
        trajectories,
        EpisodeConfig(threshold=0.0),
        np.asarray([0.0]),
        horizon_s=20.0,
        decorrelation_time_s=1.0,
    )[0]
    payload = _point_with_bootstrap_or_unevaluable(
        point,
        trajectories,
        horizon_s=20.0,
        decorrelation_s=1.0,
    )
    assert payload["trajectory_bootstrap_status"] == ("unevaluable: no finite replicates")
