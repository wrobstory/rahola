from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.experiments.common import (
    load_result,
    result_graph_lock,
    trajectory_forecasts,
    write_result,
)
from rahola_lab.experiments.detector_common import detector_risk_end_s, window_auc

from rahola.dataset import SimulationDataset


def test_result_writer_serializes_non_finite_values_as_null(tmp_path: Path) -> None:
    path = write_result(tmp_path, "nonfinite", {"missing": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8"))["missing"] is None
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert load_result(tmp_path, "nonfinite")["missing"] is None


def test_result_loader_rejects_stale_provenance(tmp_path: Path) -> None:
    path = write_result(tmp_path, "stale", {"value": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_provenance"]["source_sha256"] = "stale"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="current source"):
        load_result(tmp_path, "stale")


def test_result_loader_rejects_content_mutation_and_records_upstream_digest(
    tmp_path: Path,
) -> None:
    upstream_path = write_result(tmp_path, "upstream", {"value": 1})
    upstream = load_result(tmp_path, "upstream")
    downstream_path = write_result(
        tmp_path,
        "downstream",
        {"derived": 2},
        upstream_results={"upstream": upstream},
    )
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    assert downstream["_provenance"]["upstream_artifacts"] == {
        "upstream": upstream["_artifact_sha256"]
    }

    mutated = json.loads(upstream_path.read_text(encoding="utf-8"))
    mutated["value"] = 99
    upstream_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest"):
        load_result(tmp_path, "upstream")


def test_result_loader_rejects_validly_replaced_upstream(tmp_path: Path) -> None:
    write_result(tmp_path, "upstream", {"value": 1})
    write_result(
        tmp_path,
        "downstream",
        {"derived": 1},
        upstream_results={"upstream": load_result(tmp_path, "upstream")},
    )
    write_result(tmp_path, "upstream", {"value": 2})
    with pytest.raises(ValueError, match="current upstream artifact"):
        load_result(tmp_path, "downstream")


def test_result_writer_rejects_stale_in_memory_upstream(tmp_path: Path) -> None:
    write_result(tmp_path, "upstream", {"value": 1})
    stale = load_result(tmp_path, "upstream")
    write_result(tmp_path, "upstream", {"value": 2})
    with pytest.raises(ValueError, match="no longer current"):
        write_result(
            tmp_path,
            "downstream",
            {"derived": 1},
            upstream_results={"upstream": stale},
        )


def test_result_writer_rejects_dependency_cycle(tmp_path: Path) -> None:
    write_result(tmp_path, "a", {"value": 1})
    write_result(
        tmp_path,
        "b",
        {"value": 2},
        upstream_results={"a": load_result(tmp_path, "a")},
    )
    with pytest.raises(ValueError, match="cyclic upstream"):
        write_result(
            tmp_path,
            "a",
            {"value": 3},
            upstream_results={"b": load_result(tmp_path, "b")},
        )


def test_result_loader_rejects_nonstandard_nonfinite_json(tmp_path: Path) -> None:
    path = write_result(tmp_path, "finite", {"missing": None})
    path.write_text(path.read_text(encoding="utf-8").replace("null", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_result(tmp_path, "finite")


def test_result_graph_lock_blocks_concurrent_writer(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    done = tmp_path / "done"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from rahola_lab.experiments.common import write_result\n"
        "root, ready, done = map(Path, sys.argv[1:])\n"
        "ready.write_text('ready')\n"
        "write_result(root, 'concurrent', {'value': 1})\n"
        "done.write_text('done')\n"
    )
    with result_graph_lock(tmp_path):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(ready), str(done)]
        )
        deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        time.sleep(0.1)
        assert not done.exists()
    assert process.wait(timeout=5.0) == 0
    assert done.exists()


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "..", ""])
def test_result_artifact_names_cannot_escape_output_root(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match="invalid result artifact name"):
        load_result(tmp_path, name)


def test_result_replacement_is_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_result(tmp_path, "result", {"value": 1})
    original = path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr("rahola_lab.experiments.common.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        write_result(tmp_path, "result", {"value": 2})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".result.json.*.tmp"))


class _ZeroForecaster:
    def predict(self, histories: np.ndarray) -> np.ndarray:
        return np.zeros((len(histories), 1), dtype=np.float64)


def test_trajectory_forecast_record_ends_at_last_scored_endpoint() -> None:
    time_s = np.arange(201, dtype=np.float64)
    angle = np.zeros((1, len(time_s)), dtype=np.float64)
    dataset = SimulationDataset(
        time_s=time_s,
        angle_rad=angle,
        rate_rad_s=angle.copy(),
        seeds=np.array([1], dtype=np.uint64),
        capsized=np.array([False]),
        t_capsize_s=np.array([np.nan]),
        metadata=({"seed": 1},),
        config={"escape_angle_rad": 0.5},
    )
    stream = trajectory_forecasts(
        dataset,
        {"zero": _ZeroForecaster()},
        60.0,
        stride_s=10.0,
        first_history_end_s=120.0,
    )[0]
    assert stream.times_s.tolist() == [120.0, 130.0, 140.0]
    assert stream.record_end_s == stream.times_s[-1]


def test_detector_risk_end_uses_common_horizon_complete_cutoff() -> None:
    times = np.arange(240.0, 601.0, 10.0)
    assert detector_risk_end_s(
        times,
        t_capsize_s=np.nan,
        raw_record_end_s=600.0,
        horizon_s=200.0,
        record_start_s=240.0,
    ) == 400.0


def test_window_auc_excludes_inference_tail_and_ambiguity_buffer() -> None:
    times = np.array([300.0, 330.0, 400.0, 500.0])
    trajectories = [
        TrajectoryScores(
            times_s=times,
            scores=np.array([0.0, 1.0, 0.0, 1.0]),
            record_start_s=300.0,
            record_end_s=400.0,
            t_capsize_s=550.0,
        ),
        TrajectoryScores(
            times_s=times,
            scores=np.array([0.0, 0.0, 0.0, 0.0]),
            record_start_s=300.0,
            record_end_s=400.0,
        ),
    ]
    # At 330 s the event is 220 s away and lies in the 20 s exclusion buffer.
    # The 500 s clock-only tail is beyond the common horizon-complete cutoff.
    assert window_auc(trajectories) == pytest.approx(0.5)
    assert detector_risk_end_s(
        times[times < 550.0],
        t_capsize_s=550.0,
        raw_record_end_s=600.0,
        horizon_s=200.0,
        record_start_s=240.0,
    ) == 400.0
