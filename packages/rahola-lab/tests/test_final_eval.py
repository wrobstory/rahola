from __future__ import annotations

import stat

import numpy as np
import pytest
from rahola_lab.constants import SeedBlock
from rahola_lab.evaluation import ReserveBlockError, TrajectoryScores, seeds_for
from rahola_lab.experiments import final_eval
from rahola_lab.experiments.b2_chronos import _evaluate_scores
from rahola_lab.experiments.common import load_result, write_result
from rahola_lab.experiments.detector_common import relative_fpr_reduction


def _canonical_paths(root):
    return {
        "data_root": root / "data" / "reference",
        "output_root": root / "results",
        "config_root": root
        / "packages"
        / "rahola-lab"
        / "src"
        / "rahola_lab"
        / "campaigns"
        / "configs",
        "reserve_root": root / "data" / "final-reserve2",
    }


@pytest.mark.parametrize("block", [SeedBlock.RESERVE, SeedBlock.RESERVE2])
def test_development_seed_api_refuses_reserves(block: SeedBlock) -> None:
    with pytest.raises(ReserveBlockError):
        seeds_for(block, 1)


def test_final_evaluation_permanently_refuses_spent_reserve(tmp_path) -> None:
    with pytest.raises(final_eval.FinalEvaluationError, match="spent"):
        final_eval.run_final_evaluation(
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            config_root=tmp_path / "configs",
            reserve_root=tmp_path / "reserve",
            reserve_block=SeedBlock.RESERVE,
        )


def test_final_evaluation_refuses_dirty_tree_before_attestation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final_eval, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(final_eval, "_git_output", lambda *args: " M dirty.py")

    with pytest.raises(final_eval.FinalEvaluationError, match="clean"):
        final_eval.run_final_evaluation(**_canonical_paths(tmp_path))

    assert not (tmp_path / "results" / "final_reserve2_attestation.json").exists()


def test_final_evaluation_revalidates_inputs_after_expensive_preflight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    write_result(results, "p3_b2_chronos", {"survives_kill": True})
    write_result(results, "d1_operating_curves", {"selected": {}})
    expected = {
        name: load_result(results, name)["_artifact_sha256"]
        for name in ("p3_b2_chronos", "d1_operating_curves")
    }
    write_result(results, "d1_operating_curves", {"selected": {"changed": True}})
    monkeypatch.setattr(final_eval, "_git_output", lambda *args: "")

    with pytest.raises(final_eval.FinalEvaluationError, match="changed during preflight"):
        final_eval._revalidate_final_inputs(results, expected)


def test_final_evaluation_requires_frozen_survivor_before_attestation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final_eval, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(final_eval, "_git_output", lambda *args: "")
    with pytest.raises(final_eval.FinalEvaluationError, match="survivor"):
        final_eval.run_final_evaluation(**_canonical_paths(tmp_path))
    assert not (tmp_path / "results" / "final_reserve2_attestation.json").exists()


def test_final_evaluation_rejects_invalid_chunk_size_before_attestation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final_eval, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(final_eval, "_git_output", lambda *args: "")
    results = tmp_path / "results"
    results.mkdir()
    write_result(results, "p3_b2_chronos", {"survives_kill": True})
    with pytest.raises(final_eval.FinalEvaluationError, match="chunk_size"):
        final_eval.run_final_evaluation(
            **_canonical_paths(tmp_path), chunk_size=0
        )
    assert not (results / "final_reserve2_attestation.json").exists()


def _trajectory(scores: list[float], *, capsize: float | None) -> TrajectoryScores:
    return TrajectoryScores(
        times_s=np.array([10.0, 20.0, 30.0]),
        scores=np.asarray(scores),
        record_end_s=60.0,
        t_capsize_s=capsize,
    )


def test_chronos_threshold_is_invariant_to_test_outcomes() -> None:
    calibration = [
        *[_trajectory([0.1, 0.7, 0.8], capsize=40.0) for _ in range(8)],
        *[_trajectory([0.1, 0.2, 0.3], capsize=None) for _ in range(8)],
    ]
    test_a = [
        _trajectory([0.2, 0.6, 0.9], capsize=40.0),
        _trajectory([0.2, 0.6, 0.9], capsize=None),
    ]
    test_b = [
        _trajectory([0.2, 0.6, 0.9], capsize=None),
        _trajectory([0.2, 0.6, 0.9], capsize=None),
    ]
    first = _evaluate_scores(calibration, test_a)
    second = _evaluate_scores(calibration, test_b)
    assert first["threshold"] == second["threshold"]
    assert first["threshold"] == first["calibration_operating_point"]["threshold"]


def test_chronos_zero_fpr_baseline_cannot_earn_relative_improvement() -> None:
    assert relative_fpr_reduction(0.0, 0.0) is None
    assert relative_fpr_reduction(1.0, 0.0) is None
    assert relative_fpr_reduction(0.8, 1.0) == pytest.approx(0.2)


def test_attestation_claim_uses_exclusive_creation(tmp_path) -> None:
    path = tmp_path / "attestation.json"
    final_eval._write_exclusive_json(path, {"status": "started"})
    with pytest.raises(final_eval.FinalEvaluationError, match="already started"):
        final_eval._write_exclusive_json(path, {"status": "second"})


def test_attestation_claim_fsyncs_file_and_parent_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_modes = []
    original_fsync = final_eval.os.fsync

    def record_fsync(descriptor: int) -> None:
        fsync_modes.append(final_eval.os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(final_eval.os, "fsync", record_fsync)
    final_eval._write_exclusive_json(tmp_path / "attestation.json", {"status": "started"})
    assert len(fsync_modes) == 2
    assert stat.S_ISDIR(fsync_modes[-1])


def test_terminal_attestation_replacement_is_atomic(tmp_path) -> None:
    path = tmp_path / "attestation.json"
    final_eval._write_exclusive_json(path, {"status": "started"})
    final_eval._write_atomic_json(path, {"status": "complete", "result_sha256": "abc"})
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(".attestation.json.*.tmp"))
