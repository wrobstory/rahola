from __future__ import annotations

import pytest
from rahola_lab.constants import SeedBlock
from rahola_lab.evaluation import ReserveBlockError, seeds_for
from rahola_lab.experiments import final_eval


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
    monkeypatch.setattr(final_eval, "_git_output", lambda *args: " M dirty.py")

    with pytest.raises(final_eval.FinalEvaluationError, match="clean"):
        final_eval.run_final_evaluation(
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            config_root=tmp_path / "configs",
            reserve_root=tmp_path / "reserve",
        )

    assert not (tmp_path / "results" / "final_reserve2_attestation.json").exists()
