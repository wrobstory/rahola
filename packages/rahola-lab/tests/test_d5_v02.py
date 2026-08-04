from __future__ import annotations

import numpy as np
from rahola_lab.evaluation import TrajectoryScores
from rahola_lab.experiments.d5_v02 import fully_post_step


def test_d5_v02_requires_full_post_step_history_and_complete_horizon() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([530.0, 540.0, 700.0, 710.0]),
        scores=np.arange(4, dtype=np.float64),
        record_start_s=240.0,
        record_end_s=700.0,
        t_capsize_s=None,
    )
    filtered = fully_post_step({"method": [trajectory]})["method"][0]
    assert filtered.times_s.tolist() == [540.0, 700.0]
    assert filtered.record_start_s == 540.0
    assert filtered.record_end_s == 700.0
