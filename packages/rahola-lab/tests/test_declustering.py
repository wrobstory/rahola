from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.evaluation import (
    AlarmEpisode,
    decluster_episodes,
    decorrelation_lag_from_autocorrelation,
)


def test_autocorrelation_envelope_crossing_is_interpolated_by_hand() -> None:
    autocorrelation = np.array([1.0, 0.6, 0.2, 0.1, 0.04, 0.02])
    # The endpoint envelope falls linearly from 1.00 at lag 0 to 0.02 at lag 5.
    assert decorrelation_lag_from_autocorrelation(autocorrelation) == pytest.approx(
        5.0 * (1.0 - 0.05) / (1.0 - 0.02)
    )


def test_episode_declustering_merges_only_within_decorrelation_time() -> None:
    episodes = (
        AlarmEpisode(1, 2, 10.0, 20.0, 1.2),
        AlarmEpisode(4, 5, 35.0, 45.0, 2.1),
        AlarmEpisode(9, 10, 80.0, 90.0, 1.7),
    )
    clustered = decluster_episodes(episodes, 20.0)
    assert clustered == (
        AlarmEpisode(1, 5, 10.0, 45.0, 2.1),
        AlarmEpisode(9, 10, 80.0, 90.0, 1.7),
    )
