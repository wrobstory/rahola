from __future__ import annotations

from pathlib import Path

import numpy as np
from rahola_lab.campaigns import load_campaign_definition
from rahola_lab.conformal import normalized_alarm_scores
from rahola_lab.forecast import (
    EnvelopePersistenceForecaster,
    JaxLSTMQuantileForecaster,
    LinearQuantileForecaster,
    absolute_roll_escape_angle,
    extract_forecast_dataset,
)

from rahola.dataset import SimulationDataset


def _trajectory_dataset(*, capsized: bool = False) -> SimulationDataset:
    time = np.arange(21, dtype=np.float64)
    angle = 0.01 * time[None, :]
    rate = np.full_like(angle, 0.01)
    cap_time = 17.0 if capsized else np.nan
    if capsized:
        angle[0, 18:] = np.nan
        rate[0, 18:] = np.nan
    return SimulationDataset(
        time_s=time,
        angle_rad=angle,
        rate_rad_s=rate,
        seeds=np.array([4], dtype=np.uint64),
        capsized=np.array([capsized]),
        t_capsize_s=np.array([cap_time]),
        metadata=({},),
        config={"escape_angle_rad": 0.5, "negative_escape_angle_rad": None},
    )


def test_target_extraction_uses_only_history_and_marks_capsize() -> None:
    dataset = extract_forecast_dataset(
        _trajectory_dataset(capsized=True),
        history_s=5,
        horizons_s=(3.0, 6.0),
        stride_s=10,
    )
    assert dataset.histories.shape == (2, 5, 2)
    assert dataset.targets_rad[-1, 1] == 0.5
    assert np.max(dataset.histories[-1, :, 0]) < 0.5


def test_end_of_record_truncated_horizons_are_dropped() -> None:
    dataset = extract_forecast_dataset(
        _trajectory_dataset(), history_s=5, horizons_s=(6.0,), stride_s=1
    )
    assert np.max(dataset.history_end_s) <= 14.0


def test_capsize_does_not_extend_past_common_horizon_complete_cutoff() -> None:
    dataset = extract_forecast_dataset(
        _trajectory_dataset(capsized=True), history_s=5, horizons_s=(6.0,), stride_s=1
    )
    assert np.max(dataset.history_end_s) == 14.0
    assert dataset.targets_rad[-1, 0] == 0.5


def test_biased_alarm_uses_same_tighter_escape_as_scalar_target() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "rahola_lab"
        / "campaigns"
        / "configs"
        / "biased_evaluation.yaml"
    )
    definition = load_campaign_definition(config_path)
    config = definition.simulation.to_dict()
    escape = absolute_roll_escape_angle(config)

    assert escape == config["negative_escape_angle_rad"]
    assert escape < config["escape_angle_rad"]
    score = normalized_alarm_scores(np.array([0.60 * escape]), escape)
    np.testing.assert_allclose(score, [1.0])


def test_forecaster_shapes_and_lstm_budget() -> None:
    rng = np.random.default_rng(8)
    histories = rng.normal(size=(64, 20, 2))
    target = np.max(np.abs(histories[:, :, 0]), axis=1) * 1.2
    for model in (
        EnvelopePersistenceForecaster(),
        LinearQuantileForecaster(iterations=50),
        JaxLSTMQuantileForecaster(hidden_size=8, epochs=1, batch_size=16),
    ):
        model.fit(histories, target)
        prediction = model.predict(histories[:5])
        assert prediction.shape == (5, 3)
        assert np.all(np.diff(prediction, axis=1) >= 0)
    assert JaxLSTMQuantileForecaster().parameter_count() < 100_000
