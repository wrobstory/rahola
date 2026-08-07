from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from rahola_lab.campaigns.f1 import F1_TEST_SLICES, verify_f1_test_slices
from rahola_lab.campaigns.load import _REFERENCE_CHECKSUMS
from rahola_lab.experiments.f1_common import (
    StatisticRows,
    directional_growth,
    finite_time_score,
    fit_logistic,
)

from rahola.config import ForcingConfig, SimulationConfig
from rahola.dataset import SimulationDataset, TangentRollout
from rahola.dynamics import local_tangent_jacobian


def _transition_fixture(matrices: tuple[np.ndarray, ...]) -> TangentRollout:
    config = SimulationConfig(
        duration_s=6.0,
        natural_period_s=2.0,
        output_rate_hz=1.0,
        forcing=ForcingConfig(effective_wave_slope=0.0),
    )
    time_s = np.arange(7.0)
    dataset = SimulationDataset(
        time_s=time_s,
        angle_rad=np.zeros((1, len(time_s))),
        rate_rad_s=np.zeros((1, len(time_s))),
        seeds=np.array([1], dtype=np.uint64),
        capsized=np.array([False]),
        t_capsize_s=np.array([np.nan]),
        metadata=({"seed": 1},),
        config=config.to_dict(),
    )
    transitions = np.broadcast_to(np.eye(2), (1, 6, 2, 2)).copy()
    transitions[0, : len(matrices)] = matrices
    return TangentRollout(dataset, transitions, np.ones((1, len(time_s))))


def test_f1_seed_slices_are_predeclared_and_disjoint(tmp_path) -> None:
    campaign = tmp_path / "existing"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(
        json.dumps({"splits": {"test": {"offset": 0, "count": 1_000}}}),
        encoding="utf-8",
    )
    verification = verify_f1_test_slices((tmp_path,))
    assert verification["pairwise_disjoint"]
    assert verification["declared_count"] == 9_400
    assert F1_TEST_SLICES["softening_step_v02"] == (3_000, 38_000)
    assert set(f"{name}_f1" for name in F1_TEST_SLICES) <= _REFERENCE_CHECKSUMS.keys()


def test_f1_logistic_fit_is_frozen_and_monotone_on_separable_example() -> None:
    features = np.array([[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    model = fit_logistic(features, labels)
    predictions = model.predict(features)
    assert np.all(np.diff(predictions) > 0.0)
    restored = type(model).from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.predict(features), predictions)


def test_f1_tangent_composes_local_maps_in_chronological_order() -> None:
    first = np.array([[1.0, 2.0], [3.0, 4.0]])
    second = np.array([[5.0, 6.0], [7.0, 9.0]])
    rollout = _transition_fixture((first, second))
    rows = StatisticRows(
        trajectory_indices=np.array([0], dtype=np.int64),
        end_times_s=np.array([0.0]),
        labels=np.array([0], dtype=np.int8),
        features={},
        scores={},
    )
    product = second @ first
    reversed_product = first @ second
    np.testing.assert_array_equal(product, np.array([[23.0, 34.0], [34.0, 50.0]]))
    assert not np.array_equal(product, reversed_product)
    expected = np.log(np.linalg.svd(product, compute_uv=False)[0]) / (2.0 * np.pi)
    actual = finite_time_score(rollout, rows, periods=1, escape_directed=False)[0]
    assert actual == pytest.approx(expected)


def test_f1_directional_growth_uses_initial_escape_normal_as_row_vector() -> None:
    normal = np.array([1.0, 0.0])
    transition = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert directional_growth(normal, transition) == pytest.approx(np.sqrt(5.0))
    assert np.linalg.norm(transition @ normal) == pytest.approx(1.0)


def test_f1_tangent_jacobian_matches_finite_difference_at_nonzero_state() -> None:
    state = np.array([0.31, -0.47])
    damping = 0.07
    quadratic = 0.4
    stiffness = 1.3
    quintic = 0.08

    def rhs(value: np.ndarray) -> np.ndarray:
        x, velocity = value
        shape = x - x**3 + quintic * x**5
        return np.array(
            [
                velocity,
                -2.0 * damping * velocity
                - quadratic * velocity * abs(velocity)
                - stiffness * shape,
            ]
        )

    epsilon = 1e-6
    finite_difference = np.column_stack(
        [
            (
                rhs(state + epsilon * np.eye(2)[column])
                - rhs(state - epsilon * np.eye(2)[column])
            )
            / (2.0 * epsilon)
            for column in range(2)
        ]
    )
    actual = np.asarray(
        local_tangent_jacobian(
            jnp.asarray(state),
            jnp.array(0.0),
            jnp.array(stiffness),
            damping,
            quadratic,
            quintic,
            family_code=0,
            linear_restoring=False,
        )
    )
    np.testing.assert_allclose(actual, finite_difference, rtol=1e-8, atol=1e-9)
