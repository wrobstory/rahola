from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rahola.config import (
    Family,
    ForcingConfig,
    ParametricConfig,
    ProtocolConfig,
    ProtocolKind,
    SeaState,
    SeaStateStep,
    SimulationConfig,
)
from rahola.simulate import (
    _forcing_for_seed,
    simulate_batch,
    simulate_restarted_batch,
    simulate_tangent_batch,
)


def _small_config(**changes: object) -> SimulationConfig:
    base = SimulationConfig(
        duration_s=32.0,
        natural_period_s=4.0,
        output_rate_hz=2.0,
        forcing=ForcingConfig(sea_state=SeaState(hs_m=1.5, tp_s=5.0), effective_wave_slope=0.03),
    )
    return replace(base, **changes)


def test_batch_determinism_and_seed_separation() -> None:
    config = _small_config()
    first = simulate_batch(config, [10, 11])
    second = simulate_batch(config, [10, 11])
    assert np.array_equal(first.angle_rad, second.angle_rad, equal_nan=True)
    assert not np.array_equal(first.angle_rad[0], first.angle_rad[1])


def _full_record_exponent(config: SimulationConfig) -> float:
    rollout = simulate_tangent_batch(config, [10])
    transition = np.eye(2)
    for local in rollout.transition_matrices[0]:
        transition = local @ transition
    duration_tau = config.omega_n_rad_s * config.duration_s
    return float(np.log(np.linalg.svd(transition, compute_uv=False)[0]) / duration_tau)


def test_tangent_linear_oscillator_exponent_converges_to_negative_damping() -> None:
    config = _small_config(
        duration_s=400.0,
        damping_ratio=0.05,
        quadratic_damping=0.0,
        linear_restoring=True,
        forcing=ForcingConfig(effective_wave_slope=0.0),
    )
    assert _full_record_exponent(config) == pytest.approx(-config.damping_ratio, abs=2e-3)


@pytest.mark.parametrize("h0, expected_sign", [(0.08, -1), (0.24, 1)])
def test_tangent_mathieu_exponent_sign_matches_four_zeta_boundary(
    h0: float, expected_sign: int
) -> None:
    config = _small_config(
        family=Family.PARAMETRIC,
        duration_s=400.0,
        damping_ratio=0.04,
        quadratic_damping=0.0,
        linear_restoring=True,
        forcing=ForcingConfig(effective_wave_slope=0.0),
        parametric=ParametricConfig(h0=h0, excitation_ratio=2.0),
    )
    assert np.sign(_full_record_exponent(config)) == expected_sign


def test_tangent_rollout_reproduces_base_trajectory_bitwise() -> None:
    config = _small_config()
    base = simulate_batch(config, [10, 11])
    tangent = simulate_tangent_batch(config, [10, 11]).dataset
    assert np.array_equal(base.angle_rad, tangent.angle_rad, equal_nan=True)
    assert np.array_equal(base.rate_rad_s, tangent.rate_rad_s, equal_nan=True)
    assert np.array_equal(base.t_capsize_s, tangent.t_capsize_s, equal_nan=True)


def test_output_grid_honors_requested_rate_and_duration() -> None:
    dataset = simulate_batch(_small_config(duration_s=2.0, output_rate_hz=3.0), [10])
    assert dataset.time_s[-1] == pytest.approx(2.0)
    assert np.diff(dataset.time_s) == pytest.approx(np.full(6, 1.0 / 3.0))


def test_step_protocol_and_asymmetric_capsize() -> None:
    protocol = ProtocolConfig(
        kind=ProtocolKind.STEP,
        steps=(SeaStateStep(16.0, SeaState(hs_m=3.0, tp_s=4.0)),),
    )
    config = _small_config(
        family=Family.BIASED,
        bias_moment=-0.5,
        negative_escape_angle_rad=0.2,
        protocol=protocol,
    )
    dataset = simulate_batch(config, [3])
    assert dataset.metadata[0]["protocol"] == "step"
    assert dataset.capsized[0]
    cap_index = np.searchsorted(dataset.time_s, dataset.t_capsize_s[0], side="right")
    assert np.all(np.isnan(dataset.angle_rad[0, cap_index:]))


def test_restart_accepts_per_trajectory_state_and_stiffness_drift() -> None:
    config = _small_config()
    angles = np.array([0.01, -0.02])
    rates = np.array([0.003, -0.004])
    restarted = simulate_restarted_batch(
        config,
        [20, 21],
        duration_s=16.0,
        initial_angle_rad=angles,
        initial_rate_rad_s=rates,
        stiffness_multiplier=np.array([0.9, 0.8]),
        stiffness_rate_per_s=np.array([-0.001, -0.002]),
    )
    assert restarted.angle_rad[:, 0] == pytest.approx(angles)
    assert restarted.rate_rad_s[:, 0] == pytest.approx(rates)
    assert restarted.metadata[1]["restart"]["stiffness_multiplier"] == pytest.approx(0.8)
    assert restarted.metadata[1]["restart"]["stiffness_rate_per_s"] == pytest.approx(-0.002)


def test_restart_rejects_nonstationary_protocol() -> None:
    config = _small_config(
        protocol=ProtocolConfig(
            kind=ProtocolKind.STEP,
            steps=(SeaStateStep(16.0, SeaState(hs_m=3.0, tp_s=4.0)),),
        )
    )
    with pytest.raises(ValueError, match="stationary base protocol"):
        simulate_restarted_batch(
            config,
            [20],
            duration_s=16.0,
            initial_angle_rad=0.0,
            initial_rate_rad_s=0.0,
        )


def test_restart_detects_initial_escape_boundary_at_time_zero() -> None:
    config = _small_config(
        duration_s=2.0,
        escape_angle_rad=0.5,
        forcing=ForcingConfig(effective_wave_slope=0.0),
    )
    restarted = simulate_restarted_batch(
        config,
        [20],
        duration_s=2.0,
        initial_angle_rad=0.5,
        initial_rate_rad_s=-1.0,
    )
    assert restarted.capsized[0]
    assert restarted.t_capsize_s[0] == 0.0


@pytest.mark.parametrize("seeds", [[1.2, 2.8], [True, False], [-1, 2]])
def test_simulator_rejects_non_uint64_seed_values(seeds) -> None:
    with pytest.raises(ValueError, match="seeds"):
        simulate_batch(_small_config(duration_s=2.0), seeds)


@pytest.mark.slow
def test_restart_ensemble_matches_full_run_segment_statistics() -> None:
    """Predeclared check: variance within 15%, capsize fraction within 5 points."""
    config = _small_config(duration_s=256.0)
    source = simulate_batch(config, range(1_000, 1_128))
    midpoint = int(np.searchsorted(source.time_s, 128.0))
    restarted = simulate_restarted_batch(
        config,
        range(2_000, 2_128),
        duration_s=128.0,
        initial_angle_rad=source.angle_rad[:, midpoint],
        initial_rate_rad_s=source.rate_rad_s[:, midpoint],
    )
    reference = simulate_batch(config, range(3_000, 3_128))
    restart_start = int(np.searchsorted(restarted.time_s, 32.0))
    reference_start = int(np.searchsorted(reference.time_s, 160.0))
    restart_variance = np.nanmean(np.nanvar(restarted.angle_rad[:, restart_start:], axis=1))
    reference_variance = np.nanmean(np.nanvar(reference.angle_rad[:, reference_start:], axis=1))
    assert restart_variance == pytest.approx(reference_variance, rel=0.15)
    assert abs(np.mean(restarted.capsized) - np.mean(reference.capsized)) <= 0.05


@pytest.mark.slow
def test_step_halving_convergence_statistics() -> None:
    coarse = _small_config(duration_s=256.0, integration_steps_per_period=40, output_rate_hz=1)
    fine = replace(coarse, integration_steps_per_period=80)
    seeds = range(32)
    coarse_data = simulate_batch(coarse, seeds)
    fine_data = simulate_batch(fine, seeds)
    coarse_variance = np.nanmean(np.nanvar(coarse_data.angle_rad, axis=1))
    fine_variance = np.nanmean(np.nanvar(fine_data.angle_rad, axis=1))
    assert coarse_variance == pytest.approx(fine_variance, rel=0.03)
    assert abs(np.mean(coarse_data.capsized) - np.mean(fine_data.capsized)) <= 0.05


def test_fixed_cutoff_step_halving_evaluates_same_forcing_path() -> None:
    coarse = _small_config(
        duration_s=32.0,
        integration_steps_per_period=40,
        linear_restoring=True,
    )
    fine = replace(coarse, integration_steps_per_period=80)
    coarse_steps = round(coarse.duration_s / coarse.integration_dt_s)
    fine_steps = round(fine.duration_s / fine.integration_dt_s)
    coarse_slope, _ = _forcing_for_seed(coarse, 91, 2 * coarse_steps, channel=0)
    fine_slope, _ = _forcing_for_seed(fine, 91, 2 * fine_steps, channel=0)
    np.testing.assert_allclose(coarse_slope, fine_slope[::2], rtol=0.0, atol=1e-12)

    coarse_data = simulate_batch(coarse, [91])
    fine_data = simulate_batch(fine, [91])
    np.testing.assert_allclose(coarse_data.time_s, fine_data.time_s)
    rms = np.sqrt(np.mean(fine_data.angle_rad[0] ** 2))
    error = np.sqrt(np.mean((coarse_data.angle_rad[0] - fine_data.angle_rad[0]) ** 2))
    assert error / rms < 0.10
