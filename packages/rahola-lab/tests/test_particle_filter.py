from __future__ import annotations

from dataclasses import replace

import numpy as np
from rahola_lab.inference import bootstrap_particle_filter

from rahola.config import ForcingConfig, ProtocolConfig, ProtocolKind, SeaState, SimulationConfig
from rahola.simulate import simulate_batch


def test_particle_filter_recovers_known_ramped_stiffness_fixture() -> None:
    """Predeclared bounds: stiffness MAE 0.10; drift error 4e-4/s."""
    config = SimulationConfig(
        duration_s=480.0,
        natural_period_s=4.0,
        output_rate_hz=2.0,
        initial_angle_rad=0.08,
        damping_ratio=0.0,
        quadratic_damping=0.0,
        forcing=ForcingConfig(sea_state=SeaState(hs_m=1.0, tp_s=5.0), effective_wave_slope=0.0),
        protocol=ProtocolConfig(
            kind=ProtocolKind.RAMPED,
            ramp_parameter="stiffness",
            ramp_start=1.2,
            ramp_end=0.72,
        ),
    )
    observed = simulate_batch(config, [91])
    posterior = bootstrap_particle_filter(
        observed.angle_rad[0], observed.rate_rad_s[0], 0.5, config, seed=91
    )
    assert abs(np.mean(posterior.stiffness_multiplier) - 0.72) <= 0.10
    assert abs(np.mean(posterior.stiffness_rate_per_s) - (-0.001)) <= 0.0004


def test_particle_filter_validates_input_shape() -> None:
    config = replace(SimulationConfig(), output_rate_hz=2.0)
    with np.testing.assert_raises(ValueError):
        bootstrap_particle_filter(np.zeros(4), np.zeros(4), 0.5, config)
