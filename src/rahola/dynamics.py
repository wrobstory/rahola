"""JAX RK4 kernels for the nondimensional roll equations."""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


@partial(jax.jit, static_argnames=("family_code", "linear_restoring"))
def integrate_rk4_batch(
    forcing_half: jax.Array,
    modulation_half: jax.Array,
    stiffness_half: jax.Array,
    dt_tau: float,
    initial_state: jax.Array,
    damping_ratio: float,
    quadratic_damping: float,
    bias: float,
    quintic: float,
    positive_escape: float,
    negative_escape: float,
    *,
    family_code: int,
    linear_restoring: bool,
) -> tuple[jax.Array, jax.Array]:
    """Integrate independent trajectories by vmapping a lax.scan RK4 kernel."""

    def integrate_one(
        forcing: jax.Array,
        modulation: jax.Array,
        stiffness_record: jax.Array,
        state0: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def rhs(
            state: jax.Array, force: jax.Array, h: jax.Array, stiffness: jax.Array
        ) -> jax.Array:
            x, velocity = state[0], state[1]
            shape = x if linear_restoring else x - x**3 + quintic * x**5
            multiplier = 1.0 + h if family_code == 1 else 1.0
            restoring = stiffness * multiplier * shape
            acceleration = (
                force
                + (bias if family_code == 2 else 0.0)
                - 2.0 * damping_ratio * velocity
                - quadratic_damping * velocity * jnp.abs(velocity)
                - restoring
            )
            return jnp.stack((velocity, acceleration))

        n_steps = (forcing.shape[0] - 1) // 2

        def scan_step(
            carry: tuple[jax.Array, jax.Array, jax.Array], index: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            state, active, cap_step = carry
            i0 = 2 * index
            f0, fm, f1 = forcing[i0], forcing[i0 + 1], forcing[i0 + 2]
            h0, hm, h1 = modulation[i0], modulation[i0 + 1], modulation[i0 + 2]
            s0, sm, s1 = (
                stiffness_record[i0],
                stiffness_record[i0 + 1],
                stiffness_record[i0 + 2],
            )
            k1 = rhs(state, f0, h0, s0)
            k2 = rhs(state + 0.5 * dt_tau * k1, fm, hm, sm)
            k3 = rhs(state + 0.5 * dt_tau * k2, fm, hm, sm)
            k4 = rhs(state + dt_tau * k3, f1, h1, s1)
            proposal = state + dt_tau * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            next_state = jnp.where(active, proposal, state)
            crossed = active & (
                (next_state[0] >= positive_escape) | (next_state[0] <= -negative_escape)
            )
            next_cap_step = jnp.where(crossed, (index + 1).astype(jnp.int32), cap_step)
            next_active = active & ~crossed
            return (next_state, next_active, next_cap_step), next_state

        initially_crossed = (state0[0] >= positive_escape) | (
            state0[0] <= -negative_escape
        )
        (_, _, cap_step), history = jax.lax.scan(
            scan_step,
            (
                state0,
                ~initially_crossed,
                jnp.where(initially_crossed, 0, -1).astype(jnp.int32),
            ),
            jnp.arange(n_steps),
        )
        states = jnp.concatenate((state0[None, :], history), axis=0)
        return states, cap_step

    return jax.vmap(integrate_one)(forcing_half, modulation_half, stiffness_half, initial_state)
