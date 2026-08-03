"""Measure end-to-end generation throughput after one JAX warm-up."""

from __future__ import annotations

import argparse
import time

import jax

from rahola.config import SimulationConfig
from rahola.simulate import simulate_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    args = parser.parse_args()
    simulate_batch(SimulationConfig(duration_s=1.0), [2])
    config = SimulationConfig(duration_s=args.duration_s)
    start = time.perf_counter()
    result = simulate_batch(config, range(args.trajectories))
    jax.block_until_ready(result.angle_rad)
    elapsed = time.perf_counter() - start
    simulated_hours = args.trajectories * args.duration_s / 3600.0
    rate = simulated_hours / elapsed * 60.0
    print(
        f"trajectories={args.trajectories} elapsed_s={elapsed:.3f} "
        f"simulated_hours_per_wall_minute={rate:.1f}"
    )


if __name__ == "__main__":
    main()
