"""Command-line entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rahola.config import SimulationConfig
from rahola.simulate import simulate_batch
from rahola.storage import write_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rahola")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate a seeded dataset")
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--seed-start", type=int, default=0)
    generate.add_argument("--count", type=int, default=100)
    validate = subparsers.add_parser("validate", help="run every physics validation")
    validate.add_argument("--fast", action="store_true", help="exclude slow validations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        config = SimulationConfig.from_yaml(args.config)
        seeds = range(args.seed_start, args.seed_start + args.count)
        dataset = simulate_batch(config, seeds)
        manifest = write_dataset(dataset, args.out)
        print(f"wrote {dataset.batch_size} trajectories to {manifest}")
        return 0
    marker = "not slow" if args.fast else ""
    command = [sys.executable, "-m", "pytest", "-v"]
    if marker:
        command.extend(["-m", marker])
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
