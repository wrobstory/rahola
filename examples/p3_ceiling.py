"""Run or pilot the frozen Prototype #3 ceiling experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rahola_lab.experiments.ceiling import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-windows", type=int)
    args = parser.parse_args()
    pilot = args.pilot_windows is not None
    result = run(
        Path("data/reference"),
        Path("results"),
        windows_per_campaign=args.pilot_windows or 2_000,
        write=not pilot,
    )
    print(f"elapsed: {result['elapsed_s']:.1f}s")
    for name, metrics in result["methods"].items():
        print(f"{name}: {metrics['auc']:.4f} {metrics['interval']}")
    print(result["gate_verdict"])


if __name__ == "__main__":
    main()
