"""Campaign and experiment command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from rahola_lab.campaigns import generate_campaign, load_campaign_definition
from rahola_lab.experiments.final_eval import run_final_evaluation


def campaign_config_dir() -> Path:
    return Path(__file__).parent / "campaigns" / "configs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rahola-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate frozen reference campaigns")
    selection = generate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path)
    selection.add_argument("--all", action="store_true")
    generate.add_argument("--out", type=Path, default=Path("data/reference"))
    generate.add_argument("--chunk-size", type=int, default=256)
    final_eval = subparsers.add_parser(
        "final-eval", help="run the guarded one-time reserve-2 evaluation"
    )
    final_eval.add_argument("--data-root", type=Path, default=Path("data/reference"))
    final_eval.add_argument("--out", type=Path, default=Path("results"))
    final_eval.add_argument("--reserve-root", type=Path, default=Path("data/final-reserve2"))
    final_eval.add_argument("--chunk-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "final-eval":
        run_final_evaluation(
            data_root=args.data_root,
            output_root=args.out,
            config_root=campaign_config_dir(),
            reserve_root=args.reserve_root,
            chunk_size=args.chunk_size,
        )
        return 0
    paths = sorted(campaign_config_dir().glob("*.yaml")) if args.all else [args.config]
    total_seconds = 0.0
    total_bytes = 0
    for path in paths:
        definition = load_campaign_definition(path)
        result = generate_campaign(definition, args.out, chunk_size=args.chunk_size)
        total_seconds += result.elapsed_s
        total_bytes += result.bytes_written
        rates = ", ".join(
            f"{split}={fraction:.4%}" for split, fraction in result.capsize_fractions.items()
        )
        print(
            f"{definition.name}: {result.elapsed_s:.3f}s, "
            f"{result.bytes_written / 1024**2:.2f} MiB, {rates}"
        )
    print(f"total: {total_seconds:.3f}s, {total_bytes / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
