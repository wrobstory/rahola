"""Compare direct capsize boundaries with the heteroclinic Melnikov lower bound."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola.validation import (
    find_harmonic_capsize_boundary,
    melnikov_heteroclinic_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("melnikov-comparison.png"))
    args = parser.parse_args()
    damping = 0.015
    frequencies = np.linspace(0.6, 1.4, 13)
    melnikov = [melnikov_heteroclinic_threshold(damping, value) for value in frequencies]
    simulated = [find_harmonic_capsize_boundary(value, damping) for value in frequencies]
    plt.plot(frequencies, melnikov, "--", label="Melnikov necessary-condition bound")
    plt.plot(frequencies, simulated, "o-", label="50% phase-ensemble capsize")
    plt.xlabel(r"forcing ratio $\Omega$")
    plt.ylabel("nondimensional forcing amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
