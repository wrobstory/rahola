"""Render the exact-tuning cut through the principal Mathieu tongue."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola.validation import damped_mathieu_threshold, mathieu_growth_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("mathieu-tongue.png"))
    args = parser.parse_args()
    damping = 0.02
    amplitudes = np.linspace(0.04, 0.12, 33)
    rates = [mathieu_growth_rate(value, damping) for value in amplitudes]
    plt.plot(amplitudes, rates, label="RK4 Floquet-envelope estimate")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(
        damped_mathieu_threshold(damping),
        color="tab:red",
        linestyle="--",
        label=r"$h_c=4\zeta$",
    )
    plt.xlabel("stiffness modulation amplitude $h_0$")
    plt.ylabel("growth exponent per $\\tau$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
