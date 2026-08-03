"""Run the gate-open B2 Chronos transfer probe."""

from pathlib import Path

from rahola_lab.experiments.b2_chronos import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(result["kill"])
    print(result["d5_within_regime_auc"])
