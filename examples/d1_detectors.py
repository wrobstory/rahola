from __future__ import annotations

from pathlib import Path

from rahola_lab.experiments.d1 import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(result["headline_at_90_percent_sensitivity"])
