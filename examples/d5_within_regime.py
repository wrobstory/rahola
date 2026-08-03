from __future__ import annotations

from pathlib import Path

from rahola_lab.experiments.d5 import run

if __name__ == "__main__":
    print(run(Path("data/reference"), Path("results"))["methods"])
