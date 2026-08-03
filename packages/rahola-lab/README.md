# Rahola Lab

`rahola-lab` is the research layer over the validated `rahola` dynamics package. The uv workspace
keeps physics and experimental code separate: laboratory work consumes core datasets but never
changes the simulator's behavior.

## Architecture

- `rahola_lab.campaigns`: typed YAML definitions, named seed-block generation, deterministic
  chunked Parquet manifests, and verified split loading.
- `rahola_lab.evaluation`: protected train/calibration/test/reserve ranges, debounced/refractory
  alarm episodes, exposure-aware event metrics, lead times, and operating curves.
- `rahola_lab.forecast`: causal 120-second history extraction for future maximum absolute roll;
  envelope, linear-quantile, and compact JAX LSTM tiers.
- `rahola_lab.conformal`: auditable NumPy implementations of one-sided split CQR and the exact
  unprojected ACI update, plus the frozen escape-angle alarm normalization.
- `rahola_lab.experiments`: bounded-memory E1–E4 runners used by the root `examples/` scripts.

The forecast target raises any horizon containing capsize to at least the relevant asymmetric escape
angle and drops record-end-truncated horizons. All history features stop at the forecast timestamp.
CQR uses calibration seeds only. ACI consumes targets sequentially after issuing each bound.

## Install, test, and regenerate

From the repository root:

```bash
uv sync --all-packages
uv run pytest
uv run rahola-lab generate --all --out data/reference --chunk-size 256
uv run python examples/e1_coverage.py
uv run python examples/e2_operating_curve.py
uv run python examples/e3_transition.py
uv run python examples/e4_stress_test.py
```

`DATA.md` is the frozen campaign contract and includes manifest hashes. `RESULTS.md` contains the
measured experiment record, kill-criterion verdict, judgment calls, and method citations. Numeric
results and figures are checked in under `results/`; generated trajectories under `data/` are not.

## What Prototype #2 must reuse

Prototype #2 should import—not duplicate—the constants, named split utilities, campaign loader,
episode detector, exposure definition, metrics, and operating-curve generator. Its EWS methods must
use the already-frozen 60-period window, 50-period horizon, five-period exclusion buffer, campaign
manifests, and untouched reserve seed block. The reserve block remains structurally unavailable to
this prototype.

## Model-grid judgment

The LSTM is native JAX because `rahola` already depends on JAX; adding PyTorch would introduce a
second tensor runtime. It has 32 hidden states and about 4.6k parameters, far below the 100k ceiling.
The experiment grid fixes six epochs, 128-sample batches, and 750 linear-model iterations. Forecast
quality is deliberately secondary to testing the conformal layer.
