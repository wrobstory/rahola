# Rahola reference-data contract

This file freezes the data choices shared by Rahola prototypes. Generated data live under
`data/reference/` and are intentionally ignored by Git; the manifests make regeneration auditable.

## Frozen constants

| Choice | Frozen value | Rationale |
|---|---:|---|
| Forecast horizons | 30 s, 60 s | Operationally actionable and several roll cycles long |
| Forecast history | 120 s | Captures slow envelope modulation without future motion |
| Alarm angle | 0.60 × escape angle | Leaves escape margin while rejecting ordinary moderate roll |
| Prototype #2 EWS window | 60 roll periods | Long baseline for slow statistics |
| Prototype #2 EWS horizon | 50 roll periods | Planned bifurcation-warning horizon |
| Near-miss exclusion | 5 roll periods | Removes ambiguous negatives near an event |
| ACI γ grid | 0.001, 0.005, 0.01, 0.02, 0.05 | Slow through aggressive adaptation, declared before E3 |
| ACI episode explosion | >2 episodes/h **and** >4× fixed CQR | Absolute and relative operational guardrails |

Every campaign is 600 seconds at 2 Hz with a 4-second natural period. This accommodates the
110-period EWS history-plus-horizon and its transient. The biased-family campaigns start at the
stable static equilibrium; otherwise the imposed bias creates an artificial zero-state transient.

## Frozen campaigns

Rates are measured across every declared split, not pilot intentions. SHA values are the first 12
hex digits of the top-level manifest's SHA-256.

| Campaign | Role | Trajectories | Duration | Capsize fraction | Manifest SHA |
|---|---|---:|---:|---:|---|
| `softening_stationary` | stationary training | 4,000 | 600 s | 8.650% | `bc0dadbb6685` |
| `parametric_stationary` | stationary training | 4,000 | 600 s | 8.450% | `ab3291f10d92` |
| `biased_stationary` | stationary training | 4,000 | 600 s | 13.300% | `4fafc6ef37dc` |
| `softening_ramp` | Prototype #2 ramp | 2,000 | 600 s | 47.700% | `f9d21770b828` |
| `parametric_ramp` | Prototype #2 ramp | 2,000 | 600 s | 28.150% | `c742dd3c10b5` |
| `biased_ramp` | Prototype #2 ramp | 2,000 | 600 s | 24.200% | `e6e1c1df5480` |
| `softening_step` | sea-state transition | 6,000 | 600 s | 38.417% | `5e8a75387e43` |
| `softening_evaluation` | rare-event evaluation | 6,000 | 600 s | 2.000% | `2696d9359662` |
| `parametric_evaluation` | rare-event evaluation | 6,000 | 600 s | 0.950% | `2037e75e35a1` |
| `biased_evaluation` | rare-event evaluation | 6,000 | 600 s | 1.867% | `4178b2ed6fd7` |

Total: 42,000 trajectories, 1.170 GiB. The equivalent final full generation took 58.846 seconds,
versus the limits of 20 GB and one hour.

Split-specific counts and rates remain in each manifest. Stationary campaigns contain
2,000 train / 1,000 calibration / 1,000 test trajectories. Evaluation campaigns contain
1,000 calibration / 5,000 test trajectories. The step campaign contains 2,000 train / 1,000
calibration / 3,000 test trajectories. Ramps contain 2,000 train trajectories.

## Seed allocation

| Block | Half-open range | Permitted use here |
|---|---|---|
| train | `[0, 100000)` | Campaign pilots, model fitting, fixed model grid |
| calibration | `[100000, 200000)` | CQR scores and ACI γ selection |
| test | `[200000, 300000)` | Final experiment scoring |
| reserve | `[300000, 400000)` | **Untouched; reserved for Prototype #2** |

Campaign offsets in YAML allocate disjoint subranges within each block. Public split utilities take
a block name and reject `reserve`; they do not accept arbitrary seed vectors.

## Exact regeneration

From the repository root:

```bash
uv sync --all-packages
uv run rahola-lab generate --all --out data/reference --chunk-size 256
```

The generator writes deterministic Parquet shards, chunk manifests and the top-level manifests used
above. On this machine the final per-campaign timings sum to 58.846 seconds; filesystem and first-run
JAX compilation differences may change wall time without changing content.
