# Rahola Lab

`rahola-lab` is the research layer over the validated `rahola` dynamics package. The uv workspace
keeps physics and experimental code separate: laboratory work consumes core datasets but never
changes the simulator's behavior.

## Architecture

- `rahola_lab.campaigns`: typed YAML definitions, named seed-block generation, deterministic
  chunked Parquet manifests, and verified split loading.
- `rahola_lab.evaluation`: protected train/calibration/test/reserve/reserve-2 ranges, debounced/refractory
  alarm episodes, exposure-aware event metrics, exact count intervals, lead times, and curves.
- `rahola_lab.forecast`: causal 120-second history extraction for future maximum absolute roll;
  envelope, linear-quantile, compact JAX LSTM, and split-time danger-margin tiers.
- `rahola_lab.conformal`: auditable NumPy implementations of one-sided split CQR and the exact
  unprojected ACI update, deterministic DtACI, recent-score recalibration, and alarm normalization.
- `rahola_lab.detectors`: causal detector-window extraction, classical EWS, roll-power GLRT,
  neighbor loss, native-JAX CNN/gray-box models, XGBoost features, and the pinned Chronos probe.
- `rahola_lab.inference`: the fixed 2,000-particle causal stiffness/drift filter used by C2.
- `rahola_lab.experiments`: bounded-memory E1–E4, E3b, D1–D5, and Prototype #3 restart-comparison runners plus
  the guarded one-time final-reserve-2 path used by root example scripts.

The forecast target raises any complete horizon containing capsize to at least the relevant
asymmetric escape angle. Record-end-truncated horizons are dropped for both outcomes, giving
positive and negative examples common protocol-time support. All history features stop at the
forecast timestamp.
CQR uses calibration seeds only. Online conformal adapters issue each bound immediately but consume
its target only when the forecast horizon has elapsed (six 10-second score steps for the 60-second
target). For the
biased family, scalar maximum-absolute-roll targets use the smaller escape magnitude in both
directions. This is conservative but side-agnostic; signed targets remain future work.

The danger-margin baseline fits the cubic/quintic/bias restoring curve by translating to its stable
equilibrium, matching central slope and each side's first peak, and constraining the repeller line to
the configured vanishing angle. At arbitrary instants it extrapolates the fitted separatrix and uses
the nearer intermediate threshold. Its alarm score is measured outward rate minus critical rate.
The Eq. 15 forced-solution correction is implemented, but E2 uses its zero-forcing form because the
frozen experiments prohibit wave-field inputs.

## Install, test, and regenerate

From the repository root:

```bash
uv sync --all-packages --all-extras
uv run pytest
uv run rahola-lab generate --all --out data/reference --chunk-size 256
uv run python examples/e1_coverage.py
uv run python examples/e2_operating_curve.py
uv run python examples/e3_transition.py
uv run python examples/e3b_adapters.py
uv run python examples/e4_stress_test.py
uv run python examples/d1_detectors.py
uv run python examples/d2_family_generalization.py
uv run python examples/d3_bandwidth.py
uv run python examples/d4_wave_groups.py
uv run python examples/d5_within_regime.py
```

`DATA.md` is the frozen campaign contract and includes manifest hashes. `RESULTS.md` contains the
measured experiment record, kill-criterion verdict, judgment calls, and method citations. Numeric
results and figures are checked in under `results/`; generated trajectories under `data/` are not.
Every development JSON stores source-tree and reference-campaign fingerprints plus a digest of its
own serialized content. Downstream experiments also record exact upstream artifact digests and
reject stale or mutated dependencies.

## Prototype #2 detector layer

All learned and statistical methods share a 60-period, causally normalized roll/roll-rate window,
50-period capsize horizon, five-period near-miss exclusion, ten-second score stride, and one episode
implementation. Normalization at sample `t` is fitted only to samples before `t`. The vectorized
cumulative-sum transformer is regression-tested bitwise against the original per-sample loop, and
the complete detector feature path has a future-only leakage probe with a deliberately leaky
control. Every supervised endpoint requires the full outcome horizon to fit inside the source
record, including endpoints on trajectories that later capsize. A separate
inference-only mode exposes every causal pre-capsize endpoint with label `-1` for
operational scoring and current-state plots. This mode never removes endpoints based on a future
capsize or record-end censoring, so debounce timelines cannot acquire outcome-dependent gaps. An
episode begins at the score window that confirms the required uninterrupted run, not at the first
candidate window in that run.
Evaluation clips every trajectory to the same last horizon-complete warning endpoint. Later causal
scores remain available for live inference but do not enter labels, event counts, or exposure.
Each operating threshold is selected on calibration trajectories and then evaluated once, without
a test sweep. Test sensitivity may therefore differ from the 90% calibration target.

The CNN has two stride-2 temporal convolutions and global average pooling. The frozen grid contains
2,969- and 4,021-parameter variants, both far below 100k. Under the corrected complete-horizon
labels, calibration selected the 2,969-parameter 12/24-channel, kernel-9 model without auxiliary
family loss. A
danger-margin regression head is implemented, but its predeclared weight was zero throughout the
grid and it was not used. Training is deterministic weighted binary cross-entropy plus the selected
auxiliary loss.

B1 takes Kendall's tau of 12 rolling variance or lag-1-autocorrelation checkpoints; calibration
selected lag-1 autocorrelation and a local subwindow equal to 50% of the detector history. B2 adapts Galeazzi's
fixed-shape W2 scale-change GLRT to natural-frequency-band roll. The source statistic is
`roll²×pitch`, but Rahola has no pitch and forbids wave inputs, so this is explicitly a roll-power
adaptation retaining the published four-period detection interval. B3 reuses the unchanged
split-time danger margin and therefore reads the dimensional endpoint of the same causal window.

## Thesis neighbor implementation

Story's thesis pivoted from FTLE periods after observing that the spikes were caused by neighbor
loss, not trajectory stretching (Sec. 3.2.2, pp. 48–52, Figs. 38–41). The binary rule was fewer than
50 neighbors; each low-count sample incremented a cumulative flag. Chapter 3 normalized entire
roll/pitch records and Chapter 3.3 pooled 37 runs into a historical database. Those operations are
not causal within a new trajectory.

Rahola's continuous score is negative neighbor count in the causally normalized roll/rate plane.
Calibration selects radius from `{0.20, 0.35, 0.50}`; one natural period is omitted from the immediate
past so serial neighbors do not satisfy the count by themselves. The sweep always includes score
threshold −50 as the thesis binary point. The common three-window episode policy resolves flag
persistence without reproducing the thesis's cumulative slope heuristic.

The real-time algorithm searched the entire history only upon entering a new roll region, then
reused the stored count (Sec. 4.2.2, pp. 64–66). Rahola instead performs the exact search over the
frozen trailing 60-period history at every score time; this changes the cache and finite history, not
the novelty definition. Chapter 4 used backward-difference velocities but no normalization, and it
reports that the real-time cumulative flag was abandoned after nested loops corrupted recorded data
(pp. 65–67, Fig. 51). Rahola keeps the backward-looking semantics while applying the task's strict
causal normalization requirement.

## Declustering, stratification, and reserve

Calibration estimates each score's decorrelation time from the first 0.05 crossing of its absolute
autocorrelation-peak envelope; alarm episodes separated by no more than that time are merged. The
hand-computed crossing and episode merge both have unit tests. D4 reconstructs the known forcing
only in the evaluator and defines a group as `2×|Hilbert(elevation)| ≥ 0.75 Hs` for at least 1.5 Tp.
Group coincidence uses the constituent active-alarm intervals beginning at debounce confirmation,
not the preceding candidate windows or quiet span bridged by decorrelation merging.
No detector receives elevation, spectrum, or sea-state input.

Public seed utilities raise `ReserveBlockError` for both reserves. `rahola-lab final-eval` can
construct reserve-2 only; the spent reserve is refused internally. The command requires canonical
repository paths and a clean committed tree, validates all ordinary configuration and model
preconditions, verifies the committed survivor and anchored campaign hashes, revalidates the clean
tree and unchanged upstream artifacts immediately before the exclusive claim, and only then
atomically creates and directory-syncs an access-started attestation before constructing the first
seed. The current runner also holds a cross-process result-graph lock from completed-result
publication through recursive verification and atomic terminal attestation, and binds that result
by SHA-256. The two immutable historical attestations predate result-digest binding.
The repository-local guard then refuses every later invocation, including concurrent attempts.
Prototype #2's spent-reserve run completed against `843b24a`; reserve-2 completed once against
`5d4c6be` with 768 trajectories. These are code and audit safeguards, not external access control:
the reserve prohibition remains a procedural research commitment because the simulator accepts
arbitrary public seeds. The reserve-2 Chronos threshold was retrospectively selected on reserve
outcomes in the historical runner, so that immutable result is descriptive rather than a valid
prospective operating-point evaluation.

Prototype #3's C1 and C2 scores restart independent future forcing from exact or filtered endpoint
state. They discard the realized forcing phase encoded in the preceding motion history, so neither
is a Bayes-optimal motion-only ceiling and neither must upper-bound a sequence model. Their former
three-point architecture gate is retained only as historical protocol context; no information-
ceiling verdict is applied to the corrected run.

The restart-equivalence regression currently covers stationary softening only. The ceiling sampler
uses capped-equal allocation across nonempty label × absolute-time-quartile strata; small strata are
fully exhausted rather than oversampled. Its AUC bootstrap conditions on the realized stratified
window sample and rollout draws and does not propagate unequal-probability sampling-design or
additional rollout Monte Carlo uncertainty.

Prototype #3's reproducible development commands are:

```bash
uv run python examples/p3_acausal_neighbor.py
uv run python examples/p3_ceiling.py --pilot-windows 8
uv run python examples/p3_ceiling.py
uv run python examples/p3_b1_graybox.py
uv run python examples/p3_b2_chronos.py
```

The pilot does not write a result and exists only to project the frozen full-run compute budget.

Deliberately deferred work is narrow: wrapping the CNN in the existing conformal layer and building
a sea-state-conditional alarm policy. Neither is needed to answer the Prototype #2 falsification
questions, and neither may use the final reserve again.

## Model-grid judgment

The LSTM is native JAX because `rahola` already depends on JAX; adding PyTorch would introduce a
second tensor runtime. It has 32 hidden states and about 4.6k parameters, far below the 100k ceiling.
The experiment grid fixes six epochs, 128-sample batches, and 750 linear-model iterations. Forecast
quality is deliberately secondary to testing the conformal layer.
