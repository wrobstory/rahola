# D4b pre-merge adversarial audit

Date: 2026-08-08  
Audited implementation: `bc30ee5` plus the artifact-integrity and spectral-grid tests added by
this audit  
Branch: `story-d4b-critical-wave-groups`

## Merge gate

| Pass | Result | P1 / High findings |
| --- | --- | ---: |
| Adversarial physics audit | PASS | 0 |
| Adversarial numerical audit | PASS WITH DISCLOSED LIMITATIONS | 0 |
| Ponytail full pass | PASS | 0 |

The branch clears the requested no-P1/High gate. Two numerical limitations remain visible and do
not reverse a preregistered decision: the C7 boundary bootstrap is degenerate, and individual rare
capsize classifications are not bitwise stable under step halving. The findings below bound both
limitations.

## Scope and execution boundary

This was a mixed-evidence audit. It inspected equations, source, preregistration history, frozen
artifacts, seed accounting, and upstream provenance; replayed artifact validation; ran independent
metric checks; performed calibration-only group-window checks; and ran bounded step-halving probes.
It did not regenerate or replace either one-shot D4b TEST artifact.

Environment: macOS 26.5.2 arm64; Python 3.13.5; JAX 0.11.0 with 64-bit mode enabled; NumPy 2.5.1;
SciPy 1.18.0; pytest 9.1.1.

## Findings

### N-M1 — The C7 percentile bootstrap cannot express unseen response probability

Severity: Medium. Every observed library height falls below every finite empirical critical height
for its class and entry stratum. Resampling the empirical samples therefore keeps every composed
rate at zero and produces `[0, 0]`. That interval describes the observed empirical distribution; it
is not a calibrated 95% interval for an unobserved tail probability.

The branch handles this correctly. `RESULTS.md` calls the result a “transparent degenerate
bootstrap,” rejects the proposed confidence-interval contribution, and reports C5 as failure of the
predeclared gate and selected group decomposition. It does not claim that `[0, 0]` bounds the true
capsize rate. Changing the estimator after TEST would violate the preregistration, so this audit
leaves the frozen result intact.

### N-M2 — Rare-event classifications show bounded integration-grid sensitivity

Severity: Medium. On a frozen 128-seed subset, halving the RK4 step from 0.1 s to 0.05 s changed one
capsize classification: 3 coarse-grid events versus 2 fine-grid events. The normalized entry-state
error had RMS 0.0389 and maximum 0.1171. This prevents interpreting the exact reported rate as a
grid-independent physical constant.

The sensitivity does not reverse a D4b gate. C5 compares a zero empirical composition against 39
direct events; one boundary classification cannot turn the direct count into zero. C6's combined
AUC is 0.93775, 0.13775 above its 0.80 gate, and its value comes almost entirely from group
parameters. The repository's analytic RK4 fixture also recovers fourth-order convergence, and its
fixed-path stochastic step-halving test applies a 5-percentage-point capsize tolerance. A future
experiment that needs a precise rate should preregister a finer integration grid and a rare-event
classification convergence target.

The wave-side discretization is much tighter. The 8× construction agrees on common samples to
`1.11e-15` for elevation and `8.88e-16` for slope. On one full 7,200 s TRAIN record, halving the
wave grid changed the detected count from 193 to 192; 191 of 193 coarse centers matched within
0.1 s. This 0.5% count shift does not affect a reported gate, but it reinforces that the frozen
library is an operational discretization rather than a unique physical partition.

## Physics equation verification ledger

| Quantity | Intended relation | Code location | Audit result |
| --- | --- | --- | --- |
| JONSWAP realization | `a_j = sqrt(2 S(omega_j) delta_omega)` with `m0 = Hs^2 / 16` | `src/rahola/spectrum.py`; `d4b.py:100-134` | Correct normalization and FFT coefficient scaling; common-grid step halving agrees to roundoff. |
| Deep-water slope | `k = omega^2 / g`; slope is the spatial quadrature of elevation | `d4b.py:127-128` | Correct magnitude, units, and quadrature convention. |
| Nondimensional forcing | `f = r s / phi_e` for `x = phi / phi_e` | `d4b.py:554-578` | Correct escape-angle normalization and nondimensional time step `omega_n dt`. |
| Softening roll model | `x'' = f - 2 zeta x' - q x' abs(x') - (x - x^3)` | `src/rahola/dynamics.py:64-78` | Signs and coefficients match the configured softening family. |
| Escape energy reserve | `1/4 - [v^2/2 + x^2/2 - x^4/4]` | `d4b.py:591-608` | Correct potential and saddle energy for the unbiased, zero-quintic model. |
| Natural entry state | Integrate from rest through an irregular prelude and sample before target forcing | `d4b.py:307-355`, `582-612` | Entry is sampled at the zero-weight start of the blend. Embedded and unconditional entry samples are byte-identical by seed. |
| Group outcome window | First escape must occur during the target group | `d4b.py:772-816` | Calibration replay found 548/548 labeled capsizes inside the retained detected-group interval and 0 in the taper margins. |
| Encounter composition | Sum group occurrence rates times entry-averaged conditional response | `d4b.py:1037-1074` | Implements the preregistered empirical composition. Its boundary failure is disclosed under N-M1. |

The construction is consistent with the critical-wave-group and natural-initial-condition literature:
[Anastopoulos and Spyrou (2019)](https://doi.org/10.1016/j.oceaneng.2019.106213),
[Gong and Pan (2022)](https://arxiv.org/abs/2208.12907), and
[Silva and Maki (2023 preprint)](https://arxiv.org/abs/2301.09834). The spectrum follows the
[original JONSWAP report](https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_3262854).
The audit treats these papers as construction references, not as validation data for Rahola's
reduced roll model.

## Claim and oracle ledger

| Claim | Oracle or evidence | Independence | Result |
| --- | --- | --- | --- |
| Hand-placed group count and parameters | Analytic two-pulse fixture | Independent of stochastic generator | PASS |
| Prelude and plateau preservation | Byte comparisons plus parameter recovery | Direct array oracle | PASS: all 1,200 composites; maximum spectral distortion 0.001605 below 0.05. |
| Natural-entry representativeness | Same-seed unconditional state at 320 s | Paired independent construction path | PASS: descriptive KS statistic 0. |
| Bisection mechanics | Deterministic threshold fixture | Independent closed-form oracle | PASS to `1e-6`; production bracket reaches the 0.01 m target within 12 iterations. |
| Response monotonicity | Direct six-height outcome grid | Raw outcomes before isotonic repair | PASS: zero valid-prelude violations. |
| Direct C5 count | Sum of frozen TEST trial booleans | Independent of composition | 39/1,500; expected-event floor 30 met. |
| C6 AUC and Brier | SciPy Mann-Whitney U and direct squared error from frozen trial rows | Independent implementation | Exact agreement: AUC 0.513092, 0.934013, 0.937750. |
| C6 interval stability | Independent 10,000-replicate whole-prelude bootstrap | Same frozen rows, independent RNG | Stable; 1,000- versus 10,000-draw AUC endpoints differ by at most about 0.0004. |
| Artifact/source consistency | Recursive content, upstream, source, preregistration, and anchor digests | Shared loader, exercised by adversarial mutations | PASS after audit test addition. |

The C6 logistic design is rank deficient because group-shape columns take only six frozen class
patterns: rank 7/13 for group-only and 11/17 for both, including the intercept. The fixed L2 penalty
makes the prediction problem finite. Nonzero singular-value condition numbers are 2.92 and 3.32;
the maximum penalized gradients at the stored fits are `1.55e-5` and `1.09e-4`. Coefficients are
not individually identifiable, but predictions on the same frozen class support are stable. No
coefficient-level physical interpretation is claimed.

## Mutation ledger

Mutations ran in detached worktree `/private/tmp/rahola-d4b-audit.UHKlHn` at `bc30ee5`. The original
five D4b unit tests allowed a producer mutation because they did not load committed artifacts. This
audit added `test_committed_d4b_artifact_graph_is_current`; the improved seven-test file now rejects
all tested mutations before any result can be consumed.

| Mutation | Physical error represented | Result with improved suite |
| --- | --- | --- |
| Remove division by escape angle from wave forcing | Dimensional/nondimensional scale mismatch | KILLED: governing-input digest mismatch |
| Reverse wave-slope quadrature sign | Wrong propagation/slope convention | KILLED: governing-input digest mismatch |
| Replace `x - x^3` with `x + x^3` | Convert softening restoration into hardening | KILLED: source-provenance mismatch |
| Reverse forcing sign | Wrong beam-sea forcing convention | KILLED: governing-input digest mismatch |

## Provenance and preregistration

Commit `bd348b2` preregistered the protocol. Calibration-only amendments `27093d8`, `f20ad85`, and
`b79966a` precede calibration fit `07fa603` and one-shot TEST commits `0961cdc` and `f809f65`.
The committed artifacts recursively verify exact content, upstream artifact digests, source hashes,
campaign anchors, `d4b.py`, and the preregistration file. The audit test loads both terminal graphs:
C7/C5 and C6.

The TEST ranges `[202500, 204000)` and `[204000, 204200)` are disjoint from the prior ledger and
both reserves. The audit changed no result artifact, reserve, frozen prior experiment, paper, or
explainer.

## Ponytail full pass

The implementation adds no dependency, framework, registry, plugin layer, or speculative extension.
It uses existing NumPy, SciPy, JAX, campaign loading, result provenance, isotonic regression, and
RK4 machinery. The standalone producer is long because it records eight sequential experiment
components and their frozen artifacts, but its control flow is direct. Splitting it into a new
architecture would add indirection without reducing the frozen experimental surface.

The pass found no code worth deleting at a cost lower than invalidating the one-shot governing-input
hashes. It therefore made only two validation additions: one terminal artifact-graph test and one
8× spectral step-halving test.

## Commands and results

```text
.venv/bin/ruff check d4b.py packages/rahola-lab/tests/test_d4b.py
All checks passed.

.venv/bin/ruff check .
All checks passed.

.venv/bin/pytest -q packages/rahola-lab/tests/test_d4b.py
7 passed in 1.02s

.venv/bin/pytest -q
244 passed in 53.49s

PYTHONPATH=src:packages/rahola-lab/src .venv/bin/python /private/tmp/d4b_numerical_probe.py
common-grid errors: elevation 1.11e-15, slope 8.88e-16
128-seed step halving: 3 versus 2 capsizes; 1 disagreement

PYTHONPATH=src:packages/rahola-lab/src .venv/bin/python /private/tmp/d4b_group_grid_probe.py
one 7,200 s record: 193 versus 192 groups; 191 centers matched within 0.1 s

PYTHONPATH=src:packages/rahola-lab/src .venv/bin/python /private/tmp/d4b_group_window_probe.py
548 target-window capsizes; 548 retained-group capsizes; 0 before; 0 after

PYTHONPATH=src:packages/rahola-lab/src .venv/bin/python /private/tmp/d4b_statistical_probe.py
stored-fit rank/conditioning checks passed; 10,000-draw clustered AUC intervals stable

PYTHONPATH=src:packages/rahola-lab/src /Users/robstory/src/rahola/.venv/bin/pytest -q \
  packages/rahola-lab/tests/test_d4b.py
each mutation: 1 failed, 5 passed; the artifact-current test killed the mutation
```

The audit skipped a full fine-grid rerun of all one-shot TEST trajectories, a second sea state, and
external experimental validation of the reduced roll model. None is represented as a passing check.
