# Rahola

Rahola is a controlled study of a hard warning problem: can recent roll motion reveal an
approaching capsize soon enough to act, without producing an unusable number of false alarms? It
builds the problem from seeded nonlinear trajectories, then tests forecasts, statistical alarms,
deep sequence models, physical-state estimators, and transfer methods under one causal evaluation
protocol.

> Rahola is research software, not a vessel-operational safety system.

## The problem

A warning method can fail in several distinct ways. It may rank dangerous windows above ordinary
ones yet offer no useful operating threshold. A threshold may work on familiar dynamics and fail
under a different restoring law or sea spectrum. A model may appear predictive because it learned
protocol time, saw a truncated outcome window, or used future data during normalization. Rare
capsizes make each of these errors easy to hide behind a single aggregate score.

Rahola separates those questions. It first validates a small roll model against analytic limits.
It then generates fixed train, calibration, test, and guarded-reserve campaigns. Every warning
method receives the same roll and roll-rate history, the same causal normalization, the same
capsize horizon, and the same episode accounting. Thresholds are selected on calibration data and
evaluated once on test data.

The experiments support a restrained conclusion. Motion history often ranks risk, but the tested
alarm thresholds do not transfer reliably across failure mechanisms. In the tested softening-step
regime, motion-only scores approach chance discrimination after the trajectory enters the harsh
state. Protocol time can also rival the learned motion score. These results point toward exogenous
encounter sensing and action-aware policies, not another unconstrained search over motion-only
architectures.

## Start with one simulated batch

Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```sh
uv sync --all-packages --all-extras
uv run rahola validate
```

The smallest useful experiment generates 128 independently forced trajectories:

```python
from rahola import SimulationConfig, simulate_batch

dataset = simulate_batch(SimulationConfig(), seeds=range(128))
```

The result contains a common time grid, roll angle, roll rate, seed, capsize indicator, capsize
time, configuration, and per-trajectory metadata. Samples after capsize are `NaN` and cannot enter
a training or evaluation window.

## 1. Base roll dynamics

The simplest case is a damped roll oscillator with a softening restoring curve and an external
moment. All three Rahola families share the dimensional form

$$
 \ddot\phi+2\zeta\omega_n\dot\phi+\beta\dot\phi|\dot\phi|
 +R(\phi,t)=m(t).
$$

Here $\phi$ is roll angle, $\omega_n$ is the reference natural frequency, $\zeta$ is linear
damping, $\beta$ is quadratic damping, $R$ is the restoring moment per unit inertia, and $m$ is the
wave-induced moment per unit inertia. The public API uses seconds, metres, radians, and
radians/second.

Rahola changes one physical mechanism at a time:

| Family | Restoring / bias | Intended archetype | Principal sweep range |
| --- | --- | --- | --- |
| 1: softening | $`R=\omega_n^2(\phi-\phi^3/\phi_v^2)`$ | dead ship / pure-loss escape | $`\zeta=0.01\ldots0.10`$, $`r=0.01\ldots0.15`$ |
| 2: parametric | $`R=\omega_n^2[1+h(t)](\phi-\phi^3/\phi_v^2)`$ | parametric roll | $`h_0=0\ldots0.4`$, $`\omega_e/\omega_n=1.5\ldots2.5`$ |
| 3: biased | Family 1 plus constant nondimensional moment $`b`$ | damage / steady heel | $`b=-0.3\ldots0.3`$, side-specific escape angles |

Family 1 supplies the minimal escape problem. Family 2 modulates stiffness either deterministically,
$h=h_0\cos(\omega_e t)$, or with an independent narrow-band process. Family 3 adds steady bias and
unequal positive and negative escape angles. The cubic/quintic forms follow the scope of established
1-DOF nonlinear roll comparisons [R4].

For simulation, Rahola scales angle by the escape angle $\phi_v$ and time by $\omega_n$:

$$
  x=\phi/\phi_v,\qquad \tau=\omega_n t,\qquad
  x'=\dot\phi/(\phi_v\omega_n).
$$

The three families then share one nondimensional equation:

$$
 x''+2\zeta x'+q x'|x'|+\kappa(\tau)[1+h(\tau)]
 (x-x^3+\lambda x^5)=f(\tau)+b.
$$

The coefficient `q` is the configured nondimensional quadratic damping, equivalent to
$\beta\phi_v$ under the dimensional convention. `lambda` activates the optional quintic term, and
`kappa` changes only in a stiffness-ramp campaign.

Rahola integrates the smooth random ODE with fixed-step classical RK4. It uses at least 40 steps
per natural period and refines the step when the requested output rate requires an exact
decimation. Each random input is synthesized before integration as a finite trigonometric record;
the model is not an Itô SDE [R8]. Capsize occurs when the state first reaches either escape angle.
The escaped state is absorbing, and later samples are `NaN`.

## 2. Irregular-sea forcing

The one-sided angular-frequency JONSWAP spectrum is

$$
 S_\eta(\omega)=\alpha g^2\omega^{-5}
 \exp[-\tfrac54(\omega_p/\omega)^4]
 \gamma^{\exp[-(\omega-\omega_p)^2/(2\sigma^2\omega_p^2)]},
$$

with $\sigma=0.07$ below the peak and 0.09 above it. Rahola numerically
normalizes $\alpha$ so $m_0=H_s^2/16$, hence
$H_s=4\sqrt{m_0}$. The form and constants come from the JONSWAP field program
[R1].

For FFT-bin width $\Delta\omega$, the deterministic component amplitude is
$A_j=\sqrt{2S_\eta(\omega_j)\Delta\omega}$, while each phase is independently
uniform. Thus

$$
 \eta(t)=\sum_j A_j\cos(\omega_jt+\theta_j).
$$

This deterministic-amplitude/random-phase spectral representation is chosen
because every finite realization has the prescribed bin energy and converges to
the target Gaussian process as the component count grows. Random amplitudes
would represent ensemble scatter more directly but add avoidable finite-record
variance. The ergodicity and FFT tradeoff is treated by Shinozuka & Deodatis
[R2]. Rahola uses at least 200 positive-frequency components.

For a progressive Airy component, the deep-water dispersion relation gives
$k_j=\omega_j^2/g$, and spatial differentiation puts slope in quadrature:
$\alpha_j=\partial\eta_j/\partial x=-k_jA_j\sin(\cdot)$ [R3]. The simplified
roll-excitation chain is

$$
 \eta\longrightarrow \alpha=\partial\eta/\partial x
 \longrightarrow m=\omega_n^2 r\alpha
 \longrightarrow f=m/(\phi_v\omega_n^2)=r\alpha/\phi_v.
$$

The effective wave-slope coefficient $r$ is an input, not a hull-derived
quantity. This follows the established 1-DOF effective-slope abstraction in
Bulian & Francescutto [R5] and the IMO intact-stability explanatory notes [R6].

The complete forcing is synthesized once by inverse FFT on the RK4 half-step
grid; no cosine sum occurs inside the RHS. Step protocols synthesize every
piecewise-stationary sea independently and replace the boundary sample with the
new segment. State remains continuous, but forcing need not be continuous at a
declared environmental step.

## 3. Physics checks before warning models

Rahola tests the generator before it tests any warning method. Three limits expose different
failure modes in the implementation: linear response checks forcing scale, Mathieu instability
checks parametric stiffness, and the Melnikov calculation checks nonlinear escape geometry.

For the linear limit, angular-acceleration input has transfer function

$$
 H(\omega)=\frac{1}{\omega_n^2-\omega^2+i2\zeta\omega_n\omega},\qquad
 \sigma_\phi^2=\int_0^\infty|H(\omega)|^2S_m(\omega)\mathrm{d}\omega,
$$

using the standard random-vibration input/output spectral relation [R7]. At
exact principal parametric tuning, first-order averaging of
$x''+2\zeta x'+[1+h_0\cos(2\tau)]x=0$ gives the small-damping boundary
$h_{0,c}=4\zeta$ [R9].

For Family 1 with harmonic forcing,

$$
 x''+2\zeta x'+x-x^3=F\cos(\Omega\tau),
$$

the unperturbed heteroclinic orbit is
$x_h=\tanh(\tau/\sqrt2)$. Substitution into the Melnikov integral gives the
simple-zero threshold

$$
 F_M(\Omega)=\frac{4\zeta}{3\pi\Omega}
 \sinh\left(\frac{\pi\Omega}{\sqrt2}\right).
$$

Rahola computes both orbit integrals by quadrature and compares this necessary-condition lower
bound with direct phase-ensemble capsize sweeps. The global ship-roll application and interpretation
of the bound follow Falzarano, Shaw & Troesch [R10]. Rahola does not treat the bound as a sufficient
capsize condition.

| Physics component | Falsification test | Acceptance used here |
| --- | --- | --- |
| JONSWAP and FFT realization | `test_jonswap_spectral_fidelity_and_significant_height` | recovered $H_s$ within 2%; log-PSD correlation >0.95; band energy within 12% |
| Linear forcing/response chain | `test_linear_limit_variance_matches_spectral_response` | ensemble variance within 6% |
| Parametric stiffness | `test_mathieu_principal_tongue_boundary` | growth sign brackets $4\zeta$; boundary within 10% |
| Softening separatrix | Melnikov quadrature and capsize-boundary tests | formula within $10^{-5}$; capsize bound above prediction with correlated shape and narrowing low-damping gap |
| RK4 and precomputed forcing | `test_step_halving_convergence_statistics` | variance within 3%; capsize-rate change <=0.05 |
| Seed propagation | spectrum/batch determinism tests | bitwise equality |
| Causal normalization | `test_future_only_leakage_probe_has_teeth` | causal AUC within 0.08 of 0.5; leaky control near perfect |
| Parquet/manifest writer | `test_same_inputs_produce_byte_identical_dataset` | every emitted byte equal |

No acceptance test is skipped by `uv run pytest` or `rahola validate`. Slow
tests are marked only so developers can request `pytest -m 'not slow'` while
iterating; the acceptance command includes them.

## 4. Reference data and causal examples

The validated simulator feeds a fixed synthetic data program. Every reference trajectory lasts
600 seconds, is sampled at 2 Hz, and uses a four-second natural period. The checked campaign
contract contains 58,500 trajectories and 1.629 GiB of Parquet data.

| Campaign layer | Trajectories | Purpose |
| --- | ---: | --- |
| Three stationary families | 12,000 | fit forecasters and detectors under fixed dynamics |
| Three stiffness ramps | 10,500 | test warning as stability erodes |
| One sea-state step | 6,000 | test adaptation after abrupt distribution shift |
| Three rare-event evaluation campaigns | 18,000 | measure false-alarm cost at 0.95–2.00% capsize rates |
| Five forcing-bandwidth campaigns | 12,000 | separate bandwidth from failure severity |

Each stored row represents one trajectory. It contains the seed, capsize outcome and time, common
time vector, roll angle, roll rate, and metadata. A sorted manifest records the full configuration,
ordered seeds, package version, Git commit, shard hashes, and outcome counts. Generation is byte
deterministic for a fixed software environment.

The split design makes model selection explicit. Stationary campaigns use 2,000 train, 1,000
calibration, and 1,000 test trajectories per family. Evaluation campaigns use 1,000 calibration
and 5,000 test trajectories. Step, ramp, and bandwidth campaigns have their own frozen allocations.
Disjoint seed blocks protect train, calibration, test, reserve, and reserve-2 data; public utilities
refuse both reserve blocks.

### From trajectories to labels

Rahola builds examples only from information available at the scoring time. `CausalTransformer`
standardizes and optionally detrends each sample from statistics fitted strictly before that
sample. A future-only leakage probe verifies the full feature path against a deliberately leaky
control.

The forecasting experiments use 120 seconds of history and 30- or 60-second outcome horizons. The
detector experiments use 60 natural periods of roll and roll-rate history, a 50-period capsize
horizon, a five-period exclusion band, and a ten-second score stride. A positive window precedes
capsize within the horizon. A negative window has a complete scored horizon with no capsize within
the horizon plus a five-period exclusion band. Ambiguous and record-end-truncated windows are
discarded for both outcomes.

Operational inference remains separate from supervised evaluation. It emits every causal
pre-capsize score, while exposure, event counts, and labels stop at the last endpoint with a
complete outcome horizon. Three consecutive threshold crossings open an alarm episode at the
confirming window; refractory and decorrelation rules prevent dense scores from becoming a count of
duplicate alarms.

The full campaign table, split counts, capsize rates, seed ranges, and manifest hashes are in
[`DATA.md`](DATA.md).

## 5. Research program: forecasts, warnings, and state

Each stage asks a harder question while retaining the same simulator and split discipline.

### Prototype #1: forecast uncertainty under shift

The first warning layer predicts future maximum absolute roll from causal motion history. It
compares an envelope extrapolator, linear quantile regression, and a 4.6k-parameter JAX LSTM. Split
conformalized quantile regression calibrates their upper bounds; a zero-training split-time danger
margin supplies a physical baseline.

Experiments E1–E4 move from stationary marginal coverage to operational alarms, abrupt sea-state
shift, delayed-feedback adaptive conformal methods, and cross-sea-state stress tests. This sequence
distinguishes a calibrated forecast interval from a useful alarm policy: nominal marginal coverage
can coexist with poor rolling coverage or excessive alarm episodes after a shift.

### Prototype #2: direct warning from motion history

The second layer places five scores behind one threshold-selection and episode harness:

- a 2,969-parameter temporal CNN;
- classical variance and autocorrelation trends;
- a roll-power adaptation of Galeazzi's GLRT;
- the split-time danger margin;
- Story's phase-space neighbor-loss score.

D1 measures pooled in-distribution performance. D2 holds out each failure family in turn. D3 varies
forcing bandwidth while retuning severity to a common capsize band. D4 asks whether alarms coincide
with evaluator-only critical wave groups, and D5 asks whether a method can still rank risk after
entry into an established harsh regime. No detector receives wave elevation, spectrum, family, or
sea-state input.

### Prototype #3: state inference and transfer

The final layer asks what information or architecture might close the remaining gap. C1 restarts
independent futures from the exact simulated state. C2 first infers stiffness and drift with a
2,000-particle filter. An engineered-feature XGBoost model tests whether the CNN simply missed an
easy representation, and a clock-only comparator measures protocol-time confounding.

Two architecture probes follow: a 4,329-parameter gray-box network with a physical latent head, and
a pinned Chronos-T5-tiny transfer experiment with frozen and one-epoch fine-tuned encoders. These
are bounded falsification tests, not an open-ended model search. C1 and C2 also replace the future
forcing realization, so they are restart comparators rather than Bayes-optimal information ceilings.

## 6. What the experiments established

The corrected development record supports the following claims:

| Question | Result | Interpretation |
| --- | --- | --- |
| Does stationary conformal calibration work? | Mean absolute coverage error across E1 was 0.75 percentage points. | The basic forecast and calibration implementation behaves as intended. |
| Does adaptation repair an abrupt sea-state shift? | ACI, DtACI, and recent-score recalibration missed the joint rolling-coverage and alarm-cost criteria. | Marginal conformal machinery did not yield a satisfactory online alarm under this shift. |
| Which detector has the best pooled operating point? | The CNN reached 92.36% sensitivity at 15.548 false episodes/h; classical EWS reached 100% (near-always-on) at 21.391/h. | The CNN lowers pooled alarm cost, with lower sensitivity. |
| Does that threshold transfer across failure families? | The CNN missed 90% test sensitivity in all three D2 rotations. | No all-family operating point was established. |
| Does bandwidth destroy ranking skill? | CNN AUC remained 0.862–0.920, but its broadband FPR improvement was only 8.6%. | D3 is inconclusive under its predeclared 10% materiality rule. |
| Are false alarms simply critical-wave encounters? | 75–88% overlapped evaluator-defined groups. | The overlap is descriptive because groups are common and no matched null was tested. |
| Can motion rank risk inside the harsh regime? | D5 AUCs ranged from 0.474 to 0.509. | The tested motion-only scores were near chance after regime entry. |
| Do more complex state and transfer models resolve the problem? | XGBoost beat the CNN in the restart sample; the clock score also beat the CNN. B1 failed transfer, and B2 produced one development survivor. | Architecture comparisons remain confounded by protocol time and lack a valid new final holdout. |

The historical reserve and reserve-2 artifacts remain immutable. An August 2026 audit found that
their old procedures differ from the corrected calibration-only threshold protocol; reserve-2 also
selected the Chronos threshold using reserve outcomes. They remain audit records, not prospective
validation of the corrected methods.

[`RESULTS.md`](RESULTS.md) gives every interval, lead-time distribution, kill criterion, and
methodological qualification. [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) traces the full project and
the audit corrections.

## 7. Reproducibility and performance

The repository separates the validated simulator (`rahola`) from the experimental layer
(`rahola-lab`). From the repository root:

| Path | Role |
| --- | --- |
| `src/rahola/` | dynamics, spectra, simulation, causal windows, storage, and analytic validation |
| `packages/rahola-lab/` | campaigns, forecasters, detectors, evaluation, and experiment runners |
| `configs/` | small demonstration campaigns |
| `examples/` | reproducible validation and research entry points |
| `results/` | checked numeric artifacts and figures |

```sh
uv sync --all-packages --all-extras
uv run pytest
uv run rahola validate
uv run rahola-lab generate --all --out data/reference --chunk-size 256
uv run python examples/e1_coverage.py
uv run python examples/d1_detectors.py
uv run python examples/p3_ceiling.py
```

Generated trajectories stay outside Git, while frozen campaign definitions, manifest anchors,
numeric result JSON, and figures are tracked. Each development artifact records the source-tree and
reference-data fingerprints used to produce it, binds its serialized content, and records exact
upstream artifact digests. Loaders reject stale or mutated dependencies.

### Simulator throughput

Measured on this arm64 Apple-silicon host with JAX 0.11.0, CPU backend, after a
one-trajectory warm-up:

```text
uv run python examples/benchmark.py --trajectories 256 --duration-s 3600
trajectories=256 elapsed_s=1.191 simulated_hours_per_wall_minute=12899.9
```

This is an end-to-end measurement including the 256 independently seeded FFT
records and array materialization. It exceeds the 500 simulated-hours/minute
guideline by 25x. It is not an extrapolation to 10,000 trajectories; memory and
thermal behavior at that campaign size should be measured on the execution host.
JAX owns the backend choice, so selecting a GPU does not change the model code.

## 8. Explicit scope decisions

- Deterministic amplitudes plus random phases were selected for exact discrete
  spectral energy and lower finite-record variance; phases remain the stochastic
  degrees of freedom.
- The requested output rate may force an integration step smaller than
  $T_n/40$; output is never interpolated upward from a coarser state grid.
- Each step-sea segment is independently synthesized, so forcing may jump while
  roll angle and rate remain continuous.
- A ramp in `stiffness` is a nondimensional multiplier $\kappa$; the reference
  time scale $\omega_n$ is kept fixed during that trajectory.
- `simulate_restarted_batch` starts independent futures from per-trajectory roll,
  rate, stiffness, drift, and deterministic-parametric phase offsets; it is the
  validated core extension used by the Prototype #3 restart comparators. The
  synthesized forcing is temporally correlated, so motion history carries
  encounter-preview information that independent-future restarts discard. The
  restart scores are therefore not Bayes-optimal motion-history ceilings.
- Stochastic parametric modulation uses an independently phased JONSWAP
  elevation record, RMS-normalized to `stochastic_std`.
- Capsize time is zero for an initially escaped state; otherwise it is the first RK4 endpoint beyond
  the threshold, without sub-step root interpolation.
- The Melnikov comparison defines the simulated boundary at 50% capsize over
  uniformly spaced forcing phases and 120 natural periods. Melnikov remains only
  a necessary heteroclinic-intersection bound.
- Uncompressed Parquet was favored over smaller files so identical inputs remain
  byte-identical across repeat writes with a fixed PyArrow version.

## 9. Known limitations

- One roll DOF only: no heave, sway, yaw, pitch, or hull-geometry input.
- Cubic/quintic polynomial restoring is an archetype, not a vessel-specific GZ.
- No convolution-memory or frequency-dependent radiation damping.
- Deep-water, long-crested, unidirectional Airy slope; no directional spreading,
  finite-depth dispersion, encounter-frequency transform, diffraction, or wave
  nonlinearity.
- Constant effective wave-slope coefficient rather than a frequency-dependent
  hydrodynamic admittance.
- Piecewise-stationary step forcing is intentionally discontinuous.
- Fixed-step event timing has one-step resolution.
- JONSWAP stationarity and phase randomization do not model coherent wave groups
  beyond those produced by the spectrum.
- Research validation only; no operational decision or safety claim.

## References

- **R1.** Hasselmann et al. (1973), *Measurements of Wind-Wave Growth and
  Swell Decay during the Joint North Sea Wave Project (JONSWAP)*,
  [Max Planck repository](https://pure.mpg.de/view/item_3262854_4).
- **R2.** Shinozuka & Deodatis (1991), “Simulation of Stochastic Processes by
  Spectral Representation,” *Applied Mechanics Reviews* 44(4),
  [doi:10.1115/1.3119501](https://doi.org/10.1115/1.3119501).
- **R3.** Dean & Dalrymple (1991), *Water Wave Mechanics for Engineers and
  Scientists*, World Scientific, ISBN 9789810204204.
- **R4.** Bulian & Francescutto (2011), “Effect of roll modelling in beam waves
  under multi-frequency excitation,” *Ocean Engineering* 38,
  [doi:10.1016/j.oceaneng.2011.07.004](https://doi.org/10.1016/j.oceaneng.2011.07.004).
- **R5.** Bulian & Francescutto (2004), “A simplified modular approach for the
  prediction of roll motion due to wind and waves,”
  [doi:10.1243/1475090041737958](https://doi.org/10.1243/1475090041737958).
- **R6.** IMO (2008), *Explanatory Notes to the International Code on Intact
  Stability*, MSC.1/Circ.1281,
  [official circular copy](https://wwwcdn.imo.org/localresources/en/OurWork/Safety/Documents/MSC.1-CIRC.1281.pdf).
- **R7.** Bendat & Piersol (2010), *Random Data: Analysis and Measurement
  Procedures*, 4th ed., [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
- **R8.** Hairer, Nørsett & Wanner (1993), *Solving Ordinary Differential
  Equations I*, 2nd ed., [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1).
- **R9.** Nayfeh & Mook (1979), *Nonlinear Oscillations*, Wiley,
  [doi:10.1002/9783527617586](https://doi.org/10.1002/9783527617586).
- **R10.** Falzarano, Shaw & Troesch (1992), “Application of Global Methods for
  Analyzing Dynamical Systems to Ship Rolling Motion and Capsizing,”
  [doi:10.1142/S0218127492000100](https://doi.org/10.1142/S0218127492000100).
