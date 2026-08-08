# Rahola

*On the warning of capsize from measured roll motion: a synthetic-data study*

## Abstract

The study asks whether the recent roll motion of a vessel contains enough information to warn of
an approaching capsize at an acceptable false-alarm cost. A one-degree-of-freedom nonlinear roll
model with three restoring-curve families was validated against analytic limits and used to
generate a 58,500-trajectory reference record under JONSWAP forcing, followed by 13,200 versioned
replacement trajectories after a numerical-resolution audit. Five warning methods were evaluated
under a common protocol in which every operating threshold is selected on calibration data and
evaluated at one frozen point per corrected test run: a small temporal convolutional network,
classical variance and autocorrelation trend statistics, a generalized likelihood-ratio detector,
a closed-form critical-roll-rate margin, and a phase-space neighbor-count score. Four findings are
reported.
(a) Ranking skill is present at every forcing bandwidth tested; the primary network's window AUC
was 0.88 to 0.91. (b) Within an established severe regime, no motion-only method exceeded an
orientation-independent AUC of 0.556. (c) Between 75 and 88 percent of nominally false alarm
episodes overlapped evaluator-defined high-envelope wave groups. (d) Neither CNN normalization
mode retained 90 percent sensitivity across all held-out restoring families; trivial comparators
did so only near the always-on alarm cost. An audit conducted during the study found that the
operating points first reported had been selected on test data; every numerical value in this
report is the corrected value.

## On the name

The repository is named for Jaakko Rahola, whose doctoral thesis (Rahola 1939) derived working
intact-stability criteria from the systematic analysis of recorded capsize casualties, and whose
approach remains the foundation of stability regulation. The present program follows the same
method at a smaller scale: criteria are sought from a recorded population of capsizes, and the
program's own errors are part of the record.

## Administrative information

The repository is a `uv` workspace requiring Python 3.12 or newer. The validated simulator is
separated from the experimental layer as follows.

| Path | Contents |
| --- | --- |
| `src/rahola/` | dynamics, spectra, simulation, causal windows, storage, analytic validation |
| `packages/rahola-lab/` | campaigns, forecasters, detectors, evaluation, experiment runners |
| `configs/` | small demonstration campaigns |
| `examples/` | validation and research entry points |
| `results/` | checked numeric artifacts and figures |

Installation and operating instructions are given in Appendix A. The complete numerical record is
in [`RESULTS.md`](RESULTS.md), the campaign and provenance record in [`DATA.md`](DATA.md), and a
narrative account of the program, including the audit, in
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md). The frozen v0.2 repair record is
[`RESULTS_v02.md`](RESULTS_v02.md); v0.1 artifacts remain immutable historical evidence.

## 1. Introduction

A capsize warning method can fail in several distinct ways. It may rank dangerous intervals above
ordinary ones and still offer no usable operating threshold. A threshold established on one set of
dynamics may fail under a different restoring law or sea spectrum. Apparent skill may arise from
protocol time, from truncated outcome windows, or from the use of future data during
normalization. Because capsizes are rare, each of these errors is easily concealed behind a single
aggregate score. The evaluation procedure of Section 5 treats these failure modes separately.

Signal-based detection with a designed false-alarm probability has been demonstrated for the
parametric-roll problem by Galeazzi et al. (2015). For the general stability-failure
problem, the established machinery is offline and probabilistic; the split-time method of
Belenky et al. (2024) is the principal example, and its critical-roll-rate quantity is
adopted here as a real-time physics baseline. The present study addresses the intermediate
question of onboard warning from motion measurement alone, under evaluation conditions severe
enough to expose the failure modes listed above.

Section 2 describes the numerical model, Section 3 its validation, Section 4 the synthetic data
program, Section 5 the evaluation procedure, and Section 6 the findings. Limitations are collected
in Section 7 and conclusions in Section 8.

## 2. The numerical model

### 2.1 Equation of motion

The model is a damped one-degree-of-freedom roll oscillator with nonlinear restoring and an
external wave moment. All three families share the dimensional form

$$
 \ddot\phi+2\zeta\omega_n\dot\phi+\beta\dot\phi|\dot\phi|
 +R(\phi,t)=m(t),
$$

in which $\phi$ is the roll angle, $\omega_n$ the reference natural frequency, $\zeta$ the linear
damping ratio, $\beta$ the quadratic damping coefficient, $R$ the restoring moment per unit
inertia, and $m$ the wave-induced moment per unit inertia. The public interface uses seconds,
metres, radians, and radians per second. One physical mechanism is varied per family.

| Family | Restoring / bias | Archetype | Principal sweep range |
| --- | --- | --- | --- |
| 1: softening | $`R=\omega_n^2(\phi-\phi^3/\phi_v^2)`$ | dead ship, pure-loss escape | $`\zeta=0.01\ldots0.10`$, $`r=0.01\ldots0.15`$ |
| 2: parametric | $`R=\omega_n^2[1+h(t)](\phi-\phi^3/\phi_v^2)`$ | parametric roll | $`h_0=0\ldots0.4`$, $`\omega_e/\omega_n=1.5\ldots2.5`$ |
| 3: biased | Family 1 plus constant nondimensional moment $`b`$ | damage, steady heel | $`b=-0.3\ldots0.3`$, side-specific escape angles |

Family 1 is the minimal escape problem. Family 2 modulates the stiffness either deterministically,
$h=h_0\cos(\omega_e t)$, or by an independent narrow-band random process. Family 3 adds a steady
bias moment and unequal positive and negative escape angles. The cubic and quintic restoring forms
follow the scope of established one-degree-of-freedom roll comparisons (Bulian and Francescutto
2011). For simulation the angle is scaled by the escape angle $\phi_v$ and time by $\omega_n$,

$$
  x=\phi/\phi_v,\qquad \tau=\omega_n t,\qquad
  x'=\dot\phi/(\phi_v\omega_n),
$$

after which the three families share one nondimensional equation,

$$
 x''+2\zeta x'+q x'|x'|+\kappa(\tau)[1+h(\tau)]
 (x-x^3+\lambda x^5)=f(\tau)+b.
$$

The coefficient $q$ is the configured nondimensional quadratic damping, equivalent to
$\beta\phi_v$ under the dimensional convention; $\lambda$ activates the optional quintic term;
$\kappa$ departs from unity only in stiffness-ramp campaigns.

The smooth random ordinary differential equation is integrated by fixed-step classical
Runge–Kutta (Hairer, Nørsett and Wanner 1993) at no fewer than 40 steps per natural period, with
the step refined further when the requested output rate requires an exact decimation. Every random
input is synthesized as a finite trigonometric record before integration; the model is not treated
as an Itô equation. Capsize is recorded when the state first reaches either escape angle. The
escaped state is absorbing, and all later samples are stored as `NaN` so that no window can
include post-event data.

### 2.2 Irregular-sea forcing

Wave elevation is generated from the one-sided angular-frequency JONSWAP spectrum

$$
 S_\eta(\omega)=\alpha g^2\omega^{-5}
 \exp[-\tfrac54(\omega_p/\omega)^4]
 \gamma^{\exp[-(\omega-\omega_p)^2/(2\sigma^2\omega_p^2)]},
$$

with $\sigma=0.07$ below the spectral peak and $0.09$ above it, in the form and with the constants
of the JONSWAP field program (Hasselmann et al. 1973). The constant $\alpha$ is normalized
numerically so that $m_0=H_s^2/16$ and hence $H_s=4\sqrt{m_0}$.

For an FFT bin of width $\Delta\omega$, the component amplitude is deterministic,
$A_j=\sqrt{2S_\eta(\omega_j)\Delta\omega}$, and each phase is drawn independently from a uniform
distribution, so that

$$
 \eta(t)=\sum_j A_j\cos(\omega_jt+\theta_j).
$$

The deterministic-amplitude representation was selected because every finite realization then
carries the prescribed bin energy; random amplitudes would represent ensemble scatter more
directly at the cost of avoidable finite-record variance. Two consequences should be stated
plainly. Each record spans exactly one period of its discrete Fourier field, and full-record
spectral energy is fixed across seeds by construction. Every reported rate and probability is
therefore conditional on a fixed-energy, random-phase, periodic discrete-spectrum realization
rather than on unconditional draws from the JONSWAP process. A preregistered wave-field audit
(W1) of self-repetition, crossing rates against the Rice formula, and ensemble variability is
the planned check on this construction. The trade is treated by Shinozuka and
Deodatis (1991). No fewer than 200 positive-frequency components are used. For a progressive Airy
component in deep water, $k_j=\omega_j^2/g$, and spatial differentiation places the slope in
quadrature (Dean and Dalrymple 1991). The simplified excitation chain is

$$
 \eta\longrightarrow \alpha=\partial\eta/\partial x
 \longrightarrow m=\omega_n^2 r\alpha
 \longrightarrow f=m/(\phi_v\omega_n^2)=r\alpha/\phi_v,
$$

in which the effective wave-slope coefficient $r$ is an input rather than a hull-derived quantity,
following the established one-degree-of-freedom abstraction (Bulian and Francescutto 2004; IMO
2008). The component set is defined on a fixed spectral grid with an upper cutoff of
$40\omega_n$, independent of the integration step. The complete forcing record is then evaluated
on the Runge–Kutta half-step grid before integration begins. Step protocols synthesize each piecewise-stationary sea
independently; the state remains continuous across a declared environmental step, and the forcing
need not.

### 2.2.1 Equation-level provenance ledger

The following records separate equations verified from this repository from external
equation-level provenance. The checkout contains bibliography entries but no scanned or
version-pinned copies of the cited books and papers; where a page or equation number cannot be
verified locally, it is explicitly marked unavailable.

| Component | Repository-local equation and transformation chain | Primary-source status |
| --- | --- | --- |
| JONSWAP | `src/rahola/spectrum.py` evaluates $S_\eta=\alpha g^2\omega^{-5}\exp[-1.25(\omega_p/\omega)^4]\gamma^{\exp[-(\omega-\omega_p)^2/(2\sigma^2\omega_p^2)]}$, then normalizes $\int S_\eta\,d\omega$ to $H_s^2/16$, sets $A_j=\sqrt{2S_j\Delta\omega}$, and maps amplitudes/phases through the inverse FFT to $\eta(t)$. | Hasselmann et al. (1973) is a bibliography pointer only in this checkout; source version, page, and equation number are unavailable locally. |
| Deep-water slope | From the linear deep-water dispersion relation $\omega^2=gk$, $k=\omega^2/g$. For $\eta_j=A_j\cos(kx-\omega t+\theta_j)$, evaluation at the fixed location $x=0$ gives $\partial_x\eta_j=-A_jk\sin(-\omega t+\theta_j)$, or $C_{\alpha,j}=-ik_jC_{\eta,j}$. The code then applies $m=\omega_n^2r\alpha$ and $f=m/(\phi_v\omega_n^2)=r\alpha/\phi_v$. | Dean and Dalrymple (1991) is cited, but edition/page/equation evidence is unavailable locally. The effective $r$ is a configured abstraction, not a sourced hull transfer function. |
| Mathieu | The normalized local model is $x''+2\zeta x'+[1+h_0\cos(2\tau)]x=0$ and the retained first-order exact-tuning boundary is $h_{0,c}=4\zeta$; `mathieu_growth_rate` integrates the same half-step forcing and estimates the envelope exponent. | Nayfeh and Mook (1979) is a bibliography pointer only; source version, page, and equation number are unavailable locally. |
| Melnikov | For $x''+2\zeta x'+x-x^3=F\cos(\Omega\tau)$ and $x_h=\tanh(\tau/\sqrt2)$, the local derivation uses $\int x_h'^2d\tau=2\sqrt2/3$ and $\left|\int x_h'\cos(\Omega\tau)d\tau\right|=\pi\sqrt2\Omega/\sinh(\pi\Omega/\sqrt2)$, yielding $F_M=4\zeta\sinh(\pi\Omega/\sqrt2)/(3\pi\Omega)$. The code compares this closed form with quadrature and uses it only as a necessary lower bound for a phase sweep. | Falzarano, Shaw, and Troesch (1992) is cited for application context; edition/page/equation evidence is unavailable locally. The displayed formula is independently derived and tested in `tests/test_validation.py`. |

This ledger is implementation provenance, not a claim that the reduced-order model or its
effective wave-slope coefficient is externally validated.

### 2.3 Modeling decisions

The following decisions bound the scope of the model. (a) Phases are the only stochastic degrees
of freedom; see Section 2.2. (b) The requested output rate may force an integration step smaller
than $T_n/40$; output is never interpolated from a coarser state grid. (c) Step-sea forcing is
intentionally discontinuous at segment boundaries. (d) A stiffness ramp is a nondimensional
multiplier $\kappa$; the reference scale $\omega_n$ is held fixed within a trajectory.
(e) `simulate_restarted_batch` starts independent futures from stored state; because those futures
discard the realized forcing phases, they are restart comparators and not bounds on prediction
from the full motion history (Section 6). (f) Stochastic parametric modulation uses an
independently phased elevation record normalized to a prescribed standard deviation. (g) Capsize
time has one-step resolution, without sub-step root interpolation. (h) The Melnikov comparison of
Section 3 defines the simulated boundary at 50 percent capsize over uniformly spaced forcing
phases and 120 natural periods. (i) Parquet output is uncompressed so that identical inputs
produce byte-identical files under a fixed PyArrow version.

## 3. Validation

The generator is tested against analytic limits before any warning method is evaluated. Three
limits expose different implementation errors: the linear response checks the forcing scale, the
Mathieu boundary checks the parametric stiffness, and the Melnikov threshold checks the nonlinear
escape geometry.

In the linear limit, the angular-acceleration input has transfer function

$$
 H(\omega)=\frac{1}{\omega_n^2-\omega^2+i2\zeta\omega_n\omega},\qquad
 \sigma_\phi^2=\int_0^\infty|H(\omega)|^2S_m(\omega)\,\mathrm{d}\omega,
$$

by the standard input–output spectral relation (Bendat and Piersol 2010). At exact principal
parametric tuning, first-order averaging of $x''+2\zeta x'+[1+h_0\cos(2\tau)]x=0$ gives the
small-damping stability boundary $h_{0,c}=4\zeta$ (Nayfeh and Mook 1979). For Family 1 under
harmonic forcing,

$$
 x''+2\zeta x'+x-x^3=F\cos(\Omega\tau),
$$

the unperturbed heteroclinic orbit is $x_h=\tanh(\tau/\sqrt2)$, and substitution into the Melnikov
integral gives the simple-zero threshold

$$
 F_M(\Omega)=\frac{4\zeta}{3\pi\Omega}
 \sinh\left(\frac{\pi\Omega}{\sqrt2}\right).
$$

Both orbit integrals are also evaluated by quadrature, and the closed form is compared against
direct phase-ensemble capsize sweeps. The interpretation of the threshold as a necessary
condition, and its application to ship rolling, follow Falzarano, Shaw and Troesch (1992). The
bound is not treated as a sufficient capsize condition.

| Physics component | Acceptance test | Criterion |
| --- | --- | --- |
| JONSWAP and FFT realization | `test_jonswap_spectral_fidelity_and_significant_height`; `test_deep_water_slope_fourier_coefficients_have_expected_magnitude_and_sign` | recovered $H_s$ within 2%; log-PSD correlation above 0.95; band energy within 12%; $C_\alpha/C_\eta=-i\omega^2/g$ |
| Linear forcing and response | `test_linear_limit_variance_matches_spectral_response` | ensemble variance within 6% |
| Parametric stiffness | `test_mathieu_principal_tongue_boundary` | growth sign brackets $4\zeta$; boundary within 10% |
| Softening separatrix | Melnikov quadrature and capsize-boundary tests | closed form within $10^{-5}$ of quadrature; simulated boundary above the prediction, with correlated shape and a narrowing low-damping gap |
| Integration and forcing | `test_step_halving_convergence_statistics`; `test_linear_harmonic_forcing_converges_at_fourth_order_asymptotically` | aggregate coarse/fine variance within 3% and capsize-rate change at most 0.05; observed order at least 3.5 on the two finest refinements |
| Seed propagation | spectrum and batch determinism tests | bitwise equality |
| Causal normalization | `test_future_only_leakage_probe_has_teeth` | causal AUC within 0.08 of 0.5; leaky control near perfect |
| Parquet and manifest writer | `test_same_inputs_produce_byte_identical_dataset` | every emitted byte equal |

No acceptance test is skipped by `uv run pytest` or by `rahola validate` when either command runs
from the repository root. Slow tests carry a marker only so that a developer may exclude them
while iterating; the acceptance commands include them.

## 4. The synthetic data program

Every reference trajectory lasts 600 seconds, is sampled at 2 Hz, and uses a four-second natural
period. The checked campaign contract comprises 58,500 trajectories and 1.629 GiB of Parquet
data; the most recent full generation required 94.5 seconds on the development host.

| Campaign layer | Trajectories | Purpose |
| --- | ---: | --- |
| Three stationary families | 12,000 | fitting under fixed dynamics |
| Three stiffness ramps | 10,500 | warning as stability erodes |
| One sea-state step | 6,000 | adaptation after abrupt shift |
| Three rare-event evaluation campaigns | 18,000 | false-alarm cost at 0.95–2.00% capsize rates |
| Five forcing-bandwidth campaigns | 12,000 | bandwidth separated from severity |

Each stored row holds one trajectory: seed, capsize outcome and time, common time vector, roll
angle, roll rate, and metadata. A sorted manifest records the full configuration, ordered seeds,
package version, Git commit, shard hashes, and outcome counts; generation is byte-deterministic
for a fixed software environment. Disjoint seed blocks separate training, calibration, test, and
two guarded reserve allocations, and the public loading utilities refuse both reserve blocks. The
complete campaign table, split counts, measured capsize rates, seed ranges, and manifest hashes
are given in [`DATA.md`](DATA.md).

Supervised examples are constructed only from information available at the scoring time. Physics
comparators use configured nondimensional units. Learned detectors use either one trend and scale
fitted to the complete past-only scoring window or the historical cumulative-online transform;
the former is primary, while the latter carries state from the full observed history. A
future-only leakage probe verifies all three paths against a deliberately leaky control.
Forecasting experiments use 120 seconds of history with 30-
and 60-second outcome horizons; detector experiments use 60 natural periods of roll and roll-rate
history, a 50-period capsize horizon, a five-period exclusion band, and a ten-second score stride.
A positive window precedes capsize within the horizon. A negative window has a complete scored
horizon with no capsize within the horizon and the exclusion band. Ambiguous and record-truncated
windows are discarded.

## 5. Evaluation procedure

### 5.1 Episodes and metrics

Operational scoring is separated from supervised evaluation. Every causal pre-capsize score is
emitted, while exposure, event counts, and labels stop at the last endpoint with a complete
outcome horizon. Three consecutive threshold crossings open an alarm episode at the confirming
window; refractory and decorrelation rules prevent dense scores from being counted as repeated
alarms. The reported quantities are episode sensitivity, false episodes per exposure hour, and
lead time. Uncertainty resamples complete trajectories within campaigns and recomputes the full
episode logic; exact binomial intervals are retained only for event-level capsize counts. These
intervals condition on the calibration-selected policy frozen before test scoring.

### 5.2 Threshold selection

All operating controls and thresholds are selected on calibration data, frozen, and evaluated at
one point in each corrected test run. Conformal calibration of forecast upper bounds follows
Romano, Patterson and Candès (2019); the online adaptation experiments follow Gibbs and Candès
(2021).

### 5.3 Revision notice

An audit conducted in August 2026 found that the operating points first reported by this program
had been selected on test outcomes, and that one guarded holdout evaluation had selected its
threshold using the holdout labels. All development results were regenerated under the corrected
protocol of Section 5.2, and the correction is enforced by the test suite rather than by
convention. Both reserve blocks are now expended. Their result artifacts are retained unchanged
as historical records and are so labeled; they do not constitute prospective validation of the
corrected methods. The ordinary development-test splits had already been inspected during v0.1,
so the corrected development results are retrospective reanalyses, not fresh prospective
validation.

### 5.4 Information sets

The methods do not all receive the same information. In particular, the danger margin and the
engineered XGBoost baseline use the true configured vessel model; they are comparators, not
motion-only detectors.

| Method or feature | Motion window | Full motion history | Vessel configuration | Protocol clock | Sea state | Wave field | Realized future forcing |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN, fixed-window primary | yes | no | no | no | no | no | no |
| CNN, cumulative-online | yes | yes, through normalization | no | no | no | no | no |
| Classical EWS, GLRT, neighbor score | yes | no | scale constants only | no | no | no | no |
| Danger margin | endpoint state | no | yes; known-configuration physics comparator | no | no | no | no |
| XGBoost engineered features | yes | mode dependent | yes; configuration-assisted features | no | no | no | no |
| Protocol-clock comparator | no | no | no | yes | no | no | no |
| C1/C2 restart comparators | endpoint or filtered state | C2 filtering only | yes | ramp state only | yes | no | independent replacement future |
| D4 evaluator | detector dependent | detector dependent | detector dependent | no | yes | evaluator only | evaluator reconstruction only |

No warning method observes the realized future forcing. D4's reconstructed wave field is used
only after scoring to characterize events, and the C1/C2 rollouts replace—not reveal—the realized
future.

## 6. Findings

| Question | Result | Assessment |
| --- | --- | --- |
| Stationary conformal calibration | regenerated LSTM mean absolute coverage error 0.89 percentage points (E1_v02) | implementation behaves as intended |
| Online adaptation after an abrupt sea-state step | ACI, DtACI, and recent-score recalibration each missed the joint rolling-coverage and alarm-cost criteria | not attained under this shift |
| Best pooled operating point | fixed-window network: 91.00% sensitivity at 13.409 false episodes/h; cumulative-online network: 93.83% at 16.451/h; classical statistics: 100% (near-always-on) at 21.391/h | the primary network lowers pooled alarm cost |
| Threshold transfer across families | fixed-window missed 90% on biased; cumulative-online missed it on softening and biased | no all-family operating point established |
| Effect of forcing bandwidth | primary network window AUC 0.883–0.913 at every bandwidth | ranking survives; operating-cost claim remains inconclusive |
| Character of false alarms | 75–88% of false episodes overlapped evaluator-defined critical wave groups | descriptive; no matched null was tested |
| Discrimination inside the severe regime | no motion-only method exceeded orientation-independent AUC 0.556 after a full 60-period post-step wash-in | near chance after regime entry; preregistered prediction retained |
| State-inference and transfer probes | exact-state restart AUC 0.850; fixed-window/cumulative XGBoost 0.723/0.768; fixed-window/cumulative CNN 0.652/0.622; protocol clock 0.656 | architecture comparisons remain confounded by protocol time and unequal information sets |

Two qualifications attach to the final row. The restart comparators replace the realized forcing,
and since the seaway is temporally correlated, the motion history carries information about the
near future that such comparators cannot represent; they are therefore reference points, not
ceilings. Common future-forcing seeds reduce C1/C2 Monte Carlo noise but do not equalize their
information. The filtered-state comparator reached AUC 0.486 and remained limited by its particle
filter.

Taken as a whole, the results indicate that motion history carries usable information about the
slowly varying stability state, and near-chance information about the timing of the terminal wave
encounter once a severe regime is established. Most false episodes overlap a broad,
evaluator-defined high-envelope wave-group proxy. Without a prevalence-matched null, that
descriptive overlap does not establish encounter detection or causation.

## 7. Limitations

The model has one roll degree of freedom, with no heave, sway, yaw, pitch, or hull-geometry
input, and its polynomial restoring is an archetype rather than a vessel-specific GZ curve. There
is no radiation-damping memory. The seaway is deep-water, long-crested, and unidirectional, with
no directional spreading, finite-depth dispersion, encounter-frequency transformation,
diffraction, or wave nonlinearity, and the effective wave-slope coefficient is constant rather
than a frequency-dependent admittance. Step forcing is intentionally discontinuous; event timing
has one-step resolution; the stationary phase-randomized seaway produces only those wave groups
implied by the spectrum. The results are research findings on a reduced-order model and support
no operational or safety claim.

## 8. Conclusions and recommendations

Three conclusions are considered established within the scope of the model: motion history ranks
capsize risk well above chance at every forcing bandwidth tested; after a complete post-step
wash-in, no motion-only method examined here exceeds orientation-independent AUC 0.556; and most
false alarm episodes overlap the declared high-envelope wave-group proxy. One negative conclusion
is equally definite: neither CNN normalization mode retains 90 percent sensitivity across all
held-out families. The physics comparators reach that sensitivity only at roughly 20.8–21.7 false
episodes per hour, close to always on.

It is recommended that further work address the information deficit rather than the detector
architecture. The natural instrument-side question, staged so that information sufficiency is
tested before any particular sensor, is whether a single spatially advanced measurement of the
oncoming seaway restores within-regime discrimination. The evaluation harness, campaign
definitions, and frozen splits of this repository are suitable for that study without
modification.

## Appendix A. Installation and operation

Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) are required. From a source checkout,
install the workspace and run its development validation suite:

```sh
uv sync --all-packages --all-extras
uv run rahola validate
```

`rahola validate` invokes `pytest` in the current source checkout. It is a developer command, not
a self-test embedded in the installed package.

A minimal batch of 128 independently forced trajectories:

```python
from rahola import SimulationConfig, simulate_batch

dataset = simulate_batch(SimulationConfig(), seeds=range(128))
```

The result contains a common time grid, roll angle, roll rate, seed, capsize indicator, capsize
time, configuration, and per-trajectory metadata. Samples after capsize are `NaN`.

The following commands reproduce the historical v0.1 record only. They write unsuffixed v0.1 paths,
so run them in a disposable checkout and output directory; do not run them against the frozen
repository artifacts:

```sh
uv run rahola-lab generate --all --out data/reference --chunk-size 256
uv run python examples/e1_coverage.py
uv run python examples/d1_detectors.py
uv run python examples/p3_ceiling.py
```

The v0.2 selective-regeneration commands are documented in [`DATA.md`](DATA.md), and the exact
supersession record is in [`RESULTS_v02.md`](RESULTS_v02.md). Generated trajectories stay outside
version control; frozen campaign definitions, manifest anchors, numeric result files, and figures
are tracked. Historical artifacts record source-tree and reference-data fingerprints. The tracked,
self-digested v0.2 provenance manifest additionally binds both reference anchors, every v0.2
artifact digest, and declared result dependencies. Loaders reject stale or mutated content.
Measured throughput on an Apple-silicon host (JAX 0.11.0, CPU backend, after a one-trajectory
warm-up):

```text
uv run python examples/benchmark.py --trajectories 256 --duration-s 3600
trajectories=256 elapsed_s=1.191 simulated_hours_per_wall_minute=12899.9
```

This is an end-to-end measurement including synthesis of the 256 seeded forcing records. Memory
and thermal behavior at full campaign size should be measured on the execution host.

## References

- Anastopoulos, P. A., and Spyrou, K. J. (2017). Ship dynamic stability assessment based on
  realistic wave groups. *Ocean Engineering* 134.
  [doi:10.1016/j.oceaneng.2016.10.042](https://doi.org/10.1016/j.oceaneng.2016.10.042)
- Belenky, V., Weems, K., Lin, W.-M., Pipiras, V., and Sapsis, T. (2024). Estimation of probability
  of capsizing with split-time method. *Ocean Engineering* 292, 116452.
  [doi:10.1016/j.oceaneng.2023.116452](https://doi.org/10.1016/j.oceaneng.2023.116452)
- Bendat, J. S., and Piersol, A. G. (2010). *Random Data: Analysis and Measurement Procedures*,
  4th ed. Wiley. [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428)
- Bulian, G., and Francescutto, A. (2004). A simplified modular approach for the prediction of
  roll motion due to wind and waves.
  [doi:10.1243/1475090041737958](https://doi.org/10.1243/1475090041737958)
- Bulian, G., and Francescutto, A. (2011). Effect of roll modelling in beam waves under
  multi-frequency excitation. *Ocean Engineering* 38.
  [doi:10.1016/j.oceaneng.2011.07.004](https://doi.org/10.1016/j.oceaneng.2011.07.004)
- Dean, R. G., and Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and Scientists*.
  World Scientific. ISBN 9789810204204.
- Falzarano, J. M., Shaw, S. W., and Troesch, A. W. (1992). Application of global methods for
  analyzing dynamical systems to ship rolling motion and capsizing.
  [doi:10.1142/S0218127492000100](https://doi.org/10.1142/S0218127492000100)
- Galeazzi, R., Blanke, M., Falkenberg, T., Poulsen, N. K., Violaris, N., Storhaug, G., and Huss,
  M. (2015). Parametric roll resonance monitoring using signal-based detection. *Ocean
  Engineering* 109.
  [doi:10.1016/j.oceaneng.2015.08.037](https://www.sciencedirect.com/science/article/abs/pii/S0029801815004357)
- Gibbs, I., and Candès, E. (2021). Adaptive conformal inference under distribution shift.
  [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
- Hairer, E., Nørsett, S. P., and Wanner, G. (1993). *Solving Ordinary Differential Equations I*,
  2nd ed. Springer. [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
- Hasselmann, K., et al. (1973). Measurements of wind-wave growth and swell decay during the Joint
  North Sea Wave Project (JONSWAP).
  [Max Planck repository](https://pure.mpg.de/view/item_3262854_4)
- IMO (2008). *Explanatory Notes to the International Code on Intact Stability*, MSC.1/Circ.1281.
  [official circular](https://wwwcdn.imo.org/localresources/en/OurWork/Safety/Documents/MSC.1-CIRC.1281.pdf)
- Nayfeh, A. H., and Mook, D. T. (1979). *Nonlinear Oscillations*. Wiley.
  [doi:10.1002/9783527617586](https://doi.org/10.1002/9783527617586)
- Rahola, J. (1939). *The Judging of the Stability of Ships and the Determination of the Minimum
  Amount of Stability*. Doctoral thesis, Technical University of Finland, Helsinki.
- Romano, Y., Patterson, E., and Candès, E. (2019). Conformalized quantile regression.
  [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
- Shinozuka, M., and Deodatis, G. (1991). Simulation of stochastic processes by spectral
  representation. *Applied Mechanics Reviews* 44(4).
  [doi:10.1115/1.3119501](https://doi.org/10.1115/1.3119501)
- Story, W. R. (2009). *Predicting Ship Capsize Using Lyapunov Exponents*. Master's thesis,
  Virginia Tech. [repository record](https://vtechworks.lib.vt.edu/items/7eee36dd-055b-4aec-b49d-b173c2232278)
