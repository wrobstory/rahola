# Rahola

Rahola is a falsification-first synthetic 1-DOF ship-roll dynamics library. It
generates seeded nonlinear roll trajectories for softening, parametric, and
biased-restoring archetypes. Phase 0 contains the shared data engine and its
analytic validation suite; it deliberately contains no alarm or machine-learning
model. Every physics component is paired with a test intended to disprove it.

> This research software is not a vessel-operational safety system.

## Install and run

Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```sh
uv sync --all-extras
uv run pytest
uv run rahola validate
uv run rahola generate --config configs/family1_stationary.yaml --out data/demo
```

The public API is:

```python
from rahola import SimulationConfig, simulate_batch

dataset = simulate_batch(SimulationConfig(), seeds=range(128))
```

Three campaign files live in `configs/`. The validation figures are reproducible
scripts in `examples/`; for example, `uv run python examples/basin_erosion.py`.

## Units and nondimensionalization

The public boundary uses SI seconds, metres, radians, and radians/second. YAML
comments show degree equivalents where useful, but values stored and returned by
the library are radians. Internally,

\[
  x=\phi/\phi_v,\qquad \tau=\omega_n t,\qquad
  x'=\dot\phi/(\phi_v\omega_n).
\]

This makes the common equation

\[
 x''+2\zeta x'+q x'|x'|+\kappa(\tau)[1+h(\tau)]
 (x-x^3+\lambda x^5)=f(\tau)+b.
\]

Here `q` is the configured nondimensional quadratic-damping coefficient
(equivalent to \(\beta\phi_v\) under the dimensional damping convention in the
mission), `lambda` is zero unless the quintic option is enabled, and `kappa` is
one except in a stiffness-ramp campaign. Fixed-step classical RK4 is evaluated
at the beginning, midpoint, midpoint, and end of every step. The implementation
uses at least 40 steps per natural period and also refines the step when needed
to make the requested output rate a true decimation. RK4 is appropriate here
because every random input is realized first as a finite, smooth trigonometric
record; this is a random ODE, not an Itô SDE. See Hairer, Nørsett & Wanner for the
classical RK family [R8].

## Roll families

The dimensional models use

\[
 \ddot\phi+2\zeta\omega_n\dot\phi+\beta\dot\phi|\dot\phi|
 +R(\phi,t)=m(t).
\]

| Family | Restoring / bias | Intended archetype | Principal sweep range |
| --- | --- | --- | --- |
| 1: softening | \(R=\omega_n^2(\phi-\phi^3/\phi_v^2)\) | dead ship / pure-loss escape | \(\zeta=0.01\ldots0.10\), \(r=0.01\ldots0.15\) |
| 2: parametric | \(R=\omega_n^2[1+h(t)](\phi-\phi^3/\phi_v^2)\) | parametric roll | \(h_0=0\ldots0.4\), \(\omega_e/\omega_n=1.5\ldots2.5\) |
| 3: biased | Family 1 plus constant nondimensional moment \(b\) | damage / steady heel | \(b=-0.3\ldots0.3\), side-specific escape angles |

The cubic/quintic 1-DOF form and the limits of alternative excitation models are
consistent with Bulian & Francescutto's nonlinear roll-model comparison [R4].
Family 2 accepts deterministic \(h=h_0\cos(\omega_e t)\), or an independent
narrow-band JONSWAP realization normalized to the requested modulation standard
deviation. Family 3 can use unequal positive and negative escape angles.

Capsize is the first integration endpoint at which either applicable escape
angle is reached. That state is absorbing. The event time is recorded; emitted
samples strictly after it are NaN and cannot enter a window.

## Irregular-sea forcing

The one-sided angular-frequency JONSWAP spectrum is

\[
 S_\eta(\omega)=\alpha g^2\omega^{-5}
 \exp[-\tfrac54(\omega_p/\omega)^4]
 \gamma^{\exp[-(\omega-\omega_p)^2/(2\sigma^2\omega_p^2)]},
\]

with \(\sigma=0.07\) below the peak and 0.09 above it. Rahola numerically
normalizes \(\alpha\) so \(m_0=H_s^2/16\), hence
\(H_s=4\sqrt{m_0}\). The form and constants come from the JONSWAP field program
[R1].

For FFT-bin width \(\Delta\omega\), the deterministic component amplitude is
\(A_j=\sqrt{2S_\eta(\omega_j)\Delta\omega}\), while each phase is independently
uniform. Thus

\[
 \eta(t)=\sum_j A_j\cos(\omega_jt+\theta_j).
\]

This deterministic-amplitude/random-phase spectral representation is chosen
because every finite realization has the prescribed bin energy and converges to
the target Gaussian process as the component count grows. Random amplitudes
would represent ensemble scatter more directly but add avoidable finite-record
variance. The ergodicity and FFT tradeoff is treated by Shinozuka & Deodatis
[R2]. Rahola uses at least 200 positive-frequency components.

For a progressive Airy component, the deep-water dispersion relation gives
\(k_j=\omega_j^2/g\), and spatial differentiation puts slope in quadrature:
\(\alpha_j=\partial\eta_j/\partial x=-k_jA_j\sin(\cdot)\) [R3]. The simplified
roll-excitation chain is

\[
 \eta\longrightarrow \alpha=\partial\eta/\partial x
 \longrightarrow m=\omega_n^2 r\alpha
 \longrightarrow f=m/(\phi_v\omega_n^2)=r\alpha/\phi_v.
\]

The effective wave-slope coefficient \(r\) is an input, not a hull-derived
quantity. This follows the established 1-DOF effective-slope abstraction in
Bulian & Francescutto [R5] and the IMO intact-stability explanatory notes [R6].

The complete forcing is synthesized once by inverse FFT on the RK4 half-step
grid; no cosine sum occurs inside the RHS. Step protocols synthesize every
piecewise-stationary sea independently and replace the boundary sample with the
new segment. State remains continuous, but forcing need not be continuous at a
declared environmental step.

## Analytic validation

For the linear limit, angular-acceleration input has transfer function

\[
 H(\omega)=\frac{1}{\omega_n^2-\omega^2+i2\zeta\omega_n\omega},\qquad
 \sigma_\phi^2=\int_0^\infty|H(\omega)|^2S_m(\omega)\,d\omega,
\]

using the standard random-vibration input/output spectral relation [R7]. At
exact principal parametric tuning, first-order averaging of
\(x''+2\zeta x'+[1+h_0\cos(2\tau)]x=0\) gives the small-damping boundary
\(h_{0,c}=4\zeta\) [R9].

For Family 1 with harmonic forcing,

\[
 x''+2\zeta x'+x-x^3=F\cos(\Omega\tau),
\]

the unperturbed heteroclinic orbit is
\(x_h=\tanh(\tau/\sqrt2)\). Substitution into the Melnikov integral gives the
simple-zero threshold

\[
 F_M(\Omega)=\frac{4\zeta}{3\pi\Omega}
 \sinh\!\left(\frac{\pi\Omega}{\sqrt2}\right).
\]

Rahola independently quadratures both orbit integrals and compares this
necessary-condition lower bound with direct phase-ensemble capsize sweeps. The
global ship-roll application and interpretation of the bound follow Falzarano,
Shaw & Troesch [R10]. It is not treated as a sufficient capsize condition.

| Physics component | Falsification test | Acceptance used here |
| --- | --- | --- |
| JONSWAP and FFT realization | `test_jonswap_spectral_fidelity_and_significant_height` | recovered \(H_s\) within 2%; log-PSD correlation >0.95; band energy within 12% |
| Linear forcing/response chain | `test_linear_limit_variance_matches_spectral_response` | ensemble variance within 6% |
| Parametric stiffness | `test_mathieu_principal_tongue_boundary` | growth sign brackets \(4\zeta\); boundary within 10% |
| Softening separatrix | Melnikov quadrature and capsize-boundary tests | formula within \(10^{-5}\); capsize bound above prediction with correlated shape and narrowing low-damping gap |
| RK4 and precomputed forcing | `test_step_halving_convergence_statistics` | variance within 3%; capsize-rate change <=0.05 |
| Seed propagation | spectrum/batch determinism tests | bitwise equality |
| Causal normalization | `test_future_only_leakage_probe_has_teeth` | causal AUC within 0.08 of 0.5; leaky control near perfect |
| Parquet/manifest writer | `test_same_inputs_produce_byte_identical_dataset` | every emitted byte equal |

No acceptance test is skipped by `uv run pytest` or `rahola validate`. Slow
tests are marked only so developers can request `pytest -m 'not slow'` while
iterating; the acceptance command includes them.

## Campaigns, windows, and storage

Stationary, linear-ramp, and multi-step sea-state protocols are first-class
configuration types. Ramps can change nondimensional stiffness or forcing scale.
Step segments use independent phases. A configuration plus ordered unique seeds
fully determines the batch.

`CausalTransformer` walks forward once. Each output sample is standardized and,
optionally, linearly detrended using sums fitted strictly before that sample.
Callers cannot fit it on a future slice. `make_windows` then cuts that already
causal stream. A window is positive when capsize occurs within its horizon,
negative when the run is non-capsizing or capsize lies beyond horizon plus
buffer, and discarded inside the exclusion buffer. Windows stop before capsize.

Storage is uncompressed sharded Parquet plus a sorted JSON manifest containing
the full configuration, seed list, package version, Git commit, SHA-256 per
shard, and family/protocol outcome counts. Parquet was selected over Zarr for
portable tabular metadata and simple independent shards. Compression and
dictionary encoding are disabled to make byte determinism straightforward.

## Performance

Measured on this arm64 Apple-silicon host with JAX 0.11.0, CPU backend, after a
one-trajectory warm-up:

```text
uv run python examples/benchmark.py --trajectories 256 --duration-s 3600
trajectories=256 elapsed_s=1.274 simulated_hours_per_wall_minute=12058.5
```

This is an end-to-end measurement including the 256 independently seeded FFT
records and array materialization. It exceeds the 500 simulated-hours/minute
guideline by 24x. It is not an extrapolation to 10,000 trajectories; memory and
thermal behavior at that campaign size should be measured on the execution host.
JAX owns the backend choice, so selecting a GPU does not change the model code.

## Explicit judgment calls

- Deterministic amplitudes plus random phases were selected for exact discrete
  spectral energy and lower finite-record variance; phases remain the stochastic
  degrees of freedom.
- The requested output rate may force an integration step smaller than
  \(T_n/40\); output is never interpolated upward from a coarser state grid.
- Each step-sea segment is independently synthesized, so forcing may jump while
  roll angle and rate remain continuous.
- A ramp in `stiffness` is a nondimensional multiplier \(\kappa\); the reference
  time scale \(\omega_n\) is kept fixed during that trajectory.
- Stochastic parametric modulation uses an independently phased JONSWAP
  elevation record, RMS-normalized to `stochastic_std`.
- Capsize time is the first RK4 endpoint beyond the threshold, without
  sub-step root interpolation.
- The Melnikov comparison defines the simulated boundary at 50% capsize over
  uniformly spaced forcing phases and 120 natural periods. Melnikov remains only
  a necessary heteroclinic-intersection bound.
- Uncompressed Parquet was favored over smaller files so identical inputs remain
  byte-identical across repeat writes with a fixed PyArrow version.

## Known limitations

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
