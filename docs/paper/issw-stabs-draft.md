<!-- DRAFT v0.4 — Ocean Engineering submission. Revised 2026-08-08 against the second
     adversarial pre-submission review: abstract/Discussion/Conclusion aligned with corrected
     body claims (2.5x factor removed, 92% heralded denominator, rate-factor language, one-shot
     scope), experiment ledger added, F1 formula and censor counts inlined, Tier-1 references
     added after verification. -->

# Motion-Only Capsize Warning on an Audited Synthetic Benchmark: Ranking Skill Without Threshold Transfer

**W. R. Story**

## Abstract

Can the measured roll motion of a ship warn of capsize soon enough to act, at a false-alarm
cost a crew could accept? We study this question on a synthetic benchmark validated against
analytic limits and internal numerical checks; external ship-level validity is out of scope: a
one-degree-of-freedom nonlinear roll model, three restoring-curve families representing
distinct ways a ship loses stability, and a baseline detector corpus of 58,500 seeded
trajectories under JONSWAP forcing, with later preregistered experiments materializing
separate additional campaigns.
Five warning methods compete under one protocol in which every operating threshold is chosen on
calibration data and frozen before each test evaluation, with every historical deviation from
that protocol recorded in the repository rather than repaired silently: a small convolutional
network, classical variance
and autocorrelation trend statistics, a likelihood-ratio detector, a closed-form
critical-roll-rate margin, and a phase-space neighbor-count score. Four findings emerge.
(a) Ranking skill is real at every forcing bandwidth tested; the fixed-window primary network
separates dangerous from ordinary intervals with window AUC 0.883 to 0.913. (b) Inside an
established severe regime, no tested motion-only statistic predicts which encounter will be fatal;
all score near chance. (c) Most nominally
false alarms, 75 to 88 percent in a mixed-generation historical diagnostic, coincide with
high-envelope wave-group episodes, a coincidence proxy without a matched null, that the vessel
survived. (d) No method's threshold transfers across failure modes at fixed sensitivity. We
then operate the split-time upcrossing decomposition as an onboard rate estimator, in online
and offline-calibrated forms; both fail preregistered calibration tests for a reason we
localize. In an in-sample check, the decomposition reproduced realized counts: given the
empirically measured probability that a
threshold crossing is terminal, it reproduces realized counts within 3 in 71. But the
closed-form criticality model overestimates the lethality of a crossing nearly fivefold, and
in 92 percent of heralded capsizes (221 of 240) no ten-second emission opportunity separated
the terminal crossing from the recorded capsize. An audit during the study found our own first results inflated by test-set
threshold selection; every number reported here is the corrected value. The picture that
survives is simple, and a final embedded-group experiment quantifies it within its own design.
Embedding a frozen library of six wave-group shapes at six prescribed heights into independent
irregular preludes, so that the vessel arrives at each group naturally, capsize within the
group is predicted at AUC 0.513 from the vessel's entry state, 0.934 from the group's
engineered descriptors, and 0.938 from both. Within that balanced single-configuration
experiment, motion reveals the slowly building vulnerability and the arriving group's
description dominates the rest.

*Highlights:*

- Five motion-only capsize scores were compared with frozen thresholds.
- Window AUC stayed 0.883 to 0.913 across tested forcing bandwidths.
- Tested scores were near chance inside an established severe regime.
- No 10-s emission opportunity preceded 92 percent of heralded capsizes.
- In one six-shape library, wave-group descriptors gave AUC 0.934.

*Keywords:* capsize, intact stability, real-time warning, split-time method, preregistered
evaluation, synthetic benchmark.

## 1. Introduction

Begin from first principles. A ship in beam seas is a lightly damped nonlinear oscillator
(Nayfeh and Mook 1979): waves push, the righting arm restores, and capsize is the escape of
that oscillator over the rim of its potential well. Two different questions hide inside "can we warn?" One asks whether
the well is becoming shallow — vulnerability, a slow process. The other asks when a particular
group of waves will push the ship over — the encounter, a fast one. Every result in this paper
answers one of those two questions, and the paper's central finding is that motion measurement
answers only the first.

Interest in warning of capsize from onboard motion measurement is at least two decades old. The
Lyapunov-exponent programme of the mid-2000s (McCue and Troesch 2004, 2006) sought a precursor
of capsize in the divergence of measured trajectories; the author's
thesis in that programme (Story 2009) found empirically that the useful signal lay not in
estimated exponents but in the loss of phase-space neighbors, and reported a neighbor-count
warning flag with encouraging lead times and an honestly documented false-alarm problem. The
thread was not taken up. The signal-based detectors of Galeazzi et al. (2013, 2015) for
parametric roll, with designed false-alarm probabilities and full-scale blind validation, remain
the most complete fielded success of motion-based stability-failure detection, though onboard
monitoring efforts are again active, for damaged ships (Mak et al. 2025) and naval vessels
(Santiago Caamaño et al. 2025), and the field's general
machinery moved instead to offline probabilistic assessment (Belenky and Sevastianov 2007), of
which the split-time method (Belenky, Weems and Lin 2016; Belenky et al. 2024) is the principal
example, and to operational guidance derived from precomputed simulation under the
second-generation intact stability criteria (Bačkalov et al. 2016; IMO 2020, 2022; Petacco and
Gualeni 2020; Shigunov 2023).

The relationship of the present work to the split-time method deserves statement at the outset,
since that method supplies both the strongest available machinery and the sharpest open question.
Belenky et al. (2024) compute the probability of capsizing offline. Their central object is an
upcrossing: the roll angle passing outward through a fixed intermediate threshold, here the
angle of maximum righting arm. The metric of danger, a critical roll rate at those upcrossings,
is evaluated along simulated histories, its tail is
extrapolated with a physics-informed exponential form (Glotzer et al. 2024), and the estimate
is validated by confidence-interval capture against direct counting (Weems et al. 2023; on
direct-counting interval estimands see Wandji et al. 2024). Three transfers from that programme are
attempted here. The metric itself is operated, to our knowledge for the first time, as a
continuously evaluated onboard alarm statistic computed from measured state alone, and is
benchmarked against learned and statistical competitors on equal terms. The statistical
discipline of the split-time programme, declustering, interval estimates on every rate, and
validation against direct counting, is transplanted to warning evaluation, where it remains
rare; an uncertainty-aware short-horizon warning benchmark has appeared in parallel (Joo et
al. 2026). And the boundary of the transfer is itself measured: the engineering-fidelity form of the
metric requires re-simulation with the incident wave realization known, and the present results
quantify operationally what that requirement conceals, namely that the information carried by the
future waves is absent from measured motion precisely where event timing is decided.

The present study returns to the motion-only question with tools that did not exist in 2009 and
with an evaluation protocol designed against the failure modes that make warning research easy to
overstate. A warning method may rank dangerous intervals well yet possess no usable threshold; a
threshold may fail to transfer across failure modes or sea states; apparent skill may arise from
protocol time, truncated outcome windows, or the use of future data during normalization; and,
because capsizes are rare, an aggregate score conceals all of these. Each failure mode is tested
separately here. The study also reports, deliberately, an audit of its own methodology: the
operating points first obtained were later found to have been selected on test outcomes, an error
we believe is widespread in the published literature on learned motion prediction, and
the corrected results quantify its size.

Three questions are addressed. First, does motion history rank capsize risk under forcing broad
enough that classical critical-slowing-down indicators fail? Second, do the resulting operating
thresholds transfer across stability-failure modes at fixed sensitivity? Third, what do the false
alarms of such systems consist of?

## 2. The synthetic benchmark

The benchmark is the simplest ship that can capsize (Figure 1): a one-degree-of-freedom
nonlinear roll model, validated against analytic limits, generating seeded trajectories under
JONSWAP forcing (Hasselmann et al. 1973). Three restoring
families represent failure archetypes: softening restoring (dead-ship escape), time-varying
stiffness (parametric roll), and biased restoring with asymmetric escape angles (damage, steady
heel), in the scope of established one-degree-of-freedom roll comparisons (Bulian and
Francescutto 2004, 2011). Following-sea failure modes, broaching foremost (Umeda et al. 2016),
are outside the benchmark's scope, a limitation the transfer results of Section 5 should be read
against. Forcing is synthesized spectrally with deterministic amplitudes and
seeded random phases (Shinozuka and Deodatis 1991); integration is fixed-step Runge-Kutta at no
fewer than 40 steps per natural period; capsize is an absorbing state and post-event samples are
excluded from every window. The benchmark takes its name from Rahola (1939), whose thesis
distilled casualty records into judged stability criteria; the aim here is the same empirical
discipline applied to warning.

The model is nondimensional, and its forcing deserves a dimensional account. The excitation
represents an effective wave-slope moment process: a JONSWAP elevation spectrum is converted to
a wave-slope spectrum, scaled by an effective-wave-slope coefficient and the square of the
natural frequency, and applied as a roll moment — the linear-limit validation gate below checks
exactly this chain against the spectral calculation. The natural period (four seconds at
reference), damping ratio, and escape-angle geometry are set to representative small-vessel
values; the three restoring shapes and the effective-wave-slope coefficient are chosen for
archetypal behavior, not calibrated to a hull. No coefficient in the benchmark is claimed as a
measured property of any ship, which is one reason external ship-level validity is declared out
of scope.

Validation gates precede all warning experiments: recovered significant wave height within 2
percent; linear-limit response variance within 6 percent of the spectral calculation; the
principal Mathieu boundary within 10 percent of the averaging result; and the simulated harmonic
capsize boundary at or above the heteroclinic Melnikov threshold (Falzarano, Shaw and Troesch
1992) with the correct frequency shape, the bound being treated as a necessary condition only.
The acceptance run at the submission commit passes the full suite of 244 tests with none
skipped, and the attained residuals of each gate are serialized in a versioned artifact
(`results/validation_acceptance.json`) alongside their thresholds.

The baseline detector corpus comprises 58,500 trajectories of 600 seconds at a four-second
natural period:
stationary campaigns per family, stiffness-ramp campaigns representing slow stability erosion,
one sea-state step campaign, rare-event evaluation campaigns at 0.95 to 2.0 percent capsize
rates, and five forcing-bandwidth campaigns in which JONSWAP peak enhancement is varied from 1.0
to 30 with terminal ramp severity retuned per bandwidth so that outcome rates remain in a common
band; the last decision prevents severity from acting as a hidden bandwidth axis. Disjoint seed
blocks separate training, calibration, test, and guarded holdout data. The later preregistered
experiments of Section 5 — the split-time operation, the tangent-exponent trial, the wave-field
audit, and the embedded-group experiment — materialized separate additional campaigns, each
recorded against the same seed ledger.

Supervised windows use 60 natural periods of causally normalized roll and roll-rate history, a
50-period outcome horizon, and an exclusion band for near misses; a future-only leakage probe
verifies the entire feature path against a deliberately leaky control. The label discipline
responds directly to critiques of early-warning-signal studies in other fields (Boettiger and
Hastings 2012; Dakos et al. 2012).

![The benchmark. (a) The three restoring families: the softening curve with its escape angles (dashed red), the constant bias moment that shifts the biased family's equilibria, and the parametric family's stiffness modulation envelope (dotted). (b) Two 600-second lives of the same ship in the same sea state, different seeded seas: the capsizing life ends during a wave group in its first minute, while its twin survives larger excursions for the full record. The encounter, not the general severity, decides the moment.](figs/fig_model.png)

## 3. Five warning methods

All five methods consume identical windows and emit a scalar score which is thresholded into
alarm episodes (three consecutive crossings to open; refractory closure). No method receives wave
elevation, spectrum, family identity, or sea-state input.

(a) A temporal convolutional network of 2,969 parameters trained on the horizon labels, in the
spirit of learned early-warning classifiers (Bury et al. 2021); neural roll prediction has
in-community precedent (Míguez González et al. 2023). (b) Rolling variance and lag-one
autocorrelation with Kendall trend statistics, the classical critical-slowing-down indicators
(Dakos et al. 2012), whose rate-of-change and noise-model limits are themselves documented
(Radhakrishnan et al. 2025; Layritz et al. 2025). (c) A roll-power adaptation of the double-Weibull generalized
likelihood-ratio detector of Galeazzi et al. (2013, 2015). (d) The split-time critical
roll rate (Belenky et al. 2024), computed in closed form from the known restoring parameters and
the measured state, with the separatrix line extrapolated between threshold crossings; to our
knowledge this quantity has not previously been operated as a continuously evaluated
online alarm statistic, and it functions here as a zero-training physics baseline. (e) The
neighbor-count score of Story (2009), reimplemented faithfully in the (roll, roll-rate) plane
under strictly causal normalization, with the thesis's fixed flag retained as one point on a full
operating curve; phase-space identification of dangerous ship states has since been developed
independently in the Spyrou school (Kontolefas and Spyrou 2020).

## 4. Evaluation protocol and the audit

Metrics are episode sensitivity, false episodes per exposure hour, and lead time, with
trajectory-block bootstrap intervals and fixed campaign weights; exposure ends at capsize; an
episode overlapping the pre-capsize horizon is event-associated. Final thresholds for the
baseline detector experiments were selected on calibration data and frozen before the final
corrected rescoring; those development-test blocks had already been accessed during method
correction, so the baseline results are corrected retrospective evaluations, not one-shot
tests. The later split-time, tangent-exponent, wave-field, and embedded-group experiments used
separately recorded preregistered fresh-data procedures, detailed with each result.

The protocol was not followed from the outset, and the failure is reported as a result. The
study's first operating points were obtained by scanning test-set metrics for the lowest
false-episode rate at 90 percent sensitivity or better, which is threshold selection on test
outcomes. Reported false-episode rates under test-informed selection were several times lower
than the calibration-frozen rates reported after correction; because the affected evaluations
differ in population as well as threshold provenance, we report the effect qualitatively rather
than as a single controlled factor. One guarded holdout evaluation was additionally invalidated
by threshold selection on the holdout labels, and its artifacts are retained as historical
records only. The corrected protocol is encoded in guard tests, and the acceptance run recorded
with the validation artifact passes the full suite with none skipped.
We suggest that reported operating points in the learned motion-prediction literature should be
read with this mechanism in mind wherever threshold provenance is not stated.

Because the study accumulated experiments with different data-access histories, Table 1 states,
for each, the primary question, the data it was evaluated on, how its operating choices were
selected, what its intervals mean, and its preregistered verdict. Throughout Section 5 each
experiment is named by what it measures; the parenthesized labels (D1 through D5, U1, H1, F1,
W1, D4b) are the repository identifiers under which every artifact, preregistration, and dated
addendum is filed.

| Experiment | Primary question | Data access | Selection rule | Intervals | Verdict |
|---|---|---|---|---|---|
| D1 pooled operating points | sensitivity vs. false-episode cost | corrected development-test rescoring | calibration-frozen thresholds | trajectory-block bootstrap | descriptive |
| D2 threshold transfer | does a frozen threshold transfer across families | corrected development-test rescoring | calibration-frozen thresholds | trajectory-block bootstrap | failed at fixed sensitivity |
| D3 bandwidth sweep | does ranking survive broadband forcing | corrected development-test rescoring | preregistered 10% operating gate | trajectory-block bootstrap | AUC held; operating gate missed |
| D4 false-alarm character | what false episodes coincide with | mixed-generation historical diagnostic | envelope criterion, predeclared | none (descriptive) | descriptive; no matched null |
| D5 within-regime timing | does motion time capsize inside the severe regime | corrected development-test rescoring | preregistered 0.58 trigger | trajectory-block bootstrap | chance-level |
| U1/U1-r2 online split-time | is the online decomposition calibrated | preregistered fresh test | preregistered capture criterion | parameter-uncertainty capture | failed; error localized |
| H1 offline-calibrated hybrid | is the offline-calibrated form calibrated | preregistered fresh test, one shot | preregistered capture and kill criteria | parameter-uncertainty capture diagnostic; reliability MAE | failed capture |
| F1 tangent exponents | do finite-time tangent statistics time capsize | preregistered, oracle and causal | preregistered ablation | trajectory-block bootstrap; survivor-conditioned | chance-level for tested family |
| W1 wave-field audit | is the synthetic seaway statistically sound | preregistered fresh wave-only seeds | preregistered gates | predictive | passed except one continuous-Rice gate |
| D4b embedded groups | how is predictability split between entry state and group | preregistered fresh preludes, one shot | preregistered predictions, frozen library | prelude-clustered percentile bootstrap | C5 composition failed; C6 evaluated |
Table: The experiment ledger. Every experiment's primary question, the data it was evaluated on, how its operating choices were selected, what its intervals mean, and its verdict. Confirmatory preregistered gates are distinguished from descriptive diagnostics; the D4 row is descriptive on both counts.

## 5. Results

**Ranking under changing forcing bandwidth (D3).** Classical critical-slowing-down indicators
depend on narrow-band structure, so the first question is whether any motion score ranks risk
once that structure is removed. Five campaigns vary the JONSWAP peak-enhancement factor from 30,
a nearly monochromatic sea, down to 1.0, fully broadband; every method scores the same windows,
and window AUC measures how well each score ranks the intervals that end badly above those that
do not. The fixed-window primary network's window AUC was 0.883 to
0.913 at every peak enhancement from 30 down to 1.0, while the classical trend indicators collapsed on ramped
campaigns. Ranking skill from motion history survives broadband forcing. The predeclared
operating-cost criterion at the broadband end (a 10 percent false-episode improvement) was missed
at 7.6 percent (14.061442 versus 15.212539 false episodes per hour), and the operating-cost claim
is reported as inconclusive: ranking skill and a deployable threshold are different assets
(Figure 2).

![Ranking survives bandwidth. Window AUC on the five forcing-bandwidth campaigns, corrected record: the fixed-window primary network holds 0.883 to 0.913 from narrow-band forcing down to fully broadband, while the classical indicators and physics scores range from near chance to well below it on these ramped campaigns — the fixed-window scoring that is honest about information also strips the trends those baselines rely on.](../../results/d3_bandwidth_skill_v02.png)

**Operating points at frozen thresholds (D1).** Ranking skill is not an alarm. Here each
method's score stream is thresholded into alarm episodes — three consecutive crossings open an
episode, a refractory period closes it — with the threshold chosen on calibration data and
frozen, and the resulting alarm system is charged its full operating cost: episode sensitivity
against false episodes per exposure hour on the pooled rare-event campaigns. At
calibration-frozen thresholds the fixed-window primary network attained 91.00 percent sensitivity at 13.409 false
episodes per hour; the classical indicators reached 100 percent sensitivity only at a
near-always-on threshold costing
21.4 per hour; the physics margin and the 2009 neighbor score occupied the same
high-sensitivity, high-cost region. No method reached 90 percent sensitivity at a false-episode
rate below several per hour (Figure 3).

![The pooled operating curves, corrected record: capsize sensitivity against declustered false episodes per exposure hour at calibration-frozen thresholds. The network variants dominate the low-cost region; every classical and physics score buys sensitivity only near the always-on corner. Even the best curve pays several false episodes per hour at 90 percent sensitivity.](../../results/d1_operating_curves_v02.png)

**Threshold transfer across failure modes (D2).** A deployed system cannot know which failure
mode the ship will face, so a threshold chosen on one pair of restoring families is applied
unchanged to the held-out third, in rotation, and judged at the same fixed sensitivity target.
The fixed-window CNN reached 99.15 percent, 90.83 percent, and
76.21 percent test sensitivity across the softening, parametric, and biased holdouts. Two
rotations met 90 percent, but only near always-on alarm cost; the biased holdout failed. No
deployable transfer was established.

**Timing inside the severe regime (D5).** The two clocks of the Introduction are separated by
construction: evaluation is restricted to trajectories already washed into the established
severe regime, where the slow vulnerability signal has saturated, so any remaining skill must
come from reading the fast clock — which particular encounter will be terminal. After the full
wash-in, the fixed-window CNN's
orientation-independent AUC was 0.556, below the 0.58 trigger; cumulative-online CNN reached
0.538 and classical scores remained near chance. Motion-only scores carried no usable information
about which encounter would be terminal inside the established severe regime.

**The character of false alarms (D4).** What is a false alarm made of? Because the seaway is
synthetic and fully known, every false episode can be checked for coincidence with a
high-envelope wave group, defined by a Hilbert-envelope exceedance criterion held for a minimum
run of cycles. Between 75 and 88 percent of nominally false episodes
coincided with such groups, which the vessel survived. This is a coincidence proxy, not the
critical-wave-groups method proper (Themelis and Spyrou 2007; Anastopoulos et al. 2016). The
pooled range also mixes artifact generations: the danger-margin row is v0.2-regenerated while
the remaining rows are v0.1-corrected, as the repository's supersession tables record. The
figure is descriptive, since no matched null was
tested and groups are common in a narrow-band sea; its qualitative content is that outcome-based
false-positive accounting charges the detectors for hazard exposure the vessel happened to
survive, the base-rate structure anticipated in Boettiger and Hastings (2012).

**Reference comparators.** To locate the learned methods, three references were scored on a
design-balanced window set: a restart comparator that re-simulates from the window's exact end
state under independent future forcing, an upper reference for state information alone; a
gradient-boosted tree on engineered features, among them an online stiffness estimate; and the
fixed-window network itself. The exact-state independent-
future restart reached AUC 0.850, fixed-window XGBoost 0.723, and fixed-window CNN 0.652; a
protocol-time comparator also exceeded the network. Two qualifications apply: the restart
comparators replace the realized forcing and are therefore reference points rather than bounds,
since the temporally correlated seaway leaves encounter information in the motion history; and
part of the network's advantage on natural test populations evidently derived from population
structure rather than state inference. The headroom that exists lies in state inference, and
simple tools claimed it first.

**Operating the split-time decomposition onboard (U1, H1).** The split-time method writes the
capsize rate as a product: the rate at which roll upcrosses an intermediate threshold, times
the probability that an upcrossing goes critical. Three preregistered experiments operated that
product as a continuously updated onboard estimate computed from measured motion alone: one
with the conditional estimated online from the observed exceedance tail (U1), one attributing
the resulting error term by term against measured quantities (U1-r2), and one with the
conditional calibrated offline from design-stage data and multiplied onboard by the measured
crossing rate (H1). Each was judged on fresh test data by the split-time program's own
confidence-interval-capture standard: does the estimate's interval capture the realized capsize
count of each campaign? The
online-estimated form failed its capture criterion, and a predeclared comparison found the
upcrossing rate alone more reliable than the full decomposition, so the corresponding kill
criterion fired. A measured attribution then localized the error: with the empirical
terminal-crossing probability substituted for the modeled one, the decomposition predicted 68
events against 71 realized, while the closed-form criticality model (the unforced
piecewise-linear critical roll rate with an exponential tail) overpredicted by 4.8-fold; a
motion-derived estimate of the forced correction made the error larger, not smaller.
A final hybrid, with the conditional calibrated offline from design-stage data and multiplied
onboard by the measured crossing rate, also failed its capture and transfer criteria. A timing
audit supplied the mechanism: in 221 of 240 heralded capsizes (92 percent; 19 exceptions), no
ten-second emission opportunity lay between the retained terminal crossing and the recorded
capsize — our reading is that the severity information the conditional carries arrives too
late, on this emission grid, to act on. Two methodological
qualifications attach: fresh test slices of 30 to 60 expected events realized counts 21 to 37
percent from expectation, and the capture intervals carried parameter uncertainty without
realization noise — a frozen parameter-uncertainty diagnostic rather than calibrated predictive
coverage — so the capture criterion on slices of this size is severe even for a
correctly calibrated estimator; the reliability errors (0.039848 for the hybrid, 0.042806 for the
rate-only map, and 0.242303 for a variance-based map) are the fairer calibration comparison. All
verdicts were preregistered and stand as recorded. Figure 4 shows the reliability comparison:
the hybrid and rate-only maps had similar pooled reliability error and both outperformed the
variance map, but no method passed the campaign count-capture criterion, and neither form was
validated as a calibrated operational estimator.

![Reliability of the offline-calibrated hybrid and its comparators on fresh test data (H1): predicted capsize probability per bin against the realized fraction. All three methods failed campaign count capture. The hybrid and rate-only maps show similar pooled reliability error (0.0398 and 0.0428) and both outperform the variance map (0.2423); no method was validated as a calibrated operational estimator.](../../results/h1_reliability_h1.png)

**The exponent's clean trial (F1).** The 2009 thread deserved one experiment its era could not
run: the finite-time Lyapunov exponent computed exactly, from the model's analytic Jacobian,
with tangent perturbations receiving the identical realized forcing (the standard construction
of Babaee et al. 2017). One preregistered ablation evaluated the margin level, its closure
rate, time to closure, energy reserve and depletion, level-plus-rate combinations, the
instantaneous strain normal to the escape boundary, and generic and escape-directed finite-time
exponents, in both an operational causal setting and an oracle setting with the true state and
tangent map. On the fully post-step severe-regime geometry, every statistic scored between
0.501 and 0.507 — including the acausal true-tangent exponents, which use information from the
future itself. The conditional plan to audit the 2009 estimator's implementation was therefore
retired by its own trigger: within the tested family of statistics, the exactly computed
tangent quantities carried no timing information, so any historical implementation defect was
immaterial to the thesis's conclusion. The estimand deserves exactness. The evaluated
population is 27,290 severe-regime windows (19,202 nonterminal, 8,088 terminal); finite-time
scores are evaluated only on windows whose trajectory survives the tangent horizon, which
censors 191, 419, and 1,018 terminal windows (and no nonterminal ones) at horizons of one,
two, and five natural periods — a censoring that preferentially removes the fastest events.
The generic statistic is the finite-time maximum singular value of the tangent propagator
$\Phi$; the escape-directed statistic is the covector projection $\lVert n_0^{T} \Phi \rVert$
onto the initial-state escape normal, normalized over the horizon — an escape-normal
finite-time tangent amplification, not the conventional singular-value exponent (Shadden et
al. 2005). On this survivor-conditioned, fully post-step geometry, none of the tested
finite-time tangent statistics discriminated terminal from nonterminal windows; no
information-theoretic ceiling is computed or claimed.

**The other side of the decomposition (D4b).** A final experiment asked the question every
negative above implies: if the arriving wave group were known, how much would it decide? A
frozen library of six envelope-mined group shapes was embedded into hundreds of independent
irregular preludes, in the natural-initial-condition manner of Silva and Maki (2021, 2024;
extended to free-running vessels in Silva and Maki 2026), so that
the vessel and its response history arrive at each group naturally; group heights were swept
across the critical range and monotone response maps fitted per entry-state stratum. Evaluated
once on held-out preludes (7,164 trials, 580 capsizes), capsize within the group was predicted
at AUC 0.513 [0.503, 0.524] from the vessel's entry state alone — near-chance, statistically
above one half but operationally negligible — at 0.934
[0.930, 0.938] from the group's engineered descriptors alone, and at 0.938 [0.934, 0.941] from
both, with intervals from prelude-level bootstrap resampling. The three predictors are
fixed-penalty L2-regularized logistic models on entry-state features, group descriptors, and
their union, fitted on calibration preludes and evaluated once on held-out preludes; over the
six-shape library the group and combined designs are rank-deficient, so individual
coefficients are not identifiable, though in-support predictions are stable under the fixed
penalty. The monotone response maps serve the rate composition below, not these AUCs. The
preregistered expectation
that the combination would substantially exceed either input was honestly not confirmed: the
point estimate rises by 0.004 once the group is known, a difference for which no paired
interval is reported. The estimand deserves precision: this is
a balanced design over six frozen shapes and six prescribed heights, in-support (shapes and
heights are not held out; prelude seeds are), for one softening-family configuration in beam
seas, and the group descriptors include the imposed height — the experimentally manipulated
driver. Within that design, group descriptors dominate the entry state; no universal division
of predictability follows. A companion
rate composition over the library's naturally occurring groups failed its capture criterion in
the informative direction: the composition predicted zero capsizes while direct counting on
matched campaigns recorded 39 in 1,500 trials — every library class fell below its critical
height. This six-class implementation failed coverage of the direct capsize mechanism; the
frozen artifacts do not identify which omitted representation — sequences outside the mined
classes, the entry-state stratification, or the fitted boundary — is responsible, and no
conclusion about other libraries or parameterizations follows. Broadband group representations
with initial-condition variability (Hafezi et al. 2026; cf. Anastopoulos and Spyrou 2019,
2023) indicate what a broader library must include. One numerical limitation attaches: a
step-halving audit on a frozen 128-seed subset changed one capsize classification, so exact
counts and rates are grid-conditioned rather than grid-independent constants, though no
headline comparison could be reversed by a sensitivity of that size. A preregistered
wave-field audit (W1) preceded these runs: the periodic record's
recurrence begins beyond every labeled lag, a check the self-repetition discussion of Umeda et
al. (2025) motivates; elevation upcrossing rates match the Rice (1945) formula
in a passing-rate sense, though one preregistered continuous-Rice mean gate failed as a
consequence of the fixed-energy conditioning; and that conditioning of the
deterministic-amplitude ensemble is stated as part of the estimand.

A companion study of conformalized forecast intervals on the same benchmark (following Romano,
Patterson and Candès 2019; Gibbs and Candès 2021; sequential and nonexchangeable variants in
Zaffran et al. 2022, Xu and Xie 2023, and Barber et al. 2023) found stationary marginal coverage accurate to
0.75 percentage points in the mean, while every online adaptation scheme tested failed a
predeclared joint rolling-coverage and alarm-cost criterion through the sea-state step. Those
results are reported in the project record and not further discussed here.

## 6. Discussion

The results organize into a two-clock description. The slowly varying stability state, the
eroding restoring that determines whether the vessel is in danger, is observable from motion
history: ranking skill survives every forcing bandwidth, and the most informative engineered
feature is in effect an online stiffness estimate, a quantity this community has long extracted
from the roll period (Terada et al. 2016; Míguez González et al. 2017; full-scale voyage
evidence in Higo et al. 2025). For the tested family of
motion-only statistics, the terminal encounter, the particular wave group that converts
vulnerability into capsize, was not recovered once the severe regime was established; the
within-regime result is chance-level for learned, classical, and physical scores alike, consistent
with the stochastic-Melnikov view that a deterministic separatrix loses its predictive meaning
under random forcing (Frey and Simiu 1993). The practical reading is that motion-only systems should be understood, and
evaluated, as vulnerability monitors rather than event predictors, a role compatible with the
operational measures framework of the second-generation criteria (IMO 2020) rather than with an
alarm bell.

Read against the split-time programme, the results locate both the value and the limit of
operating its metric onboard. As a zero-training detector the closed-form critical roll rate was
sensitivity-dominant and cost-competitive with every learned alternative, which is at once a
tribute to the physics and a caution against claims for learned detectors that have not faced it
as a baseline. Its one systematic failure, degradation on stability-erosion campaigns where the
assumed restoring becomes stale, identifies the missing component for operational use: online
estimation of the current restoring state, whose feasibility the comparator results demonstrate,
the roll period alone recovering most of the available headroom; full-scale data-driven
parameter updating is developing in exactly this direction (Takami et al. 2026). The within-regime chance result
then explains why the engineering-fidelity metric must re-simulate with the waves known: the
future-dependent quantity used by the motion perturbation method was not recovered by the tested
motion-only statistics. The U1 and H1 experiments made that boundary quantitative from
inside the method itself. Among the tested components, the crossing rate — an observable of the
slowly varying stability state — retained the more encouraging qualitative reliability pattern,
but the rate-only map failed every campaign count-capture check and was not validated as a
calibrated operational estimator; the severity factor, which would time the event, was in 92
percent of heralded capsizes not emitted with a ten-second opportunity before the recorded
event. The two-clock description is therefore not an external criticism of the split-time
programme but a property its own decomposition exhibits when operated online: the observable
factor reads the slow clock, while the tested online severity factor requires future
re-simulation.

The false-alarm finding bears on how such monitors should be judged, though it supports a
hypothesis rather than a conclusion. The high overlap between false episodes and high-envelope
wave groups motivates hazard-conditioned evaluation, but without a prevalence-, duration-, and
regime-matched null it cannot distinguish hazard sensitivity from the background prevalence of
envelope groups in narrow-band forcing; whether conventional false-alarm accounting is biased,
and in which direction, awaits that matched-null experiment. If the overlap survives it,
metrics conditioned on hazard exposure, and alarm policies whose budgets are conditioned on
regime — designed within the bridge alert-management framework (IMO 2010), with decision
support under uncertainty developing in parallel (Louvros et al. 2025) — appear to us a more
productive direction than further detector refinement.

Finally, the audit deserves a place in the discussion rather than an apology. The evaluation
stack that produced the leaked operating points had predeclared kill criteria, confidence
intervals on every rate, frozen data splits, and guarded holdouts, and it still admitted
materially lower reported false-alarm rates through test-informed threshold selection; because
the affected evaluations differ in population as well as threshold provenance, the historical
record is not a paired counterfactual and no single optimism factor is reported. The mechanism
is not exotic and is not confined to this study. The community's evaluation practice for learned warning methods would benefit from
treating threshold provenance as a reportable property of every operating point.

## 7. Conclusions

On a synthetic benchmark validated against analytic limits and internal numerical checks,
spanning three stability-failure archetypes, motion-only warning methods rank capsize risk well
above chance at every forcing bandwidth tested, fail to
transfer operating thresholds across failure modes, and discriminate event timing at chance
inside an established severe regime; the majority of their false alarms coincide, in a
mixed-generation diagnostic without a matched null, with high-envelope wave-group episodes. A
test-selected evaluation had earlier reported materially lower false-alarm rates for the
leading detector; the affected evaluations differ in population as well as threshold
provenance, so the effect is reported qualitatively. The split-time experiments
sharpen the same conclusion from inside the method: the upcrossing rate retained the more
encouraging reliability pattern among the tested components, though no tested form passed a
count-capture check, while the severity term that would time the event was, in 92
percent of heralded capsizes, not emitted with a ten-second opportunity before the recorded
event. The natural-initial-condition experiment then
argues the same point from the other side: within its balanced single-configuration design,
with the arriving group's engineered description fully known, capsize within the group is
predicted at AUC 0.934 while the vessel's entry state raises the point estimate by 0.004. That motivates, but does not
price, a forward-sensing experiment: what was tested is a complete, noiseless, exactly aligned
episode representation, and whether any realizable sensor recovers a useful fraction of it —
under range limits, directional ambiguity, and estimation error (Nielsen et al. 2024; Lee and
Kim 2025) — is the proposed next study. Further progress on the timing
question appears to require information about the incident wave field rather than refinement of
motion-only architectures; shipboard sea-state and wave-field estimation is an active
literature (Nielsen 2017; Nielsen et al. 2024; Lopac et al. 2026), with real-time wave-elevation
reconstruction from measured motion now demonstrated in simulation and basin conditions
(Zhang et al. 2026), and the benchmark's frozen
splits and audited protocol provide the evaluation discipline for that study, though an
encounter-sensing experiment would require model extensions the benchmark does not yet
contain.

## Declarations

**Data availability.** The benchmark code, campaign definitions, preregistrations, and the
complete numerical record, including corrected and superseded results retained as immutable
history, will be public at github.com/wrobstory/rahola on publication.

**Declaration of competing interest.** The author declares no competing financial interests or
personal relationships that could have influenced this work.

**CRediT author statement.** W. R. Story: conceptualization, methodology, software,
validation, formal analysis, investigation, data curation, writing, visualization.

**Funding.** This research received no external funding.

## Acknowledgments

The benchmark code, campaign definitions, and the complete numerical record, including the
pre-audit results retained as historical artifacts, are available from the author.

## References

<!-- Verified against primary records 2026-08-03; Tier-1 additions verified against
     publisher/DOI records and the official ISSW proceedings PDFs 2026-08-08 (ISSW page ranges
     confirmed against the proceedings; Míguez González et al. 2017 ends p. 229). Remaining
     flags: Themelis & Spyrou end page (181-204 vs -206); Nielsen (2017) details from secondary
     sources; McCue & Troesch (2004) page range unavailable in the public workshop record. -->

- Anastopoulos, P. A., and Spyrou, K. J. (2019). Evaluation of the critical wave groups method in calculating the probability of ship capsize in beam seas. *Ocean Engineering* 187, 106213.
- Anastopoulos, P. A., and Spyrou, K. J. (2023). Extrapolation of ship capsize probability over significant wave height: foundation on wave groups theory. *Ocean Engineering* 281, 114766.
- Anastopoulos, P. A., Spyrou, K. J., Bassler, C. C., and Belenky, V. (2016). Towards an improved critical wave groups method for the probabilistic assessment of large ship motions in irregular seas. *Probabilistic Engineering Mechanics* 44, 18–27.
- Babaee, H., Farazmand, M., Haller, G., and Sapsis, T. P. (2017). Reduced-order description of transient instabilities and computation of finite-time Lyapunov exponents. *Chaos* 27, 063103.
- Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics* 51(2), 816–845.
- Bačkalov, I., Bulian, G., Rosén, A., Shigunov, V., and Themelis, N. (2016). Improvement of ship stability and safety in intact condition through operational measures: challenges and opportunities. *Ocean Engineering* 120, 353–361.
- Belenky, V. L., and Sevastianov, N. B. (2007). *Stability and Safety of Ships: Risk of Capsizing*, 2nd ed. SNAME, Jersey City.
- Belenky, V., Weems, K. M., Lin, W.-M., Pipiras, V., and Sapsis, T. P. (2024). Estimation of probability of capsizing with split-time method. *Ocean Engineering* 292, 116452.
- Belenky, V., Weems, K., and Lin, W.-M. (2016). Split-time method for estimation of probability of capsizing caused by pure loss of stability. *Ocean Engineering* 122, 333–343.
- Boettiger, C., and Hastings, A. (2012). Early warning signals and the prosecutor's fallacy. *Proceedings of the Royal Society B* 279(1748), 4734–4739.
- Bulian, G., and Francescutto, A. (2004). A simplified modular approach for the prediction of the roll motion due to the combined action of wind and waves. *Proc. IMechE Part M: Journal of Engineering for the Maritime Environment* 218(3), 189–212.
- Bulian, G., and Francescutto, A. (2011). Effect of roll modelling in beam waves under multi-frequency excitation. *Ocean Engineering* 38(13), 1448–1463.
- Bury, T. M., Sujith, R. I., Pavithran, I., Scheffer, M., Lenton, T. M., Anand, M., and Bauch, C. T. (2021). Deep learning for early warning signals of tipping points. *PNAS* 118(39), e2106140118.
- Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S., Livina, V., Seekell, D. A., van Nes, E. H., and Scheffer, M. (2012). Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data. *PLoS ONE* 7(7), e41010.
- Falzarano, J. M., Shaw, S. W., and Troesch, A. W. (1992). Application of global methods for analyzing dynamical systems to ship rolling motion and capsizing. *International Journal of Bifurcation and Chaos* 2(1), 101–115.
- Frey, M., and Simiu, E. (1993). Noise-induced chaos and phase space flux. *Physica D* 63(3–4), 321–340.
- Galeazzi, R., Blanke, M., and Poulsen, N. K. (2013). Early detection of parametric roll resonance on container ships. *IEEE Transactions on Control Systems Technology* 21(2), 489–503.
- Galeazzi, R., Blanke, M., Falkenberg, T., Poulsen, N. K., Violaris, N., Storhaug, G., and Huss, M. (2015). Parametric roll resonance monitoring using signal-based detection. *Ocean Engineering* 109, 355–371.
- Gibbs, I., and Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems* 34, 1660–1672.
- Glotzer, D., Pipiras, V., Sapsis, T. P., and Belenky, V. (2024). Distributions and extreme value analysis of critical response rate and split-time metric in nonlinear oscillators with stochastic excitation. *Ocean Engineering* 292, 116538.
- Hafezi, S., Gong, X., and Pan, Y. (2026). Efficient estimation of temporal exceeding probability for ship responses in broadband wave fields. *Applied Ocean Research* 166, 104898.
- Hasselmann, K., et al. (1973). Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP). *Ergänzungsheft zur Deutschen Hydrographischen Zeitschrift*, Reihe A(8°), Nr. 12.
- Higo, Y., Chikamori, R., Hashimoto, H., Yamamoto, Y., Masutani, Y., and Matsuda, A. (2025). Estimation of the natural roll period of a large containership from ship motion observed during voyages. *Proceedings of the 20th International Ship Stability Workshop*, Chania, 139–147.
- IMO (2010). *Adoption of Performance Standards for Bridge Alert Management*. Resolution MSC.302(87), International Maritime Organization, London.
- IMO (2020). *Interim Guidelines on the Second Generation Intact Stability Criteria*. MSC.1/Circ.1627, International Maritime Organization, London.
- IMO (2022). *Explanatory Notes to the Interim Guidelines on the Second Generation Intact Stability Criteria*. MSC.1/Circ.1652, International Maritime Organization, London.
- Joo, H., Song, B., Park, K., Kwon, K., and Im, T. (2026). Uncertainty-aware short-horizon warning of large-inclination exceedance in small fishing vessels: a simulation-based multi-model benchmark. *Journal of Marine Science and Engineering* 14(13), 1195.
- Kontolefas, I., and Spyrou, K. J. (2020). Probability of ship high-runs from phase-space data. *Journal of Ship Research* 64(1), 81–97.
- Layritz, L. S., Rammig, A., Pavlyukevich, I., and Kuehn, C. (2025). Early warning signs for tipping points in systems with non-Gaussian alpha-stable noise. *Scientific Reports* 15, 13758.
- Lee, J., and Kim, Y. (2025). Application of spatiotemporal wave field-based neural network for predicting parametric roll motions. *Ocean Engineering* 342, 122845.
- Lopac, N., Severinski, K., Palaić, D., and Lerga, J. (2026). Machine learning for ship-motion-based sea-state estimation. *Ocean Engineering* 362, 126362.
- Louvros, P., Stefanou, E., Htein, N. M., Boulougouris, E., and Vassalos, D. (2025). Uncertainty-aware decision support for real-time damage stability assessment in maritime emergencies. *Proceedings of the 20th International Ship Stability Workshop*, Chania, 269–277.
- Mak, B., Scholcz, T. P., and van 't Veer, R. (2025). Development of an onboard machine learning based early warning system for dynamic ship stability loss after damage. *Proceedings of the 20th International Ship Stability Workshop*, Chania, 263–268.
- McCue, L. S., and Troesch, A. W. (2004). Use of Lyapunov exponents to predict chaotic vessel motions. *Proceedings of the 7th International Ship Stability Workshop*, Shanghai.
- McCue, L. S., and Troesch, A. W. (2006). A combined numerical–empirical method to calculate finite-time Lyapunov exponents from experimental time series with application to vessel capsizing. *Ocean Engineering* 33(13), 1796–1813.
- Míguez González, M., Bulian, G., Santiago Caamaño, L., and Díaz Casás, V. (2017). Towards real-time identification of initial stability from ship roll motion analysis. *Proceedings of the 16th International Ship Stability Workshop*, Belgrade, 221–229.
- Míguez González, M., Díaz Casás, V., López Peña, F., and Pérez Rojas, L. (2023). On the application of artificial neural networks for the real time prediction of parametric roll resonance. In Spyrou, K. J., et al. (eds.), *Contemporary Ideas on Ship Stability: From Dynamics to Criteria*, Springer, 335–349.
- Nayfeh, A. H., and Mook, D. T. (1979). *Nonlinear Oscillations*. Wiley-Interscience, New York.
- Nielsen, U. D. (2017). A concise account of techniques available for shipboard sea state estimation. *Ocean Engineering* 129, 352–362.
- Nielsen, U. D., Iwase, K., Mounet, R. E. G., and Storhaug, G. (2024). Uncertainty-associated directional wave spectrum estimation from wave-induced ship responses using machine learning methods. *Ocean Engineering* 313, 119543.
- Petacco, N., and Gualeni, P. (2020). IMO second generation intact stability criteria: general overview and focus on operational measures. *Journal of Marine Science and Engineering* 8(7), 494.
- Radhakrishnan, R., Pavithran, I., Livina, V., Kurths, J., and Sujith, R. I. (2025). Early warnings are too late when parameters change rapidly. *Scientific Reports* 15, 20256.
- Rahola, J. (1939). *The Judging of the Stability of Ships and the Determination of the Minimum Amount of Stability — Especially Considering the Vessels Navigating Finnish Waters*. Doctoral thesis, Technical University of Finland, Helsinki.
- Rice, S. O. (1945). Mathematical analysis of random noise. *Bell System Technical Journal* 24(1), 46–156.
- Romano, Y., Patterson, E., and Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems* 32, 3538–3548.
- Santiago Caamaño, L., Díaz González, I., and Míguez González, M. (2025). Application of a stability monitoring system to naval vessels. *Proceedings of the 20th International Ship Stability Workshop*, Chania, 149–156.
- Shadden, S. C., Lekien, F., and Marsden, J. E. (2005). Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows. *Physica D* 212(3–4), 271–304.
- Shigunov, V. (2023). Intact stability operational measures: criteria, standards and examples. *Ocean Engineering* 279, 114446.
- Shinozuka, M., and Deodatis, G. (1991). Simulation of stochastic processes by spectral representation. *Applied Mechanics Reviews* 44(4), 191–204.
- Silva, K. M., and Maki, K. J. (2021). Towards a computational fluid dynamics implementation of the critical wave groups method. *Ocean Engineering* 235, 109451.
- Silva, K. M., and Maki, K. J. (2024). Implementation of the critical wave groups method with computational fluid dynamics and neural networks. *Ocean Engineering* 292, 116468.
- Silva, K. M., and Maki, K. J. (2026). Towards a statistical validation of the critical wave groups method for free-running vessels in beam seas. *Ocean Engineering* 362, 126163.
- Story, W. R. (2009). *Application of Lyapunov Exponents to Strange Attractors and Intact & Damaged Ship Stability*. M.S. thesis, Aerospace and Ocean Engineering, Virginia Polytechnic Institute and State University, Blacksburg.
- Takami, T., Nielsen, U. D., Mounet, R. E. G., and Sasa, K. (2026). A data-driven scheme for updating of parameters for ship roll motion. *Ocean Engineering* 356, Part 2, 125346.
- Terada, D., Tamashima, M., Nakao, I., and Matsuda, A. (2016). Estimation of the metacentric height by using onboard monitoring roll data based on time series analysis. *Proceedings of the 15th International Ship Stability Workshop*, Stockholm, 209–215.
- Themelis, N., and Spyrou, K. J. (2007). Probabilistic assessment of ship stability. *SNAME Transactions* 115, 181–204.
- Umeda, N., Maruyama, Y., Belenky, V., and Weems, K. (2025). Discussion to self-repeating effect in reconstruction of irregular waves for direct stability assessment. *Proceedings of the 20th International Ship Stability Workshop*, Chania, 215–225.
- Umeda, N., Usada, S., Mizumoto, K., and Matsuda, A. (2016). Broaching probability for a ship in irregular stern-quartering waves: theoretical prediction and experimental validation. *Journal of Marine Science and Technology* 21(1), 23–37.
- Wandji, C., Shigunov, V., Pipiras, V., and Belenky, V. (2024). Benchmarking of direct counting approaches. *Ocean Engineering* 296, 116649.
- Weems, K., Belenky, V., Campbell, B., and Pipiras, V. (2023). Statistical validation of the split-time method with volume-based numerical simulation. In Spyrou, K. J., Belenky, V. L., Katayama, T., Bačkalov, I., and Francescutto, A. (eds.), *Contemporary Ideas on Ship Stability: From Dynamics to Criteria*, Springer, Cham, 225–243.
- Xu, C., and Xie, Y. (2023). Sequential predictive conformal inference for time series. *Proceedings of Machine Learning Research* 202, 38707–38727.
- Zaffran, M., Feron, O., Goude, Y., Josse, J., and Dieuleveut, A. (2022). Adaptive conformal predictions for time series. *Proceedings of Machine Learning Research* 162, 25834–25866.
- Zhang, Y., Huang, L., Jiang, F., Duan, W., Tang, L., and Ni, T. (2026). Real-time estimation of wave elevation and statistics from ship motions based on machine learning. *Ocean Engineering* 349, 124144.
