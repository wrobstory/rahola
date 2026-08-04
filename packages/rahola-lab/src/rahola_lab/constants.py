"""Frozen experimental constants shared by every Rahola prototype."""

from __future__ import annotations

from enum import StrEnum

# Operational alarm horizons: short and actionable, while spanning several rolls.
FORECAST_HORIZONS_S = (30.0, 60.0)
# Two minutes captures slow envelope modulation without using future motion.
FORECAST_HISTORY_S = 120.0
# 0.6*phi_v leaves escape margin while rejecting ordinary moderate roll.
ALARM_THRESHOLD_ESCAPE_FRACTION = 0.60
# Prototype #2 needs a long baseline for slow early-warning statistics.
EWS_WINDOW_PERIODS = 60.0
# Roughly 50 cycles preserves the planned bifurcation-warning horizon.
EWS_HORIZON_PERIODS = 50.0
# Five periods remove ambiguous near-misses from the negative class.
EXCLUSION_BUFFER_PERIODS = 5.0

# v0.2 intervals resample whole trajectories within campaigns, conditional on
# calibration-frozen policies. Freeze these values before v0.2 scoring.
TRAJECTORY_BOOTSTRAP_REPLICATES = 1_000
TRAJECTORY_BOOTSTRAP_SEED = 20_260_804

# v0.2 preprocessing freeze, declared before any v0.2 test scoring.
PRIMARY_NORMALIZATION_MODE = {
    "classical_ews": "physical",
    "neighbor_2009": "physical",
    "galeazzi_glrt": "physical",
    "danger_margin": "physical",
    "cnn": "fixed_window_causal",
    "xgboost": "fixed_window_causal",
}
REPORTED_NORMALIZATION_MODES = {
    "cnn": ("fixed_window_causal", "cumulative_online"),
    "xgboost": ("fixed_window_causal", "cumulative_online"),
}

# v0.2 forcing audit, frozen before the paired simulations.
FORCING_PREVALENCE_TOLERANCE = 0.01
FORCING_PAIRED_TRAJECTORIES_PER_CAMPAIGN = 512

# v0.2 established-regime geometry, frozen before campaign generation.
D5_V02_TRANSITION_S = 300.0
D5_V02_DURATION_S = 900.0
D5_V02_FIRST_ENDPOINT_S = 540.0
D5_V02_LAST_ENDPOINT_S = 700.0
D5_V02_AUC_LIMIT = 0.58
D5_V02_PREREGISTERED_PREDICTION = (
    "Prediction (preregistered): fully post-step within-regime discrimination will "
    "remain near chance (AUC below 0.58) for every motion-only method, consistent with "
    "the immediate-post-transition result it replaces."
)

# ACI is operationally unacceptable only if it exceeds both this absolute rate...
ACI_EXPLOSION_FPR_PER_HOUR = 2.0
# ...and four times the corresponding fixed-CQR episode rate.
ACI_EXPLOSION_FACTOR = 4.0
# Small predeclared grid spans slow adaptation through deliberately aggressive updates.
ACI_GAMMA_GRID = (0.001, 0.005, 0.01, 0.02, 0.05)
# Gibbs--Candes' published DtACI expert grid; aggregation removes manual gamma selection.
DTACI_GAMMA_EXPERTS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
# Calibration-only E3b grid for recent-score recalibration memory.
SLIDING_RECALIBRATION_WINDOWS = (25, 50, 100)
# Fitted-config bound brackets always-on through selective physics alarms.
DANGER_SCORE_THRESHOLDS_RAD_S = tuple(-1.75 + 0.025 * index for index in range(91))

# Prototype #2 detector grids are frozen before any development-test scoring.
EWS_SUBWINDOW_FRACTIONS = (0.20, 0.35, 0.50)
NEIGHBOR_RADIUS_GRID = (0.20, 0.35, 0.50)
CNN_GRID = (
    {"channels": (12, 24), "kernel_size": 9, "family_head_weight": 0.0},
    {"channels": (16, 32), "kernel_size": 7, "family_head_weight": 0.10},
)
DETECTOR_MATCHED_SENSITIVITY = 0.90
# "Materially above B1" requires both methods to retain the 90% calibration target on test,
# plus at least 10% lower FPR/h at their calibration-selected thresholds.
D2_MATERIAL_FPR_REDUCTION = 0.10
D3_MATERIAL_AUC_MARGIN = 0.02
D3_BROADBAND_VERDICT = (
    "If all motion-only detectors' skill collapses toward the broadband end, the "
    "encounter-driven objection to precursor-based warning is quantitatively confirmed for "
    "this system class."
)
D3_SURVIVAL_VERDICT = (
    "If skill survives at gamma=1.0 materially above the B1 floor, motion history contains "
    "precursor information beyond critical slowing down."
)
D3_INCONCLUSIVE_VERDICT = (
    "Broadband ranking skill remains above the B1 floor, but the predeclared operating-cost "
    "materiality criterion is not met; neither survival nor collapse is established."
)

# Historical Prototype #3 restart-comparison gap, frozen before test scoring.
CEILING_AUC_GAP = 0.03
ORACLE_ROLLOUTS = 200
PF_PARTICLES = 2_000
CEILING_WINDOWS_PER_CAMPAIGN = 2_000
CEILING_BOOTSTRAP_REPLICATES = 2_000
CEILING_BOOTSTRAP_SEED = 31_415
# Gate-open B1 grid and kills, frozen before any B1 calibration or test scoring.
GRAYBOX_AUXILIARY_WEIGHT_GRID = (0.25, 1.0)
GRAYBOX_TRANSFER_FPR_REDUCTION = 0.15
GRAYBOX_TRANSFER_ROTATIONS_REQUIRED = 2
GRAYBOX_STIFFNESS_MAE_LIMIT = 0.10
# Gate-open B2 protocol, frozen before downloading or scoring the checkpoint.
CHRONOS_CHECKPOINT = "amazon/chronos-t5-tiny"
CHRONOS_REVISION = "29d808298f1a62493e7b9a5e08529d0d930fa189"
CHRONOS_LICENSE = "Apache-2.0"
B2_FPR_IMPROVEMENT = 0.10
B2_D5_LEAKAGE_AUC = 0.58
B2_ADAPTATION_CAPSIZES = 20
B2_TRAJECTORIES_PER_CAMPAIGN = 128
B2_CONTEXT_SAMPLES = 256
B2_FROZEN_HEAD_EPOCHS = 25
B2_FINETUNE_EPOCHS = 1
B2_FINETUNE_MAX_WINDOWS = 1_024
# Wave-group stratification treats twice the elevation-envelope amplitude as
# instantaneous wave height. A critical run must exceed 0.75 Hs for 1.5 Tp.
WAVE_GROUP_HEIGHT_HS_FRACTION = 0.75
WAVE_GROUP_MIN_PERIODS = 1.5

# Disjoint 100k-wide ranges prevent phase reuse across statistical roles.
SEED_BLOCK_SIZE = 100_000


class SeedBlock(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"
    RESERVE = "reserve"
    RESERVE2 = "reserve2"


SEED_BLOCK_START = {
    SeedBlock.TRAIN: 0,
    SeedBlock.CALIBRATION: 100_000,
    SeedBlock.TEST: 200_000,
    # Frozen for Prototype #2 final evaluation; this task must never materialize it.
    SeedBlock.RESERVE: 300_000,
    # Prototype #3 final holdout and standing holdout for future automated search.
    SeedBlock.RESERVE2: 400_000,
}
