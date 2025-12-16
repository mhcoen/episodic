# Episodic topic detection and commitment

Internal note

## Purpose

Episodic needs topic boundaries to drive memory management actions such as segmenting history for retrieval and avoiding cross topic contamination. The topic detector is designed to be robust to transient digressions and to separate detection from commitment.

## Core design

The system separates three concerns.

1. **Scoring**
   A neural boundary scorer estimates whether a boundary exists around the current turn given a frozen reference window. This scorer is treated as a signal source, not as the decision logic.

2. **Triggering**
   An optional embedding based semantic drift signal serves as an early warning trigger to start investigating a possible change. Drift is never sufficient to commit a boundary.

3. **Commitment**
   A commitment policy turns a stream of scores into committed boundaries via a small state machine with evidence accumulation and abort logic.

## State machine

The operational machine has two states.

**State STABLE**
No active suspicion of a boundary.

**State SUSPECT**
A boundary is being evaluated. The reference context is frozen so that scores remain comparable across turns.

### Entry conditions for STABLE to SUSPECT

Enter SUSPECT when either condition holds.

**A. Neural trigger**
Neural boundary confidence exceeds the suspect threshold.

**B. Drift trigger**
Semantic drift exceeds the drift threshold. Drift trigger is treated as noisy, so it always requires confirmation before commit.

On entry to SUSPECT, the policy records:
- suspect_cause: neural or drift
- entry_confidence: the neural confidence at the moment SUSPECT is entered
- frozen_before: the pre boundary context window used as reference
- frozen_straddle: the boundary adjacent anchor message representing the last message of the old topic
- accumulated_evidence, low_confidence_streak, and peak confidence tracking. Peak confidence is updated over the course of the SUSPECT episode and records the maximum neural confidence observed. It is used by return_drop_ratio to detect relative confidence drops.

### Frozen reference invariant

While in SUSPECT, scoring is always computed against the frozen reference. Both the pre boundary context and the straddle anchor are held fixed. This preserves the scorer's training semantics and prevents the reference from sliding into the new topic.

### Evidence accumulation

Each turn in SUSPECT produces a neural confidence score computed against the frozen reference. The policy maintains an accumulated evidence statistic with decay. Conceptually, the policy asks whether the dialogue has persistently departed from the frozen old topic.

### Abort logic

If the dialogue appears to return to the old topic before commitment, the policy aborts the suspicion and returns to STABLE without emitting a boundary.

Return detection uses either:
- **Absolute return threshold**: abort if confidence drops below return_threshold after evidence is high enough to consider commitment
- **Relative drop detection**: abort if confidence falls sufficiently relative to the peak confidence observed during SUSPECT

### Commit logic

A boundary is committed when:
1. accumulated evidence reaches min_evidence, and
2. either
   a. bypass is allowed, or
   b. persistence is satisfied

Persistence is defined by commit_persistence, the number of consecutive turns that must maintain confidence above the return threshold after evidence exceeds min_evidence before commitment is allowed.

### Bypass rule: conditional cooldown

To avoid sacrificing recall while still handling digressions, the policy includes a bypass based on SUSPECT entry confidence.

- If suspect_cause is neural and entry_confidence is at least high_conf_commit_threshold, the policy can commit without waiting for additional persistence.
- Otherwise, the policy enforces a short persistence requirement and allows abort if return is detected during this cooldown.

This implements a two sided test. A strong initial boundary signal can commit immediately. Weaker signals require brief confirmation and can be canceled if the conversation reverts.

### Backdating

When a commit occurs, the emitted boundary is placed at the SUSPECT entry point, not at the later confirmation turn. This corrects for the common behavior of window based neural scorers that often detect topic commitment after the first initiating turn.

### Cause conditioning

Some thresholds are conditioned on suspect_cause.

- Drift triggered SUSPECT is treated as higher risk and can require stronger evidence and faster abort behavior than neural triggered SUSPECT.
- Neural triggered SUSPECT can be more permissive, relying on the scorer's contextual signal.

## Granularity versus stability

**Stability knobs** determine how the system handles transient shifts and when it commits.
- commit_persistence
- return_threshold and return_drop_ratio
- high_conf_commit_threshold
- drift trigger threshold and drift specific evidence or abort thresholds

**Granularity knobs** control how many boundaries are produced.
- base confidence thresholds for candidate boundaries
- minimum spacing or min_gap constraints
- min_evidence and evidence decay parameters

Operationally, stability should be tuned to avoid pathological false commits, then granularity should be chosen to match the downstream application cost profile.

## Evaluation artifacts and interpretation

The harness evaluates both segmentation style metrics and system metrics.

**Segmentation style metrics**
- W-F1, BOR, purity, coverage

**System oriented metrics**
- delay_mean and delay_coverage computed on matched boundaries
- time to SUSPECT, time to COMMIT
- churn and abort rates

Delay is only meaningful when reported with delay_coverage. A low delay_mean with low coverage is not informative.

## What is guaranteed by design

- Drift cannot directly commit a boundary. It can only trigger SUSPECT.
- Scores are comparable during SUSPECT because the scoring reference is frozen.
- Transient digressions can be aborted via return detection rather than requiring globally high evidence thresholds.
- Strong neural signals can commit quickly via entry confidence bypass without adding latency.

## What is intentionally application dependent

- The exact operating point for high_conf_commit_threshold defines a precision recall regime.
- Drift thresholds and cause conditioned evidence parameters define how aggressively the system investigates sharp shifts.
- Granularity targets depend on whether boundaries drive summarization, retrieval indexing, or eviction.
