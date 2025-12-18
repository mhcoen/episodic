# SuperDialseg Leaderboard Audit Summary

Generated: 2025-12-18 04:17:38
Random seed: 42

## Claims Audited

- **Claim A** (dialseg711): TextTiling vs. CSM
  - ΔF1: +0.135, ΔW-F1: +0.220, ΔBOR: +2.13
  - Regime shift: Conservative → Aggressive

- **Claim B** (tiage): TextTiling vs. CSM
  - ΔF1: +0.128, ΔW-F1: +0.219, ΔBOR: +1.26
  - Regime shift: Conservative → Aggressive

- **Claim C** (dialseg711): TextTiling vs. Even
  - ΔF1: +0.277, ΔW-F1: +0.132, ΔBOR: +1.69
  - Regime shift: Balanced → Aggressive

## Key Observations

For runnable published methods, the analysis examines whether F1/W-F1 differences
align with differences in boundary density (BOR) and granularity regime.

- Claim A: Higher F1 method (TextTiling) does NOT have BOR closer to 1.0
- Claim B: Higher F1 method (TextTiling) does NOT have BOR closer to 1.0
- Claim C: Higher F1 method (TextTiling) does NOT have BOR closer to 1.0

## Non-Reproduced Methods

- **RoBERTa**: Requires training from scratch; not reproduced.
- **ChatGPT**: Requires proprietary API; not reproduced.

## Interpretation

This analysis does not claim invalidation of prior work. It only reports
whether observed F1/W-F1 differences are accompanied by shifts in BOR
and granularity regime.