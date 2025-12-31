# Alpha Tuning Comparison Record

This document records both result sets for the external methods evaluation,
demonstrating how F1-based threshold tuning affects boundary density.

## Dataset Differences

| Source | DialSeg711 N | TIAGE N | Alpha | Notes |
|--------|--------------|---------|-------|-------|
| Tuned (external_methods.csv) | 704 | 400 | Dev-tuned | Combined train+val splits |
| Fixed (leaderboard_reanalysis.csv) | 711 | 100 | 0.0 | Test split only |

## Tuned-Alpha Results (external_methods.csv)

Threshold tuned on dev set to maximize F1.

| Method | Dataset | N | W-F1 | BOR | F1 | Purity | Coverage | Regime |
|--------|---------|---|------|-----|-----|--------|----------|--------|
| TextTiling | DialSeg711 | 704 | 0.648 | **1.88** | 0.550 | 0.968 | 0.746 | Aggressive |
| CSM | DialSeg711 | 704 | 0.412 | **0.64** | 0.373 | 0.749 | 0.921 | Conservative |
| Random | DialSeg711 | 704 | 0.267 | 0.55 | 0.098 | 0.711 | 0.890 | Conservative |
| Even | DialSeg711 | 704 | 0.483 | 1.00 | 0.211 | 0.806 | 0.809 | Balanced |
| TextTiling | TIAGE | 400 | 0.612 | **2.04** | 0.445 | 0.949 | 0.693 | Aggressive |
| CSM | TIAGE | 400 | 0.615 | **3.97** | 0.346 | 0.976 | 0.454 | **Aggressive** |
| Random | TIAGE | 400 | 0.256 | 0.33 | 0.121 | 0.697 | 0.921 | Conservative |
| Even | TIAGE | 400 | 0.620 | 1.00 | 0.300 | 0.832 | 0.838 | Balanced |

## Fixed-Alpha Results (leaderboard_reanalysis.csv)

Fixed threshold alpha=0.0 (no tuning).

| Method | Dataset | N | W-F1 | BOR | F1 | Purity | Coverage | Regime |
|--------|---------|---|------|-----|-----|--------|----------|--------|
| TextTiling | DialSeg711 | 711 | 0.613 | **2.69** | 0.488 | 0.988 | 0.638 | Aggressive |
| CSM | DialSeg711 | 711 | 0.393 | **0.56** | 0.353 | 0.732 | 0.931 | Conservative |
| Even | DialSeg711 | 711 | 0.481 | 1.00 | 0.211 | 0.806 | 0.809 | Balanced |
| TextTiling | TIAGE | 100 | 0.628 | **1.84** | 0.434 | 0.933 | 0.717 | Aggressive |
| CSM | TIAGE | 100 | 0.409 | **0.58** | 0.306 | 0.760 | 0.909 | **Conservative** |
| Even | TIAGE | 100 | 0.633 | 1.00 | 0.276 | 0.824 | 0.826 | Balanced |

## Key Finding: CSM Regime Variation is a Tuning Artifact

| Dataset | CSM BOR (Tuned) | CSM BOR (Fixed) | Regime (Tuned) | Regime (Fixed) |
|---------|-----------------|-----------------|----------------|----------------|
| DialSeg711 | 0.64 | 0.56 | Conservative | Conservative |
| TIAGE | **3.97** | **0.58** | **Aggressive** | **Conservative** |

With fixed alpha=0, CSM is **conservative on both datasets** (BOR ~0.56-0.58).

The cross-dataset regime variation (Conservative on DialSeg711, Aggressive on TIAGE)
only appears when alpha is tuned to maximize F1. This demonstrates that F1-based
threshold tuning implicitly tunes for density matching.

## Pairwise Deltas (from fixed-alpha audit)

| Comparison | Dataset | Delta-W-F1 | Delta-BOR | Regime Shift |
|------------|---------|------------|-----------|--------------|
| TT vs CSM | DialSeg711 | +0.220 [0.202, 0.237] | +2.13 [2.07, 2.18] | Conservative -> Aggressive |
| TT vs CSM | TIAGE | +0.219 [0.155, 0.283] | +1.26 [1.11, 1.44] | Conservative -> Aggressive |
| TT vs Even | DialSeg711 | +0.132 [0.114, 0.150] | +1.69 [1.64, 1.74] | Balanced -> Aggressive |
