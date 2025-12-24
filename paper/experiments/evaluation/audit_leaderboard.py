#!/usr/bin/env python3
"""
Audit SuperDialseg leaderboard claims under a granularity-aware (BOR) framework.

This script re-evaluates published methods from SuperDialseg Table 3 to analyze
whether F1/W-F1 differences align with boundary density (BOR) differences.

Claims audited:
- Claim A: TextTiling vs. CSM on DialSeg711
- Claim B: TextTiling vs. CSM on TIAGE
- Claim C: TextTiling vs. Even on DialSeg711

Non-reproduced methods (documented only):
- RoBERTa: Requires training from scratch; not reproduced.
- ChatGPT: Requires proprietary API; not reproduced.

Usage:
    python -m tacl.experiments.evaluation.audit_leaderboard
"""

import csv
import json
import sys
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RANDOM_SEED = 42
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

# Bootstrap CI configuration
# Bootstrap unit: dialogues (resample dialogues with replacement)
# CI method: percentile bootstrap (2.5%, 97.5%)
# For ΔBOR: resample dialogues, recompute BOR_A and BOR_B on resample, take difference
# For ΔW-F1: resample dialogues, recompute macro W-F1 for both methods, take difference
BOOTSTRAP_N_REPLICATES = 1000  # Number of bootstrap resamples (configurable)
BOOTSTRAP_SEED = 42  # Fixed random seed for reproducibility
BOOTSTRAP_CI_ALPHA = 0.05  # 95% CI (2.5%, 97.5% percentiles)

# Expected dataset paths
EXPECTED_PATHS = {
    "dialseg711": DATASETS_DIR / "dialseg711" / "segmentation_file_test.json",
    "tiage": DATASETS_DIR / "tiage" / "segmentation_file_test.json",
}

# Claims to audit
CLAIMS = [
    {"id": "A", "dataset": "dialseg711", "method1": "TextTiling", "method2": "CSM"},
    {"id": "B", "dataset": "tiage", "method1": "TextTiling", "method2": "CSM"},
    {"id": "C", "dataset": "dialseg711", "method1": "TextTiling", "method2": "Even"},
]


@dataclass
class EvalResult:
    """Evaluation result for one method on one dataset."""
    method: str
    dataset: str
    n_dialogues: int
    n_gold: int
    n_pred: int
    strict_f1: float
    w_f1: float
    bor: float
    purity: float
    coverage: float
    regime: str  # Conservative, Balanced, Aggressive


@dataclass
class EvalResultWithPerDialogue:
    """
    Evaluation result with per-dialogue data for bootstrap CI computation.

    Per-dialogue data is stored as lists aligned by dialogue index, enabling
    dialogue-level bootstrap resampling without re-running inference.

    For bootstrap ΔW-F1: resample dialogue indices, compute macro W-F1 on resample.
    For bootstrap ΔBOR: resample dialogue indices, recompute BOR = sum(pred_d) / sum(gold_d).
    """
    method: str
    dataset: str
    n_dialogues: int
    n_gold: int  # Total gold boundaries (dataset level)
    n_pred: int  # Total predicted boundaries (dataset level)
    strict_f1: float
    w_f1: float  # Macro-averaged W-F1 over dialogues
    bor: float  # Dataset-level BOR = n_pred / n_gold
    purity: float
    coverage: float
    regime: str
    # Per-dialogue data (lists of length n_dialogues)
    per_dialogue_wf1: List[float] = field(default_factory=list)
    per_dialogue_gold_count: List[int] = field(default_factory=list)
    per_dialogue_pred_count: List[int] = field(default_factory=list)

    def to_eval_result(self) -> EvalResult:
        """Convert to basic EvalResult (drops per-dialogue data)."""
        return EvalResult(
            method=self.method,
            dataset=self.dataset,
            n_dialogues=self.n_dialogues,
            n_gold=self.n_gold,
            n_pred=self.n_pred,
            strict_f1=self.strict_f1,
            w_f1=self.w_f1,
            bor=self.bor,
            purity=self.purity,
            coverage=self.coverage,
            regime=self.regime,
        )


def log(msg: str):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def validate_paths() -> bool:
    """Validate that all required dataset paths exist."""
    log("Validating dataset paths...")
    missing = []
    for name, path in EXPECTED_PATHS.items():
        if not path.exists():
            missing.append(f"  - {name}: {path}")

    if missing:
        log("ERROR: Missing dataset files:")
        for m in missing:
            print(m)
        return False

    log("All dataset paths validated.")
    return True


def load_dataset(dataset_name: str) -> List[Dict]:
    """
    Load dataset and convert to canonical format.

    Returns list of dicts with:
    - messages: List[Dict] with 'role' and 'content'
    - gold_boundaries: Set[int] - canonical boundary indices
    - num_messages: int
    """
    filepath = EXPECTED_PATHS[dataset_name]
    log(f"Loading {dataset_name} from {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    # Extract dialogues from nested structure
    dial_data = data.get("dial_data", {})
    dialogues_raw = dial_data.get(dataset_name, [])

    dialogues = []
    for dial in dialogues_raw:
        turns = dial.get("turns", [])

        # Convert to messages format
        messages = []
        for turn in turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("utterance", ""),
            })

        # Extract gold boundaries
        # segmentation_label=1 at turn t means boundary AFTER turn t
        # In canonical format: boundary at index t+1 (between t and t+1)
        gold_boundaries = set()
        for i, turn in enumerate(turns):
            if turn.get("segmentation_label", 0) == 1:
                # Boundary after this turn = at position i+1 (0-indexed turns)
                # But we need canonical between-message index
                boundary_idx = i + 1  # Position after message i
                if 1 <= boundary_idx < len(turns):
                    gold_boundaries.add(boundary_idx)

        dialogues.append({
            "messages": messages,
            "gold_boundaries": gold_boundaries,
            "num_messages": len(messages),
        })

    log(f"  Loaded {len(dialogues)} dialogues")
    return dialogues


def get_segmenter(method: str):
    """Get segmenter instance by name with default parameters (no tuning)."""
    from tacl.experiments.segmenters import (
        TextTilingSegmenter,
        CSMSegmenter,
        RandomSegmenter,
        EvenSegmenter,
    )

    if method.lower() == "texttiling":
        # Default alpha=0.0 (threshold = mean depth score)
        return TextTilingSegmenter(alpha=0.0)
    elif method.lower() in ("csm", "csm_nsp"):
        # Default alpha=0.0 (threshold = mean depth score)
        return CSMSegmenter(alpha=0.0)
    elif method.lower() == "random":
        # Fixed seed for reproducibility
        log(f"  Random seed: {RANDOM_SEED}")
        return RandomSegmenter(target_ratio=0.1, seed=RANDOM_SEED)
    elif method.lower() == "even":
        return EvenSegmenter(match_gold=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_windowed_metrics(
    gold: Set[int],
    pred: Set[int],
    window: int = 1
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 with tolerance window."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0

    matched_pred = set()
    matched_gold = set()

    for p in pred:
        for g in gold:
            if abs(p - g) <= window:
                matched_pred.add(p)
                matched_gold.add(g)
                break

    precision = len(matched_pred) / len(pred) if pred else 0.0
    recall = len(matched_gold) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def boundaries_to_segments(boundaries: Set[int], num_messages: int) -> List[Set[int]]:
    """Convert boundary positions to segment membership sets."""
    sorted_bounds = sorted(boundaries)
    segments = []
    prev = 0

    for bound in sorted_bounds:
        if bound > prev:
            segments.append(set(range(prev, bound)))
        prev = bound

    if prev < num_messages:
        segments.append(set(range(prev, num_messages)))

    return segments if segments else [set(range(num_messages))]


def compute_purity_coverage(
    gold_segments: List[Set[int]],
    pred_segments: List[Set[int]]
) -> Tuple[float, float]:
    """Compute segment purity and coverage."""
    if not gold_segments or not pred_segments:
        return 0.0, 0.0

    purities = []
    for pred_seg in pred_segments:
        if pred_seg:
            max_overlap = max(len(pred_seg & gold_seg) for gold_seg in gold_segments)
            purities.append(max_overlap / len(pred_seg))

    coverages = []
    for gold_seg in gold_segments:
        if gold_seg:
            max_overlap = max(len(gold_seg & pred_seg) for pred_seg in pred_segments)
            coverages.append(max_overlap / len(gold_seg))

    purity = sum(purities) / len(purities) if purities else 0.0
    coverage = sum(coverages) / len(coverages) if coverages else 0.0

    return purity, coverage


def get_regime(bor: float) -> str:
    """Get granularity regime label from BOR."""
    if bor < 0.8:
        return "Conservative"
    elif bor > 1.2:
        return "Aggressive"
    else:
        return "Balanced"


def run_method_on_dataset(
    method_name: str,
    dataset_name: str,
    dialogues: List[Dict]
) -> EvalResultWithPerDialogue:
    """
    Run a method on a dataset and compute all metrics with per-dialogue storage.

    Returns EvalResultWithPerDialogue which includes:
    - Aggregate metrics (same as before)
    - Per-dialogue W-F1 values (for bootstrap ΔW-F1)
    - Per-dialogue gold/pred counts (for bootstrap ΔBOR)

    Per-dialogue data enables bootstrap CI computation without re-running inference.
    """
    log(f"Running {method_name} on {dataset_name}...")

    segmenter = get_segmenter(method_name)

    # Accumulators for aggregate metrics
    total_gold = 0
    total_pred = 0
    total_tp = 0
    w_f1_sum = 0.0
    purity_sum = 0.0
    coverage_sum = 0.0

    # Per-dialogue storage for bootstrap CI
    per_dialogue_wf1 = []
    per_dialogue_gold_count = []
    per_dialogue_pred_count = []

    for dialogue in tqdm(dialogues, desc=f"  {method_name}", leave=False):
        messages = dialogue["messages"]
        gold = dialogue["gold_boundaries"]
        num_msg = dialogue["num_messages"]

        # Get prediction
        result = segmenter.predict_boundaries(
            messages,
            num_gold_boundaries=len(gold),  # For EvenSegmenter
        )
        pred = result.to_set()

        # Per-dialogue boundary counts (for BOR recomputation in bootstrap)
        n_gold_d = len(gold)
        n_pred_d = len(pred)
        per_dialogue_gold_count.append(n_gold_d)
        per_dialogue_pred_count.append(n_pred_d)

        # Strict F1 components
        tp = len(pred & gold)
        total_tp += tp
        total_gold += n_gold_d
        total_pred += n_pred_d

        # Windowed F1 (per-dialogue)
        _, _, w_f1 = compute_windowed_metrics(gold, pred, window=1)
        w_f1_sum += w_f1
        per_dialogue_wf1.append(w_f1)

        # Purity/coverage
        gold_segs = boundaries_to_segments(gold, num_msg)
        pred_segs = boundaries_to_segments(pred, num_msg)
        purity, coverage = compute_purity_coverage(gold_segs, pred_segs)
        purity_sum += purity
        coverage_sum += coverage

    n = len(dialogues)

    # Micro-averaged strict F1
    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    strict_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # BOR (dataset-level)
    bor = total_pred / total_gold if total_gold > 0 else (float('inf') if total_pred > 0 else 1.0)

    result = EvalResultWithPerDialogue(
        method=method_name,
        dataset=dataset_name,
        n_dialogues=n,
        n_gold=total_gold,
        n_pred=total_pred,
        strict_f1=strict_f1,
        w_f1=w_f1_sum / n,  # Macro-averaged W-F1
        bor=bor,
        purity=purity_sum / n,
        coverage=coverage_sum / n,
        regime=get_regime(bor),
        per_dialogue_wf1=per_dialogue_wf1,
        per_dialogue_gold_count=per_dialogue_gold_count,
        per_dialogue_pred_count=per_dialogue_pred_count,
    )

    log(f"  F1={result.strict_f1:.3f}, W-F1={result.w_f1:.3f}, BOR={result.bor:.2f} [{result.regime}]")
    return result


def write_csv_row(filepath: Path, row: Dict, write_header: bool = False):
    """Append a row to CSV file and flush."""
    mode = 'w' if write_header else 'a'
    with open(filepath, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def generate_per_method_csv(results: List[EvalResult], filepath: Path):
    """Generate per-method results CSV."""
    log(f"Writing per-method CSV to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        row = {
            "method": r.method,
            "dataset": r.dataset,
            "n_dialogues": r.n_dialogues,
            "n_gold": r.n_gold,
            "n_pred": r.n_pred,
            "f1": f"{r.strict_f1:.3f}",
            "w_f1": f"{r.w_f1:.3f}",
            "bor": f"{r.bor:.2f}",
            "purity": f"{r.purity:.3f}",
            "coverage": f"{r.coverage:.3f}",
            "regime": r.regime,
        }
        write_csv_row(filepath, row, write_header=(i == 0))


def generate_per_method_latex(results: List[EvalResult], filepath: Path):
    """Generate per-method LaTeX table."""
    log(f"Writing per-method LaTeX to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Per-method results for SuperDialseg leaderboard audit",
        "% Generated by audit_leaderboard.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Method & Dataset & F1 & W-F1 & BOR & Purity & Coverage & Regime \\\\",
        "\\midrule",
    ]

    for r in results:
        line = f"{r.method} & {r.dataset} & {r.strict_f1:.3f} & {r.w_f1:.3f} & {r.bor:.2f} & {r.purity:.3f} & {r.coverage:.3f} & {r.regime} \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Per-method results for SuperDialseg leaderboard audit.}",
        "\\label{tab:leaderboard-per-method}",
        "\\end{table}",
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS FOR AUDIT DELTAS
# =============================================================================

def compute_bootstrap_delta_ci(
    result1: EvalResultWithPerDialogue,
    result2: EvalResultWithPerDialogue,
    n_replicates: int = BOOTSTRAP_N_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_CI_ALPHA,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap CIs for ΔW-F1 and ΔBOR between two methods.

    Bootstrap methodology:
    - Unit: dialogues (resample dialogue indices with replacement)
    - CRITICAL: Same dialogue indices are used for both methods in each replicate
    - CI method: percentile bootstrap (2.5%, 97.5% for alpha=0.05)

    For ΔW-F1:
        1. Resample dialogue indices
        2. Compute macro W-F1_A = mean(W-F1_A[indices]) on resample
        3. Compute macro W-F1_B = mean(W-F1_B[indices]) on resample
        4. ΔW-F1 = W-F1_A - W-F1_B

    For ΔBOR:
        1. Resample dialogue indices
        2. Compute BOR_A = sum(pred_A[indices]) / sum(gold[indices]) on resample
        3. Compute BOR_B = sum(pred_B[indices]) / sum(gold[indices]) on resample
        4. ΔBOR = BOR_A - BOR_B

    Args:
        result1: EvalResultWithPerDialogue for method A
        result2: EvalResultWithPerDialogue for method B
        n_replicates: Number of bootstrap resamples (default 1000)
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dict with keys "delta_wf1" and "delta_bor", each mapping to
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n_dialogues = result1.n_dialogues

    # Verify both methods evaluated on same number of dialogues
    assert result2.n_dialogues == n_dialogues, \
        f"Dialogue count mismatch: {result1.method}={n_dialogues}, {result2.method}={result2.n_dialogues}"

    # Convert to numpy arrays for efficient resampling
    wf1_A = np.array(result1.per_dialogue_wf1)
    wf1_B = np.array(result2.per_dialogue_wf1)
    gold_A = np.array(result1.per_dialogue_gold_count)
    gold_B = np.array(result2.per_dialogue_gold_count)
    pred_A = np.array(result1.per_dialogue_pred_count)
    pred_B = np.array(result2.per_dialogue_pred_count)

    # Verify gold counts are identical (same dataset)
    assert np.array_equal(gold_A, gold_B), \
        "Gold boundary counts should be identical for same dataset"

    # Point estimates (should match original computation)
    delta_wf1_point = float(np.mean(wf1_A) - np.mean(wf1_B))
    total_gold = np.sum(gold_A)
    bor_A_point = np.sum(pred_A) / total_gold if total_gold > 0 else 0.0
    bor_B_point = np.sum(pred_B) / total_gold if total_gold > 0 else 0.0
    delta_bor_point = float(bor_A_point - bor_B_point)

    # Bootstrap resampling
    bootstrap_delta_wf1 = np.zeros(n_replicates)
    bootstrap_delta_bor = np.zeros(n_replicates)

    for i in range(n_replicates):
        # Resample dialogue indices with replacement
        # CRITICAL: Same indices for both methods to maintain pairing
        indices = rng.choice(n_dialogues, size=n_dialogues, replace=True)

        # Recompute ΔW-F1 on resample (macro-average)
        wf1_A_resample = np.mean(wf1_A[indices])
        wf1_B_resample = np.mean(wf1_B[indices])
        bootstrap_delta_wf1[i] = wf1_A_resample - wf1_B_resample

        # Recompute ΔBOR on resample
        # BOR = sum(predicted) / sum(gold) on resampled dialogues
        gold_resample = np.sum(gold_A[indices])
        if gold_resample > 0:
            bor_A_resample = np.sum(pred_A[indices]) / gold_resample
            bor_B_resample = np.sum(pred_B[indices]) / gold_resample
            bootstrap_delta_bor[i] = bor_A_resample - bor_B_resample
        else:
            # Edge case: no gold boundaries in resample
            bootstrap_delta_bor[i] = 0.0

    # Percentile CI (2.5th and 97.5th percentiles for 95% CI)
    ci_low_pct = 100 * alpha / 2
    ci_high_pct = 100 * (1 - alpha / 2)

    delta_wf1_ci_low = float(np.percentile(bootstrap_delta_wf1, ci_low_pct))
    delta_wf1_ci_high = float(np.percentile(bootstrap_delta_wf1, ci_high_pct))

    delta_bor_ci_low = float(np.percentile(bootstrap_delta_bor, ci_low_pct))
    delta_bor_ci_high = float(np.percentile(bootstrap_delta_bor, ci_high_pct))

    return {
        "delta_wf1": (delta_wf1_point, delta_wf1_ci_low, delta_wf1_ci_high),
        "delta_bor": (delta_bor_point, delta_bor_ci_low, delta_bor_ci_high),
    }


def format_delta_with_ci(
    point: float,
    ci_low: float,
    ci_high: float,
    decimals: int = 3,
    show_sign: bool = True
) -> str:
    """
    Format a delta value with CI brackets.

    Examples:
        +0.220 [0.19, 0.25]
        -0.05 [-0.08, -0.02]
    """
    fmt = f"+.{decimals}f" if show_sign else f".{decimals}f"
    point_str = f"{point:{fmt}}"
    ci_str = f"[{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"
    return f"{point_str} {ci_str}"


def compute_claim_deltas_with_ci(
    results: List[EvalResultWithPerDialogue],
    n_replicates: int = BOOTSTRAP_N_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> List[Dict]:
    """
    Compute pairwise deltas with bootstrap CIs for each claim.

    This extends compute_claim_deltas() by adding 95% CIs for ΔW-F1 and ΔBOR.
    """
    # Index results by (method, dataset)
    results_idx = {(r.method, r.dataset): r for r in results}

    deltas = []
    for claim in CLAIMS:
        key1 = (claim["method1"], claim["dataset"])
        key2 = (claim["method2"], claim["dataset"])

        if key1 not in results_idx or key2 not in results_idx:
            log(f"  Warning: Missing results for claim {claim['id']}")
            continue

        r1 = results_idx[key1]
        r2 = results_idx[key2]

        # Point estimates (unchanged from original)
        delta_f1 = r1.strict_f1 - r2.strict_f1
        delta_w_f1 = r1.w_f1 - r2.w_f1
        delta_bor = r1.bor - r2.bor
        regime_shift = f"{r2.regime} → {r1.regime}"

        # Compute bootstrap CIs
        log(f"  Computing bootstrap CIs for claim {claim['id']} ({n_replicates} replicates)...")
        ci_results = compute_bootstrap_delta_ci(r1, r2, n_replicates, seed)

        # Unpack CI results
        wf1_point, wf1_ci_low, wf1_ci_high = ci_results["delta_wf1"]
        bor_point, bor_ci_low, bor_ci_high = ci_results["delta_bor"]

        # Format with CI brackets
        delta_wf1_with_ci = format_delta_with_ci(wf1_point, wf1_ci_low, wf1_ci_high, decimals=3)
        delta_bor_with_ci = format_delta_with_ci(bor_point, bor_ci_low, bor_ci_high, decimals=2)

        deltas.append({
            "claim_id": claim["id"],
            "method1": claim["method1"],
            "method2": claim["method2"],
            "dataset": claim["dataset"],
            "n_dialogues": r1.n_dialogues,
            # Point estimates (raw, for verification)
            "delta_f1": f"{delta_f1:+.3f}",
            "delta_w_f1_point": delta_w_f1,
            "delta_bor_point": delta_bor,
            # Formatted with CI
            "delta_w_f1": delta_wf1_with_ci,
            "delta_bor": delta_bor_with_ci,
            # CI bounds (raw, for artifact storage)
            "delta_w_f1_ci_low": wf1_ci_low,
            "delta_w_f1_ci_high": wf1_ci_high,
            "delta_bor_ci_low": bor_ci_low,
            "delta_bor_ci_high": bor_ci_high,
            # Metadata
            "regime_shift": regime_shift,
            "method1_bor": f"{r1.bor:.2f}",
            "method2_bor": f"{r2.bor:.2f}",
            "bootstrap_n": n_replicates,
            "bootstrap_seed": seed,
        })

    return deltas


def compute_claim_deltas(results: List[EvalResultWithPerDialogue]) -> List[Dict]:
    """
    Compute pairwise deltas for each claim (legacy, without CIs).

    Kept for backward compatibility. Use compute_claim_deltas_with_ci() for CIs.
    """
    # Index results by (method, dataset)
    results_idx = {(r.method, r.dataset): r for r in results}

    deltas = []
    for claim in CLAIMS:
        key1 = (claim["method1"], claim["dataset"])
        key2 = (claim["method2"], claim["dataset"])

        if key1 not in results_idx or key2 not in results_idx:
            log(f"  Warning: Missing results for claim {claim['id']}")
            continue

        r1 = results_idx[key1]
        r2 = results_idx[key2]

        # Deltas: method1 - method2
        delta_f1 = r1.strict_f1 - r2.strict_f1
        delta_w_f1 = r1.w_f1 - r2.w_f1
        delta_bor = r1.bor - r2.bor
        regime_shift = f"{r2.regime} → {r1.regime}"

        deltas.append({
            "claim_id": claim["id"],
            "method1": claim["method1"],
            "method2": claim["method2"],
            "dataset": claim["dataset"],
            "delta_f1": f"{delta_f1:+.3f}",
            "delta_w_f1": f"{delta_w_f1:+.3f}",
            "delta_bor": f"{delta_bor:+.2f}",
            "regime_shift": regime_shift,
            "method1_bor": f"{r1.bor:.2f}",
            "method2_bor": f"{r2.bor:.2f}",
        })

    return deltas


def generate_deltas_csv(deltas: List[Dict], filepath: Path):
    """Generate pairwise claim deltas CSV."""
    log(f"Writing deltas CSV to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(deltas):
        write_csv_row(filepath, d, write_header=(i == 0))


def generate_deltas_latex(deltas: List[Dict], filepath: Path):
    """Generate pairwise claim deltas LaTeX table."""
    log(f"Writing deltas LaTeX to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Pairwise claim deltas for SuperDialseg leaderboard audit",
        "% Generated by audit_leaderboard.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{clcccc}",
        "\\toprule",
        "Claim & Comparison & Dataset & $\\Delta$F1 & $\\Delta$W-F1 & $\\Delta$BOR & Regime Shift \\\\",
        "\\midrule",
    ]

    for d in deltas:
        comparison = f"{d['method1']} vs. {d['method2']}"
        line = f"{d['claim_id']} & {comparison} & {d['dataset']} & {d['delta_f1']} & {d['delta_w_f1']} & {d['delta_bor']} & {d['regime_shift']} \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Pairwise claim deltas. Positive $\\Delta$ indicates Method 1 $>$ Method 2.}",
        "\\label{tab:leaderboard-deltas}",
        "\\end{table}",
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def generate_deltas_csv_with_ci(deltas: List[Dict], filepath: Path):
    """
    Generate pairwise claim deltas CSV with bootstrap CI data.

    CSV columns include:
    - claim_id, method1, method2, dataset, n_dialogues
    - delta_f1 (point estimate only)
    - delta_w_f1_point, delta_w_f1_ci_low, delta_w_f1_ci_high
    - delta_bor_point, delta_bor_ci_low, delta_bor_ci_high
    - bootstrap_n, bootstrap_seed
    """
    log(f"Writing deltas CSV with CIs to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Select columns for CSV (exclude formatted strings, include raw values)
    csv_columns = [
        "claim_id", "method1", "method2", "dataset", "n_dialogues",
        "delta_f1",
        "delta_w_f1_point", "delta_w_f1_ci_low", "delta_w_f1_ci_high",
        "delta_bor_point", "delta_bor_ci_low", "delta_bor_ci_high",
        "regime_shift", "method1_bor", "method2_bor",
        "bootstrap_n", "bootstrap_seed",
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        for d in deltas:
            writer.writerow(d)
        f.flush()


def generate_deltas_latex_with_ci(deltas: List[Dict], filepath: Path):
    """
    Generate pairwise claim deltas LaTeX table with CI brackets.

    Format: ΔW-F1 and ΔBOR shown as "point [ci_low, ci_high]"
    """
    log(f"Writing deltas LaTeX with CIs to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Pairwise claim deltas with 95% bootstrap CIs for SuperDialseg leaderboard audit",
        "% Generated by audit_leaderboard.py",
        f"% Bootstrap: {BOOTSTRAP_N_REPLICATES} replicates, seed={BOOTSTRAP_SEED}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{clccc}",
        "\\toprule",
        "Claim & Comparison & Dataset & $\\Delta$W-F1 [95\\% CI] & $\\Delta$BOR [95\\% CI] \\\\",
        "\\midrule",
    ]

    for d in deltas:
        comparison = f"{d['method1']} vs. {d['method2']}"
        # Use the pre-formatted strings with CI brackets
        line = f"{d['claim_id']} & {comparison} & {d['dataset']} & {d['delta_w_f1']} & {d['delta_bor']} \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Pairwise claim deltas with 95\\% bootstrap CIs ({BOOTSTRAP_N_REPLICATES} replicates). "
        "Positive $\\Delta$ indicates Method 1 $>$ Method 2.}}",
        "\\label{tab:leaderboard-deltas-ci}",
        "\\end{table}",
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def save_ci_artifact(deltas: List[Dict], filepath: Path):
    """
    Save bootstrap CI artifact to JSON for reproducibility.

    The artifact contains all information needed to reproduce the CIs:
    - Point estimates and CI bounds
    - Bootstrap configuration (n_replicates, seed)
    - Dataset and method metadata
    """
    log(f"Writing CI artifact to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "description": "Bootstrap 95% CIs for audit pairwise deltas",
            "bootstrap_n_replicates": BOOTSTRAP_N_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci_alpha": BOOTSTRAP_CI_ALPHA,
            "ci_method": "percentile",
            "bootstrap_unit": "dialogues",
        },
        "claims": [],
    }

    for d in deltas:
        claim_data = {
            "claim_id": d["claim_id"],
            "method1": d["method1"],
            "method2": d["method2"],
            "dataset": d["dataset"],
            "n_dialogues": d["n_dialogues"],
            "delta_w_f1": {
                "point_estimate": d["delta_w_f1_point"],
                "ci_low": d["delta_w_f1_ci_low"],
                "ci_high": d["delta_w_f1_ci_high"],
            },
            "delta_bor": {
                "point_estimate": d["delta_bor_point"],
                "ci_low": d["delta_bor_ci_low"],
                "ci_high": d["delta_bor_ci_high"],
            },
            "regime_shift": d["regime_shift"],
        }
        artifact["claims"].append(claim_data)

    with open(filepath, 'w') as f:
        json.dump(artifact, f, indent=2)


def generate_summary(results: List[EvalResultWithPerDialogue], deltas: List[Dict], filepath: Path):
    """Generate summary note with bootstrap CI information."""
    log(f"Writing summary to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if deltas have CI data
    has_ci = "delta_w_f1_ci_low" in deltas[0] if deltas else False

    lines = [
        "# SuperDialseg Leaderboard Audit Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed: {RANDOM_SEED}",
        "",
    ]

    if has_ci:
        lines.extend([
            "## Bootstrap Confidence Intervals",
            "",
            f"- Bootstrap replicates: {BOOTSTRAP_N_REPLICATES}",
            f"- Bootstrap seed: {BOOTSTRAP_SEED}",
            f"- CI method: percentile bootstrap (2.5%, 97.5%)",
            f"- Resampling unit: dialogues (with replacement)",
            "",
        ])

    lines.extend([
        "## Claims Audited",
        "",
    ])

    for d in deltas:
        lines.append(f"- **Claim {d['claim_id']}** ({d['dataset']}): {d['method1']} vs. {d['method2']}")
        lines.append(f"  - ΔF1: {d['delta_f1']}")
        lines.append(f"  - ΔW-F1: {d['delta_w_f1']}")
        lines.append(f"  - ΔBOR: {d['delta_bor']}")
        lines.append(f"  - Regime shift: {d['regime_shift']}")
        lines.append("")

    lines.extend([
        "## Key Observations",
        "",
        "For runnable published methods, the analysis examines whether F1/W-F1 differences",
        "align with differences in boundary density (BOR) and granularity regime.",
        "",
    ])

    # Check if higher F1 correlates with BOR closer to 1
    bor_correlation_notes = []
    for d in deltas:
        delta_f1 = float(d['delta_f1'])
        m1_bor = float(d['method1_bor'])
        m2_bor = float(d['method2_bor'])

        # Which method has BOR closer to 1?
        m1_bor_dist = abs(m1_bor - 1.0)
        m2_bor_dist = abs(m2_bor - 1.0)

        if delta_f1 > 0:
            higher_f1_method = d['method1']
            closer_bor_method = d['method1'] if m1_bor_dist < m2_bor_dist else d['method2']
        else:
            higher_f1_method = d['method2']
            closer_bor_method = d['method2'] if m2_bor_dist < m1_bor_dist else d['method1']

        aligned = higher_f1_method == closer_bor_method
        bor_correlation_notes.append(
            f"- Claim {d['claim_id']}: Higher F1 method ({higher_f1_method}) "
            f"{'has' if aligned else 'does NOT have'} BOR closer to 1.0"
        )

    lines.extend(bor_correlation_notes)
    lines.append("")

    lines.extend([
        "## Non-Reproduced Methods",
        "",
        "- **RoBERTa**: Requires training from scratch; not reproduced.",
        "- **ChatGPT**: Requires proprietary API; not reproduced.",
        "",
        "## Interpretation",
        "",
        "This analysis does not claim invalidation of prior work. It only reports",
        "whether observed F1/W-F1 differences are accompanied by shifts in BOR",
        "and granularity regime.",
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def main():
    """Main entry point."""
    log("=" * 60)
    log("SuperDialseg Leaderboard Audit")
    log("=" * 60)
    log(f"Random seed: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)

    # Validate paths
    if not validate_paths():
        log("Stopping: Missing required dataset files.")
        sys.exit(1)

    # Load datasets
    datasets = {}
    for name in EXPECTED_PATHS:
        datasets[name] = load_dataset(name)

    # Determine which methods to run
    methods_to_run = set()
    for claim in CLAIMS:
        methods_to_run.add(claim["method1"])
        methods_to_run.add(claim["method2"])

    log(f"Methods to evaluate: {', '.join(sorted(methods_to_run))}")

    # Run each method on each dataset
    all_results = []
    csv_path = RESULTS_DIR / "leaderboard_reanalysis.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    first_row = True
    for dataset_name, dialogues in datasets.items():
        log(f"\n{'='*60}")
        log(f"Dataset: {dataset_name} ({len(dialogues)} dialogues)")
        log("=" * 60)

        for method_name in sorted(methods_to_run):
            try:
                result = run_method_on_dataset(method_name, dataset_name, dialogues)
                all_results.append(result)

                # Incremental CSV write
                row = {
                    "method": result.method,
                    "dataset": result.dataset,
                    "n_dialogues": result.n_dialogues,
                    "n_gold": result.n_gold,
                    "n_pred": result.n_pred,
                    "f1": f"{result.strict_f1:.3f}",
                    "w_f1": f"{result.w_f1:.3f}",
                    "bor": f"{result.bor:.2f}",
                    "purity": f"{result.purity:.3f}",
                    "coverage": f"{result.coverage:.3f}",
                    "regime": result.regime,
                }
                write_csv_row(csv_path, row, write_header=first_row)
                first_row = False

            except Exception as e:
                log(f"  ERROR running {method_name}: {e}")
                import traceback
                traceback.print_exc()

    # Generate outputs
    log(f"\n{'='*60}")
    log("Generating outputs")
    log("=" * 60)

    # Per-method LaTeX (convert to EvalResult for compatibility)
    eval_results = [r.to_eval_result() for r in all_results]
    generate_per_method_latex(eval_results, PAPER_TABLES_DIR / "leaderboard_reanalysis.tex")

    # Pairwise deltas with bootstrap CIs
    log(f"\n--- Computing Bootstrap CIs ({BOOTSTRAP_N_REPLICATES} replicates, seed={BOOTSTRAP_SEED}) ---")
    deltas_with_ci = compute_claim_deltas_with_ci(
        all_results,
        n_replicates=BOOTSTRAP_N_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    # Generate outputs with CIs
    generate_deltas_csv_with_ci(deltas_with_ci, RESULTS_DIR / "leaderboard_claim_deltas_ci.csv")
    generate_deltas_latex_with_ci(deltas_with_ci, PAPER_TABLES_DIR / "leaderboard_claim_deltas_ci.tex")
    save_ci_artifact(deltas_with_ci, RESULTS_DIR / "leaderboard_audit_ci_artifact.json")

    # Also generate legacy outputs without CIs for backward compatibility
    deltas_legacy = compute_claim_deltas(all_results)
    generate_deltas_csv(deltas_legacy, RESULTS_DIR / "leaderboard_claim_deltas.csv")
    generate_deltas_latex(deltas_legacy, PAPER_TABLES_DIR / "leaderboard_claim_deltas.tex")

    # Summary (uses CI-enabled deltas)
    generate_summary(all_results, deltas_with_ci, RESULTS_DIR / "leaderboard_audit_summary.md")

    # Print summary table
    log("\n" + "=" * 90)
    log(f"{'Method':<15} {'Dataset':<15} {'F1':>6} {'W-F1':>6} {'BOR':>5} {'Purity':>7} {'Coverage':>8} {'Regime':<12}")
    log("=" * 90)
    for r in all_results:
        log(f"{r.method:<15} {r.dataset:<15} {r.strict_f1:>6.3f} {r.w_f1:>6.3f} {r.bor:>5.2f} {r.purity:>7.3f} {r.coverage:>8.3f} [{r.regime}]")
    log("=" * 90)

    # Print delta summary with CIs
    log("\n" + "=" * 90)
    log("Pairwise Deltas with 95% Bootstrap CIs")
    log("=" * 90)
    for d in deltas_with_ci:
        log(f"Claim {d['claim_id']}: {d['method1']} vs. {d['method2']} on {d['dataset']}")
        log(f"  ΔW-F1: {d['delta_w_f1']}")
        log(f"  ΔBOR:  {d['delta_bor']}")
    log("=" * 90)

    log("\nDone.")
    log(f"Results: {csv_path}")
    log(f"Deltas (with CI): {RESULTS_DIR / 'leaderboard_claim_deltas_ci.csv'}")
    log(f"CI Artifact: {RESULTS_DIR / 'leaderboard_audit_ci_artifact.json'}")
    log(f"Summary: {RESULTS_DIR / 'leaderboard_audit_summary.md'}")


if __name__ == "__main__":
    main()
