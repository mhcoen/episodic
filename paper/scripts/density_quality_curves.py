#!/usr/bin/env python3
"""
Density-Quality Curve Analysis for Dialogue Topic Segmentation

This script sweeps the boundary selection threshold τ to vary BOR (Boundary
Oversegmentation Ratio) from ~0.2 to ~5.0 while holding the minimum spacing
parameter g fixed. For each τ, it computes BOR, W-F1, coverage, and purity.

Key constraint: This sweep isolates selection granularity by only varying τ.
Candidate generation is frozen (all inter-utterance positions are candidates).

BOR Definition:
    BOR = #predicted_boundaries / #gold_boundaries
    - BOR < 1: undersegmentation (fewer boundaries than gold)
    - BOR = 1: matched density
    - BOR > 1: oversegmentation (more boundaries than gold)

Models evaluated:
    1. Neural scorer (proposed model with calibrated temperature)
    2. TextTiling baseline (depth scores)
    3. CSM baseline (NSP-based depth scores)
    4. Random scorer (uniform random scores, 10 seeds)

Usage:
    python paper/density_quality_curves.py

Outputs:
    - paper/figures/density_quality_{dataset}.pdf/png
    - paper/results/sweep_{dataset}_{model}.csv
    - paper/results/auc_summary.csv

Author: Generated for paper experiments
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
    compute_windowed_metrics_one_to_one,
    compute_exact_f1,
)

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
PAPER_DIR = Path(__file__).parent.parent  # paper/ directory
FIGURES_DIR = PAPER_DIR / "figures"
RESULTS_DIR = PAPER_DIR / "results"

# Sweep configuration
MIN_GAP = 2  # Fixed minimum spacing between boundaries (in user turns)
N_SWEEP_STEPS = 200  # Number of threshold steps
MAX_BOR = 5.0  # Stop sweep when BOR exceeds this
BOR_RANGE = (0.2, 5.0)  # Range for AUC computation
RANDOM_SEEDS = list(range(10))  # Seeds for random baseline

# Bootstrap CI configuration
# Bootstrap unit: dialogues (resample dialogues with replacement)
# CI method: percentile bootstrap (2.5%, 97.5%)
# BOR: treated as fixed per operating point (dataset-level quantity)
BOOTSTRAP_N_REPLICATES = 1000  # Number of bootstrap resamples (configurable)
BOOTSTRAP_SEED = 42  # Fixed random seed for reproducibility
BOOTSTRAP_CI_ALPHA = 0.05  # 95% CI (2.5%, 97.5% percentiles)

# Datasets to evaluate
DATASETS = {
    "dialseg711": {"display": "DialSeg711", "type": "dense"},
    "superseg": {"display": "SuperSeg", "type": "sparse"},
}

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "neural": "#0072B2",      # Blue
    "texttiling": "#E69F00",  # Orange
    "csm": "#009E73",         # Green
    "random": "#CC79A7",      # Pink/magenta
}

# BOR regime colors (qualitative, matching paper definitions)
REGIME_COLORS = {
    "conservative": "#4393c3",  # Blue-ish (undersegmentation)
    "aggressive": "#d6604d",    # Red-ish (oversegmentation)
}


def add_bor_regime_overlays(ax, *, alpha=0.10):
    """
    Add qualitative BOR regime overlays to a plot.

    Regime definitions (from paper/topicDetection.tex Table 2):
    - Conservative: BOR < 1 (undersegmentation)
    - Balanced: BOR ≈ 1 (rendered as BOR=1 reference line)
    - Aggressive: BOR > 1 (oversegmentation)

    Args:
        ax: Matplotlib axis to annotate
        alpha: Transparency for shaded bands (default 0.10)
    """
    x0, x1 = ax.get_xlim()
    trans = ax.get_xaxis_transform()

    # BOR=1 reference line (always draw if in range)
    if x0 <= 1.0 <= x1:
        ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1.5,
                   zorder=1, label="_nolegend_")
        # Label at 1/3 height of plot
        ax.text(1.02, 0.33, "BOR=1\n(density-matched)", transform=trans,
                fontsize=9, ha="left", va="center", color="#444444", alpha=0.9)

    # Shade BOR < 1 region (if data exists there)
    if x0 < 1.0:
        ax.axvspan(x0, min(1.0, x1), color=REGIME_COLORS["conservative"],
                   alpha=alpha, zorder=0, label="_nolegend_")
        # Centered annotation in blue region
        center_x = (x0 + min(1.0, x1)) / 2
        ax.text(center_x, 0.97, "BOR<1", transform=trans, fontsize=9,
                ha="center", va="top", color=REGIME_COLORS["conservative"], alpha=0.7)

    # Shade BOR > 1 region only if xlim extends nontrivially beyond 1
    if x1 > 1.2:
        ax.axvspan(max(1.0, x0), x1, color=REGIME_COLORS["aggressive"],
                   alpha=alpha, zorder=0, label="_nolegend_")
        # Centered annotation in red region
        center_x = (max(1.0, x0) + x1) / 2
        ax.text(center_x, 0.97, "BOR>1", transform=trans, fontsize=9,
                ha="center", va="top", color=REGIME_COLORS["aggressive"], alpha=0.7)

    # Reset xlim in case axvspan changed it
    ax.set_xlim(x0, x1)


@dataclass
class DialogueData:
    """Container for a single dialogue with boundaries."""
    dialogue_id: int
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_messages: int  # Number of user turns


@dataclass
class SweepPoint:
    """Single point in the sweep."""
    dataset: str
    model: str
    g: int
    step: int
    tau: float
    bor: float
    wf1: float  # Many-to-one matching (original)
    wf1_1to1: float  # One-to-one matching (standard tolerant matching)
    exact_f1: float  # Strict position matching (no tolerance)
    coverage: float
    purity: float
    n_pred_boundaries: int
    n_gold_boundaries: int
    seed: Optional[int] = None


@dataclass
class SweepPointWithPerDialogue:
    """
    Sweep point with per-dialogue metrics for bootstrap CI computation.

    Per-dialogue metrics are stored as lists aligned with the dialogue index.
    This enables dialogue-level bootstrap resampling without re-running inference.
    """
    dataset: str
    model: str
    g: int
    step: int
    tau: float
    bor: float  # Fixed dataset-level BOR for this operating point
    wf1: float  # Macro-averaged W-F1 (many-to-one matching)
    wf1_1to1: float  # Macro-averaged W-F1 (one-to-one matching)
    exact_f1: float  # Macro-averaged exact F1 (strict position matching)
    coverage: float  # Macro-averaged Coverage
    purity: float  # Macro-averaged Purity
    n_pred_boundaries: int
    n_gold_boundaries: int
    n_dialogues: int
    # Per-dialogue metrics (lists of length n_dialogues)
    per_dialogue_wf1: List[float]
    per_dialogue_wf1_1to1: List[float]
    per_dialogue_exact_f1: List[float]
    per_dialogue_coverage: List[float]
    seed: Optional[int] = None


def log(msg: str):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name: str) -> List[DialogueData]:
    """Load dataset dialogues with gold boundaries."""
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Dataset not found: {test_file}")

    with open(test_file) as f:
        data = json.load(f)

    dialogues = []
    dial_data = data.get("dial_data", data)
    dialogue_id = 0

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            # Extract boundaries at user turn positions
            boundaries = set()
            prev_topic = None
            user_idx = 0

            for turn in turns:
                if turn.get("role") == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            messages = [
                {"role": t["role"], "content": t.get("utterance", t.get("text", ""))}
                for t in turns
            ]

            num_user_turns = sum(1 for m in messages if m["role"] == "user")

            dialogues.append(DialogueData(
                dialogue_id=dialogue_id,
                messages=messages,
                gold_boundaries=boundaries,
                num_messages=num_user_turns
            ))
            dialogue_id += 1

    return dialogues


# =============================================================================
# SCORE COMPUTATION
# =============================================================================

def get_neural_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """
    Get per-position scores from the neural model.

    Returns: List of dicts mapping user_turn_idx -> confidence score
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    # Load calibrated model
    model_path = TACL_DIR / "experiments" / "models" / "final_calibrated.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create examples
    all_scores = []

    for dialogue in tqdm(dialogues, desc="  Neural scoring", leave=False):
        messages = dialogue.messages
        scores = {}

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    # Create window
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    # Format text
                    context_parts = []
                    for m in window[-6:]:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        context_parts.append(f"{role}: {content}")

                    curr_content = msg.get("content", "")
                    text = " [SEP] ".join(context_parts) + f" [SEP] current: {curr_content}"

                    # Tokenize
                    encoding = tokenizer(
                        text, max_length=256, padding="max_length",
                        truncation=True, return_tensors="pt"
                    )

                    # Get score
                    with torch.no_grad():
                        inputs = {k: v.to(device) for k, v in encoding.items()}
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits / temperature, dim=-1)
                        score = probs[0, 1].item()  # Probability of boundary

                    scores[user_idx] = score

                user_idx += 1

        all_scores.append(scores)

    return all_scores


def get_texttiling_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """
    Get per-position depth scores from TextTiling.

    Returns: List of dicts mapping user_turn_idx -> depth score
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    all_scores = []

    for dialogue in tqdm(dialogues, desc="  TextTiling scoring", leave=False):
        # Get user utterances only
        user_utterances = [
            msg["content"] for msg in dialogue.messages if msg["role"] == "user"
        ]

        if len(user_utterances) < 3:
            all_scores.append({})
            continue

        # Encode utterances
        embeddings = model.encode(user_utterances, convert_to_numpy=True)

        # Compute similarities between adjacent utterances
        similarities = []
        for i in range(len(embeddings) - 1):
            e1, e2 = embeddings[i], embeddings[i + 1]
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            similarities.append(sim)

        # Compute depth scores
        depth_scores = []
        for i in range(len(similarities)):
            left_flag = similarities[i]
            right_flag = similarities[i]

            for left_idx in range(i - 1, -1, -1):
                if similarities[left_idx] >= left_flag:
                    left_flag = similarities[left_idx]
                else:
                    break

            for right_idx in range(i + 1, len(similarities)):
                if similarities[right_idx] >= right_flag:
                    right_flag = similarities[right_idx]
                else:
                    break

            depth = 0.5 * (left_flag + right_flag - 2 * similarities[i])
            depth_scores.append(depth)

        # Map to user turn indices (depth_scores[i] -> boundary at position i+1)
        scores = {}
        for i, depth in enumerate(depth_scores):
            pos = i + 1  # Canonical boundary position
            if 1 <= pos < len(user_utterances):
                scores[pos] = depth

        all_scores.append(scores)

    return all_scores


def get_csm_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """
    Get per-position depth scores from CSM (NSP-based).

    Returns: List of dicts mapping user_turn_idx -> depth score
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForNextSentencePrediction

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    all_scores = []

    for dialogue in tqdm(dialogues, desc="  CSM scoring", leave=False):
        user_utterances = [
            msg["content"] for msg in dialogue.messages if msg["role"] == "user"
        ]

        if len(user_utterances) < 3:
            all_scores.append({})
            continue

        # Compute NSP scores
        nsp_scores = []
        with torch.no_grad():
            for i in range(len(user_utterances) - 1):
                sent1 = user_utterances[i]
                sent2 = user_utterances[i + 1]

                tokenized = tokenizer(
                    sent1, sent2, padding="max_length", max_length=128,
                    truncation=True, return_tensors="pt"
                )
                tokenized = {k: v.to(device) for k, v in tokenized.items()}

                outputs = model(**tokenized)
                probs = torch.softmax(outputs.logits, dim=1)
                continuation_prob = probs[0, 0].item()
                nsp_scores.append(continuation_prob)

        # Compute depth scores (same as TextTiling)
        depth_scores = []
        for i in range(len(nsp_scores)):
            left_flag = nsp_scores[i]
            right_flag = nsp_scores[i]

            for left_idx in range(i - 1, -1, -1):
                if nsp_scores[left_idx] >= left_flag:
                    left_flag = nsp_scores[left_idx]
                else:
                    break

            for right_idx in range(i + 1, len(nsp_scores)):
                if nsp_scores[right_idx] >= right_flag:
                    right_flag = nsp_scores[right_idx]
                else:
                    break

            depth = 0.5 * (left_flag + right_flag - 2 * nsp_scores[i])
            depth_scores.append(depth)

        scores = {}
        for i, depth in enumerate(depth_scores):
            pos = i + 1
            if 1 <= pos < len(user_utterances):
                scores[pos] = depth

        all_scores.append(scores)

    return all_scores


def get_random_scores(dialogues: List[DialogueData], seed: int) -> List[Dict[int, float]]:
    """
    Get random uniform scores for each position.

    Returns: List of dicts mapping user_turn_idx -> random score in [0, 1]
    """
    rng = np.random.RandomState(seed)
    all_scores = []

    for dialogue in dialogues:
        num_user_turns = dialogue.num_messages
        scores = {}
        for pos in range(1, num_user_turns):
            scores[pos] = rng.random()
        all_scores.append(scores)

    return all_scores


# =============================================================================
# THRESHOLD SWEEP WITH MIN_GAP ENFORCEMENT
# =============================================================================

def apply_threshold_with_gap(
    scores: Dict[int, float],
    threshold: float,
    min_gap: int,
    num_messages: int
) -> Set[int]:
    """
    Apply threshold to scores with min_gap enforcement.

    Args:
        scores: Dict mapping position -> score
        threshold: Only positions with score >= threshold are candidates
        min_gap: Minimum spacing between selected boundaries
        num_messages: Total number of messages

    Returns:
        Set of selected boundary positions
    """
    # Get all positions above threshold, sorted by score (descending)
    candidates = [
        (pos, score) for pos, score in scores.items()
        if score >= threshold and 1 <= pos < num_messages
    ]
    candidates.sort(key=lambda x: -x[1])  # Highest score first

    # Greedily select with gap enforcement
    selected = set()
    for pos, _ in candidates:
        # Check gap from all already-selected boundaries
        ok = True
        for existing in selected:
            if abs(pos - existing) < min_gap:
                ok = False
                break
        if ok:
            selected.add(pos)

    return selected


def run_sweep(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    dataset_name: str,
    model_name: str,
    min_gap: int = MIN_GAP,
    n_steps: int = N_SWEEP_STEPS,
    max_bor: float = MAX_BOR,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sweep threshold τ and compute metrics at each point.

    Returns: DataFrame with one row per sweep step
    """
    # Collect all scores to determine threshold range
    all_scores_flat = []
    for scores in scores_list:
        all_scores_flat.extend(scores.values())

    if not all_scores_flat:
        return pd.DataFrame()

    # Use quantiles for sweep (robust to score distribution)
    thresholds = np.percentile(all_scores_flat, np.linspace(100, 0, n_steps))
    thresholds = np.unique(thresholds)  # Remove duplicates

    # Count total gold boundaries
    total_gold = sum(len(d.gold_boundaries) for d in dialogues)

    results = []
    prev_bor = None

    for step, tau in enumerate(thresholds):
        # Apply threshold to all dialogues
        all_preds = []
        all_wf1 = []
        all_wf1_1to1 = []
        all_exact_f1 = []
        all_purity = []
        all_coverage = []
        total_pred = 0

        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(
                scores, tau, min_gap, dialogue.num_messages
            )
            all_preds.append(pred)
            total_pred += len(pred)

            # W-F1 (many-to-one matching - original)
            _, _, wf1 = compute_windowed_metrics(
                dialogue.gold_boundaries, pred, dialogue.num_messages, window=1
            )
            all_wf1.append(wf1)

            # W-F1 (one-to-one matching - standard tolerant matching)
            _, _, wf1_1to1 = compute_windowed_metrics_one_to_one(
                dialogue.gold_boundaries, pred, dialogue.num_messages, window=1
            )
            all_wf1_1to1.append(wf1_1to1)

            # Exact F1 (strict position matching - no tolerance)
            _, _, exact_f1 = compute_exact_f1(
                dialogue.gold_boundaries, pred, dialogue.num_messages
            )
            all_exact_f1.append(exact_f1)

            # Purity/Coverage
            gold_segments = boundaries_to_segments(
                dialogue.gold_boundaries, dialogue.num_messages
            )
            pred_segments = boundaries_to_segments(pred, dialogue.num_messages)
            purity, coverage = compute_purity_coverage(gold_segments, pred_segments)
            all_purity.append(purity)
            all_coverage.append(coverage)

        bor = total_pred / total_gold if total_gold > 0 else 0.0

        # Early stopping if BOR exceeds max
        if bor > max_bor:
            break

        # Skip if identical to previous step (saturation)
        if prev_bor is not None and abs(bor - prev_bor) < 1e-6:
            continue

        results.append(SweepPoint(
            dataset=dataset_name,
            model=model_name,
            g=min_gap,
            step=step,
            tau=float(tau),
            bor=bor,
            wf1=float(np.mean(all_wf1)),
            wf1_1to1=float(np.mean(all_wf1_1to1)),
            exact_f1=float(np.mean(all_exact_f1)),
            coverage=float(np.mean(all_coverage)),
            purity=float(np.mean(all_purity)),
            n_pred_boundaries=total_pred,
            n_gold_boundaries=total_gold,
            seed=seed,
        ))

        prev_bor = bor

    return pd.DataFrame([asdict(r) for r in results])


def run_sweep_with_per_dialogue(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    dataset_name: str,
    model_name: str,
    min_gap: int = MIN_GAP,
    n_steps: int = N_SWEEP_STEPS,
    max_bor: float = MAX_BOR,
    seed: Optional[int] = None,
) -> List[SweepPointWithPerDialogue]:
    """
    Sweep threshold τ and store per-dialogue metrics for bootstrap CI.

    This extends run_sweep() by storing per-dialogue W-F1 and Coverage
    for each operating point, enabling dialogue-level bootstrap resampling.

    Point-estimate equivalence: The aggregate metrics (wf1, coverage) are
    computed identically to run_sweep() via np.mean() over per-dialogue values.

    Empty-set handling: Per compute_windowed_metrics(), if both gold and
    predicted boundaries are empty, W-F1=1.0. Per compute_purity_coverage(),
    if segments are empty, coverage=0.0. These edge cases are propagated
    through the bootstrap CI computation.

    Returns: List of SweepPointWithPerDialogue objects
    """
    # Collect all scores to determine threshold range
    all_scores_flat = []
    for scores in scores_list:
        all_scores_flat.extend(scores.values())

    if not all_scores_flat:
        return []

    # Use quantiles for sweep (robust to score distribution)
    thresholds = np.percentile(all_scores_flat, np.linspace(100, 0, n_steps))
    thresholds = np.unique(thresholds)

    # Count total gold boundaries
    total_gold = sum(len(d.gold_boundaries) for d in dialogues)
    n_dialogues = len(dialogues)

    results = []
    prev_bor = None

    for step, tau in enumerate(thresholds):
        # Apply threshold to all dialogues and collect per-dialogue metrics
        per_dialogue_wf1 = []
        per_dialogue_wf1_1to1 = []
        per_dialogue_exact_f1 = []
        per_dialogue_coverage = []
        total_pred = 0

        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(
                scores, tau, min_gap, dialogue.num_messages
            )
            total_pred += len(pred)

            # W-F1 per dialogue (many-to-one matching)
            _, _, wf1 = compute_windowed_metrics(
                dialogue.gold_boundaries, pred, dialogue.num_messages, window=1
            )
            per_dialogue_wf1.append(wf1)

            # W-F1 per dialogue (one-to-one matching)
            _, _, wf1_1to1 = compute_windowed_metrics_one_to_one(
                dialogue.gold_boundaries, pred, dialogue.num_messages, window=1
            )
            per_dialogue_wf1_1to1.append(wf1_1to1)

            # Exact F1 per dialogue (strict position matching)
            _, _, exact_f1 = compute_exact_f1(
                dialogue.gold_boundaries, pred, dialogue.num_messages
            )
            per_dialogue_exact_f1.append(exact_f1)

            # Coverage per dialogue
            gold_segments = boundaries_to_segments(
                dialogue.gold_boundaries, dialogue.num_messages
            )
            pred_segments = boundaries_to_segments(pred, dialogue.num_messages)
            _, coverage = compute_purity_coverage(gold_segments, pred_segments)
            per_dialogue_coverage.append(coverage)

        bor = total_pred / total_gold if total_gold > 0 else 0.0

        # Early stopping if BOR exceeds max
        if bor > max_bor:
            break

        # Skip if identical to previous step (saturation)
        if prev_bor is not None and abs(bor - prev_bor) < 1e-6:
            continue

        results.append(SweepPointWithPerDialogue(
            dataset=dataset_name,
            model=model_name,
            g=min_gap,
            step=step,
            tau=float(tau),
            bor=bor,
            wf1=float(np.mean(per_dialogue_wf1)),
            wf1_1to1=float(np.mean(per_dialogue_wf1_1to1)),
            exact_f1=float(np.mean(per_dialogue_exact_f1)),
            coverage=float(np.mean(per_dialogue_coverage)),
            purity=0.0,  # Not needed for CI, skip for efficiency
            n_pred_boundaries=total_pred,
            n_gold_boundaries=total_gold,
            n_dialogues=n_dialogues,
            per_dialogue_wf1=per_dialogue_wf1,
            per_dialogue_wf1_1to1=per_dialogue_wf1_1to1,
            per_dialogue_exact_f1=per_dialogue_exact_f1,
            per_dialogue_coverage=per_dialogue_coverage,
            seed=seed,
        ))

        prev_bor = bor

    return results


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def compute_bootstrap_ci(
    per_dialogue_values: List[float],
    n_replicates: int = BOOTSTRAP_N_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_CI_ALPHA,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a macro-averaged metric.

    This implements dialogue-level percentile bootstrap:
    1. Resample dialogues with replacement
    2. Compute macro-average (mean) of the metric on the resample
    3. Repeat n_replicates times
    4. Return percentile-based CI bounds

    Args:
        per_dialogue_values: List of per-dialogue metric values
        n_replicates: Number of bootstrap resamples (default 1000)
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (point_estimate, ci_low, ci_high)

    Note:
        BOR is treated as fixed per operating point since it's a dataset-level
        quantity determined by the threshold. Bootstrap measures uncertainty
        in the metric (W-F1, Coverage) given dialogue sampling variability.
    """
    rng = np.random.RandomState(seed)
    n_dialogues = len(per_dialogue_values)
    values = np.array(per_dialogue_values)

    # Point estimate is the original macro-average
    point_estimate = float(np.mean(values))

    # Bootstrap resampling
    bootstrap_means = np.zeros(n_replicates)
    for i in range(n_replicates):
        # Resample dialogue indices with replacement
        indices = rng.choice(n_dialogues, size=n_dialogues, replace=True)
        # Compute macro-average on the resample
        bootstrap_means[i] = np.mean(values[indices])

    # Percentile CI (2.5th and 97.5th percentiles for 95% CI)
    ci_low = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return point_estimate, ci_low, ci_high


def compute_sweep_bootstrap_cis(
    sweep_points: List[SweepPointWithPerDialogue],
    n_replicates: int = BOOTSTRAP_N_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Compute bootstrap CIs for all operating points in a sweep.

    Args:
        sweep_points: List of SweepPointWithPerDialogue from run_sweep_with_per_dialogue
        n_replicates: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            dataset, model, step, tau, bor, metric, estimate, ci_low, ci_high,
            n_dialogues, bootstrap_n, bootstrap_seed
    """
    results = []

    for point in sweep_points:
        # Compute CI for W-F1 (many-to-one matching)
        wf1_est, wf1_lo, wf1_hi = compute_bootstrap_ci(
            point.per_dialogue_wf1, n_replicates, seed
        )
        results.append({
            "dataset": point.dataset,
            "model": point.model,
            "step": point.step,
            "tau": point.tau,
            "bor": point.bor,
            "metric": "wf1",
            "estimate": wf1_est,
            "ci_low": wf1_lo,
            "ci_high": wf1_hi,
            "n_dialogues": point.n_dialogues,
            "bootstrap_n": n_replicates,
            "bootstrap_seed": seed,
        })

        # Compute CI for W-F1 (one-to-one matching)
        wf1_1to1_est, wf1_1to1_lo, wf1_1to1_hi = compute_bootstrap_ci(
            point.per_dialogue_wf1_1to1, n_replicates, seed
        )
        results.append({
            "dataset": point.dataset,
            "model": point.model,
            "step": point.step,
            "tau": point.tau,
            "bor": point.bor,
            "metric": "wf1_1to1",
            "estimate": wf1_1to1_est,
            "ci_low": wf1_1to1_lo,
            "ci_high": wf1_1to1_hi,
            "n_dialogues": point.n_dialogues,
            "bootstrap_n": n_replicates,
            "bootstrap_seed": seed,
        })

        # Compute CI for Exact F1 (strict position matching)
        exact_f1_est, exact_f1_lo, exact_f1_hi = compute_bootstrap_ci(
            point.per_dialogue_exact_f1, n_replicates, seed
        )
        results.append({
            "dataset": point.dataset,
            "model": point.model,
            "step": point.step,
            "tau": point.tau,
            "bor": point.bor,
            "metric": "exact_f1",
            "estimate": exact_f1_est,
            "ci_low": exact_f1_lo,
            "ci_high": exact_f1_hi,
            "n_dialogues": point.n_dialogues,
            "bootstrap_n": n_replicates,
            "bootstrap_seed": seed,
        })

        # Compute CI for Coverage
        cov_est, cov_lo, cov_hi = compute_bootstrap_ci(
            point.per_dialogue_coverage, n_replicates, seed
        )
        results.append({
            "dataset": point.dataset,
            "model": point.model,
            "step": point.step,
            "tau": point.tau,
            "bor": point.bor,
            "metric": "coverage",
            "estimate": cov_est,
            "ci_low": cov_lo,
            "ci_high": cov_hi,
            "n_dialogues": point.n_dialogues,
            "bootstrap_n": n_replicates,
            "bootstrap_seed": seed,
        })

    return pd.DataFrame(results)


def save_per_dialogue_data(
    sweep_points: List[SweepPointWithPerDialogue],
    output_path: Path,
):
    """
    Save per-dialogue metrics to JSON for reproducibility.

    The JSON structure allows recomputing bootstrap CIs with different
    parameters without re-running model inference.
    """
    data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "n_points": len(sweep_points),
            "dataset": sweep_points[0].dataset if sweep_points else None,
            "model": sweep_points[0].model if sweep_points else None,
            "n_dialogues": sweep_points[0].n_dialogues if sweep_points else 0,
        },
        "points": [
            {
                "step": p.step,
                "tau": p.tau,
                "bor": p.bor,
                "wf1": p.wf1,
                "wf1_1to1": p.wf1_1to1,
                "exact_f1": p.exact_f1,
                "coverage": p.coverage,
                "n_pred_boundaries": p.n_pred_boundaries,
                "n_gold_boundaries": p.n_gold_boundaries,
                "per_dialogue_wf1": p.per_dialogue_wf1,
                "per_dialogue_wf1_1to1": p.per_dialogue_wf1_1to1,
                "per_dialogue_exact_f1": p.per_dialogue_exact_f1,
                "per_dialogue_coverage": p.per_dialogue_coverage,
                "seed": p.seed,
            }
            for p in sweep_points
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_per_dialogue_data(input_path: Path) -> List[SweepPointWithPerDialogue]:
    """Load per-dialogue metrics from JSON."""
    with open(input_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    points = []

    for p in data["points"]:
        # Handle backwards compatibility: old files may not have wf1_1to1 or exact_f1
        per_dialogue_wf1_1to1 = p.get("per_dialogue_wf1_1to1", [])
        wf1_1to1 = p.get("wf1_1to1", 0.0)
        per_dialogue_exact_f1 = p.get("per_dialogue_exact_f1", [])
        exact_f1 = p.get("exact_f1", 0.0)
        points.append(SweepPointWithPerDialogue(
            dataset=metadata["dataset"],
            model=metadata["model"],
            g=MIN_GAP,
            step=p["step"],
            tau=p["tau"],
            bor=p["bor"],
            wf1=p["wf1"],
            wf1_1to1=wf1_1to1,
            exact_f1=exact_f1,
            coverage=p["coverage"],
            purity=0.0,
            n_pred_boundaries=p["n_pred_boundaries"],
            n_gold_boundaries=p["n_gold_boundaries"],
            n_dialogues=metadata["n_dialogues"],
            per_dialogue_wf1=p["per_dialogue_wf1"],
            per_dialogue_wf1_1to1=per_dialogue_wf1_1to1,
            per_dialogue_exact_f1=per_dialogue_exact_f1,
            per_dialogue_coverage=p["per_dialogue_coverage"],
            seed=p.get("seed"),
        ))

    return points


# =============================================================================
# PLOTTING
# =============================================================================

def plot_density_quality_curves(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
    display_name: str,
    output_dir: Path,
    *,
    with_regime_overlays: bool = False,
    output_suffix: str = "",
):
    """
    Generate publication-ready density-quality curve plots.

    Args:
        results: Dict mapping model name to sweep DataFrame
        dataset_name: Dataset identifier (e.g., "dialseg711")
        display_name: Human-readable name (e.g., "DialSeg711")
        output_dir: Directory to save figures
        with_regime_overlays: If True, add BOR regime shading and labels
        output_suffix: Suffix to add to output filenames (e.g., "_regimes")
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Models to plot (in order)
    model_order = ["neural", "texttiling", "csm", "random"]
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    # Track maximum x value across all plotted data
    xmax_data = 0.0

    for model in model_order:
        if model not in results:
            continue

        df = results[model]
        if df.empty:
            continue

        color = COLORS.get(model, "#333333")
        label = model_labels.get(model, model)

        if model == "random":
            # Aggregate random runs
            grouped = df.groupby("step").agg({
                "bor": "mean",
                "wf1": ["mean", "std"],
                "coverage": ["mean", "std"],
            }).reset_index()

            bor = grouped["bor"]["mean"].values
            wf1_mean = grouped["wf1"]["mean"].values
            wf1_std = grouped["wf1"]["std"].values
            cov_mean = grouped["coverage"]["mean"].values
            cov_std = grouped["coverage"]["std"].values

            # Sort by BOR
            order = np.argsort(bor)
            bor = bor[order]
            wf1_mean = wf1_mean[order]
            wf1_std = wf1_std[order]
            cov_mean = cov_mean[order]
            cov_std = cov_std[order]

            # Track xmax
            if len(bor) > 0:
                xmax_data = max(xmax_data, bor.max())

            # W-F1 plot
            axes[0].plot(bor, wf1_mean, color=color, label=label, linewidth=1.5)
            axes[0].fill_between(bor, wf1_mean - wf1_std, wf1_mean + wf1_std,
                                color=color, alpha=0.2)

            # Coverage plot
            axes[1].plot(bor, cov_mean, color=color, label=label, linewidth=1.5)
            axes[1].fill_between(bor, cov_mean - cov_std, cov_mean + cov_std,
                                color=color, alpha=0.2)
        else:
            # Sort by BOR
            df_sorted = df.sort_values("bor")

            # Track xmax
            if len(df_sorted) > 0:
                xmax_data = max(xmax_data, df_sorted["bor"].max())

            # W-F1 plot
            axes[0].plot(df_sorted["bor"], df_sorted["wf1"],
                        color=color, label=label, linewidth=1.5)

            # Coverage plot
            axes[1].plot(df_sorted["bor"], df_sorted["coverage"],
                        color=color, label=label, linewidth=1.5)

    # Autoscale x-limits to data (with 5% padding)
    if xmax_data > 0:
        xlim_upper = xmax_data * 1.05
    else:
        xlim_upper = 5.5  # Fallback

    # Configure axes
    for ax in axes:
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(0, xlim_upper)
        ax.set_ylim(0, 1.05)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add regime overlays or simple BOR=1 line
        if with_regime_overlays:
            add_bor_regime_overlays(ax)
        else:
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    axes[0].set_ylabel("W-F1")
    axes[0].set_title(f"{display_name}: W-F1 vs Boundary Density")

    axes[1].set_ylabel("Coverage")
    axes[1].set_title(f"{display_name}: Coverage vs Boundary Density")

    # Add text annotation for gold density line (only if not using regime overlays)
    if not with_regime_overlays:
        for ax in axes:
            ax.text(1.02, 0.02, "gold\ndensity", fontsize=8, color="gray",
                   transform=ax.get_xaxis_transform(), va="bottom")

    # Legend outside plot (reduced whitespace)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"density_quality_{dataset_name}{output_suffix}.pdf"
    png_path = output_dir / f"density_quality_{dataset_name}{output_suffix}.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")
    log(f"  Saved: {png_path}")


def plot_density_quality_curves_with_ci(
    results: Dict[str, pd.DataFrame],
    ci_data: Dict[str, pd.DataFrame],
    dataset_name: str,
    display_name: str,
    output_dir: Path,
    *,
    with_regime_overlays: bool = False,
    output_suffix: str = "",
    ci_alpha: float = 0.15,
    show_error_bars: bool = False,
    error_bar_spacing: int = 10,
):
    """
    Generate density-quality curves with bootstrap confidence intervals.

    This extends plot_density_quality_curves() by adding CI bands (shaded regions)
    or error bars at selected operating points.

    Args:
        results: Dict mapping model name to sweep DataFrame (for line data)
        ci_data: Dict mapping model name to CI DataFrame from compute_sweep_bootstrap_cis
        dataset_name: Dataset identifier (e.g., "dialseg711")
        display_name: Human-readable name (e.g., "DialSeg711")
        output_dir: Directory to save figures
        with_regime_overlays: If True, add BOR regime shading
        output_suffix: Suffix for output filenames
        ci_alpha: Transparency for CI bands (default 0.15)
        show_error_bars: If True, show error bars instead of bands (less clutter)
        error_bar_spacing: Show error bars every N points (if show_error_bars=True)

    Note:
        Bootstrap CIs are computed by resampling dialogues with replacement and
        recomputing the macro-average metric. BOR is fixed per operating point.
        See compute_bootstrap_ci() for methodology details.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    model_order = ["neural", "texttiling", "csm", "random"]
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    xmax_data = 0.0

    for model in model_order:
        if model not in results:
            continue

        df = results[model]
        if df.empty:
            continue

        color = COLORS.get(model, "#333333")
        label = model_labels.get(model, model)

        # Get CI data for this model if available
        model_ci = ci_data.get(model, pd.DataFrame())

        if model == "random":
            # Aggregate random runs (existing behavior)
            grouped = df.groupby("step").agg({
                "bor": "mean",
                "wf1": ["mean", "std"],
                "coverage": ["mean", "std"],
            }).reset_index()

            bor = grouped["bor"]["mean"].values
            wf1_mean = grouped["wf1"]["mean"].values
            wf1_std = grouped["wf1"]["std"].values
            cov_mean = grouped["coverage"]["mean"].values
            cov_std = grouped["coverage"]["std"].values

            order = np.argsort(bor)
            bor = bor[order]
            wf1_mean = wf1_mean[order]
            wf1_std = wf1_std[order]
            cov_mean = cov_mean[order]
            cov_std = cov_std[order]

            if len(bor) > 0:
                xmax_data = max(xmax_data, bor.max())

            # W-F1 plot with std bands (random uses std, not bootstrap CI)
            axes[0].plot(bor, wf1_mean, color=color, label=label, linewidth=1.5)
            axes[0].fill_between(bor, wf1_mean - wf1_std, wf1_mean + wf1_std,
                                color=color, alpha=0.2)

            # Coverage plot
            axes[1].plot(bor, cov_mean, color=color, label=label, linewidth=1.5)
            axes[1].fill_between(bor, cov_mean - cov_std, cov_mean + cov_std,
                                color=color, alpha=0.2)
        else:
            # Sort by BOR
            df_sorted = df.sort_values("bor")

            if len(df_sorted) > 0:
                xmax_data = max(xmax_data, df_sorted["bor"].max())

            bor = df_sorted["bor"].values
            wf1 = df_sorted["wf1"].values
            cov = df_sorted["coverage"].values

            # Plot main lines
            axes[0].plot(bor, wf1, color=color, label=label, linewidth=1.5)
            axes[1].plot(bor, cov, color=color, label=label, linewidth=1.5)

            # Add CI bands/error bars if available
            if not model_ci.empty:
                # Extract W-F1 CIs
                wf1_ci = model_ci[model_ci["metric"] == "wf1"].sort_values("bor")
                cov_ci = model_ci[model_ci["metric"] == "coverage"].sort_values("bor")

                if not wf1_ci.empty:
                    ci_bor = wf1_ci["bor"].values
                    ci_lo = wf1_ci["ci_low"].values
                    ci_hi = wf1_ci["ci_high"].values

                    if show_error_bars:
                        # Error bars at selected points
                        indices = range(0, len(ci_bor), error_bar_spacing)
                        axes[0].errorbar(
                            ci_bor[list(indices)],
                            wf1_ci["estimate"].values[list(indices)],
                            yerr=[
                                wf1_ci["estimate"].values[list(indices)] - ci_lo[list(indices)],
                                ci_hi[list(indices)] - wf1_ci["estimate"].values[list(indices)]
                            ],
                            fmt='none', ecolor=color, alpha=0.5, capsize=2
                        )
                    else:
                        # Shaded CI band
                        axes[0].fill_between(ci_bor, ci_lo, ci_hi,
                                            color=color, alpha=ci_alpha)

                if not cov_ci.empty:
                    ci_bor = cov_ci["bor"].values
                    ci_lo = cov_ci["ci_low"].values
                    ci_hi = cov_ci["ci_high"].values

                    if show_error_bars:
                        indices = range(0, len(ci_bor), error_bar_spacing)
                        axes[1].errorbar(
                            ci_bor[list(indices)],
                            cov_ci["estimate"].values[list(indices)],
                            yerr=[
                                cov_ci["estimate"].values[list(indices)] - ci_lo[list(indices)],
                                ci_hi[list(indices)] - cov_ci["estimate"].values[list(indices)]
                            ],
                            fmt='none', ecolor=color, alpha=0.5, capsize=2
                        )
                    else:
                        axes[1].fill_between(ci_bor, ci_lo, ci_hi,
                                            color=color, alpha=ci_alpha)

    # Fixed x-limits for consistent alignment across datasets
    xlim_upper = 2.1  # Fixed upper bound for all figures

    # Configure axes
    for ax in axes:
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(0, xlim_upper)
        ax.set_ylim(0, 1.05)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if with_regime_overlays:
            add_bor_regime_overlays(ax)
        else:
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    axes[0].set_ylabel("W-F1")
    # Title removed per request

    axes[1].set_ylabel("Coverage")
    # Title removed per request

    if not with_regime_overlays:
        for ax in axes:
            ax.text(1.02, 0.02, "gold\ndensity", fontsize=8, color="gray",
                   transform=ax.get_xaxis_transform(), va="bottom")

    # Legend (reduced whitespace)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Add dataset label below y-axis, left-aligned with x=0
    # Use blended transform: x in data coords, y in axes coords
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    ylabel_fontsize = axes[0].yaxis.label.get_size()
    axes[0].text(0, -0.12, display_name, transform=trans,
                fontsize=ylabel_fontsize, ha='left', va='top')

    # Save with v2 suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"density_quality_{dataset_name}_ci{output_suffix}_v2.pdf"
    png_path = output_dir / f"density_quality_{dataset_name}_ci{output_suffix}_v2.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")
    log(f"  Saved: {png_path}")


def plot_matching_comparison(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
    display_name: str,
    output_dir: Path,
    ci_data: Optional[Dict[str, pd.DataFrame]] = None,
    with_regime_overlays: bool = True,
    ci_alpha: float = 0.15,
):
    """
    Generate comparison plot of many-to-one vs one-to-one tolerant matching.

    This plot demonstrates that the density-dominates effect holds regardless
    of matching semantics, addressing the reviewer concern about metric choice.

    Left panel: W-F1 with many-to-one matching (paper's main metric)
    Right panel: W-F1 with one-to-one matching (standard tolerant matching)

    Args:
        results: Dict mapping model name to sweep DataFrame
        dataset_name: Dataset identifier
        display_name: Human-readable dataset name
        output_dir: Directory to save figures
        ci_data: Optional dict mapping model name to CI DataFrame
        with_regime_overlays: If True, add BOR regime shading
        ci_alpha: Transparency for CI bands
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if ci_data is None:
        ci_data = {}

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    model_order = ["neural", "texttiling", "csm", "random"]
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    # Define which metrics to plot in each panel
    # Map sweep column to CI metric name (CI uses "wf1" for many-to-one)
    panels = [
        ("wf1", "wf1", "W-F1 (Many-to-One Matching)"),
        ("wf1_1to1", "wf1_1to1", "W-F1 (One-to-One Matching)"),
    ]

    for ax_idx, (sweep_metric, ci_metric, title) in enumerate(panels):
        ax = axes[ax_idx]
        xmax_data = 0.0

        for model in model_order:
            if model not in results:
                continue

            df = results[model]
            if df.empty or sweep_metric not in df.columns:
                continue

            color = COLORS.get(model, "#333333")
            label = model_labels.get(model, model)
            model_ci = ci_data.get(model, pd.DataFrame())

            if model == "random":
                # Aggregate random runs
                grouped = df.groupby("step").agg({
                    "bor": "mean",
                    sweep_metric: ["mean", "std"],
                }).reset_index()

                bor = grouped["bor"]["mean"].values
                val_mean = grouped[sweep_metric]["mean"].values
                val_std = grouped[sweep_metric]["std"].values

                order = np.argsort(bor)
                bor = bor[order]
                val_mean = val_mean[order]
                val_std = val_std[order]

                if len(bor) > 0:
                    xmax_data = max(xmax_data, bor.max())

                ax.plot(bor, val_mean, color=color, label=label, linewidth=1.5)
                ax.fill_between(bor, val_mean - val_std, val_mean + val_std,
                               color=color, alpha=0.2)
            else:
                df_sorted = df.sort_values("bor")

                if len(df_sorted) > 0:
                    xmax_data = max(xmax_data, df_sorted["bor"].max())

                ax.plot(df_sorted["bor"], df_sorted[sweep_metric],
                       color=color, label=label, linewidth=1.5)

                # Add CI bands if available
                if not model_ci.empty and "metric" in model_ci.columns:
                    metric_ci = model_ci[model_ci["metric"] == ci_metric].sort_values("bor")
                    if not metric_ci.empty:
                        ci_bor = metric_ci["bor"].values
                        ci_lo = metric_ci["ci_low"].values
                        ci_hi = metric_ci["ci_high"].values
                        ax.fill_between(ci_bor, ci_lo, ci_hi,
                                       color=color, alpha=ci_alpha)

        # Configure axis
        xlim_upper = min(xmax_data * 1.05, 2.5) if xmax_data > 0 else 2.1
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(0, xlim_upper)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("W-F1")
        ax.set_title(title)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if with_regime_overlays:
            add_bor_regime_overlays(ax)
        else:
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Add dataset label below y-axis, left-aligned with x=0
    # Use blended transform: x in data coords, y in axes coords
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(axes[0].transData, axes[0].transAxes)
    ylabel_fontsize = axes[0].yaxis.label.get_size()
    axes[0].text(0, -0.12, display_name, transform=trans,
                fontsize=ylabel_fontsize, ha='left', va='top')

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"matching_comparison_{dataset_name}.pdf"
    png_path = output_dir / f"matching_comparison_{dataset_name}.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")
    log(f"  Saved: {png_path}")


def plot_exact_f1_curves(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
    display_name: str,
    output_dir: Path,
    ci_data: Optional[Dict[str, pd.DataFrame]] = None,
    with_regime_overlays: bool = True,
    ci_alpha: float = 0.15,
):
    """
    Generate Exact F1 (strict position matching) density-quality curves.

    This plot demonstrates that the density confound persists even under
    strict position matching with zero tolerance window.

    Args:
        results: Dict mapping model name to sweep DataFrame
        dataset_name: Dataset identifier
        display_name: Human-readable dataset name
        output_dir: Directory to save figures
        ci_data: Optional dict mapping model name to CI DataFrame
        with_regime_overlays: If True, add BOR regime shading
        ci_alpha: Transparency for CI bands
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if ci_data is None:
        ci_data = {}

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    model_order = ["neural", "texttiling", "csm", "random"]
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    xmax_data = 0.0

    for model in model_order:
        if model not in results:
            continue

        df = results[model]
        if df.empty or "exact_f1" not in df.columns:
            continue

        color = COLORS.get(model, "#333333")
        label = model_labels.get(model, model)
        model_ci = ci_data.get(model, pd.DataFrame())

        if model == "random":
            # Aggregate random runs
            grouped = df.groupby("step").agg({
                "bor": "mean",
                "exact_f1": ["mean", "std"],
            }).reset_index()

            bor = grouped["bor"]["mean"].values
            val_mean = grouped["exact_f1"]["mean"].values
            val_std = grouped["exact_f1"]["std"].values

            order = np.argsort(bor)
            bor = bor[order]
            val_mean = val_mean[order]
            val_std = val_std[order]

            if len(bor) > 0:
                xmax_data = max(xmax_data, bor.max())

            ax.plot(bor, val_mean, color=color, label=label, linewidth=1.5)
            ax.fill_between(bor, val_mean - val_std, val_mean + val_std,
                           color=color, alpha=0.2)
        else:
            df_sorted = df.sort_values("bor")

            if len(df_sorted) > 0:
                xmax_data = max(xmax_data, df_sorted["bor"].max())

            ax.plot(df_sorted["bor"], df_sorted["exact_f1"],
                   color=color, label=label, linewidth=1.5)

            # Add CI bands if available
            if not model_ci.empty and "metric" in model_ci.columns:
                metric_ci = model_ci[model_ci["metric"] == "exact_f1"].sort_values("bor")
                if not metric_ci.empty:
                    ci_bor = metric_ci["bor"].values
                    ci_lo = metric_ci["ci_low"].values
                    ci_hi = metric_ci["ci_high"].values
                    ax.fill_between(ci_bor, ci_lo, ci_hi,
                                   color=color, alpha=ci_alpha)

    # Configure axis
    xlim_upper = min(xmax_data * 1.05, 2.5) if xmax_data > 0 else 2.1
    ax.set_xlabel("BOR (Boundary Density)")
    ax.set_xlim(0, xlim_upper)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact F1")
    ax.set_title(f"{display_name}: Exact F1 vs Boundary Density")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if with_regime_overlays:
        add_bor_regime_overlays(ax)
    else:
        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"density_quality_exact_f1_{dataset_name}.pdf"
    png_path = output_dir / f"density_quality_exact_f1_{dataset_name}.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")
    log(f"  Saved: {png_path}")


def plot_exact_f1_combined(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    ci_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    with_regime_overlays: bool = True,
    ci_alpha: float = 0.15,
):
    """
    Generate combined Exact F1 plot for multiple datasets (side-by-side panels).

    Args:
        all_results: Dict mapping dataset_name to Dict[model_name, DataFrame]
        output_dir: Directory to save figures
        ci_data: Optional dict mapping dataset_name to Dict[model_name, CI DataFrame]
        with_regime_overlays: If True, add BOR regime shading
        ci_alpha: Transparency for CI bands
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if ci_data is None:
        ci_data = {}

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]

    display_names = {
        "dialseg711": "DialSeg711",
        "superseg": "SuperSeg",
    }

    model_order = ["neural", "texttiling", "csm", "random"]
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    for ax_idx, dataset_name in enumerate(datasets):
        ax = axes[ax_idx]
        results = all_results[dataset_name]
        dataset_ci = ci_data.get(dataset_name, {})
        display_name = display_names.get(dataset_name, dataset_name)

        xmax_data = 0.0

        for model in model_order:
            if model not in results:
                continue

            df = results[model]
            if df.empty or "exact_f1" not in df.columns:
                continue

            color = COLORS.get(model, "#333333")
            label = model_labels.get(model, model)
            model_ci = dataset_ci.get(model, pd.DataFrame())

            if model == "random":
                grouped = df.groupby("step").agg({
                    "bor": "mean",
                    "exact_f1": ["mean", "std"],
                }).reset_index()

                bor = grouped["bor"]["mean"].values
                val_mean = grouped["exact_f1"]["mean"].values
                val_std = grouped["exact_f1"]["std"].values

                order = np.argsort(bor)
                bor = bor[order]
                val_mean = val_mean[order]
                val_std = val_std[order]

                if len(bor) > 0:
                    xmax_data = max(xmax_data, bor.max())

                ax.plot(bor, val_mean, color=color, label=label, linewidth=1.5)
                ax.fill_between(bor, val_mean - val_std, val_mean + val_std,
                               color=color, alpha=0.2)
            else:
                df_sorted = df.sort_values("bor")

                if len(df_sorted) > 0:
                    xmax_data = max(xmax_data, df_sorted["bor"].max())

                ax.plot(df_sorted["bor"], df_sorted["exact_f1"],
                       color=color, label=label, linewidth=1.5)

                if not model_ci.empty and "metric" in model_ci.columns:
                    metric_ci = model_ci[model_ci["metric"] == "exact_f1"].sort_values("bor")
                    if not metric_ci.empty:
                        ci_bor = metric_ci["bor"].values
                        ci_lo = metric_ci["ci_low"].values
                        ci_hi = metric_ci["ci_high"].values
                        ax.fill_between(ci_bor, ci_lo, ci_hi,
                                       color=color, alpha=ci_alpha)

        xlim_upper = min(xmax_data * 1.05, 2.5) if xmax_data > 0 else 2.1
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(0, xlim_upper)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Exact F1")
        ax.set_title(f"{display_name}")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if with_regime_overlays:
            add_bor_regime_overlays(ax)
        else:
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Shared legend (from first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "density_quality_exact_f1.pdf"
    png_path = output_dir / "density_quality_exact_f1.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")
    log(f"  Saved: {png_path}")


# =============================================================================
# AUC COMPUTATION
# =============================================================================

def compute_auc(df: pd.DataFrame, metric: str, bor_range: Tuple[float, float]) -> float:
    """
    Compute AUC as trapezoidal integral of metric vs log(BOR).

    Args:
        df: Sweep results with 'bor' and metric columns
        metric: Column name for metric ('wf1' or 'coverage')
        bor_range: (min_bor, max_bor) for integration

    Returns:
        AUC value
    """
    if df.empty:
        return 0.0

    # Filter to BOR range
    mask = (df["bor"] >= bor_range[0]) & (df["bor"] <= bor_range[1])
    df_filtered = df[mask].sort_values("bor")

    if len(df_filtered) < 2:
        return 0.0

    # Interpolate to common grid (log-spaced)
    log_bor = np.log(df_filtered["bor"].values)
    values = df_filtered[metric].values

    # Trapezoidal integration
    auc = np.trapz(values, log_bor)

    # Normalize by log range
    log_range = np.log(bor_range[1]) - np.log(bor_range[0])
    auc_normalized = auc / log_range

    return float(auc_normalized)


# =============================================================================
# CSV-BASED FIGURE GENERATION
# =============================================================================

def load_sweep_from_csv(dataset: str, method: str) -> pd.DataFrame:
    """Load sweep data from pre-computed CSV file."""
    csv_path = RESULTS_DIR / f"sweep_{dataset}_{method}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def generate_exact_f1_from_csv():
    """
    Generate exact F1 density-quality figure from pre-computed sweep CSVs.

    This reads directly from paper/results/sweep_{dataset}_{method}.csv
    and uses the exact_f1 column, without recomputing predictions.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    log("=" * 70)
    log("Generating Exact F1 Figure from Sweep CSVs")
    log("=" * 70)

    # Publication settings
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    datasets = ["dialseg711", "superseg"]
    methods = ["neural", "texttiling", "csm", "random"]
    display_names = {
        "dialseg711": "DialSeg711",
        "superseg": "SuperSeg",
    }
    model_labels = {
        "neural": "Proposed (Neural)",
        "texttiling": "TextTiling",
        "csm": "CSM (NSP)",
        "random": "Random",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        display_name = display_names[dataset]
        xmax_data = 0.0

        log(f"\n--- {display_name} ---")

        for method in methods:
            try:
                df = load_sweep_from_csv(dataset, method)
            except FileNotFoundError as e:
                log(f"  {method}: SKIP - {e}")
                continue

            if "exact_f1" not in df.columns:
                log(f"  {method}: SKIP - no exact_f1 column")
                continue

            color = COLORS.get(method, "#333333")
            label = model_labels.get(method, method)

            if method == "random":
                # Aggregate random runs by step
                grouped = df.groupby("step").agg({
                    "bor": "mean",
                    "exact_f1": ["mean", "std"],
                }).reset_index()

                bor = grouped["bor"]["mean"].values
                val_mean = grouped["exact_f1"]["mean"].values
                val_std = grouped["exact_f1"]["std"].values

                order = np.argsort(bor)
                bor = bor[order]
                val_mean = val_mean[order]
                val_std = val_std[order]

                if len(bor) > 0:
                    xmax_data = max(xmax_data, bor.max())
                    best_idx = np.argmax(val_mean)
                    log(f"  {method}: best exact_f1={val_mean[best_idx]:.3f} at BOR={bor[best_idx]:.2f}, "
                        f"BOR range=[{bor.min():.2f}, {bor.max():.2f}]")

                ax.plot(bor, val_mean, color=color, label=label, linewidth=1.5)
                ax.fill_between(bor, val_mean - val_std, val_mean + val_std,
                               color=color, alpha=0.2)
            else:
                df_sorted = df.sort_values("bor")
                bor = df_sorted["bor"].values
                exact_f1 = df_sorted["exact_f1"].values

                if len(bor) > 0:
                    xmax_data = max(xmax_data, bor.max())
                    best_idx = np.argmax(exact_f1)
                    log(f"  {method}: best exact_f1={exact_f1[best_idx]:.3f} at BOR={bor[best_idx]:.2f}, "
                        f"BOR range=[{bor.min():.2f}, {bor.max():.2f}]")

                ax.plot(bor, exact_f1, color=color, label=label, linewidth=1.5)

        # Configure axis
        xlim_upper = min(xmax_data * 1.05, 2.5) if xmax_data > 0 else 2.1
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(0, xlim_upper)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Exact F1")
        ax.set_title(f"{display_name}")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add regime overlays
        add_bor_regime_overlays(ax)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "density_quality_exact_f1.pdf"
    png_path = FIGURES_DIR / "density_quality_exact_f1.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"\nSaved: {pdf_path}")
    log(f"Saved: {png_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    log("=" * 70)
    log("Density-Quality Curve Analysis")
    log(f"Min gap (g): {MIN_GAP}")
    log(f"Sweep steps: {N_SWEEP_STEPS}")
    log(f"Max BOR: {MAX_BOR}")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_auc_results = []
    all_results_by_dataset = {}  # For combined exact F1 plot

    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info["display"]
        log(f"\n{'='*70}")
        log(f"Dataset: {display_name}")
        log("=" * 70)

        # Load data
        try:
            dialogues = load_dataset(dataset_name)
            log(f"Loaded {len(dialogues)} dialogues")
            total_gold = sum(len(d.gold_boundaries) for d in dialogues)
            log(f"Total gold boundaries: {total_gold}")
        except FileNotFoundError as e:
            log(f"SKIP: {e}")
            continue

        results = {}

        # Neural scorer
        log("\n--- Neural Scorer ---")
        try:
            neural_scores = get_neural_scores(dialogues)
            df_neural = run_sweep(dialogues, neural_scores, dataset_name, "neural")
            results["neural"] = df_neural
            log(f"  Sweep points: {len(df_neural)}")
            df_neural.to_csv(RESULTS_DIR / f"sweep_{dataset_name}_neural.csv", index=False)
        except Exception as e:
            log(f"  ERROR: {e}")

        # TextTiling
        log("\n--- TextTiling ---")
        try:
            tt_scores = get_texttiling_scores(dialogues)
            df_tt = run_sweep(dialogues, tt_scores, dataset_name, "texttiling")
            results["texttiling"] = df_tt
            log(f"  Sweep points: {len(df_tt)}")
            df_tt.to_csv(RESULTS_DIR / f"sweep_{dataset_name}_texttiling.csv", index=False)
        except Exception as e:
            log(f"  ERROR: {e}")

        # CSM
        log("\n--- CSM (NSP) ---")
        try:
            csm_scores = get_csm_scores(dialogues)
            df_csm = run_sweep(dialogues, csm_scores, dataset_name, "csm")
            results["csm"] = df_csm
            log(f"  Sweep points: {len(df_csm)}")
            df_csm.to_csv(RESULTS_DIR / f"sweep_{dataset_name}_csm.csv", index=False)
        except Exception as e:
            log(f"  ERROR: {e}")

        # Random (10 seeds)
        log("\n--- Random (10 seeds) ---")
        try:
            random_dfs = []
            for seed in RANDOM_SEEDS:
                random_scores = get_random_scores(dialogues, seed)
                df_rand = run_sweep(dialogues, random_scores, dataset_name, "random", seed=seed)
                random_dfs.append(df_rand)

            df_random = pd.concat(random_dfs, ignore_index=True)
            results["random"] = df_random
            log(f"  Total sweep points: {len(df_random)} (across {len(RANDOM_SEEDS)} seeds)")
            df_random.to_csv(RESULTS_DIR / f"sweep_{dataset_name}_random.csv", index=False)
        except Exception as e:
            log(f"  ERROR: {e}")

        # Generate plots
        log("\n--- Generating plots ---")
        plot_density_quality_curves(results, dataset_name, display_name, FIGURES_DIR)

        # Generate plots with regime overlays
        log("\n--- Generating plots with regime overlays ---")
        plot_density_quality_curves(
            results, dataset_name, display_name, FIGURES_DIR,
            with_regime_overlays=True, output_suffix="_regimes"
        )

        # Generate matching comparison plot (many-to-one vs one-to-one)
        log("\n--- Generating matching comparison plot ---")
        plot_matching_comparison(results, dataset_name, display_name, FIGURES_DIR)

        # Generate exact F1 plot
        log("\n--- Generating exact F1 plot ---")
        plot_exact_f1_curves(results, dataset_name, display_name, FIGURES_DIR)

        # Store results for combined plot
        all_results_by_dataset[dataset_name] = results

        # Compute AUC
        log("\n--- AUC Summary ---")
        for model, df in results.items():
            if df.empty:
                continue

            if model == "random":
                # Aggregate first
                grouped = df.groupby("step").agg({
                    "bor": "mean", "wf1": "mean", "wf1_1to1": "mean", "coverage": "mean"
                }).reset_index()
                auc_wf1 = compute_auc(grouped, "wf1", BOR_RANGE)
                auc_wf1_1to1 = compute_auc(grouped, "wf1_1to1", BOR_RANGE)
                auc_cov = compute_auc(grouped, "coverage", BOR_RANGE)
            else:
                auc_wf1 = compute_auc(df, "wf1", BOR_RANGE)
                auc_wf1_1to1 = compute_auc(df, "wf1_1to1", BOR_RANGE)
                auc_cov = compute_auc(df, "coverage", BOR_RANGE)

            log(f"  {model}: AUC(W-F1)={auc_wf1:.3f}, AUC(W-F1-1to1)={auc_wf1_1to1:.3f}, AUC(Coverage)={auc_cov:.3f}")

            all_auc_results.append({
                "dataset": dataset_name,
                "model": model,
                "auc_wf1": auc_wf1,
                "auc_wf1_1to1": auc_wf1_1to1,
                "auc_coverage": auc_cov,
            })

    # Save AUC summary
    if all_auc_results:
        auc_df = pd.DataFrame(all_auc_results)
        auc_path = RESULTS_DIR / "auc_summary.csv"
        auc_df.to_csv(auc_path, index=False)
        log(f"\nAUC summary saved: {auc_path}")

        # Print final table
        log("\n" + "=" * 70)
        log("AUC SUMMARY")
        log("=" * 70)
        print(auc_df.to_string(index=False))

    # Generate combined exact F1 plot for all datasets
    if all_results_by_dataset:
        log("\n--- Generating combined exact F1 plot ---")
        plot_exact_f1_combined(all_results_by_dataset, FIGURES_DIR)

    log("\nDone.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--exact-f1-only":
        # Generate exact F1 figure from existing CSVs without recomputing
        generate_exact_f1_from_csv()
    else:
        main()
