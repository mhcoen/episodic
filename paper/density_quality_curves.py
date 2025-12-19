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
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
)

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

# Sweep configuration
MIN_GAP = 2  # Fixed minimum spacing between boundaries (in user turns)
N_SWEEP_STEPS = 200  # Number of threshold steps
MAX_BOR = 5.0  # Stop sweep when BOR exceeds this
BOR_RANGE = (0.2, 5.0)  # Range for AUC computation
RANDOM_SEEDS = list(range(10))  # Seeds for random baseline

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
        # Small label for the line
        ax.text(1.02, 0.92, "BOR=1\n(density-matched)", transform=trans,
                fontsize=7, ha="left", va="top", color="#444444", alpha=0.9)

    # Shade BOR < 1 region (if data exists there)
    if x0 < 1.0:
        ax.axvspan(x0, min(1.0, x1), color=REGIME_COLORS["conservative"],
                   alpha=alpha, zorder=0, label="_nolegend_")
        # Small corner annotation
        ax.text(x0 + 0.05, 0.97, "BOR<1", transform=trans, fontsize=7,
                ha="left", va="top", color=REGIME_COLORS["conservative"], alpha=0.7)

    # Shade BOR > 1 region only if xlim extends nontrivially beyond 1
    if x1 > 1.2:
        ax.axvspan(max(1.0, x0), x1, color=REGIME_COLORS["aggressive"],
                   alpha=alpha, zorder=0, label="_nolegend_")
        # Small corner annotation
        ax.text(x1 - 0.05, 0.97, "BOR>1", transform=trans, fontsize=7,
                ha="right", va="top", color=REGIME_COLORS["aggressive"], alpha=0.7)

    # Reset xlim in case axvspan changed it
    ax.set_xlim(x0, x1)


@dataclass
class DialogueData:
    """Container for a single dialogue with boundaries."""
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
    wf1: float
    coverage: float
    purity: float
    n_pred_boundaries: int
    n_gold_boundaries: int
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
                messages=messages,
                gold_boundaries=boundaries,
                num_messages=num_user_turns
            ))

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
    model_path = PROJECT_ROOT / "paper" / "experiments" / "models" / "final_calibrated.pt"
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
        all_purity = []
        all_coverage = []
        total_pred = 0

        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(
                scores, tau, min_gap, dialogue.num_messages
            )
            all_preds.append(pred)
            total_pred += len(pred)

            # W-F1
            _, _, wf1 = compute_windowed_metrics(
                dialogue.gold_boundaries, pred, dialogue.num_messages, window=1
            )
            all_wf1.append(wf1)

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
            coverage=float(np.mean(all_coverage)),
            purity=float(np.mean(all_purity)),
            n_pred_boundaries=total_pred,
            n_gold_boundaries=total_gold,
            seed=seed,
        ))

        prev_bor = bor

    return pd.DataFrame([asdict(r) for r in results])


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

    # Legend outside plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"density_quality_{dataset_name}{output_suffix}.pdf"
    png_path = output_dir / f"density_quality_{dataset_name}{output_suffix}.png"

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

        # Compute AUC
        log("\n--- AUC Summary ---")
        for model, df in results.items():
            if df.empty:
                continue

            if model == "random":
                # Aggregate first
                grouped = df.groupby("step").agg({
                    "bor": "mean", "wf1": "mean", "coverage": "mean"
                }).reset_index()
                auc_wf1 = compute_auc(grouped, "wf1", BOR_RANGE)
                auc_cov = compute_auc(grouped, "coverage", BOR_RANGE)
            else:
                auc_wf1 = compute_auc(df, "wf1", BOR_RANGE)
                auc_cov = compute_auc(df, "coverage", BOR_RANGE)

            log(f"  {model}: AUC(W-F1)={auc_wf1:.3f}, AUC(Coverage)={auc_cov:.3f}")

            all_auc_results.append({
                "dataset": dataset_name,
                "model": model,
                "auc_wf1": auc_wf1,
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

    log("\nDone.")


if __name__ == "__main__":
    main()
