#!/usr/bin/env python3
"""
Density-Quality Diagnostics: Additional figures for granularity-mismatch analysis.

This script generates supplementary diagnostic figures using the existing sweep
data from paper/results/sweep_{dataset}_{model}.csv.

Figure A: Local Robustness Around Gold Density (BOR≈1)
    Shows W-F1 and Coverage vs BOR in a narrow window around BOR=1.0 to
    demonstrate whether conclusions at "gold density" are stable or threshold-fragile.
    Window: BOR ∈ [0.7, 1.3]

Figure B: Per-Dialogue BOR Distribution at Operating Point
    Shows violin/histogram of per-dialogue BOR values at the sweep point closest
    to global BOR=1.0. Demonstrates that global BOR can hide substantial
    per-dialogue over/under-segmentation.

    Zero-gold-boundary handling: Dialogues with 0 gold boundaries are EXCLUDED
    from per-dialogue BOR computation (BOR undefined). The number of excluded
    dialogues is logged and noted in the figure caption.

Usage:
    python paper/density_quality_diagnostics.py

Outputs:
    - paper/figures/robustness_near_bor1_{dataset}.pdf/png
    - paper/figures/per_dialogue_bor_{dataset}.pdf/png
    - paper/results/per_dialogue_bor_{dataset}_{model}.csv

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

# Configuration
MIN_GAP = 2
BOR_WINDOW = (0.7, 1.3)  # Window around gold density for Figure A
RANDOM_SEEDS = list(range(10))

# Datasets
DATASETS = {
    "dialseg711": {"display": "DialSeg711"},
    "superseg": {"display": "SuperSeg"},
}

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "neural": "#0072B2",      # Blue
    "texttiling": "#E69F00",  # Orange
    "csm": "#009E73",         # Green
    "random": "#CC79A7",      # Pink/magenta
}

MODEL_LABELS = {
    "neural": "Proposed (Neural)",
    "texttiling": "TextTiling",
    "csm": "CSM (NSP)",
    "random": "Random",
}


@dataclass
class DialogueData:
    """Container for a single dialogue with boundaries."""
    dialogue_id: int
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_messages: int


def log(msg: str):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sweep_csv(dataset: str, model: str) -> pd.DataFrame:
    """Load existing sweep CSV."""
    path = RESULTS_DIR / f"sweep_{dataset}_{model}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
# SCORING FUNCTIONS (reused from main script)
# =============================================================================

def get_neural_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """Get per-position scores from the neural model."""
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

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
    all_scores = []

    for dialogue in tqdm(dialogues, desc="  Neural scoring", leave=False):
        messages = dialogue.messages
        scores = {}

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    context_parts = []
                    for m in window[-6:]:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        context_parts.append(f"{role}: {content}")

                    curr_content = msg.get("content", "")
                    text = " [SEP] ".join(context_parts) + f" [SEP] current: {curr_content}"

                    encoding = tokenizer(
                        text, max_length=256, padding="max_length",
                        truncation=True, return_tensors="pt"
                    )

                    with torch.no_grad():
                        inputs = {k: v.to(device) for k, v in encoding.items()}
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits / temperature, dim=-1)
                        score = probs[0, 1].item()

                    scores[user_idx] = score
                user_idx += 1

        all_scores.append(scores)

    return all_scores


def get_texttiling_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """Get per-position depth scores from TextTiling."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    all_scores = []

    for dialogue in tqdm(dialogues, desc="  TextTiling scoring", leave=False):
        user_utterances = [
            msg["content"] for msg in dialogue.messages if msg["role"] == "user"
        ]

        if len(user_utterances) < 3:
            all_scores.append({})
            continue

        embeddings = model.encode(user_utterances, convert_to_numpy=True)

        similarities = []
        for i in range(len(embeddings) - 1):
            e1, e2 = embeddings[i], embeddings[i + 1]
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            similarities.append(sim)

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

        scores = {}
        for i, depth in enumerate(depth_scores):
            pos = i + 1
            if 1 <= pos < len(user_utterances):
                scores[pos] = depth

        all_scores.append(scores)

    return all_scores


def get_csm_scores(dialogues: List[DialogueData]) -> List[Dict[int, float]]:
    """Get per-position depth scores from CSM (NSP-based)."""
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
    """Get random uniform scores for each position."""
    rng = np.random.RandomState(seed)
    all_scores = []

    for dialogue in dialogues:
        num_user_turns = dialogue.num_messages
        scores = {}
        for pos in range(1, num_user_turns):
            scores[pos] = rng.random()
        all_scores.append(scores)

    return all_scores


def apply_threshold_with_gap(
    scores: Dict[int, float],
    threshold: float,
    min_gap: int,
    num_messages: int
) -> Set[int]:
    """Apply threshold to scores with min_gap enforcement."""
    candidates = [
        (pos, score) for pos, score in scores.items()
        if score >= threshold and 1 <= pos < num_messages
    ]
    candidates.sort(key=lambda x: -x[1])

    selected = set()
    for pos, _ in candidates:
        ok = True
        for existing in selected:
            if abs(pos - existing) < min_gap:
                ok = False
                break
        if ok:
            selected.add(pos)

    return selected


# =============================================================================
# FIGURE A: LOCAL ROBUSTNESS AROUND BOR≈1
# =============================================================================

def plot_robustness_near_bor1(dataset_name: str, display_name: str, output_dir: Path):
    """
    Plot W-F1 and Coverage vs BOR in a narrow window around BOR=1.0.

    Demonstrates whether conclusions at "gold density" are stable or threshold-fragile.
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ["neural", "texttiling", "csm", "random"]

    for model in models:
        df = load_sweep_csv(dataset_name, model)
        if df.empty:
            continue

        # For random, aggregate across seeds
        if model == "random":
            df_agg = df.groupby("step").agg({
                "bor": "mean",
                "wf1": ["mean", "std"],
                "coverage": ["mean", "std"],
            }).reset_index()
            df_agg.columns = ["step", "bor", "wf1_mean", "wf1_std", "cov_mean", "cov_std"]

            # Filter to BOR window
            mask = (df_agg["bor"] >= BOR_WINDOW[0]) & (df_agg["bor"] <= BOR_WINDOW[1])
            df_filtered = df_agg[mask].sort_values("bor")

            if len(df_filtered) < 2:
                continue

            color = COLORS.get(model, "#333333")
            label = MODEL_LABELS.get(model, model)

            axes[0].plot(df_filtered["bor"], df_filtered["wf1_mean"],
                        color=color, label=label, linewidth=1.5)
            axes[0].fill_between(df_filtered["bor"],
                                df_filtered["wf1_mean"] - df_filtered["wf1_std"],
                                df_filtered["wf1_mean"] + df_filtered["wf1_std"],
                                color=color, alpha=0.2)

            axes[1].plot(df_filtered["bor"], df_filtered["cov_mean"],
                        color=color, label=label, linewidth=1.5)
            axes[1].fill_between(df_filtered["bor"],
                                df_filtered["cov_mean"] - df_filtered["cov_std"],
                                df_filtered["cov_mean"] + df_filtered["cov_std"],
                                color=color, alpha=0.2)
        else:
            # Filter to BOR window
            mask = (df["bor"] >= BOR_WINDOW[0]) & (df["bor"] <= BOR_WINDOW[1])
            df_filtered = df[mask].sort_values("bor")

            if len(df_filtered) < 2:
                continue

            color = COLORS.get(model, "#333333")
            label = MODEL_LABELS.get(model, model)

            axes[0].plot(df_filtered["bor"], df_filtered["wf1"],
                        color=color, label=label, linewidth=1.5)
            axes[1].plot(df_filtered["bor"], df_filtered["coverage"],
                        color=color, label=label, linewidth=1.5)

    # Configure axes
    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("BOR (Boundary Density)")
        ax.set_xlim(BOR_WINDOW[0] - 0.05, BOR_WINDOW[1] + 0.05)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel("W-F1")
    axes[0].set_title(f"{display_name}: W-F1 Near Gold Density")

    axes[1].set_ylabel("Coverage")
    axes[1].set_title(f"{display_name}: Coverage Near Gold Density")

    # Add annotation
    for ax in axes:
        ax.text(1.02, 0.02, "BOR=1", fontsize=8, color="gray",
               transform=ax.get_xaxis_transform(), va="bottom")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5),
              frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"robustness_near_bor1_{dataset_name}.pdf"
    png_path = output_dir / f"robustness_near_bor1_{dataset_name}.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")


# =============================================================================
# FIGURE B: PER-DIALOGUE BOR DISTRIBUTION
# =============================================================================

def compute_per_dialogue_bor(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    threshold: float,
    min_gap: int = MIN_GAP,
) -> pd.DataFrame:
    """
    Compute per-dialogue BOR at a given threshold.

    Dialogues with 0 gold boundaries are EXCLUDED (BOR undefined).
    """
    records = []

    for dialogue, scores in zip(dialogues, scores_list):
        n_gold = len(dialogue.gold_boundaries)

        # Skip dialogues with no gold boundaries
        if n_gold == 0:
            continue

        pred = apply_threshold_with_gap(scores, threshold, min_gap, dialogue.num_messages)
        n_pred = len(pred)
        bor = n_pred / n_gold

        records.append({
            "dialogue_id": dialogue.dialogue_id,
            "n_pred_boundaries": n_pred,
            "n_gold_boundaries": n_gold,
            "bor_dialogue": bor,
        })

    return pd.DataFrame(records)


def find_threshold_for_bor_target(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    target_bor: float = 1.0,
    min_gap: int = MIN_GAP,
) -> Tuple[float, float]:
    """Find threshold that produces closest to target global BOR."""
    # Collect all scores
    all_scores = []
    for scores in scores_list:
        all_scores.extend(scores.values())

    if not all_scores:
        return 0.5, 1.0

    total_gold = sum(len(d.gold_boundaries) for d in dialogues)

    # Binary search for threshold
    thresholds = np.percentile(all_scores, np.linspace(100, 0, 100))

    best_threshold = 0.5
    best_diff = float('inf')
    best_bor = 1.0

    for tau in thresholds:
        total_pred = 0
        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(scores, tau, min_gap, dialogue.num_messages)
            total_pred += len(pred)

        bor = total_pred / total_gold if total_gold > 0 else 0
        diff = abs(bor - target_bor)

        if diff < best_diff:
            best_diff = diff
            best_threshold = tau
            best_bor = bor

    return best_threshold, best_bor


def plot_per_dialogue_bor(
    dataset_name: str,
    display_name: str,
    output_dir: Path,
    dialogues: List[DialogueData],
    model_scores: Dict[str, List[Dict[int, float]]],
):
    """
    Plot violin plot of per-dialogue BOR at operating point closest to BOR=1.0.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10

    # Prepare data for each model
    all_data = {}
    n_excluded = 0

    # Count excluded dialogues (those with 0 gold boundaries)
    for d in dialogues:
        if len(d.gold_boundaries) == 0:
            n_excluded += 1

    for model, scores_list in model_scores.items():
        if not scores_list:
            continue

        # Find threshold for BOR≈1
        threshold, achieved_bor = find_threshold_for_bor_target(
            dialogues, scores_list, target_bor=1.0
        )

        log(f"    {model}: threshold={threshold:.4f}, achieved global BOR={achieved_bor:.3f}")

        # Compute per-dialogue BOR
        df_per_dial = compute_per_dialogue_bor(dialogues, scores_list, threshold)

        if not df_per_dial.empty:
            all_data[model] = df_per_dial["bor_dialogue"].values

            # Save audit CSV
            csv_path = RESULTS_DIR / f"per_dialogue_bor_{dataset_name}_{model}.csv"
            df_per_dial.to_csv(csv_path, index=False)
            log(f"    Saved: {csv_path}")

    if not all_data:
        log(f"  No data to plot for {dataset_name}")
        return

    # Create violin plot
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    data_arrays = []
    colors_list = []
    labels_list = []

    model_order = ["neural", "texttiling", "csm", "random"]
    pos = 1

    for model in model_order:
        if model not in all_data:
            continue
        positions.append(pos)
        data_arrays.append(all_data[model])
        colors_list.append(COLORS.get(model, "#333333"))
        labels_list.append(MODEL_LABELS.get(model, model))
        pos += 1

    # Violin plot
    parts = ax.violinplot(data_arrays, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)

    # Style the other parts
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1)

    # Add horizontal line at BOR=1
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.02, 1.05, "BOR=1", fontsize=8, color="gray", transform=ax.get_yaxis_transform())

    # Configure
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("Per-Dialogue BOR")
    ax.set_xlabel("Model")
    ax.set_title(f"{display_name}: Per-Dialogue BOR Distribution\n(at global BOR≈1.0 operating point)")

    # Add note about excluded dialogues
    if n_excluded > 0:
        ax.text(0.02, 0.02, f"Note: {n_excluded} dialogues with 0 gold boundaries excluded",
               fontsize=8, color="gray", transform=ax.transAxes)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis to show reasonable range
    ax.set_ylim(0, min(10, max(max(d) for d in data_arrays) + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"per_dialogue_bor_{dataset_name}.pdf"
    png_path = output_dir / f"per_dialogue_bor_{dataset_name}.png"

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    log(f"  Saved: {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    log("=" * 70)
    log("Density-Quality Diagnostics")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info["display"]
        log(f"\n{'='*70}")
        log(f"Dataset: {display_name}")
        log("=" * 70)

        # =====================================================================
        # Figure A: Local robustness around BOR≈1
        # =====================================================================
        log("\n--- Figure A: Local Robustness Near BOR=1 ---")
        try:
            plot_robustness_near_bor1(dataset_name, display_name, FIGURES_DIR)
        except Exception as e:
            log(f"  ERROR: {e}")

        # =====================================================================
        # Figure B: Per-dialogue BOR distribution
        # =====================================================================
        log("\n--- Figure B: Per-Dialogue BOR Distribution ---")
        try:
            # Load dialogues
            dialogues = load_dataset(dataset_name)
            log(f"  Loaded {len(dialogues)} dialogues")

            n_zero_gold = sum(1 for d in dialogues if len(d.gold_boundaries) == 0)
            log(f"  Dialogues with 0 gold boundaries (excluded): {n_zero_gold}")

            # Compute scores for each model
            model_scores = {}

            log("  Computing scores for neural model...")
            model_scores["neural"] = get_neural_scores(dialogues)

            log("  Computing scores for TextTiling...")
            model_scores["texttiling"] = get_texttiling_scores(dialogues)

            log("  Computing scores for CSM...")
            model_scores["csm"] = get_csm_scores(dialogues)

            # For random, use seed 0 as representative
            log("  Computing scores for Random (seed=0)...")
            model_scores["random"] = get_random_scores(dialogues, seed=0)

            # Generate plot
            log("  Generating violin plot...")
            plot_per_dialogue_bor(dataset_name, display_name, FIGURES_DIR,
                                 dialogues, model_scores)

        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    log("\nDone.")


if __name__ == "__main__":
    main()
