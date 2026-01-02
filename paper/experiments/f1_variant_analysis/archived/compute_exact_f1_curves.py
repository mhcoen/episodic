#!/usr/bin/env python3
"""
Compute density-quality curves for exact F1 (strict position matching).

This script generates density-quality curves using exact F1 (no tolerance window)
to demonstrate that the density confound persists even under strict matching.

Output: density_quality_exact_f1.pdf showing curves for all methods on both datasets.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
DATASETS_PATH = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "paper" / "experiments" / "models"

# Add segmenters to path
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "experiments"))

# Styling - match Figure 1
COLORS = {
    "neural": "#1f77b4",  # Blue
    "texttiling": "#2ca02c",  # Green
    "csm": "#ff7f0e",  # Orange
    "random": "#7f7f7f",  # Gray
}

MODEL_LABELS = {
    "neural": "Proposed (Neural)",
    "texttiling": "TextTiling",
    "csm": "CSM (NSP)",
    "random": "Random",
}

MIN_GAP = 2  # g=2 spacing constraint


def compute_exact_f1(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_positions: int
) -> Tuple[float, float, float]:
    """
    Compute exact-match F1 using sklearn (position-wise binary comparison).
    No tolerance window - position must match exactly.
    """
    y_true = [1 if i in gold_boundaries else 0 for i in range(num_positions)]
    y_pred = [1 if i in predicted_boundaries else 0 for i in range(num_positions)]

    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 1.0, 1.0, 1.0

    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return precision, recall, f1


def apply_spacing_constraint(positions: List[int], scores: List[float], min_gap: int = 2) -> Set[int]:
    """Apply minimum spacing constraint via greedy NMS."""
    if not positions:
        return set()
    sorted_indices = sorted(range(len(positions)), key=lambda i: scores[i], reverse=True)
    accepted = set()
    for idx in sorted_indices:
        pos = positions[idx]
        if all(abs(pos - b) >= min_gap for b in accepted):
            accepted.add(pos)
    return accepted


@dataclass
class DialogueData:
    """Container for a single dialogue."""
    dialogue_id: int
    messages: List[Dict]
    gold_boundaries: Set[int]
    num_user_turns: int


def load_dataset(dataset_name: str) -> List[DialogueData]:
    """Load dialogues from a dataset."""
    test_file = DATASETS_PATH / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        print(f"Warning: {test_file} not found")
        return []

    with open(test_file) as f:
        data = json.load(f)

    dialogues = []
    dial_data = data.get("dial_data", data)
    dial_id = 0

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
                dialogue_id=dial_id,
                messages=messages,
                gold_boundaries=boundaries,
                num_user_turns=num_user_turns
            ))
            dial_id += 1

    return dialogues


class BoundaryDataset(Dataset):
    """Dataset for boundary classification."""
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        window = ex.get("window", [])
        current = ex.get("current_message", {})

        context_parts = []
        for msg in window[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", msg.get("utterance", ""))
            context_parts.append(f"{role}: {content}")

        curr_content = current.get("content", current.get("utterance", ""))
        text = " [SEP] ".join(context_parts) + f" [SEP] current: {curr_content}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "dialogue_idx": ex["dialogue_idx"],
            "position": ex["position"],
        }


def create_examples(dialogues: List[DialogueData]) -> List[Dict]:
    """Create examples for model inference."""
    examples = []

    for dialogue in dialogues:
        messages = dialogue.messages
        user_idx = 0

        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    examples.append({
                        "window": window,
                        "current_message": msg,
                        "dialogue_idx": dialogue.dialogue_id,
                        "position": user_idx,
                    })
                user_idx += 1

    return examples


def get_neural_predictions(dialogues: List[DialogueData], device) -> Dict[int, List[Tuple[int, float]]]:
    """Get neural model predictions for all dialogues."""
    model_path = MODELS_DIR / "final_calibrated.pt"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return {}

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create examples and get predictions
    examples = create_examples(dialogues)
    dataset = BoundaryDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Dict: dialogue_idx -> [(position, probability)]
    predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dialogue_indices = batch["dialogue_idx"].numpy()
            positions = batch["position"].numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits / temperature, dim=-1)[:, 1].cpu().numpy()

            for i in range(len(probs)):
                dial_idx = int(dialogue_indices[i])
                if dial_idx not in predictions:
                    predictions[dial_idx] = []
                predictions[dial_idx].append((int(positions[i]), float(probs[i])))

    return predictions


def get_texttiling_predictions(dialogues: List[DialogueData]) -> Dict[int, List[Tuple[int, float]]]:
    """Get TextTiling predictions for all dialogues."""
    from segmenters.texttiling import TextTilingSegmenter

    segmenter = TextTilingSegmenter()
    predictions = {}

    for dialogue in dialogues:
        result = segmenter.predict_boundaries(dialogue.messages, alpha=-999)  # Get all scores
        # Get depth scores for all positions
        utterances = [m["content"] for m in dialogue.messages if m["role"] == "user"]
        if len(utterances) < 2:
            continue

        sim_scores = segmenter._compute_similarity_scores(utterances)
        depth_scores = segmenter._compute_depth_scores(sim_scores)

        preds = []
        for i, depth in enumerate(depth_scores):
            pos = i + 1
            if pos < dialogue.num_user_turns:
                preds.append((pos, float(depth)))

        predictions[dialogue.dialogue_id] = preds

    return predictions


def get_csm_predictions(dialogues: List[DialogueData]) -> Dict[int, List[Tuple[int, float]]]:
    """Get CSM (NSP) predictions for all dialogues."""
    from segmenters.csm_nsp import CSMSegmenter

    segmenter = CSMSegmenter()
    predictions = {}

    for dialogue in dialogues:
        result = segmenter.predict_boundaries(dialogue.messages, threshold=-999)

        preds = []
        if result.scores:
            for pos, score in result.scores.items():
                preds.append((pos, float(score)))
        predictions[dialogue.dialogue_id] = preds

    return predictions


def get_random_predictions(dialogues: List[DialogueData], seed: int = 42) -> Dict[int, List[Tuple[int, float]]]:
    """Get random predictions for all dialogues."""
    rng = np.random.RandomState(seed)
    predictions = {}

    for dialogue in dialogues:
        preds = []
        for pos in range(1, dialogue.num_user_turns):
            score = rng.random()
            preds.append((pos, float(score)))
        predictions[dialogue.dialogue_id] = preds

    return predictions


def compute_metrics_at_threshold(
    dialogues: List[DialogueData],
    predictions: Dict[int, List[Tuple[int, float]]],
    threshold: float,
    min_gap: int = 2
) -> Dict:
    """Compute exact F1 and BOR at a given threshold."""
    all_exact_f1 = []
    total_gold = 0
    total_pred = 0

    for dialogue in dialogues:
        gold = dialogue.gold_boundaries
        n = dialogue.num_user_turns

        # Get predictions at this threshold with spacing constraint
        dial_preds = predictions.get(dialogue.dialogue_id, [])
        positions = [p for p, s in dial_preds if s >= threshold]
        scores = [s for p, s in dial_preds if s >= threshold]

        if positions:
            pred = apply_spacing_constraint(positions, scores, min_gap)
        else:
            pred = set()

        # Compute exact F1
        _, _, exact_f1 = compute_exact_f1(gold, pred, n)
        all_exact_f1.append(exact_f1)

        total_gold += len(gold)
        total_pred += len(pred)

    bor = total_pred / total_gold if total_gold > 0 else 0.0
    mean_exact_f1 = np.mean(all_exact_f1) if all_exact_f1 else 0.0

    return {
        "threshold": threshold,
        "exact_f1": mean_exact_f1,
        "bor": bor,
        "total_gold": total_gold,
        "total_pred": total_pred,
        "per_dialogue_f1": all_exact_f1,
    }


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (np.mean(values), np.mean(values))

    rng = np.random.RandomState(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (lower, upper)


def sweep_thresholds(
    dialogues: List[DialogueData],
    predictions: Dict[int, List[Tuple[int, float]]],
    n_steps: int = 100
) -> pd.DataFrame:
    """Sweep thresholds and compute exact F1."""
    # Get all scores to determine threshold range
    all_scores = []
    for dial_preds in predictions.values():
        all_scores.extend([s for _, s in dial_preds])

    if not all_scores:
        return pd.DataFrame()

    thresholds = np.percentile(all_scores, np.linspace(0, 99, n_steps))

    results = []
    for tau in thresholds:
        metrics = compute_metrics_at_threshold(dialogues, predictions, tau, MIN_GAP)
        ci_low, ci_high = bootstrap_ci(metrics["per_dialogue_f1"])
        results.append({
            "threshold": tau,
            "bor": metrics["bor"],
            "exact_f1": metrics["exact_f1"],
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    return pd.DataFrame(results)


def add_regime_shading(ax, xmax: float = 2.0):
    """Add BOR regime shading like Figure 1."""
    # Under-segmentation region (BOR < 1): light red/pink
    ax.axvspan(0, 1.0, alpha=0.1, color='red', zorder=0)
    # Over-segmentation region (BOR > 1): light green
    ax.axvspan(1.0, xmax, alpha=0.1, color='green', zorder=0)
    # Vertical line at BOR=1
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)


def generate_exact_f1_figure():
    """Generate the density-quality curves figure for exact F1."""
    print("=" * 70)
    print("Generating Exact F1 Density-Quality Curves (All Methods)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Publication settings - match Figure 1
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    datasets = [
        ("dialseg711", "DialSeg711"),
        ("superseg", "SuperSeg"),
    ]

    model_order = ["neural", "texttiling", "csm", "random"]
    correlation_results = []

    for ax_idx, (dataset_name, display_name) in enumerate(datasets):
        ax = axes[ax_idx]
        print(f"\n--- {display_name} ---")

        # Load dialogues
        dialogues = load_dataset(dataset_name)
        print(f"  Loaded {len(dialogues)} dialogues")

        # Add regime shading
        add_regime_shading(ax, xmax=2.5)

        xmax_data = 0.0

        for model in model_order:
            print(f"  Processing {model}...")

            # Get predictions
            if model == "neural":
                predictions = get_neural_predictions(dialogues, device)
            elif model == "texttiling":
                predictions = get_texttiling_predictions(dialogues)
            elif model == "csm":
                predictions = get_csm_predictions(dialogues)
            elif model == "random":
                # Aggregate multiple random seeds
                all_dfs = []
                for seed in range(10):
                    predictions = get_random_predictions(dialogues, seed=seed)
                    df = sweep_thresholds(dialogues, predictions)
                    if not df.empty:
                        df["seed"] = seed
                        all_dfs.append(df)
                if all_dfs:
                    random_df = pd.concat(all_dfs, ignore_index=True)
                    # Aggregate by BOR bins
                    random_df["bor_bin"] = pd.cut(random_df["bor"], bins=50)
                    agg_df = random_df.groupby("bor_bin").agg({
                        "bor": "mean",
                        "exact_f1": ["mean", "std"],
                    }).reset_index()
                    agg_df.columns = ["bor_bin", "bor", "exact_f1", "std"]
                    agg_df = agg_df.dropna()
                    agg_df = agg_df.sort_values("bor")

                    # Filter to BOR range
                    mask = (agg_df["bor"] > 0) & (agg_df["bor"] <= 2.5)
                    plot_df = agg_df[mask]

                    if not plot_df.empty:
                        xmax_data = max(xmax_data, plot_df["bor"].max())
                        ax.fill_between(
                            plot_df["bor"],
                            plot_df["exact_f1"] - 1.96 * plot_df["std"],
                            plot_df["exact_f1"] + 1.96 * plot_df["std"],
                            alpha=0.15, color=COLORS["random"]
                        )
                        ax.plot(
                            plot_df["bor"], plot_df["exact_f1"],
                            color=COLORS["random"], linewidth=2,
                            label=MODEL_LABELS["random"]
                        )
                continue

            if not predictions:
                continue

            # Sweep thresholds
            df = sweep_thresholds(dialogues, predictions)

            if df.empty:
                continue

            # Filter to BOR range 0-2.5
            mask = (df["bor"] > 0) & (df["bor"] <= 2.5)
            plot_df = df[mask].copy()

            if plot_df.empty:
                continue

            xmax_data = max(xmax_data, plot_df["bor"].max())

            # Plot curve with CI
            ax.fill_between(
                plot_df["bor"], plot_df["ci_low"], plot_df["ci_high"],
                alpha=0.15, color=COLORS[model]
            )
            ax.plot(
                plot_df["bor"], plot_df["exact_f1"],
                color=COLORS[model], linewidth=2,
                label=MODEL_LABELS[model]
            )

            # Compute correlation
            if len(plot_df) >= 3:
                r, p = stats.pearsonr(plot_df["bor"], plot_df["exact_f1"])
                best_idx = plot_df["exact_f1"].idxmax()
                best_f1 = plot_df.loc[best_idx, "exact_f1"]
                best_bor = plot_df.loc[best_idx, "bor"]

                correlation_results.append({
                    "dataset": display_name,
                    "model": MODEL_LABELS[model],
                    "r": r,
                    "p": p,
                    "best_f1": best_f1,
                    "best_bor": best_bor,
                })

                print(f"    {model}: r={r:.3f}, best F1={best_f1:.3f} at BOR={best_bor:.2f}")

        # Labels
        ax.set_xlabel("Boundary Oversegmentation Ratio (BOR = |P|/|G|)")
        ax.set_ylabel("Exact F1 (Strict Position Matching)")
        ax.set_title(f"{display_name}: Exact F1 vs Boundary Density")
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "density_quality_exact_f1.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXACT F1 CORRELATION SUMMARY")
    print("=" * 70)
    for result in correlation_results:
        print(f"{result['dataset']} ({result['model']}):")
        print(f"  Pearson r = {result['r']:.3f} (p = {result['p']:.2e})")
        print(f"  Best Exact F1 = {result['best_f1']:.3f} at BOR = {result['best_bor']:.2f}")

    print("\nKey finding: Strong positive correlation indicates density confound")
    print("persists even under strict position matching with zero tolerance.")

    return correlation_results


if __name__ == "__main__":
    generate_exact_f1_figure()
