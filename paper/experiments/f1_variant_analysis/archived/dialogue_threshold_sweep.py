#!/usr/bin/env python3
"""
Dialogue Segmentation Threshold Sweep: F1 vs BOR Analysis

Tests whether boundary density (BOR = |P|/|G|) correlates with F1
in dialogue topic segmentation.

Computes BOTH:
1. W-F1 (many-to-one with window=1) - Michael's paper formulation
2. Exact F1 (sklearn position-wise binary) - CSM/SuperDialseg formulation

Critical question: Does the density confound exist under exact F1,
or is it specific to many-to-one W-F1?
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import f1_score as sklearn_f1_score

# Copy W-F1 computation directly to avoid dependency issues
def compute_windowed_metrics(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int,
    window: int = 1
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 with tolerance window (many-to-one matching).

    A predicted boundary at t is considered correct if there's a gold
    boundary in [t-window, t+window]. Multiple predictions can match the
    same gold boundary (many-to-one), which makes this variant recall-favoring.
    """
    if not gold_boundaries and not predicted_boundaries:
        return 1.0, 1.0, 1.0

    # For each predicted boundary, check if it matches any gold within window
    matched_predictions = set()
    matched_golds = set()

    for pred in predicted_boundaries:
        for gold in gold_boundaries:
            if abs(pred - gold) <= window:
                matched_predictions.add(pred)
                matched_golds.add(gold)
                break

    # Precision: what fraction of predictions are near a gold boundary
    precision = len(matched_predictions) / len(predicted_boundaries) if predicted_boundaries else 0.0

    # Recall: what fraction of gold boundaries have a nearby prediction
    recall = len(matched_golds) / len(gold_boundaries) if gold_boundaries else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_exact_f1(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int
) -> Tuple[float, float, float]:
    """
    Compute exact-match F1 using sklearn (position-wise binary comparison).

    This is what CSM and SuperDialseg use - no tolerance window,
    strict position matching.
    """
    # Convert to binary labels (0/1 for each position)
    y_true = [1 if i in gold_boundaries else 0 for i in range(num_messages)]
    y_pred = [1 if i in predicted_boundaries else 0 for i in range(num_messages)]

    # Use sklearn's f1_score with binary average (like SuperDialseg)
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 1.0, 1.0, 1.0

    precision = sklearn_f1_score(y_true, y_pred, average='binary', zero_division=0)
    # Actually sklearn f1_score returns F1, not precision. Let me compute properly:
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = sklearn_f1_score(y_true, y_pred, average='binary', zero_division=0)

    return precision, recall, f1


@dataclass
class DialogueData:
    """Container for a single dialogue with boundaries."""
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_messages: int


class BoundaryDataset(Dataset):
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
            "label": torch.tensor(ex.get("label", 0), dtype=torch.float),
            "dialogue_idx": ex.get("dialogue_idx", 0),
            "position": ex.get("position", 0),
        }


def load_dataset_dialogues(datasets_path: Path, dataset_name: str) -> List[DialogueData]:
    """Load dialogues from a dataset."""
    test_file = datasets_path / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        print(f"Warning: {test_file} not found")
        return []

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


def create_examples_from_dialogues(dialogues: List[DialogueData]) -> List[Dict]:
    """Create window examples from dialogues."""
    examples = []

    for dial_idx, dialogue in enumerate(dialogues):
        messages = dialogue.messages
        gold_boundaries = dialogue.gold_boundaries

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    examples.append({
                        "window": window,
                        "current_message": msg,
                        "label": 1 if user_idx in gold_boundaries else 0,
                        "dialogue_idx": dial_idx,
                        "position": user_idx,
                    })
                user_idx += 1

    return examples


def get_model_probabilities(model, dataloader, device, temperature: float) -> List[Dict]:
    """Get probability scores for all examples."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dialogue_indices = batch["dialogue_idx"].numpy()
            positions = batch["position"].numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits / temperature, dim=-1)[:, 1].cpu().numpy()

            for i in range(len(probs)):
                results.append({
                    "dialogue_idx": int(dialogue_indices[i]),
                    "position": int(positions[i]),
                    "prob": float(probs[i])
                })

    return results


def compute_metrics_at_threshold(
    dialogues: List[DialogueData],
    probabilities: List[Dict],
    threshold: float
) -> Dict[str, float]:
    """Compute W-F1, exact F1, and BOR at a given probability threshold."""
    # Group probabilities by dialogue
    prob_by_dialogue = {}
    for p in probabilities:
        dial_idx = p["dialogue_idx"]
        if dial_idx not in prob_by_dialogue:
            prob_by_dialogue[dial_idx] = []
        prob_by_dialogue[dial_idx].append(p)

    all_w1_f1 = []
    all_exact_f1 = []
    total_gold = 0
    total_pred = 0

    for dial_idx, dialogue in enumerate(dialogues):
        gold = dialogue.gold_boundaries
        n = dialogue.num_messages

        # Get predictions at this threshold
        pred = set()
        if dial_idx in prob_by_dialogue:
            for p in prob_by_dialogue[dial_idx]:
                if p["prob"] >= threshold:
                    pred.add(p["position"])

        # Compute W-F1 (many-to-one with window)
        _, _, w1_f1 = compute_windowed_metrics(gold, pred, n, window=1)
        all_w1_f1.append(w1_f1)

        # Compute exact F1 (sklearn position-wise)
        _, _, exact_f1 = compute_exact_f1(gold, pred, n)
        all_exact_f1.append(exact_f1)

        total_gold += len(gold)
        total_pred += len(pred)

    bor = total_pred / total_gold if total_gold > 0 else 1.0
    mean_w1_f1 = np.mean(all_w1_f1) if all_w1_f1 else 0.0
    mean_exact_f1 = np.mean(all_exact_f1) if all_exact_f1 else 0.0

    return {
        "threshold": threshold,
        "w_f1": mean_w1_f1,
        "exact_f1": mean_exact_f1,
        "bor": bor,
        "total_gold": total_gold,
        "total_pred": total_pred,
    }


def main():
    print("=" * 70)
    print("Dialogue Segmentation Threshold Sweep: W-F1 vs BOR")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths - relative to project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent  # f1_variant_analysis -> experiments -> paper -> episodic
    experiments_dir = project_root / "paper" / "experiments"
    model_path = experiments_dir / "models" / "final_calibrated.pt"
    datasets_path = project_root / "datasets"
    output_dir = script_dir  # Output to same directory as script

    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Falling back to random probability simulation...")
        use_model = False
    else:
        use_model = True

    # Datasets to test
    dataset_names = ["dialseg711", "superseg"]

    # Threshold sweep values
    THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    results_by_dataset = {}

    if use_model:
        # Load model
        print("\nLoading calibrated model...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        temperature = checkpoint.get("temperature", 1.0)
        print(f"Temperature: {temperature:.4f}")

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name.upper()}")
        print("="*50)

        # Load dialogues
        dialogues = load_dataset_dialogues(datasets_path, dataset_name)
        if not dialogues:
            print(f"  No dialogues loaded for {dataset_name}")
            continue

        print(f"  Loaded {len(dialogues)} dialogues")

        if use_model:
            # Create examples and get probabilities
            examples = create_examples_from_dialogues(dialogues)
            dataset = BoundaryDataset(examples, tokenizer)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
            probabilities = get_model_probabilities(model, dataloader, device, temperature)
        else:
            # Simulate probabilities
            examples = create_examples_from_dialogues(dialogues)
            np.random.seed(42)
            probabilities = [
                {"dialogue_idx": ex["dialogue_idx"],
                 "position": ex["position"],
                 "prob": np.random.random()}
                for ex in examples
            ]

        # Sweep thresholds
        results = []
        print(f"\n  {'Threshold':>10} {'BOR':>10} {'W-F1':>10} {'Exact-F1':>10}")
        print("  " + "-" * 50)

        for threshold in THRESHOLDS:
            metrics = compute_metrics_at_threshold(dialogues, probabilities, threshold)
            results.append(metrics)
            print(f"  {threshold:>10.2f} {metrics['bor']:>10.2f} {metrics['w_f1']:>10.3f} {metrics['exact_f1']:>10.3f}")

        results_by_dataset[dataset_name] = results

    # Correlation analysis and plotting
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Create 2x2 plot: W-F1 and Exact-F1 for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"dialseg711": "steelblue", "superseg": "coral"}

    for i, (dataset_name, results) in enumerate(results_by_dataset.items()):
        bors = np.array([r["bor"] for r in results])
        w_f1s = np.array([r["w_f1"] for r in results])
        exact_f1s = np.array([r["exact_f1"] for r in results])

        # Filter out zero BOR (no predictions)
        mask = bors > 0
        bors = bors[mask]
        w_f1s = w_f1s[mask]
        exact_f1s = exact_f1s[mask]

        if len(bors) < 3:
            print(f"\n{dataset_name}: Not enough data points")
            continue

        # Compute correlations for both metrics
        r_w, p_w = stats.pearsonr(bors, w_f1s)
        r_e, p_e = stats.pearsonr(bors, exact_f1s)

        print(f"\n{dataset_name.upper()}:")
        print(f"  W-F1 (many-to-one, window=1):")
        print(f"    Pearson r = {r_w:.3f} (p = {p_w:.4f})")
        print(f"    Range: {w_f1s.min():.3f} - {w_f1s.max():.3f}")
        best_w_idx = np.argmax(w_f1s)
        print(f"    Best: {w_f1s[best_w_idx]:.3f} at BOR = {bors[best_w_idx]:.2f}")

        print(f"  Exact-F1 (sklearn position-wise):")
        print(f"    Pearson r = {r_e:.3f} (p = {p_e:.4f})")
        print(f"    Range: {exact_f1s.min():.3f} - {exact_f1s.max():.3f}")
        best_e_idx = np.argmax(exact_f1s)
        print(f"    Best: {exact_f1s[best_e_idx]:.3f} at BOR = {bors[best_e_idx]:.2f}")

        # Plot W-F1 (top row)
        ax = axes[0, i]
        ax.plot(bors, w_f1s, 'o-', markersize=8, linewidth=2,
                color=colors[dataset_name], label=f'W-F1 (r={r_w:.2f})')
        ax.axvline(x=1.0, color='green', linestyle=':', alpha=0.7, label='BOR = 1')
        ax.scatter([bors[best_w_idx]], [w_f1s[best_w_idx]], color='red', s=150,
                   zorder=10, edgecolors='black', label=f'Best: BOR={bors[best_w_idx]:.2f}')
        ax.set_xlabel('BOR (|P| / |G|)', fontsize=12)
        ax.set_ylabel('W-F1 (many-to-one, window=1)', fontsize=12)
        ax.set_title(f'{dataset_name}: W-F1 vs BOR\nr = {r_w:.3f}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)

        # Plot Exact-F1 (bottom row)
        ax = axes[1, i]
        ax.plot(bors, exact_f1s, 's-', markersize=8, linewidth=2,
                color=colors[dataset_name], label=f'Exact-F1 (r={r_e:.2f})')
        ax.axvline(x=1.0, color='green', linestyle=':', alpha=0.7, label='BOR = 1')
        ax.scatter([bors[best_e_idx]], [exact_f1s[best_e_idx]], color='red', s=150,
                   zorder=10, edgecolors='black', label=f'Best: BOR={bors[best_e_idx]:.2f}')
        ax.set_xlabel('BOR (|P| / |G|)', fontsize=12)
        ax.set_ylabel('Exact-F1 (sklearn position-wise)', fontsize=12)
        ax.set_title(f'{dataset_name}: Exact-F1 vs BOR\nr = {r_e:.3f}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'dialogue_threshold_sweep_both_f1.png',
                dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'dialogue_threshold_sweep_both_f1.pdf',
                bbox_inches='tight')

    print(f"\n\nPlots saved to {output_dir}:")
    print("  - dialogue_threshold_sweep_both_f1.png")
    print("  - dialogue_threshold_sweep_both_f1.pdf")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: W-F1 vs Exact-F1 Comparison")
    print("=" * 70)
    print("""
Key question: Does the density confound exist under exact F1,
or is it specific to many-to-one W-F1?

W-F1 (many-to-one, window=1):
- Multiple predictions can match same gold boundary
- More lenient, rewards over-prediction

Exact-F1 (sklearn position-wise):
- Position must match exactly
- Used by CSM and SuperDialseg
- No tolerance window

If exact-F1 peaks sharply at BOR≈1 while W-F1 shows positive correlation,
then the "gameable zone" is specific to many-to-one matching.
""")


if __name__ == "__main__":
    main()
