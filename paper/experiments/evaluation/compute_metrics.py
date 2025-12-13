"""
Compute metrics for paper tables: Purity/Coverage and Baselines.

Uses the calibrated model checkpoint and evaluates on DialSeg711, SuperSeg, TIAGE.
"""

import torch
import json
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

import sys
# Add project root to path for episodic imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
    compute_bor,
)


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
    """Load dialogues from a dataset, returning structured data.

    Boundaries are counted in USER TURN indices (matching training format).
    """
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

            # Extract boundaries using topic_id or topic_name
            # Boundaries are indexed by USER TURN position (matching training)
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

            # num_user_turns is the dimension we evaluate on
            num_user_turns = sum(1 for m in messages if m["role"] == "user")

            dialogues.append(DialogueData(
                messages=messages,
                gold_boundaries=boundaries,
                num_messages=num_user_turns  # Use user turn count
            ))

    return dialogues


def create_examples_from_dialogues(dialogues: List[DialogueData]) -> List[Dict]:
    """Create window examples from dialogues for model inference.

    Only creates examples for user messages (matching training format).
    Position is USER TURN index, not message index.
    """
    examples = []

    for dial_idx, dialogue in enumerate(dialogues):
        messages = dialogue.messages
        gold_boundaries = dialogue.gold_boundaries

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:  # Skip first user message
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    examples.append({
                        "window": window,
                        "current_message": msg,
                        "label": 1 if user_idx in gold_boundaries else 0,
                        "dialogue_idx": dial_idx,
                        "position": user_idx,  # User turn index
                    })
                user_idx += 1

    return examples


def get_model_predictions(
    model,
    dataloader,
    device,
    temperature: float,
    dialogues: List[DialogueData]
) -> Dict[int, Set[int]]:
    """Run model and get predicted boundaries per dialogue."""
    model.eval()
    predictions_by_dialogue = {i: set() for i in range(len(dialogues))}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dialogue_indices = batch["dialogue_idx"].numpy()
            positions = batch["position"].numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits / temperature, dim=-1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            for i in range(len(preds)):
                if preds[i] == 1:
                    dial_idx = dialogue_indices[i]
                    pos = positions[i]
                    predictions_by_dialogue[dial_idx].add(pos)

    return predictions_by_dialogue


def compute_all_metrics(
    dialogues: List[DialogueData],
    predictions_by_dialogue: Dict[int, Set[int]]
) -> Dict[str, float]:
    """Compute aggregated metrics across all dialogues."""
    all_w1_prec, all_w1_rec, all_w1_f1 = [], [], []
    all_purity, all_coverage = [], []
    total_gold, total_pred = 0, 0

    for dial_idx, dialogue in enumerate(dialogues):
        gold = dialogue.gold_boundaries
        pred = predictions_by_dialogue.get(dial_idx, set())
        n = dialogue.num_messages

        # Windowed F1
        w1_prec, w1_rec, w1_f1 = compute_windowed_metrics(gold, pred, n, window=1)
        all_w1_prec.append(w1_prec)
        all_w1_rec.append(w1_rec)
        all_w1_f1.append(w1_f1)

        # Purity/Coverage
        gold_segments = boundaries_to_segments(gold, n)
        pred_segments = boundaries_to_segments(pred, n)
        purity, coverage = compute_purity_coverage(gold_segments, pred_segments)
        all_purity.append(purity)
        all_coverage.append(coverage)

        total_gold += len(gold)
        total_pred += len(pred)

    bor = total_pred / total_gold if total_gold > 0 else 1.0

    return {
        "w_f1": np.mean(all_w1_f1),
        "w_prec": np.mean(all_w1_prec),
        "w_rec": np.mean(all_w1_rec),
        "bor": bor,
        "purity": np.mean(all_purity),
        "coverage": np.mean(all_coverage),
        "total_gold": total_gold,
        "total_pred": total_pred,
        "n_dialogues": len(dialogues),
    }


# =============================================================================
# BASELINE PREDICTORS
# =============================================================================

def baseline_no_boundary(dialogues: List[DialogueData]) -> Dict[int, Set[int]]:
    """Predict no boundaries at all."""
    return {i: set() for i in range(len(dialogues))}


def baseline_every_n(dialogues: List[DialogueData], n: int) -> Dict[int, Set[int]]:
    """Predict a boundary every N user turns."""
    predictions = {}
    for i, dialogue in enumerate(dialogues):
        pred = set()
        # Start from position n (after first n user turns)
        for pos in range(n, dialogue.num_messages, n):
            pred.add(pos)
        predictions[i] = pred
    return predictions


def baseline_random_at_rate(dialogues: List[DialogueData], seed: int = 42) -> Dict[int, Set[int]]:
    """Predict boundaries randomly at the same rate as gold (BOR=1 by construction)."""
    random.seed(seed)
    predictions = {}

    for i, dialogue in enumerate(dialogues):
        n_gold = len(dialogue.gold_boundaries)
        n_user_turns = dialogue.num_messages

        # Sample random positions (excluding position 0, user turn indices)
        possible_positions = list(range(1, n_user_turns))
        if n_gold <= len(possible_positions):
            pred_positions = random.sample(possible_positions, n_gold)
        else:
            pred_positions = possible_positions

        predictions[i] = set(pred_positions)

    return predictions


def find_good_every_n(dialogues: List[DialogueData]) -> int:
    """Find N for every-N baseline that gives reasonable BOR (not degenerate)."""
    # Target BOR around 1.0
    total_gold = sum(len(d.gold_boundaries) for d in dialogues)
    total_user_turns = sum(d.num_messages for d in dialogues)

    if total_gold == 0:
        return 5

    # Average segment length in user turns
    avg_segment_length = total_user_turns / (total_gold + len(dialogues))
    n = max(2, int(round(avg_segment_length)))
    return n


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics for paper tables")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to calibrated model checkpoint")
    parser.add_argument("--datasets-dir", type=str, default=None,
                       help="Path to datasets directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for LaTeX tables")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup paths
    experiments_dir = Path(__file__).parent.parent

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = experiments_dir / "models" / "final_calibrated.pt"

    if args.datasets_dir:
        datasets_path = Path(args.datasets_dir)
    else:
        # Default: project root / datasets
        datasets_path = experiments_dir.parent.parent / "datasets"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiments_dir.parent  # paper/

    # Load model
    print("Loading calibrated model...")
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

    # Datasets to evaluate
    dataset_names = ["dialseg711", "superseg", "tiage"]

    # Store results
    model_results = {}
    baseline_results = {"no_boundary": {}, "every_n": {}, "random_rate": {}}
    every_n_values = {}

    print("\n" + "="*70)
    print("EVALUATING MODEL AND BASELINES")
    print("="*70)

    for dataset_name in dataset_names:
        print(f"\n--- {dataset_name.upper()} ---")

        # Load dialogues
        dialogues = load_dataset_dialogues(datasets_path, dataset_name)
        if not dialogues:
            print(f"  No dialogues loaded for {dataset_name}")
            continue

        print(f"  Loaded {len(dialogues)} dialogues")

        # Create examples and dataloader for model
        examples = create_examples_from_dialogues(dialogues)
        dataset = BoundaryDataset(examples, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Get model predictions
        model_preds = get_model_predictions(model, dataloader, device, temperature, dialogues)
        model_metrics = compute_all_metrics(dialogues, model_preds)
        model_results[dataset_name] = model_metrics

        print(f"  Model: W-F1={model_metrics['w_f1']:.3f}, BOR={model_metrics['bor']:.2f}, "
              f"Purity={model_metrics['purity']:.3f}, Coverage={model_metrics['coverage']:.3f}")

        # Baseline: No boundary
        no_bound_preds = baseline_no_boundary(dialogues)
        no_bound_metrics = compute_all_metrics(dialogues, no_bound_preds)
        baseline_results["no_boundary"][dataset_name] = no_bound_metrics

        # Baseline: Every-N
        n = find_good_every_n(dialogues)
        every_n_values[dataset_name] = n
        every_n_preds = baseline_every_n(dialogues, n)
        every_n_metrics = compute_all_metrics(dialogues, every_n_preds)
        baseline_results["every_n"][dataset_name] = every_n_metrics

        # Baseline: Random at rate
        random_preds = baseline_random_at_rate(dialogues)
        random_metrics = compute_all_metrics(dialogues, random_preds)
        baseline_results["random_rate"][dataset_name] = random_metrics

        print(f"  No-boundary: W-F1={no_bound_metrics['w_f1']:.3f}, BOR={no_bound_metrics['bor']:.2f}, "
              f"Purity={no_bound_metrics['purity']:.3f}, Coverage={no_bound_metrics['coverage']:.3f}")
        print(f"  Every-{n}: W-F1={every_n_metrics['w_f1']:.3f}, BOR={every_n_metrics['bor']:.2f}, "
              f"Purity={every_n_metrics['purity']:.3f}, Coverage={every_n_metrics['coverage']:.3f}")
        print(f"  Random-rate: W-F1={random_metrics['w_f1']:.3f}, BOR={random_metrics['bor']:.2f}, "
              f"Purity={random_metrics['purity']:.3f}, Coverage={random_metrics['coverage']:.3f}")

    # ==========================================================================
    # OUTPUT RESULTS
    # ==========================================================================

    print("\n" + "="*70)
    print("RAW NUMBERS FOR AUDIT")
    print("="*70)

    print("\n--- Model Results (Segment Coherence) ---")
    print(f"{'Dataset':<12} {'W-F1':>8} {'BOR':>8} {'Purity':>8} {'Coverage':>8}")
    print("-" * 50)
    for ds in dataset_names:
        if ds in model_results:
            m = model_results[ds]
            print(f"{ds:<12} {m['w_f1']:>8.3f} {m['bor']:>8.2f} {m['purity']:>8.3f} {m['coverage']:>8.3f}")

    print("\n--- Baseline Results ---")
    for baseline_name, baseline_data in baseline_results.items():
        print(f"\n{baseline_name}:")
        print(f"{'Dataset':<12} {'W-F1':>8} {'BOR':>8} {'Purity':>8} {'Coverage':>8}")
        print("-" * 50)
        for ds in dataset_names:
            if ds in baseline_data:
                m = baseline_data[ds]
                print(f"{ds:<12} {m['w_f1']:>8.3f} {m['bor']:>8.2f} {m['purity']:>8.3f} {m['coverage']:>8.3f}")

    print(f"\nEvery-N values used: {every_n_values}")

    # ==========================================================================
    # GENERATE LATEX TABLES
    # ==========================================================================

    latex_output = []

    # Table 1: Segment Coherence Diagnostics
    latex_output.append("% Table: Segment Coherence Diagnostics")
    latex_output.append("% Columns: Dataset, Purity, Coverage")
    latex_output.append("")
    latex_output.append("\\begin{table}[t]")
    latex_output.append("\\centering")
    latex_output.append("\\begin{tabular}{lcc}")
    latex_output.append("\\toprule")
    latex_output.append("Dataset & Purity & Coverage \\\\")
    latex_output.append("\\midrule")

    ds_display = {"dialseg711": "DialSeg711", "superseg": "SuperSeg", "tiage": "TIAGE"}
    for ds in dataset_names:
        if ds in model_results:
            m = model_results[ds]
            latex_output.append(f"{ds_display[ds]} & {m['purity']:.3f} & {m['coverage']:.3f} \\\\")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\caption{Segment coherence diagnostics for the calibrated model.}")
    latex_output.append("\\label{tab:coherence}")
    latex_output.append("\\end{table}")
    latex_output.append("")

    # Table 2: Baseline Comparisons (averaged across datasets)
    latex_output.append("% Table: Baseline Comparisons")
    latex_output.append("% Columns: Baseline, W-F1, BOR, Purity, Coverage")
    latex_output.append("")
    latex_output.append("\\begin{table}[t]")
    latex_output.append("\\centering")
    latex_output.append("\\begin{tabular}{lcccc}")
    latex_output.append("\\toprule")
    latex_output.append("Baseline & W-F1 & BOR & Purity & Coverage \\\\")
    latex_output.append("\\midrule")

    # Compute averages for baselines
    def avg_metrics(baseline_data):
        w_f1s = [baseline_data[ds]["w_f1"] for ds in dataset_names if ds in baseline_data]
        bors = [baseline_data[ds]["bor"] for ds in dataset_names if ds in baseline_data]
        purities = [baseline_data[ds]["purity"] for ds in dataset_names if ds in baseline_data]
        coverages = [baseline_data[ds]["coverage"] for ds in dataset_names if ds in baseline_data]
        return {
            "w_f1": np.mean(w_f1s) if w_f1s else 0,
            "bor": np.mean(bors) if bors else 0,
            "purity": np.mean(purities) if purities else 0,
            "coverage": np.mean(coverages) if coverages else 0,
        }

    # Average N used for every-N
    avg_n = int(round(np.mean(list(every_n_values.values()))))

    no_bound_avg = avg_metrics(baseline_results["no_boundary"])
    every_n_avg = avg_metrics(baseline_results["every_n"])
    random_avg = avg_metrics(baseline_results["random_rate"])
    model_avg = avg_metrics(model_results)

    latex_output.append(f"No boundary & {no_bound_avg['w_f1']:.3f} & {no_bound_avg['bor']:.2f} & {no_bound_avg['purity']:.3f} & {no_bound_avg['coverage']:.3f} \\\\")
    latex_output.append(f"Every-{avg_n} & {every_n_avg['w_f1']:.3f} & {every_n_avg['bor']:.2f} & {every_n_avg['purity']:.3f} & {every_n_avg['coverage']:.3f} \\\\")
    latex_output.append(f"Random (BOR=1) & {random_avg['w_f1']:.3f} & {random_avg['bor']:.2f} & {random_avg['purity']:.3f} & {random_avg['coverage']:.3f} \\\\")
    latex_output.append("\\midrule")
    latex_output.append(f"Calibrated model & {model_avg['w_f1']:.3f} & {model_avg['bor']:.2f} & {model_avg['purity']:.3f} & {model_avg['coverage']:.3f} \\\\")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append(f"\\caption{{Baseline comparisons (averaged across DialSeg711, SuperSeg, TIAGE). Every-N uses N={avg_n}.}}")
    latex_output.append("\\label{tab:baselines}")
    latex_output.append("\\end{table}")
    latex_output.append("")

    # Also add per-dataset baseline table for completeness
    latex_output.append("% Table: Per-dataset baseline breakdown (optional)")
    latex_output.append("")
    latex_output.append("\\begin{table*}[t]")
    latex_output.append("\\centering")
    latex_output.append("\\small")
    latex_output.append("\\begin{tabular}{ll cccc}")
    latex_output.append("\\toprule")
    latex_output.append("Dataset & Baseline & W-F1 & BOR & Purity & Coverage \\\\")
    latex_output.append("\\midrule")

    for ds in dataset_names:
        ds_name = ds_display[ds]
        n = every_n_values.get(ds, avg_n)

        if ds in baseline_results["no_boundary"]:
            m = baseline_results["no_boundary"][ds]
            latex_output.append(f"{ds_name} & No boundary & {m['w_f1']:.3f} & {m['bor']:.2f} & {m['purity']:.3f} & {m['coverage']:.3f} \\\\")

        if ds in baseline_results["every_n"]:
            m = baseline_results["every_n"][ds]
            latex_output.append(f" & Every-{n} & {m['w_f1']:.3f} & {m['bor']:.2f} & {m['purity']:.3f} & {m['coverage']:.3f} \\\\")

        if ds in baseline_results["random_rate"]:
            m = baseline_results["random_rate"][ds]
            latex_output.append(f" & Random (BOR=1) & {m['w_f1']:.3f} & {m['bor']:.2f} & {m['purity']:.3f} & {m['coverage']:.3f} \\\\")

        if ds in model_results:
            m = model_results[ds]
            latex_output.append(f" & \\textbf{{Model}} & \\textbf{{{m['w_f1']:.3f}}} & {m['bor']:.2f} & \\textbf{{{m['purity']:.3f}}} & \\textbf{{{m['coverage']:.3f}}} \\\\")

        if ds != dataset_names[-1]:
            latex_output.append("\\midrule")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\caption{Per-dataset baseline comparison.}")
    latex_output.append("\\label{tab:baselines-detail}")
    latex_output.append("\\end{table*}")

    # Write to file
    output_path = output_dir / "results_tables.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(latex_output))

    print(f"\n\nLaTeX tables saved to: {output_path}")
    print("\n" + "="*70)
    print("LATEX OUTPUT")
    print("="*70)
    print("\n".join(latex_output))


if __name__ == "__main__":
    main()
