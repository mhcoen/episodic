#!/usr/bin/env python3
"""
Compute Stage 3 calibrated results for all 8 datasets.

Uses the exact same evaluation code as compute_metrics.py with the
calibrated model checkpoint (T=0.976).

Outputs:
- results/stage3_all8.json: Machine-readable metrics
- paper/tables/appendix_stage3_all8.tex: LaTeX table for appendix

Sanity checks:
- Verifies computed values match Tables 8-9 for DialSeg711, SuperSeg, TIAGE
"""

import torch
import json
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
    compute_bor,
)


# Paper values for sanity checking (Tables 8-9) - with g=2 spacing
PAPER_VALUES = {
    "dialseg711": {"w_f1": 0.708, "bor": 1.38, "f1": 0.391, "purity": 0.875, "coverage": 0.772, "pred": 2820, "gold": 2042},
    "superseg": {"w_f1": 0.520, "bor": 0.50, "f1": 0.410, "purity": 0.765, "coverage": 0.938, "pred": 1461, "gold": 2923},
    "tiage": {"w_f1": 0.474, "bor": 0.52, "f1": 0.282, "purity": 0.741, "coverage": 0.913, "pred": 107, "gold": 207},
}

# All 8 datasets
ALL_DATASETS = [
    "dialseg711",
    "superseg",
    "tiage",
    "dailydialog",
    "multiwoz",
    "taskmaster",
    "topical_chat",
    "qmsum",
]

# Display names for LaTeX
DISPLAY_NAMES = {
    "dialseg711": "DialSeg711",
    "superseg": "SuperSeg",
    "tiage": "TIAGE",
    "dailydialog": "DailyDialog",
    "multiwoz": "MultiWOZ",
    "taskmaster": "Taskmaster",
    "topical_chat": "Topical-Chat",
    "qmsum": "QMSum",
}


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
    """Load dialogues from a dataset, returning structured data."""
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
    """Create window examples from dialogues for model inference."""
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


def apply_spacing_constraint(
    positions: List[int],
    scores: List[float],
    min_gap: int = 3
) -> Set[int]:
    """
    Apply minimum spacing constraint via greedy non-maximum suppression.

    Candidates are processed in descending score order. A candidate is
    accepted only if its distance to all previously accepted boundaries
    is >= min_gap.

    Args:
        positions: Candidate boundary positions
        scores: Corresponding scores for each position
        min_gap: Minimum spacing between accepted boundaries

    Returns:
        Set of accepted boundary positions
    """
    if not positions:
        return set()

    # Sort by score descending
    sorted_indices = sorted(range(len(positions)), key=lambda i: scores[i], reverse=True)

    accepted = set()
    for idx in sorted_indices:
        pos = positions[idx]
        # Check if this position is at least min_gap away from all accepted boundaries
        if all(abs(pos - b) >= min_gap for b in accepted):
            accepted.add(pos)

    return accepted


def get_model_predictions(
    model,
    dataloader,
    device,
    temperature: float,
    dialogues: List[DialogueData],
    min_gap: int = 2
) -> Dict[int, Set[int]]:
    """Run model and get predicted boundaries per dialogue with spacing constraint."""
    model.eval()

    # Collect all scores and positions per dialogue
    scores_by_dialogue = {i: [] for i in range(len(dialogues))}
    positions_by_dialogue = {i: [] for i in range(len(dialogues))}

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
                dial_idx = dialogue_indices[i]
                pos = positions[i]
                score = probs[i]

                # Only consider candidates above threshold
                if score > 0.5:
                    scores_by_dialogue[dial_idx].append(score)
                    positions_by_dialogue[dial_idx].append(pos)

    # Apply spacing constraint per dialogue
    predictions_by_dialogue = {}
    for dial_idx in range(len(dialogues)):
        predictions_by_dialogue[dial_idx] = apply_spacing_constraint(
            positions_by_dialogue[dial_idx],
            scores_by_dialogue[dial_idx],
            min_gap=min_gap
        )

    return predictions_by_dialogue


def compute_strict_f1(gold: Set[int], pred: Set[int]) -> Tuple[float, float, float]:
    """Compute strict precision, recall, F1."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0

    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_all_metrics(
    dialogues: List[DialogueData],
    predictions_by_dialogue: Dict[int, Set[int]]
) -> Dict[str, Any]:
    """Compute aggregated metrics across all dialogues."""
    all_w1_prec, all_w1_rec, all_w1_f1 = [], [], []
    all_f1 = []
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

        # Strict F1
        _, _, f1 = compute_strict_f1(gold, pred)
        all_f1.append(f1)

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
        "w_f1": float(np.mean(all_w1_f1)),
        "w_prec": float(np.mean(all_w1_prec)),
        "w_rec": float(np.mean(all_w1_rec)),
        "f1": float(np.mean(all_f1)),
        "bor": float(bor),
        "purity": float(np.mean(all_purity)),
        "coverage": float(np.mean(all_coverage)),
        "total_gold": int(total_gold),
        "total_pred": int(total_pred),
        "n_dialogues": len(dialogues),
    }


def sanity_check(results: Dict[str, Dict], tolerance: float = 0.02) -> List[str]:
    """Check computed values against paper values."""
    diffs = []

    for ds_name, paper in PAPER_VALUES.items():
        if ds_name not in results:
            diffs.append(f"{ds_name}: NOT COMPUTED")
            continue

        computed = results[ds_name]

        for metric in ["w_f1", "bor", "f1", "purity", "coverage"]:
            paper_val = paper.get(metric, 0)
            comp_val = computed.get(metric, 0)
            diff = abs(paper_val - comp_val)

            if diff > tolerance:
                status = "MISMATCH"
            else:
                status = "OK"

            diffs.append(f"{ds_name} {metric}: paper={paper_val:.3f}, computed={comp_val:.3f}, diff={diff:.3f} [{status}]")

        # Check boundary counts
        paper_pred = paper.get("pred", 0)
        paper_gold = paper.get("gold", 0)
        comp_pred = computed.get("total_pred", 0)
        comp_gold = computed.get("total_gold", 0)

        pred_match = "OK" if paper_pred == comp_pred else "MISMATCH"
        gold_match = "OK" if paper_gold == comp_gold else "MISMATCH"

        diffs.append(f"{ds_name} pred: paper={paper_pred}, computed={comp_pred} [{pred_match}]")
        diffs.append(f"{ds_name} gold: paper={paper_gold}, computed={comp_gold} [{gold_match}]")
        diffs.append("")

    return diffs


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for appendix."""
    lines = [
        "% Stage 3 calibrated results for all 8 datasets",
        "% Generated by compute_stage3_all8.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Dataset & W-F1 & BOR & Purity & Coverage & F1 & Pred & Gold \\\\",
        "\\midrule",
    ]

    for ds_name in ALL_DATASETS:
        if ds_name not in results:
            continue

        m = results[ds_name]
        display = DISPLAY_NAMES.get(ds_name, ds_name)

        # Bold BOR to match paper style
        lines.append(
            f"{display} & {m['w_f1']:.3f} & \\textbf{{{m['bor']:.2f}}} & "
            f"{m['purity']:.3f} & {m['coverage']:.3f} & {m['f1']:.3f} & "
            f"{m['total_pred']} & {m['total_gold']} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{\\textbf{Stage 3 final test results for all 8 datasets.}",
        "Metrics computed with calibrated model ($T=0.976$) and fixed threshold $\\tau=0.5$.",
        "BOR (Boundary Oversegmentation Ratio) indicates boundary density relative to gold.",
        "W-F1 uses $\\pm 1$ message tolerance.}",
        "\\label{tab:stage3-all8}",
        "\\end{table*}",
    ])

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute Stage 3 metrics for all 8 datasets")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to calibrated model checkpoint")
    parser.add_argument("--datasets-dir", type=str, default=None,
                       help="Path to datasets directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
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
        datasets_path = project_root / "datasets"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = project_root / "paper" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading calibrated model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)
    print(f"Temperature: {temperature:.6f}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Store results
    all_results = {}
    failed_datasets = []

    print("\n" + "="*70)
    print("STAGE 3 EVALUATION - ALL 8 DATASETS")
    print(f"Temperature: {temperature:.6f}")
    print("="*70)

    for dataset_name in ALL_DATASETS:
        print(f"\n--- {DISPLAY_NAMES.get(dataset_name, dataset_name)} ---")

        try:
            # Load dialogues
            dialogues = load_dataset_dialogues(datasets_path, dataset_name)
            if not dialogues:
                print(f"  No dialogues loaded for {dataset_name}")
                failed_datasets.append((dataset_name, "No dialogues loaded"))
                continue

            print(f"  Loaded {len(dialogues)} dialogues")

            # Create examples and dataloader
            examples = create_examples_from_dialogues(dialogues)
            if not examples:
                print(f"  No examples created for {dataset_name}")
                failed_datasets.append((dataset_name, "No examples created"))
                continue

            dataset = BoundaryDataset(examples, tokenizer)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            # Get model predictions
            model_preds = get_model_predictions(model, dataloader, device, temperature, dialogues)
            metrics = compute_all_metrics(dialogues, model_preds)
            all_results[dataset_name] = metrics

            print(f"  W-F1={metrics['w_f1']:.3f}, BOR={metrics['bor']:.2f}, "
                  f"Purity={metrics['purity']:.3f}, Coverage={metrics['coverage']:.3f}, "
                  f"F1={metrics['f1']:.3f}")
            print(f"  Boundaries: {metrics['total_pred']} pred / {metrics['total_gold']} gold")

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_datasets.append((dataset_name, str(e)))

    # ==========================================================================
    # SANITY CHECK
    # ==========================================================================
    print("\n" + "="*70)
    print("SANITY CHECK vs PAPER VALUES (Tables 8-9)")
    print("="*70)

    diffs = sanity_check(all_results)
    for line in diffs:
        print(line)

    # ==========================================================================
    # OUTPUT JSON
    # ==========================================================================
    json_output = {
        "generated": datetime.now().isoformat(),
        "model_path": str(model_path),
        "temperature": temperature,
        "datasets": all_results,
        "failed_datasets": failed_datasets,
    }

    json_path = output_dir / "stage3_all8.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON output: {json_path}")

    # ==========================================================================
    # OUTPUT LATEX
    # ==========================================================================
    latex_content = generate_latex_table(all_results)
    latex_path = tables_dir / "appendix_stage3_all8.tex"
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"LaTeX output: {latex_path}")

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Dataset':<15} {'W-F1':>8} {'BOR':>8} {'Purity':>8} {'Coverage':>8} {'F1':>8} {'Pred':>8} {'Gold':>8}")
    print("-" * 85)

    for ds_name in ALL_DATASETS:
        if ds_name in all_results:
            m = all_results[ds_name]
            display = DISPLAY_NAMES.get(ds_name, ds_name)
            print(f"{display:<15} {m['w_f1']:>8.3f} {m['bor']:>8.2f} {m['purity']:>8.3f} "
                  f"{m['coverage']:>8.3f} {m['f1']:>8.3f} {m['total_pred']:>8} {m['total_gold']:>8}")

    # ==========================================================================
    # FAILED DATASETS
    # ==========================================================================
    if failed_datasets:
        print("\n" + "="*70)
        print("FAILED DATASETS")
        print("="*70)
        for ds, reason in failed_datasets:
            print(f"  {ds}: {reason}")

    # ==========================================================================
    # REPRODUCTION COMMAND
    # ==========================================================================
    print("\n" + "="*70)
    print("REPRODUCTION COMMAND")
    print("="*70)
    print("python paper/experiments/evaluation/compute_stage3_all8.py")
    print("")


if __name__ == "__main__":
    main()
