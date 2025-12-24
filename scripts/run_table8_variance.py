#!/usr/bin/env python3
"""
Run Stage-3 evaluation N times to compute variance statistics for Table 8.

This script runs the exact same evaluation pipeline as compute_stage3_all8.py
for N independent runs, then computes mean±std across runs.

Key finding: The Stage-3 evaluation is DETERMINISTIC given:
- Fixed model checkpoint (final_calibrated.pt with T=0.976)
- Fixed data (segmentation_file_test.json for each dataset)
- shuffle=False in DataLoader
- model.eval() disables dropout
- Fixed threshold τ=0.5

Therefore std=0 for all metrics, which we verify empirically.

Usage:
    python scripts/run_table8_variance.py --runs 5

Outputs:
    - results/table8_variance.csv
    - paper/tables/table8_variance.tex
    - results/table8_variance_notes.txt
"""

import torch
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Set, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from datetime import datetime
from dataclasses import dataclass
import argparse

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
)


# Datasets for Table 8 (main paper)
TABLE8_DATASETS = ["dialseg711", "superseg", "tiage"]

DISPLAY_NAMES = {
    "dialseg711": "DialSeg711",
    "superseg": "SuperSeg",
    "tiage": "TIAGE",
}


@dataclass
class DialogueData:
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
            "dialogue_idx": ex.get("dialogue_idx", 0),
            "position": ex.get("position", 0),
        }


def load_dataset_dialogues(datasets_path: Path, dataset_name: str) -> List[DialogueData]:
    test_file = datasets_path / dataset_name / "segmentation_file_test.json"
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
                        "dialogue_idx": dial_idx,
                        "position": user_idx,
                    })
                user_idx += 1
    return examples


def get_predictions(model, dataloader, device, temperature, n_dialogues) -> Dict[int, Set[int]]:
    model.eval()
    predictions = {i: set() for i in range(n_dialogues)}

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
                    predictions[dialogue_indices[i]].add(positions[i])

    return predictions


def compute_metrics(dialogues, predictions) -> Dict[str, float]:
    all_wf1 = []
    all_f1 = []
    all_purity = []
    all_coverage = []
    total_gold = 0
    total_pred = 0

    for dial_idx, dialogue in enumerate(dialogues):
        gold = dialogue.gold_boundaries
        pred = predictions.get(dial_idx, set())
        n = dialogue.num_messages

        # W-F1
        _, _, wf1 = compute_windowed_metrics(gold, pred, n, window=1)
        all_wf1.append(wf1)

        # Strict F1
        if not gold and not pred:
            f1 = 1.0
        elif not gold or not pred:
            f1 = 0.0
        else:
            tp = len(gold & pred)
            prec = tp / len(pred) if pred else 0
            rec = tp / len(gold) if gold else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        all_f1.append(f1)

        # Purity/Coverage
        gold_seg = boundaries_to_segments(gold, n)
        pred_seg = boundaries_to_segments(pred, n)
        purity, coverage = compute_purity_coverage(gold_seg, pred_seg)
        all_purity.append(purity)
        all_coverage.append(coverage)

        total_gold += len(gold)
        total_pred += len(pred)

    return {
        "w_f1": float(np.mean(all_wf1)),
        "bor": total_pred / total_gold if total_gold > 0 else 1.0,
        "f1": float(np.mean(all_f1)),
        "purity": float(np.mean(all_purity)),
        "coverage": float(np.mean(all_coverage)),
        "total_pred": total_pred,
        "total_gold": total_gold,
    }


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_evaluation(
    model, tokenizer, device, temperature, datasets_path, seed: int
) -> Dict[str, Dict[str, float]]:
    """Run evaluation for all Table 8 datasets with given seed."""
    set_all_seeds(seed)
    results = {}

    for dataset_name in TABLE8_DATASETS:
        dialogues = load_dataset_dialogues(datasets_path, dataset_name)
        examples = create_examples_from_dialogues(dialogues)

        dataset = BoundaryDataset(examples, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions = get_predictions(model, dataloader, device, temperature, len(dialogues))
        metrics = compute_metrics(dialogues, predictions)
        results[dataset_name] = metrics

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()

    N_RUNS = args.runs
    SEEDS = list(range(42, 42 + N_RUNS))

    print(f"Stage-3 Table 8 Variance Analysis")
    print(f"N={N_RUNS} runs with seeds {SEEDS}")
    print("="*70)

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = project_root / "paper" / "experiments" / "models" / "final_calibrated.pt"
    datasets_path = project_root / "datasets"

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)
    print(f"Temperature: {temperature:.6f}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Run N evaluations
    all_runs = []
    for run_idx, seed in enumerate(SEEDS):
        print(f"\nRun {run_idx + 1}/{N_RUNS} (seed={seed})...")
        results = run_single_evaluation(
            model, tokenizer, device, temperature, datasets_path, seed
        )
        all_runs.append(results)

        for ds in TABLE8_DATASETS:
            m = results[ds]
            print(f"  {ds}: W-F1={m['w_f1']:.4f}, BOR={m['bor']:.4f}")

    # Compute statistics
    print("\n" + "="*70)
    print("COMPUTING STATISTICS")
    print("="*70)

    stats = {}
    metrics = ["w_f1", "bor", "purity", "coverage", "f1"]

    for ds in TABLE8_DATASETS:
        stats[ds] = {}
        for metric in metrics:
            values = [run[ds][metric] for run in all_runs]
            stats[ds][f"{metric}_mean"] = np.mean(values)
            stats[ds][f"{metric}_std"] = np.std(values)
            stats[ds][f"{metric}_values"] = values

        # Add total_pred and total_gold (should be constant)
        stats[ds]["total_pred"] = all_runs[0][ds]["total_pred"]
        stats[ds]["total_gold"] = all_runs[0][ds]["total_gold"]

    # Check if deterministic
    is_deterministic = True
    for ds in TABLE8_DATASETS:
        for metric in metrics:
            if stats[ds][f"{metric}_std"] > 1e-10:
                is_deterministic = False
                break

    # =========================================================================
    # OUTPUT CSV
    # =========================================================================
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "table8_variance.csv"
    with open(csv_path, "w") as f:
        headers = ["dataset"]
        for m in metrics:
            headers.extend([f"{m}_mean", f"{m}_std"])
        headers.extend(["total_pred", "total_gold"])
        f.write(",".join(headers) + "\n")

        for ds in TABLE8_DATASETS:
            row = [ds]
            for m in metrics:
                row.append(f"{stats[ds][f'{m}_mean']:.6f}")
                row.append(f"{stats[ds][f'{m}_std']:.6f}")
            row.append(str(stats[ds]["total_pred"]))
            row.append(str(stats[ds]["total_gold"]))
            f.write(",".join(row) + "\n")

    print(f"CSV saved: {csv_path}")

    # =========================================================================
    # OUTPUT LATEX
    # =========================================================================
    tables_dir = project_root / "paper" / "tables"
    tables_dir.mkdir(exist_ok=True)

    latex_path = tables_dir / "table8_variance.tex"
    with open(latex_path, "w") as f:
        f.write("% Table 8 with variance statistics (N={} runs)\n".format(N_RUNS))
        f.write("% Generated by scripts/run_table8_variance.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Dataset & W-F1 & BOR & Purity & Coverage & Pred & Gold \\\\\n")
        f.write("\\midrule\n")

        for ds in TABLE8_DATASETS:
            s = stats[ds]
            display = DISPLAY_NAMES[ds]

            # Format: mean±std or just mean if std=0
            def fmt(metric):
                mean = s[f"{metric}_mean"]
                std = s[f"{metric}_std"]
                if std < 1e-6:
                    return f"{mean:.3f}"
                else:
                    return f"{mean:.3f}$\\pm${std:.3f}"

            def fmt_bor(metric):
                mean = s[f"{metric}_mean"]
                std = s[f"{metric}_std"]
                if std < 1e-6:
                    return f"\\textbf{{{mean:.2f}}}"
                else:
                    return f"\\textbf{{{mean:.2f}$\\pm${std:.2f}}}"

            f.write(f"{display} & {fmt('w_f1')} & {fmt_bor('bor')} & "
                   f"{fmt('purity')} & {fmt('coverage')} & "
                   f"{s['total_pred']} & {s['total_gold']} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{\\textbf{Stage 3 final test results.}\n")
        f.write(f"Metrics computed with calibrated model ($T={temperature:.3f}$) and threshold $\\tau=0.5$.\n")
        if is_deterministic:
            f.write("Results are deterministic; std=0 across $N={}$ runs.}}\n".format(N_RUNS))
        else:
            f.write("Mean$\\pm$std over $N={}$ runs.}}\n".format(N_RUNS))
        f.write("\\label{tab:stage3-variance}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX saved: {latex_path}")

    # =========================================================================
    # OUTPUT NOTES
    # =========================================================================
    notes_path = results_dir / "table8_variance_notes.txt"
    with open(notes_path, "w") as f:
        f.write("Table 8 Variance Analysis Notes\n")
        f.write("="*50 + "\n\n")
        f.write(f"N runs: {N_RUNS}\n")
        f.write(f"Seeds used: {SEEDS}\n")
        f.write(f"Model: final_calibrated.pt\n")
        f.write(f"Temperature: {temperature:.6f}\n")
        f.write(f"Threshold: 0.5 (fixed)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("Determinism Analysis:\n")
        f.write("-"*50 + "\n")
        if is_deterministic:
            f.write("RESULT: Evaluation is DETERMINISTIC (std=0 for all metrics)\n\n")
            f.write("Reasons:\n")
            f.write("1. DataLoader uses shuffle=False (fixed order)\n")
            f.write("2. model.eval() disables dropout\n")
            f.write("3. Fixed threshold τ=0.5 for boundary selection\n")
            f.write("4. torch.no_grad() during inference\n")
            f.write("5. No stochastic operations in forward pass\n\n")
            f.write("Implication: Reporting mean±std is redundant since std=0.\n")
            f.write("The paper can simply report point estimates.\n")
        else:
            f.write("RESULT: Evaluation has non-zero variance\n\n")
            f.write("Sources of variance:\n")
            for ds in TABLE8_DATASETS:
                f.write(f"\n{ds}:\n")
                for m in metrics:
                    std = stats[ds][f"{m}_std"]
                    if std > 1e-10:
                        f.write(f"  {m}: std={std:.6f}\n")

        f.write("\n" + "="*50 + "\n")
        f.write("Reproduction Command:\n")
        f.write("-"*50 + "\n")
        f.write(f"python scripts/run_table8_variance.py --runs {N_RUNS}\n")

    print(f"Notes saved: {notes_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Dataset':<12} {'W-F1':>15} {'BOR':>15} {'Purity':>15} {'Coverage':>15}")
    print("-"*70)

    for ds in TABLE8_DATASETS:
        s = stats[ds]

        def fmt(m):
            mean = s[f"{m}_mean"]
            std = s[f"{m}_std"]
            if std < 1e-6:
                return f"{mean:.3f}"
            return f"{mean:.3f}±{std:.3f}"

        print(f"{DISPLAY_NAMES[ds]:<12} {fmt('w_f1'):>15} {fmt('bor'):>15} "
              f"{fmt('purity'):>15} {fmt('coverage'):>15}")

    if is_deterministic:
        print("\n*** Results are DETERMINISTIC (std=0) ***")

    print(f"\nOutput files:")
    print(f"  CSV:   {csv_path}")
    print(f"  LaTeX: {latex_path}")
    print(f"  Notes: {notes_path}")


if __name__ == "__main__":
    main()
