#!/usr/bin/env python3
"""
Orchestrate training variance experiment for Table 8.

This script measures training-run variability by training K independently
initialized models (different fine-tuning seeds) and evaluating each on
the Stage-3 benchmark datasets (DialSeg711, SuperSeg, TIAGE).

Unlike run_table8_variance.py (which measures evaluation repeatability on
a fixed checkpoint), this script measures training variance - the variability
introduced by different random seeds during training.

Features:
- Crash-safe: Progress saved to JSONL after each seed completes
- Resumable: --resume skips seeds already in results file
- Aggregates mean±std across completed runs

Usage:
    python scripts/run_table8_trainvar.py --seeds 11,22,33 --outdir results/table8_trainvar
    python scripts/run_table8_trainvar.py --seeds 11,22,33 --outdir results/table8_trainvar --resume

Outputs:
    - results/table8_trainvar_runs.jsonl: Per-seed results (one JSON object per line)
    - results/table8_trainvar_aggregate.csv: Aggregated mean±std statistics
    - paper/tables/table8_trainvar.tex: LaTeX table with mean±std

Training pipeline (per seed):
    1. Stage 1: Pretrain on synthetic splice boundaries
    2. Stage 2: Fine-tune on benchmark datasets (DialSeg711, SuperSeg, TIAGE)
    3. Stage 3: Temperature calibration on validation data

Author: Generated for training variance analysis
"""

import argparse
import json
import subprocess
import sys
import os
import shutil
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "paper" / "experiments" / "training" / "train_3stage.py"
DATASETS_DIR = PROJECT_ROOT / "datasets"
SYNTHETIC_DIR = PROJECT_ROOT / "paper" / "experiments" / "data" / "synthetic"

# Datasets for Table 8 (main paper)
TABLE8_DATASETS = ["dialseg711", "superseg", "tiage"]

DISPLAY_NAMES = {
    "dialseg711": "DialSeg711",
    "superseg": "SuperSeg",
    "tiage": "TIAGE",
}


# ============================================================================
# DATA LOADING (adapted from compute_stage3_all8.py)
# ============================================================================

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
    """Load dialogues from a dataset for evaluation."""
    test_file = datasets_path / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

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
                        "dialogue_idx": dial_idx,
                        "position": user_idx,
                    })
                user_idx += 1
    return examples


# ============================================================================
# EVALUATION (adapted from compute_stage3_all8.py)
# ============================================================================

# Import evaluation functions from the project
sys.path.insert(0, str(PROJECT_ROOT))
from episodic.topics.evaluation import (
    compute_purity_coverage,
    boundaries_to_segments,
    compute_windowed_metrics,
)


def get_predictions(model, dataloader, device, temperature, n_dialogues) -> Dict[int, Set[int]]:
    """Run model inference and get predicted boundaries per dialogue."""
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


def compute_metrics(dialogues: List[DialogueData], predictions: Dict[int, Set[int]]) -> Dict[str, float]:
    """Compute evaluation metrics for a dataset."""
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

        # W-F1 with ±1 window
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
        "total_pred": int(total_pred),
        "total_gold": int(total_gold),
    }


def evaluate_checkpoint(
    model_path: Path,
    datasets_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a trained checkpoint on Table 8 datasets.

    Returns dict with:
        - temperature: learned temperature from checkpoint
        - per-dataset metrics (w_f1, bor, purity, coverage, f1, total_pred, total_gold)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Evaluate each dataset
    results = {"temperature": temperature, "datasets": {}}

    for dataset_name in TABLE8_DATASETS:
        dialogues = load_dataset_dialogues(datasets_dir, dataset_name)
        examples = create_examples_from_dialogues(dialogues)

        dataset = BoundaryDataset(examples, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions = get_predictions(model, dataloader, device, temperature, len(dialogues))
        metrics = compute_metrics(dialogues, predictions)
        results["datasets"][dataset_name] = metrics

    return results


# ============================================================================
# TRAINING (invokes train_3stage.py with modified seed)
# ============================================================================

def convert_synthetic_examples(raw_data: Dict) -> List[Dict]:
    """
    Convert synthetic data format to BoundaryDataset format.

    Input format (synthetic):
        - before: list of messages (window)
        - after: list of messages (first is current)
        - is_boundary: bool
        - conversation_id: str
        - boundary_index: int

    Output format (BoundaryDataset):
        - window: list of messages
        - current_message: dict with role/content
        - label: int (0 or 1)
        - conversation_id: str
        - source: str
    """
    examples = []

    # Handle both dict with training_examples and raw list
    if isinstance(raw_data, dict):
        if "training_examples" in raw_data:
            raw_examples = raw_data["training_examples"]
        elif "train" in raw_data:
            raw_examples = raw_data["train"]
        else:
            raw_examples = []
    else:
        raw_examples = raw_data

    for ex in raw_examples:
        # Get window (before messages)
        window = ex.get("before", ex.get("window", []))

        # Get current message (first of after, or current_message field)
        if "current_message" in ex:
            current_message = ex["current_message"]
        elif "after" in ex and len(ex["after"]) > 0:
            current_message = ex["after"][0]
        else:
            continue  # Skip malformed examples

        # Get label
        if "label" in ex:
            label = ex["label"]
        elif "is_boundary" in ex:
            label = 1 if ex["is_boundary"] else 0
        else:
            label = 0

        examples.append({
            "window": window,
            "current_message": current_message,
            "label": label,
            "conversation_id": ex.get("conversation_id", "unknown"),
            "source": "synthetic",
            "splice_type": ex.get("splice_type", "unknown"),
        })

    return examples


def train_with_seed(seed: int, output_dir: Path) -> Path:
    """
    Train a model with the specified seed.

    This function imports and modifies the training config to use
    the specified seed, then runs the full 3-stage training pipeline.

    Returns the path to the final calibrated checkpoint.
    """
    # Import the training module
    sys.path.insert(0, str(TRAINING_SCRIPT.parent))

    # We need to modify TrainingConfig to use our seed
    # Import the module dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_3stage", TRAINING_SCRIPT)
    train_module = importlib.util.module_from_spec(spec)

    # Patch the config before loading
    original_training_config = None

    try:
        # Load the module
        spec.loader.exec_module(train_module)

        # Create a modified config with our seed
        config = train_module.TrainingConfig()
        config.seed = seed

        # Set all random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Determine device
        device = torch.device("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        tokenizer = train_module.DistilBertTokenizer.from_pretrained(config.model_name)

        # Initialize model
        model = train_module.DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2
        )
        model.to(device)

        # ================================================================
        # Stage 1: Pretrain on splice boundaries
        # ================================================================
        print(f"[Seed {seed}] Stage 1: Pretraining on splice boundaries...")

        splice_train = SYNTHETIC_DIR / "synthetic_large_train.json"
        splice_val = SYNTHETIC_DIR / "synthetic_large_val.json"

        if not splice_train.exists():
            raise FileNotFoundError(f"Synthetic training data not found: {splice_train}")

        with open(splice_train) as f:
            train_data = json.load(f)
        with open(splice_val) as f:
            val_data = json.load(f)

        # Convert synthetic data format to BoundaryDataset format
        train_examples = convert_synthetic_examples(train_data)
        val_examples = convert_synthetic_examples(val_data)

        print(f"[Seed {seed}] Loaded {len(train_examples)} train, {len(val_examples)} val examples")

        train_dataset = train_module.BoundaryDataset(train_examples, tokenizer, config.max_length)
        val_dataset = train_module.BoundaryDataset(val_examples, tokenizer, config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=config.pretrain_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.pretrain_batch_size)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay
        )
        total_steps = len(train_loader) * config.pretrain_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = train_module.get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_f1 = 0
        for epoch in range(config.pretrain_epochs):
            loss = train_module.train_epoch(
                model, train_loader, optimizer, scheduler, device,
                desc=f"[Seed {seed}] Pretrain {epoch+1}/{config.pretrain_epochs}"
            )
            metrics, _, _ = train_module.evaluate(model, val_loader, device)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "stage": "pretrain",
                    "best_f1": best_f1,
                }, output_dir / "pretrained_splice.pt")

        print(f"[Seed {seed}] Stage 1 complete. Best F1: {best_f1:.3f}")

        # ================================================================
        # Stage 2: Fine-tune on benchmark datasets
        # ================================================================
        print(f"[Seed {seed}] Stage 2: Fine-tuning on benchmark datasets...")

        benchmark_data = train_module.load_benchmark_data(DATASETS_DIR)

        if len(benchmark_data["train"]) == 0:
            raise ValueError("No benchmark training data found!")

        train_dataset = train_module.BoundaryDataset(benchmark_data["train"], tokenizer, config.max_length)
        val_dataset = train_module.BoundaryDataset(benchmark_data["val"], tokenizer, config.max_length)
        test_dataset = train_module.BoundaryDataset(benchmark_data["test"], tokenizer, config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=config.finetune_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.finetune_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.finetune_batch_size)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.finetune_lr,
            weight_decay=config.weight_decay
        )
        total_steps = len(train_loader) * config.finetune_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = train_module.get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_f1 = 0
        for epoch in range(config.finetune_epochs):
            loss = train_module.train_epoch(
                model, train_loader, optimizer, scheduler, device,
                desc=f"[Seed {seed}] Finetune {epoch+1}/{config.finetune_epochs}"
            )
            metrics, _, _ = train_module.evaluate(model, val_loader, device)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "stage": "finetune",
                    "best_f1": best_f1,
                }, output_dir / "finetuned_benchmark.pt")

        print(f"[Seed {seed}] Stage 2 complete. Best F1: {best_f1:.3f}")

        # Load best finetuned model
        if (output_dir / "finetuned_benchmark.pt").exists():
            checkpoint = torch.load(output_dir / "finetuned_benchmark.pt",
                                   map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        # ================================================================
        # Stage 3: Temperature calibration
        # ================================================================
        print(f"[Seed {seed}] Stage 3: Temperature calibration...")

        temperature = train_module.calibrate_model(model, val_loader, device)
        print(f"[Seed {seed}] Learned temperature: {temperature:.4f}")

        # Final evaluation
        metrics, preds, labels = train_module.evaluate(model, test_loader, device, temperature)

        # Save final calibrated model
        final_path = output_dir / "final_calibrated.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "temperature": temperature,
            "test_metrics": metrics,
            "seed": seed,
        }, final_path)

        print(f"[Seed {seed}] Stage 3 complete. Temperature: {temperature:.4f}")
        print(f"[Seed {seed}] Saved checkpoint: {final_path}")

        return final_path

    finally:
        # Clean up sys.path
        if str(TRAINING_SCRIPT.parent) in sys.path:
            sys.path.remove(str(TRAINING_SCRIPT.parent))


# ============================================================================
# CRASH-SAFE I/O
# ============================================================================

def load_completed_seeds(jsonl_path: Path) -> Set[int]:
    """Load seeds that have already been completed from the JSONL file."""
    completed = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["seed"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def append_result(jsonl_path: Path, result: Dict):
    """Append a result record to the JSONL file with immediate flush."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_all_results(jsonl_path: Path) -> List[Dict]:
    """Load all results from the JSONL file."""
    results = []
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and dirty status."""
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        commit_hash = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        is_dirty = len(result.stdout.strip()) > 0 if result.returncode == 0 else True

        return {
            "commit": commit_hash,
            "dirty": is_dirty,
            "commit_display": f"{commit_hash}{'*' if is_dirty else ''}"
        }
    except Exception:
        return {"commit": "unknown", "dirty": True, "commit_display": "unknown*"}


# ============================================================================
# AGGREGATION
# ============================================================================

def compute_aggregate_stats(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean±std across seeds for each dataset and metric."""
    stats = {}
    metrics = ["w_f1", "bor", "purity", "coverage", "f1"]

    for ds_name in TABLE8_DATASETS:
        stats[ds_name] = {}

        for metric in metrics:
            values = []
            for r in results:
                if ds_name in r.get("datasets", {}):
                    values.append(r["datasets"][ds_name][metric])

            if values:
                stats[ds_name][f"{metric}_mean"] = float(np.mean(values))
                stats[ds_name][f"{metric}_std"] = float(np.std(values))
                stats[ds_name][f"{metric}_n"] = len(values)
            else:
                stats[ds_name][f"{metric}_mean"] = 0.0
                stats[ds_name][f"{metric}_std"] = 0.0
                stats[ds_name][f"{metric}_n"] = 0

        # Pred/Gold counts (report mean if they vary, otherwise just the value)
        pred_values = [r["datasets"][ds_name]["total_pred"]
                      for r in results if ds_name in r.get("datasets", {})]
        gold_values = [r["datasets"][ds_name]["total_gold"]
                      for r in results if ds_name in r.get("datasets", {})]

        if pred_values:
            stats[ds_name]["pred_mean"] = float(np.mean(pred_values))
            stats[ds_name]["pred_std"] = float(np.std(pred_values))
        if gold_values:
            stats[ds_name]["gold_mean"] = float(np.mean(gold_values))
            stats[ds_name]["gold_std"] = float(np.std(gold_values))

    # Temperature statistics
    temp_values = [r.get("temperature", 1.0) for r in results]
    stats["_meta"] = {
        "temperature_mean": float(np.mean(temp_values)),
        "temperature_std": float(np.std(temp_values)),
        "n_runs": len(results),
        "seeds": [r["seed"] for r in results],
    }

    return stats


def write_aggregate_csv(stats: Dict, csv_path: Path):
    """Write aggregated statistics to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        metrics = ["w_f1", "bor", "purity", "coverage", "f1"]
        header = ["dataset"]
        for m in metrics:
            header.extend([f"{m}_mean", f"{m}_std"])
        header.extend(["pred_mean", "pred_std", "gold_mean", "gold_std"])
        writer.writerow(header)

        # Data rows
        for ds_name in TABLE8_DATASETS:
            row = [ds_name]
            for m in metrics:
                row.append(f"{stats[ds_name][f'{m}_mean']:.6f}")
                row.append(f"{stats[ds_name][f'{m}_std']:.6f}")
            row.append(f"{stats[ds_name]['pred_mean']:.1f}")
            row.append(f"{stats[ds_name]['pred_std']:.1f}")
            row.append(f"{stats[ds_name]['gold_mean']:.1f}")
            row.append(f"{stats[ds_name]['gold_std']:.1f}")
            writer.writerow(row)

        # Meta row
        meta = stats["_meta"]
        writer.writerow([])
        writer.writerow(["# Meta information"])
        writer.writerow(["n_runs", meta["n_runs"]])
        writer.writerow(["seeds", ",".join(map(str, meta["seeds"]))])
        writer.writerow(["temperature_mean", f"{meta['temperature_mean']:.6f}"])
        writer.writerow(["temperature_std", f"{meta['temperature_std']:.6f}"])


def write_latex_table(stats: Dict, tex_path: Path):
    """Write LaTeX table with mean±std (3 decimal places)."""
    meta = stats["_meta"]
    n_runs = meta["n_runs"]
    temp_mean = meta["temperature_mean"]
    temp_std = meta["temperature_std"]

    def fmt(ds, metric, decimals=3):
        """Format metric as mean±std or just mean if std≈0."""
        mean = stats[ds][f"{metric}_mean"]
        std = stats[ds][f"{metric}_std"]
        if std < 0.0005:  # Effectively zero
            return f"{mean:.{decimals}f}"
        else:
            return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"

    def fmt_bor(ds):
        """Format BOR with 2 decimal places, bold."""
        mean = stats[ds]["bor_mean"]
        std = stats[ds]["bor_std"]
        if std < 0.005:
            return f"\\textbf{{{mean:.2f}}}"
        else:
            return f"\\textbf{{{mean:.2f}$\\pm${std:.2f}}}"

    def fmt_count(ds, which):
        """Format Pred/Gold counts."""
        mean = stats[ds][f"{which}_mean"]
        std = stats[ds][f"{which}_std"]
        if std < 0.5:
            return f"{int(round(mean))}"
        else:
            return f"{mean:.0f}$\\pm${std:.0f}"

    lines = [
        f"% Table 8 with training variance statistics (K={n_runs} seeds)",
        "% Generated by scripts/run_table8_trainvar.py",
        f"% Seeds: {', '.join(map(str, meta['seeds']))}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & W-F1 & BOR & Purity & Coverage & Pred & Gold \\\\",
        "\\midrule",
    ]

    for ds in TABLE8_DATASETS:
        display = DISPLAY_NAMES[ds]
        lines.append(
            f"{display} & {fmt(ds, 'w_f1')} & {fmt_bor(ds)} & "
            f"{fmt(ds, 'purity')} & {fmt(ds, 'coverage')} & "
            f"{fmt_count(ds, 'pred')} & {fmt_count(ds, 'gold')} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{\\textbf{Stage 3 final test results (training variance).}",
    ])

    if temp_std < 0.001:
        lines.append(f"Mean$\\pm$std over $K={n_runs}$ independently trained models ($T={temp_mean:.3f}$, $\\tau=0.5$).")
    else:
        lines.append(f"Mean$\\pm$std over $K={n_runs}$ independently trained models ($T={temp_mean:.3f}\\pm{temp_std:.3f}$, $\\tau=0.5$).")

    lines.extend([
        "Variance reflects training RNG only (different fine-tuning seeds).}",
        "\\label{tab:stage3-trainvar}",
        "\\end{table}",
    ])

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run training variance experiment for Table 8"
    )
    parser.add_argument(
        "--seeds", type=str, default="11,22,33",
        help="Comma-separated list of training seeds (default: 11,22,33)"
    )
    parser.add_argument(
        "--outdir", type=str, default="results/table8_trainvar",
        help="Output directory for results (default: results/table8_trainvar)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip seeds that already exist in results file"
    )
    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Setup output paths
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outdir / "table8_trainvar_runs.jsonl"
    csv_path = outdir / "table8_trainvar_aggregate.csv"
    tex_path = PROJECT_ROOT / "paper" / "tables" / "table8_trainvar.tex"
    notes_path = outdir / "table8_trainvar_notes.txt"

    # Get git info
    git_info = get_git_info()

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("TRAINING VARIANCE EXPERIMENT FOR TABLE 8")
    print("="*70)
    print(f"Seeds: {seeds}")
    print(f"Output directory: {outdir}")
    print(f"Device: {device}")
    print(f"Git: {git_info['commit_display']}")
    print(f"Resume mode: {args.resume}")
    print("="*70)

    # Check which seeds are already done
    completed_seeds = load_completed_seeds(jsonl_path) if args.resume else set()
    if completed_seeds:
        print(f"Already completed seeds: {sorted(completed_seeds)}")

    # Process each seed
    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n[Seed {seed}] Already completed, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"PROCESSING SEED {seed}")
        print(f"{'='*70}")

        # Create seed-specific output directory
        seed_dir = outdir / f"seed_{seed}"

        try:
            # Train model
            print(f"[Seed {seed}] Training model...")
            checkpoint_path = train_with_seed(seed, seed_dir)

            # Evaluate model
            print(f"[Seed {seed}] Evaluating checkpoint...")
            eval_results = evaluate_checkpoint(checkpoint_path, DATASETS_DIR, device)

            # Build result record
            result = {
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "git_commit": git_info["commit"],
                "git_dirty": git_info["dirty"],
                "checkpoint_path": str(checkpoint_path),
                "temperature": eval_results["temperature"],
                "tau": 0.5,  # Fixed threshold
                "datasets": eval_results["datasets"],
            }

            # Append to JSONL (crash-safe)
            append_result(jsonl_path, result)
            print(f"[Seed {seed}] Results saved to {jsonl_path}")

            # Print summary
            print(f"[Seed {seed}] Summary:")
            print(f"  Temperature: {result['temperature']:.4f}")
            for ds_name in TABLE8_DATASETS:
                m = result["datasets"][ds_name]
                print(f"  {DISPLAY_NAMES[ds_name]}: W-F1={m['w_f1']:.3f}, BOR={m['bor']:.2f}")

        except Exception as e:
            print(f"[Seed {seed}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Recompute aggregates after each seed
        all_results = load_all_results(jsonl_path)
        if all_results:
            stats = compute_aggregate_stats(all_results)

            # Write CSV
            write_aggregate_csv(stats, csv_path)
            print(f"[Aggregate] Updated {csv_path}")

            # Write LaTeX
            write_latex_table(stats, tex_path)
            print(f"[Aggregate] Updated {tex_path}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    all_results = load_all_results(jsonl_path)
    if not all_results:
        print("No results available.")
        return

    stats = compute_aggregate_stats(all_results)
    meta = stats["_meta"]

    print(f"Completed runs: {meta['n_runs']}")
    print(f"Seeds: {meta['seeds']}")
    print(f"Temperature: {meta['temperature_mean']:.4f} ± {meta['temperature_std']:.4f}")
    print()

    print(f"{'Dataset':<12} {'W-F1':>18} {'BOR':>15} {'Purity':>18} {'Coverage':>18}")
    print("-"*80)

    for ds in TABLE8_DATASETS:
        s = stats[ds]
        def fmt(m):
            mean = s[f"{m}_mean"]
            std = s[f"{m}_std"]
            if std < 0.001:
                return f"{mean:.3f}"
            return f"{mean:.3f}±{std:.3f}"

        print(f"{DISPLAY_NAMES[ds]:<12} {fmt('w_f1'):>18} {fmt('bor'):>15} "
              f"{fmt('purity'):>18} {fmt('coverage'):>18}")

    # Write notes file
    with open(notes_path, "w") as f:
        f.write("Table 8 Training Variance Analysis Notes\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("TRAINING COMMAND TEMPLATE:\n")
        f.write("-"*50 + "\n")
        f.write("The training is invoked programmatically via train_with_seed() which:\n")
        f.write("1. Imports paper/experiments/training/train_3stage.py\n")
        f.write("2. Modifies TrainingConfig.seed to the specified value\n")
        f.write("3. Runs the full 3-stage pipeline:\n")
        f.write("   - Stage 1: Pretrain on synthetic splice boundaries\n")
        f.write("   - Stage 2: Fine-tune on DialSeg711, SuperSeg, TIAGE\n")
        f.write("   - Stage 3: Temperature calibration on validation data\n")
        f.write("\n")
        f.write("Equivalent manual command (if --seed were supported):\n")
        f.write("  python paper/experiments/training/train_3stage.py --seed <SEED> --output-dir <DIR>\n")
        f.write("\n")

        f.write("EVALUATION COMMAND TEMPLATE:\n")
        f.write("-"*50 + "\n")
        f.write("python paper/experiments/evaluation/compute_stage3_all8.py \\\n")
        f.write("    --model-path <seed_dir>/final_calibrated.pt \\\n")
        f.write("    --datasets-dir datasets\n")
        f.write("\n")

        f.write("SEEDS USED:\n")
        f.write("-"*50 + "\n")
        f.write(f"Seeds: {meta['seeds']}\n")
        f.write(f"Number of runs: {meta['n_runs']}\n")
        f.write("\n")

        f.write("WHAT VARIES ACROSS RUNS:\n")
        f.write("-"*50 + "\n")
        f.write("- Training RNG seed (affects weight initialization, data shuffling, dropout)\n")
        f.write("- Learned temperature T (from Stage 3 calibration)\n")
        f.write("- Model weights (from training)\n")
        f.write("\n")
        f.write("WHAT IS FIXED:\n")
        f.write("- Model architecture (DistilBERT)\n")
        f.write("- Hyperparameters (lr, epochs, batch size, etc.)\n")
        f.write("- Decision threshold τ=0.5\n")
        f.write("- Evaluation datasets and preprocessing\n")
        f.write("- Window size (8 messages)\n")
        f.write("\n")

        f.write("RESUME BEHAVIOR:\n")
        f.write("-"*50 + "\n")
        f.write("With --resume flag:\n")
        f.write("- Checks table8_trainvar_runs.jsonl for completed seeds\n")
        f.write("- Skips any seed already present in the file\n")
        f.write("- Continues with remaining seeds\n")
        f.write("- Recomputes aggregates after each new seed\n")
        f.write("\n")
        f.write("Without --resume:\n")
        f.write("- Starts fresh (existing JSONL is appended to, not overwritten)\n")
        f.write("- Warning: May create duplicate entries for seeds\n")
        f.write("\n")

        f.write("RESULTS:\n")
        f.write("-"*50 + "\n")
        f.write(f"Temperature: {meta['temperature_mean']:.4f} ± {meta['temperature_std']:.4f}\n\n")

        for ds in TABLE8_DATASETS:
            s = stats[ds]
            f.write(f"{DISPLAY_NAMES[ds]}:\n")
            for m in ["w_f1", "bor", "purity", "coverage", "f1"]:
                f.write(f"  {m}: {s[f'{m}_mean']:.4f} ± {s[f'{m}_std']:.4f}\n")
            f.write(f"  Pred: {s['pred_mean']:.0f} ± {s['pred_std']:.1f}\n")
            f.write(f"  Gold: {s['gold_mean']:.0f} ± {s['gold_std']:.1f}\n")
            f.write("\n")

        f.write("OUTPUT FILES:\n")
        f.write("-"*50 + "\n")
        f.write(f"- {jsonl_path} (per-seed results, JSONL)\n")
        f.write(f"- {csv_path} (aggregated statistics, CSV)\n")
        f.write(f"- {tex_path} (LaTeX table)\n")
        f.write(f"- {notes_path} (this file)\n")
        f.write("\n")

        f.write("REPRODUCTION COMMAND:\n")
        f.write("-"*50 + "\n")
        seeds_str = ",".join(map(str, meta['seeds']))
        f.write(f"python scripts/run_table8_trainvar.py --seeds {seeds_str} --outdir {args.outdir}\n")

    print(f"\nNotes saved: {notes_path}")
    print(f"\nOutput files:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  LaTeX: {tex_path}")
    print(f"  Notes: {notes_path}")


if __name__ == "__main__":
    main()
