#!/usr/bin/env python3
"""
LODO + 5-Fold CV Experiments - Standalone Script for Local Execution

This script runs DistilBERT training experiments that cannot run in a sandbox
environment due to memory/time constraints.

Usage:
    cd /path/to/episodic
    python paper/experiments/run_lodo_local.py --experiment lodo1
    python paper/experiments/run_lodo_local.py --experiment lodo2
    python paper/experiments/run_lodo_local.py --experiment cv5
    python paper/experiments/run_lodo_local.py --experiment all

Experiments:
    lodo1: Train on SuperSeg (2k) + TIAGE -> eval on DialSeg711
    lodo2: Train on DialSeg711 + TIAGE -> eval on SuperSeg
    cv5:   5-fold cross-validation on DialSeg711
    all:   Run all experiments
"""

import os
import sys
import json
import time
import argparse
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
warnings.filterwarnings("ignore", message=".*Token indices.*")
warnings.filterwarnings("ignore", message=".*Be aware.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_GAP = 2
TOLERANT_WINDOW = 3

# Training config
FINETUNE_EPOCHS = 3
FINETUNE_LR = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42
SUPERSEG_SUBSAMPLE = 2000

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments" / "results"

# GPT-5.2 caches
GPT52_CACHES = {
    "dialseg711": PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json",
    "superseg": PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DialogueData:
    dialogue_id: int
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_user_turns: int


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name: str, split: str = "test") -> List[DialogueData]:
    """Load a dialogue segmentation dataset."""
    file_path = DATASETS_DIR / dataset_name / f"segmentation_file_{split}.json"

    if not file_path.exists():
        print(f"  Warning: {file_path} does not exist")
        return []

    with open(file_path) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    dialogues = []
    dialogue_id = 0

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            # Extract messages
            messages = []
            gold_boundaries = set()
            prev_topic = None
            user_idx = 0

            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("utterance", turn.get("content", ""))
                messages.append({"role": role, "content": content})

                if role == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        gold_boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            if user_idx > 0:
                dialogues.append(DialogueData(
                    dialogue_id=dialogue_id,
                    messages=messages,
                    gold_boundaries=gold_boundaries,
                    num_user_turns=user_idx
                ))
                dialogue_id += 1

    return dialogues


def convert_dialogues_to_examples(dialogues: List[DialogueData], window_size: int = 3) -> List[Dict]:
    """Convert dialogues to training examples for boundary detection."""
    examples = []

    for dialogue in dialogues:
        window = []
        user_idx = 0

        for msg in dialogue.messages:
            if msg["role"] == "user":
                # Create example for this position
                examples.append({
                    "window": list(window),
                    "current_message": msg,
                    "label": 1 if user_idx in dialogue.gold_boundaries else 0,
                    "conversation_id": str(dialogue.dialogue_id),
                    "turn_index": user_idx,
                })

            # Update window
            window.append(msg)
            if len(window) > window_size * 2:
                window = window[-window_size * 2:]

            if msg["role"] == "user":
                user_idx += 1

    return examples


# =============================================================================
# DATASET CLASS
# =============================================================================

class BoundaryDataset(Dataset):
    """Dataset for boundary detection training."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format window as text
        window_text = self._format_window(ex["window"])
        current_text = ex["current_message"]["content"]

        # Suppress overflow warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoding = self.tokenizer(
                window_text,
                current_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                verbose=False
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
            "conv_id": ex.get("conversation_id", "unknown"),
            "turn_index": ex.get("turn_index", 0),
        }

    def _format_window(self, window: List[Dict]) -> str:
        parts = []
        for msg in window:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{role}: {content}")
        return " ".join(parts)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    train_examples: List[Dict],
    val_examples: List[Dict],
    output_path: Path,
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """Train a DistilBERT model on given examples."""

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Training on {len(train_examples)} examples, validating on {len(val_examples)}")
    print(f"  Device: {device}")

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    # Create datasets
    train_dataset = BoundaryDataset(train_examples, tokenizer, MAX_LENGTH)
    val_dataset = BoundaryDataset(val_examples, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * FINETUNE_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_f1 = 0
    best_model_state = None

    print(f"  Total batches per epoch: {len(train_loader)}")
    sys.stdout.flush()

    for epoch in range(FINETUNE_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                print(f"    Batch {batch_idx+1}/{len(train_loader)}, elapsed: {elapsed:.1f}s")
                sys.stdout.flush()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Compute F1
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"val_F1={f1:.3f}, val_P={precision:.3f}, val_R={recall:.3f}, "
              f"time={epoch_time:.1f}s")
        sys.stdout.flush()

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
    }, output_path)

    return model, best_f1


# =============================================================================
# SCORING
# =============================================================================

def score_dialogues_distilbert(
    model: nn.Module,
    dialogues: List[DialogueData],
    tokenizer,
    device: torch.device
) -> Dict[int, Dict[int, float]]:
    """Score all candidate boundaries in dialogues using DistilBERT."""
    model.eval()
    scores_by_dialogue = {}

    for dialogue in dialogues:
        examples = convert_dialogues_to_examples([dialogue])
        if not examples:
            continue

        dataset = BoundaryDataset(examples, tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        dialogue_scores = {}
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                turn_indices = batch["turn_index"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

                for i, turn_idx in enumerate(turn_indices):
                    dialogue_scores[int(turn_idx)] = float(probs[i].cpu())

        scores_by_dialogue[dialogue.dialogue_id] = dialogue_scores

    return scores_by_dialogue


def load_gpt52_scores(dataset_name: str) -> Dict[int, Dict[int, float]]:
    """Load GPT-5.2 scores from cache."""
    cache_path = GPT52_CACHES.get(dataset_name)
    if not cache_path or not cache_path.exists():
        print(f"  Warning: No GPT-5.2 cache for {dataset_name}")
        return {}

    with open(cache_path) as f:
        cache_data = json.load(f)

    scores_by_dialogue = defaultdict(dict)

    for key, entry in cache_data.items():
        if entry.get("invalid_first_token") or entry.get("missing_yn_in_toplogprobs"):
            continue

        # Read dialogue_id and position directly from entry (not from key)
        dialogue_id = entry.get("dialogue_id")
        turn_idx = entry.get("position")
        score = entry.get("score", 0)

        if dialogue_id is not None and turn_idx is not None:
            scores_by_dialogue[dialogue_id][turn_idx] = score

    print(f"  Loaded GPT-5.2 scores: {len(scores_by_dialogue)} dialogues, "
          f"{sum(len(s) for s in scores_by_dialogue.values())} positions")

    return dict(scores_by_dialogue)


# =============================================================================
# EVALUATION
# =============================================================================

def greedy_nms(scores: Dict[int, float], tau: float, min_gap: int = 2) -> Set[int]:
    """Apply greedy NMS to select boundaries."""
    candidates = [(idx, score) for idx, score in scores.items() if score >= tau]
    candidates.sort(key=lambda x: -x[1])

    selected = set()
    for idx, score in candidates:
        # Check if too close to already selected
        too_close = any(abs(idx - s) < min_gap for s in selected)
        if not too_close:
            selected.add(idx)

    return selected


def compute_exact_f1(pred: Set[int], gold: Set[int]) -> float:
    """Compute exact match F1."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_tolerant_f1(pred: Set[int], gold: Set[int], n: int, window: int = 3) -> float:
    """Compute tolerant (windowed) F1 with one-to-one matching."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # Build bipartite matching
    pred_list = sorted(pred)
    gold_list = sorted(gold)

    # Greedy one-to-one matching (closest first)
    matched_pred = set()
    matched_gold = set()

    pairs = []
    for p in pred_list:
        for g in gold_list:
            if abs(p - g) <= window:
                pairs.append((abs(p - g), p, g))

    pairs.sort()

    for dist, p, g in pairs:
        if p not in matched_pred and g not in matched_gold:
            matched_pred.add(p)
            matched_gold.add(g)

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_at_bor(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    target_bor: float = 1.0
) -> Dict:
    """Evaluate at a target BOR level."""
    # Collect all scores
    all_scores = []
    for dialogue_scores in scores_by_dialogue.values():
        all_scores.extend(dialogue_scores.values())

    if not all_scores:
        return {"exact_f1": 0, "tolerant_f1": 0, "bor": 0}

    # Binary search for tau that gives target BOR
    best_tau = None
    best_bor_diff = float("inf")

    for pct in range(0, 100, 2):
        tau = np.percentile(all_scores, pct)

        total_pred = 0
        total_gold = 0

        for dialogue_id, dialogue_scores in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            pred = greedy_nms(dialogue_scores, tau, MIN_GAP)
            total_pred += len(pred)
            total_gold += len(gold)

        if total_gold == 0:
            continue

        bor = total_pred / total_gold
        bor_diff = abs(bor - target_bor)

        if bor_diff < best_bor_diff:
            best_bor_diff = bor_diff
            best_tau = tau

    if best_tau is None:
        return {"exact_f1": 0, "tolerant_f1": 0, "bor": 0}

    # Evaluate at best tau
    exact_f1s = []
    tolerant_f1s = []
    total_pred = 0
    total_gold = 0

    for dialogue_id, dialogue_scores in scores_by_dialogue.items():
        gold = gold_by_dialogue.get(dialogue_id, set())
        if not gold:
            continue

        pred = greedy_nms(dialogue_scores, best_tau, MIN_GAP)
        n = max(max(pred) if pred else 0, max(gold) if gold else 0) + 1

        exact_f1s.append(compute_exact_f1(pred, gold))
        tolerant_f1s.append(compute_tolerant_f1(pred, gold, n, TOLERANT_WINDOW))

        total_pred += len(pred)
        total_gold += len(gold)

    bor = total_pred / total_gold if total_gold > 0 else 0

    return {
        "exact_f1": np.mean(exact_f1s) if exact_f1s else 0,
        "tolerant_f1": np.mean(tolerant_f1s) if tolerant_f1s else 0,
        "bor": bor,
        "n_dialogues": len(exact_f1s),
    }


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_lodo1(device: torch.device):
    """LODO-1: Train on SuperSeg + TIAGE -> eval on DialSeg711."""
    print("\n" + "="*60)
    print("LODO-1: Train on SuperSeg + TIAGE -> Evaluate on DialSeg711")
    print("="*60)

    start_time = time.time()

    # Load training data
    print("\nLoading training data...")
    superseg_train_full = load_dataset("superseg", "train")
    tiage_train = load_dataset("tiage", "train")

    # Subsample SuperSeg
    rng = np.random.default_rng(SEED)
    if len(superseg_train_full) > SUPERSEG_SUBSAMPLE:
        indices = rng.choice(len(superseg_train_full), SUPERSEG_SUBSAMPLE, replace=False)
        superseg_train = [superseg_train_full[i] for i in sorted(indices)]
        print(f"  SuperSeg train: {len(superseg_train)} dialogues (subsampled from {len(superseg_train_full)})")
    else:
        superseg_train = superseg_train_full
        print(f"  SuperSeg train: {len(superseg_train)} dialogues")
    print(f"  TIAGE train: {len(tiage_train)} dialogues")

    # Validation data
    superseg_val = load_dataset("superseg", "test")[:200]
    tiage_val = load_dataset("tiage", "test")[:50]

    # Convert to examples
    train_examples = (
        convert_dialogues_to_examples(superseg_train) +
        convert_dialogues_to_examples(tiage_train)
    )
    val_examples = (
        convert_dialogues_to_examples(superseg_val) +
        convert_dialogues_to_examples(tiage_val)
    )

    print(f"  Total training examples: {len(train_examples)}")
    print(f"  Total validation examples: {len(val_examples)}")

    # Train
    print("\nTraining...")
    model_path = OUTPUT_DIR / "models" / "lodo1_superseg_tiage.pt"

    model, best_f1 = train_model(train_examples, val_examples, model_path, device)

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Evaluate on DialSeg711
    print("\nEvaluating on DialSeg711...")
    dialseg_test = load_dataset("dialseg711", "test")
    print(f"  Test dialogues: {len(dialseg_test)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_scores = score_dialogues_distilbert(model, dialseg_test, tokenizer, device)
    gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in dialseg_test}

    # Load GPT-5.2 scores
    gpt52_scores = load_gpt52_scores("dialseg711")

    # Diagnostic: check ID alignment
    db_ids = set(distilbert_scores.keys())
    gpt_ids = set(gpt52_scores.keys())
    gold_ids = set(gold_by_dialogue.keys())
    common_ids = db_ids & gpt_ids & gold_ids
    print(f"  ID alignment: DistilBERT={len(db_ids)}, GPT-5.2={len(gpt_ids)}, "
          f"Gold={len(gold_ids)}, Common={len(common_ids)}")
    if len(common_ids) == 0:
        print(f"  WARNING: No common IDs! Sample DistilBERT IDs: {sorted(db_ids)[:5]}")
        print(f"  WARNING: Sample GPT-5.2 IDs: {sorted(gpt_ids)[:5]}")

    # Evaluate at BOR ~ 1.0
    distilbert_results = evaluate_at_bor(distilbert_scores, gold_by_dialogue, target_bor=1.0)
    gpt52_results = evaluate_at_bor(gpt52_scores, gold_by_dialogue, target_bor=1.0)

    print("\n" + "="*60)
    print("LODO-1 Results (at BOR ~ 1.0)")
    print("="*60)
    print(f"DistilBERT: Exact={distilbert_results['exact_f1']:.3f}, "
          f"Tolerant={distilbert_results['tolerant_f1']:.3f}, "
          f"BOR={distilbert_results['bor']:.2f}")
    print(f"GPT-5.2:    Exact={gpt52_results['exact_f1']:.3f}, "
          f"Tolerant={gpt52_results['tolerant_f1']:.3f}, "
          f"BOR={gpt52_results['bor']:.2f}")

    # Save results
    results = {
        "experiment": "lodo1",
        "train_data": "superseg_2k + tiage",
        "test_data": "dialseg711",
        "training_time_minutes": train_time / 60,
        "distilbert": distilbert_results,
        "gpt52": gpt52_results,
        "config": {
            "superseg_subsample": SUPERSEG_SUBSAMPLE,
            "batch_size": BATCH_SIZE,
            "epochs": FINETUNE_EPOCHS,
            "min_gap": MIN_GAP,
            "tolerant_window": TOLERANT_WINDOW,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "lodo1_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def run_lodo2(device: torch.device):
    """LODO-2: Train on DialSeg711 + TIAGE -> eval on SuperSeg."""
    print("\n" + "="*60)
    print("LODO-2: Train on DialSeg711 + TIAGE -> Evaluate on SuperSeg")
    print("="*60)

    start_time = time.time()

    # Load training data (use DialSeg711 test as training since no train split exists)
    print("\nLoading training data...")
    dialseg_train = load_dataset("dialseg711", "test")  # Use test as train
    tiage_train = load_dataset("tiage", "train")

    print(f"  DialSeg711 (as train): {len(dialseg_train)} dialogues")
    print(f"  TIAGE train: {len(tiage_train)} dialogues")

    # Validation data
    tiage_val = load_dataset("tiage", "test")[:50]

    # Convert to examples
    train_examples = (
        convert_dialogues_to_examples(dialseg_train) +
        convert_dialogues_to_examples(tiage_train)
    )
    val_examples = convert_dialogues_to_examples(tiage_val)

    print(f"  Total training examples: {len(train_examples)}")
    print(f"  Total validation examples: {len(val_examples)}")

    # Train
    print("\nTraining...")
    model_path = OUTPUT_DIR / "models" / "lodo2_dialseg_tiage.pt"

    model, best_f1 = train_model(train_examples, val_examples, model_path, device)

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Evaluate on SuperSeg
    print("\nEvaluating on SuperSeg...")
    superseg_test = load_dataset("superseg", "test")
    print(f"  Test dialogues: {len(superseg_test)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_scores = score_dialogues_distilbert(model, superseg_test, tokenizer, device)
    gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in superseg_test}

    # Load GPT-5.2 scores
    gpt52_scores = load_gpt52_scores("superseg")

    # Diagnostic: check ID alignment
    db_ids = set(distilbert_scores.keys())
    gpt_ids = set(gpt52_scores.keys())
    gold_ids = set(gold_by_dialogue.keys())
    common_ids = db_ids & gpt_ids & gold_ids
    print(f"  ID alignment: DistilBERT={len(db_ids)}, GPT-5.2={len(gpt_ids)}, "
          f"Gold={len(gold_ids)}, Common={len(common_ids)}")
    if len(common_ids) == 0:
        print(f"  WARNING: No common IDs! Sample DistilBERT IDs: {sorted(db_ids)[:5]}")
        print(f"  WARNING: Sample GPT-5.2 IDs: {sorted(gpt_ids)[:5]}")

    # Evaluate at BOR ~ 1.0
    distilbert_results = evaluate_at_bor(distilbert_scores, gold_by_dialogue, target_bor=1.0)
    gpt52_results = evaluate_at_bor(gpt52_scores, gold_by_dialogue, target_bor=1.0)

    print("\n" + "="*60)
    print("LODO-2 Results (at BOR ~ 1.0)")
    print("="*60)
    print(f"DistilBERT: Exact={distilbert_results['exact_f1']:.3f}, "
          f"Tolerant={distilbert_results['tolerant_f1']:.3f}, "
          f"BOR={distilbert_results['bor']:.2f}")
    print(f"GPT-5.2:    Exact={gpt52_results['exact_f1']:.3f}, "
          f"Tolerant={gpt52_results['tolerant_f1']:.3f}, "
          f"BOR={gpt52_results['bor']:.2f}")

    # Save results
    results = {
        "experiment": "lodo2",
        "train_data": "dialseg711 + tiage",
        "test_data": "superseg",
        "training_time_minutes": train_time / 60,
        "distilbert": distilbert_results,
        "gpt52": gpt52_results,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": FINETUNE_EPOCHS,
            "min_gap": MIN_GAP,
            "tolerant_window": TOLERANT_WINDOW,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "lodo2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def run_cv5(device: torch.device):
    """5-fold cross-validation on DialSeg711."""
    print("\n" + "="*60)
    print("5-Fold Cross-Validation on DialSeg711")
    print("="*60)

    start_time = time.time()

    # Load all DialSeg711 data
    print("\nLoading DialSeg711...")
    all_dialogues = load_dataset("dialseg711", "test")
    print(f"  Total dialogues: {len(all_dialogues)}")

    # Create stratified folds based on number of boundaries
    n_boundaries = [len(d.gold_boundaries) for d in all_dialogues]
    # Bin into groups for stratification
    boundary_bins = np.digitize(n_boundaries, bins=[1, 2, 3, 4, 5])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    gpt52_scores = load_gpt52_scores("dialseg711")

    fold_results = []
    all_distilbert_scores = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_dialogues, boundary_bins)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        fold_start = time.time()

        train_dialogues = [all_dialogues[i] for i in train_idx]
        test_dialogues = [all_dialogues[i] for i in test_idx]

        print(f"  Train: {len(train_dialogues)} dialogues")
        print(f"  Test: {len(test_dialogues)} dialogues")

        # Split training into train/val
        val_size = int(len(train_dialogues) * 0.1)
        val_dialogues = train_dialogues[:val_size]
        train_dialogues = train_dialogues[val_size:]

        train_examples = convert_dialogues_to_examples(train_dialogues)
        val_examples = convert_dialogues_to_examples(val_dialogues)

        # Train
        model_path = OUTPUT_DIR / "models" / f"cv5_fold{fold_idx+1}.pt"
        model, best_f1 = train_model(train_examples, val_examples, model_path, device)

        # Score test dialogues
        fold_scores = score_dialogues_distilbert(model, test_dialogues, tokenizer, device)
        all_distilbert_scores.update(fold_scores)

        # Evaluate this fold
        gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in test_dialogues}
        distilbert_results = evaluate_at_bor(fold_scores, gold_by_dialogue, target_bor=1.0)

        # GPT-5.2 for same dialogues
        fold_gpt52_scores = {d.dialogue_id: gpt52_scores.get(d.dialogue_id, {})
                            for d in test_dialogues}
        gpt52_results = evaluate_at_bor(fold_gpt52_scores, gold_by_dialogue, target_bor=1.0)

        fold_time = time.time() - fold_start
        print(f"  Fold {fold_idx+1} ({fold_time/60:.1f} min): "
              f"DistilBERT Exact={distilbert_results['exact_f1']:.3f}, "
              f"GPT-5.2 Exact={gpt52_results['exact_f1']:.3f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_dialogues),
            "test_size": len(test_dialogues),
            "distilbert": distilbert_results,
            "gpt52": gpt52_results,
            "time_minutes": fold_time / 60,
        })

    total_time = time.time() - start_time

    # Aggregate results
    print("\n" + "="*60)
    print("5-Fold CV Aggregate Results (at BOR ~ 1.0)")
    print("="*60)

    db_exact = [f["distilbert"]["exact_f1"] for f in fold_results]
    db_tol = [f["distilbert"]["tolerant_f1"] for f in fold_results]
    gpt_exact = [f["gpt52"]["exact_f1"] for f in fold_results]
    gpt_tol = [f["gpt52"]["tolerant_f1"] for f in fold_results]

    print(f"DistilBERT: Exact={np.mean(db_exact):.3f} +/- {np.std(db_exact):.3f}, "
          f"Tolerant={np.mean(db_tol):.3f} +/- {np.std(db_tol):.3f}")
    print(f"GPT-5.2:    Exact={np.mean(gpt_exact):.3f} +/- {np.std(gpt_exact):.3f}, "
          f"Tolerant={np.mean(gpt_tol):.3f} +/- {np.std(gpt_tol):.3f}")

    # Save results
    results = {
        "experiment": "cv5",
        "dataset": "dialseg711",
        "n_folds": 5,
        "total_time_minutes": total_time / 60,
        "folds": fold_results,
        "aggregate": {
            "distilbert_exact_mean": float(np.mean(db_exact)),
            "distilbert_exact_std": float(np.std(db_exact)),
            "distilbert_tolerant_mean": float(np.mean(db_tol)),
            "distilbert_tolerant_std": float(np.std(db_tol)),
            "gpt52_exact_mean": float(np.mean(gpt_exact)),
            "gpt52_exact_std": float(np.std(gpt_exact)),
            "gpt52_tolerant_mean": float(np.mean(gpt_tol)),
            "gpt52_tolerant_std": float(np.std(gpt_tol)),
        },
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": FINETUNE_EPOCHS,
            "min_gap": MIN_GAP,
            "tolerant_window": TOLERANT_WINDOW,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "cv5_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def print_header():
    """Print startup information."""
    print("="*60)
    print("LODO + 5-Fold CV Experiments")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
        est_time = {"lodo1": "5-8 min", "lodo2": "3-5 min", "cv5": "20-25 min"}
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        est_time = {"lodo1": "3-5 min", "lodo2": "2-3 min", "cv5": "15-20 min"}
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        est_time = {"lodo1": "15-20 min", "lodo2": "10-15 min", "cv5": "60-90 min"}

    print(f"Device: {device_name}")
    print(f"\nEstimated times:")
    print(f"  LODO-1 (SuperSeg+TIAGE -> DialSeg711): {est_time['lodo1']}")
    print(f"  LODO-2 (DialSeg711+TIAGE -> SuperSeg): {est_time['lodo2']}")
    print(f"  5-fold CV (DialSeg711): {est_time['cv5']}")
    print(f"\nConfig: batch_size={BATCH_SIZE}, epochs={FINETUNE_EPOCHS}, "
          f"superseg_subsample={SUPERSEG_SUBSAMPLE}")
    print(f"Selector: greedy NMS, MIN_GAP={MIN_GAP}")
    print(f"Metrics: Exact F1 (w=0), Tolerant W-F1 (w={TOLERANT_WINDOW})")
    print("="*60)

    return device


def main():
    parser = argparse.ArgumentParser(description="Run LODO and CV experiments")
    parser.add_argument("--experiment", choices=["lodo1", "lodo2", "cv5", "all"],
                       required=True, help="Which experiment to run")
    args = parser.parse_args()

    device = print_header()

    if args.experiment == "lodo1":
        run_lodo1(device)
    elif args.experiment == "lodo2":
        run_lodo2(device)
    elif args.experiment == "cv5":
        run_cv5(device)
    elif args.experiment == "all":
        run_lodo1(device)
        run_lodo2(device)
        run_cv5(device)

    print("\n" + "="*60)
    print("All requested experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
