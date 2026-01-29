#!/usr/bin/env python3
"""
LODO + 5-fold CV Experiments for Fair DistilBERT vs GPT-5.2 Comparison

Experiments:
1. LODO-1: Train on SuperSeg + TIAGE → evaluate on DialSeg711
2. LODO-2: Train on DialSeg711 + TIAGE → evaluate on SuperSeg
3. 5-fold CV on DialSeg711 (in-domain upper bound)

Usage:
    python paper/experiments/lodo_cv_experiments.py --experiment lodo1
    python paper/experiments/lodo_cv_experiments.py --experiment lodo2
    python paper/experiments/lodo_cv_experiments.py --experiment cv5
    python paper/experiments/lodo_cv_experiments.py --experiment all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import subprocess
import warnings
import logging

# Suppress tokenizer warnings completely
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
warnings.filterwarnings("ignore", message=".*Token indices.*")
warnings.filterwarnings("ignore", message=".*Be aware.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_GAP = 2
TOLERANT_WINDOW = 3

# Training config - lighter for faster iteration
FINETUNE_EPOCHS = 3
FINETUNE_LR = 2e-5
BATCH_SIZE = 16  # Reduced from 32
MAX_LENGTH = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42
SUPERSEG_SUBSAMPLE = 2000  # Subsample SuperSeg to reduce training time

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "paper" / "experiments" / "models"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments"
SYNTHETIC_DIR = PROJECT_ROOT / "paper" / "experiments" / "data" / "synthetic"

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
    """Load a dataset split."""
    file_path = DATASETS_DIR / dataset_name / f"segmentation_file_{split}.json"
    if not file_path.exists():
        return []

    with open(file_path) as f:
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
                num_user_turns=num_user_turns
            ))
            dialogue_id += 1

    return dialogues


def convert_dialogues_to_examples(dialogues: List[DialogueData], window_size: int = 8) -> List[Dict]:
    """Convert dialogues to training examples."""
    examples = []

    for dialogue in dialogues:
        messages = dialogue.messages
        user_idx = 0

        for i, msg in enumerate(messages):
            if msg["role"] == "user" and user_idx > 0:
                window_start = max(0, i - window_size)
                window = messages[window_start:i]

                examples.append({
                    "window": window,
                    "current_message": msg,
                    "label": 1 if user_idx in dialogue.gold_boundaries else 0,
                    "conversation_id": str(dialogue.dialogue_id),
                    "turn_index": user_idx,
                })

            if msg["role"] == "user":
                user_idx += 1

    return examples


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

        # Suppress overflow warnings with verbose=False
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
    start_from_pretrained: bool = True,
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

    if start_from_pretrained and (MODELS_DIR / "final_calibrated.pt").exists():
        print("  Loading from final_calibrated.pt")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        checkpoint = torch.load(MODELS_DIR / "final_calibrated.pt",
                               map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("  Training from scratch (distilbert-base-uncased)")
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
    epoch_start_time = time.time()

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

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
    }, output_path)

    return model, best_f1


def score_dialogues_distilbert(
    model: nn.Module,
    dialogues: List[DialogueData],
    tokenizer,
    device: torch.device
) -> Dict[int, Dict[int, float]]:
    """Score all dialogues with DistilBERT model."""
    model.eval()
    scores_by_dialogue = {}

    for dialogue in tqdm(dialogues, desc="  Scoring dialogues", leave=False):
        messages = dialogue.messages
        scores = {}

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    # Format text
                    window_text = " ".join([f"{m['role']}: {m['content']}" for m in window])
                    current_text = msg["content"]

                    encoding = tokenizer(
                        window_text, current_text,
                        truncation=True, max_length=MAX_LENGTH,
                        padding="max_length", return_tensors="pt"
                    )

                    with torch.no_grad():
                        inputs = {k: v.to(device) for k, v in encoding.items()}
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        score = probs[0, 1].item()

                    scores[user_idx] = score

                user_idx += 1

        scores_by_dialogue[dialogue.dialogue_id] = scores

    return scores_by_dialogue


def load_gpt52_scores(dataset_name: str) -> Dict[int, Dict[int, float]]:
    """Load GPT-5.2 scores from cache."""
    cache_file = GPT52_CACHES.get(dataset_name)
    if not cache_file or not cache_file.exists():
        raise FileNotFoundError(f"GPT-5.2 cache not found for {dataset_name}")

    with open(cache_file) as f:
        cache = json.load(f)

    scores_by_dialogue = defaultdict(dict)

    for key, entry in cache.items():
        if entry.get("missing_yn_in_toplogprobs") or entry.get("invalid_first_token"):
            continue

        dialogue_id = entry["dialogue_id"]
        position = entry["position"]
        score = entry["score"]

        scores_by_dialogue[dialogue_id][position] = score

    return dict(scores_by_dialogue)


# =============================================================================
# EVALUATION
# =============================================================================

def greedy_nms_predict(scores: Dict[int, float], tau: float, min_gap: int) -> Set[int]:
    """Greedy NMS prediction."""
    candidates = [(pos, score) for pos, score in scores.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    predicted = set()
    for pos, score in candidates:
        too_close = any(abs(pos - p) < min_gap for p in predicted)
        if not too_close:
            predicted.add(pos)

    return predicted


def compute_exact_f1(gold: Set[int], pred: Set[int]) -> float:
    """Compute exact F1."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def compute_tolerant_f1(gold: Set[int], pred: Set[int], window: int = 3) -> float:
    """Compute tolerant F1 with one-to-one matching."""
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0 if pred else 1.0
    if not pred:
        return 0.0

    candidates = []
    for g in gold:
        for p in pred:
            dist = abs(g - p)
            if dist <= window:
                candidates.append((dist, g, p))

    candidates.sort(key=lambda x: x[0])
    matched_gold = set()
    matched_pred = set()

    for dist, g, p in candidates:
        if g not in matched_gold and p not in matched_pred:
            matched_gold.add(g)
            matched_pred.add(p)

    tp = len(matched_gold)
    precision = tp / len(pred)
    recall = tp / len(gold)

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def evaluate_at_bor(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    target_bor: float = 1.0
) -> Dict[str, float]:
    """Evaluate at target BOR."""
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    # Find tau for target BOR
    best_tau = None
    best_bor_diff = float('inf')

    for pct in range(1, 100):
        tau = np.percentile(all_scores, pct)

        total_pred = 0
        total_gold = 0

        for dialogue_id, scores in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            pred = greedy_nms_predict(scores, tau, MIN_GAP)
            total_pred += len(pred)
            total_gold += len(gold)

        bor = total_pred / total_gold if total_gold > 0 else 0
        bor_diff = abs(bor - target_bor)

        if bor_diff < best_bor_diff:
            best_bor_diff = bor_diff
            best_tau = tau
            actual_bor = bor

    # Compute metrics at best tau
    exact_f1s = []
    tolerant_f1s = []

    for dialogue_id, scores in scores_by_dialogue.items():
        gold = gold_by_dialogue.get(dialogue_id, set())
        pred = greedy_nms_predict(scores, best_tau, MIN_GAP)

        exact_f1s.append(compute_exact_f1(gold, pred))
        tolerant_f1s.append(compute_tolerant_f1(gold, pred, TOLERANT_WINDOW))

    return {
        "tau": float(best_tau),
        "bor": actual_bor,
        "exact_f1": float(np.mean(exact_f1s)),
        "tolerant_f1": float(np.mean(tolerant_f1s)),
    }


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_lodo1():
    """LODO-1: Train on SuperSeg + TIAGE → evaluate on DialSeg711."""
    print("\n" + "="*60)
    print("LODO-1: Train on SuperSeg + TIAGE → Evaluate on DialSeg711")
    print("="*60)

    start_time = time.time()

    # Load training data
    print("\nLoading training data...")
    superseg_train_full = load_dataset("superseg", "train")
    tiage_train = load_dataset("tiage", "train")

    # Subsample SuperSeg for faster training
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
    superseg_val = load_dataset("superseg", "test")[:200]  # Use subset for validation
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
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")

    model_path = OUTPUT_DIR / "models" / "lodo1_superseg_tiage.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model, best_f1 = train_model(train_examples, val_examples, model_path,
                                  start_from_pretrained=False, device=device)

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

    # Evaluate at BOR ≈ 1.0
    distilbert_results = evaluate_at_bor(distilbert_scores, gold_by_dialogue, target_bor=1.0)
    gpt52_results = evaluate_at_bor(gpt52_scores, gold_by_dialogue, target_bor=1.0)

    print("\n" + "="*60)
    print("LODO-1 Results (at BOR ≈ 1.0)")
    print("="*60)
    print(f"DistilBERT: Exact={distilbert_results['exact_f1']:.3f}, "
          f"Tolerant={distilbert_results['tolerant_f1']:.3f}, "
          f"BOR={distilbert_results['bor']:.2f}")
    print(f"GPT-5.2:    Exact={gpt52_results['exact_f1']:.3f}, "
          f"Tolerant={gpt52_results['tolerant_f1']:.3f}, "
          f"BOR={gpt52_results['bor']:.2f}")
    print(f"\nTraining time: {train_time/60:.1f} minutes")

    return {
        "experiment": "LODO-1",
        "train_data": "SuperSeg + TIAGE",
        "test_data": "DialSeg711",
        "distilbert": distilbert_results,
        "gpt52": gpt52_results,
        "training_time_min": train_time / 60,
    }


def run_lodo2():
    """LODO-2: Train on DialSeg711 + TIAGE → evaluate on SuperSeg."""
    print("\n" + "="*60)
    print("LODO-2: Train on DialSeg711 + TIAGE → Evaluate on SuperSeg")
    print("="*60)

    start_time = time.time()

    # Load training data - use DialSeg711 test as training (no train split exists)
    print("\nLoading training data...")
    dialseg_train = load_dataset("dialseg711", "test")  # All 711 as training
    tiage_train = load_dataset("tiage", "train")

    print(f"  DialSeg711 (as train): {len(dialseg_train)} dialogues")
    print(f"  TIAGE train: {len(tiage_train)} dialogues")

    # Validation data
    dialseg_val = dialseg_train[-100:]  # Use last 100 as validation
    dialseg_train = dialseg_train[:-100]  # Rest as train
    tiage_val = load_dataset("tiage", "test")[:50]

    # Convert to examples
    train_examples = (
        convert_dialogues_to_examples(dialseg_train) +
        convert_dialogues_to_examples(tiage_train)
    )
    val_examples = (
        convert_dialogues_to_examples(dialseg_val) +
        convert_dialogues_to_examples(tiage_val)
    )

    print(f"  Total training examples: {len(train_examples)}")
    print(f"  Total validation examples: {len(val_examples)}")

    # Train
    print("\nTraining...")
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")

    model_path = OUTPUT_DIR / "models" / "lodo2_dialseg_tiage.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model, best_f1 = train_model(train_examples, val_examples, model_path,
                                  start_from_pretrained=False, device=device)

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

    # Evaluate at BOR ≈ 1.0
    distilbert_results = evaluate_at_bor(distilbert_scores, gold_by_dialogue, target_bor=1.0)
    gpt52_results = evaluate_at_bor(gpt52_scores, gold_by_dialogue, target_bor=1.0)

    print("\n" + "="*60)
    print("LODO-2 Results (at BOR ≈ 1.0)")
    print("="*60)
    print(f"DistilBERT: Exact={distilbert_results['exact_f1']:.3f}, "
          f"Tolerant={distilbert_results['tolerant_f1']:.3f}, "
          f"BOR={distilbert_results['bor']:.2f}")
    print(f"GPT-5.2:    Exact={gpt52_results['exact_f1']:.3f}, "
          f"Tolerant={gpt52_results['tolerant_f1']:.3f}, "
          f"BOR={gpt52_results['bor']:.2f}")
    print(f"\nTraining time: {train_time/60:.1f} minutes")

    return {
        "experiment": "LODO-2",
        "train_data": "DialSeg711 + TIAGE",
        "test_data": "SuperSeg",
        "distilbert": distilbert_results,
        "gpt52": gpt52_results,
        "training_time_min": train_time / 60,
    }


def run_cv5():
    """5-fold CV on DialSeg711."""
    print("\n" + "="*60)
    print("5-Fold CV on DialSeg711")
    print("="*60)

    start_time = time.time()

    # Load all DialSeg711 dialogues
    print("\nLoading DialSeg711...")
    all_dialogues = load_dataset("dialseg711", "test")
    print(f"  Total dialogues: {len(all_dialogues)}")

    # Create stratification labels (based on number of boundaries)
    n_boundaries = [len(d.gold_boundaries) for d in all_dialogues]

    # Create 5-fold split
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load GPT-5.2 scores once
    gpt52_scores = load_gpt52_scores("dialseg711")

    fold_results = []
    all_distilbert_scores = {}

    for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(all_dialogues, n_boundaries)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")

        train_dialogues = [all_dialogues[i] for i in train_indices]
        test_dialogues = [all_dialogues[i] for i in test_indices]

        print(f"  Train: {len(train_dialogues)} dialogues")
        print(f"  Test: {len(test_dialogues)} dialogues")

        # Use last 10% of train as validation
        val_dialogues = train_dialogues[-len(train_dialogues)//10:]
        train_dialogues = train_dialogues[:-len(train_dialogues)//10]

        train_examples = convert_dialogues_to_examples(train_dialogues)
        val_examples = convert_dialogues_to_examples(val_dialogues)

        # Train
        model_path = OUTPUT_DIR / "models" / f"cv5_fold{fold_idx}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model, best_f1 = train_model(train_examples, val_examples, model_path,
                                      start_from_pretrained=False, device=device)

        # Score test dialogues
        fold_distilbert_scores = score_dialogues_distilbert(model, test_dialogues, tokenizer, device)

        # Store scores with original dialogue IDs
        for d, dialogue in zip(test_dialogues, test_dialogues):
            all_distilbert_scores[d.dialogue_id] = fold_distilbert_scores[d.dialogue_id]

        # Evaluate fold
        gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in test_dialogues}

        # Filter GPT-5.2 scores to test dialogues
        fold_gpt52_scores = {d.dialogue_id: gpt52_scores.get(d.dialogue_id, {})
                            for d in test_dialogues}

        distilbert_results = evaluate_at_bor(fold_distilbert_scores, gold_by_dialogue, target_bor=1.0)
        gpt52_results = evaluate_at_bor(fold_gpt52_scores, gold_by_dialogue, target_bor=1.0)

        print(f"  Fold {fold_idx + 1}: DistilBERT Exact={distilbert_results['exact_f1']:.3f}, "
              f"GPT-5.2 Exact={gpt52_results['exact_f1']:.3f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "distilbert": distilbert_results,
            "gpt52": gpt52_results,
        })

    train_time = time.time() - start_time

    # Aggregate results
    print("\n" + "="*60)
    print("5-Fold CV Aggregate Results (at BOR ≈ 1.0)")
    print("="*60)

    distilbert_exact = [r["distilbert"]["exact_f1"] for r in fold_results]
    distilbert_tol = [r["distilbert"]["tolerant_f1"] for r in fold_results]
    gpt52_exact = [r["gpt52"]["exact_f1"] for r in fold_results]
    gpt52_tol = [r["gpt52"]["tolerant_f1"] for r in fold_results]

    print(f"DistilBERT: Exact={np.mean(distilbert_exact):.3f}±{np.std(distilbert_exact):.3f}, "
          f"Tolerant={np.mean(distilbert_tol):.3f}±{np.std(distilbert_tol):.3f}")
    print(f"GPT-5.2:    Exact={np.mean(gpt52_exact):.3f}±{np.std(gpt52_exact):.3f}, "
          f"Tolerant={np.mean(gpt52_tol):.3f}±{np.std(gpt52_tol):.3f}")
    print(f"\nTotal training time: {train_time/60:.1f} minutes")

    return {
        "experiment": "5-Fold CV",
        "train_data": "DialSeg711 (4/5)",
        "test_data": "DialSeg711 (1/5)",
        "fold_results": fold_results,
        "aggregate": {
            "distilbert_exact_mean": float(np.mean(distilbert_exact)),
            "distilbert_exact_std": float(np.std(distilbert_exact)),
            "distilbert_tolerant_mean": float(np.mean(distilbert_tol)),
            "distilbert_tolerant_std": float(np.std(distilbert_tol)),
            "gpt52_exact_mean": float(np.mean(gpt52_exact)),
            "gpt52_exact_std": float(np.std(gpt52_exact)),
            "gpt52_tolerant_mean": float(np.mean(gpt52_tol)),
            "gpt52_tolerant_std": float(np.std(gpt52_tol)),
        },
        "training_time_min": train_time / 60,
    }


def get_git_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True)
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["lodo1", "lodo2", "cv5", "all"],
                       help="Which experiment to run")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("="*60)
    print("LODO + 5-Fold CV Experiments")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Git commit: {get_git_hash()}")
    print(f"Selector: greedy NMS, MIN_GAP={MIN_GAP}")
    print(f"Metrics: Exact F1 (w=0), Tolerant W-F1 (w={TOLERANT_WINDOW})")

    results = []

    if args.experiment in ["lodo1", "all"]:
        results.append(run_lodo1())

    if args.experiment in ["lodo2", "all"]:
        results.append(run_lodo2())

    if args.experiment in ["cv5", "all"]:
        results.append(run_cv5())

    # Save results
    output_file = OUTPUT_DIR / "lodo_cv_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "git_commit": get_git_hash(),
                "selector": {"type": "greedy_nms", "min_gap": MIN_GAP},
                "metrics": {"exact": "w=0", "tolerant": f"w={TOLERANT_WINDOW}"},
            },
            "results": results,
        }, f, indent=2)

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE (at BOR ≈ 1.0)")
    print("="*100)
    header = f"{'Experiment':<12} | {'Train Data':<18} | {'Test Data':<12} | {'DB Exact':>9} | {'GPT Exact':>9} | {'DB W-F1':>8} | {'GPT W-F1':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        if "fold_results" in r:  # CV results
            db_exact = r["aggregate"]["distilbert_exact_mean"]
            db_tol = r["aggregate"]["distilbert_tolerant_mean"]
            gpt_exact = r["aggregate"]["gpt52_exact_mean"]
            gpt_tol = r["aggregate"]["gpt52_tolerant_mean"]
        else:
            db_exact = r["distilbert"]["exact_f1"]
            db_tol = r["distilbert"]["tolerant_f1"]
            gpt_exact = r["gpt52"]["exact_f1"]
            gpt_tol = r["gpt52"]["tolerant_f1"]

        row = f"{r['experiment']:<12} | {r['train_data']:<18} | {r['test_data']:<12} | {db_exact:>9.3f} | {gpt_exact:>9.3f} | {db_tol:>8.3f} | {gpt_tol:>8.3f}"
        print(row)

    print("="*100)
    print(f"\nResults saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
