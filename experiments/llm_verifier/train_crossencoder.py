#!/usr/bin/env python3
"""
Train a CrossEncoder for topic coverage verification.

Uses plain PyTorch training with a cross-encoder architecture
(sequence classification on [query, candidate] pairs).
"""

import json
import sqlite3
import random
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"
MODEL_DIR = EXPERIMENT_DIR / "crossencoder_model"

# MS MARCO cross-encoder - pre-trained for relevance matching
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_training_data():
    """Load training data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM statements")
    statements = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    examples = []
    for case in cases:
        query = case["query"]
        gold = case.get("gold_relevant", {})
        hard_negs = set(case.get("hard_negatives", []))
        candidates = case["candidates"]

        for cid in candidates:
            if cid not in statements:
                continue
            text = statements[cid]
            label = gold.get(str(cid), 0)

            # Weight hard negatives more heavily in training
            weight = 3.0 if cid in hard_negs else 1.0
            examples.append((query, text, label, weight))

    return examples, statements


def augment_data(examples):
    """Augment with additional hard negatives."""
    augmented = list(examples)
    all_texts = list(set(e[1] for e in examples))
    queries = list(set(e[0] for e in examples))

    for query in queries:
        query_texts = set(e[1] for e in examples if e[0] == query)
        available = [t for t in all_texts if t not in query_texts]
        if available:
            negatives = random.sample(available, min(10, len(available)))
            for neg in negatives:
                augmented.append((query, neg, 0, 1.0))

    return augmented


class PairDataset(Dataset):
    """Dataset for query-candidate pairs."""
    def __init__(self, pairs, labels, tokenizer, max_length=256):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, candidate = self.pairs[idx]
        encoding = self.tokenizer(
            query, candidate,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def train_crossencoder():
    """Train the CrossEncoder using plain PyTorch."""
    print("Loading training data...")
    examples, statements = load_training_data()
    print(f"  Loaded {len(examples)} examples")

    examples = augment_data(examples)
    print(f"  After augmentation: {len(examples)} examples")

    positives = [e for e in examples if e[2] == 1]
    negatives = [e for e in examples if e[2] == 0]
    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Balance
    if len(negatives) > len(positives) * 3:
        negatives = random.sample(negatives, len(positives) * 3)
        examples = positives + negatives
        print(f"  After balancing: {len(examples)} examples")

    random.shuffle(examples)

    # Prepare for training
    train_pairs = [[e[0], e[1]] for e in examples]
    train_labels = [float(e[2]) for e in examples]

    # Split
    train_pairs, eval_pairs, train_y, eval_y = train_test_split(
        train_pairs, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    print(f"  Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Load model and tokenizer
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Datasets
    train_dataset = PairDataset(train_pairs, train_y, tokenizer)
    eval_dataset = PairDataset(eval_pairs, eval_y, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    print("\nTraining...")
    num_epochs = 5
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy().astype(int))

        # Metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))

    print(f"\nBest F1: {best_f1:.3f}")
    print(f"Model saved to {MODEL_DIR}")

    return model, tokenizer


def evaluate_on_test_cases(model, tokenizer):
    """Evaluate on test cases."""
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST CASES")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM statements")
    statements = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    def predict_score(query, text):
        """Get relevance score for a query-text pair."""
        inputs = tokenizer(
            query, text, truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits.squeeze(-1)).item()
        return score

    # Find best threshold
    print("\nFinding optimal threshold...")
    all_scores = []
    all_labels = []

    for case in cases:
        query = case["query"]
        gold = case.get("gold_relevant", {})
        candidates = case["candidates"][:20]

        for cid in candidates:
            if cid not in statements:
                continue
            text = statements[cid]
            score = predict_score(query, text)
            label = gold.get(str(cid), 0)
            all_scores.append(score)
            all_labels.append(label)

    # Try different thresholds
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = [1 if s > thresh else 0 for s in all_scores]
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"  Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

    # Evaluate with best threshold
    total_hard_negatives = 0
    hard_negative_accepted = 0

    print("\nHard negative cases:")
    for case in cases:
        query = case["query"]
        gold = case.get("gold_relevant", {})
        hard_negs = set(case.get("hard_negatives", []))
        candidates = case["candidates"][:10]

        if not hard_negs:
            continue

        for cid in candidates:
            if cid not in statements:
                continue
            if cid not in hard_negs:
                continue

            total_hard_negatives += 1
            text = statements[cid]
            score = predict_score(query, text)

            if score > best_threshold:
                hard_negative_accepted += 1

        # Count for display
        hn_accepted = 0
        for cid in candidates:
            if cid in hard_negs and cid in statements:
                score = predict_score(query, statements[cid])
                if score > best_threshold:
                    hn_accepted += 1

        status = "✓ PASS" if hn_accepted == 0 else f"✗ FAIL ({hn_accepted})"
        print(f"  {query:20s} {status}")

    print()
    if total_hard_negatives > 0:
        print(f"Hard negative FPR: {hard_negative_accepted}/{total_hard_negatives} = {hard_negative_accepted/total_hard_negatives:.1%}")


def benchmark_latency(model, tokenizer, n_samples=100):
    """Benchmark inference latency."""
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM statements LIMIT 10")
    texts = [row[0] for row in cursor.fetchall()]
    conn.close()

    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    query = "python programming"

    # Warmup
    for text in texts[:3]:
        inputs = tokenizer(query, text, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    # Single inference
    times = []
    for _ in range(n_samples):
        text = random.choice(texts)
        inputs = tokenizer(query, text, truncation=True, max_length=256, return_tensors="pt").to(device)

        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"\nSingle inference (n={n_samples}):")
    print(f"  Mean: {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  P95: {np.percentile(times, 95):.1f} ms")

    # Batch inference
    batch_times = []
    for _ in range(n_samples // 5):
        batch_texts = random.sample(texts, min(5, len(texts)))
        inputs = tokenizer(
            [query] * len(batch_texts), batch_texts,
            truncation=True, max_length=256, padding=True, return_tensors="pt"
        ).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        end = time.perf_counter()
        batch_times.append((end - start) * 1000)

    print(f"\nBatch inference (5 candidates, n={len(batch_times)}):")
    print(f"  Mean: {np.mean(batch_times):.1f} ms")
    print(f"  Per candidate: {np.mean(batch_times)/5:.1f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.train:
        model, tokenizer = train_crossencoder()
        if args.eval:
            evaluate_on_test_cases(model, tokenizer)
        if args.benchmark:
            benchmark_latency(model, tokenizer)
    elif args.eval or args.benchmark:
        print(f"Loading model from {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        if args.eval:
            evaluate_on_test_cases(model, tokenizer)
        if args.benchmark:
            benchmark_latency(model, tokenizer)
    else:
        print("Usage: python train_crossencoder.py --train [--eval] [--benchmark]")
