#!/usr/bin/env python3
"""
Train a local cross-encoder classifier for topic coverage verification.

Uses plain PyTorch training (no Trainer dependency issues).
"""

import json
import sqlite3
import random
from pathlib import Path
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"
MODEL_DIR = EXPERIMENT_DIR / "classifier_model"

# Use a small, fast model - DistilBERT base
BASE_MODEL = "distilbert-base-uncased"


@dataclass
class TrainingExample:
    query: str
    candidate: str
    label: int  # 1 = related, 0 = unrelated


def load_training_data() -> list[TrainingExample]:
    """Load training data from synthetic corpus."""
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
        candidates = case["candidates"]

        for cid in candidates:
            if cid not in statements:
                continue
            text = statements[cid]
            label = gold.get(str(cid), 0)
            examples.append(TrainingExample(query=query, candidate=text, label=label))

    return examples


def augment_training_data(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Augment with random negatives."""
    augmented = list(examples)
    all_candidates = list(set(e.candidate for e in examples))
    queries = list(set(e.query for e in examples))

    for query in queries:
        query_candidates = set(e.candidate for e in examples if e.query == query)
        available = [c for c in all_candidates if c not in query_candidates]
        if available:
            negatives = random.sample(available, min(5, len(available)))
            for neg in negatives:
                augmented.append(TrainingExample(query=query, candidate=neg, label=0))

    return augmented


class TopicDataset(Dataset):
    def __init__(self, examples: list[TrainingExample], tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.query,
            ex.candidate,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(ex.label, dtype=torch.long),
        }


def train_classifier():
    """Train the topic coverage classifier using plain PyTorch."""
    print("Loading training data...")
    examples = load_training_data()
    print(f"  Loaded {len(examples)} examples")

    examples = augment_training_data(examples)
    print(f"  After augmentation: {len(examples)} examples")

    positives = [e for e in examples if e.label == 1]
    negatives = [e for e in examples if e.label == 0]
    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Balance classes
    if len(negatives) > len(positives) * 2:
        negatives = random.sample(negatives, len(positives) * 2)
        examples = positives + negatives
        print(f"  After balancing: {len(examples)} examples")

    random.shuffle(examples)

    # Split
    train_examples, eval_examples = train_test_split(
        examples, test_size=0.2, random_state=42, stratify=[e.label for e in examples]
    )
    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Datasets
    train_dataset = TopicDataset(train_examples, tokenizer)
    eval_dataset = TopicDataset(eval_examples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
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
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

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
    """Evaluate on original test cases."""
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

    total_hard_negatives = 0
    hard_negative_accepted = 0
    total_relevant = 0
    relevant_accepted = 0

    print("\nHard negative cases:")
    for case in cases:
        query = case["query"]
        gold = case.get("gold_relevant", {})
        hard_negs = set(case.get("hard_negatives", []))
        candidates = case["candidates"][:10]

        if not hard_negs:
            continue

        accepted = []
        for cid in candidates:
            if cid not in statements:
                continue
            text = statements[cid]

            inputs = tokenizer(
                query, text, truncation=True, max_length=256,
                padding=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()

            if pred == 1:
                accepted.append(cid)
                if cid in hard_negs:
                    hard_negative_accepted += 1

        for cid in candidates:
            if cid in hard_negs:
                total_hard_negatives += 1
            if gold.get(str(cid), 0) == 1:
                total_relevant += 1
                if cid in accepted:
                    relevant_accepted += 1

        hn_in_accepted = len([c for c in accepted if c in hard_negs])
        status = "✓ PASS" if hn_in_accepted == 0 else f"✗ FAIL ({hn_in_accepted})"
        print(f"  {query:20s} {status}")

    print()
    print(f"Hard negative FPR: {hard_negative_accepted}/{total_hard_negatives} = {hard_negative_accepted/total_hard_negatives:.1%}" if total_hard_negatives > 0 else "No hard negatives")


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
        model, tokenizer = train_classifier()
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
        print("Usage: python train_classifier.py --train [--eval] [--benchmark]")
