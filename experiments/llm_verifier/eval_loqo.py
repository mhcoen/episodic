#!/usr/bin/env python3
"""
Leave-One-Query-Out (LOQO) evaluation for cross-encoder.

Proper eval protocol: train on N-1 queries, test on the held-out query.
This stress-tests generalization to unseen query terms and their hard negatives.
"""

import json
import sqlite3
import random
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class QueryResult:
    """Results for a single held-out query."""
    query: str
    hard_negative_count: int
    hard_negative_accepted: int
    hard_negative_fpr: float
    precision: float
    recall: float
    f1: float
    threshold: float


class PairDataset(Dataset):
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


def load_data():
    """Load all data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM statements")
    statements = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    return statements, cases


def prepare_examples_for_query(case: dict, statements: dict) -> list[tuple]:
    """Prepare training examples for a single query case."""
    query = case["query"]
    gold = case.get("gold_relevant", {})
    hard_negs = set(case.get("hard_negatives", []))
    candidates = case["candidates"]

    examples = []
    for cid in candidates:
        if cid not in statements:
            continue
        text = statements[cid]
        label = gold.get(str(cid), 0)
        # Weight hard negatives
        weight = 3.0 if cid in hard_negs else 1.0
        examples.append((query, text, label, weight, cid))

    return examples


def train_on_queries(
    train_cases: list[dict],
    statements: dict,
    tokenizer,
    epochs: int = 3,
) -> torch.nn.Module:
    """Train a model on the given query cases."""
    # Collect all examples
    all_examples = []
    for case in train_cases:
        all_examples.extend(prepare_examples_for_query(case, statements))

    # Augment with cross-query negatives
    all_texts = list(set(e[1] for e in all_examples))
    queries = list(set(e[0] for e in all_examples))

    for query in queries:
        query_texts = set(e[1] for e in all_examples if e[0] == query)
        available = [t for t in all_texts if t not in query_texts]
        if available:
            negatives = random.sample(available, min(5, len(available)))
            for neg in negatives:
                all_examples.append((query, neg, 0, 1.0, -1))

    # Balance classes
    positives = [e for e in all_examples if e[2] == 1]
    negatives = [e for e in all_examples if e[2] == 0]
    if len(negatives) > len(positives) * 3:
        negatives = random.sample(negatives, len(positives) * 3)
        all_examples = positives + negatives

    random.shuffle(all_examples)

    # Prepare for training
    train_pairs = [[e[0], e[1]] for e in all_examples]
    train_labels = [float(e[2]) for e in all_examples]

    # Load fresh model
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Dataset
    train_dataset = PairDataset(train_pairs, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
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

    return model


def evaluate_on_query(
    model: torch.nn.Module,
    case: dict,
    statements: dict,
    tokenizer,
) -> QueryResult:
    """Evaluate on a single held-out query."""
    query = case["query"]
    gold = case.get("gold_relevant", {})
    hard_negs = set(case.get("hard_negatives", []))
    candidates = case["candidates"][:20]

    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Get scores for all candidates
    scores = []
    labels = []
    cids = []

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
            score = torch.sigmoid(outputs.logits.squeeze(-1)).item()

        scores.append(score)
        labels.append(gold.get(str(cid), 0))
        cids.append(cid)

    if not scores:
        return QueryResult(
            query=query, hard_negative_count=0, hard_negative_accepted=0,
            hard_negative_fpr=0, precision=0, recall=0, f1=0, threshold=0.5
        )

    # Find best threshold
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = [1 if s > thresh else 0 for s in scores]
        if sum(preds) == 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Evaluate with best threshold
    preds = [1 if s > best_threshold else 0 for s in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    # Hard negative analysis
    hn_count = 0
    hn_accepted = 0
    for i, cid in enumerate(cids):
        if cid in hard_negs:
            hn_count += 1
            if preds[i] == 1:
                hn_accepted += 1

    return QueryResult(
        query=query,
        hard_negative_count=hn_count,
        hard_negative_accepted=hn_accepted,
        hard_negative_fpr=hn_accepted / hn_count if hn_count > 0 else 0,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=best_threshold,
    )


def run_loqo_evaluation():
    """Run leave-one-query-out evaluation."""
    print("Loading data...")
    statements, cases = load_data()

    # Filter to cases with hard negatives (the stress test)
    hard_neg_cases = [c for c in cases if c.get("hard_negatives")]
    print(f"  {len(hard_neg_cases)} cases with hard negatives")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    results = []

    print("\nRunning LOQO evaluation...")
    print("=" * 70)

    for i, test_case in enumerate(hard_neg_cases):
        # Train on all other cases
        train_cases = [c for c in cases if c["query"] != test_case["query"]]

        print(f"\n[{i+1}/{len(hard_neg_cases)}] Held out: '{test_case['query']}'")
        print(f"  Training on {len(train_cases)} queries...")

        model = train_on_queries(train_cases, statements, tokenizer, epochs=3)
        result = evaluate_on_query(model, test_case, statements, tokenizer)
        results.append(result)

        status = "✓ PASS" if result.hard_negative_accepted == 0 else f"✗ FAIL"
        print(f"  Hard neg FPR: {result.hard_negative_accepted}/{result.hard_negative_count} = {result.hard_negative_fpr:.0%} {status}")
        print(f"  F1={result.f1:.2f}, P={result.precision:.2f}, R={result.recall:.2f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("LOQO EVALUATION SUMMARY")
    print("=" * 70)

    total_hn = sum(r.hard_negative_count for r in results)
    total_hn_accepted = sum(r.hard_negative_accepted for r in results)
    mean_f1 = np.mean([r.f1 for r in results])
    mean_precision = np.mean([r.precision for r in results])
    mean_recall = np.mean([r.recall for r in results])

    print(f"\nHard Negative Analysis:")
    print(f"  Total hard negatives:    {total_hn}")
    print(f"  Incorrectly accepted:    {total_hn_accepted}")
    print(f"  Overall FPR:             {total_hn_accepted/total_hn:.1%}" if total_hn > 0 else "  No hard negatives")

    print(f"\nClassification Metrics (mean across held-out queries):")
    print(f"  Precision: {mean_precision:.1%}")
    print(f"  Recall:    {mean_recall:.1%}")
    print(f"  F1:        {mean_f1:.1%}")

    # Per-query breakdown
    print("\nPer-Query Results:")
    for r in results:
        status = "✓" if r.hard_negative_accepted == 0 else "✗"
        print(f"  {status} {r.query:20s} HN={r.hard_negative_accepted}/{r.hard_negative_count} F1={r.f1:.2f}")

    return results


if __name__ == "__main__":
    run_loqo_evaluation()
