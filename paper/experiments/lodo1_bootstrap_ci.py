#!/usr/bin/env python3
"""
Bootstrap CIs for LODO-1 Scorer Difference

Computes procedure-consistent bootstrap confidence intervals for the
difference between GPT-5.2 and DistilBERT on DialSeg711.

Usage:
    python paper/experiments/lodo1_bootstrap_ci.py
    python paper/experiments/lodo1_bootstrap_ci.py --n-bootstrap 200  # faster
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"
GPT52_CACHE = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"
LODO1_MODEL = PROJECT_ROOT / "paper" / "experiments" / "results" / "models" / "lodo1_superseg_tiage.pt"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments" / "results"

# Config
MIN_GAP = 2
TOLERANT_WINDOW = 3
SEED = 42


@dataclass
class DialogueData:
    dialogue_id: int
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_user_turns: int


def load_dataset() -> List[DialogueData]:
    """Load DialSeg711 dataset."""
    file_path = DATASETS_DIR / "dialseg711" / "segmentation_file_test.json"

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


def load_gpt52_scores() -> Dict[int, Dict[int, float]]:
    """Load GPT-5.2 scores from cache."""
    with open(GPT52_CACHE) as f:
        cache = json.load(f)

    scores_by_dialogue = defaultdict(dict)
    for key, entry in cache.items():
        if entry.get("invalid_first_token") or entry.get("missing_yn_in_toplogprobs"):
            continue
        dialogue_id = entry.get("dialogue_id")
        position = entry.get("position")
        score = entry.get("score", 0)
        if dialogue_id is not None and position is not None:
            scores_by_dialogue[dialogue_id][position] = score

    return dict(scores_by_dialogue)


def load_distilbert_scores(dialogues: List[DialogueData]) -> Dict[int, Dict[int, float]]:
    """Score dialogues with LODO-1 DistilBERT model."""
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    checkpoint = torch.load(LODO1_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scores_by_dialogue = {}

    with torch.no_grad():
        for dialogue in dialogues:
            user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]
            dialogue_scores = {}

            for position in range(1, len(user_turns)):
                # Build context
                context_before = user_turns[max(0, position-4):position]
                context_after = user_turns[position]

                window_text = " ".join([f"user: {t}" for t in context_before])
                current_text = context_after

                # Tokenize
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encoding = tokenizer(
                        window_text,
                        current_text,
                        truncation=True,
                        max_length=256,
                        padding="max_length",
                        return_tensors="pt",
                        verbose=False
                    )

                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
                dialogue_scores[position] = prob

            scores_by_dialogue[dialogue.dialogue_id] = dialogue_scores

    return scores_by_dialogue


def greedy_nms(scores: Dict[int, float], tau: float, min_gap: int = 2) -> Set[int]:
    """Apply greedy NMS to select boundaries."""
    candidates = [(idx, score) for idx, score in scores.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    selected = set()
    for idx, score in candidates:
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


def compute_tolerant_f1(pred: Set[int], gold: Set[int], window: int = 3) -> float:
    """Compute tolerant (windowed) F1 with one-to-one matching."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    pred_list = sorted(pred)
    gold_list = sorted(gold)

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


def find_tau_for_bor(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    dialogue_ids: List[int],
    target_bor: float = 1.0
) -> float:
    """Find tau that gives BOR closest to target on given dialogues."""
    # Collect all scores from selected dialogues
    all_scores = []
    for did in dialogue_ids:
        if did in scores_by_dialogue:
            all_scores.extend(scores_by_dialogue[did].values())

    if not all_scores:
        return 0.5

    best_tau = None
    best_bor_diff = float("inf")

    for pct in range(0, 100, 2):
        tau = np.percentile(all_scores, pct)

        total_pred = 0
        total_gold = 0

        for did in dialogue_ids:
            if did not in scores_by_dialogue:
                continue
            gold = gold_by_dialogue.get(did, set())
            pred = greedy_nms(scores_by_dialogue[did], tau, MIN_GAP)
            total_pred += len(pred)
            total_gold += len(gold)

        if total_gold == 0:
            continue

        bor = total_pred / total_gold
        bor_diff = abs(bor - target_bor)

        if bor_diff < best_bor_diff:
            best_bor_diff = bor_diff
            best_tau = tau

    return best_tau if best_tau is not None else 0.5


def evaluate_on_dialogues(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    dialogue_ids: List[int],
    tau: float
) -> Tuple[float, float, float]:
    """Evaluate on given dialogues with given tau. Returns (exact_f1, tolerant_f1, bor)."""
    exact_f1s = []
    tolerant_f1s = []
    total_pred = 0
    total_gold = 0

    for did in dialogue_ids:
        if did not in scores_by_dialogue:
            continue
        gold = gold_by_dialogue.get(did, set())
        if not gold:
            continue

        pred = greedy_nms(scores_by_dialogue[did], tau, MIN_GAP)

        exact_f1s.append(compute_exact_f1(pred, gold))
        tolerant_f1s.append(compute_tolerant_f1(pred, gold, TOLERANT_WINDOW))

        total_pred += len(pred)
        total_gold += len(gold)

    bor = total_pred / total_gold if total_gold > 0 else 0

    return (
        np.mean(exact_f1s) if exact_f1s else 0,
        np.mean(tolerant_f1s) if tolerant_f1s else 0,
        bor
    )


def run_bootstrap(
    distilbert_scores: Dict[int, Dict[int, float]],
    gpt52_scores: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    dialogue_ids: List[int],
    n_bootstrap: int = 1000,
    seed: int = SEED
) -> Dict:
    """Run bootstrap procedure."""
    rng = np.random.default_rng(seed)

    delta_exact = []
    delta_tolerant = []
    db_bors = []
    gpt_bors = []

    n_dialogues = len(dialogue_ids)

    for b in range(n_bootstrap):
        # Resample dialogues with replacement
        sampled_ids = rng.choice(dialogue_ids, size=n_dialogues, replace=True).tolist()

        # Find tau for each scorer on resampled set
        db_tau = find_tau_for_bor(distilbert_scores, gold_by_dialogue, sampled_ids, target_bor=1.0)
        gpt_tau = find_tau_for_bor(gpt52_scores, gold_by_dialogue, sampled_ids, target_bor=1.0)

        # Evaluate each scorer
        db_exact, db_tol, db_bor = evaluate_on_dialogues(
            distilbert_scores, gold_by_dialogue, sampled_ids, db_tau
        )
        gpt_exact, gpt_tol, gpt_bor = evaluate_on_dialogues(
            gpt52_scores, gold_by_dialogue, sampled_ids, gpt_tau
        )

        # Record deltas (GPT-5.2 - DistilBERT)
        delta_exact.append(gpt_exact - db_exact)
        delta_tolerant.append(gpt_tol - db_tol)
        db_bors.append(db_bor)
        gpt_bors.append(gpt_bor)

        if (b + 1) % 100 == 0:
            print(f"  Bootstrap: {b+1}/{n_bootstrap}")

    return {
        "delta_exact": delta_exact,
        "delta_tolerant": delta_tolerant,
        "db_bors": db_bors,
        "gpt_bors": gpt_bors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap replicates")
    args = parser.parse_args()

    print("="*60)
    print("LODO-1 Bootstrap CIs for Scorer Difference")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Bootstrap replicates: {args.n_bootstrap}")

    # Load data
    print("\nLoading data...")
    dialogues = load_dataset()
    gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in dialogues}
    dialogue_ids = [d.dialogue_id for d in dialogues]
    print(f"  Dialogues: {len(dialogues)}")

    # Load GPT-5.2 scores
    print("  Loading GPT-5.2 scores...")
    gpt52_scores = load_gpt52_scores()
    print(f"  GPT-5.2 dialogues: {len(gpt52_scores)}")

    # Load/compute DistilBERT scores
    print("  Scoring with DistilBERT (LODO-1 model)...")
    distilbert_scores = load_distilbert_scores(dialogues)
    print(f"  DistilBERT dialogues: {len(distilbert_scores)}")

    # Find common dialogues
    common_ids = list(set(dialogue_ids) & set(gpt52_scores.keys()) & set(distilbert_scores.keys()))
    print(f"  Common dialogues: {len(common_ids)}")

    # Point estimates first
    print("\nComputing point estimates...")
    db_tau = find_tau_for_bor(distilbert_scores, gold_by_dialogue, common_ids, target_bor=1.0)
    gpt_tau = find_tau_for_bor(gpt52_scores, gold_by_dialogue, common_ids, target_bor=1.0)

    db_exact, db_tol, db_bor = evaluate_on_dialogues(distilbert_scores, gold_by_dialogue, common_ids, db_tau)
    gpt_exact, gpt_tol, gpt_bor = evaluate_on_dialogues(gpt52_scores, gold_by_dialogue, common_ids, gpt_tau)

    print(f"  DistilBERT: Exact={db_exact:.3f}, Tolerant={db_tol:.3f}, BOR={db_bor:.2f}")
    print(f"  GPT-5.2:    Exact={gpt_exact:.3f}, Tolerant={gpt_tol:.3f}, BOR={gpt_bor:.2f}")
    print(f"  Δ_exact:    {gpt_exact - db_exact:+.3f}")
    print(f"  Δ_tolerant: {gpt_tol - db_tol:+.3f}")

    # Run bootstrap
    print(f"\nRunning bootstrap ({args.n_bootstrap} replicates)...")
    bootstrap_results = run_bootstrap(
        distilbert_scores, gpt52_scores, gold_by_dialogue,
        common_ids, n_bootstrap=args.n_bootstrap
    )

    # Compute CIs
    delta_exact = np.array(bootstrap_results["delta_exact"])
    delta_tolerant = np.array(bootstrap_results["delta_tolerant"])
    db_bors = np.array(bootstrap_results["db_bors"])
    gpt_bors = np.array(bootstrap_results["gpt_bors"])

    ci_exact = (np.percentile(delta_exact, 2.5), np.percentile(delta_exact, 97.5))
    ci_tolerant = (np.percentile(delta_tolerant, 2.5), np.percentile(delta_tolerant, 97.5))

    # Print results
    print("\n" + "="*60)
    print(f"LODO-1 Bootstrap CIs (B={args.n_bootstrap}, N={len(common_ids)} dialogues)")
    print("="*60)
    print(f"Δ_exact:    {gpt_exact - db_exact:+.3f} (95% CI: [{ci_exact[0]:.3f}, {ci_exact[1]:.3f}])")
    print(f"Δ_tolerant: {gpt_tol - db_tol:+.3f} (95% CI: [{ci_tolerant[0]:.3f}, {ci_tolerant[1]:.3f}])")
    print()
    print("BOR distribution at τ*:")
    print(f"  DistilBERT: mean={np.mean(db_bors):.2f}, std={np.std(db_bors):.2f}, "
          f"range=[{np.min(db_bors):.2f}, {np.max(db_bors):.2f}]")
    print(f"  GPT-5.2:    mean={np.mean(gpt_bors):.2f}, std={np.std(gpt_bors):.2f}, "
          f"range=[{np.min(gpt_bors):.2f}, {np.max(gpt_bors):.2f}]")

    # Save results
    results = {
        "experiment": "lodo1_bootstrap_ci",
        "n_bootstrap": args.n_bootstrap,
        "n_dialogues": len(common_ids),
        "seed": SEED,
        "point_estimates": {
            "distilbert": {"exact_f1": db_exact, "tolerant_f1": db_tol, "bor": db_bor},
            "gpt52": {"exact_f1": gpt_exact, "tolerant_f1": gpt_tol, "bor": gpt_bor},
            "delta_exact": gpt_exact - db_exact,
            "delta_tolerant": gpt_tol - db_tol,
        },
        "bootstrap_ci": {
            "delta_exact_mean": float(np.mean(delta_exact)),
            "delta_exact_std": float(np.std(delta_exact)),
            "delta_exact_ci_95": [float(ci_exact[0]), float(ci_exact[1])],
            "delta_tolerant_mean": float(np.mean(delta_tolerant)),
            "delta_tolerant_std": float(np.std(delta_tolerant)),
            "delta_tolerant_ci_95": [float(ci_tolerant[0]), float(ci_tolerant[1])],
        },
        "bor_distribution": {
            "distilbert": {
                "mean": float(np.mean(db_bors)),
                "std": float(np.std(db_bors)),
                "min": float(np.min(db_bors)),
                "max": float(np.max(db_bors)),
            },
            "gpt52": {
                "mean": float(np.mean(gpt_bors)),
                "std": float(np.std(gpt_bors)),
                "min": float(np.min(gpt_bors)),
                "max": float(np.max(gpt_bors)),
            },
        },
        "config": {
            "min_gap": MIN_GAP,
            "tolerant_window": TOLERANT_WINDOW,
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_path = OUTPUT_DIR / "lodo1_bootstrap_ci.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
