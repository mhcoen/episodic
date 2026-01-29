#!/usr/bin/env python3
"""
Verification script for adaptive window experiment results.

Reproduces table and adds additional diagnostics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
GPT52_DIALSEG_CACHE = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"
GPT52_SUPERSEG_CACHE = PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json"

MIN_GAP = 2


def load_dataset_gold(dataset_name: str) -> Dict[int, Set[int]]:
    """Load gold boundaries for a dataset."""
    if dataset_name == "dialseg711":
        path = DATASETS_DIR / "dialseg711" / "segmentation_file_test.json"
    else:
        path = DATASETS_DIR / "superseg" / "segmentation_file_test.json"

    with open(path) as f:
        data = json.load(f)

    gold_by_dialogue = {}
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

            gold_by_dialogue[dialogue_id] = boundaries
            dialogue_id += 1

    return gold_by_dialogue


def load_gpt52_scores(dataset_name: str) -> Dict[int, Dict[int, float]]:
    """Load GPT-5.2 scores from cache."""
    if dataset_name == "dialseg711":
        cache_path = GPT52_DIALSEG_CACHE
        prefix = "dialseg711_"
    else:
        cache_path = GPT52_SUPERSEG_CACHE
        prefix = "superseg_"

    with open(cache_path) as f:
        cache = json.load(f)

    scores_by_dialogue = defaultdict(dict)
    for key, entry in cache.items():
        if key.startswith(prefix):
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]
            score = entry["score"]
            if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                scores_by_dialogue[dialogue_id][position] = score

    return dict(scores_by_dialogue)


def load_distilbert_scores(dataset_name: str) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Set[int]]]:
    """Load DistilBERT scores by running the model."""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    if dataset_name == "dialseg711":
        path = DATASETS_DIR / "dialseg711" / "segmentation_file_test.json"
    else:
        path = DATASETS_DIR / "superseg" / "segmentation_file_test.json"

    with open(path) as f:
        data = json.load(f)

    model_path = PROJECT_ROOT / "paper" / "experiments" / "models" / "final_calibrated.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    scores_by_dialogue = {}
    gold_by_dialogue = {}

    dial_data = data.get("dial_data", data)
    dialogue_id = 0

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in tqdm(source_dialogs, desc=f"  Scoring {dataset_name}", leave=False):
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            messages = [
                {"role": t["role"], "content": t.get("utterance", t.get("text", ""))}
                for t in turns
            ]

            # Extract gold
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
            gold_by_dialogue[dialogue_id] = boundaries

            # Score each user turn position
            scores = {}
            user_idx = 0

            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    if user_idx > 0:
                        window_start = max(0, i - 8)
                        window = messages[window_start:i]
                        pre_text = " ".join([m["content"] for m in window])
                        post_text = msg["content"]
                        text = f"{pre_text} [SEP] {post_text}"

                        inputs = tokenizer(
                            text, truncation=True, max_length=512,
                            return_tensors="pt"
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits / temperature
                            probs = torch.softmax(logits, dim=-1)
                            score = probs[0, 1].item()

                        scores[user_idx] = score

                    user_idx += 1

            scores_by_dialogue[dialogue_id] = scores
            dialogue_id += 1

    return scores_by_dialogue, gold_by_dialogue


def greedy_nms(scores_by_pos: Dict[int, float], tau: float, min_gap: int = MIN_GAP) -> Set[int]:
    """Apply greedy NMS to select boundaries."""
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < min_gap for p in predicted):
            predicted.add(pos)

    return predicted


def compute_offsets(predictions: Set[int], gold: Set[int]) -> List[int]:
    """Compute absolute offset from each prediction to nearest gold WITHIN dialogue."""
    if not predictions or not gold:
        return []

    offsets = []
    for p in predictions:
        min_dist = min(abs(p - g) for g in gold)
        offsets.append(min_dist)

    return offsets


def compute_wf1_with_details(predicted: Set[int], gold: Set[int], w: int) -> Tuple[float, float, float, int, int]:
    """
    Compute W-F1 with window w (many-to-one matching).
    Returns: (wf1, precision, recall, tp, matched_gold)
    """
    if not gold:
        return (0.0, 0.0, 0.0, 0, 0) if predicted else (1.0, 1.0, 1.0, 0, 0)

    # Count TPs: predictions within w of any gold
    tp = sum(1 for p in predicted if any(abs(p - g) <= w for g in gold))

    # Count matched gold
    matched_gold = sum(1 for g in gold if any(abs(p - g) <= w for p in predicted))

    fp = len(predicted) - tp
    fn = len(gold) - matched_gold

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = matched_gold / len(gold) if gold else 0.0

    wf1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return wf1, precision, recall, tp, matched_gold


def find_tau_for_bor(scores_by_dialogue: Dict[int, Dict[int, float]],
                     gold_by_dialogue: Dict[int, Set[int]],
                     target_bor: float) -> Tuple[float, float]:
    """Find tau that achieves closest BOR to target."""
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    total_gold = sum(len(g) for g in gold_by_dialogue.values())

    best_tau = None
    best_bor = None
    best_diff = float('inf')

    for pct in range(1, 100):
        tau = np.percentile(all_scores, pct)

        total_pred = 0
        for did, scores in scores_by_dialogue.items():
            pred = greedy_nms(scores, tau)
            total_pred += len(pred)

        bor = total_pred / total_gold if total_gold > 0 else 0
        diff = abs(bor - target_bor)

        if diff < best_diff:
            best_diff = diff
            best_tau = tau
            best_bor = bor

    return best_tau, best_bor


def run_verification(dataset_name: str, scorer: str):
    """Run verification for one dataset/scorer combination."""
    print(f"\n{'='*60}")
    print(f"Verifying: {scorer} on {dataset_name}")
    print(f"{'='*60}")

    # Load data
    if scorer == "GPT-5.2":
        scores_by_dialogue = load_gpt52_scores(dataset_name)
        gold_by_dialogue = load_dataset_gold(dataset_name)
    else:
        scores_by_dialogue, gold_by_dialogue = load_distilbert_scores(dataset_name)

    total_gold = sum(len(g) for g in gold_by_dialogue.values())
    print(f"Dialogues with scores: {len(scores_by_dialogue)}")
    print(f"Total gold boundaries: {total_gold}")

    # Find tau for BOR≈1.0
    tau, actual_bor = find_tau_for_bor(scores_by_dialogue, gold_by_dialogue, 1.0)
    print(f"\nτ* selection: searched percentiles 1-99, picked closest to BOR=1.0")
    print(f"τ* = {tau:.6f}, actual BOR = {actual_bor:.4f}")

    # Compute predictions at tau*
    all_predictions = {}
    all_gold = {}
    all_offsets = []
    total_pred = 0

    for did, scores in scores_by_dialogue.items():
        pred = greedy_nms(scores, tau)
        gold = gold_by_dialogue.get(did, set())

        all_predictions[did] = pred
        all_gold[did] = gold
        total_pred += len(pred)

        offsets = compute_offsets(pred, gold)
        all_offsets.extend(offsets)

    all_offsets = np.array(all_offsets)
    print(f"Total predictions: {total_pred}")
    print(f"Total offsets computed: {len(all_offsets)}")

    # Compute quantiles
    w_50 = int(np.ceil(np.median(all_offsets))) if len(all_offsets) > 0 else 0
    w_90 = int(np.ceil(np.percentile(all_offsets, 90))) if len(all_offsets) > 0 else 0
    w_95 = int(np.ceil(np.percentile(all_offsets, 95))) if len(all_offsets) > 0 else 0

    print(f"\nOffset quantiles:")
    print(f"  median (w_50): {np.median(all_offsets):.2f} → ceil = {w_50}")
    print(f"  90th (w_90): {np.percentile(all_offsets, 90):.2f} → ceil = {w_90}")
    print(f"  95th (w_95): {np.percentile(all_offsets, 95):.2f} → ceil = {w_95}")

    # Histogram of offsets
    print(f"\nOffset histogram:")
    for bucket in [0, 1, 2, 3]:
        count = np.sum(all_offsets == bucket)
        pct = 100 * count / len(all_offsets)
        print(f"  offset={bucket}: {count} ({pct:.1f}%)")
    count_gt3 = np.sum(all_offsets > 3)
    pct_gt3 = 100 * count_gt3 / len(all_offsets)
    print(f"  offset>3: {count_gt3} ({pct_gt3:.1f}%)")

    # Compute W-F1 at different windows
    results = {}
    for w in [0, 1, 3]:
        total_wf1 = 0.0
        total_prec = 0.0
        total_rec = 0.0
        n = 0

        for did, pred in all_predictions.items():
            gold = all_gold.get(did, set())
            if gold:
                wf1, prec, rec, tp, mg = compute_wf1_with_details(pred, gold, w)
                total_wf1 += wf1
                total_prec += prec
                total_rec += rec
                n += 1

        avg_wf1 = total_wf1 / n if n > 0 else 0.0
        avg_prec = total_prec / n if n > 0 else 0.0
        avg_rec = total_rec / n if n > 0 else 0.0
        results[w] = {"wf1": avg_wf1, "prec": avg_prec, "rec": avg_rec}
        print(f"\nW-F1(w={w}): {avg_wf1:.4f}  (Prec={avg_prec:.4f}, Rec={avg_rec:.4f})")

    # Compute unmatched fraction
    unmatched_w1 = np.sum(all_offsets > 1) / len(all_offsets)
    unmatched_w3 = np.sum(all_offsets > 3) / len(all_offsets)
    print(f"\nUnmatched predictions:")
    print(f"  within ±1: {100*(1-unmatched_w1):.1f}% matched, {100*unmatched_w1:.1f}% unmatched")
    print(f"  within ±3: {100*(1-unmatched_w3):.1f}% matched, {100*unmatched_w3:.1f}% unmatched")

    # Predictions matching at distance 2-3 (explains w=1 to w=3 jump)
    match_2_3 = np.sum((all_offsets >= 2) & (all_offsets <= 3))
    print(f"\nPredictions matching at distance 2-3: {match_2_3} ({100*match_2_3/len(all_offsets):.1f}%)")

    gap = results[3]["wf1"] - results[1]["wf1"]
    print(f"\nGap W-F1(w=3) - W-F1(w=1): {gap:.4f}")

    return {
        "dataset": dataset_name,
        "scorer": scorer,
        "tau": tau,
        "bor": actual_bor,
        "w_50": w_50,
        "w_90": w_90,
        "w_95": w_95,
        "wf1_w0": results[0]["wf1"],
        "wf1_w1": results[1]["wf1"],
        "wf1_w3": results[3]["wf1"],
        "prec_w1": results[1]["prec"],
        "rec_w1": results[1]["rec"],
        "prec_w3": results[3]["prec"],
        "rec_w3": results[3]["rec"],
        "gap": gap,
        "offset_histogram": {
            0: int(np.sum(all_offsets == 0)),
            1: int(np.sum(all_offsets == 1)),
            2: int(np.sum(all_offsets == 2)),
            3: int(np.sum(all_offsets == 3)),
            ">3": int(np.sum(all_offsets > 3)),
        },
        "match_2_3": int(match_2_3),
        "n_offsets": len(all_offsets),
    }


def main():
    print("=" * 70)
    print("VERIFICATION REPORT: Adaptive Window Experiment")
    print("=" * 70)

    print("\n### 1. Code Location")
    print("- Script: paper/experiments/adaptive_window_experiment.py")
    print("- Selector: greedy NMS, MIN_GAP=2 (confirmed in code line 49, 165-175)")
    print("- τ* selection: search percentiles 1-99, pick closest BOR to target")
    print("- W-F1 variant: many-to-one matching (confirmed in compute_wf1, lines 192-209)")
    print("- Boundary indexing: user turn index, gold extracted from topic_id changes")
    print("- Nearest-gold: computed WITHIN each dialogue (confirmed in compute_offsets, lines 178-189)")

    all_results = []

    # Run verification for all 4 conditions
    for dataset in ["dialseg711", "superseg"]:
        for scorer in ["GPT-5.2", "DistilBERT"]:
            try:
                result = run_verification(dataset, scorer)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Print summary table
    print("\n" + "=" * 70)
    print("### 2. Reproduced Table (BOR≈1.0)")
    print("=" * 70)

    # Original claimed values
    claimed = {
        ("dialseg711", "GPT-5.2"): {"w_50": 1, "wf1_w1": 0.756, "wf1_w3": 0.923, "gap": 0.167},
        ("dialseg711", "DistilBERT"): {"w_50": 1, "wf1_w1": 0.648, "wf1_w3": 0.858, "gap": 0.210},
        ("superseg", "GPT-5.2"): {"w_50": 1, "wf1_w1": 0.796, "wf1_w3": 0.975, "gap": 0.179},
        ("superseg", "DistilBERT"): {"w_50": 0, "wf1_w1": 0.531, "wf1_w3": 0.973, "gap": 0.442},
    }

    print(f"\n| Dataset    | Scorer     | w_50 | W-F1(w=1) | W-F1(w=3) | gap   | MATCH? |")
    print(f"|------------|------------|------|-----------|-----------|-------|--------|")

    for r in all_results:
        key = (r["dataset"], r["scorer"])
        c = claimed.get(key, {})

        w50_match = "✓" if r["w_50"] == c.get("w_50") else "✗"
        wf1_w1_match = "✓" if abs(r["wf1_w1"] - c.get("wf1_w1", 0)) < 0.01 else "✗"
        wf1_w3_match = "✓" if abs(r["wf1_w3"] - c.get("wf1_w3", 0)) < 0.01 else "✗"
        gap_match = "✓" if abs(r["gap"] - c.get("gap", 0)) < 0.02 else "✗"

        all_match = "✓" if w50_match == wf1_w1_match == wf1_w3_match == gap_match == "✓" else "✗"

        print(f"| {r['dataset']:<10} | {r['scorer']:<10} | {r['w_50']:>4} | {r['wf1_w1']:>9.4f} | {r['wf1_w3']:>9.4f} | {r['gap']:>5.3f} | {all_match}      |")

    # Print diagnostics table
    print("\n" + "=" * 70)
    print("### 4. Diagnostics")
    print("=" * 70)

    print(f"\n| Dataset    | Scorer     | w_50 | w_90 | w_95 | Prec(w=1) | Rec(w=1) | Prec(w=3) | Rec(w=3) |")
    print(f"|------------|------------|------|------|------|-----------|----------|-----------|----------|")
    for r in all_results:
        print(f"| {r['dataset']:<10} | {r['scorer']:<10} | {r['w_50']:>4} | {r['w_90']:>4} | {r['w_95']:>4} | {r['prec_w1']:>9.4f} | {r['rec_w1']:>8.4f} | {r['prec_w3']:>9.4f} | {r['rec_w3']:>8.4f} |")

    # SuperSeg DistilBERT offset histogram
    print("\n### SuperSeg DistilBERT Offset Histogram")
    for r in all_results:
        if r["dataset"] == "superseg" and r["scorer"] == "DistilBERT":
            hist = r["offset_histogram"]
            n = r["n_offsets"]
            print(f"| Offset | Count | Percentage |")
            print(f"|--------|-------|------------|")
            for k in [0, 1, 2, 3, ">3"]:
                count = hist[k]
                pct = 100 * count / n
                print(f"| {k:>6} | {count:>5} | {pct:>10.1f}% |")
            print(f"\nPredictions matching at distance 2-3: {r['match_2_3']} ({100*r['match_2_3']/n:.1f}%)")
            print(f"This explains the +0.44 gap: {100*r['match_2_3']/n:.1f}% of predictions match at distance 2-3")

    print("\n### 5. Conclusion")
    print("Verification complete. Check MATCH? column for discrepancies.")


if __name__ == "__main__":
    main()
