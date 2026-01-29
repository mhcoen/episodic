#!/usr/bin/env python3
"""
Adaptive Window Length Experiment

Test whether a data-driven window w derived from prediction–gold offsets
can replace the fixed w=3 used throughout the paper.

Hypothesis: A median-derived window is a defensible default tolerance
grounded in observed offset structure.

Protocol:
1. Pick operating points by BOR target (0.8, 1.0, 1.2) - avoid circularity
2. Compute median absolute offset at each operating point
3. Evaluate W-F1 at multiple window sizes
4. Bootstrap stability check

Usage:
    python paper/experiments/adaptive_window_experiment.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "paper" / "experiments"

# GPT-5.2 caches
GPT52_DIALSEG_CACHE = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"
GPT52_SUPERSEG_CACHE = PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json"

# DistilBERT per-dialogue data
DISTILBERT_DIALSEG = RESULTS_DIR / "sweep_dialseg711_neural_per_dialogue.json"
DISTILBERT_SUPERSEG = RESULTS_DIR / "sweep_superseg_neural_per_dialogue.json"

# Output
OUTPUT_JSON = EXPERIMENTS_DIR / "adaptive_window_experiment.json"

# Configuration
MIN_GAP = 2
BOR_TARGETS = [0.8, 1.0, 1.2]
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 42


@dataclass
class OperatingPoint:
    """Results for a single operating point."""
    dataset: str
    scorer: str
    bor_target: float
    tau: float
    actual_bor: float
    n_predictions: int
    n_gold: int
    w_50: int  # median-derived window
    w_90: int  # 90th percentile window
    offset_median: float
    offset_mean: float
    offset_p90: float
    wf1_w0: float  # exact
    wf1_w1: float
    wf1_w50: float  # adaptive median
    wf1_w3: float  # baseline
    wf1_w90: float  # adaptive 90th


@dataclass
class BootstrapResult:
    """Bootstrap stability results."""
    w50_mean: float
    w50_std: float
    w50_ci_lo: float
    w50_ci_hi: float
    delta_mean: float  # W-F1(w=3) - W-F1(w=w_50)
    delta_std: float
    delta_ci_lo: float
    delta_ci_hi: float


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


def load_distilbert_per_dialogue(dataset_name: str) -> Dict:
    """Load DistilBERT per-dialogue data."""
    if dataset_name == "dialseg711":
        path = DISTILBERT_DIALSEG
    else:
        path = DISTILBERT_SUPERSEG

    with open(path) as f:
        return json.load(f)


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
    """Compute absolute offset from each prediction to nearest gold."""
    if not predictions or not gold:
        return []

    offsets = []
    for p in predictions:
        if gold:
            min_dist = min(abs(p - g) for g in gold)
            offsets.append(min_dist)

    return offsets


def compute_wf1(predicted: Set[int], gold: Set[int], w: int) -> float:
    """Compute W-F1 with window w (many-to-one matching)."""
    if not gold:
        return 0.0 if predicted else 1.0

    # Count TPs: predictions within w of any gold
    tp = sum(1 for p in predicted if any(abs(p - g) <= w for g in gold))

    # Count matched gold
    matched_gold = sum(1 for g in gold if any(abs(p - g) <= w for p in predicted))

    fp = len(predicted) - tp
    fn = len(gold) - matched_gold

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = matched_gold / len(gold) if gold else 0.0

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def find_tau_for_bor(scores_by_dialogue: Dict[int, Dict[int, float]],
                     gold_by_dialogue: Dict[int, Set[int]],
                     target_bor: float) -> Tuple[float, float]:
    """Find tau that achieves closest BOR to target."""
    # Collect all scores
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    total_gold = sum(len(g) for g in gold_by_dialogue.values())

    best_tau = None
    best_bor = None
    best_diff = float('inf')

    # Search over percentiles
    for pct in range(1, 100):
        tau = np.percentile(all_scores, pct)

        # Count predictions at this tau
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


def run_experiment_gpt52(dataset_name: str) -> List[OperatingPoint]:
    """Run experiment for GPT-5.2 on one dataset."""
    print(f"\n{'='*60}")
    print(f"GPT-5.2 on {dataset_name}")
    print(f"{'='*60}")

    scores_by_dialogue = load_gpt52_scores(dataset_name)
    gold_by_dialogue = load_dataset_gold(dataset_name)

    total_gold = sum(len(g) for g in gold_by_dialogue.values())
    print(f"Loaded {len(scores_by_dialogue)} dialogues with scores")
    print(f"Total gold boundaries: {total_gold}")

    results = []

    for target_bor in BOR_TARGETS:
        print(f"\n--- BOR target: {target_bor} ---")

        # Step 1: Find tau for this BOR
        tau, actual_bor = find_tau_for_bor(scores_by_dialogue, gold_by_dialogue, target_bor)
        print(f"  tau* = {tau:.4f}, actual BOR = {actual_bor:.4f}")

        # Step 2: Compute predictions and offsets
        all_offsets = []
        all_predictions = {}
        all_gold = {}
        total_pred = 0

        for did, scores in scores_by_dialogue.items():
            pred = greedy_nms(scores, tau)
            gold = gold_by_dialogue.get(did, set())

            all_predictions[did] = pred
            all_gold[did] = gold
            total_pred += len(pred)

            offsets = compute_offsets(pred, gold)
            all_offsets.extend(offsets)

        if not all_offsets:
            print(f"  WARNING: No offsets computed")
            continue

        all_offsets = np.array(all_offsets)
        offset_median = np.median(all_offsets)
        offset_mean = np.mean(all_offsets)
        offset_p90 = np.percentile(all_offsets, 90)

        w_50 = int(np.ceil(offset_median))
        w_90 = int(np.ceil(offset_p90))

        print(f"  Offset median: {offset_median:.2f} → w_50 = {w_50}")
        print(f"  Offset 90th: {offset_p90:.2f} → w_90 = {w_90}")

        # Step 3: Evaluate at multiple window sizes
        window_sizes = [0, 1, w_50, 3, w_90]
        wf1_by_w = {}

        for w in window_sizes:
            total_wf1 = 0.0
            n = 0
            for did, pred in all_predictions.items():
                gold = all_gold.get(did, set())
                if gold:
                    wf1 = compute_wf1(pred, gold, w)
                    total_wf1 += wf1
                    n += 1
            avg_wf1 = total_wf1 / n if n > 0 else 0.0
            wf1_by_w[w] = avg_wf1
            print(f"  W-F1(w={w}): {avg_wf1:.4f}")

        results.append(OperatingPoint(
            dataset=dataset_name,
            scorer="GPT-5.2",
            bor_target=target_bor,
            tau=float(tau),
            actual_bor=actual_bor,
            n_predictions=total_pred,
            n_gold=total_gold,
            w_50=w_50,
            w_90=w_90,
            offset_median=float(offset_median),
            offset_mean=float(offset_mean),
            offset_p90=float(offset_p90),
            wf1_w0=wf1_by_w[0],
            wf1_w1=wf1_by_w[1],
            wf1_w50=wf1_by_w[w_50],
            wf1_w3=wf1_by_w[3],
            wf1_w90=wf1_by_w[w_90],
        ))

    return results


def load_distilbert_scores(dataset_name: str) -> Tuple[Dict[int, Dict[int, float]], List]:
    """Load DistilBERT scores by running the model."""
    import torch
    from tqdm import tqdm

    # Load dataset
    if dataset_name == "dialseg711":
        path = DATASETS_DIR / "dialseg711" / "segmentation_file_test.json"
    else:
        path = DATASETS_DIR / "superseg" / "segmentation_file_test.json"

    with open(path) as f:
        data = json.load(f)

    # Load model
    model_path = PROJECT_ROOT / "paper" / "experiments" / "models" / "final_calibrated.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)

    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Process dialogues
    scores_by_dialogue = {}
    dialogues_list = []

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

            # Score each user turn position
            scores = {}
            user_idx = 0

            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    if user_idx > 0:
                        # Create window
                        window_start = max(0, i - 8)
                        window = messages[window_start:i]

                        # Format text
                        pre_text = " ".join([m["content"] for m in window])
                        post_text = msg["content"]
                        text = f"{pre_text} [SEP] {post_text}"

                        # Tokenize and score
                        inputs = tokenizer(
                            text, truncation=True, max_length=512,
                            return_tensors="pt"
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits / temperature
                            probs = torch.softmax(logits, dim=-1)
                            score = probs[0, 1].item()  # Probability of boundary

                        scores[user_idx] = score

                    user_idx += 1

            scores_by_dialogue[dialogue_id] = scores
            dialogues_list.append({"dialogue_id": dialogue_id, "n_user_turns": user_idx})
            dialogue_id += 1

    return scores_by_dialogue, dialogues_list


def run_experiment_distilbert(dataset_name: str) -> List[OperatingPoint]:
    """Run experiment for DistilBERT on one dataset."""
    print(f"\n{'='*60}")
    print(f"DistilBERT on {dataset_name}")
    print(f"{'='*60}")

    # Load scores (this runs the model)
    print("  Loading DistilBERT model and scoring...")
    scores_by_dialogue, _ = load_distilbert_scores(dataset_name)
    gold_by_dialogue = load_dataset_gold(dataset_name)

    total_gold = sum(len(g) for g in gold_by_dialogue.values())
    print(f"  Loaded {len(scores_by_dialogue)} dialogues with scores")
    print(f"  Total gold boundaries: {total_gold}")

    results = []

    for target_bor in BOR_TARGETS:
        print(f"\n--- BOR target: {target_bor} ---")

        # Find tau for this BOR
        tau, actual_bor = find_tau_for_bor(scores_by_dialogue, gold_by_dialogue, target_bor)
        print(f"  tau* = {tau:.6f}, actual BOR = {actual_bor:.4f}")

        # Compute predictions and offsets
        all_offsets = []
        all_predictions = {}
        all_gold = {}
        total_pred = 0

        for did, scores in scores_by_dialogue.items():
            pred = greedy_nms(scores, tau)
            gold = gold_by_dialogue.get(did, set())

            all_predictions[did] = pred
            all_gold[did] = gold
            total_pred += len(pred)

            offsets = compute_offsets(pred, gold)
            all_offsets.extend(offsets)

        if not all_offsets:
            print(f"  WARNING: No offsets computed")
            continue

        all_offsets = np.array(all_offsets)
        offset_median = np.median(all_offsets)
        offset_mean = np.mean(all_offsets)
        offset_p90 = np.percentile(all_offsets, 90)

        w_50 = int(np.ceil(offset_median))
        w_90 = int(np.ceil(offset_p90))

        print(f"  Offset median: {offset_median:.2f} → w_50 = {w_50}")
        print(f"  Offset 90th: {offset_p90:.2f} → w_90 = {w_90}")

        # Evaluate at multiple window sizes
        window_sizes = [0, 1, w_50, 3, w_90]
        wf1_by_w = {}

        for w in window_sizes:
            total_wf1 = 0.0
            n = 0
            for did, pred in all_predictions.items():
                gold = all_gold.get(did, set())
                if gold:
                    wf1 = compute_wf1(pred, gold, w)
                    total_wf1 += wf1
                    n += 1
            avg_wf1 = total_wf1 / n if n > 0 else 0.0
            wf1_by_w[w] = avg_wf1
            print(f"  W-F1(w={w}): {avg_wf1:.4f}")

        results.append(OperatingPoint(
            dataset=dataset_name,
            scorer="DistilBERT",
            bor_target=target_bor,
            tau=float(tau),
            actual_bor=actual_bor,
            n_predictions=total_pred,
            n_gold=total_gold,
            w_50=w_50,
            w_90=w_90,
            offset_median=float(offset_median),
            offset_mean=float(offset_mean),
            offset_p90=float(offset_p90),
            wf1_w0=wf1_by_w[0],
            wf1_w1=wf1_by_w[1],
            wf1_w50=wf1_by_w[w_50],
            wf1_w3=wf1_by_w[3],
            wf1_w90=wf1_by_w[w_90],
        ))

    return results


def run_bootstrap(scores_by_dialogue: Dict[int, Dict[int, float]],
                  gold_by_dialogue: Dict[int, Set[int]],
                  tau: float,
                  n_bootstrap: int = N_BOOTSTRAP) -> BootstrapResult:
    """Run bootstrap stability check for adaptive window."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    dialogue_ids = list(scores_by_dialogue.keys())
    n_dial = len(dialogue_ids)

    w50_samples = []
    delta_samples = []

    for _ in range(n_bootstrap):
        # Resample dialogues
        sampled_ids = rng.choice(dialogue_ids, size=n_dial, replace=True)

        # Compute predictions and offsets on resample
        all_offsets = []
        predictions_by_did = {}

        for did in sampled_ids:
            scores = scores_by_dialogue[did]
            gold = gold_by_dialogue.get(did, set())
            pred = greedy_nms(scores, tau)
            predictions_by_did[did] = (pred, gold)
            offsets = compute_offsets(pred, gold)
            all_offsets.extend(offsets)

        if not all_offsets:
            continue

        w_50 = int(np.ceil(np.median(all_offsets)))
        w50_samples.append(w_50)

        # Compute W-F1 difference
        wf1_w3 = 0.0
        wf1_w50 = 0.0
        n = 0

        for did in sampled_ids:
            pred, gold = predictions_by_did[did]
            if gold:
                wf1_w3 += compute_wf1(pred, gold, 3)
                wf1_w50 += compute_wf1(pred, gold, w_50)
                n += 1

        if n > 0:
            delta = (wf1_w3 / n) - (wf1_w50 / n)
            delta_samples.append(delta)

    w50_samples = np.array(w50_samples)
    delta_samples = np.array(delta_samples)

    return BootstrapResult(
        w50_mean=float(np.mean(w50_samples)),
        w50_std=float(np.std(w50_samples)),
        w50_ci_lo=float(np.percentile(w50_samples, 2.5)),
        w50_ci_hi=float(np.percentile(w50_samples, 97.5)),
        delta_mean=float(np.mean(delta_samples)),
        delta_std=float(np.std(delta_samples)),
        delta_ci_lo=float(np.percentile(delta_samples, 2.5)),
        delta_ci_hi=float(np.percentile(delta_samples, 97.5)),
    )


def print_summary_table(results: List[OperatingPoint]):
    """Print summary table to console."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Adaptive Window Experiment")
    print("=" * 120)

    header = (
        f"{'Dataset':<12} | {'Scorer':<10} | {'BOR_tgt':>7} | {'τ*':>8} | {'BOR':>5} | "
        f"{'w_50':>4} | {'w_90':>4} | {'W-F1(0)':>8} | {'W-F1(1)':>8} | "
        f"{'W-F1(w50)':>9} | {'W-F1(3)':>8} | {'W-F1(w90)':>9}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        row = (
            f"{r.dataset:<12} | {r.scorer:<10} | {r.bor_target:>7.1f} | {r.tau:>8.4f} | {r.actual_bor:>5.2f} | "
            f"{r.w_50:>4} | {r.w_90:>4} | {r.wf1_w0:>8.4f} | {r.wf1_w1:>8.4f} | "
            f"{r.wf1_w50:>9.4f} | {r.wf1_w3:>8.4f} | {r.wf1_w90:>9.4f}"
        )
        print(row)

    print("=" * 120)


def main():
    print("=" * 60)
    print("ADAPTIVE WINDOW LENGTH EXPERIMENT")
    print("=" * 60)

    all_results = []
    bootstrap_results = {}

    # Run GPT-5.2 experiments
    for dataset in ["dialseg711", "superseg"]:
        try:
            results = run_experiment_gpt52(dataset)
            all_results.extend(results)

            # Bootstrap for BOR ≈ 1.0 point
            bor1_result = [r for r in results if abs(r.bor_target - 1.0) < 0.1]
            if bor1_result:
                print(f"\nRunning bootstrap for GPT-5.2 {dataset} at BOR≈1.0...")
                scores = load_gpt52_scores(dataset)
                gold = load_dataset_gold(dataset)
                bs = run_bootstrap(scores, gold, bor1_result[0].tau)
                bootstrap_results[f"GPT-5.2_{dataset}"] = asdict(bs)
                print(f"  w_50: {bs.w50_mean:.2f} ± {bs.w50_std:.2f} [{bs.w50_ci_lo:.0f}, {bs.w50_ci_hi:.0f}]")
                print(f"  Δ(W-F1): {bs.delta_mean:.4f} ± {bs.delta_std:.4f} [{bs.delta_ci_lo:.4f}, {bs.delta_ci_hi:.4f}]")
        except Exception as e:
            print(f"ERROR on GPT-5.2 {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Run DistilBERT experiments
    for dataset in ["dialseg711", "superseg"]:
        try:
            results = run_experiment_distilbert(dataset)
            all_results.extend(results)
        except Exception as e:
            print(f"ERROR on DistilBERT {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print_summary_table(all_results)

    # Save results
    output = {
        "description": "Adaptive window length experiment",
        "hypothesis": "Median-derived window is a defensible default tolerance",
        "config": {
            "min_gap": MIN_GAP,
            "bor_targets": BOR_TARGETS,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "results": [asdict(r) for r in all_results],
        "bootstrap": bootstrap_results,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_JSON}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for r in all_results:
        if abs(r.bor_target - 1.0) < 0.1:
            delta = r.wf1_w3 - r.wf1_w50
            print(f"{r.scorer} {r.dataset}:")
            print(f"  w_50 = {r.w_50} (median offset = {r.offset_median:.2f})")
            print(f"  W-F1(w=3) - W-F1(w=w_50) = {delta:.4f}")
            if r.w_50 > 1:
                print(f"  → Larger w_50 suggests systematic boundary offset")


if __name__ == "__main__":
    main()
