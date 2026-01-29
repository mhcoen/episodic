#!/usr/bin/env python3
"""
Pilot: DistilBERT vs GPT-5.2 on Exact F1 vs Tolerant W-F1

Test if the "SuperSeg pattern" generalizes: GPT-5.2 wins on tolerant W-F1 (w=3),
DistilBERT wins on Exact F1 (w=0).

Hard Constraints:
- Identical selector: greedy NMS, MIN_GAP=2
- Dialogue-level bootstrap (200 resamples)
- Only compare over intersection of BOR supports

Usage:
    python paper/experiments/pilot_exact_vs_tolerant.py
"""

import os
import sys

# Disable output buffering for progress visibility
sys.stdout.reconfigure(line_buffering=True)

import json
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_GAP = 2
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 42
TAU_PERCENTILES = list(range(1, 100, 3))  # 1, 4, 7, ..., 97
TOLERANT_WINDOW = 3

DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_FILE = PROJECT_ROOT / "paper" / "experiments" / "pilot_exact_vs_tolerant.json"

# GPT-5.2 caches (both have correct polarity: logP(N) - logP(Y))
GPT52_CACHES = {
    "dialseg711": PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json",
    "superseg": PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json",
}

# DistilBERT model
DISTILBERT_MODEL_PATH = PROJECT_ROOT / "paper" / "experiments" / "models" / "final_calibrated.pt"


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

def load_dataset(dataset_name: str) -> List[DialogueData]:
    """Load a dataset and extract dialogues with gold boundaries."""
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Dataset not found: {test_file}")

    with open(test_file) as f:
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


def load_gpt52_scores(dataset_name: str, dialogues: List[DialogueData]) -> Dict[int, Dict[int, float]]:
    """Load GPT-5.2 scores from cache, organized by dialogue."""
    cache_file = GPT52_CACHES.get(dataset_name)
    if not cache_file or not cache_file.exists():
        raise FileNotFoundError(f"GPT-5.2 cache not found for {dataset_name}")

    with open(cache_file) as f:
        cache = json.load(f)

    # Count valid entries by dialogue
    scores_by_dialogue: Dict[int, Dict[int, float]] = defaultdict(dict)
    invalid_count = 0

    for key, entry in cache.items():
        # Filter out invalid entries
        if entry.get("missing_yn_in_toplogprobs") or entry.get("invalid_first_token"):
            invalid_count += 1
            continue

        dialogue_id = entry["dialogue_id"]
        position = entry["position"]
        score = entry["score"]

        scores_by_dialogue[dialogue_id][position] = score

    print(f"  Loaded {sum(len(s) for s in scores_by_dialogue.values())} valid scores")
    print(f"  Skipped {invalid_count} invalid entries")

    return dict(scores_by_dialogue)


def load_distilbert_scores(dialogues: List[DialogueData]) -> Dict[int, Dict[int, float]]:
    """Run DistilBERT inference on dialogues and return scores."""
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Load calibrated model
    checkpoint = torch.load(DISTILBERT_MODEL_PATH, map_location=device, weights_only=False)
    temperature = checkpoint.get("temperature", 1.0)
    print(f"  Temperature: {temperature}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    scores_by_dialogue: Dict[int, Dict[int, float]] = {}

    for idx, dialogue in enumerate(tqdm(dialogues, desc="  DistilBERT scoring")):
        messages = dialogue.messages
        scores = {}

        user_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if user_idx > 0:
                    # Create window
                    window_start = max(0, i - 8)
                    window = messages[window_start:i]

                    # Format text
                    context_parts = []
                    for m in window[-6:]:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        context_parts.append(f"{role}: {content}")

                    curr_content = msg.get("content", "")
                    text = " [SEP] ".join(context_parts) + f" [SEP] current: {curr_content}"

                    # Tokenize
                    encoding = tokenizer(
                        text, max_length=256, padding="max_length",
                        truncation=True, return_tensors="pt"
                    )

                    # Get score
                    with torch.no_grad():
                        inputs = {k: v.to(device) for k, v in encoding.items()}
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits / temperature, dim=-1)
                        score = probs[0, 1].item()  # Probability of boundary

                    scores[user_idx] = score

                user_idx += 1

        scores_by_dialogue[dialogue.dialogue_id] = scores

        # Progress report every 100 dialogues
        if (idx + 1) % 100 == 0:
            print(f"  DistilBERT scoring: {idx + 1}/{len(dialogues)} dialogues ({100*(idx+1)//len(dialogues)}%)")

    return scores_by_dialogue


# =============================================================================
# METRICS
# =============================================================================

def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float, min_gap: int) -> Set[int]:
    """Greedy NMS prediction with min gap constraint."""
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    predicted = set()
    for pos, score in candidates:
        too_close = any(abs(pos - p) < min_gap for p in predicted)
        if not too_close:
            predicted.add(pos)

    return predicted


def compute_exact_f1(gold: Set[int], pred: Set[int]) -> float:
    """Compute exact match F1 (window=0)."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_tolerant_f1(gold: Set[int], pred: Set[int], window: int = 3) -> float:
    """Compute tolerant F1 with one-to-one matching (window=3)."""
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0 if pred else 1.0
    if not pred:
        return 0.0

    # Build candidate matches within window
    candidates = []
    for g in gold:
        for p in pred:
            dist = abs(g - p)
            if dist <= window:
                candidates.append((dist, g, p))

    # Greedy one-to-one matching
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

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# =============================================================================
# SWEEP COMPUTATION
# =============================================================================

def compute_sweep(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    tau_percentiles: List[int]
) -> Dict[str, List]:
    """
    Compute metric sweep over tau percentiles.

    Returns dict with keys: tau_pct, tau, bor, exact_f1, tolerant_f1,
                            per_dialogue_exact, per_dialogue_tolerant
    """
    # Collect all scores for percentile computation
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    results = {
        "tau_pct": [],
        "tau": [],
        "bor": [],
        "exact_f1": [],
        "tolerant_f1": [],
        "per_dialogue_exact": [],
        "per_dialogue_tolerant": [],
    }

    for pct in tau_percentiles:
        tau = np.percentile(all_scores, pct)

        # Per-dialogue metrics
        exact_f1s = []
        tolerant_f1s = []
        total_pred = 0
        total_gold = 0

        for dialogue_id, scores in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            pred = greedy_nms_predict(scores, tau, MIN_GAP)

            total_pred += len(pred)
            total_gold += len(gold)

            exact_f1s.append(compute_exact_f1(gold, pred))
            tolerant_f1s.append(compute_tolerant_f1(gold, pred, TOLERANT_WINDOW))

        # BOR = #pred / #gold
        bor = total_pred / total_gold if total_gold > 0 else 0.0

        # Macro-averaged F1
        mean_exact = np.mean(exact_f1s)
        mean_tolerant = np.mean(tolerant_f1s)

        results["tau_pct"].append(pct)
        results["tau"].append(float(tau))
        results["bor"].append(bor)
        results["exact_f1"].append(mean_exact)
        results["tolerant_f1"].append(mean_tolerant)
        results["per_dialogue_exact"].append(exact_f1s)
        results["per_dialogue_tolerant"].append(tolerant_f1s)

    return results


def find_bor_intersection(sweep1: Dict, sweep2: Dict) -> Tuple[float, float]:
    """Find intersection of BOR support ranges."""
    bor1 = sweep1["bor"]
    bor2 = sweep2["bor"]

    min_bor = max(min(bor1), min(bor2))
    max_bor = min(max(bor1), max(bor2))

    return min_bor, max_bor


def get_indices_in_bor_range(sweep: Dict, min_bor: float, max_bor: float) -> List[int]:
    """Get indices of sweep points within BOR range."""
    indices = []
    for i, bor in enumerate(sweep["bor"]):
        if min_bor <= bor <= max_bor:
            indices.append(i)
    return indices


# =============================================================================
# BOOTSTRAP CI
# =============================================================================

def compute_bootstrap_ci(
    distilbert_sweep: Dict,
    gpt52_sweep: Dict,
    min_bor: float,
    max_bor: float,
    n_bootstrap: int = 200,
    seed: int = 42
) -> Dict:
    """
    Compute bootstrap confidence intervals for deltas.

    Returns per-BOR-bin and overall results.
    """
    rng = np.random.default_rng(seed)

    # Get indices in BOR range
    distilbert_indices = get_indices_in_bor_range(distilbert_sweep, min_bor, max_bor)
    gpt52_indices = get_indices_in_bor_range(gpt52_sweep, min_bor, max_bor)

    if not distilbert_indices or not gpt52_indices:
        return {"error": "No overlapping BOR range"}

    # For each BOR bin, find matching indices
    # We'll use interpolation to match BOR values
    results_by_bor = []

    # Sample at representative BOR points (0.5, 0.8, 1.0, 1.2, 1.5)
    target_bors = [b for b in [0.5, 0.8, 1.0, 1.2, 1.5] if min_bor <= b <= max_bor]

    for target_bor in target_bors:
        # Find closest BOR in each sweep
        distilbert_idx = min(range(len(distilbert_sweep["bor"])),
                            key=lambda i: abs(distilbert_sweep["bor"][i] - target_bor))
        gpt52_idx = min(range(len(gpt52_sweep["bor"])),
                       key=lambda i: abs(gpt52_sweep["bor"][i] - target_bor))

        actual_distilbert_bor = distilbert_sweep["bor"][distilbert_idx]
        actual_gpt52_bor = gpt52_sweep["bor"][gpt52_idx]

        # Skip if BOR mismatch is too large
        if abs(actual_distilbert_bor - target_bor) > 0.2 or abs(actual_gpt52_bor - target_bor) > 0.2:
            continue

        # Get per-dialogue metrics
        distilbert_exact = np.array(distilbert_sweep["per_dialogue_exact"][distilbert_idx])
        distilbert_tol = np.array(distilbert_sweep["per_dialogue_tolerant"][distilbert_idx])
        gpt52_exact = np.array(gpt52_sweep["per_dialogue_exact"][gpt52_idx])
        gpt52_tol = np.array(gpt52_sweep["per_dialogue_tolerant"][gpt52_idx])

        n_dialogues = len(distilbert_exact)
        if n_dialogues != len(gpt52_exact):
            # Find common dialogues
            n_dialogues = min(len(distilbert_exact), len(gpt52_exact))
            distilbert_exact = distilbert_exact[:n_dialogues]
            distilbert_tol = distilbert_tol[:n_dialogues]
            gpt52_exact = gpt52_exact[:n_dialogues]
            gpt52_tol = gpt52_tol[:n_dialogues]

        # Bootstrap
        delta_exact_samples = []
        delta_tol_samples = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n_dialogues, size=n_dialogues, replace=True)

            distilbert_exact_mean = np.mean(distilbert_exact[indices])
            distilbert_tol_mean = np.mean(distilbert_tol[indices])
            gpt52_exact_mean = np.mean(gpt52_exact[indices])
            gpt52_tol_mean = np.mean(gpt52_tol[indices])

            # Delta = DistilBERT - GPT-5.2 (positive = DistilBERT wins)
            delta_exact_samples.append(distilbert_exact_mean - gpt52_exact_mean)
            delta_tol_samples.append(distilbert_tol_mean - gpt52_tol_mean)

        delta_exact_samples = np.array(delta_exact_samples)
        delta_tol_samples = np.array(delta_tol_samples)

        results_by_bor.append({
            "target_bor": target_bor,
            "actual_distilbert_bor": actual_distilbert_bor,
            "actual_gpt52_bor": actual_gpt52_bor,
            "distilbert_exact": float(np.mean(distilbert_exact)),
            "distilbert_tolerant": float(np.mean(distilbert_tol)),
            "gpt52_exact": float(np.mean(gpt52_exact)),
            "gpt52_tolerant": float(np.mean(gpt52_tol)),
            "delta_exact_mean": float(np.mean(delta_exact_samples)),
            "delta_exact_ci": (float(np.percentile(delta_exact_samples, 2.5)),
                              float(np.percentile(delta_exact_samples, 97.5))),
            "delta_tolerant_mean": float(np.mean(delta_tol_samples)),
            "delta_tolerant_ci": (float(np.percentile(delta_tol_samples, 2.5)),
                                 float(np.percentile(delta_tol_samples, 97.5))),
        })

    return {"by_bor": results_by_bor}


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_pattern(bootstrap_results: Dict) -> Dict:
    """
    Detect if the "SuperSeg pattern" holds:
    - DistilBERT wins on Exact F1 (positive delta_exact)
    - GPT-5.2 wins on Tolerant F1 (negative delta_tolerant)
    """
    by_bor = bootstrap_results.get("by_bor", [])
    if not by_bor:
        return {"pattern_holds": False, "reason": "No BOR data"}

    # Check pattern at each BOR point
    exact_wins = 0
    tolerant_wins = 0
    both_hold = 0

    for entry in by_bor:
        delta_exact = entry["delta_exact_mean"]
        delta_tol = entry["delta_tolerant_mean"]
        exact_ci = entry["delta_exact_ci"]
        tol_ci = entry["delta_tolerant_ci"]

        # DistilBERT wins exact if delta > 0 and CI mostly > 0
        if delta_exact > 0.02 and exact_ci[0] > -0.02:
            exact_wins += 1

        # GPT-5.2 wins tolerant if delta < 0 (i.e., GPT-5.2 > DistilBERT)
        if delta_tol < 0:
            tolerant_wins += 1

        # Both conditions
        if delta_exact > 0.02 and delta_tol < 0:
            both_hold += 1

    n_points = len(by_bor)
    pattern_holds = both_hold >= n_points * 0.5  # Holds if true for >= 50% of BOR points

    return {
        "pattern_holds": pattern_holds,
        "exact_wins_count": exact_wins,
        "tolerant_wins_count": tolerant_wins,
        "both_hold_count": both_hold,
        "n_bor_points": n_points,
        "reason": f"Both hold at {both_hold}/{n_points} BOR points"
    }


# =============================================================================
# MAIN
# =============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def process_dataset(dataset_name: str) -> Dict:
    """Process a single dataset."""
    print(f"\n{'='*60}")
    print(f"=== Processing {dataset_name} ===")
    print(f"{'='*60}")

    # Load data
    print("\nLoading dataset...")
    dialogues = load_dataset(dataset_name)
    print(f"  Dialogues: {len(dialogues)}")
    total_gold = sum(len(d.gold_boundaries) for d in dialogues)
    print(f"  Total gold boundaries: {total_gold}")

    # Create gold_by_dialogue mapping
    gold_by_dialogue = {d.dialogue_id: d.gold_boundaries for d in dialogues}

    # Load GPT-5.2 scores
    print("\nLoading GPT-5.2 scores...")
    gpt52_scores = load_gpt52_scores(dataset_name, dialogues)
    print(f"  Dialogues with scores: {len(gpt52_scores)}")
    print(f"  Estimated cost: $0.00 (all cached)")

    # Load DistilBERT scores
    print("\nRunning DistilBERT inference...")
    distilbert_scores = load_distilbert_scores(dialogues)
    print(f"  Dialogues scored: {len(distilbert_scores)}")

    # Compute sweeps
    print("\nComputing sweeps...")
    print("  Computing GPT-5.2 sweep...")
    gpt52_sweep = compute_sweep(gpt52_scores, gold_by_dialogue, TAU_PERCENTILES)
    print(f"  GPT-5.2 BOR range: [{min(gpt52_sweep['bor']):.2f}, {max(gpt52_sweep['bor']):.2f}]")

    print("  Computing DistilBERT sweep...")
    distilbert_sweep = compute_sweep(distilbert_scores, gold_by_dialogue, TAU_PERCENTILES)
    print(f"  DistilBERT BOR range: [{min(distilbert_sweep['bor']):.2f}, {max(distilbert_sweep['bor']):.2f}]")

    # Find BOR intersection
    min_bor, max_bor = find_bor_intersection(distilbert_sweep, gpt52_sweep)
    print(f"\nBOR intersection: [{min_bor:.2f}, {max_bor:.2f}]")

    # Bootstrap CIs
    print(f"\nComputing bootstrap CIs ({N_BOOTSTRAP} resamples)...")
    bootstrap_results = compute_bootstrap_ci(
        distilbert_sweep, gpt52_sweep, min_bor, max_bor, N_BOOTSTRAP, BOOTSTRAP_SEED
    )

    # Pattern detection
    pattern_result = detect_pattern(bootstrap_results)
    pattern_holds = pattern_result["pattern_holds"]

    # Print results
    print(f"\n{'='*60}")
    print(f"=== {dataset_name} Results ===")
    print(f"{'='*60}")
    print(f"BOR support: DistilBERT [{min(distilbert_sweep['bor']):.2f}, {max(distilbert_sweep['bor']):.2f}], "
          f"GPT-5.2 [{min(gpt52_sweep['bor']):.2f}, {max(gpt52_sweep['bor']):.2f}]")
    print(f"Intersection: [{min_bor:.2f}, {max_bor:.2f}]")

    for entry in bootstrap_results.get("by_bor", []):
        bor = entry["target_bor"]
        print(f"\nAt BOR={bor:.1f}:")
        print(f"  DistilBERT: Exact={entry['distilbert_exact']:.3f}, Tolerant={entry['distilbert_tolerant']:.3f}")
        print(f"  GPT-5.2:    Exact={entry['gpt52_exact']:.3f}, Tolerant={entry['gpt52_tolerant']:.3f}")
        delta_exact = entry["delta_exact_mean"]
        delta_tol = entry["delta_tolerant_mean"]
        exact_ci = entry["delta_exact_ci"]
        tol_ci = entry["delta_tolerant_ci"]

        exact_winner = "DistilBERT" if delta_exact > 0 else "GPT-5.2"
        tol_winner = "DistilBERT" if delta_tol > 0 else "GPT-5.2"

        print(f"  Delta exact:    {delta_exact:+.3f} [{exact_ci[0]:+.3f}, {exact_ci[1]:+.3f}]  <- {exact_winner} wins")
        print(f"  Delta tolerant: {delta_tol:+.3f} [{tol_ci[0]:+.3f}, {tol_ci[1]:+.3f}]  <- {tol_winner} wins")

    print(f"\nPattern holds: {'YES' if pattern_holds else 'NO'}")
    print(f"  {pattern_result['reason']}")

    # Return results
    return {
        "dataset": dataset_name,
        "n_dialogues": len(dialogues),
        "n_gold_boundaries": total_gold,
        "distilbert_bor_range": [min(distilbert_sweep["bor"]), max(distilbert_sweep["bor"])],
        "gpt52_bor_range": [min(gpt52_sweep["bor"]), max(gpt52_sweep["bor"])],
        "bor_intersection": [min_bor, max_bor],
        "bootstrap_results": bootstrap_results,
        "pattern_result": pattern_result,
        "gpt52_cost": 0.0,
        "sweeps": {
            "distilbert": {
                "tau_pct": distilbert_sweep["tau_pct"],
                "tau": distilbert_sweep["tau"],
                "bor": distilbert_sweep["bor"],
                "exact_f1": distilbert_sweep["exact_f1"],
                "tolerant_f1": distilbert_sweep["tolerant_f1"],
            },
            "gpt52": {
                "tau_pct": gpt52_sweep["tau_pct"],
                "tau": gpt52_sweep["tau"],
                "bor": gpt52_sweep["bor"],
                "exact_f1": gpt52_sweep["exact_f1"],
                "tolerant_f1": gpt52_sweep["tolerant_f1"],
            }
        }
    }


def print_summary_table(results: List[Dict]):
    """Print summary table."""
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = f"{'dataset':<12} | {'NMS BOR_max':>11} | {'intersection':>14} | {'winner_exact':>12} | {'winner_tolerant':>15} | {'notes':>15} | {'cost':>6}"
    print(header)
    print("-" * 100)

    for r in results:
        dataset = r["dataset"]
        bor_max = max(r["gpt52_bor_range"])
        intersection = f"[{r['bor_intersection'][0]:.1f}, {r['bor_intersection'][1]:.1f}]"

        # Determine winners from BOR=1.0 results
        by_bor = r["bootstrap_results"].get("by_bor", [])
        winner_exact = "N/A"
        winner_tolerant = "N/A"

        for entry in by_bor:
            if abs(entry["target_bor"] - 1.0) < 0.3:
                winner_exact = "DistilBERT" if entry["delta_exact_mean"] > 0 else "GPT-5.2"
                winner_tolerant = "DistilBERT" if entry["delta_tolerant_mean"] > 0 else "GPT-5.2"
                break

        pattern = "Pattern YES" if r["pattern_result"]["pattern_holds"] else "Pattern NO"
        cost = f"${r['gpt52_cost']:.2f}"

        row = f"{dataset:<12} | {bor_max:>11.1f} | {intersection:>14} | {winner_exact:>12} | {winner_tolerant:>15} | {pattern:>15} | {cost:>6}"
        print(row)

    print(f"{'='*100}")


def main():
    print("=" * 60)
    print("Pilot: DistilBERT vs GPT-5.2 on Exact F1 vs Tolerant W-F1")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Git commit: {get_git_commit_hash()}")
    print()
    print("Configuration:")
    print(f"  MIN_GAP: {MIN_GAP}")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")
    print(f"  Tolerant window: {TOLERANT_WINDOW}")
    print(f"  Tau percentiles: {len(TAU_PERCENTILES)} points")

    datasets = ["dialseg711", "superseg"]
    results = []

    for dataset_name in datasets:
        result = process_dataset(dataset_name)
        results.append(result)

    # Print summary table
    print_summary_table(results)

    # Save results
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "git_commit": get_git_commit_hash(),
            "selector": {"type": "greedy_nms", "min_gap": MIN_GAP},
            "metrics": {"exact": "w=0", "tolerant": f"w={TOLERANT_WINDOW}, one_to_one"},
            "bootstrap": {"n": N_BOOTSTRAP, "ci_level": 0.95, "seed": BOOTSTRAP_SEED}
        },
        "summary_table": [
            {
                "dataset": r["dataset"],
                "nms_bor_max": max(r["gpt52_bor_range"]),
                "intersection_bor": f"[{r['bor_intersection'][0]:.1f}, {r['bor_intersection'][1]:.1f}]",
                "winner_exact": "DistilBERT" if any(
                    e["delta_exact_mean"] > 0 for e in r["bootstrap_results"].get("by_bor", [])
                    if abs(e["target_bor"] - 1.0) < 0.3
                ) else "GPT-5.2",
                "winner_tolerant": "DistilBERT" if any(
                    e["delta_tolerant_mean"] > 0 for e in r["bootstrap_results"].get("by_bor", [])
                    if abs(e["target_bor"] - 1.0) < 0.3
                ) else "GPT-5.2",
                "notes": "Pattern YES" if r["pattern_result"]["pattern_holds"] else "Pattern NO",
                "gpt52_cost": r["gpt52_cost"]
            }
            for r in results
        ],
        "datasets": {r["dataset"]: r for r in results}
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
