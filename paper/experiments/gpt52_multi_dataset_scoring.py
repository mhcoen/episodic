#!/usr/bin/env python3
"""
Score DailyDialog, TopicalChat, and TIAGE with GPT-5.2.
Generate Figure 4 panels for each dataset.
"""

import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".gpt52_multi_cache"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments"

# Config
GPT52_INPUT_PRICE_PER_1M = 1.75
GPT52_OUTPUT_PRICE_PER_1M = 14.0
HARD_BUDGET_USD = 3.0
TOP_LOGPROBS = 5
PROGRESS_INTERVAL = 100
SAVE_INTERVAL = 100
MIN_GAP = 2
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TAU_PERCENTILES = [1, 2, 3] + list(range(5, 100, 5))

# Datasets to process
DATASETS = [
    {
        "name": "DailyDialog",
        "file": PROJECT_ROOT / "datasets" / "dailydialog" / "segmentation_file_test.json",
        "prefix": "dailydialog",
    },
    {
        "name": "TopicalChat",
        "file": PROJECT_ROOT / "datasets" / "topical_chat" / "segmentation_file_test.json",
        "prefix": "topicalchat",
    },
    {
        "name": "TIAGE",
        "file": PROJECT_ROOT / "datasets" / "tiage" / "segmentation_file_test.json",
        "prefix": "tiage",
    },
]

# Prompt template
SYSTEM_PROMPT = """You are a discourse segmentation expert. A segment boundary occurs when the conversation shifts to a new topic, task, or phase. Return ONLY 'Y' or 'N' (single token)."""

def build_user_prompt(context_before: List[str], context_after: str) -> str:
    context_lines = []
    for i, turn in enumerate(context_before[-4:]):
        context_lines.append(f"[{i+1}] {turn}")
    context_str = "\n".join(context_lines)
    return f"""Context before boundary:
{context_str}

--- CANDIDATE BOUNDARY ---

Next turn:
{context_after}

Is this a topic boundary? Decision:"""

def compute_prompt_hash(system: str, user: str) -> str:
    content = f"{system}|{user}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def get_cache_key(prefix: str, dialogue_id: int, position: int, prompt_hash: str) -> str:
    return f"{prefix}_{dialogue_id}_{position}_{prompt_hash}"

@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_calls: int = 0

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * GPT52_INPUT_PRICE_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * GPT52_OUTPUT_PRICE_PER_1M
        return input_cost + output_cost

def load_dataset(file_path: Path):
    """Load dataset and extract dialogues with gold boundaries."""
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
            messages = []

            for turn in turns:
                role = turn.get("role")
                content = turn.get("utterance", turn.get("text", ""))
                messages.append({"role": role, "content": content})

                if role == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            dialogues.append({
                "dialogue_id": dialogue_id,
                "messages": messages,
                "gold_boundaries": boundaries,
                "num_user_turns": user_idx
            })
            dialogue_id += 1

    return dialogues

def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float) -> Set[int]:
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

def compute_wf1(predicted: Set[int], gold: Set[int], one_to_one: bool = False, k: int = 3) -> float:
    if not gold:
        return 0.0 if predicted else 1.0

    if one_to_one:
        pred_list = sorted(predicted)
        gold_list = sorted(gold)
        pairs = []
        for p in pred_list:
            for g in gold_list:
                if abs(p - g) <= k:
                    pairs.append((abs(p - g), p, g))
        pairs.sort()
        matched_pred = set()
        matched_gold = set()
        for dist, p, g in pairs:
            if p not in matched_pred and g not in matched_gold:
                matched_pred.add(p)
                matched_gold.add(g)
        tp = len(matched_pred)
    else:
        matched_pred = set()
        matched_gold = set()
        for p in predicted:
            for g in gold:
                if abs(p - g) <= k and g not in matched_gold:
                    matched_pred.add(p)
                    matched_gold.add(g)
                    break
        tp = len(matched_pred)

    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold) if one_to_one else len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_sweep(scores_by_dialogue: Dict, gold_by_dialogue: Dict) -> List[Dict]:
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    sweep_points = []
    for pct in TAU_PERCENTILES:
        tau = np.percentile(all_scores, pct)

        total_wf1_m2o = 0.0
        total_wf1_1to1 = 0.0
        total_pred = 0
        total_gold = 0
        n = 0

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            if not gold:
                continue

            predicted = greedy_nms_predict(scores_by_pos, tau)
            wf1_m2o = compute_wf1(predicted, gold, one_to_one=False)
            wf1_1to1 = compute_wf1(predicted, gold, one_to_one=True)

            total_wf1_m2o += wf1_m2o
            total_wf1_1to1 += wf1_1to1
            total_pred += len(predicted)
            total_gold += len(gold)
            n += 1

        if n > 0:
            sweep_points.append({
                "percentile": pct,
                "tau": float(tau),
                "bor": total_pred / total_gold if total_gold > 0 else 0,
                "wf1_m2o": total_wf1_m2o / n,
                "wf1_1to1": total_wf1_1to1 / n,
            })

    return sweep_points

def generate_figure4_panel(dataset_name: str, scores_by_dialogue: Dict, gold_by_dialogue: Dict):
    """Generate Figure 4 panel for a dataset."""
    print(f"\n  Generating Figure 4 panel for {dataset_name}...", flush=True)

    n_dialogues = len(scores_by_dialogue)
    n_boundaries = sum(len(s) for s in scores_by_dialogue.values())
    print(f"    Loaded {n_boundaries} scores across {n_dialogues} dialogues", flush=True)

    # Compute main sweep
    print("    Computing threshold sweep...", flush=True)
    sweep_points = compute_sweep(scores_by_dialogue, gold_by_dialogue)

    if not sweep_points:
        print("    ERROR: No sweep points computed", flush=True)
        return

    best_m2o = max(sweep_points, key=lambda x: x["wf1_m2o"])
    best_1to1 = max(sweep_points, key=lambda x: x["wf1_1to1"])
    print(f"    Peak W-F1 (m2o): {best_m2o['wf1_m2o']:.4f} at BOR={best_m2o['bor']:.3f}", flush=True)
    print(f"    Peak W-F1 (1to1): {best_1to1['wf1_1to1']:.4f} at BOR={best_1to1['bor']:.3f}", flush=True)

    # Bootstrap CIs
    print(f"    Bootstrap ({N_BOOTSTRAP} iterations)...", flush=True)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    dialogue_ids = list(scores_by_dialogue.keys())
    n_dial = len(dialogue_ids)

    bootstrap_wf1_m2o = defaultdict(list)
    bootstrap_wf1_1to1 = defaultdict(list)
    bootstrap_bor = defaultdict(list)

    for b in range(N_BOOTSTRAP):
        if (b + 1) % 200 == 0:
            print(f"      Bootstrap: {b + 1}/{N_BOOTSTRAP}", flush=True)

        sampled_ids = rng.choice(dialogue_ids, size=n_dial, replace=True)
        resampled_scores = {}
        resampled_gold = {}
        for new_id, orig_id in enumerate(sampled_ids):
            resampled_scores[new_id] = scores_by_dialogue[orig_id]
            resampled_gold[new_id] = gold_by_dialogue.get(orig_id, set())

        sweep = compute_sweep(resampled_scores, resampled_gold)
        for sp in sweep:
            pct = sp["percentile"]
            bootstrap_wf1_m2o[pct].append(sp["wf1_m2o"])
            bootstrap_wf1_1to1[pct].append(sp["wf1_1to1"])
            bootstrap_bor[pct].append(sp["bor"])

    # Compute CIs
    ci_data = {}
    for pct in TAU_PERCENTILES:
        if pct in bootstrap_wf1_m2o:
            m2o = np.array(bootstrap_wf1_m2o[pct])
            o2o = np.array(bootstrap_wf1_1to1[pct])
            bor = np.array(bootstrap_bor[pct])

            ci_data[pct] = {
                "wf1_m2o_lo": float(np.percentile(m2o, 2.5)),
                "wf1_m2o_hi": float(np.percentile(m2o, 97.5)),
                "wf1_1to1_lo": float(np.percentile(o2o, 2.5)),
                "wf1_1to1_hi": float(np.percentile(o2o, 97.5)),
                "bor_lo": float(np.percentile(bor, 2.5)),
                "bor_hi": float(np.percentile(bor, 97.5)),
            }

    # Save JSON
    output_json = OUTPUT_DIR / f"gpt52_{dataset_name.lower()}_figure4.json"
    print(f"    Saving results to {output_json}...", flush=True)
    results = {
        "description": f"GPT-5.2 {dataset_name} Figure 4",
        "n_dialogues": n_dialogues,
        "n_boundaries": n_boundaries,
        "n_bootstrap": N_BOOTSTRAP,
        "peak_wf1_m2o": best_m2o["wf1_m2o"],
        "peak_bor_m2o": best_m2o["bor"],
        "peak_wf1_1to1": best_1to1["wf1_1to1"],
        "peak_bor_1to1": best_1to1["bor"],
        "sweep_points": sweep_points,
        "bootstrap_ci": ci_data,
    }
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plot
    output_png = OUTPUT_DIR / f"gpt52_{dataset_name.lower()}_figure4.png"
    print(f"    Generating plot {output_png}...", flush=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        bors = [sp["bor"] for sp in sweep_points]
        idx = np.argsort(bors)
        bors = np.array(bors)[idx]

        for ax_idx, (metric, title) in enumerate([("wf1_m2o", "W-F1 (many-to-one)"), ("wf1_1to1", "W-F1 (one-to-one)")]):
            ax = axes[ax_idx]

            wf1s = np.array([sp[metric] for sp in sweep_points])[idx]
            ci_lo = np.array([ci_data[sp["percentile"]][f"{metric}_lo"] for sp in sweep_points])[idx]
            ci_hi = np.array([ci_data[sp["percentile"]][f"{metric}_hi"] for sp in sweep_points])[idx]

            ax.fill_between(bors, ci_lo, ci_hi, color='#8B5CF6', alpha=0.2, label='95% CI')
            ax.plot(bors, wf1s, '-', color='#8B5CF6', linewidth=2.5, marker='o', markersize=4, label='GPT-5.2')

            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel('BOR')
            ax.set_ylabel(title)
            ax.set_title(f'{dataset_name}: {title}')
            ax.set_xlim(0, max(2.5, max(bors) * 1.1))
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

        fig.suptitle(f'GPT-5.2 Boundary Scoring: {dataset_name}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_png, dpi=150)
        plt.close(fig)
        print("    Plot saved!", flush=True)

    except Exception as e:
        print(f"    Plot error: {e}", flush=True)

    return results

# ============================================================
# MAIN
# ============================================================

print("=" * 60, flush=True)
print("=== GPT-5.2 Multi-Dataset Scoring ===", flush=True)
print("=" * 60, flush=True)

# Load/create cache
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_file = CACHE_DIR / "cache.json"
cache = {}
if cache_file.exists():
    with open(cache_file) as f:
        cache = json.load(f)
print(f"Cached entries: {len(cache)}", flush=True)

# Also check old cache for DailyDialog entries
old_cache_file = PROJECT_ROOT / ".gpt52_cache" / "cache.json"
old_cache = {}
if old_cache_file.exists():
    with open(old_cache_file) as f:
        old_cache = json.load(f)
    print(f"Old cache entries (may contain DailyDialog): {len(old_cache)}", flush=True)

# Process each dataset
import openai
client = openai.OpenAI()
cost_tracker = CostTracker()
start_time = time.time()

all_results = {}

for dataset_info in DATASETS:
    dataset_name = dataset_info["name"]
    dataset_file = dataset_info["file"]
    prefix = dataset_info["prefix"]

    print(f"\n{'=' * 60}", flush=True)
    print(f"Processing {dataset_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if not dataset_file.exists():
        print(f"  Skipping: file not found", flush=True)
        continue

    # Load dataset
    dialogues = load_dataset(dataset_file)
    total_boundaries = sum(d["num_user_turns"] - 1 for d in dialogues)
    print(f"  Dialogues: {len(dialogues)}", flush=True)
    print(f"  Total boundaries: {total_boundaries}", flush=True)

    # Build gold by dialogue
    gold_by_dialogue = {d["dialogue_id"]: d["gold_boundaries"] for d in dialogues}

    # Identify uncached boundaries
    boundaries_to_score = []
    scores_by_dialogue = defaultdict(dict)

    for dialogue in dialogues:
        user_turns = [m["content"] for m in dialogue["messages"] if m["role"] == "user"]

        for position in range(1, dialogue["num_user_turns"]):
            context_before = []
            for i in range(max(0, position - 4), position):
                context_before.append(user_turns[i])
            context_after = user_turns[position] if position < len(user_turns) else ""

            user_prompt = build_user_prompt(context_before, context_after)
            prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)
            cache_key = get_cache_key(prefix, dialogue["dialogue_id"], position, prompt_hash)

            # Check cache (new and old)
            if cache_key in cache:
                entry = cache[cache_key]
                if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                    scores_by_dialogue[dialogue["dialogue_id"]][position] = entry["score"]
            elif prefix == "dailydialog" and cache_key in old_cache:
                # Flip score from old cache (it used wrong polarity)
                entry = old_cache[cache_key]
                if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                    flipped_score = -entry["score"]
                    scores_by_dialogue[dialogue["dialogue_id"]][position] = flipped_score
                    # Save to new cache with correct polarity
                    cache[cache_key] = {**entry, "score": flipped_score}
            else:
                boundaries_to_score.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "position": position,
                    "context_before": context_before,
                    "context_after": context_after,
                    "cache_key": cache_key,
                })

    print(f"  Already cached: {total_boundaries - len(boundaries_to_score)}", flush=True)
    print(f"  Need to score: {len(boundaries_to_score)}", flush=True)

    if boundaries_to_score:
        # Score uncached boundaries
        for i, boundary in enumerate(boundaries_to_score):
            user_prompt = build_user_prompt(boundary["context_before"], boundary["context_after"])

            try:
                response = client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    top_p=1,
                    max_completion_tokens=10,
                    logprobs=True,
                    top_logprobs=TOP_LOGPROBS,
                    reasoning_effort="none",
                )

                choice = response.choices[0]
                raw_token = choice.message.content.strip() if choice.message.content else ""

                logprob_y = None
                logprob_n = None

                if choice.logprobs and choice.logprobs.content:
                    first_token_logprobs = choice.logprobs.content[0].top_logprobs
                    for lp in first_token_logprobs:
                        token = lp.token.strip().upper()
                        if token == "Y":
                            logprob_y = lp.logprob
                        elif token == "N":
                            logprob_n = lp.logprob

                missing_yn = logprob_y is None or logprob_n is None

                # FLIPPED polarity
                if missing_yn:
                    score = 0.0
                else:
                    score = logprob_n - logprob_y

                cache[boundary["cache_key"]] = {
                    "dialogue_id": boundary["dialogue_id"],
                    "position": boundary["position"],
                    "score": score,
                    "raw_token": raw_token,
                    "logprob_y": logprob_y if logprob_y else 0.0,
                    "logprob_n": logprob_n if logprob_n else 0.0,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "missing_yn_in_toplogprobs": missing_yn,
                    "invalid_first_token": raw_token.strip().upper()[:1] not in ("Y", "N"),
                }

                if not missing_yn and raw_token.strip().upper()[:1] in ("Y", "N"):
                    scores_by_dialogue[boundary["dialogue_id"]][boundary["position"]] = score

                cost_tracker.num_calls += 1
                cost_tracker.total_input_tokens += response.usage.prompt_tokens
                cost_tracker.total_output_tokens += response.usage.completion_tokens

            except Exception as e:
                print(f"  API error at {i}: {e}", flush=True)
                continue

            # Progress
            if (i + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = cost_tracker.num_calls / (elapsed / 60) if elapsed > 0 else 0
                remaining = len(boundaries_to_score) - (i + 1)
                eta_minutes = remaining / rate if rate > 0 else 0
                print(f"  [{dataset_name}] {i+1}/{len(boundaries_to_score)} | ${cost_tracker.total_cost_usd:.2f} | {rate:.0f}/min | ETA: {eta_minutes:.0f}min", flush=True)

            # Save cache periodically
            if (i + 1) % SAVE_INTERVAL == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)

            # Budget check
            if cost_tracker.total_cost_usd > HARD_BUDGET_USD:
                print(f"\n  [ABORT] Budget exceeded: ${cost_tracker.total_cost_usd:.2f} > ${HARD_BUDGET_USD:.2f}", flush=True)
                break

        # Final save
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

    # Generate Figure 4 panel
    result = generate_figure4_panel(dataset_name, dict(scores_by_dialogue), gold_by_dialogue)
    if result:
        all_results[dataset_name] = result

    # Check budget
    if cost_tracker.total_cost_usd > HARD_BUDGET_USD:
        break

# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time
print("\n" + "=" * 60, flush=True)
print("SCORING COMPLETE", flush=True)
print(f"Total API calls: {cost_tracker.num_calls}", flush=True)
print(f"Total cost: ${cost_tracker.total_cost_usd:.2f}", flush=True)
print(f"Total time: {elapsed/60:.1f} min", flush=True)
print("=" * 60, flush=True)

print("\nResults Summary:", flush=True)
print("-" * 60, flush=True)
for name, result in all_results.items():
    print(f"  {name}:", flush=True)
    print(f"    Peak W-F1 (m2o): {result['peak_wf1_m2o']:.4f} at BOR={result['peak_bor_m2o']:.3f}", flush=True)
    print(f"    Peak W-F1 (1to1): {result['peak_wf1_1to1']:.4f} at BOR={result['peak_bor_1to1']:.3f}", flush=True)
