#!/usr/bin/env python3
"""
GPT-5.2 Full-Scale Scoring for Figure 4 Replication

Scores full test sets for DialSeg711 and SuperSeg with bootstrap CIs.
Reuses cached scores from sanity check and validation runs.

Usage:
    python paper/experiments/gpt52_figure4_replication.py
    python paper/experiments/gpt52_figure4_replication.py --dry-run
"""

import os
import sys

# Disable output buffering for progress visibility
sys.stdout.reconfigure(line_buffering=True)

import json
import hashlib
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPT-5.2 pricing
GPT52_INPUT_PRICE_PER_1M = 1.75
GPT52_OUTPUT_PRICE_PER_1M = 14.0

# Budget
HARD_BUDGET_USD = 10.0
COST_CHECK_INTERVAL = 500

# Top logprobs
TOP_LOGPROBS = 5

# Greedy NMS min gap
MIN_GAP = 2

# Bootstrap
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# Tau sweep (using percentiles for more stable BOR coverage)
TAU_PERCENTILES = list(range(1, 100, 3))  # 1, 4, 7, ..., 97

# Paths
# NOTE: The sanity check cache (.gpt52_cache) uses ORIGINAL score formula (logP(Y) - logP(N))
# The superseg cache (.gpt52_superseg_cache) uses FLIPPED formula (logP(N) - logP(Y))
# We need to flip scores from the sanity check cache when loading them.
CACHE_DIRS_WITH_FLIP = [
    (PROJECT_ROOT / ".gpt52_cache", True),            # Sanity check - NEEDS FLIP
    (PROJECT_ROOT / ".gpt52_superseg_cache", False),  # SuperSeg - already flipped
]
OUTPUT_CACHE_DIR = PROJECT_ROOT / ".gpt52_figure4_cache"
OUTPUT_FILE = PROJECT_ROOT / "paper" / "experiments" / "gpt52_figure4_replication.json"
OUTPUT_FIGURE = PROJECT_ROOT / "paper" / "experiments" / "gpt52_figure4_replication.png"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments" / "gpt52_figure4_curves"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Existing curves for overlay
EXISTING_CURVES = {
    "dialseg711": {
        "neural": PROJECT_ROOT / "paper" / "results" / "sweep_dialseg711_neural_per_dialogue.json",
        "texttiling": PROJECT_ROOT / "paper" / "results" / "sweep_dialseg711_texttiling_per_dialogue.json",
        "csm": PROJECT_ROOT / "paper" / "results" / "sweep_dialseg711_csm_per_dialogue.json",
    },
    "superseg": {
        "neural": PROJECT_ROOT / "paper" / "results" / "sweep_superseg_neural_per_dialogue.json",
        "texttiling": PROJECT_ROOT / "paper" / "results" / "sweep_superseg_texttiling_per_dialogue.json",
        "csm": PROJECT_ROOT / "paper" / "results" / "sweep_superseg_csm_per_dialogue.json",
    }
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

@dataclass
class BoundaryScore:
    dialogue_id: int
    position: int
    score: float
    raw_token: str
    logprob_y: float
    logprob_n: float
    input_tokens: int
    output_tokens: int
    missing_yn_in_toplogprobs: bool = False
    invalid_first_token: bool = False
    retried: bool = False

@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_calls: int = 0
    num_cache_hits: int = 0
    input_tokens_list: List[int] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * GPT52_INPUT_PRICE_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * GPT52_OUTPUT_PRICE_PER_1M
        return input_cost + output_cost

    @property
    def cost_per_call(self) -> float:
        return self.total_cost_usd / self.num_calls if self.num_calls > 0 else 0.0

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = """You are a discourse segmentation expert. A segment boundary occurs when the conversation shifts to a new topic, task, or phase. Return ONLY 'Y' or 'N' (single token)."""

def build_user_prompt(context_before: List[str], context_after: str) -> str:
    context_lines = []
    for i, turn in enumerate(context_before[-4:]):
        context_lines.append(f"[{i+1}] {turn}")
    context_str = "\n".join(context_lines)

    prompt = f"""Context before boundary:
{context_str}

--- CANDIDATE BOUNDARY ---

Next turn:
{context_after}

Is this a topic boundary? Decision:"""
    return prompt

def build_retry_prompt(original_prompt: str) -> str:
    return original_prompt + "\n\nIMPORTANT: Your response must be exactly 'Y' or 'N', nothing else."

# =============================================================================
# CACHING
# =============================================================================

def get_cache_key(dataset: str, dialogue_id: int, position: int, prompt_hash: str) -> str:
    return f"{dataset}_{dialogue_id}_{position}_{prompt_hash}"

def compute_prompt_hash(system: str, user: str) -> str:
    content = f"{system}|{user}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

class MultiSourceCache:
    """Cache that reads from multiple directories and writes to output cache.

    Handles score polarity flipping for caches that used the original formula.
    """

    def __init__(self, read_dirs_with_flip: List[Tuple[Path, bool]], write_dir: Path):
        self.read_dirs_with_flip = read_dirs_with_flip
        self.write_dir = write_dir
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self._caches = self._load_all_caches()
        self._write_cache = {}
        self._write_file = write_dir / "cache.json"
        self._load_write_cache()

    def _load_all_caches(self) -> Dict[str, Dict]:
        caches = {}
        for cache_dir, needs_flip in self.read_dirs_with_flip:
            cache_file = cache_dir / "cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        data = json.load(f)

                    # Flip scores if this cache used the original formula
                    if needs_flip:
                        flipped_count = 0
                        for key, entry in data.items():
                            if 'score' in entry and entry['score'] != 0:
                                entry['score'] = -entry['score']
                                flipped_count += 1
                        print(f"  Loaded {len(data)} entries from {cache_file} (FLIPPED {flipped_count} scores)")
                    else:
                        print(f"  Loaded {len(data)} entries from {cache_file}")

                    caches.update(data)
                except Exception as e:
                    print(f"  Warning: Failed to load {cache_file}: {e}")
        return caches

    def _load_write_cache(self):
        if self._write_file.exists():
            try:
                with open(self._write_file) as f:
                    self._write_cache = json.load(f)
                print(f"  Loaded {len(self._write_cache)} entries from write cache")
            except Exception:
                self._write_cache = {}

    def _save_write_cache(self):
        with open(self._write_file, 'w') as f:
            json.dump(self._write_cache, f)

    def get(self, key: str) -> Optional[Dict]:
        # Check write cache first
        if key in self._write_cache:
            return self._write_cache[key]
        # Then check read caches
        return self._caches.get(key)

    def set(self, key: str, value: Dict):
        self._write_cache[key] = value
        # Save periodically
        if len(self._write_cache) % 100 == 0:
            self._save_write_cache()

    def save(self):
        self._save_write_cache()

    def size(self) -> Tuple[int, int]:
        """Return (read_cache_size, write_cache_size)"""
        return len(self._caches), len(self._write_cache)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name: str) -> List[DialogueData]:
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

# =============================================================================
# GPT-5.2 SCORING
# =============================================================================

def score_boundary_gpt52(
    client,
    dialogue: DialogueData,
    position: int,
    cache: MultiSourceCache,
    dataset_name: str,
    dry_run: bool = False
) -> Tuple[BoundaryScore, bool]:
    """
    Score a boundary. Returns (score, was_cached).
    Uses FLIPPED polarity: s_i = logP(N) - logP(Y)
    """
    user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]

    context_before = []
    for i in range(max(0, position - 4), position):
        context_before.append(user_turns[i])

    context_after = user_turns[position] if position < len(user_turns) else ""

    user_prompt = build_user_prompt(context_before, context_after)
    prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)
    cache_key = get_cache_key(dataset_name, dialogue.dialogue_id, position, prompt_hash)

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        # Need to recompute score with flipped polarity if old cache used wrong polarity
        logprob_y = cached.get("logprob_y", 0)
        logprob_n = cached.get("logprob_n", 0)
        flipped_score = logprob_n - logprob_y  # FLIPPED

        return BoundaryScore(
            dialogue_id=dialogue.dialogue_id,
            position=position,
            score=flipped_score,
            raw_token=cached.get("raw_token", ""),
            logprob_y=logprob_y,
            logprob_n=logprob_n,
            input_tokens=cached.get("input_tokens", 0),
            output_tokens=cached.get("output_tokens", 0),
            missing_yn_in_toplogprobs=cached.get("missing_yn_in_toplogprobs", False),
            invalid_first_token=cached.get("invalid_first_token", False),
            retried=cached.get("retried", False)
        ), True

    # Dry run
    if dry_run:
        return BoundaryScore(
            dialogue_id=dialogue.dialogue_id,
            position=position,
            score=0.0,
            raw_token="Y",
            logprob_y=-0.5,
            logprob_n=-1.5,
            input_tokens=len(user_prompt.split()),
            output_tokens=1,
        ), False

    # API call
    def make_call(prompt: str, is_retry: bool = False) -> Dict:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
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

        return {
            "raw_token": raw_token,
            "logprob_y": logprob_y,
            "logprob_n": logprob_n,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    result = make_call(user_prompt)
    first_token = result["raw_token"].strip().upper()[:1]

    invalid_first_token = first_token not in ("Y", "N")
    retried = False

    if invalid_first_token:
        retry_prompt = build_retry_prompt(user_prompt)
        result = make_call(retry_prompt, is_retry=True)
        first_token = result["raw_token"].strip().upper()[:1]
        retried = True
        invalid_first_token = first_token not in ("Y", "N")

    missing_yn = result["logprob_y"] is None or result["logprob_n"] is None

    if missing_yn or invalid_first_token:
        score = 0.0
    else:
        score = result["logprob_n"] - result["logprob_y"]  # FLIPPED

    boundary_score = BoundaryScore(
        dialogue_id=dialogue.dialogue_id,
        position=position,
        score=score,
        raw_token=result["raw_token"],
        logprob_y=result["logprob_y"] if result["logprob_y"] is not None else 0.0,
        logprob_n=result["logprob_n"] if result["logprob_n"] is not None else 0.0,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        missing_yn_in_toplogprobs=missing_yn,
        invalid_first_token=invalid_first_token,
        retried=retried
    )

    # Cache
    cache.set(cache_key, asdict(boundary_score))

    return boundary_score, False

def score_dataset(
    dataset_name: str,
    dialogues: List[DialogueData],
    cache: MultiSourceCache,
    cost_tracker: CostTracker,
    dry_run: bool = False
) -> List[BoundaryScore]:
    """Score all boundaries in a dataset."""
    import openai
    client = openai.OpenAI() if not dry_run else None

    all_scores = []
    total_boundaries = sum(d.num_user_turns - 1 for d in dialogues)
    start_time = time.time()
    last_report_time = start_time

    print(f"\n  Scoring {dataset_name}: {len(dialogues)} dialogues, {total_boundaries} boundaries")

    for i, dialogue in enumerate(dialogues):
        for position in range(1, dialogue.num_user_turns):
            result, was_cached = score_boundary_gpt52(
                client, dialogue, position, cache, dataset_name, dry_run
            )

            if was_cached:
                cost_tracker.num_cache_hits += 1
            else:
                cost_tracker.num_calls += 1
                cost_tracker.total_input_tokens += result.input_tokens
                cost_tracker.total_output_tokens += result.output_tokens
                cost_tracker.input_tokens_list.append(result.input_tokens)

            all_scores.append(result)

            # Progress check every 100 boundaries or every 30 seconds
            total_processed = len(all_scores)
            current_time = time.time()
            elapsed = current_time - start_time

            if total_processed % 100 == 0 or (current_time - last_report_time) >= 30:
                last_report_time = current_time
                remaining = total_boundaries - total_processed
                rate = total_processed / (elapsed / 60) if elapsed > 0 else 0
                eta_minutes = remaining / rate if rate > 0 else 0

                if cost_tracker.num_calls > 0:
                    projected = cost_tracker.total_cost_usd + (remaining * cost_tracker.cost_per_call)
                    print(f"    [{dataset_name}] {total_processed}/{total_boundaries} | "
                          f"${cost_tracker.total_cost_usd:.2f} | "
                          f"{rate:.0f}/min | ETA: {eta_minutes:.0f}min")

                    if cost_tracker.total_cost_usd > HARD_BUDGET_USD:
                        print(f"\n[ABORT] Budget exceeded: ${cost_tracker.total_cost_usd:.2f}")
                        cache.save()
                        return all_scores
                elif total_processed % 500 == 0:
                    # Cache hits only - still report progress
                    print(f"    [{dataset_name}] {total_processed}/{total_boundaries} (from cache)")

        # Dialogue milestone
        if (i + 1) % 100 == 0:
            print(f"    Dialogues: {i + 1}/{len(dialogues)}")

    cache.save()
    return all_scores

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float, min_gap: int) -> Set[int]:
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    predicted = set()
    for pos, score in candidates:
        too_close = any(abs(pos - p) < min_gap for p in predicted)
        if not too_close:
            predicted.add(pos)

    return predicted

def compute_wf1_many_to_one(predicted: Set[int], gold: Set[int], k: int = 3) -> float:
    """Compute W-F1 with many-to-one matching (windowed)."""
    if not gold:
        return 0.0 if predicted else 1.0

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
    fn = len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_wf1_one_to_one(predicted: Set[int], gold: Set[int], k: int = 3) -> float:
    """Compute W-F1 with one-to-one matching (stricter)."""
    if not gold:
        return 0.0 if predicted else 1.0

    # Create match matrix
    pred_list = sorted(predicted)
    gold_list = sorted(gold)

    # Greedy one-to-one matching
    matched_pred = set()
    matched_gold = set()

    # Sort by distance to find best matches
    pairs = []
    for p in pred_list:
        for g in gold_list:
            dist = abs(p - g)
            if dist <= k:
                pairs.append((dist, p, g))

    pairs.sort()

    for dist, p, g in pairs:
        if p not in matched_pred and g not in matched_gold:
            matched_pred.add(p)
            matched_gold.add(g)

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_purity_coverage(predicted: Set[int], gold: Set[int]) -> Tuple[float, float]:
    """Compute purity and coverage."""
    if not gold:
        return (0.0, 0.0) if predicted else (1.0, 1.0)
    if not predicted:
        return 0.0, 0.0

    # Match predicted to closest gold
    matched = 0
    for p in predicted:
        if any(abs(p - g) <= 3 for g in gold):
            matched += 1

    purity = matched / len(predicted) if predicted else 0.0

    # Match gold to closest predicted
    covered = 0
    for g in gold:
        if any(abs(g - p) <= 3 for p in predicted):
            covered += 1

    coverage = covered / len(gold) if gold else 0.0

    return purity, coverage

def compute_sweep_for_dialogues(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    tau_percentiles: List[int]
) -> List[Dict]:
    """Compute threshold sweep metrics."""
    all_scores = []
    for dialogue_id, scores_by_pos in scores_by_dialogue.items():
        all_scores.extend(scores_by_pos.values())

    if not all_scores:
        return []

    all_scores = np.array(all_scores)
    sweep_points = []

    for pct in tau_percentiles:
        tau = np.percentile(all_scores, pct)

        total_wf1_m2o = 0.0
        total_wf1_1to1 = 0.0
        total_purity = 0.0
        total_coverage = 0.0
        total_pred = 0
        total_gold = 0
        n_dialogues = 0
        per_dialogue_wf1_m2o = {}
        per_dialogue_wf1_1to1 = {}

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            if not gold:
                continue

            predicted = greedy_nms_predict(scores_by_pos, tau, MIN_GAP)

            wf1_m2o = compute_wf1_many_to_one(predicted, gold)
            wf1_1to1 = compute_wf1_one_to_one(predicted, gold)
            purity, coverage = compute_purity_coverage(predicted, gold)

            per_dialogue_wf1_m2o[dialogue_id] = wf1_m2o
            per_dialogue_wf1_1to1[dialogue_id] = wf1_1to1
            total_wf1_m2o += wf1_m2o
            total_wf1_1to1 += wf1_1to1
            total_purity += purity
            total_coverage += coverage
            total_pred += len(predicted)
            total_gold += len(gold)
            n_dialogues += 1

        if n_dialogues == 0:
            continue

        bor = total_pred / total_gold if total_gold > 0 else 0

        sweep_points.append({
            "percentile": pct,
            "tau": float(tau),
            "bor": bor,
            "wf1_m2o": total_wf1_m2o / n_dialogues,
            "wf1_1to1": total_wf1_1to1 / n_dialogues,
            "purity": total_purity / n_dialogues,
            "coverage": total_coverage / n_dialogues,
            "n_dialogues": n_dialogues,
            "per_dialogue_wf1_m2o": per_dialogue_wf1_m2o,
            "per_dialogue_wf1_1to1": per_dialogue_wf1_1to1,
        })

    return sweep_points

def compute_bootstrap_ci(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    tau_percentiles: List[int],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Compute bootstrap CIs for the sweep curves."""
    rng = np.random.RandomState(seed)
    dialogue_ids = list(scores_by_dialogue.keys())
    n_dialogues = len(dialogue_ids)

    # Storage for bootstrap samples at each tau
    bootstrap_bors = defaultdict(list)
    bootstrap_wf1_m2o = defaultdict(list)
    bootstrap_wf1_1to1 = defaultdict(list)

    print(f"    Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"      Bootstrap: {b + 1}/{n_bootstrap}")

        # Resample dialogues with replacement
        sampled_ids = rng.choice(dialogue_ids, size=n_dialogues, replace=True)

        # Build resampled data
        resampled_scores = {}
        resampled_gold = {}
        for new_id, orig_id in enumerate(sampled_ids):
            resampled_scores[new_id] = scores_by_dialogue[orig_id]
            resampled_gold[new_id] = gold_by_dialogue[orig_id]

        # Compute sweep for this resample
        sweep = compute_sweep_for_dialogues(resampled_scores, resampled_gold, tau_percentiles)

        for sp in sweep:
            pct = sp["percentile"]
            bootstrap_bors[pct].append(sp["bor"])
            bootstrap_wf1_m2o[pct].append(sp["wf1_m2o"])
            bootstrap_wf1_1to1[pct].append(sp["wf1_1to1"])

    # Compute CIs
    ci_results = {}
    for pct in tau_percentiles:
        if pct not in bootstrap_bors:
            continue

        bors = np.array(bootstrap_bors[pct])
        wf1_m2o = np.array(bootstrap_wf1_m2o[pct])
        wf1_1to1 = np.array(bootstrap_wf1_1to1[pct])

        ci_results[pct] = {
            "bor_mean": float(np.mean(bors)),
            "bor_lo": float(np.percentile(bors, 2.5)),
            "bor_hi": float(np.percentile(bors, 97.5)),
            "wf1_m2o_mean": float(np.mean(wf1_m2o)),
            "wf1_m2o_lo": float(np.percentile(wf1_m2o, 2.5)),
            "wf1_m2o_hi": float(np.percentile(wf1_m2o, 97.5)),
            "wf1_1to1_mean": float(np.mean(wf1_1to1)),
            "wf1_1to1_lo": float(np.percentile(wf1_1to1, 2.5)),
            "wf1_1to1_hi": float(np.percentile(wf1_1to1, 97.5)),
        }

    return ci_results

# =============================================================================
# PLOTTING
# =============================================================================

def load_existing_curves(dataset_name: str) -> Dict[str, List[Dict]]:
    """Load existing method curves for overlay."""
    curves = {}
    paths = EXISTING_CURVES.get(dataset_name, {})

    for method, path in paths.items():
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                curves[method] = data.get("points", [])
            except Exception as e:
                print(f"  Warning: Failed to load {method} curve: {e}")

    return curves

def create_figure4(results: Dict):
    """Create 4-panel Figure 4 replica."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[ERROR] matplotlib not available")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    datasets = ["dialseg711", "superseg"]
    wf1_types = ["m2o", "1to1"]
    wf1_labels = ["W-F1 (many-to-one)", "W-F1 (one-to-one)"]

    colors = {
        "gpt52": "#8B5CF6",  # Purple
        "neural": "#2563EB",  # Blue
        "texttiling": "#10B981",  # Green
        "csm": "#F59E0B",  # Orange
    }

    for row, dataset in enumerate(datasets):
        dataset_results = results["datasets"].get(dataset, {})
        sweep_points = dataset_results.get("sweep_points", [])
        ci_data = dataset_results.get("bootstrap_ci", {})

        # Load existing curves
        existing = load_existing_curves(dataset)

        for col, (wf1_type, wf1_label) in enumerate(zip(wf1_types, wf1_labels)):
            ax = axes[row, col]

            # Extract GPT-5.2 curve
            bors = [sp["bor"] for sp in sweep_points]
            wf1s = [sp[f"wf1_{wf1_type}"] for sp in sweep_points]

            # Sort by BOR
            idx = np.argsort(bors)
            bors = np.array(bors)[idx]
            wf1s = np.array(wf1s)[idx]

            # Plot GPT-5.2 curve
            ax.plot(bors, wf1s, '-', color=colors["gpt52"], linewidth=2.5,
                    label='GPT-5.2', marker='o', markersize=3)

            # Plot CI band if available
            if ci_data:
                ci_bors = []
                ci_lo = []
                ci_hi = []
                for sp in sweep_points:
                    pct = sp["percentile"]
                    if pct in ci_data:
                        ci_bors.append(sp["bor"])
                        ci_lo.append(ci_data[pct][f"wf1_{wf1_type}_lo"])
                        ci_hi.append(ci_data[pct][f"wf1_{wf1_type}_hi"])

                if ci_bors:
                    idx = np.argsort(ci_bors)
                    ci_bors = np.array(ci_bors)[idx]
                    ci_lo = np.array(ci_lo)[idx]
                    ci_hi = np.array(ci_hi)[idx]
                    ax.fill_between(ci_bors, ci_lo, ci_hi, color=colors["gpt52"], alpha=0.2)

            # Plot existing methods
            for method, curve in existing.items():
                if not curve:
                    continue
                ex_bors = [p["bor"] for p in curve]
                ex_wf1s = [p.get("wf1", p.get("wf1_m2o", 0)) for p in curve] if wf1_type == "m2o" else [p.get("wf1_1to1", p.get("wf1", 0)) for p in curve]

                # Subsample for clarity
                step = max(1, len(ex_bors) // 30)
                ex_bors = np.array(ex_bors)[::step]
                ex_wf1s = np.array(ex_wf1s)[::step]

                ax.plot(ex_bors, ex_wf1s, '--', color=colors.get(method, 'gray'),
                        linewidth=1.5, alpha=0.7, label=method.replace("_", " ").title())

            # BOR=1 line and shading
            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)

            # Shading for BOR regions
            ax.axvspan(0, 1.0, alpha=0.05, color='blue')
            ax.axvspan(1.0, ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 3, alpha=0.05, color='red')

            ax.set_xlabel('BOR')
            ax.set_ylabel(wf1_label)
            ax.set_title(f'{dataset.upper()}: {wf1_label}')
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)

    fig.suptitle('GPT-5.2 Boundary Scoring: Figure 4 Replication', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close(fig)

    print(f"\n  Figure saved to: {OUTPUT_FIGURE}")

    # Save individual panels
    for row, dataset in enumerate(datasets):
        for col, (wf1_type, wf1_label) in enumerate(zip(wf1_types, wf1_labels)):
            fig, ax = plt.subplots(figsize=(8, 6))

            dataset_results = results["datasets"].get(dataset, {})
            sweep_points = dataset_results.get("sweep_points", [])

            bors = [sp["bor"] for sp in sweep_points]
            wf1s = [sp[f"wf1_{wf1_type}"] for sp in sweep_points]

            idx = np.argsort(bors)
            bors = np.array(bors)[idx]
            wf1s = np.array(wf1s)[idx]

            ax.plot(bors, wf1s, '-', color=colors["gpt52"], linewidth=2.5,
                    label='GPT-5.2', marker='o', markersize=4)

            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel('BOR')
            ax.set_ylabel(wf1_label)
            ax.set_title(f'{dataset.upper()}: {wf1_label}')
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

            panel_path = OUTPUT_DIR / f"{dataset}_{wf1_type}.png"
            fig.tight_layout()
            fig.savefig(panel_path, dpi=150)
            plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-5.2 Figure 4 Replication")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap CI computation")
    args = parser.parse_args()

    print("=" * 70)
    print("GPT-5.2 FULL-SCALE SCORING FOR FIGURE 4 REPLICATION")
    print("=" * 70)
    print(f"Budget cap: ${HARD_BUDGET_USD:.2f}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Score formula: s_i = logP(N) - logP(Y) [FLIPPED]")

    # Load caches
    print("\nLoading caches...")
    cache = MultiSourceCache(CACHE_DIRS_WITH_FLIP, OUTPUT_CACHE_DIR)
    read_size, write_size = cache.size()
    total_cached = read_size + write_size
    print(f"  Total cached entries: {total_cached}")

    # Load datasets and estimate work
    print("\nEstimating work...")
    dial_dialogues = load_dataset("dialseg711")
    super_dialogues = load_dataset("superseg")
    dial_boundaries = sum(d.num_user_turns - 1 for d in dial_dialogues)
    super_boundaries = sum(d.num_user_turns - 1 for d in super_dialogues)
    total_boundaries = dial_boundaries + super_boundaries
    new_api_calls = max(0, total_boundaries - total_cached)

    # Estimate cost (assuming ~200 tokens input, 5 tokens output per call)
    est_cost_per_call = (200 / 1_000_000) * GPT52_INPUT_PRICE_PER_1M + (5 / 1_000_000) * GPT52_OUTPUT_PRICE_PER_1M
    est_total_cost = new_api_calls * est_cost_per_call

    # Estimate time (assuming ~30 calls per minute due to API rate limits)
    calls_per_minute = 30
    est_minutes = new_api_calls / calls_per_minute

    print(f"  DialSeg711: {len(dial_dialogues)} dialogues, {dial_boundaries} boundaries")
    print(f"  SuperSeg: {len(super_dialogues)} dialogues, {super_boundaries} boundaries")
    print(f"  Total boundaries to score: {total_boundaries}")
    print(f"  Already cached: {total_cached}")
    print(f"  New API calls needed: ~{new_api_calls}")
    print(f"  Estimated cost: ~${est_total_cost:.2f}")
    print(f"  Estimated time: ~{est_minutes:.0f} min ({est_minutes/60:.1f}h) at {calls_per_minute} calls/min")

    # Validate model
    if not args.dry_run:
        print("\nValidating model access...")
        import openai
        try:
            client = openai.OpenAI()
            client.models.retrieve("gpt-5.2")
            print("  [OK] Model gpt-5.2 accessible")
        except Exception as e:
            print(f"  [ERROR] {e}")
            return 1

    # Cost tracker
    cost_tracker = CostTracker()

    # Results storage
    results = {
        "description": "GPT-5.2 Figure 4 replication with bootstrap CIs",
        "score_formula": "s_i = logP(N) - logP(Y) [FLIPPED]",
        "n_bootstrap": N_BOOTSTRAP,
        "datasets": {}
    }

    # Process each dataset
    for dataset_name in ["dialseg711", "superseg"]:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 70)

        # Load dataset
        dialogues = load_dataset(dataset_name)
        print(f"  Loaded {len(dialogues)} dialogues")
        print(f"  Total gold boundaries: {sum(len(d.gold_boundaries) for d in dialogues)}")

        # Score all boundaries
        all_scores = score_dataset(dataset_name, dialogues, cache, cost_tracker, args.dry_run)
        print(f"  Total scores: {len(all_scores)}")
        print(f"  Cache hits: {cost_tracker.num_cache_hits}")
        print(f"  API calls: {cost_tracker.num_calls}")
        print(f"  Cost so far: ${cost_tracker.total_cost_usd:.4f}")

        # Build score dictionaries
        scores_by_dialogue = defaultdict(dict)
        gold_by_dialogue = {}

        for dialogue in dialogues:
            gold_by_dialogue[dialogue.dialogue_id] = dialogue.gold_boundaries

        for score in all_scores:
            if score.missing_yn_in_toplogprobs or score.invalid_first_token:
                continue
            scores_by_dialogue[score.dialogue_id][score.position] = score.score

        # Compute sweep
        print("\n  Computing threshold sweep...")
        sweep_points = compute_sweep_for_dialogues(
            dict(scores_by_dialogue), gold_by_dialogue, TAU_PERCENTILES
        )

        # Find peak
        if sweep_points:
            best_m2o = max(sweep_points, key=lambda x: x["wf1_m2o"])
            best_1to1 = max(sweep_points, key=lambda x: x["wf1_1to1"])
            print(f"  Peak W-F1 (m2o): {best_m2o['wf1_m2o']:.4f} at BOR={best_m2o['bor']:.3f}")
            print(f"  Peak W-F1 (1to1): {best_1to1['wf1_1to1']:.4f} at BOR={best_1to1['bor']:.3f}")

        # Bootstrap CIs
        bootstrap_ci = {}
        if not args.skip_bootstrap and not args.dry_run:
            print("\n  Computing bootstrap CIs...")
            bootstrap_ci = compute_bootstrap_ci(
                dict(scores_by_dialogue), gold_by_dialogue,
                TAU_PERCENTILES, N_BOOTSTRAP, BOOTSTRAP_SEED
            )

        # Store results
        results["datasets"][dataset_name] = {
            "n_dialogues": len(dialogues),
            "n_boundaries": len(all_scores),
            "n_valid": len([s for s in all_scores if not s.missing_yn_in_toplogprobs and not s.invalid_first_token]),
            "peak_wf1_m2o": best_m2o["wf1_m2o"] if sweep_points else 0,
            "peak_bor_m2o": best_m2o["bor"] if sweep_points else 0,
            "peak_wf1_1to1": best_1to1["wf1_1to1"] if sweep_points else 0,
            "peak_bor_1to1": best_1to1["bor"] if sweep_points else 0,
            "sweep_points": [{k: v for k, v in sp.items()
                            if not k.startswith("per_dialogue")} for sp in sweep_points],
            "bootstrap_ci": bootstrap_ci,
        }

    # Final cost
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total API calls: {cost_tracker.num_calls}")
    print(f"  Total cache hits: {cost_tracker.num_cache_hits}")
    print(f"  Total cost: ${cost_tracker.total_cost_usd:.4f}")

    results["total_cost_usd"] = cost_tracker.total_cost_usd
    results["total_api_calls"] = cost_tracker.num_calls
    results["total_cache_hits"] = cost_tracker.num_cache_hits

    # Save results
    print(f"\nSaving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Create figure
    print("\nCreating Figure 4...")
    create_figure4(results)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
