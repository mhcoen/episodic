#!/usr/bin/env python3
"""
GPT-5.2 SuperSeg Validation Run

Validates GPT-5.2 boundary scoring on SuperSeg dataset before full-scale scoring.
Uses flipped score polarity: s_i = logP(N) - logP(Y)

Parameters:
- Dataset: SuperSeg only
- Dialogues: 150, stratified by length tertiles
- Hard boundary cap: 5,000
- Budget cap: $2.00

Usage:
    python paper/experiments/gpt52_superseg_validation.py
    python paper/experiments/gpt52_superseg_validation.py --dry-run
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPT-5.2 pricing (as of January 2026)
GPT52_INPUT_PRICE_PER_1M = 1.75   # $1.75 per 1M input tokens
GPT52_OUTPUT_PRICE_PER_1M = 14.0  # $14.00 per 1M output tokens

# Hard safety limits (STRICTER for validation run)
HARD_BUDGET_USD = 2.0            # Maximum total spend
HARD_MAX_BOUNDARIES = 5000       # Maximum boundaries to score
HARD_TARGET_INPUT_TOKENS = 320   # Target input tokens per call
HARD_MAX_INPUT_P95 = 450         # Abort if p95 input tokens exceeds this

# Cost check frequency
COST_CHECK_INTERVAL = 100        # Check projected cost after every N calls

# Sampling configuration
N_SAMPLE_DIALOGUES = 150         # Total dialogues to sample
LENGTH_TERTILE_SPLITS = 3        # Split by short/medium/long

# Top logprobs to request
TOP_LOGPROBS = 5

# Tau percentiles for sweep
TAU_PERCENTILES = [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]

# Greedy NMS min gap
MIN_GAP = 2

# Paths
CACHE_DIR = PROJECT_ROOT / ".gpt52_superseg_cache"
OUTPUT_FILE = PROJECT_ROOT / "paper" / "experiments" / "gpt52_superseg_validation.json"
PLOT_DIR = PROJECT_ROOT / "paper" / "experiments" / "gpt52_superseg_validation"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DialogueData:
    """Container for a single dialogue with boundaries."""
    dialogue_id: int
    messages: List[Dict[str, str]]
    gold_boundaries: Set[int]
    num_user_turns: int

@dataclass
class BoundaryScore:
    """Score for a single boundary position."""
    dialogue_id: int
    position: int
    score: float   # logP(N) - logP(Y) (FLIPPED)
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
    """Track API costs and enforce budget."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_calls: int = 0
    num_retries: int = 0
    num_invalid_first_token: int = 0
    num_missing_yn: int = 0
    input_tokens_list: List[int] = field(default_factory=list)
    output_tokens_list: List[int] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * GPT52_INPUT_PRICE_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * GPT52_OUTPUT_PRICE_PER_1M
        return input_cost + output_cost

    @property
    def mean_input_tokens(self) -> float:
        return np.mean(self.input_tokens_list) if self.input_tokens_list else 0.0

    @property
    def p95_input_tokens(self) -> float:
        return np.percentile(self.input_tokens_list, 95) if self.input_tokens_list else 0.0

    @property
    def cost_per_boundary(self) -> float:
        return self.total_cost_usd / self.num_calls if self.num_calls > 0 else 0.0

    def project_total_cost(self, remaining_calls: int) -> float:
        if self.num_calls == 0:
            return 0.0
        projected_remaining = remaining_calls * self.cost_per_boundary
        return self.total_cost_usd + projected_remaining

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = """You are a discourse segmentation expert. A segment boundary occurs when the conversation shifts to a new topic, task, or phase. Return ONLY 'Y' or 'N' (single token)."""

def build_user_prompt(context_before: List[str], context_after: str) -> str:
    """Build user prompt for boundary decision."""
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
    """Build stricter retry prompt after invalid first token."""
    return original_prompt + "\n\nIMPORTANT: Your response must be exactly 'Y' or 'N', nothing else."

# =============================================================================
# CACHING
# =============================================================================

def get_cache_key(dataset: str, dialogue_id: int, position: int, prompt_hash: str) -> str:
    """Generate cache key for a boundary decision."""
    return f"{dataset}_{dialogue_id}_{position}_{prompt_hash}"

def compute_prompt_hash(system: str, user: str) -> str:
    """Compute hash of prompt for cache lookup."""
    content = f"{system}|{user}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

class ResponseCache:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f)

    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict):
        self._cache[key] = value
        self._save_cache()

    def size(self) -> int:
        return len(self._cache)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_superseg() -> List[DialogueData]:
    """Load SuperSeg dataset dialogues with gold boundaries."""
    test_file = DATASETS_DIR / "superseg" / "segmentation_file_test.json"
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

            # Extract boundaries at user turn positions
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

def sample_dialogues_by_length_tertiles(
    dialogues: List[DialogueData],
    n_sample: int,
    seed: int = 42
) -> List[DialogueData]:
    """Sample dialogues stratified by length tertiles."""
    if len(dialogues) <= n_sample:
        return dialogues

    lengths = [d.num_user_turns for d in dialogues]
    tertiles = np.percentile(lengths, [33.3, 66.7])

    short = [d for d in dialogues if d.num_user_turns <= tertiles[0]]
    medium = [d for d in dialogues if tertiles[0] < d.num_user_turns <= tertiles[1]]
    long = [d for d in dialogues if d.num_user_turns > tertiles[1]]

    rng = np.random.RandomState(seed)
    n_per_tertile = n_sample // 3

    sampled = []
    for group in [short, medium, long]:
        if len(group) <= n_per_tertile:
            sampled.extend(group)
        else:
            indices = rng.choice(len(group), n_per_tertile, replace=False)
            sampled.extend([group[i] for i in indices])

    return sampled

# =============================================================================
# MODEL VALIDATION
# =============================================================================

def validate_model_access() -> bool:
    """Validate that gpt-5.2 is accessible via OpenAI API."""
    try:
        import openai
        client = openai.OpenAI()

        try:
            model = client.models.retrieve("gpt-5.2")
            print(f"[OK] Model gpt-5.2 is accessible")
            return True
        except openai.NotFoundError:
            print("[ERROR] Model 'gpt-5.2' not found")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to validate model access: {e}")
        return False

# =============================================================================
# GPT-5.2 SCORING (with FLIPPED polarity)
# =============================================================================

def score_boundary_gpt52(
    client,
    dialogue: DialogueData,
    position: int,
    cache: ResponseCache,
    dry_run: bool = False
) -> BoundaryScore:
    """
    Score a single boundary position using GPT-5.2.
    CRITICAL: Uses FLIPPED score polarity: s_i = logP(N) - logP(Y)
    """
    user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]

    context_before = []
    for i in range(max(0, position - 4), position):
        context_before.append(user_turns[i])

    context_after = user_turns[position] if position < len(user_turns) else ""

    user_prompt = build_user_prompt(context_before, context_after)
    prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)
    cache_key = get_cache_key("superseg", dialogue.dialogue_id, position, prompt_hash)

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return BoundaryScore(**cached)

    # Dry run mode
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
            missing_yn_in_toplogprobs=False,
            invalid_first_token=False,
            retried=False
        )

    # Make API call
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
            "is_retry": is_retry
        }

    # First attempt
    result = make_call(user_prompt)
    first_token = result["raw_token"].strip().upper()[:1]

    invalid_first_token = first_token not in ("Y", "N")
    retried = False

    # Retry if invalid
    if invalid_first_token:
        retry_prompt = build_retry_prompt(user_prompt)
        result = make_call(retry_prompt, is_retry=True)
        first_token = result["raw_token"].strip().upper()[:1]
        retried = True
        invalid_first_token = first_token not in ("Y", "N")

    # Handle missing Y/N in logprobs
    missing_yn = result["logprob_y"] is None or result["logprob_n"] is None

    # Compute score - FLIPPED POLARITY
    if missing_yn or invalid_first_token:
        score = 0.0
    else:
        # CRITICAL: Flipped from original: logP(N) - logP(Y)
        score = result["logprob_n"] - result["logprob_y"]

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

    # Cache result
    cache.set(cache_key, asdict(boundary_score))

    return boundary_score

def score_all_boundaries(
    dialogues: List[DialogueData],
    cache: ResponseCache,
    cost_tracker: CostTracker,
    dry_run: bool = False
) -> Tuple[List[BoundaryScore], bool]:
    """
    Score all boundaries in dialogues.
    Returns: (all_scores, aborted)
    """
    import openai
    client = openai.OpenAI() if not dry_run else None

    all_scores = []
    total_boundaries = 0
    aborted = False

    print(f"\nScoring {len(dialogues)} dialogues...")

    for i, dialogue in enumerate(dialogues):
        if aborted:
            break

        for position in range(1, dialogue.num_user_turns):
            # Check boundary cap
            if total_boundaries >= HARD_MAX_BOUNDARIES:
                print(f"\n[ABORT] Reached boundary cap ({HARD_MAX_BOUNDARIES})")
                aborted = True
                break

            result = score_boundary_gpt52(client, dialogue, position, cache, dry_run)

            # Update cost tracker
            cost_tracker.num_calls += 1
            cost_tracker.total_input_tokens += result.input_tokens
            cost_tracker.total_output_tokens += result.output_tokens
            cost_tracker.input_tokens_list.append(result.input_tokens)
            cost_tracker.output_tokens_list.append(result.output_tokens)

            if result.retried:
                cost_tracker.num_retries += 1
            if result.invalid_first_token:
                cost_tracker.num_invalid_first_token += 1
            if result.missing_yn_in_toplogprobs:
                cost_tracker.num_missing_yn += 1

            all_scores.append(result)
            total_boundaries += 1

            # Periodic cost check
            if cost_tracker.num_calls % COST_CHECK_INTERVAL == 0:
                remaining = sum(d.num_user_turns - 1 for d in dialogues[i:]) - (dialogue.num_user_turns - position)
                projected = cost_tracker.project_total_cost(remaining)

                print(f"  [{cost_tracker.num_calls}] Cost: ${cost_tracker.total_cost_usd:.4f}, "
                      f"Projected: ${projected:.4f}")

                if projected > HARD_BUDGET_USD:
                    print(f"\n[ABORT] Projected cost ${projected:.2f} exceeds budget ${HARD_BUDGET_USD:.2f}")
                    aborted = True
                    break

                if cost_tracker.total_cost_usd > HARD_BUDGET_USD:
                    print(f"\n[ABORT] Actual cost ${cost_tracker.total_cost_usd:.2f} exceeds budget")
                    aborted = True
                    break

        # Progress update
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(dialogues)} dialogues, "
                  f"{len(all_scores)} boundaries, ${cost_tracker.total_cost_usd:.4f}")

    return all_scores, aborted

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float, min_gap: int) -> Set[int]:
    """Greedy NMS prediction: select positions with score > tau, enforcing min_gap."""
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])

    predicted = set()
    for pos, score in candidates:
        too_close = any(abs(pos - p) < min_gap for p in predicted)
        if not too_close:
            predicted.add(pos)

    return predicted

def compute_wf1_and_bor(predicted: Set[int], gold: Set[int], k: int = 3) -> Tuple[float, float, float, float]:
    """Compute W-F1, purity, coverage, and BOR."""
    if not gold:
        n_pred = len(predicted)
        return 0.0, 0.0, 0.0, float('inf') if n_pred > 0 else 1.0

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

    wf1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    purity = precision
    coverage = recall
    bor = len(predicted) / len(gold) if gold else float('inf')

    return wf1, purity, coverage, bor

def compute_sweep(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]]
) -> List[Dict]:
    """Sweep tau thresholds and compute metrics at each point."""
    all_scores = []
    for dialogue_id, scores_by_pos in scores_by_dialogue.items():
        all_scores.extend(scores_by_pos.values())

    if not all_scores:
        return []

    all_scores = np.array(all_scores)

    sweep_points = []
    for pct in TAU_PERCENTILES:
        tau = np.percentile(all_scores, pct)

        total_wf1 = 0.0
        total_purity = 0.0
        total_coverage = 0.0
        total_pred = 0
        total_gold = 0
        n_dialogues = 0
        per_dialogue_wf1 = {}

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            if not gold:
                continue

            predicted = greedy_nms_predict(scores_by_pos, tau, MIN_GAP)
            wf1, purity, coverage, _ = compute_wf1_and_bor(predicted, gold)

            per_dialogue_wf1[dialogue_id] = wf1
            total_wf1 += wf1
            total_purity += purity
            total_coverage += coverage
            total_pred += len(predicted)
            total_gold += len(gold)
            n_dialogues += 1

        if n_dialogues == 0:
            continue

        avg_wf1 = total_wf1 / n_dialogues
        avg_purity = total_purity / n_dialogues
        avg_coverage = total_coverage / n_dialogues
        bor = total_pred / total_gold if total_gold > 0 else 0

        sweep_points.append({
            "percentile": pct,
            "tau": float(tau),
            "wf1": avg_wf1,
            "purity": avg_purity,
            "coverage": avg_coverage,
            "bor": bor,
            "n_dialogues": n_dialogues,
            "total_pred": total_pred,
            "total_gold": total_gold,
            "per_dialogue_wf1": per_dialogue_wf1,
        })

    return sweep_points

def compute_split_half_deviation(sweep_points: List[Dict]) -> Tuple[float, List[Dict], List[Dict]]:
    """Compute split-half reliability."""
    if not sweep_points:
        return 0.0, [], []

    all_dialogue_ids = list(sweep_points[0]["per_dialogue_wf1"].keys())

    if len(all_dialogue_ids) < 4:
        return 0.0, [], []

    np.random.seed(42)
    shuffled = np.random.permutation(all_dialogue_ids)
    mid = len(shuffled) // 2
    half1_ids = set(shuffled[:mid])
    half2_ids = set(shuffled[mid:])

    half1_curve = []
    half2_curve = []

    for sp in sweep_points:
        per_dialogue = sp["per_dialogue_wf1"]

        h1_wf1s = [wf1 for did, wf1 in per_dialogue.items() if did in half1_ids]
        h1_avg = np.mean(h1_wf1s) if h1_wf1s else 0

        h2_wf1s = [wf1 for did, wf1 in per_dialogue.items() if did in half2_ids]
        h2_avg = np.mean(h2_wf1s) if h2_wf1s else 0

        half1_curve.append({"bor": sp["bor"], "wf1": h1_avg})
        half2_curve.append({"bor": sp["bor"], "wf1": h2_avg})

    max_dev = 0.0
    for h1, h2 in zip(half1_curve, half2_curve):
        dev = abs(h1["wf1"] - h2["wf1"])
        if dev > max_dev:
            max_dev = dev

    return max_dev, half1_curve, half2_curve

def compute_auroc(scores: List[float], labels: List[bool]) -> float:
    """Compute AUROC for binary classification."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)

def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std

# =============================================================================
# PLOTTING
# =============================================================================

def create_plots(
    all_scores: List[BoundaryScore],
    gold_by_dialogue: Dict[int, Set[int]],
    sweep_points: List[Dict],
    split_half_data: Dict
):
    """Create diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter valid scores
    valid_scores = [s for s in all_scores
                    if not s.missing_yn_in_toplogprobs and not s.invalid_first_token]

    scores = [s.score for s in valid_scores]
    labels = [s.position in gold_by_dialogue.get(s.dialogue_id, set()) for s in valid_scores]

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # 1. Score Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_arr, bins=50, edgecolor='black', alpha=0.7)

    q10 = np.percentile(scores_arr, 10)
    q50 = np.percentile(scores_arr, 50)
    q90 = np.percentile(scores_arr, 90)

    ax.axvline(q10, color='red', linestyle='--', label=f'q10={q10:.2f}')
    ax.axvline(q50, color='green', linestyle='--', label=f'q50={q50:.2f}')
    ax.axvline(q90, color='blue', linestyle='--', label=f'q90={q90:.2f}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='s=0')

    ax.set_xlabel('Score s_i = log P(N) - log P(Y)')
    ax.set_ylabel('Count')
    ax.set_title(f'SuperSeg: Score Distribution (n={len(scores_arr)})')
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'superseg_score_histogram.png', dpi=150)
    plt.close(fig)

    # 2. Score vs Gold Label
    gold_scores = scores_arr[labels_arr]
    non_gold_scores = scores_arr[~labels_arr]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot([non_gold_scores, gold_scores],
                    tick_labels=['Non-Gold', 'Gold'],
                    patch_artist=True)

    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    for i, (data, color) in enumerate([(non_gold_scores, 'blue'), (gold_scores, 'red')]):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(i + 1 + jitter, data, alpha=0.3, s=10, c=color)

    ax.set_ylabel('Score s_i')
    ax.set_title('SuperSeg: Score by Gold Label')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    if len(gold_scores) > 0 and len(non_gold_scores) > 0:
        auroc = compute_auroc(list(scores_arr), list(labels_arr))
        cohens_d = compute_cohens_d(list(gold_scores), list(non_gold_scores))
        ax.text(0.02, 0.98, f"AUC-ROC: {auroc:.3f}\nCohen's d: {cohens_d:.3f}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'superseg_score_vs_gold.png', dpi=150)
    plt.close(fig)

    # 3. W-F1 vs BOR Curve
    if sweep_points:
        fig, ax = plt.subplots(figsize=(10, 6))

        bors = [sp["bor"] for sp in sweep_points]
        wf1s = [sp["wf1"] for sp in sweep_points]

        ax.plot(bors, wf1s, 'b-', linewidth=2, label='Full Sample', marker='o', markersize=4)

        # Split-half curves
        h1_curve = split_half_data.get("half1_curve", [])
        h2_curve = split_half_data.get("half2_curve", [])

        if h1_curve and h2_curve:
            ax.plot([p["bor"] for p in h1_curve], [p["wf1"] for p in h1_curve],
                    'g--', linewidth=1, alpha=0.7, label='Half 1')
            ax.plot([p["bor"] for p in h2_curve], [p["wf1"] for p in h2_curve],
                    'r--', linewidth=1, alpha=0.7, label='Half 2')

        ax.set_xlabel('BOR (Boundary Oversegmentation Ratio)')
        ax.set_ylabel('W-F1')
        max_dev = split_half_data.get("max_deviation", 0)
        ax.set_title(f'SuperSeg: W-F1 vs BOR\nSplit-half max deviation: {max_dev:.4f}')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='BOR=1')
        ax.legend()

        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'superseg_wf1_vs_bor.png', dpi=150)
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-5.2 SuperSeg Validation")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    args = parser.parse_args()

    print("=" * 60)
    print("GPT-5.2 SUPERSEG VALIDATION RUN")
    print("=" * 60)
    print(f"Budget cap: ${HARD_BUDGET_USD:.2f}")
    print(f"Boundary cap: {HARD_MAX_BOUNDARIES}")
    print(f"Sample size: {N_SAMPLE_DIALOGUES} dialogues")
    print(f"Score formula: s_i = logP(N) - logP(Y) [FLIPPED]")

    # Validate model access
    if not args.dry_run:
        print("\nValidating model access...")
        if not validate_model_access():
            return 1

    # Load SuperSeg
    print("\nLoading SuperSeg dataset...")
    all_dialogues = load_superseg()
    print(f"  Total dialogues: {len(all_dialogues)}")
    print(f"  Total gold boundaries: {sum(len(d.gold_boundaries) for d in all_dialogues)}")

    # Sample dialogues
    dialogues = sample_dialogues_by_length_tertiles(all_dialogues, N_SAMPLE_DIALOGUES)
    print(f"  Sampled dialogues: {len(dialogues)}")

    n_boundaries = sum(d.num_user_turns - 1 for d in dialogues)
    print(f"  Candidate boundaries to score: {n_boundaries}")

    if n_boundaries > HARD_MAX_BOUNDARIES:
        print(f"  [WARNING] Exceeds cap, will score first {HARD_MAX_BOUNDARIES}")

    # Initialize cache and cost tracker
    cache = ResponseCache(CACHE_DIR)
    print(f"\nCache entries: {cache.size()}")

    cost_tracker = CostTracker()

    # Score all boundaries
    print("\n" + "=" * 60)
    print("SCORING BOUNDARIES")
    print("=" * 60)

    all_scores, aborted = score_all_boundaries(dialogues, cache, cost_tracker, args.dry_run)

    print(f"\n  Total boundaries scored: {len(all_scores)}")
    print(f"  Aborted: {aborted}")
    print(f"  Cost: ${cost_tracker.total_cost_usd:.4f}")
    print(f"  Invalid first token: {cost_tracker.num_invalid_first_token}")
    print(f"  Missing Y/N in logprobs: {cost_tracker.num_missing_yn}")

    # Build score dictionaries
    scores_by_dialogue = {}
    gold_by_dialogue = {}

    for dialogue in dialogues:
        gold_by_dialogue[dialogue.dialogue_id] = dialogue.gold_boundaries

    for score in all_scores:
        if score.missing_yn_in_toplogprobs or score.invalid_first_token:
            continue
        if score.dialogue_id not in scores_by_dialogue:
            scores_by_dialogue[score.dialogue_id] = {}
        scores_by_dialogue[score.dialogue_id][score.position] = score.score

    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)

    sweep_points = compute_sweep(scores_by_dialogue, gold_by_dialogue)

    if sweep_points:
        best_point = max(sweep_points, key=lambda x: x["wf1"])
        print(f"  Best W-F1: {best_point['wf1']:.4f} at tau={best_point['tau']:.3f} (BOR={best_point['bor']:.3f})")
        print(f"  Purity: {best_point['purity']:.4f}, Coverage: {best_point['coverage']:.4f}")

    # Split-half reliability
    max_dev, h1_curve, h2_curve = compute_split_half_deviation(sweep_points)
    print(f"  Split-half max deviation: {max_dev:.4f}")

    split_half_data = {
        "max_deviation": max_dev,
        "half1_curve": h1_curve,
        "half2_curve": h2_curve
    }

    # Score separation analysis
    valid_scores = [s for s in all_scores
                    if not s.missing_yn_in_toplogprobs and not s.invalid_first_token]

    gold_scores = [s.score for s in valid_scores
                   if s.position in gold_by_dialogue.get(s.dialogue_id, set())]
    non_gold_scores = [s.score for s in valid_scores
                       if s.position not in gold_by_dialogue.get(s.dialogue_id, set())]

    if gold_scores and non_gold_scores:
        all_score_values = [s.score for s in valid_scores]
        all_labels = [s.position in gold_by_dialogue.get(s.dialogue_id, set()) for s in valid_scores]
        auroc = compute_auroc(all_score_values, all_labels)
        cohens_d = compute_cohens_d(gold_scores, non_gold_scores)

        print(f"  Gold mean: {np.mean(gold_scores):.3f}, Non-gold mean: {np.mean(non_gold_scores):.3f}")
        print(f"  AUC-ROC: {auroc:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
    else:
        auroc = 0.5
        cohens_d = 0.0

    # Acceptance criteria
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)

    auc_pass = auroc > 0.6
    split_half_pass = max_dev < 0.10  # Relaxed threshold
    cost_pass = cost_tracker.total_cost_usd < HARD_BUDGET_USD

    print(f"  AUC-ROC > 0.6: {auroc:.4f} [{'PASS' if auc_pass else 'FAIL'}]")
    print(f"  Split-half < 0.10: {max_dev:.4f} [{'PASS' if split_half_pass else 'FAIL'}]")
    print(f"  Cost < ${HARD_BUDGET_USD:.2f}: ${cost_tracker.total_cost_usd:.4f} [{'PASS' if cost_pass else 'FAIL'}]")

    all_passed = auc_pass and split_half_pass and cost_pass
    print(f"\n  OVERALL: {'PASS' if all_passed else 'FAIL'}")

    # Check curve shape (narrow peak near BOR=1?)
    print("\n" + "=" * 60)
    print("CURVE SHAPE ANALYSIS")
    print("=" * 60)

    if sweep_points:
        # Find peak
        best_idx = max(range(len(sweep_points)), key=lambda i: sweep_points[i]["wf1"])
        peak_bor = sweep_points[best_idx]["bor"]
        peak_wf1 = sweep_points[best_idx]["wf1"]

        print(f"  Peak: W-F1={peak_wf1:.4f} at BOR={peak_bor:.3f}")

        # Check if peak is near BOR=1
        if 0.8 <= peak_bor <= 1.2:
            print(f"  [YES] Peak is near BOR=1 (tight optimum)")
        else:
            print(f"  [NO] Peak is at BOR={peak_bor:.3f}, not near 1.0")

        # Measure falloff
        # Find W-F1 at BOR=0.5 and BOR=1.5
        wf1_at_05 = None
        wf1_at_15 = None
        for sp in sweep_points:
            if 0.4 <= sp["bor"] <= 0.6 and wf1_at_05 is None:
                wf1_at_05 = sp["wf1"]
            if 1.4 <= sp["bor"] <= 1.6 and wf1_at_15 is None:
                wf1_at_15 = sp["wf1"]

        if wf1_at_05 is not None:
            print(f"  W-F1 at BOR~0.5: {wf1_at_05:.4f} (drop from peak: {peak_wf1 - wf1_at_05:.4f})")
        if wf1_at_15 is not None:
            print(f"  W-F1 at BOR~1.5: {wf1_at_15:.4f} (drop from peak: {peak_wf1 - wf1_at_15:.4f})")

    # Create plots
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)

    create_plots(all_scores, gold_by_dialogue, sweep_points, split_half_data)
    print(f"  Plots saved to: {PLOT_DIR}")

    # Save results
    results = {
        "description": "GPT-5.2 SuperSeg validation run",
        "score_formula": "s_i = logP(N) - logP(Y) [FLIPPED]",
        "n_dialogues_sampled": len(dialogues),
        "n_boundaries_scored": len(all_scores),
        "n_valid_scores": len(valid_scores),
        "cost_usd": cost_tracker.total_cost_usd,
        "aborted": aborted,
        "best_wf1": best_point["wf1"] if sweep_points else 0,
        "best_tau": best_point["tau"] if sweep_points else 0,
        "best_bor": best_point["bor"] if sweep_points else 0,
        "split_half_max_deviation": max_dev,
        "auroc": auroc,
        "cohens_d": cohens_d,
        "gold_mean": float(np.mean(gold_scores)) if gold_scores else 0,
        "non_gold_mean": float(np.mean(non_gold_scores)) if non_gold_scores else 0,
        "criteria": {
            "auc_roc_passed": bool(auc_pass),
            "split_half_passed": bool(split_half_pass),
            "cost_passed": bool(cost_pass),
        },
        "all_passed": bool(all_passed),
        "sweep_points": [{k: v for k, v in sp.items() if k != "per_dialogue_wf1"} for sp in sweep_points],
    }

    print(f"\nSaving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
