#!/usr/bin/env python3
"""
GPT-5.2 Boundary Scorer Sanity Check

This script performs a low-cost sanity check of GPT-5.2 as a boundary scorer
in the "score -> threshold sweep -> BOR-conditioned curves" pipeline.

SAFETY CAPS (non-negotiable):
- HARD BUDGET: $10 maximum spend, checked after every 200 calls
- HARD BOUNDARIES: 5,000 boundaries max across both datasets
- HARD CONTEXT: 320 tokens target per call (will abort if p95 > 450)

REQUIREMENTS:
- OpenAI API key set in OPENAI_API_KEY environment variable
- Model: gpt-5.2 (must be accessible)
- Must enable logprobs with reasoning.effort="none"

Usage:
    python paper/experiments/gpt52_sanity_check.py [--dry-run] [--sample N]

Options:
    --dry-run       Run without making API calls (validates prompt/sampling)
    --sample N      Override dialogue sample size per dataset (default: 50)
    --cache-dir     Directory for caching API responses (default: .gpt52_cache)
    --output        Output JSON file path

Output:
    - JSON report to stdout and disk
    - PASS/FAIL acceptance criteria
    - Cost tracking and projections

Author: Generated for paper experiments
Date: 2026-01-17
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

from episodic.topics.evaluation import (
    compute_windowed_metrics,
    compute_windowed_metrics_one_to_one,
    compute_purity_coverage,
    boundaries_to_segments,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPT-5.2 pricing (as of January 2026)
GPT52_INPUT_PRICE_PER_1M = 1.75   # $1.75 per 1M input tokens
GPT52_OUTPUT_PRICE_PER_1M = 14.0  # $14.00 per 1M output tokens

# Hard safety limits
HARD_BUDGET_USD = 10.0           # Maximum total spend
HARD_MAX_BOUNDARIES = 5000       # Maximum boundaries to score
HARD_TARGET_INPUT_TOKENS = 320   # Target input tokens per call
HARD_MAX_INPUT_P95 = 450         # Abort if p95 input tokens exceeds this

# Cost check frequency
COST_CHECK_INTERVAL = 200        # Check projected cost after every N calls

# Sampling configuration
DEFAULT_SAMPLE_PER_DATASET = 50  # Dialogues per dataset
LENGTH_TERTILE_SPLITS = 3        # Split by short/medium/long

# Dataset regime classification (from paper)
DATASET_REGIMES = {
    "dialseg711": "fine",      # Higher boundary density
    "dailydialog": "coarse",   # Lower boundary density (synthetic)
}

# Top logprobs to request (ensure Y and N both appear)
# GPT-5.2 limit is 5; using max allowed value
TOP_LOGPROBS = 5

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
    position: int  # User turn index
    score: float   # log P(Y) - log P(N)
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
    def mean_output_tokens(self) -> float:
        return np.mean(self.output_tokens_list) if self.output_tokens_list else 0.0

    @property
    def p95_output_tokens(self) -> float:
        return np.percentile(self.output_tokens_list, 95) if self.output_tokens_list else 0.0

    @property
    def cost_per_boundary(self) -> float:
        return self.total_cost_usd / self.num_calls if self.num_calls > 0 else 0.0

    def project_total_cost(self, remaining_calls: int) -> float:
        """Project total cost including remaining planned calls."""
        if self.num_calls == 0:
            return 0.0
        projected_remaining = remaining_calls * self.cost_per_boundary
        return self.total_cost_usd + projected_remaining

@dataclass
class SweepPoint:
    """Single point in the threshold sweep."""
    tau: float
    bor: float
    wf1: float
    wf1_1to1: float
    coverage: float
    purity: float
    n_pred: int
    n_gold: int
    per_dialogue_wf1: List[float] = field(default_factory=list)

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

# System prompt: minimal rubric adapted from SeCom GPT-4 style
SYSTEM_PROMPT = """You are a discourse segmentation expert. A segment boundary occurs when the conversation shifts to a new topic, task, or phase. Return ONLY 'Y' or 'N' (single token)."""

def build_user_prompt(context_before: List[str], context_after: str) -> str:
    """
    Build user prompt for boundary decision.

    Args:
        context_before: List of recent turns BEFORE the candidate boundary
        context_after: The turn immediately AFTER the candidate boundary

    Returns:
        User prompt string
    """
    # Format context (limited to last 4 turns for token budget)
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

# =============================================================================
# DATA LOADING
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "datasets"

def load_dataset(dataset_name: str) -> List[DialogueData]:
    """Load dataset dialogues with gold boundaries."""
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
    """
    Sample dialogues stratified by length tertiles.

    Returns approximately n_sample dialogues (or fewer if dataset is smaller),
    with equal representation from short/medium/long dialogues.
    """
    if len(dialogues) <= n_sample:
        return dialogues

    # Compute length tertile boundaries
    lengths = [d.num_user_turns for d in dialogues]
    tertiles = np.percentile(lengths, [33.3, 66.7])

    # Split into tertiles
    short = [d for d in dialogues if d.num_user_turns <= tertiles[0]]
    medium = [d for d in dialogues if tertiles[0] < d.num_user_turns <= tertiles[1]]
    long = [d for d in dialogues if d.num_user_turns > tertiles[1]]

    # Sample from each tertile
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
    """
    Validate that gpt-5.2 is accessible via OpenAI API.
    If not found, print available gpt-5* models and return False.
    """
    try:
        import openai
        client = openai.OpenAI()

        # Try to retrieve the model
        try:
            model = client.models.retrieve("gpt-5.2")
            print(f"[OK] Model gpt-5.2 is accessible")
            return True
        except openai.NotFoundError:
            print("[ERROR] Model 'gpt-5.2' not found. Listing available gpt-5* models...")

            # List available models
            models = client.models.list()
            gpt5_models = [m.id for m in models if m.id.startswith("gpt-5")]

            if gpt5_models:
                print("Available gpt-5* models:")
                for m in sorted(gpt5_models):
                    print(f"  - {m}")
            else:
                print("No gpt-5* models found in your account.")

            return False

    except Exception as e:
        print(f"[ERROR] Failed to validate model access: {e}")
        return False

# =============================================================================
# GPT-5.2 SCORING
# =============================================================================

def score_boundary_gpt52(
    client,
    dialogue: DialogueData,
    position: int,
    cache: ResponseCache,
    dataset_name: str,
    dry_run: bool = False
) -> BoundaryScore:
    """
    Score a single boundary position using GPT-5.2.

    Args:
        client: OpenAI client
        dialogue: Dialogue containing the boundary
        position: User turn index of the boundary (turn that would start new segment)
        cache: Response cache
        dataset_name: Dataset name for cache key
        dry_run: If True, return mock response without API call

    Returns:
        BoundaryScore with log-odds score
    """
    # Build context
    user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]

    # Context before: turns 0 to position-1
    context_before = []
    for i in range(max(0, position - 4), position):
        context_before.append(user_turns[i])

    # Turn after boundary
    context_after = user_turns[position] if position < len(user_turns) else ""

    # Build prompts
    user_prompt = build_user_prompt(context_before, context_after)
    prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)
    cache_key = get_cache_key(dataset_name, dialogue.dialogue_id, position, prompt_hash)

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
            input_tokens=len(user_prompt.split()),  # Approximate
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
            max_completion_tokens=10,  # Allow a few tokens for model output
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            reasoning_effort="none",  # Required for logprobs on GPT-5.2
        )

        choice = response.choices[0]
        raw_token = choice.message.content.strip() if choice.message.content else ""

        # Extract logprobs
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

    # Compute score
    if missing_yn or invalid_first_token:
        score = 0.0  # Neutral score for missing data
    else:
        score = result["logprob_y"] - result["logprob_n"]

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
    dataset_name: str,
    cache: ResponseCache,
    cost_tracker: CostTracker,
    dry_run: bool = False
) -> Tuple[List[Dict[int, float]], bool]:
    """
    Score all boundaries in dialogues.

    Returns:
        Tuple of (scores_list, aborted)
        scores_list: List of dicts mapping position -> score per dialogue
        aborted: True if budget exceeded and scoring was stopped
    """
    import openai
    client = openai.OpenAI() if not dry_run else None

    all_scores = []
    total_boundaries = 0
    aborted = False

    for dialogue in dialogues:
        dialogue_scores = {}

        # Score each potential boundary position (user turns 1 to N-1)
        for position in range(1, dialogue.num_user_turns):
            # Check boundary cap
            if total_boundaries >= HARD_MAX_BOUNDARIES:
                print(f"\n[ABORT] Reached boundary cap ({HARD_MAX_BOUNDARIES})")
                aborted = True
                break

            # Score this boundary
            result = score_boundary_gpt52(
                client, dialogue, position, cache, dataset_name, dry_run
            )

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

            # Store score (skip if invalid after retry)
            if not result.invalid_first_token and not result.missing_yn_in_toplogprobs:
                dialogue_scores[position] = result.score

            total_boundaries += 1

            # Periodic cost check
            if cost_tracker.num_calls % COST_CHECK_INTERVAL == 0:
                remaining = HARD_MAX_BOUNDARIES - total_boundaries
                projected = cost_tracker.project_total_cost(remaining)

                print(f"\n[Cost Check] Calls: {cost_tracker.num_calls}, "
                      f"Current: ${cost_tracker.total_cost_usd:.4f}, "
                      f"Projected: ${projected:.4f}")

                if cost_tracker.total_cost_usd > HARD_BUDGET_USD:
                    print(f"[ABORT] Current cost (${cost_tracker.total_cost_usd:.2f}) exceeds budget")
                    aborted = True
                    break

                if projected > HARD_BUDGET_USD:
                    print(f"[ABORT] Projected cost (${projected:.2f}) exceeds budget")
                    aborted = True
                    break

                if cost_tracker.p95_input_tokens > HARD_MAX_INPUT_P95:
                    print(f"[ABORT] p95 input tokens ({cost_tracker.p95_input_tokens:.0f}) exceeds limit")
                    aborted = True
                    break

        all_scores.append(dialogue_scores)

        if aborted:
            break

    return all_scores, aborted

# =============================================================================
# THRESHOLD SWEEP
# =============================================================================

def apply_threshold_with_gap(
    scores: Dict[int, float],
    threshold: float,
    min_gap: int,
    num_messages: int
) -> Set[int]:
    """Apply threshold to scores with min_gap enforcement (greedy NMS)."""
    candidates = [
        (pos, score) for pos, score in scores.items()
        if score >= threshold and 1 <= pos < num_messages
    ]
    candidates.sort(key=lambda x: -x[1])

    selected = set()
    for pos, _ in candidates:
        ok = True
        for existing in selected:
            if abs(pos - existing) < min_gap:
                ok = False
                break
        if ok:
            selected.add(pos)

    return selected

def run_sweep(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    min_gap: int = 2,
    n_steps: int = 100,
    max_bor: float = 5.0
) -> List[SweepPoint]:
    """Run threshold sweep and compute metrics at each point."""
    # Collect all scores
    all_scores_flat = []
    for scores in scores_list:
        all_scores_flat.extend(scores.values())

    if not all_scores_flat:
        return []

    # Use quantiles for sweep
    thresholds = np.percentile(all_scores_flat, np.linspace(100, 0, n_steps))
    thresholds = np.unique(thresholds)

    # Count total gold boundaries
    total_gold = sum(len(d.gold_boundaries) for d in dialogues)

    results = []
    prev_bor = None

    for tau in thresholds:
        all_wf1 = []
        all_wf1_1to1 = []
        all_purity = []
        all_coverage = []
        total_pred = 0

        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(
                scores, tau, min_gap, dialogue.num_user_turns
            )
            total_pred += len(pred)

            # W-F1 (many-to-one)
            _, _, wf1 = compute_windowed_metrics(
                dialogue.gold_boundaries, pred, dialogue.num_user_turns, window=1
            )
            all_wf1.append(wf1)

            # W-F1 (one-to-one)
            _, _, wf1_1to1 = compute_windowed_metrics_one_to_one(
                dialogue.gold_boundaries, pred, dialogue.num_user_turns, window=1
            )
            all_wf1_1to1.append(wf1_1to1)

            # Purity/Coverage
            gold_segments = boundaries_to_segments(
                dialogue.gold_boundaries, dialogue.num_user_turns
            )
            pred_segments = boundaries_to_segments(pred, dialogue.num_user_turns)
            purity, coverage = compute_purity_coverage(gold_segments, pred_segments)
            all_purity.append(purity)
            all_coverage.append(coverage)

        bor = total_pred / total_gold if total_gold > 0 else 0.0

        if bor > max_bor:
            break

        if prev_bor is not None and abs(bor - prev_bor) < 1e-6:
            continue

        results.append(SweepPoint(
            tau=float(tau),
            bor=bor,
            wf1=float(np.mean(all_wf1)),
            wf1_1to1=float(np.mean(all_wf1_1to1)),
            coverage=float(np.mean(all_coverage)),
            purity=float(np.mean(all_purity)),
            n_pred=total_pred,
            n_gold=total_gold,
            per_dialogue_wf1=all_wf1
        ))

        prev_bor = bor

    return results

# =============================================================================
# STABILITY DIAGNOSTICS
# =============================================================================

def compute_bor_decile_support(
    sweep_points: List[SweepPoint],
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    min_gap: int = 2
) -> List[Dict]:
    """
    Compute support per BOR decile.

    Returns list of dicts with:
    - bor_low, bor_high: BOR range
    - n_boundaries: Number of predicted boundaries in this decile
    - n_dialogues: Number of dialogues contributing
    """
    if not sweep_points:
        return []

    bor_values = [sp.bor for sp in sweep_points]
    bor_min, bor_max = min(bor_values), max(bor_values)

    if bor_max <= bor_min:
        return []

    decile_edges = np.linspace(bor_min, bor_max, 11)
    results = []

    for i in range(10):
        bor_low, bor_high = decile_edges[i], decile_edges[i+1]

        # Find sweep points in this BOR range
        points_in_decile = [sp for sp in sweep_points
                          if bor_low <= sp.bor < bor_high or
                          (i == 9 and sp.bor == bor_high)]

        if not points_in_decile:
            results.append({
                "bor_low": float(bor_low),
                "bor_high": float(bor_high),
                "n_boundaries": 0,
                "n_dialogues": 0
            })
            continue

        # Use the middle point's predictions
        mid_point = points_in_decile[len(points_in_decile)//2]
        tau = mid_point.tau

        n_boundaries = 0
        dialogues_contributing = set()

        for dialogue, scores in zip(dialogues, scores_list):
            pred = apply_threshold_with_gap(
                scores, tau, min_gap, dialogue.num_user_turns
            )
            if pred:
                n_boundaries += len(pred)
                dialogues_contributing.add(dialogue.dialogue_id)

        results.append({
            "bor_low": float(bor_low),
            "bor_high": float(bor_high),
            "n_boundaries": n_boundaries,
            "n_dialogues": len(dialogues_contributing)
        })

    return results

def compute_split_half_reliability(
    dialogues: List[DialogueData],
    scores_list: List[Dict[int, float]],
    seed: int = 42
) -> float:
    """
    Compute split-half reliability: max absolute deviation in W-F1 at matched BOR.

    Splits dialogues into two halves, computes curves separately,
    then computes max |W-F1_A - W-F1_B| at matched BOR points.
    """
    if len(dialogues) < 4:
        return 0.0

    # Split dialogues
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dialogues))
    half = len(indices) // 2

    indices_a = set(indices[:half])
    indices_b = set(indices[half:])

    dialogues_a = [d for i, d in enumerate(dialogues) if i in indices_a]
    dialogues_b = [d for i, d in enumerate(dialogues) if i in indices_b]
    scores_a = [s for i, s in enumerate(scores_list) if i in indices_a]
    scores_b = [s for i, s in enumerate(scores_list) if i in indices_b]

    # Run sweeps
    sweep_a = run_sweep(dialogues_a, scores_a)
    sweep_b = run_sweep(dialogues_b, scores_b)

    if not sweep_a or not sweep_b:
        return 0.0

    # Compute max deviation at matched BOR
    max_deviation = 0.0

    # Use linear interpolation to match BOR values
    bor_a = np.array([sp.bor for sp in sweep_a])
    wf1_a = np.array([sp.wf1 for sp in sweep_a])
    bor_b = np.array([sp.bor for sp in sweep_b])
    wf1_b = np.array([sp.wf1 for sp in sweep_b])

    # Find common BOR range
    bor_min = max(min(bor_a), min(bor_b))
    bor_max = min(max(bor_a), max(bor_b))

    if bor_max <= bor_min:
        return 0.0

    # Sample BOR points in common range
    common_bors = np.linspace(bor_min, bor_max, 50)

    for bor in common_bors:
        # Interpolate W-F1 at this BOR
        wf1_at_bor_a = np.interp(bor, bor_a, wf1_a)
        wf1_at_bor_b = np.interp(bor, bor_b, wf1_b)

        deviation = abs(wf1_at_bor_a - wf1_at_bor_b)
        max_deviation = max(max_deviation, deviation)

    return float(max_deviation)

# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

def check_acceptance_criteria(
    cost_tracker: CostTracker,
    split_half_deviations: Dict[str, float],
    bor_decile_support: Dict[str, List[Dict]]
) -> Dict[str, Tuple[bool, str]]:
    """
    Check acceptance criteria and return PASS/FAIL with reasons.

    Criteria:
    - invalid_first_token_rate <= 1%
    - missing_YN_in_toplogprobs_rate <= 5%
    - mean input tokens <= 320 and p95 <= 450
    - actual cost <= $10
    - split_half_max_delta_wf1 <= 0.02 where BOR decile has >= 200 boundaries
    """
    results = {}

    # 1. Invalid first token rate
    rate = (cost_tracker.num_invalid_first_token / cost_tracker.num_calls
            if cost_tracker.num_calls > 0 else 0)
    passed = rate <= 0.01
    results["invalid_first_token_rate"] = (
        passed,
        f"{rate*100:.2f}% (threshold: 1%)"
    )

    # 2. Missing Y/N in toplogprobs rate
    rate = (cost_tracker.num_missing_yn / cost_tracker.num_calls
            if cost_tracker.num_calls > 0 else 0)
    passed = rate <= 0.05
    results["missing_YN_rate"] = (
        passed,
        f"{rate*100:.2f}% (threshold: 5%)"
    )

    # 3. Mean input tokens
    passed = cost_tracker.mean_input_tokens <= HARD_TARGET_INPUT_TOKENS
    results["mean_input_tokens"] = (
        passed,
        f"{cost_tracker.mean_input_tokens:.0f} (threshold: {HARD_TARGET_INPUT_TOKENS})"
    )

    # 4. P95 input tokens
    passed = cost_tracker.p95_input_tokens <= HARD_MAX_INPUT_P95
    results["p95_input_tokens"] = (
        passed,
        f"{cost_tracker.p95_input_tokens:.0f} (threshold: {HARD_MAX_INPUT_P95})"
    )

    # 5. Total cost
    passed = cost_tracker.total_cost_usd <= HARD_BUDGET_USD
    results["total_cost"] = (
        passed,
        f"${cost_tracker.total_cost_usd:.4f} (threshold: ${HARD_BUDGET_USD})"
    )

    # 6. Split-half reliability (check where decile has >= 200 boundaries)
    for dataset, deviation in split_half_deviations.items():
        # Check if any decile has sufficient support
        support = bor_decile_support.get(dataset, [])
        sufficient_deciles = [d for d in support if d["n_boundaries"] >= 200]

        if not sufficient_deciles:
            results[f"split_half_{dataset}"] = (
                True,  # Pass if no deciles have sufficient support
                f"N/A (no deciles with >=200 boundaries)"
            )
        else:
            passed = deviation <= 0.02
            results[f"split_half_{dataset}"] = (
                passed,
                f"{deviation:.4f} (threshold: 0.02)"
            )

    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.2 Boundary Scorer Sanity Check"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without API calls")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_PER_DATASET,
                        help=f"Dialogues per dataset (default: {DEFAULT_SAMPLE_PER_DATASET})")
    parser.add_argument("--cache-dir", type=str, default=".gpt52_cache",
                        help="Cache directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("GPT-5.2 BOUNDARY SCORER SANITY CHECK")
    print("=" * 70)
    print(f"Hard Budget: ${HARD_BUDGET_USD}")
    print(f"Max Boundaries: {HARD_MAX_BOUNDARIES}")
    print(f"Target Input Tokens: {HARD_TARGET_INPUT_TOKENS}")
    print(f"Dry Run: {args.dry_run}")
    print()

    # Validate model access (skip in dry run)
    if not args.dry_run:
        if not validate_model_access():
            print("\n[FATAL] Cannot access gpt-5.2. Aborting.")
            sys.exit(1)

    # Initialize
    cache_dir = PROJECT_ROOT / args.cache_dir
    cache = ResponseCache(cache_dir)
    cost_tracker = CostTracker()

    # Select datasets: one coarse, one fine
    datasets_to_use = list(DATASET_REGIMES.keys())
    print(f"\nDatasets: {datasets_to_use}")
    print(f"  {datasets_to_use[0]}: {DATASET_REGIMES[datasets_to_use[0]]} regime")
    print(f"  {datasets_to_use[1]}: {DATASET_REGIMES[datasets_to_use[1]]} regime")

    # Pre-run cost estimate
    estimated_boundaries = args.sample * 10 * len(datasets_to_use)  # ~10 boundaries per dialogue
    estimated_cost = (estimated_boundaries * HARD_TARGET_INPUT_TOKENS / 1_000_000 * GPT52_INPUT_PRICE_PER_1M +
                      estimated_boundaries * 1 / 1_000_000 * GPT52_OUTPUT_PRICE_PER_1M)
    print(f"\nPre-run cost estimate (worst case): ${estimated_cost:.4f}")

    if estimated_cost > HARD_BUDGET_USD:
        print(f"[WARNING] Estimated cost exceeds budget. Will enforce hard caps.")

    # Process each dataset
    all_results = {}
    bor_decile_support = {}
    split_half_deviations = {}

    for dataset_name in datasets_to_use:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        # Load and sample dialogues
        print("Loading dataset...")
        dialogues = load_dataset(dataset_name)
        print(f"  Total dialogues: {len(dialogues)}")

        sampled = sample_dialogues_by_length_tertiles(dialogues, args.sample)
        print(f"  Sampled dialogues: {len(sampled)}")

        total_boundaries = sum(d.num_user_turns - 1 for d in sampled)
        print(f"  Total boundaries to score: {total_boundaries}")

        # Score boundaries
        print("\nScoring boundaries...")
        scores_list, aborted = score_all_boundaries(
            sampled, dataset_name, cache, cost_tracker, args.dry_run
        )

        if aborted:
            print(f"[WARNING] Scoring aborted for {dataset_name}")

        # Run sweep
        print("\nRunning threshold sweep...")
        sweep_points = run_sweep(sampled, scores_list)
        print(f"  Sweep points: {len(sweep_points)}")

        if sweep_points:
            bor_range = (min(sp.bor for sp in sweep_points),
                        max(sp.bor for sp in sweep_points))
            print(f"  BOR range: {bor_range[0]:.2f} - {bor_range[1]:.2f}")

        # BOR decile support
        print("\nComputing BOR decile support...")
        decile_support = compute_bor_decile_support(sweep_points, sampled, scores_list)
        bor_decile_support[dataset_name] = decile_support

        # Split-half reliability
        print("Computing split-half reliability...")
        split_half = compute_split_half_reliability(sampled, scores_list)
        split_half_deviations[dataset_name] = split_half
        print(f"  Max deviation: {split_half:.4f}")

        # Store results
        all_results[dataset_name] = {
            "n_dialogues": len(sampled),
            "n_boundaries_scored": sum(len(s) for s in scores_list),
            "bor_range": [sweep_points[0].bor, sweep_points[-1].bor] if sweep_points else [0, 0],
            "sweep_points": [asdict(sp) for sp in sweep_points],
            "decile_support": decile_support,
            "split_half_max_deviation": split_half
        }

        if aborted:
            break

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    # Cost summary
    print("\nCost Summary:")
    print(f"  Total API calls: {cost_tracker.num_calls}")
    print(f"  Total input tokens: {cost_tracker.total_input_tokens:,}")
    print(f"  Total output tokens: {cost_tracker.total_output_tokens:,}")
    print(f"  Mean input tokens: {cost_tracker.mean_input_tokens:.1f}")
    print(f"  P95 input tokens: {cost_tracker.p95_input_tokens:.1f}")
    print(f"  Total cost: ${cost_tracker.total_cost_usd:.4f}")
    print(f"  Cost per boundary: ${cost_tracker.cost_per_boundary:.6f}")

    # Quality summary
    print("\nQuality Summary:")
    print(f"  Invalid first token: {cost_tracker.num_invalid_first_token} ({cost_tracker.num_invalid_first_token/cost_tracker.num_calls*100:.2f}%)" if cost_tracker.num_calls > 0 else "  Invalid first token: N/A")
    print(f"  Missing Y/N in logprobs: {cost_tracker.num_missing_yn} ({cost_tracker.num_missing_yn/cost_tracker.num_calls*100:.2f}%)" if cost_tracker.num_calls > 0 else "  Missing Y/N in logprobs: N/A")
    print(f"  Retries: {cost_tracker.num_retries}")

    # Acceptance criteria
    print("\n" + "-" * 40)
    print("ACCEPTANCE CRITERIA")
    print("-" * 40)

    criteria = check_acceptance_criteria(
        cost_tracker, split_half_deviations, bor_decile_support
    )

    all_passed = True
    for name, (passed, detail) in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 70)

    # Helper to convert numpy types for JSON serialization
    def jsonify(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Build output JSON
    output = jsonify({
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dry_run": args.dry_run,
            "sample_per_dataset": args.sample,
            "hard_budget_usd": HARD_BUDGET_USD,
            "hard_max_boundaries": HARD_MAX_BOUNDARIES,
            "hard_target_input_tokens": HARD_TARGET_INPUT_TOKENS,
            "top_logprobs": TOP_LOGPROBS
        },
        "cost_summary": {
            "total_calls": cost_tracker.num_calls,
            "total_input_tokens": cost_tracker.total_input_tokens,
            "total_output_tokens": cost_tracker.total_output_tokens,
            "mean_input_tokens": cost_tracker.mean_input_tokens,
            "p95_input_tokens": cost_tracker.p95_input_tokens,
            "mean_output_tokens": cost_tracker.mean_output_tokens,
            "p95_output_tokens": cost_tracker.p95_output_tokens,
            "total_cost_usd": cost_tracker.total_cost_usd,
            "cost_per_boundary_usd": cost_tracker.cost_per_boundary
        },
        "quality_summary": {
            "invalid_first_token_count": cost_tracker.num_invalid_first_token,
            "invalid_first_token_rate": cost_tracker.num_invalid_first_token / cost_tracker.num_calls if cost_tracker.num_calls > 0 else 0,
            "missing_yn_count": cost_tracker.num_missing_yn,
            "missing_yn_rate": cost_tracker.num_missing_yn / cost_tracker.num_calls if cost_tracker.num_calls > 0 else 0,
            "retry_count": cost_tracker.num_retries
        },
        "datasets": all_results,
        "acceptance_criteria": {
            name: {"passed": bool(passed), "detail": detail}
            for name, (passed, detail) in criteria.items()
        },
        "overall_passed": all_passed
    })

    # Write output
    output_path = args.output or (PROJECT_ROOT / "paper/experiments/gpt52_sanity_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Also print to stdout
    print("\n" + "-" * 40)
    print("JSON OUTPUT:")
    print("-" * 40)
    print(json.dumps(output, indent=2))

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
