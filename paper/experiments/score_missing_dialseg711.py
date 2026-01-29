#!/usr/bin/env python3
"""
Score missing DialSeg711 dialogues with GPT-5.2

The existing cache has 615 dialogues but DialSeg711 has 711.
This script scores the missing 96 dialogues (IDs 608-710).

Usage:
    python paper/experiments/score_missing_dialseg711.py
    python paper/experiments/score_missing_dialseg711.py --dry-run  # estimate cost only
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"
CACHE_PATH = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"

# Prompts (must match original scoring)
SYSTEM_PROMPT = """You are a discourse segmentation expert. A segment boundary occurs when the conversation shifts to a new topic, task, or phase. Return ONLY 'Y' or 'N' (single token)."""

TOP_LOGPROBS = 5

# Cost tracking (GPT-5.2 pricing estimate)
COST_PER_1K_INPUT = 0.01
COST_PER_1K_OUTPUT = 0.03


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


def load_cache() -> Dict:
    """Load existing cache."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    """Save cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_cached_dialogue_ids(cache: Dict) -> Set[int]:
    """Get set of dialogue IDs already in cache."""
    ids = set()
    for entry in cache.values():
        did = entry.get("dialogue_id")
        if did is not None:
            ids.add(did)
    return ids


def compute_prompt_hash(system: str, user: str) -> str:
    """Compute hash of prompt for cache key."""
    combined = f"{system}|||{user}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def build_user_prompt(context_before: List[str], context_after: str) -> str:
    """Build user prompt for boundary detection."""
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


def score_boundary(
    client,
    dialogue: DialogueData,
    position: int,
    dry_run: bool = False
) -> Dict:
    """Score a single boundary position."""
    user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]

    context_before = []
    for i in range(max(0, position - 4), position):
        context_before.append(user_turns[i])

    context_after = user_turns[position]
    user_prompt = build_user_prompt(context_before, context_after)
    prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)

    if dry_run:
        # Estimate tokens
        input_tokens = len(SYSTEM_PROMPT.split()) + len(user_prompt.split())
        return {
            "dialogue_id": dialogue.dialogue_id,
            "position": position,
            "input_tokens": input_tokens * 1.3,  # rough estimate
            "dry_run": True
        }

    # Make API call
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

    # Extract logprobs
    logprob_y = None
    logprob_n = None
    invalid_first_token = False
    missing_yn = False

    if choice.logprobs and choice.logprobs.content:
        first_token_logprobs = choice.logprobs.content[0].top_logprobs
        for lp in first_token_logprobs:
            token_lower = lp.token.strip().lower()
            if token_lower == "y" or token_lower == "yes":
                logprob_y = lp.logprob
            elif token_lower == "n" or token_lower == "no":
                logprob_n = lp.logprob

        if logprob_y is None or logprob_n is None:
            missing_yn = True

        first_token = first_token_logprobs[0].token.strip().lower() if first_token_logprobs else ""
        if first_token not in ["y", "yes", "n", "no"]:
            invalid_first_token = True

    # Compute score with FLIPPED polarity: s_i = logP(N) - logP(Y)
    if logprob_y is not None and logprob_n is not None:
        score = logprob_n - logprob_y
    else:
        score = 0

    return {
        "dialogue_id": dialogue.dialogue_id,
        "position": position,
        "score": score,
        "raw_token": raw_token,
        "logprob_y": logprob_y,
        "logprob_n": logprob_n,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "missing_yn_in_toplogprobs": missing_yn,
        "invalid_first_token": invalid_first_token,
        "retried": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    args = parser.parse_args()

    print("="*60)
    print("Score Missing DialSeg711 Dialogues with GPT-5.2")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Dry run: {args.dry_run}")

    # Load data
    print("\nLoading data...")
    dialogues = load_dataset()
    cache = load_cache()
    cached_ids = get_cached_dialogue_ids(cache)

    print(f"  Total dialogues: {len(dialogues)}")
    print(f"  Already cached: {len(cached_ids)}")

    # Find missing dialogues
    all_ids = set(range(len(dialogues)))
    missing_ids = all_ids - cached_ids
    missing_dialogues = [d for d in dialogues if d.dialogue_id in missing_ids]

    print(f"  Missing dialogues: {len(missing_dialogues)}")

    if not missing_dialogues:
        print("\nNo missing dialogues to score!")
        return

    # Count boundaries to score
    total_boundaries = sum(d.num_user_turns - 1 for d in missing_dialogues)
    print(f"  Boundaries to score: {total_boundaries}")

    # Estimate cost
    est_input_tokens = total_boundaries * 150  # ~150 tokens per boundary
    est_output_tokens = total_boundaries * 5   # ~5 tokens per response
    est_cost = (est_input_tokens / 1000 * COST_PER_1K_INPUT +
                est_output_tokens / 1000 * COST_PER_1K_OUTPUT)
    print(f"\nEstimated cost: ${est_cost:.2f}")

    if args.dry_run:
        print("\nDry run complete. No API calls made.")
        return

    # Initialize OpenAI client
    import openai
    client = openai.OpenAI()

    print(f"\nScoring {len(missing_dialogues)} dialogues...")

    total_input_tokens = 0
    total_output_tokens = 0
    scored = 0
    start_time = time.time()

    for i, dialogue in enumerate(missing_dialogues):
        for position in range(1, dialogue.num_user_turns):
            result = score_boundary(client, dialogue, position, dry_run=False)

            # Build cache key
            user_turns = [m["content"] for m in dialogue.messages if m["role"] == "user"]
            context_before = [user_turns[j] for j in range(max(0, position - 4), position)]
            context_after = user_turns[position]
            user_prompt = build_user_prompt(context_before, context_after)
            prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)
            cache_key = f"dialseg711_{dialogue.dialogue_id}_{position}_{prompt_hash}"

            cache[cache_key] = result
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            scored += 1

            # Progress report every 100 boundaries
            if scored % 100 == 0:
                elapsed = time.time() - start_time
                cost_so_far = (total_input_tokens / 1000 * COST_PER_1K_INPUT +
                              total_output_tokens / 1000 * COST_PER_1K_OUTPUT)
                print(f"  Progress: {scored}/{total_boundaries} boundaries, "
                      f"elapsed: {elapsed:.1f}s, cost: ${cost_so_far:.2f}")

                # Save cache periodically
                save_cache(cache)

        # Progress by dialogue
        if (i + 1) % 10 == 0:
            print(f"  Completed dialogue {i+1}/{len(missing_dialogues)} (ID: {dialogue.dialogue_id})")

    # Final save
    save_cache(cache)

    # Final report
    elapsed = time.time() - start_time
    final_cost = (total_input_tokens / 1000 * COST_PER_1K_INPUT +
                  total_output_tokens / 1000 * COST_PER_1K_OUTPUT)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Scored: {scored} boundaries across {len(missing_dialogues)} dialogues")
    print(f"Time: {elapsed:.1f}s")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Cost: ${final_cost:.2f}")
    print(f"Cache saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
