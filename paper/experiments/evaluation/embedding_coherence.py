#!/usr/bin/env python3
"""
Embedding-Based Coherence Sanity Check for Dialogue Segmentations

This module provides an annotation-independent quality check for predicted
dialogue segmentations using pretrained sentence embeddings. It measures
whether predicted boundaries separate semantically distinct turn pairs.

Boundary Convention:
- Boundary position b is the between-turn index between turns u_b and u_{b+1}
- For each boundary b, the across-boundary pair is exactly (u_b, u_{b+1})

Method:
- For each predicted boundary b, extract across-boundary pairs (u_b, u_{b+1})
- Sample matched within-segment turn pairs at similar distances
- Compute cosine similarity for both sets using frozen sentence embeddings
- Report mean similarities, delta, and AUC for discriminating within vs across

This is a SANITY CHECK, not a primary evaluation metric. It validates that
predicted boundaries correspond to semantic discontinuities without requiring
gold annotations.

Usage:
    python -m paper.experiments.evaluation.embedding_coherence \\
        --datasets dialseg711 superseg tiage \\
        --k 3 --seed 0 --out results/embedding_sanity.csv [--audit]

Author: Paper experiments
"""

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
CACHE_DIR = PROJECT_ROOT / "paper" / "cache"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

# Default configuration
DEFAULT_K = 3  # Max distance for within-segment pairs
DEFAULT_SEED = 0
N_BOOTSTRAP = 2000
MIN_GAP = 2


@dataclass
class DialogueWithPredictions:
    """Dialogue with turn texts and predicted boundaries."""
    dialogue_id: int
    turns: List[str]  # All turn texts (user + assistant interleaved)
    user_turns: List[str]  # User turn texts only
    user_turn_indices: List[int]  # Indices of user turns in full dialogue
    predicted_boundaries: Set[int]  # Between-turn indices: b means gap between u_b and u_{b+1}
    num_user_turns: int


@dataclass
class CoherenceMetrics:
    """Coherence metrics for a dataset."""
    dataset: str
    n_dialogues: int
    n_boundaries: int
    n_across_pairs: int
    n_within_pairs: int
    mean_sim_within: float
    mean_sim_across: float
    delta: float  # within - across
    auc: float
    ci_delta_lower: float
    ci_delta_upper: float
    ci_auc_lower: float
    ci_auc_upper: float
    embedding_model: str


def load_dataset_turns(dataset_name: str) -> List[Dict]:
    """
    Load dataset dialogues with turn texts.

    Returns list of dicts with:
        - turns: List of all turn texts
        - user_turns: List of user turn texts only
        - user_turn_indices: Indices of user turns
    """
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Dataset not found: {test_file}")

    with open(test_file) as f:
        data = json.load(f)

    dialogues = []
    dial_data = data.get("dial_data", data)

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            all_turns = []
            user_turns = []
            user_indices = []

            for i, turn in enumerate(turns):
                content = turn.get("utterance", turn.get("text", ""))
                all_turns.append(content)
                if turn.get("role") == "user":
                    user_turns.append(content)
                    user_indices.append(i)

            dialogues.append({
                "turns": all_turns,
                "user_turns": user_turns,
                "user_turn_indices": user_indices,
            })

    return dialogues


def get_neural_predictions(
    dataset_name: str,
    dialogues: List[Dict],
    target_bor: float = 1.0,
    min_gap: int = MIN_GAP,
) -> List[Set[int]]:
    """
    Get predicted boundaries from neural model at target BOR.

    Returns boundaries as between-turn indices: b means gap between u_b and u_{b+1}.
    """
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

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

    # Get scores for each dialogue
    # Score at position p means "probability of boundary BEFORE turn p"
    all_scores = []

    for dialogue in tqdm(dialogues, desc=f"  Scoring {dataset_name}", leave=False):
        turns = dialogue["turns"]
        user_indices = dialogue["user_turn_indices"]
        scores = {}

        for user_idx_pos, global_idx in enumerate(user_indices):
            if user_idx_pos > 0:  # Can't have boundary before first user turn
                # Create context window
                window_start = max(0, global_idx - 8)
                window = turns[window_start:global_idx]

                # Format text
                context_parts = []
                for j, t in enumerate(window[-6:]):
                    abs_idx = window_start + len(window) - 6 + j
                    if abs_idx >= 0 and abs_idx in user_indices:
                        role = "user"
                    else:
                        role = "assistant"
                    context_parts.append(f"{role}: {t}")

                curr_content = turns[global_idx]
                text = " [SEP] ".join(context_parts) + f" [SEP] current: {curr_content}"

                encoding = tokenizer(
                    text, max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt"
                )

                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in encoding.items()}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits / temperature, dim=-1)
                    score = probs[0, 1].item()

                # Score at position p means boundary before turn p
                # Convert to between-turn index: boundary before turn p = between-turn index (p-1)
                scores[user_idx_pos] = score

        all_scores.append(scores)

    # Find threshold for target BOR
    all_scores_flat = []
    for scores in all_scores:
        all_scores_flat.extend(scores.values())

    if not all_scores_flat:
        return [set() for _ in dialogues]

    thresholds = np.percentile(all_scores_flat, np.linspace(100, 0, 200))

    best_threshold = 0.5
    best_bor_diff = float('inf')

    for tau in thresholds:
        total_pred = 0
        total_positions = 0

        for dialogue, scores in zip(dialogues, all_scores):
            num_user = len(dialogue["user_turns"])
            pred = _apply_threshold_with_gap(scores, tau, min_gap, num_user)
            total_pred += len(pred)
            total_positions += max(0, num_user - 1)

        if total_positions > 0:
            avg_pred = total_pred / len(dialogues)
            avg_possible = total_positions / len(dialogues)
            approx_bor = avg_pred / max(1, avg_possible / 3)

            if abs(approx_bor - target_bor) < best_bor_diff:
                best_bor_diff = abs(approx_bor - target_bor)
                best_threshold = tau

    # Apply best threshold and convert to between-turn indices
    all_predictions = []
    for dialogue, scores in zip(dialogues, all_scores):
        num_user = len(dialogue["user_turns"])
        # Scores are at positions 1, 2, ..., meaning boundary before that turn
        # Position p boundary -> between-turn index (p-1)
        pred_positions = _apply_threshold_with_gap(scores, best_threshold, min_gap, num_user)
        # Convert: position p -> between-turn index (p-1)
        pred_between_indices = {p - 1 for p in pred_positions if p > 0}
        all_predictions.append(pred_between_indices)

    return all_predictions


def _apply_threshold_with_gap(
    scores: Dict[int, float],
    threshold: float,
    min_gap: int,
    num_messages: int
) -> Set[int]:
    """Apply threshold with minimum gap enforcement."""
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


def get_embedding_model():
    """Load sentence embedding model, with fallback to TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model, "all-MiniLM-L6-v2"
    except ImportError:
        print("WARNING: sentence-transformers not available, using TF-IDF fallback")
        return None, "tfidf"


def embed_turns(
    turns: List[str],
    model,
    model_name: str,
    cache_key: str,
) -> np.ndarray:
    """Embed turns using sentence model or TF-IDF fallback. Caches embeddings."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"embeddings_{cache_key}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if model is not None:
        embeddings = model.encode(turns, convert_to_numpy=True, show_progress_bar=False)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vectorizer = TfidfVectorizer(max_features=512)
        embeddings = vectorizer.fit_transform(turns).toarray()
        embeddings = normalize(embeddings)

    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings


def compute_cache_key(dataset_name: str, dialogue_idx: int, turns: List[str]) -> str:
    """Compute cache key for embeddings."""
    content = f"{dataset_name}_{dialogue_idx}_" + "_".join(turns[:3])
    return hashlib.md5(content.encode()).hexdigest()[:16]


def extract_turn_pairs(
    dialogue: DialogueWithPredictions,
    k: int,
    rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract within-segment and across-boundary turn pairs.

    Boundary Convention:
    - Boundary b is the between-turn index between u_b and u_{b+1}
    - Across-boundary pair for boundary b is exactly (b, b+1) = (u_b, u_{b+1})

    Args:
        dialogue: Dialogue with predicted boundaries (as between-turn indices)
        k: Max distance for within-segment pairs
        rng: Random state for sampling

    Returns:
        (across_pairs, within_pairs) - lists of (i, j) index pairs into user_turns
    """
    n = dialogue.num_user_turns
    boundaries = dialogue.predicted_boundaries

    # Across-boundary pairs: (b, b+1) for each between-turn index b
    # Boundary b is between u_b and u_{b+1}, so across-pair is (b, b+1)
    across_pairs = []
    for b in sorted(boundaries):
        if b >= 0 and b + 1 < n:  # Valid: u_b and u_{b+1} both exist
            across_pairs.append((b, b + 1))

    if not across_pairs:
        return [], []

    # Build segment assignments
    # Segment changes AFTER turn b if b is a boundary
    # Turn i is in segment = count of boundaries with index < i
    sorted_bounds = sorted(boundaries)
    segment_ids = []
    seg = 0
    bound_ptr = 0
    for i in range(n):
        # Advance past boundaries that are < i
        while bound_ptr < len(sorted_bounds) and sorted_bounds[bound_ptr] < i:
            seg += 1
            bound_ptr += 1
        segment_ids.append(seg)

    # Within-segment pairs: match distance to across pairs (distance = 1)
    # All across-pairs have distance 1, so prefer within-pairs with distance 1
    within_pairs = []
    used_pairs = set()

    # First pass: find distance-1 within-pairs
    for seg_id in range(max(segment_ids) + 1):
        seg_indices = [i for i, s in enumerate(segment_ids) if s == seg_id]
        for idx in range(len(seg_indices) - 1):
            i, j = seg_indices[idx], seg_indices[idx + 1]
            if abs(i - j) == 1:  # Distance 1
                within_pairs.append((i, j))
                used_pairs.add((i, j))

    # Truncate or extend to match across count
    if len(within_pairs) > len(across_pairs):
        # Randomly sample to match
        indices = rng.choice(len(within_pairs), size=len(across_pairs), replace=False)
        within_pairs = [within_pairs[i] for i in sorted(indices)]
    elif len(within_pairs) < len(across_pairs):
        # Need more pairs - sample from distance 2..k
        additional = []
        for seg_id in range(max(segment_ids) + 1):
            seg_indices = [i for i, s in enumerate(segment_ids) if s == seg_id]
            for ii in range(len(seg_indices)):
                for jj in range(ii + 1, len(seg_indices)):
                    i, j = seg_indices[ii], seg_indices[jj]
                    dist = abs(i - j)
                    if 1 < dist <= k and (i, j) not in used_pairs:
                        additional.append((i, j))

        need = len(across_pairs) - len(within_pairs)
        if additional and need > 0:
            sample_size = min(need, len(additional))
            sampled = rng.choice(len(additional), size=sample_size, replace=False)
            within_pairs.extend([additional[i] for i in sampled])

    return across_pairs, within_pairs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / (norm + 1e-8)


def compute_auc(within_sims: List[float], across_sims: List[float]) -> float:
    """
    Compute AUC for discriminating within vs across using similarity.
    Higher similarity should predict "within" (label=1).
    """
    if not within_sims or not across_sims:
        return 0.5

    n_pos = len(within_sims)
    n_neg = len(across_sims)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending, count concordant pairs
    pairs = [(s, 1) for s in within_sims] + [(s, 0) for s in across_sims]
    pairs.sort(key=lambda x: -x[0])

    concordant = 0
    pos_seen = 0

    for score, label in pairs:
        if label == 1:
            pos_seen += 1
        else:
            concordant += pos_seen

    auc = concordant / (n_pos * n_neg)
    return auc


def audit_auc_symmetry(within_sims: List[float], across_sims: List[float]) -> bool:
    """
    Audit A: Verify AUC symmetry property.
    AUC with labels swapped should equal 1 - AUC.
    """
    if not within_sims or not across_sims:
        return True

    auc_normal = compute_auc(within_sims, across_sims)
    auc_swapped = compute_auc(across_sims, within_sims)  # Swap labels

    expected = 1 - auc_normal
    diff = abs(auc_swapped - expected)

    # Use 1e-3 tolerance to handle floating-point precision issues with tied scores
    if diff > 1e-3:
        raise ValueError(
            f"AUC symmetry check failed: AUC={auc_normal:.6f}, "
            f"swapped={auc_swapped:.6f}, expected 1-AUC={expected:.6f}, diff={diff:.6f}"
        )
    return True


def compute_distance_distribution(pairs: List[Tuple[int, int]], k: int) -> Dict[int, int]:
    """Compute distribution of |i-j| distances for pairs."""
    dist_counts = {d: 0 for d in range(1, k + 2)}
    for (i, j) in pairs:
        d = abs(i - j)
        if d in dist_counts:
            dist_counts[d] += 1
        else:
            dist_counts[d] = 1
    return dist_counts


def bootstrap_ci(
    dialogues: List[Dict],
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Bootstrap confidence intervals for delta and AUC.

    Returns:
        (mean_within, mean_across, delta, auc,
         ci_delta_lower, ci_delta_upper, ci_auc_lower, ci_auc_upper)
    """
    rng = np.random.RandomState(seed)

    # Collect dialogue-level data
    dialogue_results = []
    for d in dialogues:
        ws = d.get("within_sims", [])
        acs = d.get("across_sims", [])
        if not ws and not acs:
            continue

        mean_w = np.mean(ws) if ws else 0.5
        mean_a = np.mean(acs) if acs else 0.5
        delta = mean_w - mean_a
        auc = compute_auc(ws, acs)

        dialogue_results.append({
            "mean_within": mean_w,
            "mean_across": mean_a,
            "delta": delta,
            "auc": auc,
            "n_within": len(ws),
            "n_across": len(acs),
        })

    if not dialogue_results:
        return 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5

    # Overall metrics (weighted)
    total_within = sum(r["n_within"] for r in dialogue_results)
    total_across = sum(r["n_across"] for r in dialogue_results)

    overall_within = sum(r["mean_within"] * r["n_within"] for r in dialogue_results) / max(total_within, 1)
    overall_across = sum(r["mean_across"] * r["n_across"] for r in dialogue_results) / max(total_across, 1)
    overall_delta = overall_within - overall_across

    # Aggregate AUC
    all_within_sims = []
    all_across_sims = []
    for d in dialogues:
        all_within_sims.extend(d.get("within_sims", []))
        all_across_sims.extend(d.get("across_sims", []))
    overall_auc = compute_auc(all_within_sims, all_across_sims)

    # Bootstrap
    n_dialogues = len(dialogue_results)
    bootstrap_deltas = []
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_dialogues, size=n_dialogues, replace=True)
        sample = [dialogue_results[i] for i in indices]

        sample_within = sum(r["n_within"] for r in sample)
        sample_across = sum(r["n_across"] for r in sample)

        if sample_within > 0 and sample_across > 0:
            sw = sum(r["mean_within"] * r["n_within"] for r in sample) / sample_within
            sa = sum(r["mean_across"] * r["n_across"] for r in sample) / sample_across
            bootstrap_deltas.append(sw - sa)
            bootstrap_aucs.append(np.mean([r["auc"] for r in sample]))

    if bootstrap_deltas:
        ci_delta_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_delta_upper = np.percentile(bootstrap_deltas, 97.5)
    else:
        ci_delta_lower = ci_delta_upper = overall_delta

    if bootstrap_aucs:
        ci_auc_lower = np.percentile(bootstrap_aucs, 2.5)
        ci_auc_upper = np.percentile(bootstrap_aucs, 97.5)
    else:
        ci_auc_lower = ci_auc_upper = overall_auc

    return (overall_within, overall_across, overall_delta, overall_auc,
            ci_delta_lower, ci_delta_upper, ci_auc_lower, ci_auc_upper)


def evaluate_dataset(
    dataset_name: str,
    k: int,
    seed: int,
    embedding_model,
    model_name: str,
    audit: bool = False,
) -> Tuple[Optional[CoherenceMetrics], Optional[Dict]]:
    """
    Evaluate embedding coherence for a dataset.

    Returns:
        (metrics, audit_data) where audit_data is None if audit=False
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print('='*60)

    # Load dialogues
    try:
        dialogues_raw = load_dataset_turns(dataset_name)
        print(f"  Loaded {len(dialogues_raw)} dialogues")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None, None

    # Get predictions
    predictions = get_neural_predictions(dataset_name, dialogues_raw)

    # Create dialogue objects
    rng = np.random.RandomState(seed)
    dialogues = []

    for i, (d, pred) in enumerate(zip(dialogues_raw, predictions)):
        dialogues.append(DialogueWithPredictions(
            dialogue_id=i,
            turns=d["turns"],
            user_turns=d["user_turns"],
            user_turn_indices=d["user_turn_indices"],
            predicted_boundaries=pred,
            num_user_turns=len(d["user_turns"]),
        ))

    # Extract pairs and compute similarities
    total_across = 0
    total_within = 0
    all_across_distances = []
    all_within_distances = []

    # For audit C: sample dialogues
    audit_sample_dialogues = []
    if audit:
        sample_rng = np.random.RandomState(seed)
        sample_indices = sample_rng.choice(
            len(dialogues), size=min(5, len(dialogues)), replace=False
        )
        audit_sample_dialogues = list(sample_indices)

    audit_pair_dumps = []

    for dialogue in tqdm(dialogues, desc="  Processing dialogues", leave=False):
        across_pairs, within_pairs = extract_turn_pairs(dialogue, k, rng)

        if not across_pairs:
            dialogue.within_sims = []
            dialogue.across_sims = []
            dialogue.across_pairs = []
            dialogue.within_pairs = []
            continue

        # Track distances for audit B
        for (i, j) in across_pairs:
            all_across_distances.append(abs(i - j))
        for (i, j) in within_pairs:
            all_within_distances.append(abs(i - j))

        # Embed user turns
        cache_key = compute_cache_key(dataset_name, dialogue.dialogue_id, dialogue.user_turns)
        embeddings = embed_turns(dialogue.user_turns, embedding_model, model_name, cache_key)

        # Compute similarities
        within_sims = []
        for (i, j) in within_pairs:
            if i < len(embeddings) and j < len(embeddings):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                within_sims.append(sim)

        across_sims = []
        for (i, j) in across_pairs:
            if i < len(embeddings) and j < len(embeddings):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                across_sims.append(sim)

        dialogue.within_sims = within_sims
        dialogue.across_sims = across_sims
        dialogue.across_pairs = across_pairs
        dialogue.within_pairs = within_pairs
        total_within += len(within_sims)
        total_across += len(across_sims)

        # Audit C: dump pairs for sample dialogues
        if audit and dialogue.dialogue_id in audit_sample_dialogues:
            dump = {
                "dialogue_id": dialogue.dialogue_id,
                "num_user_turns": dialogue.num_user_turns,
                "boundaries": sorted(dialogue.predicted_boundaries),
                "within_pairs": [],
                "across_pairs": [],
            }
            for idx, (i, j) in enumerate(within_pairs[:10]):
                if i < len(dialogue.user_turns) and j < len(dialogue.user_turns):
                    dump["within_pairs"].append({
                        "indices": (i, j),
                        "sim": within_sims[idx] if idx < len(within_sims) else None,
                        "u_i": dialogue.user_turns[i][:100],
                        "u_j": dialogue.user_turns[j][:100],
                    })
            for idx, (i, j) in enumerate(across_pairs[:10]):
                if i < len(dialogue.user_turns) and j < len(dialogue.user_turns):
                    dump["across_pairs"].append({
                        "indices": (i, j),
                        "sim": across_sims[idx] if idx < len(across_sims) else None,
                        "u_i": dialogue.user_turns[i][:100],
                        "u_j": dialogue.user_turns[j][:100],
                    })
            audit_pair_dumps.append(dump)

    print(f"  Total pairs: {total_within} within, {total_across} across")

    # Audit A: AUC symmetry
    if audit:
        print("  Running Audit A (AUC symmetry)...")
        all_within_sims = []
        all_across_sims = []
        for d in dialogues:
            all_within_sims.extend(getattr(d, "within_sims", []))
            all_across_sims.extend(getattr(d, "across_sims", []))
        audit_auc_symmetry(all_within_sims, all_across_sims)
        print("    PASSED")

    # Audit B: Distance distribution
    audit_data = None
    if audit:
        print("  Running Audit B (distance matching)...")
        across_dist = compute_distance_distribution(
            [(i, j) for d in dialogues for (i, j) in getattr(d, "across_pairs", [])], k
        )
        within_dist = compute_distance_distribution(
            [(i, j) for d in dialogues for (i, j) in getattr(d, "within_pairs", [])], k
        )
        print(f"    Across-pair distances: {across_dist}")
        print(f"    Within-pair distances: {within_dist}")

        audit_data = {
            "across_distances": across_dist,
            "within_distances": within_dist,
            "pair_dumps": audit_pair_dumps,
        }

    # Compute metrics with bootstrap CI
    print("  Computing bootstrap CI...")
    dialogue_data = []
    for d in dialogues:
        dialogue_data.append({
            "within_sims": getattr(d, "within_sims", []),
            "across_sims": getattr(d, "across_sims", []),
        })

    (mean_within, mean_across, delta, auc,
     ci_delta_lower, ci_delta_upper,
     ci_auc_lower, ci_auc_upper) = bootstrap_ci(dialogue_data, N_BOOTSTRAP, seed)

    n_boundaries = sum(len(d.predicted_boundaries) for d in dialogues)

    print(f"  Mean sim (within):  {mean_within:.4f}")
    print(f"  Mean sim (across):  {mean_across:.4f}")
    print(f"  Delta:              {delta:.4f} [{ci_delta_lower:.4f}, {ci_delta_upper:.4f}]")
    print(f"  AUC:                {auc:.4f} [{ci_auc_lower:.4f}, {ci_auc_upper:.4f}]")

    metrics = CoherenceMetrics(
        dataset=dataset_name,
        n_dialogues=len(dialogues),
        n_boundaries=n_boundaries,
        n_across_pairs=total_across,
        n_within_pairs=total_within,
        mean_sim_within=mean_within,
        mean_sim_across=mean_across,
        delta=delta,
        auc=auc,
        ci_delta_lower=ci_delta_lower,
        ci_delta_upper=ci_delta_upper,
        ci_auc_lower=ci_auc_lower,
        ci_auc_upper=ci_auc_upper,
        embedding_model=model_name,
    )

    return metrics, audit_data


def generate_latex_table(results: List[CoherenceMetrics]) -> str:
    """Generate LaTeX tabular snippet with confidence intervals."""
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & $N$ & $\bar{s}_\text{within}$ & $\bar{s}_\text{across}$ & $\Delta$ [95\% CI] & AUC [95\% CI] \\",
        r"\midrule",
    ]

    for r in results:
        # Format dataset name
        ds_name = r.dataset.replace("_", r"\_")
        if ds_name == "dialseg711":
            ds_name = "DialSeg711"
        elif ds_name == "superseg":
            ds_name = "SuperSeg"
        elif ds_name == "tiage":
            ds_name = "TIAGE"

        # Format delta and AUC with CIs
        delta_str = f"{r.delta:.3f} [{r.ci_delta_lower:.3f}, {r.ci_delta_upper:.3f}]"
        auc_str = f"{r.auc:.3f} [{r.ci_auc_lower:.3f}, {r.ci_auc_upper:.3f}]"

        lines.append(
            f"{ds_name} & {r.n_dialogues} & {r.mean_sim_within:.3f} & "
            f"{r.mean_sim_across:.3f} & {delta_str} & {auc_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines)


def generate_method_paragraph() -> str:
    """Generate method description paragraph."""
    return """
\\paragraph{Embedding Coherence Sanity Check.}
As an annotation-independent validation, we measure whether predicted boundaries
separate semantically distinct turn pairs. For each predicted boundary position $b$
(the between-turn index between $u_b$ and $u_{b+1}$), we extract across-boundary
pairs $(u_b, u_{b+1})$ and sample matched within-segment pairs $(u_i, u_j)$ where
$|i - j| \\le 3$. Using frozen sentence embeddings (all-MiniLM-L6-v2), we compute
cosine similarities for both sets. We report the mean within-segment similarity
$\\bar{s}_{\\text{within}}$, mean across-boundary similarity $\\bar{s}_{\\text{across}}$,
their difference $\\Delta$, and the AUC for discriminating within vs.\\ across pairs
using similarity as the score. Positive $\\Delta$ and AUC $> 0.5$ indicate that
predicted boundaries correspond to semantic discontinuities. This is a sanity check
validating segmentation quality without gold annotations; it does not replace
standard evaluation metrics.
""".strip()


def write_audit_files(
    dataset_name: str,
    audit_data: Dict,
    out_dir: Path,
):
    """Write audit files for a dataset."""
    # Audit C: Pair dumps
    dump_path = out_dir / f"embedding_pair_dumps_{dataset_name}.txt"
    with open(dump_path, 'w') as f:
        f.write(f"# Pair Dumps for {dataset_name}\n")
        f.write(f"# Sampled dialogues with within-segment and across-boundary pairs\n\n")

        for dump in audit_data.get("pair_dumps", []):
            f.write(f"=" * 60 + "\n")
            f.write(f"Dialogue {dump['dialogue_id']}\n")
            f.write(f"  Num user turns: {dump['num_user_turns']}\n")
            f.write(f"  Boundaries (between-turn indices): {dump['boundaries']}\n")
            f.write("\n  WITHIN-SEGMENT PAIRS:\n")
            for p in dump["within_pairs"]:
                f.write(f"    ({p['indices'][0]}, {p['indices'][1]}) sim={p['sim']:.4f}\n")
                f.write(f"      u_{p['indices'][0]}: {p['u_i']}\n")
                f.write(f"      u_{p['indices'][1]}: {p['u_j']}\n")
            f.write("\n  ACROSS-BOUNDARY PAIRS:\n")
            for p in dump["across_pairs"]:
                f.write(f"    ({p['indices'][0]}, {p['indices'][1]}) sim={p['sim']:.4f}\n")
                f.write(f"      u_{p['indices'][0]}: {p['u_i']}\n")
                f.write(f"      u_{p['indices'][1]}: {p['u_j']}\n")
            f.write("\n")

    print(f"  Wrote pair dumps: {dump_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based coherence sanity check for dialogue segmentations"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dialseg711", "superseg", "tiage"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Max distance for within-segment pairs (default: {DEFAULT_K})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/embedding_sanity.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Enable audit checks (A: AUC symmetry, B: distance matching, C: pair dumps)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Embedding-Based Coherence Sanity Check")
    print("=" * 60)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"k (max within-pair distance): {args.k}")
    print(f"Seed: {args.seed}")
    print(f"Audit mode: {args.audit}")

    # Load embedding model
    embedding_model, model_name = get_embedding_model()
    print(f"Embedding model: {model_name}")

    # Evaluate each dataset
    results = []
    all_audit_data = {}

    for dataset in args.datasets:
        metrics, audit_data = evaluate_dataset(
            dataset,
            args.k,
            args.seed,
            embedding_model,
            model_name,
            audit=args.audit,
        )
        if metrics is not None:
            results.append(metrics)
        if audit_data is not None:
            all_audit_data[dataset] = audit_data

    if not results:
        print("\nNo results to save.")
        return

    # Save CSV
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_path, index=False)
    print(f"\nSaved CSV: {out_path}")

    # Generate LaTeX
    latex_table = generate_latex_table(results)
    latex_path = out_path.with_suffix(".tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX: {latex_path}")

    # Write audit files
    if args.audit:
        audit_summary_path = out_path.parent / "embedding_sanity_audit.txt"
        with open(audit_summary_path, 'w') as f:
            f.write("# Embedding Coherence Audit Summary\n\n")
            for dataset, audit_data in all_audit_data.items():
                f.write(f"## {dataset}\n")
                f.write(f"Across-pair distances: {audit_data['across_distances']}\n")
                f.write(f"Within-pair distances: {audit_data['within_distances']}\n\n")
        print(f"Saved audit summary: {audit_summary_path}")

        for dataset, audit_data in all_audit_data.items():
            write_audit_files(dataset, audit_data, out_path.parent)

    # Print outputs
    print("\n" + "=" * 60)
    print("CSV CONTENT:")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("LATEX TABLE:")
    print("=" * 60)
    print(latex_table)

    print("\n" + "=" * 60)
    print("METHOD PARAGRAPH:")
    print("=" * 60)
    print(generate_method_paragraph())


if __name__ == "__main__":
    main()
