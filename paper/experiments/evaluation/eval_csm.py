#!/usr/bin/env python3
"""
Evaluate CSM (Coherence Scoring Model) on dialogue segmentation datasets
using our metric framework (W-F1, BOR, Purity, Coverage).

This script:
1. Runs CSM with NSP mode (bert-base-uncased) to get segment predictions
2. Converts CSM's segment format to boundary indices
3. Computes our standard metrics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Add CSM repo to path
CSM_REPO = Path("/tmp/Dialogue-Topic-Segmenter")
sys.path.insert(0, str(CSM_REPO))

# Add project root for our metrics
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_csm_data(filepath: str) -> List[Dict]:
    """Load CSM format dataset."""
    with open(filepath) as f:
        return json.load(f)


def segments_to_boundaries(segment_sizes: List[int]) -> Set[int]:
    """
    Convert segment sizes to boundary indices.

    CSM format: segments = [4, 6, 3] means 3 segments of those sizes.
    Boundaries are at cumulative positions: after 4, after 10 (4+6).
    The last segment doesn't have a boundary after it.

    Returns set of boundary indices (0-indexed, position AFTER which boundary occurs).
    """
    boundaries = set()
    cumsum = 0
    for i, size in enumerate(segment_sizes[:-1]):  # Don't add boundary after last segment
        cumsum += size
        boundaries.add(cumsum - 1)  # 0-indexed: boundary at end of segment
    return boundaries


def boundaries_to_segments(boundaries: Set[int], num_utterances: int) -> List[Set[int]]:
    """
    Convert boundary positions to segment membership sets.

    Args:
        boundaries: Set of positions where boundaries occur (0-indexed)
        num_utterances: Total number of utterances

    Returns:
        List of sets, where each set contains utterance indices in that segment.
    """
    sorted_bounds = sorted(boundaries)
    segments = []
    prev = 0

    for bound in sorted_bounds:
        segment = set(range(prev, bound + 1))
        segments.append(segment)
        prev = bound + 1

    # Last segment
    if prev < num_utterances:
        segments.append(set(range(prev, num_utterances)))

    return segments


def run_csm_evaluation(dataset_path: str, encoder: str = "bert-base-uncased", mode: str = "NSP"):
    """
    Run CSM evaluation and collect per-dialogue predictions.

    Returns:
        List of dicts with 'gold_segments', 'pred_segments', 'num_utterances' per dialogue
    """
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForNextSentencePrediction
    from neural_texttiling import TextTiling
    from tqdm import tqdm

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if mode == 'SC':
        text_encoder = AutoModel.from_pretrained(encoder).to(device)
    elif mode == 'NSP':
        text_encoder = AutoModelForNextSentencePrediction.from_pretrained(encoder).to(device)
    else:
        raise ValueError(f"Mode {mode} not supported without checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(encoder)

    # Load data
    dialogue_data = load_csm_data(dataset_path)
    dev_data = [d for d in dialogue_data if d.get('set') == 'dev']
    test_data = [d for d in dialogue_data if d.get('set') != 'dev']

    # Alpha search on dev set
    print(f"Searching for best alpha on {len(dev_data)} dev dialogues...")
    best_alpha = None
    best_pk = float('inf')

    for alpha in tqdm(np.arange(-2, 2, 0.1), desc='Alpha search'):
        total_pk = 0
        for dialogue in dev_data:
            pk, _, _, _ = TextTiling(
                dialogue['utterances'], dialogue['segments'],
                text_encoder, tokenizer, alpha, mode, device
            )
            total_pk += pk
        mean_pk = total_pk / len(dev_data)
        if mean_pk < best_pk:
            best_pk = mean_pk
            best_alpha = alpha

    print(f"Best alpha: {best_alpha:.2f}")

    # Evaluate test set and collect predictions
    results = []
    print(f"Evaluating {len(test_data)} test dialogues...")

    for dialogue in tqdm(test_data, desc='Evaluating'):
        pk, wd, f1, pred_segments = TextTiling(
            dialogue['utterances'], dialogue['segments'],
            text_encoder, tokenizer, best_alpha, mode, device
        )

        results.append({
            'gold_segments': dialogue['segments'],
            'pred_segments': pred_segments,
            'num_utterances': len(dialogue['utterances']),
            'csm_pk': pk,
            'csm_wd': wd,
            'csm_f1': f1,
        })

    return results


def compute_our_metrics(results: List[Dict]) -> Dict:
    """
    Compute ALL metrics from our evaluation pipeline.

    This ensures consistent boundary indexing across all metrics.
    We do NOT use CSM's own F1 computation (which is macro F1 on binary
    position labels - a different metric entirely).
    """
    from episodic.topics.evaluation import (
        compute_windowed_metrics,
        compute_bor,
        compute_purity_coverage,
    )

    total_gold = 0
    total_pred = 0
    total_tp = 0  # For strict F1: exact boundary matches

    # Per-dialogue accumulators for macro-averaging
    strict_prec_sum, strict_rec_sum, strict_f1_sum = 0, 0, 0
    w_prec_sum, w_rec_sum, w_f1_sum = 0, 0, 0
    purity_sum, coverage_sum = 0, 0

    for r in results:
        # Convert segment sizes to boundaries
        gold_bounds = segments_to_boundaries(r['gold_segments'])
        pred_bounds = segments_to_boundaries(r['pred_segments'])

        num_utt = r['num_utterances']

        # Convert to segment membership for purity/coverage
        gold_segs = boundaries_to_segments(gold_bounds, num_utt)
        pred_segs = boundaries_to_segments(pred_bounds, num_utt)

        # STRICT F1: exact set intersection (no tolerance)
        tp = len(gold_bounds & pred_bounds)
        strict_prec = tp / len(pred_bounds) if pred_bounds else 0.0
        strict_rec = tp / len(gold_bounds) if gold_bounds else 0.0
        strict_f1 = 2 * strict_prec * strict_rec / (strict_prec + strict_rec) if (strict_prec + strict_rec) > 0 else 0.0

        strict_prec_sum += strict_prec
        strict_rec_sum += strict_rec
        strict_f1_sum += strict_f1
        total_tp += tp

        # Windowed metrics (window=1 tolerance)
        w_prec, w_rec, w_f1 = compute_windowed_metrics(gold_bounds, pred_bounds, num_utt, window=1)
        w_prec_sum += w_prec
        w_rec_sum += w_rec
        w_f1_sum += w_f1

        # Purity/coverage
        purity, coverage = compute_purity_coverage(gold_segs, pred_segs)
        purity_sum += purity
        coverage_sum += coverage

        total_gold += len(gold_bounds)
        total_pred += len(pred_bounds)

    n = len(results)

    # Micro-averaged strict metrics (across all boundaries)
    micro_strict_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_strict_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_strict_f1 = 2 * micro_strict_prec * micro_strict_rec / (micro_strict_prec + micro_strict_rec) if (micro_strict_prec + micro_strict_rec) > 0 else 0.0

    return {
        'n_dialogues': n,
        'total_gold_boundaries': total_gold,
        'total_pred_boundaries': total_pred,
        'total_true_positives': total_tp,
        'bor': compute_bor(total_gold, total_pred),
        # Strict boundary F1 (micro-averaged)
        'strict_precision': micro_strict_prec,
        'strict_recall': micro_strict_rec,
        'strict_f1': micro_strict_f1,
        # Strict F1 (macro-averaged per dialogue)
        'strict_f1_macro': strict_f1_sum / n,
        # Windowed F1 (macro-averaged per dialogue)
        'w_precision': w_prec_sum / n,
        'w_recall': w_rec_sum / n,
        'w_f1': w_f1_sum / n,
        # Segment quality
        'purity': purity_sum / n,
        'coverage': coverage_sum / n,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CSM with our metrics")
    parser.add_argument("--dataset", type=str, default="dialseg_711",
                        choices=["dialseg_711", "tiage", "doc2dial"])
    parser.add_argument("--encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--mode", type=str, default="NSP", choices=["SC", "NSP"])
    args = parser.parse_args()

    # Dataset path
    dataset_path = CSM_REPO / "data" / "eval" / f"{args.dataset}.json"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"CSM Evaluation: {args.dataset}")
    print(f"Encoder: {args.encoder}, Mode: {args.mode}")
    print(f"=" * 60)

    # Run CSM and get predictions
    results = run_csm_evaluation(str(dataset_path), args.encoder, args.mode)

    # Compute our metrics
    metrics = compute_our_metrics(results)

    # Report
    print(f"\n{'=' * 60}")
    print("RESULTS (all metrics from our evaluation pipeline)")
    print(f"{'=' * 60}")
    print(f"\nDataset: {args.dataset}")
    print(f"Dialogues: {metrics['n_dialogues']}")
    print(f"Total gold boundaries: {metrics['total_gold_boundaries']}")
    print(f"Total pred boundaries: {metrics['total_pred_boundaries']}")
    print(f"Total exact matches (TP): {metrics['total_true_positives']}")

    print(f"\n--- Boundary Detection Metrics ---")
    print(f"Strict F1 (micro):  {metrics['strict_f1']:.3f}")
    print(f"Strict Precision:   {metrics['strict_precision']:.3f}")
    print(f"Strict Recall:      {metrics['strict_recall']:.3f}")
    print(f"Strict F1 (macro):  {metrics['strict_f1_macro']:.3f}")

    print(f"\n--- Windowed Metrics (±1 tolerance) ---")
    print(f"W-F1:       {metrics['w_f1']:.3f}")
    print(f"W-Prec:     {metrics['w_precision']:.3f}")
    print(f"W-Recall:   {metrics['w_recall']:.3f}")

    print(f"\n--- Granularity & Segment Quality ---")
    print(f"BOR:        {metrics['bor']:.3f}")
    print(f"Purity:     {metrics['purity']:.3f}")
    print(f"Coverage:   {metrics['coverage']:.3f}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    bor = metrics['bor']
    if bor < 0.8:
        regime = "CONSERVATIVE (under-segmenting)"
    elif bor > 1.2:
        regime = "AGGRESSIVE (over-segmenting)"
    else:
        regime = "BALANCED"
    print(f"Regime: {regime}")

    # Sanity check: W-F1 should be >= strict F1
    if metrics['w_f1'] < metrics['strict_f1']:
        print(f"\n⚠️  WARNING: W-F1 < Strict F1 - this should not happen!")
        print(f"    Check boundary indexing consistency.")
    else:
        print(f"\n✓ Sanity check passed: W-F1 ({metrics['w_f1']:.3f}) >= Strict F1 ({metrics['strict_f1']:.3f})")

    # LaTeX table row
    print(f"\n--- LaTeX Table Row ---")
    print(f"CSM ({args.mode}) & {args.dataset} & "
          f"{metrics['strict_f1']:.3f} & {metrics['w_f1']:.3f} & "
          f"{metrics['bor']:.2f} & {metrics['purity']:.3f} & {metrics['coverage']:.3f} \\\\")


if __name__ == "__main__":
    main()
