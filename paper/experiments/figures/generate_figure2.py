#!/usr/bin/env python3
"""Generate figure demonstrating adaptive commitment behavior.

2x2 layout showing:
- Columns: Fine base scoring | Coarse base scoring
- Rows: Rolling candidate rate | Commit distribution (density)

Key insight: Fine vs coarse produces different candidate distributions.
Adaptive commitment normalizes commit rate, but spatial distribution differs.

I don't love this figure because it is easy to misinterpret as
trivial. Searching for a better example.

"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add project root to path for episodic imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from episodic.topics.strategies.neural_strategy import NeuralStrategy

# Granularity thresholds (from calibration.py)
GRANULARITY_THRESHOLDS = {'fine': 0.3, 'medium': 0.5, 'coarse': 0.7}


def load_dialseg711(max_dialogues: int = 100, datasets_dir: Path = None) -> List[List[Dict[str, Any]]]:
    """Load DialSeg711 dialogues."""
    if datasets_dir is None:
        datasets_dir = project_root / "datasets"
    path = datasets_dir / "dialseg711" / "segmentation_file_test.json"
    if not path.exists():
        print(f"Dataset not found: {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    dialogues = []
    role_map = {'user': 'user', 'agent': 'assistant'}

    for dataset_key, dial_list in data.get('dial_data', {}).items():
        for dialogue in dial_list[:max_dialogues]:
            turns = dialogue.get('turns', [])
            if len(turns) < 4:
                continue

            messages = []
            for turn in turns:
                role = role_map.get(turn.get('role', 'user'), 'user')
                content = turn.get('utterance', '')
                messages.append({'role': role, 'content': content})

            dialogues.append(messages)

    return dialogues


def run_with_threshold(
    dialogues: List[List[Dict]],
    granularity: str,
    commit_threshold: float,
    min_gap: int = 2,
) -> Dict[str, Any]:
    """
    Run with fixed threshold and min_gap spacing.

    This simulates what an adaptive controller would produce after converging:
    - Base scorer produces confidence scores
    - Commits fire when confidence >= commit_threshold
    - min_gap prevents rapid-fire commits

    Returns per-message metrics and global commit indices.
    """
    base = NeuralStrategy({'granularity': granularity})
    candidate_threshold = GRANULARITY_THRESHOLDS[granularity]

    metrics = {
        'message_idx': [],
        'boundary_committed': [],
        'is_candidate': [],
        'confidence_scores': [],
    }

    # Track global commit indices (message stream position)
    commit_indices = []
    dialogue_lengths = []

    global_msg_idx = 0

    for dialogue in dialogues:
        dialogue_len = len(dialogue)
        dialogue_lengths.append(dialogue_len)
        message_history = []
        local_idx = 0
        last_commit_idx = -min_gap - 1  # Allow first commit

        for msg in dialogue:
            if msg['role'] == 'user' and len(message_history) >= 2:
                decision = base.get_decision(
                    query=msg['content'],
                    messages=message_history,
                    current_thread=None
                )

                confidence = decision.confidence_score
                metrics['message_idx'].append(global_msg_idx)
                metrics['confidence_scores'].append(confidence)

                # Candidate = would trigger at granularity threshold
                is_candidate = confidence >= candidate_threshold
                metrics['is_candidate'].append(1 if is_candidate else 0)

                # Commit = exceeds commit_threshold AND respects min_gap
                can_commit = (local_idx - last_commit_idx) > min_gap
                is_commit = confidence >= commit_threshold and can_commit

                metrics['boundary_committed'].append(1 if is_commit else 0)

                if is_commit:
                    commit_indices.append(global_msg_idx)
                    last_commit_idx = local_idx

            message_history.append(msg)
            global_msg_idx += 1
            local_idx += 1

    # Compute summary stats
    total_commits = sum(metrics['boundary_committed'])
    total_candidates = sum(metrics['is_candidate'])
    n_samples = len(metrics['boundary_committed'])

    return {
        'metrics': metrics,
        'total_commits': total_commits,
        'total_candidates': total_candidates,
        'candidate_rate': total_candidates / n_samples if n_samples > 0 else 0,
        'commit_rate': total_commits / n_samples if n_samples > 0 else 0,
        'commit_indices': commit_indices,
        'dialogue_lengths': dialogue_lengths,
        'n_samples': n_samples,
    }


def find_threshold_for_rate(
    dialogues: List[List[Dict]],
    granularity: str,
    target_rate: float,
    min_gap: int = 2,
) -> Tuple[float, Dict]:
    """
    Binary search to find commit_threshold that produces target_rate.

    This simulates what the adaptive controller would converge to.
    """
    low, high = 0.1, 0.95
    best_threshold = 0.5
    best_result = None
    best_diff = float('inf')

    for _ in range(15):
        mid = (low + high) / 2
        result = run_with_threshold(dialogues, granularity, mid, min_gap)
        rate = result['commit_rate']
        diff = abs(rate - target_rate)

        if diff < best_diff:
            best_diff = diff
            best_threshold = mid
            best_result = result

        if rate > target_rate:
            low = mid  # Higher threshold = fewer commits
        else:
            high = mid  # Lower threshold = more commits

    return best_threshold, best_result


def compute_rolling_rate(values: List[int], window: int) -> np.ndarray:
    """Compute rolling rate (fraction of 1s in window)."""
    arr = np.array(values, dtype=float)
    rolling = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        rolling.append(np.mean(arr[start:i+1]))
    return np.array(rolling)


def plot_2x2_figure(
    results: Dict[str, Dict],
    target_rate: float,
    output_path: str,
):
    """
    Generate 2x2 figure.

    Rows: Rolling candidate rate | Commit density
    Columns: Fine | Coarse

    Both rows use same x-axis: message index (across dialogues).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.rcParams.update({'font.size': 11})

    window = 50
    granularities = ['fine', 'coarse']
    col_labels = ['Fine scoring', 'Coarse scoring']

    # Get global x-axis range (same for both rows)
    all_msg_idx = []
    for gran in granularities:
        all_msg_idx.extend(results[gran]['metrics']['message_idx'])
    x_max = max(all_msg_idx) if all_msg_idx else 100

    # Compute y-limits for top row (candidate rate)
    all_candidate_rates = []
    for gran in granularities:
        result = results[gran]
        rolling_cand = compute_rolling_rate(result['metrics']['is_candidate'], window)
        all_candidate_rates.extend(rolling_cand)
    cand_y_max = min(max(all_candidate_rates) * 1.1, 1.0) if all_candidate_rates else 0.5

    # Compute KDE for bottom row using global message indices
    kde_fine = None
    kde_coarse = None
    x_kde = np.linspace(0, x_max, 200)

    if results['fine']['commit_indices']:
        kde_fine = stats.gaussian_kde(results['fine']['commit_indices'], bw_method='scott')
    if results['coarse']['commit_indices']:
        kde_coarse = stats.gaussian_kde(results['coarse']['commit_indices'], bw_method='scott')

    # Compute density y-limit
    density_y_max = 0
    if kde_fine is not None:
        density_y_max = max(density_y_max, kde_fine(x_kde).max())
    if kde_coarse is not None:
        density_y_max = max(density_y_max, kde_coarse(x_kde).max())
    density_y_max *= 1.15  # Add headroom

    # Plot each column
    for col, gran in enumerate(granularities):
        result = results[gran]
        metrics = result['metrics']

        # ===== TOP ROW: Rolling candidate rate =====
        ax_top = axes[0, col]

        x = np.array(metrics['message_idx'])
        rolling_cand = compute_rolling_rate(metrics['is_candidate'], window)

        ax_top.plot(x, rolling_cand, color='steelblue', linewidth=1.5, alpha=0.8)
        ax_top.axhline(y=target_rate, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Title with stats
        title = (
            f"{col_labels[col]}\n"
            f"Cand: {result['candidate_rate']*100:.1f}% | "
            f"Commit: {result['commit_rate']*100:.1f}% | "
            f"N={result['total_commits']}"
        )
        ax_top.set_title(title, fontsize=11, fontweight='bold')

        ax_top.set_ylabel('Candidate rate' if col == 0 else '', fontsize=11)
        ax_top.set_ylim(0, cand_y_max)
        ax_top.set_xlim(0, x_max)
        ax_top.grid(True, alpha=0.3, which='major')
        ax_top.set_xlabel('Message index (across dialogues)', fontsize=11)

        # ===== BOTTOM ROW: Commit density =====
        ax_bot = axes[1, col]

        commit_indices = result['commit_indices']

        if commit_indices:
            # KDE density over message stream
            if gran == 'fine' and kde_fine is not None:
                y_kde = kde_fine(x_kde)
                ax_bot.fill_between(x_kde, y_kde, alpha=0.6, color='steelblue')
                ax_bot.plot(x_kde, y_kde, color='steelblue', linewidth=1.5)
            elif gran == 'coarse' and kde_coarse is not None:
                y_kde = kde_coarse(x_kde)
                ax_bot.fill_between(x_kde, y_kde, alpha=0.6, color='steelblue')
                ax_bot.plot(x_kde, y_kde, color='steelblue', linewidth=1.5)

            # Rug plot at y=0
            ax_bot.scatter(commit_indices, [0] * len(commit_indices),
                          marker='|', color='darkblue', alpha=0.3, s=30, linewidths=0.5)

        ax_bot.set_xlabel('Message index (across dialogues)', fontsize=11)
        ax_bot.set_ylabel('Selection density' if col == 0 else '', fontsize=11)
        ax_bot.set_xlim(0, x_max)
        ax_bot.set_ylim(0, density_y_max)
        ax_bot.grid(True, alpha=0.3, which='major')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")


def check_distribution_difference(results: Dict[str, Dict]) -> Tuple[bool, str]:
    """
    Check if fine and coarse commit distributions visibly differ.

    Returns (differs, explanation).
    """
    fine_idx = results['fine']['commit_indices']
    coarse_idx = results['coarse']['commit_indices']

    if len(fine_idx) < 10 or len(coarse_idx) < 10:
        return False, f"Too few commits: fine={len(fine_idx)}, coarse={len(coarse_idx)}"

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(fine_idx, coarse_idx)

    # Compare means and stds
    fine_mean, fine_std = np.mean(fine_idx), np.std(fine_idx)
    coarse_mean, coarse_std = np.mean(coarse_idx), np.std(coarse_idx)

    explanation = (
        f"Fine: mean={fine_mean:.1f}, std={fine_std:.1f}, N={len(fine_idx)}\n"
        f"Coarse: mean={coarse_mean:.1f}, std={coarse_std:.1f}, N={len(coarse_idx)}\n"
        f"KS statistic: {ks_stat:.3f}, p-value: {ks_pval:.4f}"
    )

    # Consider distributions different if KS p-value < 0.05 or mean differs > 5%
    max_idx = max(max(fine_idx), max(coarse_idx)) if fine_idx and coarse_idx else 1
    differs = ks_pval < 0.05 or abs(fine_mean - coarse_mean) / max_idx > 0.05

    return differs, explanation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Figure 2: Adaptive Commitment")
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--target-rate", type=float, default=0.10)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--max-dialogues", type=int, default=250)
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else project_root / "datasets"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DialSeg711 dataset...")
    dialogues = load_dialseg711(max_dialogues=args.max_dialogues, datasets_dir=datasets_dir)
    print(f"Loaded {len(dialogues)} dialogues")

    # Find thresholds that produce target commit rate for each granularity
    print(f"\nFinding thresholds for target_rate={args.target_rate}, min_gap={args.min_gap}...")

    print("\n  Fine granularity...")
    threshold_fine, results_fine = find_threshold_for_rate(
        dialogues, 'fine',
        target_rate=args.target_rate,
        min_gap=args.min_gap
    )
    print(f"    Commit threshold: {threshold_fine:.3f}")
    print(f"    Candidates: {results_fine['total_candidates']} ({results_fine['candidate_rate']*100:.1f}%)")
    print(f"    Commits: {results_fine['total_commits']} ({results_fine['commit_rate']*100:.1f}%)")

    print("\n  Coarse granularity...")
    threshold_coarse, results_coarse = find_threshold_for_rate(
        dialogues, 'coarse',
        target_rate=args.target_rate,
        min_gap=args.min_gap
    )
    print(f"    Commit threshold: {threshold_coarse:.3f}")
    print(f"    Candidates: {results_coarse['total_candidates']} ({results_coarse['candidate_rate']*100:.1f}%)")
    print(f"    Commits: {results_coarse['total_commits']} ({results_coarse['commit_rate']*100:.1f}%)")

    results = {'fine': results_fine, 'coarse': results_coarse}

    # Check if we have enough commits
    total_commits = results_fine['total_commits'] + results_coarse['total_commits']
    if total_commits < 50:
        print(f"\n*** WARNING: Only {total_commits} total commits. Expected 100-300+ ***")
        print("Consider adjusting target_rate or min_gap parameters.")

    # Check if distributions differ
    print("\n" + "="*60)
    print("DISTRIBUTION COMPARISON")
    print("="*60)
    differs, explanation = check_distribution_difference(results)
    print(explanation)

    if not differs:
        print("\n*** WARNING: Fine and coarse distributions appear identical ***")
        print("The bottom density plots may not show visible differences.")

    # Generate figure
    # The 3x3 figure was too confusing
    print("\nGenerating 2x2 figure...")
    output_path = output_dir / "adaptive_commitment_granularity.png"
    plot_2x2_figure(results, args.target_rate, str(output_path))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTarget rate: {args.target_rate*100:.0f}%")
    print(f"Min gap: {args.min_gap}")
    print(f"\nThresholds found by binary search (simulate adaptive convergence):")
    print(f"  Fine:   threshold={threshold_fine:.3f} -> {results_fine['total_commits']:3d} commits ({results_fine['commit_rate']*100:.1f}%)")
    print(f"  Coarse: threshold={threshold_coarse:.3f} -> {results_coarse['total_commits']:3d} commits ({results_coarse['commit_rate']*100:.1f}%)")

    print(f"\nCandidate rates (determined by base scorer + granularity threshold):")
    print(f"  Fine (threshold={GRANULARITY_THRESHOLDS['fine']}):   {results_fine['candidate_rate']*100:.1f}%")
    print(f"  Coarse (threshold={GRANULARITY_THRESHOLDS['coarse']}): {results_coarse['candidate_rate']*100:.1f}%")

    print("\nInterpretation:")
    print("  - Top row shows rolling candidate rate (base scorer sensitivity)")
    print("  - Bottom row shows WHERE in dialogues commits occur")
    print("  - Both granularities converge to same ~10% commit rate")
    print("  - But require different commit thresholds (fine higher, coarse lower)")
    print("  - Spatial distribution may still differ based on when candidates occur")
