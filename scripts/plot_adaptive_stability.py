#!/usr/bin/env python3
"""
Generate stability plot for adaptive commitment strategy.

Shows rate vs time (messages) to visualize controller behavior.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.topics.evaluation import Message
from episodic.topics.strategies.neural_strategy import NeuralStrategy
from episodic.topics.strategies.adaptive_commitment import (
    AdaptiveCommitmentStrategy,
    AdaptivePolicy,
)


def load_dialseg711(max_dialogues: int = 50) -> List[List[Dict[str, Any]]]:
    """Load DialSeg711 dialogues."""
    path = Path("datasets/dialseg711/segmentation_file_test.json")
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


def run_adaptive_and_collect_metrics(dialogues: List[List[Dict]]) -> Dict[str, List]:
    """Run adaptive strategy and collect rate/evidence over time."""

    base = NeuralStrategy({'granularity': 'fine'})

    # Create adaptive with tuned bounds for this dataset
    from episodic.topics.strategies.commitment_strategy import CommitmentPolicy

    adaptive = AdaptiveCommitmentStrategy(
        base,
        AdaptivePolicy(
            target_rate=0.10,  # Slightly lower target
            rate_window=30,  # Smaller window for faster response
            adaptation_rate=0.20,  # Faster adaptation
            tolerance=0.30,  # Wider tolerance
            fixed_min_gap=2,
            warmup_messages=8,
            warmup_calibrate=False,
            min_evidence_bounds=(0.3, 1.2),  # Tighter range
        ),
        initial_policy=CommitmentPolicy(
            min_gap=2,
            evidence_window=2,
            min_evidence=0.5,  # Start lower
            evidence_decay=0.9,  # Slower decay
        )
    )

    # Collect metrics across all dialogues (continuous timeline)
    metrics = {
        'message_idx': [],
        'current_rate': [],
        'target_rate': [],
        'min_evidence': [],
        'boundary_committed': [],
        'base_detected': [],
        'base_confidence': [],
        'dialogue_boundaries': [],  # Marks where dialogues start
    }

    global_msg_idx = 0

    for dial_idx, dialogue in enumerate(dialogues):
        # Don't reset between dialogues to show cross-dialogue adaptation
        # (In real use, you might reset per conversation)

        metrics['dialogue_boundaries'].append(global_msg_idx)

        message_history = []
        for i, msg in enumerate(dialogue):
            if msg['role'] == 'user' and len(message_history) >= 2:
                decision = adaptive.get_decision(
                    query=msg['content'],
                    messages=message_history,
                    current_thread=None
                )

                metrics['message_idx'].append(global_msg_idx)
                metrics['current_rate'].append(decision.signals.get('current_rate', 0))
                metrics['target_rate'].append(decision.signals.get('target_rate', 0.10))
                metrics['min_evidence'].append(decision.signals.get('current_min_evidence', 0.7))
                metrics['boundary_committed'].append(1 if decision.topic_changed else 0)
                metrics['base_detected'].append(1 if decision.signals.get('base_detected', False) else 0)
                metrics['base_confidence'].append(decision.signals.get('confidence_score', 0))

            message_history.append(msg)
            global_msg_idx += 1

    # Print base detection stats
    base_detections = sum(metrics['base_detected'])
    print(f"\nBase strategy stats:")
    print(f"  Base detections: {base_detections} / {len(metrics['base_detected'])} ({100*base_detections/len(metrics['base_detected']):.1f}%)")

    return metrics


def plot_stability(metrics: Dict[str, List], output_path: str = "adaptive_stability.png"):
    """Generate stability plot."""

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    x = metrics['message_idx']

    # Plot 1: Rate vs Target
    ax1 = axes[0]
    ax1.plot(x, metrics['current_rate'], 'b-', alpha=0.7, label='Committed Rate', linewidth=1)
    ax1.axhline(y=metrics['target_rate'][0], color='r', linestyle='--', label='Target Rate', linewidth=1.5)

    # Add tolerance band
    target = metrics['target_rate'][0]
    tolerance = 0.30
    ax1.axhspan(target * (1 - tolerance), target * (1 + tolerance),
                alpha=0.1, color='green', label='Tolerance Band')

    # Mark dialogue boundaries
    for boundary in metrics['dialogue_boundaries'][1:]:
        ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.3)

    ax1.set_ylabel('Committed Rate')
    ax1.set_title('Adaptive Commitment Strategy: Rate Stability on DialSeg711 (50 dialogues)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.35)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Base detection rate (rolling)
    ax2 = axes[1]
    # Calculate rolling base detection rate
    window = 30
    base_rolling = []
    for i in range(len(metrics['base_detected'])):
        start = max(0, i - window)
        base_rolling.append(sum(metrics['base_detected'][start:i+1]) / (i - start + 1) if i > start else 0)

    ax2.plot(x, base_rolling, 'purple', alpha=0.7, label='Base Detection Rate (rolling)', linewidth=1)
    ax2.axhline(y=target, color='r', linestyle='--', alpha=0.5, label='Target')

    for boundary in metrics['dialogue_boundaries'][1:]:
        ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.3)

    ax2.set_ylabel('Base Det. Rate')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: min_evidence over time
    ax3 = axes[2]
    ax3.plot(x, metrics['min_evidence'], 'g-', alpha=0.7, label='min_evidence', linewidth=1)
    ax3.axhline(y=0.5, color='orange', linestyle='--', label='Initial Value', linewidth=1)
    ax3.axhline(y=0.3, color='red', linestyle=':', label='Lower Bound', linewidth=1, alpha=0.5)

    for boundary in metrics['dialogue_boundaries'][1:]:
        ax3.axvline(x=boundary, color='gray', linestyle=':', alpha=0.3)

    ax3.set_ylabel('min_evidence')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Boundary commitments vs base detections
    ax4 = axes[3]
    committed_x = [x[i] for i, c in enumerate(metrics['boundary_committed']) if c]
    base_det_x = [x[i] for i, c in enumerate(metrics['base_detected']) if c]

    ax4.scatter(base_det_x, [0.7] * len(base_det_x), marker='|', s=30, c='purple', alpha=0.4, label='Base Detection')
    ax4.scatter(committed_x, [0.3] * len(committed_x), marker='|', s=50, c='red', alpha=0.8, label='Committed')

    for boundary in metrics['dialogue_boundaries'][1:]:
        ax4.axvline(x=boundary, color='gray', linestyle=':', alpha=0.3)

    ax4.set_ylabel('Boundaries')
    ax4.set_xlabel('Message Index (across dialogues)')
    ax4.set_yticks([0.3, 0.7])
    ax4.set_yticklabels(['Committed', 'Base'])
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary stats
    rates = metrics['current_rate']
    if len(rates) > 10:
        print(f"\nStability Statistics:")
        print(f"  Mean rate: {np.mean(rates):.4f} (target: {target:.4f})")
        print(f"  Std dev:   {np.std(rates):.4f}")
        print(f"  Final min_evidence: {metrics['min_evidence'][-1]:.3f}")
        print(f"  Total boundaries: {sum(metrics['boundary_committed'])}")
        print(f"  Total messages evaluated: {len(x)}")


def run_with_granularity(dialogues: List[List[Dict]], granularity: str) -> Dict[str, List]:
    """Run adaptive strategy with specified base granularity."""

    base = NeuralStrategy({'granularity': granularity})

    from episodic.topics.strategies.commitment_strategy import CommitmentPolicy

    adaptive = AdaptiveCommitmentStrategy(
        base,
        AdaptivePolicy(
            target_rate=0.10,
            rate_window=30,
            adaptation_rate=0.20,
            tolerance=0.30,
            fixed_min_gap=2,
            warmup_messages=8,
            warmup_calibrate=False,
            min_evidence_bounds=(0.3, 1.2),
        ),
        initial_policy=CommitmentPolicy(
            min_gap=2,
            evidence_window=2,
            min_evidence=0.5,
            evidence_decay=0.9,
        )
    )

    metrics = {
        'message_idx': [],
        'current_rate': [],
        'target_rate': [],
        'min_evidence': [],
        'boundary_committed': [],
        'base_detected': [],
        'dialogue_boundaries': [],
    }

    global_msg_idx = 0

    for dial_idx, dialogue in enumerate(dialogues):
        metrics['dialogue_boundaries'].append(global_msg_idx)

        message_history = []
        for i, msg in enumerate(dialogue):
            if msg['role'] == 'user' and len(message_history) >= 2:
                decision = adaptive.get_decision(
                    query=msg['content'],
                    messages=message_history,
                    current_thread=None
                )

                metrics['message_idx'].append(global_msg_idx)
                metrics['current_rate'].append(decision.signals.get('current_rate', 0))
                metrics['target_rate'].append(decision.signals.get('target_rate', 0.10))
                metrics['min_evidence'].append(decision.signals.get('current_min_evidence', 0.7))
                metrics['boundary_committed'].append(1 if decision.topic_changed else 0)
                metrics['base_detected'].append(1 if decision.signals.get('base_detected', False) else 0)

            message_history.append(msg)
            global_msg_idx += 1

    return metrics


def plot_comparison(metrics_fine: Dict, metrics_coarse: Dict, output_path: str = "adaptive_comparison.png"):
    """Generate side-by-side comparison plot with clear axis labels."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Increase base font size
    plt.rcParams.update({'font.size': 13})

    target = 0.10

    # Column titles
    column_titles = ["Fine base scoring", "Coarse base scoring"]

    for col, (metrics, granularity) in enumerate([
        (metrics_fine, "fine"),
        (metrics_coarse, "coarse")
    ]):
        x = metrics['message_idx']

        # Calculate rolling base rate (window=30)
        window = 30
        base_rolling = []
        for i in range(len(metrics['base_detected'])):
            start = max(0, i - window)
            base_rolling.append(sum(metrics['base_detected'][start:i+1]) / (i - start + 1) if i > start else 0)

        # Row 1: Candidate boundary rate (rolling)
        ax1 = axes[0, col]
        ax1.plot(x, base_rolling, 'b-', alpha=0.7, linewidth=1.5, label='Candidate rate')
        ax1.axhline(y=target, color='r', linestyle='--', linewidth=2, label='Target rate')

        for boundary in metrics['dialogue_boundaries'][1:]:
            ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.2)

        ax1.set_ylabel('Candidate boundary rate\n(rolling window=30)', fontsize=13)
        ax1.set_title(column_titles[col], fontsize=15, fontweight='bold')
        ax1.set_ylim(0, 0.6)
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, alpha=0.3)
        if col == 0:
            ax1.legend(loc='upper right', fontsize=12)

        # Row 2: Adaptive threshold (min_evidence)
        ax2 = axes[1, col]
        ax2.plot(x, metrics['min_evidence'], 'g-', alpha=0.7, linewidth=1.5, label='min_evidence')
        ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Initial value')
        ax2.axhline(y=0.3, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Lower bound')
        ax2.axhline(y=1.2, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Upper bound')

        for boundary in metrics['dialogue_boundaries'][1:]:
            ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.2)

        ax2.set_ylabel('Selection threshold\n(min_evidence)', fontsize=13)
        ax2.set_ylim(0.2, 1.3)
        ax2.tick_params(axis='both', labelsize=11)
        ax2.grid(True, alpha=0.3)
        if col == 0:
            ax2.legend(loc='upper right', fontsize=11)

        # Row 3: Output boundary rate (committed)
        ax3 = axes[2, col]
        ax3.plot(x, metrics['current_rate'], 'purple', alpha=0.7, linewidth=1.5, label='Output rate')
        ax3.axhline(y=target, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Target rate')

        for boundary in metrics['dialogue_boundaries'][1:]:
            ax3.axvline(x=boundary, color='gray', linestyle=':', alpha=0.2)

        ax3.set_ylabel('Output boundary rate\n(committed)', fontsize=13)
        ax3.set_xlabel('Canonical boundary index', fontsize=13)
        ax3.set_ylim(0, 0.35)
        ax3.set_yticks([0, 0.1, 0.2, 0.3])
        ax3.tick_params(axis='both', labelsize=11)
        ax3.grid(True, alpha=0.3)
        if col == 0:
            ax3.legend(loc='upper right', fontsize=12)

        # Stats annotation
        base_det = sum(metrics['base_detected'])
        committed = sum(metrics['boundary_committed'])
        final_rate = metrics['current_rate'][-1] if metrics['current_rate'] else 0

        ax1.text(0.02, 0.95, f"Candidates: {100*base_det/len(metrics['base_detected']):.0f}%\n"
                              f"Committed: {committed}",
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    print("Loading DialSeg711 dataset...")
    dialogues = load_dialseg711(max_dialogues=50)
    print(f"Loaded {len(dialogues)} dialogues")

    print("\nRunning with Neural(fine)...")
    metrics_fine = run_with_granularity(dialogues, 'fine')
    base_fine = sum(metrics_fine['base_detected'])
    print(f"  Base detections: {base_fine} ({100*base_fine/len(metrics_fine['base_detected']):.1f}%)")
    print(f"  Committed: {sum(metrics_fine['boundary_committed'])}")

    print("\nRunning with Neural(coarse)...")
    metrics_coarse = run_with_granularity(dialogues, 'coarse')
    base_coarse = sum(metrics_coarse['base_detected'])
    print(f"  Base detections: {base_coarse} ({100*base_coarse/len(metrics_coarse['base_detected']):.1f}%)")
    print(f"  Committed: {sum(metrics_coarse['boundary_committed'])}")

    print("\nGenerating comparison plot...")
    output_path = Path(__file__).parent.parent / "paper" / "figures" / "adaptive_commitment_granularity.png"
    plot_comparison(metrics_fine, metrics_coarse, str(output_path))
