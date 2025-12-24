#!/usr/bin/env python3
"""
Run InstructGPT (text-davinci-003) as an illustrative baseline.

This is a configuration-matched InstructGPT reproduction for the SuperDialseg
leaderboard audit. It is presented separately from deterministic methods
and NOT mixed into pairwise-delta tables.

Configuration (matching SuperDialseg):
- Model: text-davinci-003
- Temperature: 0
- Max tokens: 512

The SuperDialseg paper does not release the exact prompt string, so this
reproduces the task intent and output format, not a verbatim prompt.

Usage:
    python -m tacl.experiments.evaluation.run_instructgpt_baseline

    # With explicit API key
    OPENAI_API_KEY=sk-... python -m tacl.experiments.evaluation.run_instructgpt_baseline

Output files:
    results/instructgpt_baseline.csv      - Per-dialogue results
    results/instructgpt_summary.csv       - Aggregate metrics
    results/instructgpt_summary.md        - Human-readable summary
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

EXPECTED_PATHS = {
    "dialseg711": DATASETS_DIR / "dialseg711" / "segmentation_file_test.json",
    "tiage": DATASETS_DIR / "tiage" / "segmentation_file_test.json",
}


@dataclass
class DialogueResult:
    """Result for a single dialogue."""
    dialogue_id: int
    dataset: str
    n_gold: int
    n_pred: int
    strict_f1: float
    w_f1: float
    bor: float
    raw_response: str


@dataclass
class AggregateResult:
    """Aggregate result for one dataset."""
    dataset: str
    n_dialogues: int
    total_gold: int
    total_pred: int
    micro_f1: float
    macro_f1: float
    mean_w_f1: float
    bor: float
    regime: str


def log(msg: str):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def check_api_access() -> Tuple[bool, str]:
    """Check if OpenAI API is accessible."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY environment variable not set"

    # Try a minimal API call to verify access
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # List models to verify API key works
        # (cheaper than a completion call)
        models = list(client.models.list())

        # Check if text-davinci-003 is available
        model_ids = [m.id for m in models]
        if "text-davinci-003" not in model_ids:
            return False, "text-davinci-003 not available (model may be deprecated)"

        return True, "API access verified"
    except Exception as e:
        return False, f"API access failed: {e}"


def load_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset in canonical format."""
    filepath = EXPECTED_PATHS[dataset_name]
    log(f"Loading {dataset_name} from {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", {})
    dialogues_raw = dial_data.get(dataset_name, [])

    dialogues = []
    for dial in dialogues_raw:
        turns = dial.get("turns", [])

        messages = []
        for turn in turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("utterance", ""),
            })

        gold_boundaries = set()
        for i, turn in enumerate(turns):
            if turn.get("segmentation_label", 0) == 1:
                boundary_idx = i + 1
                if 1 <= boundary_idx < len(turns):
                    gold_boundaries.add(boundary_idx)

        dialogues.append({
            "messages": messages,
            "gold_boundaries": gold_boundaries,
            "num_messages": len(messages),
        })

    log(f"  Loaded {len(dialogues)} dialogues")
    return dialogues


def compute_metrics(
    gold: Set[int],
    pred: Set[int],
    window: int = 1
) -> Tuple[float, float, float]:
    """Compute strict F1 and windowed F1."""
    # Strict F1
    if not gold and not pred:
        strict_f1 = 1.0
    elif not gold or not pred:
        strict_f1 = 0.0
    else:
        tp = len(pred & gold)
        prec = tp / len(pred)
        rec = tp / len(gold)
        strict_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Windowed F1
    if not gold and not pred:
        w_f1 = 1.0
    elif not gold or not pred:
        w_f1 = 0.0
    else:
        matched_pred = set()
        matched_gold = set()
        for p in pred:
            for g in gold:
                if abs(p - g) <= window:
                    matched_pred.add(p)
                    matched_gold.add(g)
                    break

        w_prec = len(matched_pred) / len(pred)
        w_rec = len(matched_gold) / len(gold)
        w_f1 = 2 * w_prec * w_rec / (w_prec + w_rec) if (w_prec + w_rec) > 0 else 0.0

    # BOR
    bor = len(pred) / len(gold) if gold else (float('inf') if pred else 1.0)

    return strict_f1, w_f1, bor


def get_regime(bor: float) -> str:
    """Get granularity regime from BOR."""
    if bor < 0.8:
        return "Conservative"
    elif bor > 1.2:
        return "Aggressive"
    else:
        return "Balanced"


def run_instructgpt_on_dataset(
    dataset_name: str,
    dialogues: List[Dict],
    segmenter,
    csv_writer,
) -> AggregateResult:
    """Run InstructGPT on a dataset and return aggregate results."""
    log(f"Running InstructGPT on {dataset_name} ({len(dialogues)} dialogues)...")

    total_gold = 0
    total_pred = 0
    total_tp = 0
    f1_sum = 0.0
    w_f1_sum = 0.0

    for i, dialogue in enumerate(tqdm(dialogues, desc=f"  {dataset_name}", leave=False)):
        messages = dialogue["messages"]
        gold = dialogue["gold_boundaries"]

        try:
            result = segmenter.predict_boundaries(messages)
            pred = result.to_set()
            raw_response = result.metadata.get("raw_response", "")
        except Exception as e:
            log(f"  ERROR on dialogue {i}: {e}")
            pred = set()
            raw_response = f"ERROR: {e}"

        strict_f1, w_f1, bor = compute_metrics(gold, pred)

        # Accumulate for micro F1
        tp = len(pred & gold)
        total_tp += tp
        total_gold += len(gold)
        total_pred += len(pred)

        # Accumulate for macro F1
        f1_sum += strict_f1
        w_f1_sum += w_f1

        # Write per-dialogue result
        csv_writer.writerow({
            "dialogue_id": i,
            "dataset": dataset_name,
            "n_messages": len(messages),
            "n_gold": len(gold),
            "n_pred": len(pred),
            "strict_f1": f"{strict_f1:.4f}",
            "w_f1": f"{w_f1:.4f}",
            "bor": f"{bor:.4f}" if bor != float('inf') else "inf",
            "raw_response": raw_response[:200],  # Truncate for CSV
        })

    n = len(dialogues)

    # Micro F1
    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Macro F1 and mean W-F1
    macro_f1 = f1_sum / n
    mean_w_f1 = w_f1_sum / n

    # BOR
    bor = total_pred / total_gold if total_gold > 0 else 1.0

    return AggregateResult(
        dataset=dataset_name,
        n_dialogues=n,
        total_gold=total_gold,
        total_pred=total_pred,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        mean_w_f1=mean_w_f1,
        bor=bor,
        regime=get_regime(bor),
    )


def write_summary_csv(results: List[AggregateResult], filepath: Path):
    """Write aggregate summary CSV."""
    log(f"Writing summary CSV to {filepath}")
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "n_dialogues", "total_gold", "total_pred",
            "micro_f1", "macro_f1", "mean_w_f1", "bor", "regime"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset": r.dataset,
                "n_dialogues": r.n_dialogues,
                "total_gold": r.total_gold,
                "total_pred": r.total_pred,
                "micro_f1": f"{r.micro_f1:.3f}",
                "macro_f1": f"{r.macro_f1:.3f}",
                "mean_w_f1": f"{r.mean_w_f1:.3f}",
                "bor": f"{r.bor:.2f}",
                "regime": r.regime,
            })


def write_summary_md(results: List[AggregateResult], filepath: Path, api_status: str):
    """Write human-readable summary."""
    log(f"Writing summary to {filepath}")

    lines = [
        "# InstructGPT Baseline Results",
        "",
        "**Configuration-matched InstructGPT reproduction**",
        "",
        "The SuperDialseg paper does not release the exact prompt string.",
        "This reproduces the task intent and output format, not a verbatim prompt.",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        "- Model: text-davinci-003",
        "- Temperature: 0",
        "- Max tokens: 512",
        f"- API status: {api_status}",
        "",
        "## Results",
        "",
        "| Dataset | Dialogues | Micro-F1 | Macro-F1 | W-F1 | BOR | Regime |",
        "|---------|-----------|----------|----------|------|-----|--------|",
    ]

    for r in results:
        lines.append(
            f"| {r.dataset} | {r.n_dialogues} | {r.micro_f1:.3f} | {r.macro_f1:.3f} | "
            f"{r.mean_w_f1:.3f} | {r.bor:.2f} | {r.regime} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This is an **illustrative baseline**, presented separately from deterministic methods.",
        "- Results are NOT mixed into pairwise-delta tables.",
        "- BOR (Boundary Over-prediction Ratio) = n_pred / n_gold",
        "- Regime: Conservative (BOR < 0.8), Balanced (0.8-1.2), Aggressive (BOR > 1.2)",
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def write_not_reproduced(filepath: Path, reason: str):
    """Write documentation that InstructGPT was not reproduced."""
    log(f"Writing 'not reproduced' note to {filepath}")

    lines = [
        "# InstructGPT Baseline: Not Reproduced",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Status",
        "",
        f"InstructGPT (text-davinci-003) was not reproduced: {reason}",
        "",
        "## Configuration (intended)",
        "",
        "- Model: text-davinci-003",
        "- Temperature: 0",
        "- Max tokens: 512",
        "",
        "## Notes",
        "",
        "- The SuperDialseg paper does not release the exact prompt string.",
        "- text-davinci-003 is an instruction-following model (not a chat model).",
        "- This baseline requires OpenAI API access with legacy completions models.",
    ]

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))


def main():
    """Main entry point."""
    log("=" * 60)
    log("InstructGPT Baseline Evaluation")
    log("Configuration-matched reproduction (text-davinci-003)")
    log("=" * 60)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check API access
    log("Checking OpenAI API access...")
    api_ok, api_status = check_api_access()
    log(f"  {api_status}")

    if not api_ok:
        log("API access not available. Documenting as 'not reproduced'.")
        write_not_reproduced(RESULTS_DIR / "instructgpt_summary.md", api_status)
        return

    # Validate dataset paths
    log("Validating dataset paths...")
    missing = []
    for name, path in EXPECTED_PATHS.items():
        if not path.exists():
            missing.append(f"  - {name}: {path}")

    if missing:
        log("ERROR: Missing dataset files:")
        for m in missing:
            print(m)
        sys.exit(1)

    log("All paths validated.")

    # Initialize segmenter
    from tacl.experiments.segmenters import InstructGPTSegmenter
    segmenter = InstructGPTSegmenter()
    log(f"Initialized: {segmenter.description}")

    # Open per-dialogue CSV
    per_dial_path = RESULTS_DIR / "instructgpt_baseline.csv"
    log(f"Per-dialogue results will be written to {per_dial_path}")

    aggregate_results = []

    with open(per_dial_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dialogue_id", "dataset", "n_messages", "n_gold", "n_pred",
            "strict_f1", "w_f1", "bor", "raw_response"
        ])
        writer.writeheader()

        # Run on each dataset
        for dataset_name in EXPECTED_PATHS:
            dialogues = load_dataset(dataset_name)
            result = run_instructgpt_on_dataset(dataset_name, dialogues, segmenter, writer)
            aggregate_results.append(result)

            log(f"  {dataset_name}: Micro-F1={result.micro_f1:.3f}, W-F1={result.mean_w_f1:.3f}, BOR={result.bor:.2f} [{result.regime}]")

    # Write aggregate outputs
    write_summary_csv(aggregate_results, RESULTS_DIR / "instructgpt_summary.csv")
    write_summary_md(aggregate_results, RESULTS_DIR / "instructgpt_summary.md", api_status)

    # Print summary
    log("\n" + "=" * 80)
    log(f"{'Dataset':<15} {'Dialogues':>10} {'Micro-F1':>10} {'W-F1':>8} {'BOR':>6} {'Regime':<12}")
    log("=" * 80)
    for r in aggregate_results:
        log(f"{r.dataset:<15} {r.n_dialogues:>10} {r.micro_f1:>10.3f} {r.mean_w_f1:>8.3f} {r.bor:>6.2f} [{r.regime}]")
    log("=" * 80)

    log("\nDone.")
    log(f"Per-dialogue results: {per_dial_path}")
    log(f"Summary: {RESULTS_DIR / 'instructgpt_summary.md'}")


if __name__ == "__main__":
    main()
