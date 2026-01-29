#!/usr/bin/env python3
"""
Main runner for LLM verifier experiment.

Usage:
    python run_experiment.py              # Run full experiment
    python run_experiment.py --setup      # Setup corpus only
    python run_experiment.py --clear-cache # Clear verifier cache
    python run_experiment.py --dry-run    # Show what would be run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add experiment dir to path for imports
EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from corpus import create_database, create_query_cases, DB_PATH, QUERY_CASES_PATH, STATEMENTS
from verifier import (
    run_verification, init_cache, CACHE_PATH, get_prompt_hash,
    VerifierStats, Relation
)
from metrics import compute_aggregate_metrics, format_metrics_report


def setup_corpus():
    """Create the synthetic database and query cases."""
    print("Setting up synthetic corpus...")
    create_database()
    create_query_cases()
    print("Done!")


def clear_cache():
    """Clear the verifier cache."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print(f"Cleared cache at {CACHE_PATH}")
    else:
        print("No cache to clear")


def show_dry_run():
    """Show what the experiment will do without running it."""
    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    print("DRY RUN - Experiment Overview")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Statements: {len(STATEMENTS)}")
    print(f"Query cases: {len(cases)}")
    print(f"Prompt hash: {get_prompt_hash()}")
    print()

    total_candidates = sum(len(c["candidates"]) for c in cases)
    total_hard_negs = sum(len(c.get("hard_negatives", [])) for c in cases)

    print(f"Total candidates to evaluate: {total_candidates}")
    print(f"Total hard negatives: {total_hard_negs}")
    print()

    print("Query breakdown:")
    for case in cases:
        hn = len(case.get("hard_negatives", []))
        hn_str = f" [HN={hn}]" if hn > 0 else ""
        print(f"  {case['query']:20s} {len(case['candidates']):3d} candidates{hn_str}")

    print()
    print("Estimated LLM calls (worst case):")
    print(f"  {len(cases)} queries × 2 max batches = {len(cases) * 2} calls")
    print(f"  With early exit, expect ~{len(cases)} calls")


def run_full_experiment(model_id: str = "gpt-4o-mini"):
    """Run the full experiment and report results."""
    print("=" * 60)
    print("LLM VERIFIER EXPERIMENT")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Prompt hash: {get_prompt_hash()}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Ensure corpus exists
    if not DB_PATH.exists():
        print("Creating synthetic corpus...")
        setup_corpus()

    # Initialize cache
    init_cache()

    # Load cases for metrics
    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    # Run verification
    print("Running verification...")
    print("-" * 40)
    all_stats = run_verification(
        model_id=model_id,
        batch_size=10,
        accept_target=3,
        max_batches=2,
    )
    print()

    # Compute metrics
    query_metrics, agg = compute_aggregate_metrics(all_stats, cases)

    # Print report
    report = format_metrics_report(query_metrics, agg)
    print(report)

    # Save detailed results
    results_path = EXPERIMENT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        "model_id": model_id,
        "prompt_hash": get_prompt_hash(),
        "timestamp": datetime.now().isoformat(),
        "aggregate": {
            "num_queries": agg.num_queries,
            "mean_precision_at_5": agg.mean_precision_at_5,
            "mean_precision_at_10": agg.mean_precision_at_10,
            "mean_recall_at_5": agg.mean_recall_at_5,
            "mean_recall_at_10": agg.mean_recall_at_10,
            "coverage_at_5": agg.coverage_at_5,
            "coverage_at_10": agg.coverage_at_10,
            "hard_negative_false_accept_rate": agg.hard_negative_false_accept_rate,
            "format_failure_rate": agg.format_failure_rate,
            "total_llm_calls": agg.total_llm_calls,
            "total_cache_hits": agg.total_cache_hits,
        },
        "per_query": [
            {
                "query": qm.query,
                "description": qm.description,
                "precision_at_5": qm.precision_at_5,
                "precision_at_10": qm.precision_at_10,
                "recall_at_5": qm.recall_at_5,
                "recall_at_10": qm.recall_at_10,
                "hard_negative_accepted": qm.hard_negative_accepted,
                "hard_negative_count": qm.hard_negative_count,
                "accepted_ids": qm.accepted_ids,
                "relevant_ids": qm.relevant_ids,
            }
            for qm in query_metrics
        ],
        "detailed_results": [
            {
                "query": stats.query,
                "results": [
                    {
                        "statement_id": r.statement_id,
                        "relation": r.relation.value,
                        "confidence": r.confidence,
                        "quote_check_passed": r.quote_check_passed,
                        "quote_check_errors": r.quote_check_errors,
                        "rationale": r.rationale,
                    }
                    for r in stats.results
                ]
            }
            for stats in all_stats
        ],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")

    # Summary verdict
    print()
    print("VERDICT")
    print("-" * 40)
    if agg.hard_negative_false_accept_rate == 0:
        print("✓ PASS: Zero hard negatives accepted")
    else:
        print(f"✗ FAIL: {agg.total_hard_negative_accepts}/{agg.total_hard_negatives} hard negatives accepted")

    if agg.mean_precision_at_5 >= 0.8:
        print(f"✓ GOOD: Precision@5 = {agg.mean_precision_at_5:.1%}")
    elif agg.mean_precision_at_5 >= 0.5:
        print(f"○ OK: Precision@5 = {agg.mean_precision_at_5:.1%}")
    else:
        print(f"✗ LOW: Precision@5 = {agg.mean_precision_at_5:.1%}")

    return all_stats, query_metrics, agg


def main():
    parser = argparse.ArgumentParser(description="LLM Verifier Experiment")
    parser.add_argument("--setup", action="store_true", help="Setup corpus only")
    parser.add_argument("--clear-cache", action="store_true", help="Clear verifier cache")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model ID for verifier")

    args = parser.parse_args()

    if args.setup:
        setup_corpus()
    elif args.clear_cache:
        clear_cache()
    elif args.dry_run:
        if not QUERY_CASES_PATH.exists():
            setup_corpus()
        show_dry_run()
    else:
        run_full_experiment(model_id=args.model)


if __name__ == "__main__":
    main()
