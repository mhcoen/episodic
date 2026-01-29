"""
CLI commands for evaluation and calibration.
"""

import click
from typing import Optional


def evaluate_command(args: str, conv_manager) -> tuple[str, bool]:
    """
    Evaluation and calibration commands.

    Usage:
        /evaluate reactivation         - Replay recent 100 turns
        /evaluate reactivation --all   - Full conversation history
        /evaluate reactivation --export <path> - Export features for analysis
    """
    parts = args.strip().split()

    if not parts or parts[0] == 'help':
        return _show_help(), False

    subcommand = parts[0].lower()

    if subcommand == 'reactivation':
        return _evaluate_reactivation(parts[1:])
    elif subcommand == 'benchmark' or subcommand == 'resume':
        return _benchmark_resume(parts[1:])
    elif subcommand == 'quality':
        return _evaluate_quality(parts[1:])
    elif subcommand == 'calibrate':
        return _calibrate_reactivation(parts[1:])
    else:
        return f"Unknown evaluation target: {subcommand}\nUse /evaluate help for usage.", False


def _show_help() -> str:
    """Show evaluation help."""
    return """Evaluation Commands
==================

/evaluate reactivation [options]
    Replay reactivation probe decisions on historical conversations.

    Options:
        --all        Replay all conversation history (not just recent)
        --limit N    Limit to N turns (default: 100)
        --export P   Export features to path P for external analysis
        --labeled    Only replay turns with ground truth labels

/evaluate benchmark [options]
/evaluate resume [options]
    Run resume benchmark comparing context recovery modes.
    Compares ancestry vs topic_local vs hybrid on resume scenarios.

    Options:
        --limit N     Maximum scenarios to test (default: 50)
        --min-gap N   Minimum turn gap to qualify as resume (default: 3)
        --export P    Export detailed results to path P (JSON)

/evaluate quality [options]
    Run quality evaluation on labeled resume moments.
    Compares ancestry vs hybrid vs topic_local with contamination tracking.

    Options:
        --category C  Filter by category (short_gap, medium_gap, long_gap, ambiguous, thin_topic)
        --export      Export side-by-side comparison for human review
        --llm         Actually call LLM for responses (expensive)

/evaluate calibrate [options]
    Run reactivation calibration loop.
    Tunes probe thresholds against labeled moments using cross-validation.

    Options:
        --seed N      Random seed for reproducibility (default: 42)
        --no-cv       Skip cross-validation, train on all data

Examples:
    /evaluate reactivation              # Replay recent 100 turns
    /evaluate reactivation --all        # Full history
    /evaluate reactivation --limit 50   # Replay last 50 turns
    /evaluate reactivation --export features.jsonl  # Export for analysis
    /evaluate benchmark                 # Run resume benchmark
    /evaluate benchmark --min-gap 10    # Only long-gap resumes
    /evaluate benchmark --export bench.json  # Export detailed results
    /evaluate quality                   # Run quality eval on all moments
    /evaluate quality --category short_gap  # Only short gap moments
    /evaluate quality --export          # Export for human review
    /evaluate calibrate                 # Run calibration loop
    /evaluate calibrate --seed 123      # Custom seed
"""


def _evaluate_reactivation(args: list) -> tuple[str, bool]:
    """Run reactivation replay evaluation."""
    from episodic.evaluation.reactivation_replay import (
        replay_conversation,
        compute_metrics,
        get_replay_summary,
        export_features,
    )

    # Parse arguments
    limit = 100
    use_all = False
    export_path = None
    labeled_only = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--all':
            use_all = True
        elif arg == '--labeled':
            labeled_only = True
        elif arg == '--limit' and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
                i += 1
            except ValueError:
                return f"Invalid limit value: {args[i + 1]}", False
        elif arg == '--export' and i + 1 < len(args):
            export_path = args[i + 1]
            i += 1
        i += 1

    if use_all:
        limit = 10000  # Effectively unlimited

    # Run replay
    results = replay_conversation(
        limit=limit,
        use_ground_truth=labeled_only
    )

    if not results:
        return "No reactivation decisions found to replay.\nMake sure enable_topic_reactivation is enabled and reactivation_log_features is true.", False

    # Compute metrics
    metrics = compute_metrics(results)

    # Export if requested
    if export_path:
        try:
            export_features(results, export_path)
            return f"Exported {len(results)} feature records to {export_path}", True
        except Exception as e:
            return f"Failed to export features: {e}", False

    # Return summary
    summary = get_replay_summary(metrics)
    return summary, True


def _benchmark_resume(args: list) -> tuple[str, bool]:
    """Run resume benchmark comparing context recovery modes."""
    from episodic.evaluation.resume_benchmark import (
        find_resume_scenarios,
        run_benchmark,
        get_benchmark_summary,
        export_benchmark_results,
    )

    # Parse arguments
    limit = 50
    min_gap = 3
    export_path = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--limit' and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
                i += 1
            except ValueError:
                return f"Invalid limit value: {args[i + 1]}", False
        elif arg == '--min-gap' and i + 1 < len(args):
            try:
                min_gap = int(args[i + 1])
                i += 1
            except ValueError:
                return f"Invalid min-gap value: {args[i + 1]}", False
        elif arg == '--export' and i + 1 < len(args):
            export_path = args[i + 1]
            i += 1
        i += 1

    # Find scenarios
    scenarios = find_resume_scenarios(
        min_gap_turns=min_gap,
        max_scenarios=limit
    )

    if not scenarios:
        return (
            "No resume scenarios found.\n"
            "Make sure:\n"
            "  - enable_topic_reactivation is enabled\n"
            "  - reactivation_log_features is true\n"
            "  - You have some reactivation decisions in the database"
        ), False

    # Run benchmark
    result = run_benchmark(scenarios=scenarios)

    # Export if requested
    if export_path:
        try:
            export_benchmark_results(result, export_path)
            return f"Exported {len(result.details)} benchmark results to {export_path}", True
        except Exception as e:
            return f"Failed to export results: {e}", False

    # Return summary
    summary = get_benchmark_summary(result)
    return summary, True


def _evaluate_quality(args: list) -> tuple[str, bool]:
    """Run quality evaluation on labeled resume moments."""
    from episodic.evaluation.quality_eval import (
        run_quality_eval,
        export_for_human_review,
        save_report,
    )
    from episodic.evaluation.resume_moments import (
        load_resume_moments,
        summarize_moments,
    )

    # Parse arguments
    category_filter = None
    do_export = False
    call_llm = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--category' and i + 1 < len(args):
            category_filter = args[i + 1]
            i += 1
        elif arg == '--export':
            do_export = True
        elif arg == '--llm':
            call_llm = True
        i += 1

    # Load moments
    moments = load_resume_moments(category=category_filter)

    if not moments:
        return (
            "No resume moments found.\n"
            f"Looked for: episodic/evaluation/fixtures/resume_moments.json\n"
            f"Category filter: {category_filter or 'none'}"
        ), False

    # Show moment summary
    output_lines = [
        "Quality Evaluation",
        "==================",
        "",
        summarize_moments(moments),
        "",
    ]

    # Run evaluation
    output_lines.append("Running evaluation across modes (ancestry, hybrid, topic_local)...")
    output_lines.append("")

    try:
        report = run_quality_eval(
            moments=moments,
            call_llm=call_llm,
            category_filter=category_filter,
        )
    except Exception as e:
        return f"Quality evaluation failed: {e}", False

    # Summary statistics
    output_lines.append("Results")
    output_lines.append("-------")
    output_lines.append(f"Moments evaluated: {report.moments_evaluated}")
    output_lines.append("")

    output_lines.append("By Mode:")
    for mode, stats in report.summary.get("by_mode", {}).items():
        output_lines.append(
            f"  {mode}: contamination={stats['contamination_rate']:.1%}, "
            f"avg_tokens={stats['avg_tokens']:.0f}, "
            f"avg_ms={stats['avg_assembly_ms']:.2f}"
        )

    output_lines.append("")
    output_lines.append("By Category:")
    for cat, cat_stats in report.summary.get("by_category", {}).items():
        mode_contam = [
            f"{m}={s['total_contamination']}"
            for m, s in cat_stats.get("modes", {}).items()
        ]
        output_lines.append(f"  {cat} ({cat_stats['count']}): {', '.join(mode_contam)}")

    # Save full report
    try:
        report_path = save_report(report)
        output_lines.append("")
        output_lines.append(f"Full report saved to: {report_path}")
    except Exception as e:
        output_lines.append(f"Warning: Failed to save report: {e}")

    # Export for human review if requested
    if do_export:
        try:
            review_path = export_for_human_review(report)
            output_lines.append(f"Human review export saved to: {review_path}")
        except Exception as e:
            output_lines.append(f"Warning: Failed to export for review: {e}")

    return "\n".join(output_lines), True


def _calibrate_reactivation(args: list) -> tuple[str, bool]:
    """Run reactivation calibration loop."""
    from episodic.evaluation.calibration import (
        run_full_calibration,
        DEFAULT_PARAM_GRID,
        OBJECTIVE_WEIGHTS,
    )

    # Parse arguments
    seed = 42
    use_cv = True

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--seed' and i + 1 < len(args):
            try:
                seed = int(args[i + 1])
                i += 1
            except ValueError:
                return f"Invalid seed value: {args[i + 1]}", False
        elif arg == '--no-cv':
            use_cv = False
        i += 1

    output_lines = [
        "Reactivation Calibration",
        "========================",
        "",
        f"Seed: {seed}",
        f"Cross-validation: {'LOBO-CV' if use_cv else 'All data'}",
        "",
        "Parameter grid:",
    ]

    for param, values in DEFAULT_PARAM_GRID.items():
        output_lines.append(f"  {param}: {values}")

    output_lines.append("")
    output_lines.append("Objective (lexicographic):")
    for metric, weight in sorted(OBJECTIVE_WEIGHTS.items(), key=lambda x: -abs(x[1])):
        direction = "maximize" if weight > 0 else "minimize"
        output_lines.append(f"  {metric}: {direction} (weight={weight})")

    output_lines.append("")
    output_lines.append("Running calibration sweep...")

    try:
        report = run_full_calibration(seed=seed)
    except Exception as e:
        return f"Calibration failed: {e}", False

    output_lines.append("")
    output_lines.append("Results")
    output_lines.append("-------")
    output_lines.append("")
    output_lines.append("Best configuration:")
    for param, value in report.best_config.items():
        output_lines.append(f"  {param}: {value}")

    output_lines.append("")
    output_lines.append("Best metrics:")
    for metric, value in report.best_metrics.items():
        output_lines.append(f"  {metric}: {value:.2%}")

    output_lines.append("")
    output_lines.append(f"Chosen reason: {report.chosen_reason}")
    output_lines.append("")
    output_lines.append(f"Dataset hash: {report.dataset_hash}")
    output_lines.append(f"Git commit: {report.git_commit}")

    output_lines.append("")
    output_lines.append("Reports saved to:")
    output_lines.append("  episodic/evaluation/reports/calibrated_params.json")
    output_lines.append("  episodic/evaluation/reports/calibration_report.csv")

    return "\n".join(output_lines), True
