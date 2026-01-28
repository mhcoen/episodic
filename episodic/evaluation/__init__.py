"""
Evaluation tools for Episodic.

Includes replay harnesses and metrics for calibrating system components.
"""

from .reactivation_replay import (
    ReplayResult,
    replay_conversation,
    compute_metrics,
)
from .resume_benchmark import (
    BenchmarkScenarioType,
    BenchmarkScenario,
    BenchmarkResult,
    find_resume_scenarios,
    run_benchmark,
    get_benchmark_summary,
    export_benchmark_results,
)
from .benchmark_fixtures import (
    FixedResumeScenario,
    load_benchmark_fixtures,
    create_default_fixtures,
    compute_and_save_embeddings,
    validate_fixtures,
)
from .benchmark_runner import (
    BenchmarkResult as DeterministicBenchmarkResult,
    run_deterministic_benchmark,
    compute_benchmark_metrics,
    format_benchmark_report,
)

__all__ = [
    # Reactivation replay
    'ReplayResult',
    'replay_conversation',
    'compute_metrics',
    # Resume benchmark (dynamic)
    'BenchmarkScenarioType',
    'BenchmarkScenario',
    'BenchmarkResult',
    'find_resume_scenarios',
    'run_benchmark',
    'get_benchmark_summary',
    'export_benchmark_results',
    # Deterministic benchmark fixtures
    'FixedResumeScenario',
    'load_benchmark_fixtures',
    'create_default_fixtures',
    'compute_and_save_embeddings',
    'validate_fixtures',
    # Deterministic benchmark runner
    'DeterministicBenchmarkResult',
    'run_deterministic_benchmark',
    'compute_benchmark_metrics',
    'format_benchmark_report',
]
