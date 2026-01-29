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
from .resume_moments import (
    ResumeMoment,
    load_resume_moments,
    save_resume_moments,
    get_moment_by_id,
    get_moments_by_category,
    validate_moments,
    summarize_moments,
)
from .quality_eval import (
    ModeResult,
    MomentEvalResult,
    QualityEvalReport,
    run_quality_eval,
    export_for_human_review,
    save_report as save_quality_report,
)
from .calibration import (
    CalibrationConfig,
    CalibrationMetrics,
    CalibrationResult,
    CalibrationReport,
    run_calibration_sweep,
    compute_calibration_metrics,
    select_best_config,
    run_full_calibration,
    load_calibrated_params,
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
    # Resume moments
    'ResumeMoment',
    'load_resume_moments',
    'save_resume_moments',
    'get_moment_by_id',
    'get_moments_by_category',
    'validate_moments',
    'summarize_moments',
    # Quality evaluation
    'ModeResult',
    'MomentEvalResult',
    'QualityEvalReport',
    'run_quality_eval',
    'export_for_human_review',
    'save_quality_report',
    # Calibration
    'CalibrationConfig',
    'CalibrationMetrics',
    'CalibrationResult',
    'CalibrationReport',
    'run_calibration_sweep',
    'compute_calibration_metrics',
    'select_best_config',
    'run_full_calibration',
    'load_calibrated_params',
]
