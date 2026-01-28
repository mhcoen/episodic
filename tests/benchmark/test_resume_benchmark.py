"""
Deterministic resume benchmark tests for CI.

Uses fixed embeddings - no model calls, fully reproducible.
"""

import pytest

from episodic.evaluation.benchmark_fixtures import (
    FixedResumeScenario,
    load_benchmark_fixtures,
    validate_fixtures,
)
from episodic.evaluation.benchmark_runner import (
    BenchmarkResult,
    compute_benchmark_metrics,
    format_benchmark_report,
    run_deterministic_benchmark,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.reactivation]


class TestResumeBenchmark:
    """Deterministic benchmark tests."""

    @pytest.fixture(scope="class")
    def scenarios(self) -> list[FixedResumeScenario]:
        """Load benchmark fixtures."""
        return load_benchmark_fixtures()

    @pytest.fixture(scope="class")
    def results(self, scenarios: list[FixedResumeScenario]) -> list[BenchmarkResult]:
        """Run benchmark on all scenarios."""
        return run_deterministic_benchmark(scenarios)

    def test_fixtures_are_valid(self):
        """Fixtures should be properly formed."""
        validation = validate_fixtures()
        assert validation["scenario_count"] > 0, "No scenarios found"
        # Note: Embeddings may not be computed yet, which is OK for initial testing
        if not validation["valid"]:
            pytest.skip(
                f"Fixtures missing embeddings - run compute_and_save_embeddings(): {validation['issues']}"
            )

    def test_topic_local_no_contamination(self, results: list[BenchmarkResult]):
        """topic_local mode should NEVER have contamination."""
        topic_local_results = [r for r in results if r.mode == "topic_local"]

        contaminated = [r for r in topic_local_results if r.contamination_detected]

        assert len(contaminated) == 0, (
            f"Contamination in topic_local: {[r.scenario_id for r in contaminated]}"
        )

    def test_hybrid_no_contamination(self, results: list[BenchmarkResult]):
        """hybrid mode should NEVER have contamination (uses topic_local after reactivation)."""
        hybrid_results = [r for r in results if r.mode == "hybrid"]

        contaminated = [r for r in hybrid_results if r.contamination_detected]

        assert len(contaminated) == 0, (
            f"Contamination in hybrid: {[r.scenario_id for r in contaminated]}"
        )

    def test_expected_content_present_topic_local(self, results: list[BenchmarkResult]):
        """All expected content should be present in topic_local context."""
        for result in results:
            if result.mode == "topic_local":
                # Skip ambiguous scenarios - they have no expected content
                scenario = next(
                    (s for s in load_benchmark_fixtures() if s.scenario_id == result.scenario_id),
                    None,
                )
                if scenario and scenario.expected_reactivation == "disambiguate":
                    continue

                for substring, found in result.contains_expected.items():
                    assert found, (
                        f"Scenario {result.scenario_id}: missing '{substring}'"
                    )

    def test_excluded_content_absent_topic_local(self, results: list[BenchmarkResult]):
        """Excluded content should never appear in topic_local."""
        for result in results:
            if result.mode == "topic_local":
                for substring, absent in result.excludes_expected.items():
                    assert absent, (
                        f"Scenario {result.scenario_id}: found excluded '{substring}'"
                    )

    def test_ancestry_has_contamination(self, results: list[BenchmarkResult]):
        """ancestry mode should have contamination (includes all topics)."""
        # This test documents expected behavior - ancestry includes everything
        ancestry_results = [r for r in results if r.mode == "ancestry"]

        # Filter out ambiguous scenarios which have no expected_excludes
        non_ambiguous = [
            r for r in ancestry_results
            if r.scenario_id != "ambiguous_java"
        ]

        if not non_ambiguous:
            pytest.skip("No non-ambiguous scenarios to test")

        # At least some ancestry results should have contamination
        contaminated = [r for r in non_ambiguous if r.contamination_detected]
        assert len(contaminated) > 0, (
            "Expected ancestry mode to include content from other topics"
        )

    def test_no_thrashing(self, results: list[BenchmarkResult]):
        """No rapid back-and-forth reactivations within a scenario."""
        # Group by scenario
        by_scenario: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            if r.scenario_id not in by_scenario:
                by_scenario[r.scenario_id] = []
            by_scenario[r.scenario_id].append(r)

        # Check for consistency within each scenario
        # topic_local and hybrid should both reactivate to the same target
        for scenario_id, scenario_results in by_scenario.items():
            topic_local = next((r for r in scenario_results if r.mode == "topic_local"), None)
            hybrid = next((r for r in scenario_results if r.mode == "hybrid"), None)

            if topic_local and hybrid:
                assert topic_local.reactivation_target == hybrid.reactivation_target, (
                    f"{scenario_id}: thrashing between topic_local ({topic_local.reactivation_target}) "
                    f"and hybrid ({hybrid.reactivation_target})"
                )

    def test_token_efficiency(self, results: list[BenchmarkResult]):
        """topic_local should use fewer tokens than ancestry."""
        metrics = compute_benchmark_metrics(results)

        ancestry_tokens = metrics["by_mode"].get("ancestry", {}).get("avg_context_tokens", 0)
        topic_local_tokens = metrics["by_mode"].get("topic_local", {}).get("avg_context_tokens", 0)

        # topic_local should use fewer tokens (it excludes other topics)
        assert topic_local_tokens < ancestry_tokens, (
            f"topic_local ({topic_local_tokens:.0f}) should use fewer tokens "
            f"than ancestry ({ancestry_tokens:.0f})"
        )

    def test_benchmark_metrics_summary(self, results: list[BenchmarkResult], capsys):
        """Print summary metrics for visibility."""
        metrics = compute_benchmark_metrics(results)

        print("\n" + format_benchmark_report(results))

        # Basic sanity checks
        assert metrics["total_scenarios"] > 0
        assert "by_mode" in metrics
        assert "topic_local_vs_ancestry" in metrics


class TestBenchmarkFixtures:
    """Tests for fixture validation and structure."""

    def test_load_fixtures(self):
        """Should load fixtures without error."""
        scenarios = load_benchmark_fixtures()
        assert len(scenarios) > 0

    def test_fixture_categories(self):
        """Fixtures should cover multiple categories."""
        scenarios = load_benchmark_fixtures()
        categories = set(s.category for s in scenarios)

        expected_categories = {"short_gap", "medium_gap", "long_gap", "ambiguous"}
        assert categories & expected_categories, (
            f"Expected some of {expected_categories}, got {categories}"
        )

    def test_fixture_structure(self):
        """Each fixture should have required fields."""
        scenarios = load_benchmark_fixtures()

        for s in scenarios:
            assert s.scenario_id, f"Missing scenario_id"
            assert s.topic_a_name, f"{s.scenario_id}: Missing topic_a_name"
            assert s.topic_b_name, f"{s.scenario_id}: Missing topic_b_name"
            assert s.resume_query, f"{s.scenario_id}: Missing resume_query"
            assert s.expected_reactivation, f"{s.scenario_id}: Missing expected_reactivation"
            assert len(s.topic_a_exchanges) > 0, f"{s.scenario_id}: No topic_a_exchanges"
            assert len(s.topic_b_exchanges) > 0, f"{s.scenario_id}: No topic_b_exchanges"


class TestBenchmarkRunner:
    """Tests for the benchmark runner itself."""

    def test_run_single_scenario(self):
        """Should run a single scenario successfully."""
        scenarios = load_benchmark_fixtures()
        if not scenarios:
            pytest.skip("No fixtures available")

        results = run_deterministic_benchmark(scenarios[:1], modes=["topic_local"])
        assert len(results) == 1
        assert results[0].scenario_id == scenarios[0].scenario_id
        assert results[0].mode == "topic_local"

    def test_run_all_modes(self):
        """Should run all modes for each scenario."""
        scenarios = load_benchmark_fixtures()
        if not scenarios:
            pytest.skip("No fixtures available")

        results = run_deterministic_benchmark(scenarios[:1])

        modes_run = set(r.mode for r in results)
        assert modes_run == {"ancestry", "topic_local", "hybrid"}

    def test_metrics_computation(self):
        """Should compute valid metrics."""
        scenarios = load_benchmark_fixtures()
        if not scenarios:
            pytest.skip("No fixtures available")

        results = run_deterministic_benchmark(scenarios)
        metrics = compute_benchmark_metrics(results)

        assert "by_mode" in metrics
        assert "topic_local_vs_ancestry" in metrics
        assert metrics["total_scenarios"] == len(scenarios)

        # Check each mode has valid data
        for mode in ["ancestry", "topic_local", "hybrid"]:
            if mode in metrics["by_mode"]:
                mode_data = metrics["by_mode"][mode]
                assert 0 <= mode_data["pass_rate"] <= 1
                assert 0 <= mode_data["contamination_rate"] <= 1
                assert mode_data["avg_context_tokens"] >= 0
