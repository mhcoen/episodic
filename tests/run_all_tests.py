#!/usr/bin/env python3
"""
Enhanced test runner for Episodic test suite.

Provides options for running different test categories and
generating coverage reports.
"""

import sys
import os
import unittest
import argparse
from pathlib import Path
import time
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_tests(test_dir: str, pattern: str = "test_*.py") -> unittest.TestSuite:
    """Discover tests in a directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern, top_level_dir=str(project_root))
    return suite


def run_test_suite(suite: unittest.TestSuite, verbosity: int = 2) -> unittest.TestResult:
    """Run a test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def print_summary(results: List[Tuple[str, unittest.TestResult]]):
    """Print summary of all test results."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for category, result in results:
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        status = "✅ PASSED" if failures == 0 and errors == 0 else "❌ FAILED"
        print(f"\n{category}:")
        print(f"  Tests run: {tests_run}")
        print(f"  Failures: {failures}")
        print(f"  Errors: {errors}")
        print(f"  Skipped: {skipped}")
        print(f"  Status: {status}")
    
    print("\n" + "-" * 70)
    print(f"TOTAL: {total_tests} tests, {total_failures} failures, {total_errors} errors, {total_skipped} skipped")
    
    overall_status = "✅ ALL TESTS PASSED" if total_failures == 0 and total_errors == 0 else "❌ SOME TESTS FAILED"
    print(f"\n{overall_status}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Episodic test suite")
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "quick", "topics", "commands", "coverage"],
        help="Test category to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase test output verbosity"
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    verbosity = 2 if args.verbose else 1
    
    # Configure test runner
    if args.failfast:
        unittest.TestResult.failfast = True
    
    results = []
    test_dir = Path(__file__).parent
    
    print(f"Running {args.category} tests...\n")
    start_time = time.time()
    
    if args.category == "all":
        # Run all test categories
        categories = [
            ("Unit Tests", str(test_dir / "unit")),
            ("Integration Tests", str(test_dir / "integration")),
            ("Core Tests", str(test_dir)),
        ]
        
        for category_name, category_dir in categories:
            if os.path.exists(category_dir):
                print(f"\n{'=' * 50}")
                print(f"Running {category_name}")
                print('=' * 50)
                suite = discover_tests(category_dir)
                if suite.countTestCases() > 0:
                    result = run_test_suite(suite, verbosity)
                    results.append((category_name, result))
                else:
                    print(f"No tests found in {category_dir}")
                    
    elif args.category == "unit":
        # Run only unit tests
        suite = discover_tests(str(test_dir / "unit"))
        result = run_test_suite(suite, verbosity)
        results.append(("Unit Tests", result))
        
    elif args.category == "integration":
        # Run only integration tests
        suite = discover_tests(str(test_dir / "integration"))
        result = run_test_suite(suite, verbosity)
        results.append(("Integration Tests", result))
        
    elif args.category == "quick":
        # Run only stable, fast tests
        quick_tests = [
            "test_config.py",
            "test_prompt_manager.py",
            "test_caching.py",
            "test_core.py"
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_file in quick_tests:
            test_path = test_dir / test_file
            if test_path.exists():
                module_name = f"tests.{test_file[:-3]}"
                try:
                    module = __import__(module_name, fromlist=[''])
                    suite.addTests(loader.loadTestsFromModule(module))
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
        
        result = run_test_suite(suite, verbosity)
        results.append(("Quick Tests", result))
        
    elif args.category == "topics":
        # Run topic-related tests
        suite = discover_tests(str(test_dir / "unit" / "topics"))
        integration_suite = discover_tests(
            str(test_dir / "integration"),
            pattern="*topic*.py"
        )
        suite.addTests(integration_suite)
        
        result = run_test_suite(suite, verbosity)
        results.append(("Topic Tests", result))
        
    elif args.category == "commands":
        # Run command-related tests
        suite = discover_tests(str(test_dir / "unit" / "commands"))
        result = run_test_suite(suite, verbosity)
        results.append(("Command Tests", result))
        
    elif args.category == "coverage":
        # Run with coverage
        try:
            import coverage
        except ImportError:
            print("Coverage not installed. Install with: pip install coverage")
            sys.exit(1)
            
        cov = coverage.Coverage(source=['episodic'])
        cov.start()
        
        # Run all tests
        all_suite = discover_tests(str(test_dir))
        result = run_test_suite(all_suite, verbosity)
        results.append(("All Tests (with coverage)", result))
        
        cov.stop()
        cov.save()
        
        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)
        cov.report()
        
        # Generate HTML report
        html_dir = test_dir / "htmlcov"
        cov.html_report(directory=str(html_dir))
        print(f"\nDetailed HTML coverage report generated in: {html_dir}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print_summary(results)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    # Exit with appropriate code
    total_failures = sum(len(r.failures) + len(r.errors) for _, r in results)
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()