#!/usr/bin/env python3
"""
Test runner for Episodic CLI tests.

This script runs all tests and provides comprehensive reporting.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColoredTextTestResult(unittest.TextTestResult):
    """Enhanced test result with colored output and timing."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.start_time = None
        self.test_times = {}
        self.verbosity = verbosity
    
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
    
    def stopTest(self, test):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.test_times[str(test)] = elapsed
        super().stopTest(test)
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            elapsed = self.test_times.get(str(test), 0)
            self.stream.write(f" \033[92m✓\033[0m ({elapsed:.3f}s)")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(" \033[91m✗ ERROR\033[0m")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(" \033[91m✗ FAIL\033[0m")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(" \033[93m⚠ SKIP\033[0m")
            self.stream.flush()


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    
    resultclass = ColoredTextTestResult
    
    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)


def discover_tests(test_dir=None):
    """Discover all test modules."""
    if test_dir is None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
    
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    return suite


def run_specific_test(test_module, test_class=None, test_method=None):
    """Run a specific test module, class, or method."""
    if test_method and test_class:
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method))
    elif test_class:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(test_module)
    
    runner = ColoredTextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests(verbosity=2):
    """Run all discovered tests."""
    print("\033[94m" + "="*60 + "\033[0m")
    print("\033[94m🧪 Running Episodic CLI Test Suite\033[0m")
    print("\033[94m" + "="*60 + "\033[0m")
    
    suite = discover_tests()
    runner = ColoredTextTestRunner(verbosity=verbosity)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "\033[94m" + "="*60 + "\033[0m")
    print("\033[94m📊 Test Summary\033[0m")
    print("\033[94m" + "="*60 + "\033[0m")
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests run: {total_tests}")
    print(f"\033[92mPassed: {passed}\033[0m")
    if failures > 0:
        print(f"\033[91mFailed: {failures}\033[0m")
    if errors > 0:
        print(f"\033[91mErrors: {errors}\033[0m")
    if skipped > 0:
        print(f"\033[93mSkipped: {skipped}\033[0m")
    
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Show success rate
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        if success_rate == 100:
            print(f"\033[92m✅ Success rate: {success_rate:.1f}%\033[0m")
        elif success_rate >= 90:
            print(f"\033[93m⚠️  Success rate: {success_rate:.1f}%\033[0m")
        else:
            print(f"\033[91m❌ Success rate: {success_rate:.1f}%\033[0m")
    
    return result


def run_quick_tests():
    """Run only quick unit tests (skip integration tests)."""
    print("\033[94m🚀 Running Quick Tests Only\033[0m")
    
    # Define quick test modules
    quick_tests = [
        'test_cli',
        'test_config', 
        'test_prompt_manager',
        'test_llm_integration',
        'test_caching'
    ]
    
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    for test_module in quick_tests:
        try:
            module = __import__(test_module, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError:
            print(f"Warning: Could not import {test_module}")
    
    runner = ColoredTextTestRunner(verbosity=2)
    return runner.run(suite)


def run_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        print("\033[91mError: coverage module not installed. Install with: pip install coverage\033[0m")
        return None
    
    print("\033[94m📈 Running Tests with Coverage\033[0m")
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    result = run_all_tests(verbosity=1)
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\n\033[94m📊 Coverage Report\033[0m")
    print("\033[94m" + "="*50 + "\033[0m")
    cov.report()
    
    return result


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            result = run_quick_tests()
        elif command == "coverage":
            result = run_coverage()
        elif command == "help":
            print("Episodic Test Runner")
            print("Usage:")
            print("  python run_tests.py [command]")
            print("")
            print("Commands:")
            print("  (no args)  - Run all tests")
            print("  quick      - Run only quick unit tests")
            print("  coverage   - Run tests with coverage reporting")
            print("  help       - Show this help message")
            return
        else:
            print(f"Unknown command: {command}")
            print("Use 'python run_tests.py help' for usage information")
            return
    else:
        result = run_all_tests()
    
    # Exit with appropriate code
    if result and (result.failures or result.errors):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()