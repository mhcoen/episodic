#!/usr/bin/env python3
"""
Basic test runner for Episodic core functionality.

Runs the comprehensive test suite for:
- Core conversation flow
- Topic detection
- Configuration management
"""

import unittest
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests(test_pattern=None, verbosity=2):
    """Run the test suite."""
    # Test modules to run
    test_modules = [
        'test_core',
        'test_config', 
        'test_conversation_flow',
        'test_topic_detection_comprehensive',
        'test_configuration_comprehensive'
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_pattern:
        # Run specific test pattern
        for module in test_modules:
            try:
                test_module = __import__(module)
                pattern_suite = loader.loadTestsFromName(test_pattern, test_module)
                suite.addTest(pattern_suite)
            except (ImportError, AttributeError):
                continue
    else:
        # Run all tests
        for module in test_modules:
            try:
                suite.addTest(loader.loadTestsFromName(module))
            except ImportError as e:
                print(f"Warning: Could not import {module}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Episodic basic test suite')
    parser.add_argument('test_pattern', nargs='?', 
                       help='Specific test pattern to run (e.g., TestCore.test_node_creation)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Increase verbosity')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Decrease verbosity')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 2  # Default
    if args.verbose:
        verbosity = 3
    elif args.quiet:
        verbosity = 1
    
    print(f"\n{'='*70}")
    print(f"Running Episodic Basic Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Run tests
    result = run_tests(args.test_pattern, verbosity)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")
        
        if result.failures:
            print(f"\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
                
        if result.errors:
            print(f"\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())