#!/usr/bin/env python3
"""
Triage test failures into buckets for analysis.

Run: pytest tests/ --tb=line -q 2>&1 | python scripts/triage_test_failures.py

Output format:
  ERRORS (62 total):
    - 45 in tests/retrieval/* : ValueError: Attempted to create database in project directory
    - 12 in tests/unit/test_resolver.py : ImportError: cannot import name 'X'

  FAILURES (55 total):
    - 23 in tests/integration/test_memory* : AssertionError: expected X got Y
    - etc.
"""

import sys
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FailureInfo:
    """Information about a single test failure or error."""
    test_path: str
    test_name: str
    exception_type: str
    exception_message: str
    is_error: bool  # True for ERROR, False for FAILURE


def parse_pytest_output(lines: List[str]) -> Tuple[List[FailureInfo], Dict[str, int]]:
    """Parse pytest output to extract failure info."""
    failures = []
    summary = {}

    # Patterns for different failure formats
    error_pattern = re.compile(r'^ERROR\s+(.+?)::(.+?)\s*-\s*(.+?):\s*(.*)$')
    failed_pattern = re.compile(r'^FAILED\s+(.+?)::(.+?)\s*-\s*(.+?):\s*(.*)$')
    short_error_pattern = re.compile(r'^ERROR at setup of (.+?)$')
    short_failed_pattern = re.compile(r'^FAILED (.+?)$')

    # Also look for inline errors like:
    # E   ValueError: Attempted to create database...
    current_test = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Match ERROR/FAILED patterns
        error_match = error_pattern.match(line)
        failed_match = failed_pattern.match(line)

        if error_match:
            test_path, test_name, exc_type, exc_msg = error_match.groups()
            failures.append(FailureInfo(
                test_path=test_path,
                test_name=test_name,
                exception_type=exc_type,
                exception_message=exc_msg,
                is_error=True
            ))
        elif failed_match:
            test_path, test_name, exc_type, exc_msg = failed_match.groups()
            failures.append(FailureInfo(
                test_path=test_path,
                test_name=test_name,
                exception_type=exc_type,
                exception_message=exc_msg,
                is_error=False
            ))

        # Look for the setup error format
        if 'ERROR at setup of' in line:
            match = re.search(r'ERROR at setup of (.+)', line)
            if match:
                test_full = match.group(1).strip()
                # Look ahead for the error message
                exc_type = "SetupError"
                exc_msg = ""
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('E '):
                        exc_line = lines[j].strip()[2:].strip()
                        if ':' in exc_line:
                            exc_type, exc_msg = exc_line.split(':', 1)
                            exc_type = exc_type.strip()
                            exc_msg = exc_msg.strip()
                        else:
                            exc_msg = exc_line
                        break

                # Parse test_full into path and name
                if '::' in test_full:
                    parts = test_full.split('::')
                    test_path = parts[0]
                    test_name = '::'.join(parts[1:])
                else:
                    test_path = test_full
                    test_name = ""

                failures.append(FailureInfo(
                    test_path=test_path,
                    test_name=test_name,
                    exception_type=exc_type,
                    exception_message=exc_msg,
                    is_error=True
                ))

        # Parse summary line
        summary_match = re.search(r'(\d+) failed.*?(\d+) passed.*?(\d+) skipped.*?(\d+) errors', line)
        if summary_match:
            summary = {
                'failed': int(summary_match.group(1)),
                'passed': int(summary_match.group(2)),
                'skipped': int(summary_match.group(3)),
                'errors': int(summary_match.group(4)),
            }

    return failures, summary


def categorize_root_cause(exc_type: str, exc_msg: str) -> str:
    """Categorize the root cause of a failure."""
    exc_msg_lower = exc_msg.lower()

    if 'database in project directory' in exc_msg_lower or ':memory:' in exc_msg_lower:
        return "PATH_ISSUE: Database path validation blocking :memory:"
    elif 'importerror' in exc_type.lower() or 'modulenotfounderror' in exc_type.lower():
        return "IMPORT_ERROR: Missing import or circular dependency"
    elif 'connectionerror' in exc_type.lower() or 'chroma' in exc_msg_lower:
        return "MISSING_DEP: Chroma/vector DB not available"
    elif 'assertionerror' in exc_type.lower():
        return "LOGIC_ERROR: Test assertion failed"
    elif 'keyerror' in exc_type.lower():
        return "LOGIC_ERROR: Missing key in dict"
    elif 'attributeerror' in exc_type.lower():
        return "LOGIC_ERROR: Attribute not found"
    elif 'typeerror' in exc_type.lower():
        return "LOGIC_ERROR: Type mismatch"
    elif 'valueerror' in exc_type.lower():
        return "LOGIC_ERROR: Invalid value"
    elif 'timeout' in exc_msg_lower or 'flaky' in exc_msg_lower:
        return "FLAKY: Timeout or intermittent failure"
    else:
        return f"UNKNOWN: {exc_type}"


def get_directory_prefix(test_path: str) -> str:
    """Extract directory prefix from test path."""
    if test_path.startswith('tests/integration/'):
        return 'tests/integration/*'
    elif test_path.startswith('tests/unit/'):
        # Get one more level
        parts = test_path.split('/')
        if len(parts) >= 3:
            return f'tests/unit/{parts[2]}/*'
        return 'tests/unit/*'
    elif test_path.startswith('tests/retrieval/'):
        return 'tests/retrieval/*'
    elif test_path.startswith('tests/'):
        return f'tests/{test_path.split("/")[1]}'
    return test_path


def group_failures(failures: List[FailureInfo]) -> Dict[str, List[FailureInfo]]:
    """Group failures by root cause + directory."""
    groups = defaultdict(list)

    for f in failures:
        root_cause = categorize_root_cause(f.exception_type, f.exception_message)
        dir_prefix = get_directory_prefix(f.test_path)
        key = f"{root_cause} | {dir_prefix}"
        groups[key].append(f)

    return groups


def print_triage_report(failures: List[FailureInfo], summary: Dict[str, int]):
    """Print the triage report."""
    errors = [f for f in failures if f.is_error]
    fails = [f for f in failures if not f.is_error]

    print("=" * 70)
    print("TEST FAILURE TRIAGE REPORT")
    print("=" * 70)

    if summary:
        print(f"\nSummary: {summary.get('passed', 0)} passed, "
              f"{summary.get('failed', 0)} failed, "
              f"{summary.get('errors', 0)} errors, "
              f"{summary.get('skipped', 0)} skipped")

    if errors:
        print(f"\n{'='*70}")
        print(f"ERRORS ({len(errors)} total)")
        print("=" * 70)

        error_groups = group_failures(errors)
        for key in sorted(error_groups.keys(), key=lambda k: -len(error_groups[k])):
            group = error_groups[key]
            parts = key.split(' | ')
            root_cause = parts[0] if parts else key
            dir_prefix = parts[1] if len(parts) > 1 else ""

            print(f"\n  [{len(group):3d}] {dir_prefix}")
            print(f"        Root cause: {root_cause}")
            if group:
                sample = group[0]
                print(f"        Sample: {sample.exception_type}: {sample.exception_message[:60]}...")

    if fails:
        print(f"\n{'='*70}")
        print(f"FAILURES ({len(fails)} total)")
        print("=" * 70)

        fail_groups = group_failures(fails)
        for key in sorted(fail_groups.keys(), key=lambda k: -len(fail_groups[k])):
            group = fail_groups[key]
            parts = key.split(' | ')
            root_cause = parts[0] if parts else key
            dir_prefix = parts[1] if len(parts) > 1 else ""

            print(f"\n  [{len(group):3d}] {dir_prefix}")
            print(f"        Root cause: {root_cause}")
            if group:
                sample = group[0]
                print(f"        Sample: {sample.exception_type}: {sample.exception_message[:60]}...")

    # Summary by root cause
    print(f"\n{'='*70}")
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)

    all_groups = group_failures(failures)
    cause_counts = defaultdict(int)
    for key, group in all_groups.items():
        root_cause = key.split(' | ')[0]
        cause_counts[root_cause] += len(group)

    for cause in sorted(cause_counts.keys(), key=lambda c: -cause_counts[c]):
        print(f"  {cause_counts[cause]:3d}  {cause}")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    if any('PATH_ISSUE' in key for key in all_groups.keys()):
        print("\n  1. PATH_ISSUE: Fix database path validation in test fixtures")
        print("     - Update conftest.py to properly set EPISODIC_DB_PATH for tests")
        print("     - Or update db_connection.py to allow :memory: in test mode")

    if any('LOGIC_ERROR' in key for key in all_groups.keys()):
        print("\n  2. LOGIC_ERROR: Review test expectations vs implementation")
        print("     - Some tests may have outdated assertions")
        print("     - Consider which tests are testing current vs deprecated behavior")

    if any('IMPORT_ERROR' in key for key in all_groups.keys()):
        print("\n  3. IMPORT_ERROR: Check module dependencies")
        print("     - Ensure all required modules are installed")
        print("     - Check for circular imports")


def main():
    """Main entry point."""
    # Read from stdin
    lines = sys.stdin.readlines()

    failures, summary = parse_pytest_output(lines)

    # If we didn't find failures in the expected format, try to count from FAILED/ERROR lines
    if not failures:
        for line in lines:
            line = line.strip()
            if line.startswith('FAILED ') or line.startswith('ERROR '):
                # Simple parse
                if '::' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        test_ref = parts[1]
                        path_parts = test_ref.split('::')
                        test_path = path_parts[0]
                        test_name = '::'.join(path_parts[1:]) if len(path_parts) > 1 else ""

                        exc_info = ' '.join(parts[3:]) if len(parts) > 3 else "Unknown"
                        exc_type = exc_info.split(':')[0] if ':' in exc_info else "Unknown"
                        exc_msg = exc_info.split(':', 1)[1] if ':' in exc_info else exc_info

                        failures.append(FailureInfo(
                            test_path=test_path,
                            test_name=test_name,
                            exception_type=exc_type,
                            exception_message=exc_msg,
                            is_error=line.startswith('ERROR ')
                        ))

    print_triage_report(failures, summary)


if __name__ == "__main__":
    main()
