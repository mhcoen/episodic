#!/usr/bin/env python3
"""
CLI Test Harness for Episodic.

Runs actual CLI sessions and verifies output conditions.
Use this to test fixes before claiming they work.

Usage:
    python tests/cli_harness.py                    # Run all scenarios
    python tests/cli_harness.py reactivation       # Run specific scenario
    python tests/cli_harness.py --list             # List available scenarios
"""

import os
import sys
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple

# Test scenarios registry
SCENARIOS = {}


def scenario(name: str, description: str):
    """Decorator to register a test scenario."""
    def decorator(func):
        SCENARIOS[name] = {"func": func, "description": description}
        return func
    return decorator


@dataclass
class CLIResult:
    """Result of a CLI session."""
    output: str
    exit_code: int
    db_path: str
    chroma_path: str


@dataclass
class TestResult:
    """Result of a test scenario."""
    name: str
    passed: bool
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (check_name, passed, details)
    output: str = ""
    error: str = ""


def run_cli_session(commands: List[str], timeout: int = 300) -> CLIResult:
    """
    Run a CLI session with the given commands.

    Creates a fresh database for each session.
    Returns the full output.
    """
    # Create temp directories for this session
    temp_dir = tempfile.mkdtemp(prefix="episodic_test_")
    db_path = os.path.join(temp_dir, "test.db")
    chroma_path = os.path.join(temp_dir, "chroma")
    os.makedirs(chroma_path, exist_ok=True)

    # Build input
    input_text = "\n".join(commands) + "\n"

    # Set up environment
    env = os.environ.copy()
    env["EPISODIC_DB_PATH"] = db_path
    env["EPISODIC_CHROMA_PATH"] = chroma_path

    # Run CLI
    try:
        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(__file__).parent.parent)
        )

        # Strip ANSI codes for easier parsing
        output = strip_ansi(result.stdout + result.stderr)

        return CLIResult(
            output=output,
            exit_code=result.returncode,
            db_path=db_path,
            chroma_path=chroma_path
        )
    except subprocess.TimeoutExpired:
        return CLIResult(
            output="TIMEOUT",
            exit_code=-1,
            db_path=db_path,
            chroma_path=chroma_path
        )
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m|\x1b\[\?[0-9;]*[a-zA-Z]|\x1b\[[0-9;]*[a-zA-Z]')
    return ansi_pattern.sub('', text)


def extract_response_after(output: str, marker: str) -> str:
    """Extract the LLM response after a specific user input marker."""
    lines = output.split('\n')
    found_marker = False
    found_box_end = False
    response_lines = []

    for i, line in enumerate(lines):
        if marker in line:  # Found the marker
            found_marker = True
            continue

        if found_marker and not found_box_end:
            # Skip until we pass the input box (ends with ╯ or ╰)
            if '╯' in line or '╰' in line:
                found_box_end = True
            continue

        if found_box_end:
            # Stop at next user input box or /quit
            if '╭' in line and '>' in line:
                break
            if '/quit' in line:
                break
            # Skip debug lines
            if '[DEBUG]' in line:
                continue
            if line.strip().startswith('[') and ']' in line:
                continue
            # Skip semantic drift indicator (it's metadata, not response)
            if 'Semantic drift:' in line:
                continue
            if 'DEBUG:' in line:
                continue
            # Skip empty lines at start
            if not response_lines and not line.strip():
                continue
            response_lines.append(line)

    return '\n'.join(response_lines).strip()


def count_words(text: str) -> int:
    """Count words in text, excluding code blocks."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Count words
    return len(text.split())


def has_pattern(text: str, pattern: str) -> bool:
    """Check if text contains a regex pattern."""
    return bool(re.search(pattern, text, re.IGNORECASE))


# ============================================================================
# TEST SCENARIOS
# ============================================================================

@scenario("reactivation_style", "Test that /style concise is respected after topic reactivation")
def test_reactivation_style() -> TestResult:
    """
    Verifies:
    1. /style concise is acknowledged
    2. Topic reactivation fires ("🔄 Resuming topic:" appears)
    3. The reactivated response is concise (under 150 words, no headers)
    """
    result = TestResult(name="reactivation_style", passed=True)

    # Need more messages to trigger topic creation and allow dormant topics
    commands = [
        "/style concise",
        "/set enable_topic_reactivation true",
        "/set show_reactivation_decisions true",
        # First topic: Python retry (need 3+ messages to create topic)
        "How do I handle retry logic in Python with exponential backoff?",
        "What libraries can I use for this?",
        "Can you show me a code example?",
        # Switch to second topic: Sourdough (creates topic, makes Python dormant)
        "Let's talk about sourdough bread. What's a good starter ratio?",
        "How long should I proof the dough?",
        "What temperature is best for proofing?",
        # Resume first topic
        "Back to that Python retry thing - should I use tenacity?",
        "/quit"
    ]

    cli_result = run_cli_session(commands, timeout=180)
    result.output = cli_result.output

    if cli_result.output == "TIMEOUT":
        result.passed = False
        result.error = "CLI session timed out"
        return result

    # Check 1: Style was set
    style_set = "Response style set to concise" in cli_result.output
    result.checks.append(("style_set", style_set,
        "Found 'Response style set to concise'" if style_set else "Style confirmation not found"))

    # Check 2: Reactivation fired
    reactivation_fired = "Resuming topic:" in cli_result.output or "🔄 Resuming topic:" in cli_result.output
    result.checks.append(("reactivation_fired", reactivation_fired,
        "Found 'Resuming topic:'" if reactivation_fired else "Reactivation message not found"))

    # Check 3: Extract the reactivated response and check length
    # Find response after "Back to that Python retry thing"
    response = extract_response_after(cli_result.output, "Back to that Python retry thing")
    word_count = count_words(response)

    # Debug: show extracted response
    print(f"\n--- Extracted reactivated response ({word_count} words) ---")
    print(response[:500] if response else "(empty)")
    print("--- End extracted response ---\n")

    is_concise = word_count < 150
    result.checks.append(("response_concise", is_concise,
        f"Response has {word_count} words (limit: 150)" if is_concise
        else f"Response too long: {word_count} words (limit: 150)"))

    # Check 4: No section headers (indicates verbose tutorial style)
    has_headers = has_pattern(response, r'^#+\s|^[A-Z][a-z]+ [A-Z][a-z]+\n')
    no_headers = not has_headers
    result.checks.append(("no_headers", no_headers,
        "No section headers found" if no_headers else "Found section headers (indicates verbose style)"))

    # Overall pass/fail
    result.passed = all(check[1] for check in result.checks)

    return result


@scenario("reactivation_basic", "Test that topic reactivation fires at all")
def test_reactivation_basic() -> TestResult:
    """
    Verifies:
    1. Topic reactivation is enabled
    2. "🔄 Resuming topic:" message appears
    3. LLM provides a response (not just recall results)
    """
    result = TestResult(name="reactivation_basic", passed=True)

    # Need enough messages for topic creation (3+) and topic switch
    commands = [
        "/set enable_topic_reactivation true",
        "/set show_reactivation_decisions true",
        # Topic 1: Python retry
        "How do I handle retry logic in Python with exponential backoff?",
        "What libraries can I use?",
        "Show me a basic example",
        # Topic 2: Sourdough
        "Let's talk about sourdough bread. What's a good starter ratio?",
        "How long to proof?",
        "What temperature?",
        # Resume topic 1
        "Back to that Python retry thing - should I use tenacity?",
        "/quit"
    ]

    cli_result = run_cli_session(commands, timeout=180)
    result.output = cli_result.output

    if cli_result.output == "TIMEOUT":
        result.passed = False
        result.error = "CLI session timed out"
        return result

    # Check 1: Reactivation enabled
    react_enabled = "enable_topic_reactivation = True" in cli_result.output
    result.checks.append(("reactivation_enabled", react_enabled,
        "Reactivation enabled" if react_enabled else "Reactivation not enabled"))

    # Check 2: Reactivation fired
    reactivation_fired = "Resuming topic:" in cli_result.output
    result.checks.append(("reactivation_fired", reactivation_fired,
        "Reactivation fired" if reactivation_fired else "No reactivation message"))

    # Check 3: Got LLM response (not recall)
    has_recall_output = "Found" in cli_result.output and "topic(s)" in cli_result.output
    has_llm_response = "tenacity" in cli_result.output.lower() and len(cli_result.output) > 500

    got_answer = has_llm_response and not (has_recall_output and not reactivation_fired)
    result.checks.append(("got_llm_answer", got_answer,
        "Got LLM response about tenacity" if got_answer else "Got recall results instead of LLM answer"))

    result.passed = all(check[1] for check in result.checks)
    return result


@scenario("resume_cue_routing", "Test that resume cues route to chat, not recall")
def test_resume_cue_routing() -> TestResult:
    """
    Verifies that queries with resume cues go to chat flow, not recall system.
    """
    result = TestResult(name="resume_cue_routing", passed=True)

    commands = [
        "/set enable_topic_reactivation true",
        "/set debug true",
        "How do I handle retry logic in Python?",
        "Back to that Python retry thing - any best practices?",
        "/quit"
    ]

    cli_result = run_cli_session(commands, timeout=120)
    result.output = cli_result.output

    if cli_result.output == "TIMEOUT":
        result.passed = False
        result.error = "CLI session timed out"
        return result

    # Check: Resume cues detected in debug output
    resume_detected = "Resume cues detected" in cli_result.output
    result.checks.append(("resume_cues_detected", resume_detected,
        "Resume cues detected" if resume_detected else "No resume cue detection in debug output"))

    # Check: Fell through to chat
    fell_through = "falling through to chat" in cli_result.output
    result.checks.append(("fell_through_to_chat", fell_through,
        "Fell through to chat flow" if fell_through else "Did not fall through to chat"))

    result.passed = all(check[1] for check in result.checks)
    return result


# ============================================================================
# MAIN
# ============================================================================

def run_scenario(name: str) -> TestResult:
    """Run a single test scenario."""
    if name not in SCENARIOS:
        return TestResult(name=name, passed=False, error=f"Unknown scenario: {name}")

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Description: {SCENARIOS[name]['description']}")
    print('='*60)

    result = SCENARIOS[name]['func']()

    # Print results
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"\nResult: {status}")

    for check_name, passed, details in result.checks:
        check_status = "✓" if passed else "✗"
        print(f"  {check_status} {check_name}: {details}")

    if result.error:
        print(f"\nError: {result.error}")

    if not result.passed:
        print(f"\n--- OUTPUT (last 2000 chars) ---")
        print(result.output[-2000:] if len(result.output) > 2000 else result.output)
        print("--- END OUTPUT ---")

    return result


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("Available scenarios:")
            for name, info in SCENARIOS.items():
                print(f"  {name}: {info['description']}")
            return 0

        # Run specific scenario
        scenario_name = sys.argv[1]
        result = run_scenario(scenario_name)
        return 0 if result.passed else 1

    # Run all scenarios
    print("Running all CLI test scenarios...")

    results = []
    for name in SCENARIOS:
        result = run_scenario(name)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.name}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
