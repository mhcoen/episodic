#!/usr/bin/env python3
"""Analyze test results and create a detailed error report."""

import re
from collections import defaultdict

# Read the log file
with open("full_test_output.log", "r") as f:
    content = f.read()

# Split by test sections
test_sections = content.split("=" * 60)

# Parse results
results = {
    "passed": [],
    "failed": [],
    "unknown_commands": [],
    "import_errors": [],
    "attribute_errors": [],
    "type_errors": [],
    "other_errors": []
}

current_command = None
for section in test_sections:
    if "Testing:" in section:
        # Extract command
        match = re.search(r"Testing: (.+)", section)
        if match:
            current_command = match.group(1)
            
            # Check for errors
            if "Error executing command:" in section:
                error_match = re.search(r"Error executing command: (.+)", section)
                if error_match:
                    error_msg = error_match.group(1)
                    
                    if "cannot import" in error_msg or "No module named" in error_msg:
                        results["import_errors"].append((current_command, error_msg))
                    elif "has no attribute" in error_msg:
                        results["attribute_errors"].append((current_command, error_msg))
                    elif "type" in error_msg and "not supported" in error_msg:
                        results["type_errors"].append((current_command, error_msg))
                    elif "got an unexpected keyword argument" in error_msg:
                        results["type_errors"].append((current_command, error_msg))
                    elif "'module' object is not callable" in error_msg:
                        results["import_errors"].append((current_command, error_msg))
                    else:
                        results["other_errors"].append((current_command, error_msg))
                else:
                    results["failed"].append(current_command)
                    
            elif "Unknown command:" in section:
                results["unknown_commands"].append(current_command)
            else:
                # Check if it actually worked
                if current_command and "Exit code: 0" in section and "Error" not in section:
                    results["passed"].append(current_command)

# Print detailed report
print("EPISODIC CLI TEST RESULTS - DETAILED ERROR REPORT")
print("=" * 80)
print()

# Summary
total_tests = sum(len(results[k]) for k in results)
print(f"Total commands tested: ~{total_tests}")
print(f"Passed: {len(results['passed'])}")
print(f"Failed: {len(results['failed']) + len(results['unknown_commands']) + len(results['import_errors']) + len(results['attribute_errors']) + len(results['type_errors']) + len(results['other_errors'])}")
print()

# Unknown commands
if results["unknown_commands"]:
    print("UNKNOWN COMMANDS (need to be implemented or removed):")
    print("-" * 80)
    for cmd in sorted(set(results["unknown_commands"])):
        print(f"  - {cmd}")
    print()

# Import errors
if results["import_errors"]:
    print("IMPORT ERRORS (missing modules or incorrect imports):")
    print("-" * 80)
    seen = set()
    for cmd, error in results["import_errors"]:
        key = (cmd.split()[0], error)  # Group by base command
        if key not in seen:
            seen.add(key)
            print(f"  - {cmd}: {error}")
    print()

# Attribute errors
if results["attribute_errors"]:
    print("ATTRIBUTE ERRORS (missing methods or properties):")
    print("-" * 80)
    for cmd, error in results["attribute_errors"]:
        print(f"  - {cmd}: {error}")
    print()

# Type errors
if results["type_errors"]:
    print("TYPE ERRORS (incorrect function signatures or parameter types):")
    print("-" * 80)
    for cmd, error in results["type_errors"]:
        print(f"  - {cmd}: {error}")
    print()

# Other errors
if results["other_errors"]:
    print("OTHER ERRORS:")
    print("-" * 80)
    for cmd, error in results["other_errors"]:
        print(f"  - {cmd}: {error}")
    print()

# Working commands
print("WORKING COMMANDS:")
print("-" * 80)
for cmd in sorted(set(results["passed"])):
    print(f"  ✓ {cmd}")

# Save to file
with open("error_summary.md", "w") as f:
    f.write("# Episodic CLI Error Summary\n\n")
    f.write("## Statistics\n")
    f.write(f"- Total commands tested: ~{total_tests}\n")
    f.write(f"- Passed: {len(results['passed'])}\n")
    f.write(f"- Failed: {len(results['failed']) + len(results['unknown_commands']) + len(results['import_errors']) + len(results['attribute_errors']) + len(results['type_errors']) + len(results['other_errors'])}\n\n")
    
    f.write("## Issues to Fix\n\n")
    
    if results["unknown_commands"]:
        f.write("### Unknown Commands\n")
        f.write("These commands are not recognized and need to be implemented or removed from help:\n\n")
        for cmd in sorted(set(results["unknown_commands"])):
            f.write(f"- [ ] `{cmd}`\n")
        f.write("\n")
    
    if results["import_errors"]:
        f.write("### Import Errors\n")
        f.write("These commands fail due to missing modules or incorrect imports:\n\n")
        seen = set()
        for cmd, error in results["import_errors"]:
            key = (cmd.split()[0], error)
            if key not in seen:
                seen.add(key)
                f.write(f"- [ ] `{cmd}`: {error}\n")
        f.write("\n")
    
    if results["attribute_errors"]:
        f.write("### Attribute Errors\n")
        f.write("These commands fail due to missing methods or properties:\n\n")
        for cmd, error in results["attribute_errors"]:
            f.write(f"- [ ] `{cmd}`: {error}\n")
        f.write("\n")
    
    if results["type_errors"]:
        f.write("### Type Errors\n")
        f.write("These commands fail due to incorrect function signatures:\n\n")
        for cmd, error in results["type_errors"]:
            f.write(f"- [ ] `{cmd}`: {error}\n")
        f.write("\n")
    
    if results["other_errors"]:
        f.write("### Other Errors\n")
        for cmd, error in results["other_errors"]:
            f.write(f"- [ ] `{cmd}`: {error}\n")
        f.write("\n")

print("\nDetailed error summary saved to error_summary.md")