#!/usr/bin/env python3
"""
Remove unused imports from Python files in the episodic codebase.

This script uses autoflake to automatically remove unused imports and variables
from Python files, excluding test files and deprecated/old files.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run autoflake to remove unused imports."""
    # Get the project root
    project_root = Path(__file__).parent.parent
    episodic_dir = project_root / "episodic"
    
    # Files to exclude
    exclude_patterns = [
        "**/test_*.py",
        "**/tests/**",
        "**/scripts/**",
        "**/*_old.py",
        "**/*_original.py",
        "**/defunct/**",
        "**/visualization.py",  # Being replaced, don't touch
    ]
    
    # Build autoflake command
    cmd = [
        "autoflake",
        "--in-place",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        "--recursive",
        "--exclude", ",".join(exclude_patterns),
        str(episodic_dir)
    ]
    
    print("Running autoflake to remove unused imports...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # First check if autoflake is installed
        subprocess.run(["autoflake", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: autoflake is not installed.")
        print("Install it with: pip install autoflake")
        sys.exit(1)
    
    # Run autoflake
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully removed unused imports!")
            if result.stdout:
                print("\nChanges made:")
                print(result.stdout)
        else:
            print("Error running autoflake:")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\nDone! Review the changes with 'git diff' before committing.")


if __name__ == "__main__":
    main()