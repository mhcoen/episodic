#!/usr/bin/env python3
"""
Remove unused imports from Python files.

This script uses autoflake to automatically remove unused imports
from Python files in the codebase.
"""

import os
import sys
import subprocess
from pathlib import Path


def remove_unused_imports(file_path: str, dry_run: bool = True):
    """Remove unused imports from a single file."""
    cmd = [
        "autoflake",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
    ]
    
    if not dry_run:
        cmd.append("--in-place")
    
    cmd.append(file_path)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Changes for {file_path}:")
            print(result.stdout)
            return True
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")
        return False


def process_directory(directory: str, dry_run: bool = True):
    """Process all Python files in a directory."""
    changes_found = False
    
    for root, dirs, files in os.walk(directory):
        # Skip test directories and __pycache__
        if '__pycache__' in root or 'tests' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if remove_unused_imports(file_path, dry_run):
                    changes_found = True
    
    return changes_found


def main():
    # Check if autoflake is installed
    try:
        subprocess.run(["autoflake", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: autoflake is not installed.")
        print("Install it with: pip install autoflake")
        sys.exit(1)
    
    # Default to dry run
    dry_run = "--apply" not in sys.argv
    
    if dry_run:
        print("Running in DRY RUN mode. Use --apply to make changes.\n")
    else:
        print("APPLYING changes to files.\n")
    
    # Process the episodic package
    episodic_dir = Path(__file__).parent.parent.parent / "episodic"
    
    if process_directory(str(episodic_dir), dry_run):
        print("\n" + "="*50)
        if dry_run:
            print("To apply these changes, run: python remove_unused_imports.py --apply")
        else:
            print("Changes applied successfully!")
    else:
        print("\nNo unused imports found.")


if __name__ == "__main__":
    main()