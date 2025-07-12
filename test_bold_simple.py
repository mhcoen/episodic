#!/usr/bin/env python3
"""Simple test script to verify bold formatting works in terminal."""

import typer
import sys

def test_bold():
    """Test different methods of bold output."""
    
    print("=== Bold Text Test ===")
    print()
    
    # Method 1: typer.secho with bold=True (what works in welcome message)
    print("1. Using typer.secho with bold=True:")
    typer.secho("This should be BOLD", bold=True)
    typer.secho("This should be normal", bold=False)
    print()
    
    # Method 2: Raw ANSI codes
    print("2. Using raw ANSI escape codes:")
    sys.stdout.write("\033[1mThis should be BOLD\033[0m\n")
    sys.stdout.write("This should be normal\n")
    print()
    
    # Method 3: typer.style 
    print("3. Using typer.style:")
    print(typer.style("This should be BOLD", bold=True))
    print(typer.style("This should be normal", bold=False))
    print()
    
    # Method 4: click.secho
    try:
        import click
        print("4. Using click.secho:")
        click.secho("This should be BOLD", bold=True)
        click.secho("This should be normal", bold=False)
    except ImportError:
        print("4. click not available")
    
    print()
    print("=== Test Complete ===")
    print("Which methods (if any) show bold text in your terminal?")

if __name__ == "__main__":
    test_bold()