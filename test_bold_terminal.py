#!/usr/bin/env python3
"""
Simple test script to verify bold formatting works in terminal
"""

import sys
import os

def test_bold_methods():
    """Test different methods of outputting bold text"""
    
    print("=" * 60)
    print("BOLD FORMATTING TEST")
    print("=" * 60)
    
    print("\n1. Raw ANSI escape codes:")
    sys.stdout.write("\033[1mThis should be bold\033[0m and this should be normal\n")
    sys.stdout.flush()
    
    print("\n2. Using typer/click:")
    try:
        import typer
        typer.secho("This should be bold (typer)", bold=True)
        typer.secho("This should be normal (typer)", bold=False)
    except ImportError:
        print("typer not available")
    
    print("\n3. Using click directly:")
    try:
        import click
        click.secho("This should be bold (click)", bold=True)
        click.secho("This should be normal (click)", bold=False)
    except ImportError:
        print("click not available")
    
    print("\n4. Using episodic's secho_color:")
    try:
        sys.path.insert(0, '/Users/mhcoen/proj/episodic')
        from episodic.color_utils import secho_color
        secho_color("This should be bold (episodic)", bold=True)
        secho_color("This should be normal (episodic)", bold=False)
    except Exception as e:
        print(f"episodic secho_color failed: {e}")
    
    print("\n5. Terminal info:")
    print(f"TERM: {os.environ.get('TERM', 'not set')}")
    print(f"COLORTERM: {os.environ.get('COLORTERM', 'not set')}")
    print(f"Terminal supports colors: {sys.stdout.isatty()}")
    
    print("\n6. Testing different bold patterns:")
    patterns = [
        ("Headers", "### This Should Be Bold"),
        ("List items", "- This should be bold: normal text after colon"),
        ("Numbered", "1. This should be bold: normal text after colon"),
        ("Italic as bold", "*This should render as bold*"),
    ]
    
    for name, text in patterns:
        print(f"\n{name}: {text}")
    
    print("\n" + "=" * 60)
    print("If you see bold text above, your terminal supports it.")
    print("If not, your terminal may not support bold formatting.")
    print("=" * 60)

if __name__ == "__main__":
    test_bold_methods()