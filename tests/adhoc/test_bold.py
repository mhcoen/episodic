#!/usr/bin/env python3
"""Test bold formatting in terminal"""

import typer

print("Testing bold formatting:\n")

# Test basic bold
typer.secho("This should be normal", fg="white")
typer.secho("This should be BOLD", fg="white", bold=True)
print()

# Test with different colors
typer.secho("Normal cyan", fg="cyan")
typer.secho("BOLD cyan", fg="cyan", bold=True)
print()

typer.secho("Normal green", fg="green")
typer.secho("BOLD green", fg="green", bold=True)
print()

# Test headers
typer.secho("### This is a normal header", fg="white")
typer.secho("### This is a BOLD header", fg="white", bold=True)
print()

# Test lists
typer.secho("1. Normal numbered list", fg="white")
typer.secho("1. BOLD numbered list", fg="white", bold=True)
print()

typer.secho("- Normal bullet list", fg="white")
typer.secho("- BOLD bullet list", fg="white", bold=True)