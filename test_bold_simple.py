#!/usr/bin/env python3
"""Test if bold works in terminal."""

import typer

print("Testing bold formatting:\n")

# Test basic bold
typer.secho("This is normal text", fg=typer.colors.GREEN)
typer.secho("This is BOLD text", fg=typer.colors.GREEN, bold=True)
typer.secho("This is normal again", fg=typer.colors.GREEN)

print("\nTesting inline bold:")
typer.secho("This is ", fg=typer.colors.GREEN, nl=False)
typer.secho("BOLD", fg=typer.colors.GREEN, bold=True, nl=False)
typer.secho(" text inline", fg=typer.colors.GREEN)