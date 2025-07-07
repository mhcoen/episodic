#!/usr/bin/env python3
"""Test secho_color function"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from episodic.color_utils import secho_color

print("Testing secho_color:\n")

# Test simple bold
secho_color("This should be BOLD cyan", fg="cyan", bold=True)

# Test printing word by word
print("\nWord by word (should be bold):")
words = ["###", "Top", "Movies", "in", "Theaters"]
for word in words:
    secho_color(word, fg="cyan", nl=False, bold=True)
    secho_color(" ", fg="cyan", nl=False)  # Space without bold
print()

print("\nWord by word with consistent style:")
for word in words:
    secho_color(word + " ", fg="cyan", nl=False, bold=True)
print()

print("\nDirect click.style test:")
import click
styled = click.style("### Top Movies in Theaters", fg="cyan", bold=True)
click.echo(styled)