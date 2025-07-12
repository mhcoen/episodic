#!/usr/bin/env python3
"""Analyze the profile.stats file"""
import pstats
import sys

# Load the profile stats
stats = pstats.Stats('profile_final.stats')

# Sort by cumulative time and print top 50
print("Top 50 functions by cumulative time:")
print("=" * 80)
stats.sort_stats('cumulative').print_stats(50)

# Also show callers of the slowest functions
print("\n\nCallers of top time-consuming functions:")
print("=" * 80)
stats.print_callers(10)