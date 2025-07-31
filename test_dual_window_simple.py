#!/usr/bin/env python3
"""
Simple test of dual-window detection by running commands in episodic.
"""

from episodic.config import config

# Set configuration
print("Setting configuration for dual-window detection...")
config.set('debug', True)
config.set('use_dual_window_detection', True)
config.set('use_sliding_window_detection', False)
config.set('dual_window_high_precision_threshold', 0.2)
config.set('dual_window_safety_net_threshold', 0.25)
config.set('skip_llm_response', True)  # Skip LLM for faster testing

print("\nConfiguration set:")
print(f"  debug: {config.get('debug')}")
print(f"  use_dual_window_detection: {config.get('use_dual_window_detection')}")
print(f"  use_sliding_window_detection: {config.get('use_sliding_window_detection')}")
print(f"  dual_window_high_precision_threshold: {config.get('dual_window_high_precision_threshold')}")
print(f"  dual_window_safety_net_threshold: {config.get('dual_window_safety_net_threshold')}")
print(f"  skip_llm_response: {config.get('skip_llm_response')}")

print("\nRunning test script...")
print("Use: python -m episodic")
print("Then: /script scripts/populate_database.txt")
print("Then: /topics")
print("Then: /exit")