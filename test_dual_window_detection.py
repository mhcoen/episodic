#!/usr/bin/env python3
"""
Test dual-window detection with the populate_database script.
"""

import subprocess
import time
import sys

def main():
    print("Testing Dual-Window Detection")
    print("=" * 60)
    
    # Enable debug mode
    print("\n1. Setting debug mode...")
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'debug', 'true'], check=True)
    
    # Enable dual-window detection
    print("\n2. Enabling dual-window detection...")
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'use_dual_window_detection', 'true'], check=True)
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'use_sliding_window_detection', 'false'], check=True)
    
    # Set threshold
    print("\n3. Setting detection thresholds...")
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'dual_window_high_precision_threshold', '0.2'], check=True)
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'dual_window_safety_net_threshold', '0.25'], check=True)
    
    # Skip LLM responses for faster testing
    print("\n4. Enabling skip_llm_response for faster testing...")
    subprocess.run(['python', '-m', 'episodic', 'config', 'set', 'skip_llm_response', 'true'], check=True)
    
    # Show current config
    print("\n5. Current topic detection configuration:")
    subprocess.run(['python', '-m', 'episodic', 'config', 'get'])
    
    print("\n6. Running script with dual-window detection...")
    print("-" * 60)
    
    # Run the script
    result = subprocess.run(
        ['python', '-m', 'episodic', 'script', 'scripts/populate_database.txt'],
        capture_output=True,
        text=True
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\n7. Listing detected topics...")
    subprocess.run(['python', '-m', 'episodic', 'topics'])
    
    print("\n8. Showing conversation tree...")
    subprocess.run(['python', '-m', 'episodic', 'tree'])

if __name__ == "__main__":
    main()