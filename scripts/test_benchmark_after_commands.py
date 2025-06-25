#!/usr/bin/env python
"""Test script to verify benchmark display after commands."""

import subprocess
import time

def test_benchmark_display():
    """Test that benchmarks display after commands when benchmark_display is enabled."""
    print("Testing benchmark display after commands...")
    
    # Create test input
    test_commands = [
        "/set benchmark true",
        "/set benchmark_display true", 
        "/summary 5",
        "/exit"
    ]
    
    # Run the CLI with test commands
    proc = subprocess.Popen(
        ["python", "-m", "episodic"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send commands
    output, error = proc.communicate(input='\n'.join(test_commands))
    
    print("=== OUTPUT ===")
    print(output)
    
    if error:
        print("=== ERROR ===")
        print(error)
    
    # Check if benchmark was displayed after summary command
    if "[Benchmark] Summary Generation:" in output and "/summary" in output:
        print("\n✅ SUCCESS: Benchmark displayed after /summary command")
    else:
        print("\n❌ FAILED: Benchmark not displayed after /summary command")
        
if __name__ == "__main__":
    test_benchmark_display()