"""
Test script for the native visualization command-line interface.

This script tests the native visualization functionality by calling the
episodic visualize command with the --native flag.
"""

import subprocess
import sys
import time

def test_native_cli():
    """Test the native visualization command-line interface."""
    print("=== NATIVE VISUALIZATION CLI TEST ===")
    print("This test will open a native window with the visualization.")
    print("The window will remain open until you close it.")
    print("Press Ctrl+C to stop the test if the window doesn't close properly.")
    
    try:
        # Call the episodic visualize command with the --native flag
        cmd = [sys.executable, "-m", "episodic", "visualize", "--native", "--width", "1200", "--height", "900", "--port", "5001"]
        print(f"Running command: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        process.wait()
        
        print("Test completed successfully.")
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        # Try to terminate the process if it's still running
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            print("Process terminated.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_native_cli()