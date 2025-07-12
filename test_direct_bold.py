#!/usr/bin/env python3
"""Direct test of secho_color function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the function directly
from episodic.color_utils import secho_color

def test_direct_bold():
    """Test secho_color function directly."""
    print("Testing secho_color function:")
    print("=" * 40)
    
    # Test bold text
    print("1. Regular text:", end=" ")
    secho_color("This is normal text", bold=False, nl=False)
    print()
    
    print("2. Bold text:", end=" ")
    secho_color("This is BOLD text", bold=True, nl=False)
    print()
    
    print("3. Bold red text:", end=" ")
    secho_color("This is BOLD RED text", fg="red", bold=True, nl=False)
    print()
    
    print("4. Bold blue text:", end=" ")
    secho_color("This is BOLD BLUE text", fg="blue", bold=True, nl=False)
    print()
    
    print("=" * 40)
    print("If any text above appears bold, then secho_color works!")

if __name__ == "__main__":
    test_direct_bold()