#!/usr/bin/env python3
"""Test real header formatting that should be bold."""

from episodic.color_utils import secho_color

def test_header_patterns():
    """Test the exact patterns that should be bold in streaming."""
    
    print("Testing header patterns that should be bold:")
    print("=" * 50)
    
    # Test markdown headers (should be bold)
    print("1. Markdown header:")
    secho_color("### Mystery/Thriller:", bold=True, nl=False)
    print(" <- This should be BOLD")
    
    # Test colon headers (should be bold until colon)
    print("\n2. Colon header:")
    secho_color("Mystery/Thriller:", bold=True, nl=False)
    print(" <- This should be BOLD")
    
    # Test numbered lists (should be bold)
    print("\n3. Numbered list:")
    secho_color("1. Advanced", bold=True, nl=False)
    print(" <- This should be BOLD")
    
    # Test bullet lists (should be bold)
    print("\n4. Bullet list:")
    secho_color("- The Silent Patient", bold=True, nl=False)
    print(" <- This should be BOLD")
    
    # Test italic markers as bold
    print("\n5. Italic markers as bold:")
    secho_color("The Silent Patient", bold=True, nl=False)
    print(" <- This should be BOLD")
    
    print("\n" + "=" * 50)
    print("All items above should appear in BOLD text.")

if __name__ == "__main__":
    test_header_patterns()