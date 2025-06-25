#!/usr/bin/env python3
"""Test word wrapping functionality."""

import shutil
from episodic.conversation import ConversationManager
from episodic.config import config

# Enable text wrapping
config.set("text_wrap", True)

cm = ConversationManager()
wrap_width = cm.get_wrap_width()
terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns

print(f"Terminal width: {terminal_width}")
print(f"Wrap width: {wrap_width}")
print(f"Text wrap enabled: {config.get('text_wrap', True)}")

# Test with a long line
test_text = "This is a very long line that should definitely wrap because it's much longer than any reasonable terminal width. It goes on and on and on and on and on and on and on and on and on and on."
print(f"\nTest text length: {len(test_text)}")

# Test wrapping
print("\nTesting wrapped_text_print:")
cm.wrapped_text_print(test_text)