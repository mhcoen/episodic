#!/usr/bin/env python3
"""Check word wrapping configuration."""

import shutil
from episodic.conversation import ConversationManager
from episodic.config import config

# Check terminal and wrap settings
terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
print(f"Terminal width: {terminal_width}")

cm = ConversationManager()
wrap_width = cm.get_wrap_width()
print(f"Calculated wrap width: {wrap_width}")
print(f"Text wrap enabled: {config.get('text_wrap', True)}")

# Test a long line
test_line = "This is a very long line that should definitely wrap at some point because it's much longer than any reasonable terminal width could possibly be."
print(f"\nTest line length: {len(test_line)}")
print(f"Should wrap: {len(test_line) > wrap_width}")