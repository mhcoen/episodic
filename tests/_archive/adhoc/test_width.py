#!/usr/bin/env python3
import shutil
import os

# Test terminal width detection
size = shutil.get_terminal_size(fallback=(80, 24))
print(f"Terminal size: {size.columns}x{size.lines}")
print(f"COLUMNS env: {os.environ.get('COLUMNS', 'not set')}")
print(f"LINES env: {os.environ.get('LINES', 'not set')}")

# Test wrapping
wrap_width = min(size.columns - 4, 100)
print(f"Calculated wrap width: {wrap_width}")

# Test line
test_line = "✨ If you're looking to catch a movie in theaters right now, there are several popular"
print(f"\nTest line length: {len(test_line)}")
print(f"Line: {test_line}")

# Check display width vs string length
import unicodedata
def display_width(text):
    """Calculate display width accounting for wide characters"""
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width

print(f"Display width: {display_width(test_line)}")
print("^" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|")
print("0         10        20        30        40        50        60        70        80")