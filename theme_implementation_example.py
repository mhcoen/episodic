#!/usr/bin/env python3
"""
Theme selector implementation with side-by-side display.
This is a working example that can be integrated into Episodic.
"""

import sys
import tty
import termios
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import shutil

# ANSI color codes
@dataclass
class ThemeColors:
    """Theme color definitions using ANSI codes."""
    heading: str
    prompt: str
    prompt_muse: str
    assistant: str
    assistant_heading: str
    system: str
    system_emphasis: str
    success: str
    warning: str
    error: str
    info: str
    label: str
    value: str
    price: str
    accent: str
    command: str
    argument: str
    dim: str
    separator: str

# Define themes
THEMES = {
    "dark": ThemeColors(
        heading="93;1",          # Bright yellow bold
        prompt="92;1",           # Bright green bold
        prompt_muse="95;1",      # Bright magenta bold
        assistant="96",          # Bright cyan
        assistant_heading="96;1", # Bright cyan bold
        system="94",             # Bright blue
        system_emphasis="94;1",   # Bright blue bold
        success="92",            # Bright green
        warning="93",            # Bright yellow
        error="91",              # Bright red
        info="96",               # Bright cyan
        label="37",              # White
        value="97",              # Bright white
        price="92",              # Bright green
        accent="95",             # Bright magenta
        command="93;1",          # Bright yellow bold
        argument="95",           # Bright magenta
        dim="90",                # Bright black (gray)
        separator="90"           # Bright black (gray)
    ),
    "light": ThemeColors(
        heading="34;1",          # Blue bold
        prompt="32;1",           # Green bold
        prompt_muse="35;1",      # Magenta bold
        assistant="34",          # Blue
        assistant_heading="34;1", # Blue bold
        system="35",             # Magenta
        system_emphasis="35;1",   # Magenta bold
        success="32",            # Green
        warning="33",            # Yellow
        error="31",              # Red
        info="36",               # Cyan
        label="30",              # Black
        value="30;1",            # Black bold
        price="32",              # Green
        accent="35",             # Magenta
        command="34;1",          # Blue bold
        argument="35",           # Magenta
        dim="37",                # Light gray
        separator="37"           # Light gray
    ),
    "gemini": ThemeColors(
        heading="96;1",          # Bright cyan bold
        prompt="92;1",           # Bright green bold
        prompt_muse="95;1",      # Bright magenta bold
        assistant="94",          # Bright blue
        assistant_heading="94;1", # Bright blue bold
        system="96",             # Bright cyan
        system_emphasis="96;1",   # Bright cyan bold
        success="92",            # Bright green
        warning="93",            # Yellow
        error="91;1",            # Bright red bold
        info="96",               # Bright cyan
        label="37",              # White
        value="97",              # Bright white
        price="92;1",            # Bright green bold
        accent="93",             # Bright yellow
        command="93;1",          # Bright yellow bold
        argument="95",           # Bright magenta
        dim="90",                # Gray
        separator="36"           # Cyan
    ),
    "solarized_dark": ThemeColors(
        heading="33;1",          # Yellow bold
        prompt="32;1",           # Green bold
        prompt_muse="35;1",      # Magenta bold
        assistant="36",          # Cyan
        assistant_heading="36;1", # Cyan bold
        system="34",             # Blue
        system_emphasis="34;1",   # Blue bold
        success="32",            # Green
        warning="33",            # Yellow
        error="31",              # Red
        info="36",               # Cyan
        label="37",              # Base0
        value="97",              # Base1
        price="32",              # Green
        accent="33",             # Yellow
        command="33;1",          # Yellow bold
        argument="35",           # Magenta
        dim="90",                # Base01
        separator="90"           # Base01
    ),
    "dracula": ThemeColors(
        heading="95;1",          # Pink bold
        prompt="92;1",           # Green bold
        prompt_muse="95;1",      # Pink bold
        assistant="97",          # Foreground
        assistant_heading="97;1", # Foreground bold
        system="96",             # Cyan
        system_emphasis="96;1",   # Cyan bold
        success="92",            # Green
        warning="93",            # Yellow
        error="91",              # Red
        info="96",               # Cyan
        label="37",              # Comment
        value="97",              # Foreground
        price="92",              # Green
        accent="95",             # Pink
        command="93;1",          # Yellow bold
        argument="95",           # Pink
        dim="90",                # Comment
        separator="35"           # Purple
    ),
    "monokai": ThemeColors(
        heading="93;1",          # Yellow bold
        prompt="92;1",           # Green bold
        prompt_muse="95;1",      # Magenta bold
        assistant="97",          # White
        assistant_heading="97;1", # White bold
        system="96",             # Cyan
        system_emphasis="96;1",   # Cyan bold
        success="92",            # Green
        warning="93",            # Yellow
        error="91",              # Red
        info="96",               # Cyan
        label="90",              # Gray
        value="97",              # White
        price="92",              # Green
        accent="95",             # Magenta
        command="93;1",          # Yellow bold
        argument="91",           # Red
        dim="90",                # Gray
        separator="90"           # Gray
    )
}

class ThemeSelector:
    """Interactive theme selector with side-by-side display."""
    
    def __init__(self):
        self.themes = list(THEMES.keys())
        self.current_index = 0
        self.selected_theme = None
        
    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")
        
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position."""
        print(f"\033[{row};{col}H", end="")
        
    def color_text(self, text: str, theme_name: str, category: str) -> str:
        """Apply theme color to text."""
        theme = THEMES.get(theme_name, THEMES["dark"])
        color_code = getattr(theme, category, "37")
        return f"\033[{color_code}m{text}\033[0m"
    
    def draw_theme_list(self, start_row: int):
        """Draw the theme list on the left."""
        self.move_cursor(start_row, 2)
        print("Theme Selection")
        
        for i, theme in enumerate(self.themes):
            self.move_cursor(start_row + 2 + i, 2)
            if i == self.current_index:
                print(f"▶ {theme:<15} ◀── selected")
            else:
                print(f"  {theme:<15}")
        
        # Instructions at bottom
        self.move_cursor(start_row + len(self.themes) + 4, 2)
        print("[↑↓] Select theme   [Enter] Apply   [Esc] Cancel")
    
    def draw_preview_box(self, start_row: int, theme_name: str):
        """Draw the preview box on the right."""
        col = 40  # Start column for the box
        width = 49  # Box width
        
        # Draw box with preview
        lines = [
            ("╭" + "─" * (width-2) + "╮", "separator"),
            ("│" + f"THEME: {theme_name.upper()}".center(width-2) + "│", "heading"),
            ("├" + "─" * (width-2) + "┤", "separator"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  > What's the capital of France?" + " " * 14 + "│", "prompt"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  The capital of France is Paris. It has" + " " * 6 + "│", "assistant"),
            ("│  been the capital since 987 AD." + " " * 15 + "│", "assistant"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  [Input: 8 • Output: 47 • Cost: $0.0021]" + " " * 5 + "│", "dim"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  » Search for latest AI breakthroughs" + " " * 8 + "│", "prompt_muse"),
            ("│  🔍 Searching web..." + " " * 26 + "│", "system"),
            ("│  ✨ Synthesizing from 8 sources..." + " " * 11 + "│", "system"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  System Status:" + " " * 31 + "│", "heading"),
            ("│    ✅ Topic saved to: notes.md" + " " * 15 + "│", "success"),
            ("│    ⚠️  Rate limit: 45/50 requests" + " " * 12 + "│", "warning"),
            ("│    ❌ Error: Connection failed" + " " * 16 + "│", "error"),
            ("│    🎭 Muse mode ENABLED" + " " * 23 + "│", "system_emphasis"),
            ("│    📌 New topic: python-debugging" + " " * 12 + "│", "system_emphasis"),
            ("│    💡 Type /help for commands" + " " * 17 + "│", "dim"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  Current Configuration:" + " " * 23 + "│", "heading"),
            ("│    Model:    gpt-4o-mini" + " " * 21 + "│", "label"),
            ("│    Cost:     $0.0234" + " " * 25 + "│", "price"),
            ("│" + " " * (width-2) + "│", ""),
            ("│  Commands:" + " " * 36 + "│", "heading"),
            ("│    /save <name>    Save current topic" + " " * 8 + "│", "command"),
            ("│    /theme gemini" + " " * 30 + "│", "dim"),
            ("│" + " " * (width-2) + "│", ""),
            ("├" + "─" * (width-2) + "┤", "separator"),
            ("│" + " ← → Navigate  │  ↵ Select  │  ESC Cancel ".center(width-2) + "│", "dim"),
            ("╰" + "─" * (width-2) + "╯", "separator"),
        ]
        
        # Draw each line with colors
        for i, (line_content, category) in enumerate(lines):
            self.move_cursor(start_row + i, col)
            
            if category and category != "":
                # Parse the line to apply colors to the content inside the box
                if "│" in line_content and not line_content.startswith("│ ←"):
                    # Extract content between box characters
                    parts = line_content.split("│")
                    if len(parts) >= 3:
                        print("│", end="")
                        content = parts[1]
                        # Apply color to the actual content
                        colored_content = self.color_text(content.strip(), theme_name, category)
                        # Repad to maintain box alignment
                        padding = width - 2 - len(content.strip())
                        print(colored_content + " " * padding, end="")
                        print("│", end="")
                    else:
                        print(line_content, end="")
                else:
                    # For borders and special lines
                    if category == "separator":
                        print(self.color_text(line_content, theme_name, category), end="")
                    else:
                        print(line_content, end="")
            else:
                print(line_content, end="")
        
        sys.stdout.flush()
    
    def get_key(self) -> str:
        """Get a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            
            # Handle special keys
            if key == '\033':  # ESC sequence
                next_chars = sys.stdin.read(2)
                if next_chars == '[A':  # Up arrow
                    return 'up'
                elif next_chars == '[B':  # Down arrow
                    return 'down'
                elif next_chars == '[C':  # Right arrow
                    return 'right'
                elif next_chars == '[D':  # Left arrow
                    return 'left'
                else:
                    return 'escape'
            elif key == '\r' or key == '\n':
                return 'enter'
            
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def run(self) -> Optional[str]:
        """Run the theme selector and return selected theme."""
        # Save cursor and use alternate screen buffer
        print("\033[?1049h", end="")  # Switch to alternate screen
        print("\033[?25l", end="")     # Hide cursor
        
        try:
            self.clear_screen()
            
            while True:
                # Draw UI
                self.draw_theme_list(2)
                self.draw_preview_box(2, self.themes[self.current_index])
                
                # Get input
                key = self.get_key()
                
                if key == 'up':
                    self.current_index = (self.current_index - 1) % len(self.themes)
                elif key == 'down':
                    self.current_index = (self.current_index + 1) % len(self.themes)
                elif key == 'enter':
                    self.selected_theme = self.themes[self.current_index]
                    break
                elif key == 'escape':
                    break
                    
        finally:
            # Restore terminal
            print("\033[?25h", end="")     # Show cursor
            print("\033[?1049l", end="")    # Return to main screen
            
        return self.selected_theme


def demo_theme_selector():
    """Demo the theme selector."""
    selector = ThemeSelector()
    selected = selector.run()
    
    if selected:
        print(f"\n✅ Selected theme: {selected}")
        # Here you would save to config:
        # config.set("color_mode", selected)
        # config._save()
    else:
        print("\n❌ Theme selection cancelled")


if __name__ == "__main__":
    demo_theme_selector()