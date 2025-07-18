"""
Theme management command for Episodic CLI.

This module provides the /theme command for listing, previewing,
and selecting color themes.
"""

import sys
import tty
import termios
import typer
from typing import Optional, List, Tuple, Dict
from episodic.config import config
from episodic.configuration import (
    COLOR_SCHEMES, get_text_color, get_system_color, 
    get_heading_color, get_llm_color
)


def theme_command(theme_name: Optional[str] = None):
    """
    Manage color themes.
    
    Usage:
        /theme          - Interactive theme selector with preview
        /theme list     - List all available themes
        /theme <name>   - Set theme directly
    """
    if theme_name is None:
        # Interactive selector
        selector = ThemeSelector()
        selected = selector.run()
        
        if selected:
            apply_theme(selected)
            typer.secho(f"\n✅ Theme changed to: {selected}", fg=get_system_color())
        else:
            typer.secho("\n❌ Theme selection cancelled", fg="yellow")
            
    elif theme_name == "list":
        # List all themes
        list_themes()
        
    else:
        # Direct theme setting
        if theme_name in COLOR_SCHEMES:
            apply_theme(theme_name)
            typer.secho(f"✅ Theme changed to: {theme_name}", fg=get_system_color())
            show_theme_sample(theme_name)
        else:
            typer.secho(f"❌ Unknown theme: {theme_name}", fg="red")
            typer.secho(f"Available themes: {', '.join(COLOR_SCHEMES.keys())}", fg=get_text_color())


def apply_theme(theme_name: str):
    """Apply and persist a theme."""
    # Set in runtime config
    config.set("color_mode", theme_name)
    
    # Persist to disk
    config.config["color_mode"] = theme_name
    config._save()


def list_themes():
    """List all available themes with color samples."""
    typer.secho("\nAvailable themes:", fg=get_heading_color(), bold=True)
    typer.echo()
    
    for theme_name in COLOR_SCHEMES:
        # Temporarily apply theme for preview
        original = config.get("color_mode")
        config.set("color_mode", theme_name)
        
        # Show theme name and sample colors
        typer.secho(f"  {theme_name:<15}", bold=True, nl=False)
        
        # Show color dots
        typer.secho("  ●", fg=get_text_color(), nl=False)
        typer.secho(" ●", fg=get_system_color(), nl=False)
        typer.secho(" ●", fg=get_llm_color(), nl=False)
        typer.secho(" ●", fg=get_heading_color(), nl=False)
        typer.secho(" ●", fg="green", nl=False)
        typer.secho(" ●", fg="yellow", nl=False)
        typer.secho(" ●", fg="red", nl=False)
        
        # Show sample text
        typer.secho("  ", nl=False)
        typer.secho("Text", fg=get_text_color(), nl=False)
        typer.secho(" ", nl=False)
        typer.secho("System", fg=get_system_color(), nl=False)
        typer.secho(" ", nl=False)
        typer.secho("Assistant", fg=get_llm_color(), nl=False)
        typer.secho(" ", nl=False)
        typer.secho("Heading", fg=get_heading_color(), bold=True)
        
        # Restore original theme
        config.set("color_mode", original)
    
    typer.echo()
    typer.secho("Use: /theme <name> to select a theme", fg=get_text_color())
    typer.secho("Use: /theme for interactive selection", fg=get_text_color(), dim=True)


def show_theme_sample(theme_name: str):
    """Show a sample of the selected theme."""
    typer.echo()
    typer.secho("Theme preview:", fg=get_heading_color(), bold=True)
    typer.secho("> Sample prompt", fg="green", bold=True)
    typer.secho("This is how assistant responses will look.", fg=get_llm_color())
    typer.secho("✅ Success message", fg="green")
    typer.secho("⚠️  Warning message", fg="yellow")
    typer.secho("❌ Error message", fg="red")
    typer.secho("System information", fg=get_system_color())
    typer.echo()


class ThemeSelector:
    """Interactive theme selector with side-by-side display."""
    
    def __init__(self):
        self.themes = list(COLOR_SCHEMES.keys())
        self.current_index = 0
        self.original_theme = config.get("color_mode", "dark")
        
    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")
        
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position."""
        print(f"\033[{row};{col}H", end="")
        
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
        # Temporarily apply theme for preview
        config.set("color_mode", theme_name)
        
        col = 40  # Start column for the box
        width = 49  # Box width
        
        # Box structure with actual Episodic UI elements
        preview_lines = self._get_preview_lines(theme_name, width)
        
        # Draw each line
        for i, line in enumerate(preview_lines):
            self.move_cursor(start_row + i, col)
            print(line, end="")
        
        sys.stdout.flush()
    
    def _get_preview_lines(self, theme_name: str, width: int) -> List[str]:
        """Generate preview lines with proper coloring."""
        lines = []
        
        # Helper to pad content to box width
        def pad(text: str, color_func=None) -> str:
            if color_func:
                colored = color_func(text)
                # Account for ANSI codes in padding
                padding = width - 2 - len(text)
                return f"│ {colored}{' ' * padding}│"
            else:
                padding = width - 2 - len(text)
                return f"│ {text}{' ' * padding}│"
        
        # Build preview
        sep_color = lambda t: f"\033[90m{t}\033[0m"  # Gray for separators
        
        lines.append(sep_color("╭" + "─" * (width-2) + "╮"))
        lines.append(f"│{typer.style(f'THEME: {theme_name.upper()}'.center(width-2), fg=get_heading_color(), bold=True)}│")
        lines.append(sep_color("├" + "─" * (width-2) + "┤"))
        lines.append("│" + " " * (width-2) + "│")
        
        # Conversation
        lines.append(pad("> What's the capital of France?", 
                        lambda t: typer.style(t, fg="green", bold=True)))
        lines.append("│" + " " * (width-2) + "│")
        lines.append(pad("The capital of France is Paris. It has",
                        lambda t: typer.style(t, fg=get_llm_color())))
        lines.append(pad("been the capital since 987 AD.",
                        lambda t: typer.style(t, fg=get_llm_color())))
        lines.append("│" + " " * (width-2) + "│")
        lines.append(pad("[Input: 8 • Output: 47 • Cost: $0.0021]",
                        lambda t: typer.style(t, fg=get_text_color(), dim=True)))
        lines.append("│" + " " * (width-2) + "│")
        
        # Muse mode
        lines.append(pad("» Search for latest AI breakthroughs",
                        lambda t: typer.style(t, fg="bright_magenta", bold=True)))
        lines.append(pad("🔍 Searching web...",
                        lambda t: typer.style(t, fg=get_system_color())))
        lines.append(pad("✨ Synthesizing from 8 sources...",
                        lambda t: typer.style(t, fg=get_system_color())))
        lines.append("│" + " " * (width-2) + "│")
        
        # Status messages
        lines.append(pad("System Status:",
                        lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        lines.append(pad("  ✅ Topic saved to: notes.md",
                        lambda t: typer.style(t, fg="green")))
        lines.append(pad("  ⚠️  Rate limit: 45/50 requests",
                        lambda t: typer.style(t, fg="yellow")))
        lines.append(pad("  ❌ Error: Connection failed",
                        lambda t: typer.style(t, fg="red")))
        lines.append(pad("  🎭 Muse mode ENABLED",
                        lambda t: typer.style(t, fg=get_system_color(), bold=True)))
        lines.append(pad("  📌 New topic: python-debugging",
                        lambda t: typer.style(t, fg=get_system_color(), bold=True)))
        lines.append(pad("  💡 Type /help for commands",
                        lambda t: typer.style(t, fg=get_text_color(), dim=True)))
        lines.append("│" + " " * (width-2) + "│")
        
        # Configuration
        lines.append(pad("Current Configuration:",
                        lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        
        # Model line
        model_line = "  Model:    gpt-4o-mini"
        colored_model = f"  {typer.style('Model:', fg=get_text_color())}    {typer.style('gpt-4o-mini', fg=get_llm_color())}"
        padding = width - 2 - len(model_line)
        lines.append(f"│ {colored_model}{' ' * padding}│")
        
        # Cost line
        cost_line = "  Cost:     $0.0234"
        colored_cost = f"  {typer.style('Cost:', fg=get_text_color())}     {typer.style('$0.0234', fg='green')}"
        padding = width - 2 - len(cost_line)
        lines.append(f"│ {colored_cost}{' ' * padding}│")
        
        lines.append("│" + " " * (width-2) + "│")
        
        # Commands
        lines.append(pad("Commands:",
                        lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        
        # Command with argument
        cmd_line = "  /save <name>    Save current topic"
        colored_cmd = f"  {typer.style('/save', fg='bright_yellow', bold=True)} {typer.style('<name>', fg='bright_magenta')}    Save current topic"
        padding = width - 2 - len(cmd_line)
        lines.append(f"│ {colored_cmd}{' ' * padding}│")
        
        lines.append(pad("  /theme gemini",
                        lambda t: typer.style(t, fg="green", italic=True)))
        lines.append("│" + " " * (width-2) + "│")
        
        lines.append(sep_color("├" + "─" * (width-2) + "┤"))
        lines.append(pad("← → Navigate  │  ↵ Select  │  ESC Cancel",
                        lambda t: typer.style(t, fg=get_text_color(), dim=True)))
        lines.append(sep_color("╰" + "─" * (width-2) + "╯"))
        
        return lines
    
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
                    return self.themes[self.current_index]
                elif key == 'escape':
                    break
                    
        finally:
            # Restore terminal and original theme
            print("\033[?25h", end="")     # Show cursor
            print("\033[?1049l", end="")    # Return to main screen
            config.set("color_mode", self.original_theme)  # Restore original
            
        return None