"""
Theme management command for Episodic CLI.

This module provides the /theme command for listing, previewing,
and selecting color themes.
"""

import sys
import tty
import termios
import select
import fcntl
import os
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
        try:
            selector = ThemeSelector()
            selected = selector.run()
            
            if selected:
                apply_theme(selected)
                typer.secho(f"\n✅ Theme changed to: {selected}", fg=get_system_color())
            else:
                typer.secho("\n❌ Theme selection cancelled", fg="yellow")
        except Exception as e:
            typer.secho(f"\n❌ Theme selector error: {e}", fg="red")
            import traceback
            traceback.print_exc()
            
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
    """Interactive theme selector with side-by-side display and scrollable list."""
    
    def __init__(self):
        self.themes = list(COLOR_SCHEMES.keys())
        self.original_theme = config.get("color_mode", "dark")
        # Start at current theme
        try:
            self.current_index = self.themes.index(self.original_theme)
        except ValueError:
            self.current_index = 0
        self.visible_items = 12  # Number of themes visible at once
        self.scroll_offset = 0   # Current scroll position
        # Adjust scroll to show current theme
        if self.current_index >= self.visible_items:
            self.scroll_offset = self.current_index - self.visible_items // 2
        
    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")
        
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position."""
        print(f"\033[{row};{col}H", end="")
        
    def draw_theme_list(self, start_row: int):
        """Draw the scrollable theme list on the left."""
        # Use blue color for visibility on white backgrounds
        blue = "\033[34m"
        reset = "\033[0m"
        
        self.move_cursor(start_row, 2)
        print(f"{blue}Select a theme:{reset}")
        self.move_cursor(start_row + 1, 2)
        print(f"{blue}{'─' * 18}{reset}")
        
        # Calculate visible range
        visible_start = self.scroll_offset
        visible_end = min(self.scroll_offset + self.visible_items, len(self.themes))
        
        # Show top scroll indicator if needed
        list_start = start_row + 3  # Always leave room for scroll indicator
        self.move_cursor(start_row + 2, 2)
        if self.scroll_offset > 0:
            print(f"{blue}  ↑ more ↑     {reset}")
        else:
            print("               ")  # Clear the line if no indicator needed
        
        # Draw visible themes
        visible_count = visible_end - visible_start
        for i, idx in enumerate(range(visible_start, visible_end)):
            theme = self.themes[idx]
            row = list_start + i
            self.move_cursor(row, 2)
            
            if idx == self.current_index:
                # Highlight selected theme (white on blue)
                print(f"\033[1;44;37m▶ {theme:<15}\033[0m   ")  # Bold + blue bg + white fg
            else:
                print(f"{blue}  {theme:<15}{reset}   ")  # Blue text + extra spaces to clear any artifacts
        
        # Clear any remaining lines if we're showing fewer themes
        for i in range(visible_count, self.visible_items):
            row = list_start + i
            self.move_cursor(row, 2)
            print(" " * 20)  # Clear the line
        
        # Show bottom scroll indicator if needed (after the themes, not on them)
        bottom_indicator_row = list_start + self.visible_items
        self.move_cursor(bottom_indicator_row, 2)
        if visible_end < len(self.themes):
            print(f"{blue}  ↓ more ↓     {reset}")  # Extra spaces to clear any leftover text
        else:
            print("               ")  # Clear the line if no indicator needed
        
        # Show position indicator (clear the line first to handle digit changes)
        self.move_cursor(start_row + self.visible_items + 6, 2)
        print(f"{blue}{self.current_index + 1}/{len(self.themes)}    {reset}")
        
        # Instructions at bottom (vertically stacked)
        self.move_cursor(start_row + self.visible_items + 8, 2)
        print(f"{blue}  ↑↓ Navigate {reset}")
        self.move_cursor(start_row + self.visible_items + 9, 2)
        print(f"{blue}  PgUp/Dn Page{reset}")
        self.move_cursor(start_row + self.visible_items + 10, 2)
        print(f"{blue}  Home/End Jump{reset}")
        self.move_cursor(start_row + self.visible_items + 11, 2)
        print(f"{blue}  Enter Select{reset}")
        self.move_cursor(start_row + self.visible_items + 12, 2)
        print(f"{blue}  Esc Cancel  {reset}")
    
    def draw_preview_box(self, start_row: int, theme_name: str):
        """Draw the preview box on the right."""
        # Save current theme
        current_theme = config.get("color_mode", "dark")
        
        # Temporarily apply theme for preview
        config.set("color_mode", theme_name)
        
        col = 25  # Start column for the box (closer now that list is compact)
        width = 50  # Box width (narrower for more compact display)
        
        # Box structure with actual Episodic UI elements
        preview_lines = self._get_preview_lines(theme_name, width)
        
        # Draw each line
        for i, line in enumerate(preview_lines):
            self.move_cursor(start_row + i, col)
            # Clear to end of line first to ensure no artifacts
            print("\033[K", end="")  
            print(line, end="")
        
        sys.stdout.flush()
        
        # Restore original theme
        config.set("color_mode", current_theme)
    
    def _get_preview_lines(self, theme_name: str, width: int) -> List[str]:
        """Generate preview lines with proper coloring."""
        lines = []
        
        # Create a format string for consistent width
        inner_width = width - 2  # Account for the borders
        
        def get_display_width(text: str) -> int:
            """Get the actual display width of text, accounting for emojis."""
            width = 0
            i = 0
            while i < len(text):
                char = text[i]
                code = ord(char)
                
                # Skip variation selectors (zero-width)
                if code == 0xFE0F or (0xFE00 <= code <= 0xFE0F):
                    i += 1
                    continue
                
                # Check if character is emoji
                if (0x1F300 <= code <= 0x1F9FF or  # Misc symbols & pictographs
                    0x1F600 <= code <= 0x1F64F or  # Emoticons
                    0x2600 <= code <= 0x26FF or    # Misc symbols
                    0x2700 <= code <= 0x27BF or    # Dingbats
                    code in [0x2705, 0x26A0, 0x274C, 0x2728, 0x2757]):  # Specific emojis
                    width += 2  # Emoji takes 2 columns
                else:
                    width += 1  # Regular character
                
                i += 1
            return width
        
        def make_line_exact(content: str, plain_text: str) -> str:
            """Make a line exactly width chars wide.
            content: the styled content
            plain_text: the unstyled text for length calculation
            """
            # No borders - just pad to width
            display_width = get_display_width(plain_text)
            padding_needed = width - display_width
            if padding_needed < 0:
                padding_needed = 0
            return f"{content}{' ' * padding_needed}"
        
        def make_box_line(text: str, style_func=None) -> str:
            """Create a box line with exact width."""
            if style_func:
                styled = style_func(text)
                return make_line_exact(styled, text)
            else:
                return make_line_exact(text, text)
        
        # Helper for empty line
        def empty_line() -> str:
            return ' ' * width
        
        # Build preview
        sep_color = lambda t: f"\033[1;90m{t}\033[0m"  # Bold gray for separators
        
        lines.append(sep_color("╭" + "─" * inner_width + "╮"))
        
        # Theme title
        title_text = f'THEME: {theme_name.upper()}'
        centered_text = title_text.center(inner_width - 2)
        title_line = f"{sep_color('│')} {typer.style(centered_text, fg=get_heading_color(), bold=True)} {sep_color('│')}"
        lines.append(title_line)
        
        lines.append(sep_color("└" + "─" * inner_width + "┘"))
        lines.append(empty_line())
        
        # Conversation
        lines.append(make_box_line("> What's the capital of France?", 
                                   lambda t: typer.style(t, fg="green", bold=True)))
        lines.append(empty_line())
        lines.append(make_box_line("The capital of France is Paris. It has",
                                   lambda t: typer.style(t, fg=get_llm_color())))
        lines.append(make_box_line("been the capital since 987 AD.",
                                   lambda t: typer.style(t, fg=get_llm_color())))
        lines.append(empty_line())
        lines.append(make_box_line("[Input: 8 • Output: 47 • Cost: $0.0021]",
                                   lambda t: typer.style(t, fg=get_text_color(), dim=True)))
        lines.append(empty_line())
        
        # Muse mode
        lines.append(make_box_line("» Search for latest AI breakthroughs",
                                   lambda t: typer.style(t, fg="bright_magenta", bold=True)))
        lines.append(make_box_line("🔍 Searching web...",
                                   lambda t: typer.style(t, fg=get_system_color())))
        lines.append(make_box_line("✨ Synthesizing from 8 sources...",
                                   lambda t: typer.style(t, fg=get_system_color())))
        lines.append(empty_line())
        
        # Status messages
        lines.append(make_box_line("System Status:",
                                   lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        lines.append(make_box_line("  ✅ Topic saved to: notes.md",
                                   lambda t: typer.style(t, fg="green")))
        lines.append(make_box_line("  ⚠️  Rate limit: 45/50 requests",
                                   lambda t: typer.style(t, fg="yellow")))
        lines.append(make_box_line("  ❌ Error: Connection failed",
                                   lambda t: typer.style(t, fg="red")))
        lines.append(make_box_line("  🎭 Muse mode ENABLED",
                                   lambda t: typer.style(t, fg=get_system_color(), bold=True)))
        lines.append(make_box_line("  📌 New topic: python-debugging",
                                   lambda t: typer.style(t, fg=get_system_color(), bold=True)))
        lines.append(make_box_line("  💡 Type /help for commands",
                                   lambda t: typer.style(t, fg=get_text_color(), dim=True)))
        lines.append(empty_line())
        
        # Configuration
        lines.append(make_box_line("Current Configuration:",
                                   lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        
        # Model line - special handling for multi-colored line
        model_text = "  Model:    gpt-4o-mini"
        model_label = typer.style('Model:', fg=get_text_color())
        model_value = typer.style('gpt-4o-mini', fg=get_llm_color())
        colored_model = f"  {model_label}    {model_value}"
        lines.append(make_line_exact(colored_model, model_text))
        
        # Cost line - special handling for multi-colored line
        cost_text = "  Cost:     $0.0234"
        cost_label = typer.style('Cost:', fg=get_text_color())
        cost_value = typer.style('$0.0234', fg='green')
        colored_cost = f"  {cost_label}     {cost_value}"
        lines.append(make_line_exact(colored_cost, cost_text))
        
        lines.append(empty_line())
        
        # Commands
        lines.append(make_box_line("Commands:",
                                   lambda t: typer.style(t, fg=get_heading_color(), bold=True)))
        
        # Command with argument - special handling
        cmd_text = "  /save <name>    Save current topic"
        save_cmd = typer.style('/save', fg='bright_yellow', bold=True)
        name_arg = typer.style('<name>', fg='bright_magenta')
        colored_cmd = f"  {save_cmd} {name_arg}    Save current topic"
        lines.append(make_line_exact(colored_cmd, cmd_text))
        
        lines.append(make_box_line("  /theme gemini",
                                   lambda t: typer.style(t, fg="green", italic=True)))
        lines.append(empty_line())
        
        lines.append(sep_color("┌" + "─" * inner_width + "┐"))
        
        # Bottom navigation bar
        nav_text = "↑↓ Navigate  │  ↵ Select  │  ESC Cancel"
        nav_line = f"{sep_color('│')} {typer.style(nav_text.center(inner_width - 2), fg=get_text_color(), dim=True)} {sep_color('│')}"
        lines.append(nav_line)
        
        lines.append(sep_color("└" + "─" * inner_width + "┘"))
        
        return lines
    
    def get_key(self) -> str:
        """Get a single keypress - working version."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            
            # Read first character
            ch1 = sys.stdin.read(1)
            if not ch1:
                return None
            
            # Handle regular keys
            if ch1 == '\r' or ch1 == '\n':
                return 'enter'
            
            # Handle ESC
            if ch1 == '\x1b':
                # Make stdin non-blocking temporarily
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                try:
                    # Try to read the rest of escape sequence
                    rest = sys.stdin.read(10)  # Read up to 10 more chars
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags)  # Restore blocking
                    
                    if len(rest) >= 2 and rest[0] == '[':
                        # It's an escape sequence
                        if rest[1] == 'A': return 'up'
                        elif rest[1] == 'B': return 'down'
                        elif rest[1] == 'C': return None  # right
                        elif rest[1] == 'D': return None  # left
                        elif rest[1] == '5': return 'pageup'
                        elif rest[1] == '6': return 'pagedown'
                        elif rest[1] == 'H': return 'home'
                        elif rest[1] == 'F': return 'end'
                    # If we got here, it's just ESC
                    return 'escape'
                except:
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags)  # Restore blocking
                    return 'escape'
            
            # Ignore everything else
            return None
            
        except Exception:
            return None
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
                # Clear and redraw UI
                self.clear_screen()
                self.draw_theme_list(2)
                self.draw_preview_box(2, self.themes[self.current_index])
                
                # Get input
                key = self.get_key()
                
                # Skip if no valid key
                if key is None:
                    continue
                
                # Process key
                if key == 'up':
                    self.current_index = (self.current_index - 1) % len(self.themes)
                    # Adjust scroll offset if needed
                    if self.current_index < self.scroll_offset:
                        self.scroll_offset = self.current_index
                    elif self.current_index == len(self.themes) - 1:
                        # Wrapped to bottom
                        self.scroll_offset = max(0, len(self.themes) - self.visible_items)
                    continue
                elif key == 'down':
                    self.current_index = (self.current_index + 1) % len(self.themes)
                    # Adjust scroll offset if needed
                    if self.current_index >= self.scroll_offset + self.visible_items:
                        self.scroll_offset = self.current_index - self.visible_items + 1
                    elif self.current_index == 0:
                        # Wrapped to top
                        self.scroll_offset = 0
                    continue
                elif key == 'pageup':
                    # Move up by visible_items
                    self.current_index = max(0, self.current_index - self.visible_items)
                    self.scroll_offset = max(0, self.scroll_offset - self.visible_items)
                    continue
                elif key == 'pagedown':
                    # Move down by visible_items
                    self.current_index = min(len(self.themes) - 1, self.current_index + self.visible_items)
                    if self.current_index >= self.scroll_offset + self.visible_items:
                        self.scroll_offset = min(self.current_index - self.visible_items + 1, 
                                               len(self.themes) - self.visible_items)
                    continue
                elif key == 'home':
                    # Go to first theme
                    self.current_index = 0
                    self.scroll_offset = 0
                    continue
                elif key == 'end':
                    # Go to last theme
                    self.current_index = len(self.themes) - 1
                    self.scroll_offset = max(0, len(self.themes) - self.visible_items)
                    continue
                elif key == 'enter':
                    return self.themes[self.current_index]
                elif key == 'escape':
                    break
                else:
                    # Any other key - ignore and continue
                    continue
                    
        except KeyboardInterrupt:
            # Handle Ctrl-C gracefully
            pass
        finally:
            # Restore terminal and original theme
            print("\033[?25h", end="")     # Show cursor
            print("\033[?1049l", end="")    # Return to main screen
            config.set("color_mode", self.original_theme)  # Restore original
            
        return None
