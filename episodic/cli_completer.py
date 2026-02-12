"""
Tab completion support for Episodic CLI.

This module provides context-aware tab completion using prompt_toolkit.
Command-specific completers are in cli_completer_commands.py.
"""

from typing import List
import os

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from episodic.commands.registry import command_registry, register_all_commands
from episodic.config import config
from episodic.cli_completer_commands import (
    complete_model_command,
    complete_web_command,
    complete_set_command,
    complete_subcommand,
    complete_mode_command,
    complete_file_path,
    complete_script_command,
    complete_save_command,
    complete_style_command,
    complete_format_command,
    complete_detail_command,
    complete_theme_command,
    complete_load_command,
    complete_summary_command,
    complete_debug_command,
    complete_memory_command,
    complete_forget_command,
    complete_help_command,
    complete_docs_command,
    complete_prompt_command,
    complete_reset_command,
    complete_dev_command,
    complete_migrate_command,
    complete_cancel_command,
    complete_dnd_command,
    complete_news_command,
    complete_play_command,
    complete_timer_command,
    complete_alarm_command,
)


class EpisodicCompleter(Completer):
    """Context-aware completer for Episodic commands."""

    def __init__(self):
        # Ensure commands are registered
        register_all_commands()

        # Build command list (main commands only, not aliases)
        self.commands = set()
        self.command_aliases = {}  # alias -> full command

        # Add all commands from the registry
        for cmd_name, cmd_info in command_registry._commands.items():
            # Only add if this is the main command, not an alias
            if cmd_name == cmd_info.name:
                self.commands.add(cmd_name)
            else:
                # This entry is an alias pointing to the command
                self.command_aliases[cmd_name] = cmd_info.name

        # Add utility commands (handled separately via handle_utility_command)
        self.utility_commands = {
            'time': 'Show current time',
            'timer': 'Set a timer or show active timers',
            'alarm': 'Set an alarm or list alarms',
            'remind': 'Set a reminder',
            'weather': 'Get current weather',
            'forecast': 'Get weather forecast',
            'news': 'Get news headlines',
            'calc': 'Calculate expression',
            'note': 'Add or list notes',
            'play': 'Play radio station',
            'cancel': 'Cancel timer or alarm',
            'undo': 'Undo last utility action',
            'dnd': 'Do not disturb mode',
            'status': 'Show system status',
            'stop': 'Stop current action',
        }

        # Add plugin slash commands
        try:
            from episodic.mcp.plugins import get_plugin_registry
            registry = get_plugin_registry()
            if not registry.initialized:
                registry.register_all()
            for reg in registry.registered():
                for sc in reg.slash_commands:
                    name = sc.name.lstrip('/')
                    self.utility_commands[name] = sc.description or f"{sc.category} command"
                    for alias in sc.aliases:
                        alias_name = alias.lstrip('/')
                        self.utility_commands[alias_name] = sc.description or f"{sc.category} command"
        except ImportError:
            pass

        self.commands.update(self.utility_commands.keys())

        # Cache commonly used completions
        self._model_cache = None
        self._model_cache_time = 0

    def get_completions(self, document: Document, complete_event) -> List[Completion]:
        """Get completions based on current context."""
        line = document.current_line_before_cursor

        # Check for @file reference completion (works anywhere in input)
        if '@' in line:
            yield from self._complete_at_file_reference(line, document)
            # Don't return - also check for slash commands if line starts with /

        # Only complete slash commands
        if not line.startswith('/'):
            return

        # Parse the command line
        parts = line.split()
        word_before_cursor = document.get_word_before_cursor()

        # If line ends with space and no current word, we're starting a new argument
        if line.endswith(' ') and not word_before_cursor:
            parts.append('')

        # Command completion
        if len(parts) <= 1:
            # Check if we're in simple mode
            from episodic.commands.interface_mode import is_simple_mode, get_simple_mode_commands

            # Complete command names
            partial = line[1:]  # Remove the /
            for cmd in sorted(self.commands):
                if cmd.startswith(partial.lower()):
                    # In simple mode, only show allowed commands
                    if is_simple_mode() and cmd not in get_simple_mode_commands():
                        continue

                    # Calculate start position
                    start_pos = -len(line)
                    yield Completion(
                        '/' + cmd,
                        start_position=start_pos,
                        display=cmd,
                        display_meta=self._get_command_meta(cmd)
                    )
        else:
            # Context-specific completion
            cmd = parts[0][1:]  # Remove the /

            # Resolve aliases to full command
            full_cmd = self.command_aliases.get(cmd, cmd)

            # Route to appropriate completer
            if full_cmd in ['model']:
                yield from complete_model_command(parts, word_before_cursor)
            elif full_cmd == 'web':
                yield from complete_web_command(parts, word_before_cursor)
            elif full_cmd in ['set', 'mset']:
                yield from complete_set_command(parts, word_before_cursor, self._get_param_meta)
            elif full_cmd in ['topics', 'compression', 'voice', 'rag', 'kg']:
                yield from complete_subcommand(full_cmd, parts, word_before_cursor)
            elif full_cmd == 'mode':
                yield from complete_mode_command(parts, word_before_cursor)
            elif full_cmd == 'index':
                yield from complete_file_path(full_cmd, parts, word_before_cursor)
            elif full_cmd == 'script':
                yield from complete_script_command(parts, word_before_cursor)
            elif full_cmd == 'save':
                yield from complete_save_command(parts, word_before_cursor)
            elif full_cmd == 'style':
                yield from complete_style_command(parts, word_before_cursor)
            elif full_cmd == 'format':
                yield from complete_format_command(parts, word_before_cursor)
            elif full_cmd == 'detail':
                yield from complete_detail_command(parts, word_before_cursor)
            elif full_cmd == 'theme':
                yield from complete_theme_command(parts, word_before_cursor)
            elif full_cmd == 'load':
                yield from complete_load_command(parts, word_before_cursor)
            elif full_cmd == 'summary':
                yield from complete_summary_command(parts, word_before_cursor)
            elif full_cmd == 'debug':
                yield from complete_debug_command(parts, word_before_cursor)
            elif full_cmd == 'memory':
                yield from complete_memory_command(parts, word_before_cursor)
            elif full_cmd == 'forget':
                yield from complete_forget_command(parts, word_before_cursor)
            elif full_cmd == 'help':
                yield from complete_help_command(parts, word_before_cursor)
            elif full_cmd == 'docs':
                yield from complete_docs_command(parts, word_before_cursor)
            elif full_cmd == 'prompt':
                yield from complete_prompt_command(parts, word_before_cursor)
            elif full_cmd == 'reset':
                yield from complete_reset_command(parts, word_before_cursor)
            elif full_cmd == 'dev':
                yield from complete_dev_command(parts, word_before_cursor)
            elif full_cmd == 'migrate':
                yield from complete_migrate_command(parts, word_before_cursor)
            # Utility command completions
            elif full_cmd == 'cancel':
                yield from complete_cancel_command(parts, word_before_cursor)
            elif full_cmd == 'dnd':
                yield from complete_dnd_command(parts, word_before_cursor)
            elif full_cmd == 'news':
                yield from complete_news_command(parts, word_before_cursor)
            elif full_cmd == 'play':
                yield from complete_play_command(parts, word_before_cursor)
            elif full_cmd == 'timer':
                yield from complete_timer_command(parts, word_before_cursor)
            elif full_cmd == 'alarm':
                yield from complete_alarm_command(parts, word_before_cursor)
            else:
                # Check plugin slash commands for completions
                yield from self._complete_plugin_command(full_cmd, parts, word_before_cursor)

    def _complete_plugin_command(self, cmd: str, parts: list, word: str):
        """Complete a plugin slash command using its defined completions."""
        try:
            from episodic.mcp.plugins import get_plugin_registry
            registry = get_plugin_registry()
            sc = registry.get_slash_command(f"/{cmd}")
            if sc and sc.completions and len(parts) == 2:
                for comp in sc.completions:
                    if comp.startswith(word.lower()):
                        yield Completion(
                            comp, start_position=-len(word),
                            display_meta=sc.category or "",
                        )
        except ImportError:
            pass

    def _get_command_meta(self, cmd: str) -> str:
        """Get command description for display."""
        # Check utility commands first
        if cmd in self.utility_commands:
            return self.utility_commands[cmd]

        # Check regular registry
        if cmd in command_registry._commands:
            return command_registry._commands[cmd].description

        # Check if it's an alias
        if cmd in self.command_aliases:
            full_cmd = self.command_aliases[cmd]
            if full_cmd in command_registry._commands:
                return command_registry._commands[full_cmd].description

        return ''

    def _get_param_meta(self, param: str, param_type: str) -> str:
        """Get parameter metadata with current value if set."""
        # Convert display param (with dashes) to config key (with underscores)
        config_key = param.replace('-', '_')

        try:
            value = config.get(config_key)
            if value is not None:
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 20:
                    value_str = value_str[:17] + '...'
                return f"={value_str}"
            else:
                return param_type
        except Exception:
            return param_type

    def _complete_at_file_reference(self, line: str, document: Document) -> List[Completion]:
        """
        Complete @file references in chat input.

        Triggers when user types @path and hits Tab.
        Example: "Explain @src/" -> completes with files in src/
        """
        import re

        # Find the last @ reference being typed
        # Match @path or @"path (for quoted paths)
        # Look for @ followed by optional " and then path characters
        match = re.search(r'@("?)([^\s@]*)$', line)
        if not match:
            return

        quote = match.group(1)  # Empty or "
        partial_path = match.group(2)
        match_start = match.start()

        # Calculate how much to replace
        replace_len = len(line) - match_start

        # Expand user home directory
        expanded_path = partial_path
        if expanded_path.startswith('~'):
            expanded_path = os.path.expanduser(expanded_path)

        # Handle :vision suffix - strip it for path completion
        vision_suffix = ''
        if ':vision' in expanded_path:
            idx = expanded_path.find(':vision')
            vision_suffix = expanded_path[idx:]
            expanded_path = expanded_path[:idx]
            partial_path = partial_path[:partial_path.find(':vision')]

        # Get directory and partial filename
        if expanded_path:
            if os.path.isdir(expanded_path):
                search_dir = expanded_path
                partial_name = ''
            else:
                search_dir = os.path.dirname(expanded_path) or '.'
                partial_name = os.path.basename(expanded_path)
        else:
            search_dir = '.'
            partial_name = ''

        # Get completions
        try:
            if os.path.isdir(search_dir):
                entries = sorted(os.listdir(search_dir))

                for entry in entries:
                    # Skip hidden files unless user started typing with .
                    if entry.startswith('.') and not partial_name.startswith('.'):
                        continue

                    if entry.lower().startswith(partial_name.lower()):
                        full_path = os.path.join(search_dir, entry)
                        is_dir = os.path.isdir(full_path)

                        # Build completion path
                        if partial_path:
                            dir_part = os.path.dirname(partial_path)
                            if dir_part:
                                completion_path = os.path.join(dir_part, entry)
                            else:
                                completion_path = entry
                        else:
                            completion_path = entry

                        # Add trailing slash for directories
                        if is_dir:
                            completion_path += '/'
                            display = entry + '/'
                            meta = 'directory'
                        else:
                            display = entry
                            # Show file type hint
                            ext = os.path.splitext(entry)[1].lower()
                            if ext in ['.pdf']:
                                meta = 'PDF (text or :vision)'
                            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                                meta = 'image (multimodal)'
                            elif ext in ['.py', '.js', '.ts', '.java', '.go', '.rs']:
                                meta = 'code'
                            elif ext in ['.md', '.txt', '.json', '.yaml', '.yml']:
                                meta = 'text'
                            else:
                                meta = 'file'

                        # Build full completion with @ prefix
                        if quote:
                            # Quoted path
                            completion_text = f'@"{completion_path}"'
                        else:
                            # Check if path needs quoting (has spaces)
                            if ' ' in completion_path:
                                completion_text = f'@"{completion_path}"'
                            else:
                                completion_text = f'@{completion_path}'

                        yield Completion(
                            completion_text,
                            start_position=-replace_len,
                            display=display,
                            display_meta=meta
                        )
        except (OSError, PermissionError):
            # Can't read directory
            pass
