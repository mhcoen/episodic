"""
Tab completion support for Episodic CLI.

This module provides context-aware tab completion using prompt_toolkit.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from episodic.commands.registry import command_registry, register_all_commands
from episodic.config import config
from episodic.llm_config import get_available_providers, get_provider_models
from episodic.db_topics import get_recent_topics
from episodic.constants import (
    WEB_SEARCH_PROVIDERS, RESPONSE_STYLES, RESPONSE_FORMATS, DETAIL_LEVELS,
    TOPIC_ACTIONS, COMPRESSION_ACTIONS, VOICE_ACTIONS, RAG_ACTIONS, SUMMARY_LENGTHS,
    COLOR_MODES, MODEL_CONTEXTS, DOCS_ACTIONS, PROMPT_ACTIONS, RESET_ACTIONS,
    DEV_ACTIONS, MIGRATE_ACTIONS
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
        
        # Cache commonly used completions
        self._model_cache = None
        self._model_cache_time = 0
        
    def get_completions(self, document: Document, complete_event) -> List[Completion]:
        """Get completions based on current context."""
        line = document.current_line_before_cursor
        
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
                yield from self._complete_model_command(parts, word_before_cursor)
            elif full_cmd == 'web':
                yield from self._complete_web_command(parts, word_before_cursor)
            elif full_cmd in ['set', 'mset']:
                yield from self._complete_set_command(parts, word_before_cursor)
            elif full_cmd in ['topics', 'compression', 'voice', 'rag']:
                yield from self._complete_subcommand(full_cmd, parts, word_before_cursor)
            elif full_cmd == 'mode':
                yield from self._complete_mode_command(parts, word_before_cursor)
            elif full_cmd == 'index':
                yield from self._complete_file_path(full_cmd, parts, word_before_cursor)
            elif full_cmd == 'script':
                yield from self._complete_script_command(parts, word_before_cursor)
            elif full_cmd == 'save':
                yield from self._complete_save_command(parts, word_before_cursor)
            elif full_cmd == 'style':
                yield from self._complete_style_command(parts, word_before_cursor)
            elif full_cmd == 'format':
                yield from self._complete_format_command(parts, word_before_cursor)
            elif full_cmd == 'detail':
                yield from self._complete_detail_command(parts, word_before_cursor)
            elif full_cmd == 'theme':
                yield from self._complete_theme_command(parts, word_before_cursor)
            elif full_cmd == 'load':
                yield from self._complete_load_command(parts, word_before_cursor)
            elif full_cmd == 'summary':
                yield from self._complete_summary_command(parts, word_before_cursor)
            elif full_cmd == 'debug':
                yield from self._complete_debug_command(parts, word_before_cursor)
            elif full_cmd == 'memory':
                yield from self._complete_memory_command(parts, word_before_cursor)
            elif full_cmd == 'forget':
                yield from self._complete_forget_command(parts, word_before_cursor)
            elif full_cmd == 'help':
                yield from self._complete_help_command(parts, word_before_cursor)
            elif full_cmd == 'docs':
                yield from self._complete_docs_command(parts, word_before_cursor)
            elif full_cmd == 'prompt':
                yield from self._complete_prompt_command(parts, word_before_cursor)
            elif full_cmd == 'reset':
                yield from self._complete_reset_command(parts, word_before_cursor)
            elif full_cmd == 'dev':
                yield from self._complete_dev_command(parts, word_before_cursor)
            elif full_cmd == 'migrate':
                yield from self._complete_migrate_command(parts, word_before_cursor)
    
    def _get_command_meta(self, cmd: str) -> str:
        """Get command description for display."""
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

    def _complete_model_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /model command."""
        if len(parts) == 2:
            # Complete context names
            contexts = ['chat', 'detection', 'compression', 'synthesis', 'critic', 'list']
            for ctx in contexts:
                if ctx.startswith(word.lower()):
                    yield Completion(
                        ctx,
                        start_position=-len(word),
                        display_meta='model context'
                    )
        elif len(parts) == 3 and parts[1] in ['chat', 'detection', 'compression', 'synthesis', 'critic']:
            # Complete model names
            yield from self._complete_model_names(word)
    
    def _complete_model_names(self, word: str) -> List[Completion]:
        """Complete available model names."""
        # Get all available models
        models = []
        providers = get_available_providers()
        
        for provider_name, provider_config in providers.items():
            provider_models = get_provider_models(provider_name)
            if provider_models:
                for model in provider_models:
                    if isinstance(model, dict):
                        model_name = model.get("name", "unknown")
                        display_name = model.get("display_name", model_name)
                    else:
                        model_name = model
                        display_name = model
                    
                    if model_name.lower().startswith(word.lower()):
                        yield Completion(
                            model_name,
                            start_position=-len(word),
                            display=f"{display_name} ({provider_name})",
                            display_meta='model'
                        )
    
    def _complete_web_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /web command."""
        if len(parts) == 2:
            # Complete subcommands
            subcommands = ['provider', 'list', 'reset']
            for sub in subcommands:
                if sub.startswith(word.lower()):
                    yield Completion(
                        sub,
                        start_position=-len(word),
                        display_meta='web subcommand'
                    )
        elif len(parts) == 3 and parts[1] == 'provider':
            # Complete provider names
            providers = WEB_SEARCH_PROVIDERS
            for provider in providers:
                if provider.startswith(word.lower()):
                    yield Completion(
                        provider,
                        start_position=-len(word),
                        display_meta='search provider'
                    )
    
    def _complete_set_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /set and /mset commands."""
        if len(parts) == 2:
            # Get parameter types from PARAM_HANDLERS to stay in sync
            from episodic.commands.settings_handlers import PARAM_HANDLERS

            # Build param_types from PARAM_HANDLERS by inspecting handler names
            param_types = {
                'debug': 'boolean/categories',  # Special case
            }

            for param, handler in PARAM_HANDLERS.items():
                # Convert underscore to dash for display
                display_param = param.replace('_', '-')

                # Determine type from handler name
                handler_str = str(handler)
                if 'handle_boolean_param' in handler_str:
                    param_types[display_param] = 'boolean'
                elif 'handle_integer_param' in handler_str:
                    param_types[display_param] = 'integer'
                elif 'handle_float_param' in handler_str:
                    param_types[display_param] = 'float'
                elif 'handle_string_param' in handler_str:
                    param_types[display_param] = 'string'
                elif 'handle_list_param' in handler_str:
                    param_types[display_param] = 'list'
                elif 'handle_rag_embedding_model' in handler_str:
                    param_types[display_param] = 'string (restart required)'
                else:
                    param_types[display_param] = 'value'

            # For /mset, add model parameter options
            if parts[0] == '/mset':
                param_types.update({
                    'chat.temperature': 'float (0.0-2.0)',
                    'chat.max_tokens': 'integer',
                    'chat.top_p': 'float (0.0-1.0)',
                    'detection.temperature': 'float (0.0-2.0)',
                    'detection.max_tokens': 'integer',
                    'compression.temperature': 'float (0.0-2.0)',
                    'compression.max_tokens': 'integer',
                    'synthesis.temperature': 'float (0.0-2.0)',
                    'synthesis.max_tokens': 'integer'
                })

            for param in sorted(param_types.keys()):
                if param.startswith(word.lower()):
                    # Get current value for display
                    meta = self._get_param_meta(param, param_types[param])
                    yield Completion(
                        param,
                        start_position=-len(word),
                        display_meta=meta
                    )
        elif len(parts) == 3:
            # Complete parameter values
            param = parts[1]
            if param == 'debug':
                # Special handling for debug parameter - can be boolean or categories
                from episodic.debug_system import debug_system
                
                # Check if we're in comma-separated mode
                if ',' in parts[2]:
                    # Complete after comma
                    prefix = parts[2][:parts[2].rfind(',')+1]
                    partial = parts[2][parts[2].rfind(',')+1:].strip()
                    used_categories = set(cat.strip() for cat in parts[2].split(',')[:-1])
                    
                    for category in debug_system.CATEGORIES.keys():
                        if category not in used_categories and category.startswith(partial.lower()):
                            yield Completion(
                                prefix + category,
                                start_position=-len(parts[2]),
                                display_meta='add category'
                            )
                else:
                    # First, offer boolean options
                    for value in ['true', 'false', 'on', 'off', 'all']:
                        if value.startswith(word.lower()):
                            yield Completion(
                                value,
                                start_position=-len(word),
                                display_meta='enable/disable all'
                            )
                    
                    # Then offer categories
                    for category in debug_system.CATEGORIES.keys():
                        if category.startswith(word.lower()):
                            yield Completion(
                                category,
                                start_position=-len(word),
                                display_meta='debug category'
                            )
            else:
                # Check if it's a boolean parameter by looking up in PARAM_HANDLERS
                from episodic.commands.settings_handlers import PARAM_HANDLERS

                # Convert dash to underscore for lookup
                param_key = param.replace('-', '_')
                handler = PARAM_HANDLERS.get(param_key)

                if handler and 'handle_boolean_param' in str(handler):
                    # Boolean parameters
                    for value in ['true', 'false']:
                        if value.startswith(word.lower()):
                            yield Completion(
                                value,
                                start_position=-len(word),
                                display_meta='boolean'
                            )
                elif param == 'color-mode':
                    # Color mode options
                    for mode in ['full', 'minimal', 'none']:
                        if mode.startswith(word.lower()):
                            yield Completion(
                                mode,
                                start_position=-len(word),
                                display_meta='color mode'
                            )
                elif param == 'topic-granularity':
                    # Topic granularity options
                    granularity_meta = {
                        'fine': 'many boundaries (0.3)',
                        'medium': 'balanced (0.5)',
                        'coarse': 'major themes (0.7)'
                    }
                    for level, desc in granularity_meta.items():
                        if level.startswith(word.lower()):
                            yield Completion(
                                level,
                                start_position=-len(word),
                                display_meta=desc
                            )
                elif param == 'topic-temperature':
                    # Topic temperature suggestions
                    temp_suggestions = {
                        '0.5': 'sharper (more confident)',
                        '0.7': 'slightly sharper',
                        '1.0': 'default (no scaling)',
                        '1.5': 'softer (less confident)',
                        '2.0': 'much softer'
                    }
                    for temp, desc in temp_suggestions.items():
                        if temp.startswith(word):
                            yield Completion(
                                temp,
                                start_position=-len(word),
                                display_meta=desc
                            )
    
    def _complete_subcommand(self, cmd: str, parts: List[str], word: str) -> List[Completion]:
        """Complete subcommands for unified commands."""
        if len(parts) == 2:
            # Get subcommands for this command
            subcommands = []

            if cmd == 'topics':
                subcommands = TOPIC_ACTIONS
            elif cmd == 'compression':
                subcommands = COMPRESSION_ACTIONS
            elif cmd == 'voice':
                subcommands = VOICE_ACTIONS
            elif cmd == 'rag':
                subcommands = RAG_ACTIONS

            for sub in subcommands:
                if sub.startswith(word.lower()):
                    yield Completion(
                        sub,
                        start_position=-len(word),
                        display_meta='subcommand'
                    )

        # Complete options for specific subcommands
        elif len(parts) >= 3 and cmd == 'topics' and parts[1] == 'reanalyze':
            # Options for /topics reanalyze
            options = {'apply': 'save to database', 'verbose': 'show merge details'}
            already_used = set(parts[2:])
            for opt, desc in options.items():
                if opt not in already_used and opt.startswith(word.lower()):
                    yield Completion(
                        opt,
                        start_position=-len(word),
                        display_meta=desc
                    )

    def _complete_mode_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /mode command."""
        if len(parts) == 2:
            modes = {
                'local': 'Free, private, offline-capable',
                'cloud': 'Paid, higher quality APIs'
            }
            for mode, desc in modes.items():
                if mode.startswith(word.lower()):
                    yield Completion(
                        mode,
                        start_position=-len(word),
                        display_meta=desc
                    )
    
    def _complete_file_path(self, cmd: str, parts: List[str], word: str) -> List[Completion]:
        """Complete file paths for import/export/index commands."""
        # Get the partial path
        if len(parts) >= 2:
            partial_path = parts[-1] if word else ''
        else:
            partial_path = ''
        
        # Expand user home directory
        if partial_path.startswith('~'):
            partial_path = os.path.expanduser(partial_path)
        
        # Get directory and partial filename
        if partial_path:
            if os.path.isdir(partial_path):
                search_dir = partial_path
                partial_name = ''
            else:
                search_dir = os.path.dirname(partial_path) or '.'
                partial_name = os.path.basename(partial_path)
        else:
            search_dir = '.'
            partial_name = ''
        
        # Get completions
        try:
            if os.path.isdir(search_dir):
                for entry in sorted(os.listdir(search_dir)):
                    if entry.startswith(partial_name):
                        full_path = os.path.join(search_dir, entry)
                        
                        # Create display text
                        if os.path.isdir(full_path):
                            display = entry + '/'
                            meta = 'directory'
                        else:
                            display = entry
                            meta = 'file'
                        
                        # Handle the completion text
                        if partial_path and not partial_path.endswith('/'):
                            # Replace the partial filename
                            completion_text = os.path.join(os.path.dirname(partial_path), entry)
                        else:
                            completion_text = entry
                        
                        yield Completion(
                            completion_text,
                            start_position=-len(word),
                            display=display,
                            display_meta=meta
                        )
        except (OSError, PermissionError):
            # Can't read directory
            pass
    
    def _complete_save_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /save command with recent topic names."""
        if len(parts) == 2:
            try:
                # Get last 5 topics
                topics = get_recent_topics(limit=5)

                for topic in topics:
                    topic_name = topic.get('name', '')

                    # Skip ongoing-* placeholder names
                    if not topic_name or topic_name.startswith('ongoing-'):
                        continue

                    # Clean up topic name for filename
                    safe_name = ''.join(c for c in topic_name.lower() if c.isalnum() or c in ' -_')
                    safe_name = safe_name.replace(' ', '-').strip('-')

                    if safe_name and safe_name.startswith(word.lower()):
                        yield Completion(
                            safe_name,
                            start_position=-len(word),
                            display_meta='topic'
                        )
            except Exception:
                pass
    
    def _complete_style_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /style command arguments."""
        if len(parts) == 2:
            # Complete style options
            styles = RESPONSE_STYLES
            for style in styles:
                if style.startswith(word.lower()):
                    yield Completion(
                        style,
                        start_position=-len(word),
                        display_meta='response style'
                    )
    
    def _complete_format_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /format command arguments."""
        if len(parts) == 2:
            # Complete format options
            formats = RESPONSE_FORMATS
            for fmt in formats:
                if fmt.startswith(word.lower()):
                    yield Completion(
                        fmt,
                        start_position=-len(word),
                        display_meta='response format'
                    )
    
    def _complete_detail_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /detail command arguments."""
        if len(parts) == 2:
            # Complete detail level options
            detail_levels = DETAIL_LEVELS
            for level in detail_levels:
                if level.startswith(word.lower()):
                    yield Completion(
                        level,
                        start_position=-len(word),
                        display_meta='detail level'
                    )
    
    def _complete_theme_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /theme command arguments."""
        if len(parts) == 2:
            # Import here to avoid circular imports
            from episodic.configuration import COLOR_SCHEMES
            
            # Complete theme names
            themes = list(COLOR_SCHEMES.keys())
            themes.append('list')  # Add 'list' as an option
            
            for theme in themes:
                if theme.startswith(word.lower()):
                    yield Completion(
                        theme,
                        start_position=-len(word),
                        display_meta='theme' if theme != 'list' else 'action'
                    )
    
    def _complete_load_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /load command with most recently modified markdown files."""
        if len(parts) == 2:
            try:
                import os
                from pathlib import Path
                from datetime import datetime
                from episodic.config import config
                
                # Get export directory
                export_dir = Path(os.path.expanduser(config.get("export_directory", "~/.episodic/exports")))
                
                if export_dir.exists():
                    # Get all markdown files with modification times
                    md_files = []
                    for file in export_dir.glob("*.md"):
                        stat = file.stat()
                        mtime = stat.st_mtime
                        md_files.append((file.stem, mtime, file.name))
                    
                    # Sort by modification time (newest first)
                    md_files.sort(key=lambda x: x[1], reverse=True)
                    
                    # Generate completions for top 10 most recent files
                    for filename, mtime, full_name in md_files[:10]:
                        if filename.lower().startswith(word.lower()):
                            # Calculate time ago
                            time_diff = datetime.now().timestamp() - mtime
                            if time_diff < 3600:
                                time_ago = f"{int(time_diff / 60)} min ago"
                            elif time_diff < 86400:
                                time_ago = f"{int(time_diff / 3600)} hours ago"
                            else:
                                time_ago = f"{int(time_diff / 86400)} days ago"
                            
                            yield Completion(
                                filename,  # Complete without .md extension
                                start_position=-len(word),
                                display_meta=time_ago
                            )
            except Exception:
                # If anything fails, just don't provide completions
                pass

    def _complete_script_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /script command with scripts from ~/.episodic/scripts/."""
        if len(parts) == 2:
            try:
                import os
                from pathlib import Path
                from datetime import datetime

                # Scripts directory in user's home
                scripts_dir = Path.home() / ".episodic" / "scripts"

                if scripts_dir.exists():
                    # Get all .txt script files with modification times
                    script_files = []
                    for file in scripts_dir.glob("*.txt"):
                        stat = file.stat()
                        mtime = stat.st_mtime
                        script_files.append((file.stem, mtime, file.name))

                    # Sort by modification time (newest first)
                    script_files.sort(key=lambda x: x[1], reverse=True)

                    # Generate completions
                    for filename, mtime, full_name in script_files:
                        if filename.lower().startswith(word.lower()):
                            # Calculate time ago
                            time_diff = datetime.now().timestamp() - mtime
                            if time_diff < 3600:
                                time_ago = f"{int(time_diff / 60)} min ago"
                            elif time_diff < 86400:
                                time_ago = f"{int(time_diff / 3600)} hours ago"
                            else:
                                time_ago = f"{int(time_diff / 86400)} days ago"

                            yield Completion(
                                filename,  # Complete without .txt extension
                                start_position=-len(word),
                                display_meta=time_ago
                            )
            except Exception:
                pass

    def _complete_summary_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /summary command arguments."""
        if len(parts) >= 2:
            # Complete length options and count options
            options = SUMMARY_LENGTHS + ['all', 'loaded', '5', '10', '20', '50']
            # Filter out already used options
            used_options = set(parts[1:])
            
            for option in options:
                if option not in used_options and option.startswith(word):
                    # Add descriptive metadata
                    if option == 'brief':
                        meta = '2-3 sentences'
                    elif option == 'short':
                        meta = 'compact paragraph'
                    elif option == 'standard':
                        meta = 'medium length'
                    elif option == 'detailed':
                        meta = 'comprehensive'
                    elif option == 'bulleted':
                        meta = 'bullet points'
                    elif option == 'all':
                        meta = 'entire history'
                    elif option == 'loaded':
                        meta = 'last loaded conversation'
                    else:
                        meta = f'last {option} exchanges'
                    
                    yield Completion(
                        option,
                        start_position=-len(word),
                        display_meta=meta
                    )
    
    def _complete_debug_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /debug command arguments."""
        if len(parts) == 2:
            # Complete subcommands
            subcommands = {
                'on': 'Enable debug categories',
                'off': 'Disable debug categories',
                'only': 'Enable only specified categories',
                'status': 'Show debug status',
                'toggle': 'Toggle a debug category'
            }
            
            for subcmd, description in subcommands.items():
                if subcmd.startswith(word.lower()):
                    yield Completion(
                        subcmd,
                        start_position=-len(word),
                        display_meta=description
                    )
        
        elif len(parts) >= 3:
            # Complete debug categories for on/off/only/toggle commands
            subcmd = parts[1]
            if subcmd in ['on', 'off', 'only', 'toggle']:
                # Import debug categories
                from episodic.debug_system import debug_system
                
                # Get categories already mentioned
                used_categories = set(parts[2:])
                
                # For 'toggle', only allow one category
                if subcmd == 'toggle' and len(parts) > 3:
                    return
                
                # Complete categories
                for category, description in debug_system.CATEGORIES.items():
                    if category not in used_categories and category.startswith(word.lower()):
                        yield Completion(
                            category,
                            start_position=-len(word),
                            display_meta=description
                        )
    
    def _complete_memory_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete memory command options."""
        if len(parts) == 2:
            # Complete subcommands
            subcommands = {
                'search': 'Search memory entries',
                'show': 'Show specific memory entry',
                'list': 'List memory entries'
            }
            for sub, desc in subcommands.items():
                if sub.startswith(word.lower()):
                    yield Completion(sub, start_position=-len(word), display_meta=desc)
    
    def _complete_forget_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete forget command options."""
        if len(parts) == 2:
            # Complete options
            options = {
                '--all': 'Clear all memories',
                '--contains': 'Forget memories containing text',
                '--source': 'Forget memories from source'
            }
            for opt, desc in options.items():
                if opt.startswith(word):
                    yield Completion(opt, start_position=-len(word), display_meta=desc)

    def _complete_help_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /help with command names."""
        if len(parts) == 2:
            # Suggest command names for help lookups
            for cmd in sorted(command_registry._commands.keys()):
                if cmd.startswith(word.lower()):
                    info = command_registry._commands[cmd]
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display_meta=info.description[:40] if info.description else ''
                    )

    def _complete_docs_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /docs subcommands."""
        if len(parts) == 2:
            descriptions = {
                'list': 'List all documents',
                'show': 'Show document content',
                'remove': 'Remove a document',
                'rm': 'Remove a document (alias)',
                'clear': 'Remove all documents'
            }
            for action in DOCS_ACTIONS:
                if action.startswith(word.lower()):
                    yield Completion(
                        action,
                        start_position=-len(word),
                        display_meta=descriptions.get(action, '')
                    )

    def _complete_prompt_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /prompt subcommands and prompt names."""
        if len(parts) == 2:
            # Complete subcommands
            descriptions = {
                'list': 'List available prompts',
                'use': 'Use a prompt',
                'show': 'Show prompt content'
            }
            for action in PROMPT_ACTIONS:
                if action.startswith(word.lower()):
                    yield Completion(
                        action,
                        start_position=-len(word),
                        display_meta=descriptions.get(action, '')
                    )
            # Also suggest prompt names directly
            try:
                from episodic.prompt_manager import get_prompt_manager
                pm = get_prompt_manager()
                for name in pm.list_prompts():
                    if name.startswith(word.lower()):
                        yield Completion(name, start_position=-len(word), display_meta='prompt')
            except:
                pass
        elif len(parts) == 3 and parts[1] in ['use', 'show']:
            # Complete prompt names for use/show
            try:
                from episodic.prompt_manager import get_prompt_manager
                pm = get_prompt_manager()
                for name in pm.list_prompts():
                    if name.startswith(word.lower()):
                        yield Completion(name, start_position=-len(word), display_meta='prompt')
            except:
                pass

    def _complete_reset_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /reset with 'all' or parameter names."""
        if len(parts) == 2:
            # Suggest 'all' first
            if 'all'.startswith(word.lower()):
                yield Completion('all', start_position=-len(word), display_meta='Reset all to defaults')
            # Also suggest parameter names from PARAM_HANDLERS
            from episodic.commands.settings_handlers import PARAM_HANDLERS
            for param in PARAM_HANDLERS.keys():
                if param.startswith(word.lower()):
                    yield Completion(param, start_position=-len(word), display_meta='parameter')

    def _complete_dev_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /dev subcommands."""
        if len(parts) == 2:
            descriptions = {
                'reindex-help': 'Reindex help documentation'
            }
            for action in DEV_ACTIONS:
                if action.startswith(word.lower()):
                    yield Completion(
                        action,
                        start_position=-len(word),
                        display_meta=descriptions.get(action, '')
                    )

    def _complete_migrate_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /migrate subcommands."""
        if len(parts) == 2:
            descriptions = {
                'run': 'Run migration',
                'dry-run': 'Preview what would be migrated',
                'rollback': 'Rollback migration'
            }
            for action in MIGRATE_ACTIONS:
                if action.startswith(word.lower()):
                    yield Completion(
                        action,
                        start_position=-len(word),
                        display_meta=descriptions.get(action, '')
                    )