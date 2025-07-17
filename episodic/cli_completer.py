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


class EpisodicCompleter(Completer):
    """Context-aware completer for Episodic commands."""
    
    def __init__(self):
        # Ensure commands are registered
        register_all_commands()
        
        # Build command list including aliases
        self.commands = set()
        self.command_aliases = {}  # alias -> full command
        
        # Add all commands from the registry
        for cmd_name, cmd_info in command_registry._commands.items():
            self.commands.add(cmd_name)
            # Add aliases
            if cmd_info.aliases:
                for alias in cmd_info.aliases:
                    self.commands.add(alias)
                    self.command_aliases[alias] = cmd_name
        
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
            # Complete command names
            partial = line[1:]  # Remove the /
            for cmd in sorted(self.commands):
                if cmd.startswith(partial.lower()):
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
            elif full_cmd in ['set', 'mset']:
                yield from self._complete_set_command(parts, word_before_cursor)
            elif full_cmd in ['topics', 'compression']:
                yield from self._complete_subcommand(full_cmd, parts, word_before_cursor)
            elif full_cmd in ['import', 'export', 'index', 'script']:
                yield from self._complete_file_path(full_cmd, parts, word_before_cursor)
            elif full_cmd == 'save':
                yield from self._complete_save_command(parts, word_before_cursor)
    
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
    
    def _complete_model_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /model command."""
        if len(parts) == 2:
            # Complete context names
            contexts = ['chat', 'detection', 'compression', 'synthesis', 'list']
            for ctx in contexts:
                if ctx.startswith(word.lower()):
                    yield Completion(
                        ctx,
                        start_position=-len(word),
                        display_meta='model context'
                    )
        elif len(parts) == 3 and parts[1] in ['chat', 'detection', 'compression', 'synthesis']:
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
    
    def _complete_set_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /set and /mset commands."""
        if len(parts) == 2:
            # Complete parameter names with type hints
            # Define parameters with their types
            param_types = {
                # Boolean parameters
                'debug': 'boolean',
                'show_cost': 'boolean',
                'show_drift': 'boolean',
                'show_topics': 'boolean',
                'automatic_topic_detection': 'boolean',
                'auto_compress_topics': 'boolean',
                'stream_responses': 'boolean',
                'stream_constant_rate': 'boolean',
                'stream_natural_rhythm': 'boolean',
                'text_wrap': 'boolean',
                'benchmark': 'boolean',
                'benchmark_display': 'boolean',
                'rag_enabled': 'boolean',
                'web_search_enabled': 'boolean',
                'muse_mode': 'boolean',
                'enable_tab_completion': 'boolean',
                # Number parameters
                'stream_rate': 'number',
                'context_depth': 'number',
                # String parameters
                'color_mode': 'choice'
            }
            
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
                    yield Completion(
                        param,
                        start_position=-len(word),
                        display_meta=param_types[param]
                    )
        elif len(parts) == 3:
            # Complete parameter values
            param = parts[1]
            if param in ['debug', 'show_cost', 'show_drift', 'show_topics',
                        'automatic_topic_detection', 'auto_compress_topics',
                        'stream_responses', 'stream_constant_rate',
                        'stream_natural_rhythm', 'text_wrap', 'benchmark',
                        'benchmark_display', 'rag_enabled', 'web_search_enabled',
                        'muse_mode', 'enable_tab_completion']:
                # Boolean parameters
                for value in ['true', 'false']:
                    if value.startswith(word.lower()):
                        yield Completion(
                            value,
                            start_position=-len(word),
                            display_meta='boolean'
                        )
            elif param == 'color_mode':
                # Color mode options
                for mode in ['full', 'minimal', 'none']:
                    if mode.startswith(word.lower()):
                        yield Completion(
                            mode,
                            start_position=-len(word),
                            display_meta='color mode'
                        )
    
    def _complete_subcommand(self, cmd: str, parts: List[str], word: str) -> List[Completion]:
        """Complete subcommands for unified commands."""
        if len(parts) == 2:
            # Get subcommands for this command
            subcommands = []
            
            if cmd == 'topics':
                subcommands = ['list', 'rename', 'compress', 'index', 'scores', 'stats']
            elif cmd == 'compression':
                subcommands = ['stats', 'queue', 'compress', 'api-stats', 'reset-api']
            
            for sub in subcommands:
                if sub.startswith(word.lower()):
                    yield Completion(
                        sub,
                        start_position=-len(word),
                        display_meta='subcommand'
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
                        
                        # For import, only show .md files and directories
                        if cmd == 'import':
                            if not (os.path.isdir(full_path) or entry.endswith('.md')):
                                continue
                        
                        # For export, suggest a default name if no input
                        if cmd == 'export' and not word:
                            # Get current topic name for default
                            topics = get_recent_topics(limit=1)
                            if topics and topics[0]['name'] != 'General':
                                topic_name = topics[0]['name'].lower().replace(' ', '_')
                                default_name = f"{topic_name}_export.md"
                                yield Completion(
                                    default_name,
                                    start_position=0,
                                    display_meta='suggested name'
                                )
                                return
                        
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
        """Complete /save command with checkpoint names."""
        if len(parts) == 2:
            # Suggest checkpoint names based on current context
            suggestions = [
                'before_refactor',
                'checkpoint',
                'backup',
                'stable',
                'working',
                'experiment'
            ]
            
            for suggestion in suggestions:
                if suggestion.startswith(word.lower()):
                    yield Completion(
                        suggestion,
                        start_position=-len(word),
                        display_meta='checkpoint name'
                    )