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
            elif full_cmd in ['topics', 'compression']:
                yield from self._complete_subcommand(full_cmd, parts, word_before_cursor)
            elif full_cmd in ['in', 'out', 'index', 'script']:
                yield from self._complete_file_path(full_cmd, parts, word_before_cursor)
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
            providers = ['duckduckgo', 'google', 'bing', 'brave', 'searx']
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
            # Complete parameter names with type hints
            # Define parameters with their types
            param_types = {
                # Boolean parameters
                'debug': 'boolean/categories',
                'show-cost': 'boolean',
                'show-drift': 'boolean',
                'show-topics': 'boolean',
                'automatic-topic-detection': 'boolean',
                'auto-compress-topics': 'boolean',
                'stream-responses': 'boolean',
                'stream-constant-rate': 'boolean',
                'stream-natural-rhythm': 'boolean',
                'text-wrap': 'boolean',
                'benchmark': 'boolean',
                'benchmark-display': 'boolean',
                'rag-enabled': 'boolean',
                'web-search-enabled': 'boolean',
                'muse-mode': 'boolean',
                'enable-tab-completion': 'boolean',
                # Number parameters
                'stream-rate': 'number',
                'context-depth': 'number',
                # String parameters
                'color-mode': 'choice'
                # muse-detail removed - use /detail command instead
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
            elif param in ['show-cost', 'show-drift', 'show-topics',
                        'automatic-topic-detection', 'auto-compress-topics',
                        'stream-responses', 'stream-constant-rate',
                        'stream-natural-rhythm', 'text-wrap', 'benchmark',
                        'benchmark-display', 'rag-enabled', 'web-search-enabled',
                        'muse-mode', 'enable-tab-completion']:
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
            # muse-detail completion removed - use /detail command instead
    
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
                        
                        # For in, only show .md files and directories
                        if cmd == 'in':
                            if not (os.path.isdir(full_path) or entry.endswith('.md')):
                                continue
                        
                        # For out, suggest a default name if no input
                        if cmd == 'out' and not word:
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
        """Complete /save command with topic-based names."""
        if len(parts) == 2:
            suggestions = []
            
            # Start simple and add complexity gradually
            try:
                from episodic.db import get_recent_topics
                
                # Get current topic
                topics = get_recent_topics(limit=1)
                if topics:
                    current_topic = topics[0]
                    topic_name = current_topic.get('name', '')
                    
                    # Simple suggestion based on topic
                    if topic_name and not topic_name.startswith('ongoing-'):
                        # Clean up topic name for filename
                        safe_name = ''.join(c for c in topic_name.lower() if c.isalnum() or c in ' -_')
                        safe_name = safe_name.replace(' ', '-').strip('-')
                        
                        if safe_name:
                            suggestions.append(safe_name)
                            suggestions.append(f"{safe_name}-final")
                    elif topic_name.startswith('ongoing-'):
                        # For ongoing topics, extract keywords from recent messages
                        from episodic.db import get_recent_nodes
                        
                        recent_messages = get_recent_nodes(limit=10)
                        if recent_messages:
                            # Focus on user messages
                            user_messages = [m for m in recent_messages if m.get('role') == 'user']
                            if user_messages:
                                # Common words to filter out
                                common_words = {
                                    'what', 'this', 'that', 'with', 'from', 'about', 'have', 'been',
                                    'will', 'would', 'could', 'should', 'there', 'their', 'they',
                                    'your', 'more', 'some', 'just', 'like', 'into', 'than', 'then',
                                    'when', 'where', 'which', 'while', 'after', 'before', 'does',
                                    'particular', 'particularly', 'specifically', 'certain', 'various'
                                }
                                
                                # Extract keywords from last few messages
                                all_words = []
                                for msg in user_messages[-3:]:
                                    words = msg.get('content', '').lower().split()
                                    all_words.extend(words)
                                
                                # Filter for meaningful words
                                keywords = [w for w in all_words 
                                          if len(w) > 4 and w not in common_words and w.isalnum()]
                                
                                # Count frequency
                                from collections import Counter
                                word_counts = Counter(keywords)
                                
                                # Use most common keywords as suggestions
                                for word, count in word_counts.most_common(2):
                                    if count > 1 or len(word) > 6:
                                        safe_word = ''.join(c for c in word if c.isalnum())[:20]
                                        if safe_word:
                                            suggestions.append(safe_word)
            except:
                pass
            
            # Add some fallback suggestions
            if not suggestions:
                suggestions = ['conversation', 'chat-export', 'notes']
            
            # Yield completions the EXACT same way as the test version
            for suggestion in suggestions:
                if suggestion.startswith(word.lower()):
                    yield Completion(
                        suggestion,
                        start_position=-len(word),
                        display_meta='suggested name'
                    )
            return  # End of function
    
    def _complete_style_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /style command arguments."""
        if len(parts) == 2:
            # Complete style options
            styles = ['concise', 'standard', 'comprehensive', 'custom']
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
            formats = ['paragraph', 'bulleted', 'mixed', 'academic']
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
            detail_levels = ['minimal', 'moderate', 'detailed', 'maximum']
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
    
    def _complete_summary_command(self, parts: List[str], word: str) -> List[Completion]:
        """Complete /summary command arguments."""
        if len(parts) >= 2:
            # Complete length options and count options
            options = ['brief', 'short', 'standard', 'detailed', 'bulleted', 'all', 'loaded', '5', '10', '20', '50']
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