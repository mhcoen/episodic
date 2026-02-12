"""
Command-specific tab completion functions for Episodic CLI.

Extracted from cli_completer.py to keep files under the 600-line limit.
Each function is a standalone completer that yields prompt_toolkit Completions.
"""

from typing import Callable, List
import os

from prompt_toolkit.completion import Completion

from episodic.debug_utils import debug_print
from episodic.llm_config import get_available_providers, get_provider_models
from episodic.db_topics import get_recent_topics
from episodic.constants import (
    WEB_SEARCH_PROVIDERS, RESPONSE_STYLES, RESPONSE_FORMATS, DETAIL_LEVELS,
    TOPIC_ACTIONS, COMPRESSION_ACTIONS, VOICE_ACTIONS, RAG_ACTIONS, SUMMARY_LENGTHS,
    DOCS_ACTIONS, PROMPT_ACTIONS, DEV_ACTIONS, MIGRATE_ACTIONS, KG_ACTIONS
)


def _yield_options(options: dict, word: str, meta_label: str = '') -> List[Completion]:
    """Yield completions for a dict of {value: description} matching word."""
    for value, desc in options.items():
        if value.startswith(word.lower()):
            yield Completion(
                value, start_position=-len(word),
                display_meta=desc if not meta_label else meta_label
            )


def _yield_list(items: list, word: str, meta: str) -> List[Completion]:
    """Yield completions for a list of strings matching word."""
    for item in items:
        if item.startswith(word.lower()):
            yield Completion(item, start_position=-len(word), display_meta=meta)


def complete_model_names(word: str) -> List[Completion]:
    """Complete available model names."""
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
                        model_name, start_position=-len(word),
                        display=f"{display_name} ({provider_name})",
                        display_meta='model'
                    )


def complete_model_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /model command."""
    if len(parts) == 2:
        contexts = ['chat', 'detection', 'compression', 'synthesis', 'intent', 'critic', 'list']
        yield from _yield_list(contexts, word, 'model context')
    elif len(parts) == 3 and parts[1] in ['chat', 'detection', 'compression', 'synthesis', 'intent', 'critic']:
        yield from complete_model_names(word)


def complete_web_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /web command."""
    if len(parts) == 2:
        yield from _yield_list(['provider', 'list', 'reset'], word, 'web subcommand')
    elif len(parts) == 3 and parts[1] == 'provider':
        yield from _yield_list(list(WEB_SEARCH_PROVIDERS), word, 'search provider')


def complete_set_command(
    parts: List[str], word: str, get_param_meta: Callable[[str, str], str]
) -> List[Completion]:
    """Complete /set and /mset commands."""
    if len(parts) == 2:
        from episodic.commands.settings_handlers import PARAM_HANDLERS

        param_types = {'debug': 'boolean/categories'}
        for param, handler in PARAM_HANDLERS.items():
            display_param = param.replace('_', '-')
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

        param_types.update({
            'gpt.verbosity': 'low/medium/high',
            'gpt.reasoning-effort': 'minimal/low/medium/high',
            'reasoning': 'on/off',
            'reasoning-effort': 'minimal/low/medium/high',
            'reasoning-verbosity': 'low/medium/high'
        })

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
                meta = get_param_meta(param, param_types[param])
                yield Completion(param, start_position=-len(word), display_meta=meta)
    elif len(parts) == 3:
        yield from _complete_set_value(parts, word)


def _complete_set_value(parts: List[str], word: str) -> List[Completion]:
    """Complete value for /set param <value>."""
    param = parts[1]
    if param == 'debug':
        from episodic.debug_system import debug_system

        if ',' in parts[2]:
            prefix = parts[2][:parts[2].rfind(',') + 1]
            partial = parts[2][parts[2].rfind(',') + 1:].strip()
            used_categories = set(cat.strip() for cat in parts[2].split(',')[:-1])
            for category in debug_system.CATEGORIES.keys():
                if category not in used_categories and category.startswith(partial.lower()):
                    yield Completion(
                        prefix + category, start_position=-len(parts[2]),
                        display_meta='add category'
                    )
        else:
            for value in ['true', 'false', 'on', 'off', 'all']:
                if value.startswith(word.lower()):
                    yield Completion(value, start_position=-len(word), display_meta='enable/disable all')
            for category in debug_system.CATEGORIES.keys():
                if category.startswith(word.lower()):
                    yield Completion(category, start_position=-len(word), display_meta='debug category')
        return

    from episodic.commands.settings_handlers import PARAM_HANDLERS
    param_key = param.replace('-', '_')
    handler = PARAM_HANDLERS.get(param_key)

    if handler and 'handle_boolean_param' in str(handler):
        yield from _yield_list(['true', 'false'], word, 'boolean')
    elif param == 'color-mode':
        yield from _yield_options({'full': 'color mode', 'minimal': 'color mode', 'none': 'color mode'}, word)
    elif param == 'topic-granularity':
        yield from _yield_options({
            'fine': 'many boundaries (0.3)', 'medium': 'balanced (0.5)', 'coarse': 'major themes (0.7)'
        }, word)
    elif param == 'topic-temperature':
        for temp, desc in {'0.5': 'sharper (more confident)', '0.7': 'slightly sharper',
                           '1.0': 'default (no scaling)', '1.5': 'softer (less confident)',
                           '2.0': 'much softer'}.items():
            if temp.startswith(word):
                yield Completion(temp, start_position=-len(word), display_meta=desc)
    elif param == 'topic-strategy':
        yield from _yield_options({
            'default': 'Neural + Commitment (recommended)', 'neural': 'Fine-tuned DistilBERT',
            'dual_window': 'Embedding drift (4,1)+(4,2)', 'ensemble': 'Weighted combination',
            'time_aware': 'Temporal gap detection', 'cusum': 'CUSUM change detection',
            'delta': 'Delta change detection', 'keyword': 'Keyword-based detection',
            'null': 'No detection (testing)',
        }, word)
    elif param == 'gpt.verbosity':
        yield from _yield_options({
            'low': 'concise answers, code generation', 'medium': 'standard responses (default)',
            'high': 'detailed explanations'
        }, word)
    elif param in ('gpt.reasoning-effort', 'reasoning-effort'):
        meta = {
            'minimal': 'fastest, fewest reasoning tokens' if param.startswith('gpt.') else 'fastest, less thorough',
            'low': 'favors speed', 'medium': 'balanced (default)',
            'high': 'thorough reasoning' if param.startswith('gpt.') else 'most thorough'
        }
        yield from _yield_options(meta, word)
    elif param == 'reasoning':
        yield from _yield_list(['on', 'off', 'true', 'false'], word, 'toggle reasoning')
    elif param == 'reasoning-verbosity':
        yield from _yield_options({
            'low': 'concise answers', 'medium': 'standard (default)', 'high': 'detailed explanations'
        }, word)


def complete_subcommand(cmd: str, parts: List[str], word: str) -> List[Completion]:
    """Complete subcommands for unified commands."""
    if len(parts) == 2:
        action_map = {
            'topics': TOPIC_ACTIONS, 'compression': COMPRESSION_ACTIONS,
            'voice': VOICE_ACTIONS, 'rag': RAG_ACTIONS, 'kg': KG_ACTIONS
        }
        subcommands = action_map.get(cmd, [])
        yield from _yield_list(subcommands, word, 'subcommand')
    elif len(parts) >= 3 and cmd == 'topics' and parts[1] == 'reanalyze':
        already_used = set(parts[2:])
        for opt, desc in {'apply': 'save to database', 'verbose': 'show merge details'}.items():
            if opt not in already_used and opt.startswith(word.lower()):
                yield Completion(opt, start_position=-len(word), display_meta=desc)


def complete_mode_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /mode command."""
    if len(parts) == 2:
        yield from _yield_options({
            'local': 'Free, private, offline-capable', 'cloud': 'Paid, higher quality APIs'
        }, word)


def complete_file_path(cmd: str, parts: List[str], word: str) -> List[Completion]:
    """Complete file paths for import/export/index commands."""
    partial_path = parts[-1] if (len(parts) >= 2 and word) else ''
    if partial_path.startswith('~'):
        partial_path = os.path.expanduser(partial_path)

    if partial_path:
        if os.path.isdir(partial_path):
            search_dir, partial_name = partial_path, ''
        else:
            search_dir = os.path.dirname(partial_path) or '.'
            partial_name = os.path.basename(partial_path)
    else:
        search_dir, partial_name = '.', ''

    try:
        if os.path.isdir(search_dir):
            for entry in sorted(os.listdir(search_dir)):
                if entry.startswith(partial_name):
                    full_path = os.path.join(search_dir, entry)
                    is_dir = os.path.isdir(full_path)
                    display = entry + '/' if is_dir else entry
                    meta = 'directory' if is_dir else 'file'
                    if partial_path and not partial_path.endswith('/'):
                        completion_text = os.path.join(os.path.dirname(partial_path), entry)
                    else:
                        completion_text = entry
                    yield Completion(
                        completion_text, start_position=-len(word),
                        display=display, display_meta=meta
                    )
    except (OSError, PermissionError):
        pass


def complete_save_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /save command with recent topic names."""
    if len(parts) == 2:
        try:
            for topic in get_recent_topics(limit=5):
                topic_name = topic.get('name', '')
                if not topic_name or topic_name.startswith('ongoing-'):
                    continue
                safe_name = ''.join(c for c in topic_name.lower() if c.isalnum() or c in ' -_')
                safe_name = safe_name.replace(' ', '-').strip('-')
                if safe_name and safe_name.startswith(word.lower()):
                    yield Completion(safe_name, start_position=-len(word), display_meta='topic')
        except Exception:
            pass


def complete_style_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /style command arguments."""
    if len(parts) == 2:
        yield from _yield_list(list(RESPONSE_STYLES), word, 'response style')


def complete_format_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /format command arguments."""
    if len(parts) == 2:
        yield from _yield_list(list(RESPONSE_FORMATS), word, 'response format')


def complete_detail_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /detail command arguments."""
    if len(parts) == 2:
        yield from _yield_list(list(DETAIL_LEVELS), word, 'detail level')


def complete_theme_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /theme command arguments."""
    if len(parts) == 2:
        from episodic.configuration import COLOR_SCHEMES
        themes = list(COLOR_SCHEMES.keys()) + ['list']
        for theme in themes:
            if theme.startswith(word.lower()):
                yield Completion(
                    theme, start_position=-len(word),
                    display_meta='theme' if theme != 'list' else 'action'
                )


def _format_time_ago(mtime: float) -> str:
    """Format a modification timestamp as a human-readable 'X ago' string."""
    from datetime import datetime
    time_diff = datetime.now().timestamp() - mtime
    if time_diff < 3600:
        return f"{int(time_diff / 60)} min ago"
    elif time_diff < 86400:
        return f"{int(time_diff / 3600)} hours ago"
    return f"{int(time_diff / 86400)} days ago"


def complete_load_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /load command with most recently modified markdown files."""
    if len(parts) == 2:
        try:
            from pathlib import Path
            from episodic.config import config
            export_dir = Path(os.path.expanduser(config.get("export_directory", "~/.episodic/exports")))
            if export_dir.exists():
                md_files = [(f.stem, f.stat().st_mtime) for f in export_dir.glob("*.md")]
                md_files.sort(key=lambda x: x[1], reverse=True)
                for filename, mtime in md_files[:10]:
                    if filename.lower().startswith(word.lower()):
                        yield Completion(
                            filename, start_position=-len(word),
                            display_meta=_format_time_ago(mtime)
                        )
        except Exception:
            pass


def complete_script_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /script command with scripts from ~/.episodic/scripts/."""
    if len(parts) == 2:
        try:
            from pathlib import Path
            scripts_dir = Path.home() / ".episodic" / "scripts"
            if scripts_dir.exists():
                script_files = []
                for file in scripts_dir.glob("*.txt"):
                    try:
                        if not file.is_symlink():
                            script_files.append((file.stem, file.stat().st_mtime))
                    except (OSError, FileNotFoundError):
                        continue
                script_files.sort(key=lambda x: x[1], reverse=True)
                for filename, mtime in script_files:
                    if filename.lower().startswith(word.lower()):
                        yield Completion(
                            filename, start_position=-len(word),
                            display_meta=_format_time_ago(mtime)
                        )
        except Exception:
            pass


def complete_summary_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /summary command arguments."""
    if len(parts) >= 2:
        options = SUMMARY_LENGTHS + ['all', 'loaded', '5', '10', '20', '50']
        used_options = set(parts[1:])
        meta_map = {
            'brief': '2-3 sentences', 'short': 'compact paragraph',
            'standard': 'medium length', 'detailed': 'comprehensive',
            'bulleted': 'bullet points', 'all': 'entire history',
            'loaded': 'last loaded conversation'
        }
        for option in options:
            if option not in used_options and option.startswith(word):
                meta = meta_map.get(option, f'last {option} exchanges')
                yield Completion(option, start_position=-len(word), display_meta=meta)


def complete_debug_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /debug command arguments."""
    if len(parts) == 2:
        yield from _yield_options({
            'on': 'Enable debug categories', 'off': 'Disable debug categories',
            'only': 'Enable only specified categories', 'status': 'Show debug status',
            'toggle': 'Toggle a debug category'
        }, word)
    elif len(parts) >= 3 and parts[1] in ['on', 'off', 'only', 'toggle']:
        from episodic.debug_system import debug_system
        if parts[1] == 'toggle' and len(parts) > 3:
            return
        used_categories = set(parts[2:])
        for category, description in debug_system.CATEGORIES.items():
            if category not in used_categories and category.startswith(word.lower()):
                yield Completion(category, start_position=-len(word), display_meta=description)


def complete_memory_command(parts: List[str], word: str) -> List[Completion]:
    """Complete memory command options."""
    if len(parts) == 2:
        yield from _yield_options({
            'search': 'Search memory entries', 'show': 'Show specific memory entry',
            'list': 'List memory entries'
        }, word)


def complete_forget_command(parts: List[str], word: str) -> List[Completion]:
    """Complete forget command options."""
    if len(parts) == 2:
        for opt, desc in {'--all': 'Clear all memories', '--contains': 'Forget memories containing text',
                          '--source': 'Forget memories from source'}.items():
            if opt.startswith(word):
                yield Completion(opt, start_position=-len(word), display_meta=desc)


def complete_help_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /help with categories and command names."""
    from episodic.commands.registry import command_registry
    if len(parts) == 2:
        yield from _yield_options({
            'chat': 'Mode switching and conversation', 'voice': 'Voice mode, STT/TTS providers',
            'assistant': 'Timers, alarms, weather, news', 'settings': 'Configuration and system',
            'search': 'Knowledge base and muse', 'history': 'Navigation and history',
            'topics': 'Topic detection and management', 'markdown': 'Markdown file operations',
            'calendar': 'Calendar and email commands', 'all': 'Show all available commands',
        }, word)
        for cmd in sorted(command_registry._commands.keys()):
            if cmd.startswith(word.lower()):
                info = command_registry._commands[cmd]
                yield Completion(
                    cmd, start_position=-len(word),
                    display_meta=info.description[:40] if info.description else ''
                )


def complete_docs_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /docs subcommands."""
    if len(parts) == 2:
        descriptions = {
            'list': 'List all documents', 'show': 'Show document content',
            'remove': 'Remove a document', 'rm': 'Remove a document (alias)',
            'clear': 'Remove all documents'
        }
        for action in DOCS_ACTIONS:
            if action.startswith(word.lower()):
                yield Completion(
                    action, start_position=-len(word),
                    display_meta=descriptions.get(action, '')
                )


def complete_prompt_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /prompt subcommands and prompt names."""
    if len(parts) == 2:
        descriptions = {'list': 'List available prompts', 'use': 'Use a prompt', 'show': 'Show prompt content'}
        for action in PROMPT_ACTIONS:
            if action.startswith(word.lower()):
                yield Completion(
                    action, start_position=-len(word), display_meta=descriptions.get(action, '')
                )
        yield from _complete_prompt_names(word)
    elif len(parts) == 3 and parts[1] in ['use', 'show']:
        yield from _complete_prompt_names(word)


def _complete_prompt_names(word: str) -> List[Completion]:
    """Complete prompt names from the prompt manager."""
    try:
        from episodic.prompt_manager import get_prompt_manager
        pm = get_prompt_manager()
        for name in pm.list_prompts():
            if name.startswith(word.lower()):
                yield Completion(name, start_position=-len(word), display_meta='prompt')
    except Exception as e:
        debug_print(f"Prompt completion error: {e}", category="cli")


def complete_reset_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /reset with 'all' or parameter names."""
    if len(parts) == 2:
        if 'all'.startswith(word.lower()):
            yield Completion('all', start_position=-len(word), display_meta='Reset all to defaults')
        from episodic.commands.settings_handlers import PARAM_HANDLERS
        for param in PARAM_HANDLERS.keys():
            if param.startswith(word.lower()):
                yield Completion(param, start_position=-len(word), display_meta='parameter')


def complete_dev_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /dev subcommands."""
    if len(parts) == 2:
        descriptions = {'reindex-help': 'Reindex help documentation'}
        for action in DEV_ACTIONS:
            if action.startswith(word.lower()):
                yield Completion(
                    action, start_position=-len(word), display_meta=descriptions.get(action, '')
                )


def complete_migrate_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /migrate subcommands."""
    if len(parts) == 2:
        descriptions = {'run': 'Run migration', 'dry-run': 'Preview what would be migrated',
                        'rollback': 'Rollback migration'}
        for action in MIGRATE_ACTIONS:
            if action.startswith(word.lower()):
                yield Completion(
                    action, start_position=-len(word), display_meta=descriptions.get(action, '')
                )


# =========================================================================
# Utility Command Completions
# =========================================================================

def complete_cancel_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /cancel command arguments."""
    if len(parts) == 2:
        yield from _yield_options({
            'timer': 'Cancel active timer', 'alarm': 'Cancel active alarm'
        }, word)


def complete_dnd_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /dnd command arguments."""
    if len(parts) == 2:
        yield from _yield_options({
            'on': 'Enable do not disturb', 'off': 'Disable do not disturb',
            '30m': 'DND for 30 minutes', '1h': 'DND for 1 hour', '2h': 'DND for 2 hours',
        }, word)


def complete_news_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /news command arguments."""
    if len(parts) == 2:
        yield from _yield_options({
            'general': 'General news', 'technology': 'Tech news', 'business': 'Business news',
            'health': 'Health news', 'science': 'Science news', 'politics': 'Political news',
            'world': 'World news',
        }, word)


def complete_play_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /play command with radio station options."""
    if len(parts) == 2:
        yield from _yield_options({
            'npr': 'NPR News', 'bbc': 'BBC World Service', 'wnyc': 'WNYC New York',
            'wbez': 'WBEZ Chicago', 'kexp': 'KEXP Seattle', 'kusc': 'KUSC Classical',
            'wfmt': 'WFMT Classical Chicago', 'wbgo': 'WBGO Jazz',
            'jazz': 'Jazz music', 'classical': 'Classical music',
        }, word)


def complete_timer_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /timer command with common durations."""
    if len(parts) == 2:
        yield from _yield_options({
            '1m': '1 minute', '2m': '2 minutes', '3m': '3 minutes', '5m': '5 minutes',
            '10m': '10 minutes', '15m': '15 minutes', '20m': '20 minutes',
            '30m': '30 minutes', '45m': '45 minutes', '1h': '1 hour',
        }, word)


def complete_alarm_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /alarm command with common time formats."""
    if len(parts) == 2:
        yield from _yield_options({
            '6am': '6:00 AM', '6:30am': '6:30 AM', '7am': '7:00 AM', '7:30am': '7:30 AM',
            '8am': '8:00 AM', '8:30am': '8:30 AM', '9am': '9:00 AM',
            '12pm': '12:00 PM (noon)', '1pm': '1:00 PM', '6pm': '6:00 PM',
        }, word)


def complete_calendar_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /cal, /calendar, /calendars, /schedule commands."""
    if len(parts) == 2:
        yield from _yield_options({
            'today': 'Today\'s events',
            'tomorrow': 'Tomorrow\'s events',
            'this week': 'This week\'s events',
            'next week': 'Next week\'s events',
        }, word)


def complete_email_command(parts: List[str], word: str) -> List[Completion]:
    """Complete /email, /mail, /inbox, /draft, /reply, /forward commands."""
    if len(parts) == 2:
        yield from _yield_options({
            'unread': 'Unread messages',
            'from': 'Search by sender',
            'about': 'Search by subject',
            'recent': 'Recent messages',
        }, word)
