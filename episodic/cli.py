"""
Streamlined CLI module for Episodic.

This module serves as a thin wrapper that imports functionality from
specialized CLI modules for backward compatibility.
"""

import os
# Prevent tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Import main components for backward compatibility
from episodic.cli_main import (
    app,
    main,
    talk_loop,
    handle_chat_message,
    conversation_manager
)

from episodic.cli_command_router import handle_command

from episodic.cli_session import (
    save_session_script,
    execute_script,
    save_to_history,
    session_commands
)

from episodic.cli_display import (
    setup_environment,
    display_welcome,
    display_model_info,
    get_prompt
)

# Re-export everything for backward compatibility
__all__ = [
    'app',
    'main',
    'talk_loop',
    'handle_chat_message',
    'handle_command',
    'save_session_script',
    'execute_script',
    'save_to_history',
    'setup_environment',
    'display_welcome',
    'display_model_info',
    'get_prompt',
    'conversation_manager',
    'session_commands'
]

# For direct execution
if __name__ == "__main__":
    app()