"""Grammar rules contributed by the gsuite plugin.

Bridge: imports rule builders from grammar_calendar_email.py and
returns them as a list for the plugin framework.
"""

from typing import List

from episodic.utility.voice.grammar_types import GrammarRule
from episodic.utility.voice.grammar_calendar_email import (
    build_calendar_rules,
    build_email_rules,
    extract_calendar_args,
    extract_email_args,
)


def get_grammar_rules() -> List[GrammarRule]:
    """Return all calendar and email grammar rules."""
    return [*build_calendar_rules(), *build_email_rules()]


def get_arg_extractors() -> dict:
    """Return command-prefix -> extractor callable mapping."""
    return {
        "calendar.": extract_calendar_args,
        "email.": extract_email_args,
    }
