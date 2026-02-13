"""Keyword gate for MCP intent extraction.

Computes matched_domains(utterance) -> Set[str]. If the set is empty,
no extraction LLM call is made and the utterance goes directly to chat.
"""

import re
import unicodedata
from typing import Dict, FrozenSet, List, Set, Tuple


# --- Domain keyword and phrase definitions ---

CALENDAR_KEYWORDS: FrozenSet[str] = frozenset({
    "calendar", "schedule", "meeting", "appointment", "free", "busy",
    "event", "agenda", "book", "reserve",
})

CALENDAR_PHRASES: List[List[str]] = [
    ["set", "up", "a", "meeting"],
    ["am", "i", "free"],
    ["what", "on", "my"],
]

EMAIL_KEYWORDS: FrozenSet[str] = frozenset({
    "email", "mail", "inbox", "unread", "draft", "reply",
    "forward", "send", "message", "sent",
})

EMAIL_PHRASES: List[List[str]] = [
    ["follow", "up"],
    ["get", "back", "to"],
    ["check", "my", "mail"],
]

# Core domain registry: domain name -> (keywords, phrases)
_CORE_DOMAIN_REGISTRY: Dict[str, Tuple[FrozenSet[str], List[List[str]]]] = {
    "calendar": (CALENDAR_KEYWORDS, CALENDAR_PHRASES),
    "email": (EMAIL_KEYWORDS, EMAIL_PHRASES),
}

# Live registry, extended by plugins via register_plugin_domains()
DOMAIN_REGISTRY: Dict[str, Tuple[FrozenSet[str], List[List[str]]]] = dict(
    _CORE_DOMAIN_REGISTRY
)

_plugins_registered = False


def register_plugin_domains() -> None:
    """Extend DOMAIN_REGISTRY with new domains from plugin extraction contributions.

    Only adds domains that are NOT already in the core registry.
    Core domains (calendar, email) are already correctly defined and
    should not be modified by plugin keywords (which may be a flat
    merged list across multiple domains).

    Idempotent — only runs once.
    """
    global _plugins_registered
    if _plugins_registered:
        return
    _plugins_registered = True

    try:
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        for reg in registry.registered():
            contrib = reg.extraction_contribution
            if contrib is None:
                continue
            # Infer domains from intent names (e.g. "calendar.query" → "calendar")
            for intent in contrib.intents:
                if "." in intent.name:
                    domain = intent.name.split(".")[0]
                    if domain not in DOMAIN_REGISTRY:
                        # New domain from plugin — add with all plugin keywords
                        DOMAIN_REGISTRY[domain] = (
                            frozenset(contrib.gate_keywords),
                            list(contrib.gate_phrases),
                        )
    except ImportError:
        pass

# --- Plural/singular suffix normalization ---

# Minimal suffix rules for domain vocabulary. NOT a stemmer.
_SUFFIX_RULES: List[Tuple[str, str]] = [
    ("ies", "y"),       # e.g. entries -> entry (not needed now but cheap)
    ("sses", "ss"),     # e.g. addresses -> address... actually keep "es" rule
    ("es", "e"),        # e.g. schedules -> schedule, reserves -> reserve
    ("s", ""),          # e.g. appointments -> appointment, emails -> email
]

# Words where naive suffix stripping gives wrong results
_SUFFIX_EXCEPTIONS: FrozenSet[str] = frozenset({
    "bus", "busy", "free", "this", "his", "is", "was", "has",
    "yes", "us", "as", "plus",
})

# Speech-to-text filler words to strip
_FILLERS: FrozenSet[str] = frozenset({
    "umm", "uhh", "uh", "um", "hmm", "ah", "er", "erm", "like",
})

# Punctuation stripping pattern
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_token(token: str) -> str:
    """Apply plural/singular normalization to a single token."""
    if token in _SUFFIX_EXCEPTIONS or len(token) <= 2:
        return token
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[: -len(suffix)] + replacement
    return token


def _tokenize(utterance: str) -> List[str]:
    """Tokenize an utterance for gate matching.

    Pipeline: lowercase -> NFKC normalize -> strip punctuation ->
    split whitespace -> remove fillers -> plural/singular normalize.
    """
    text = utterance.lower()
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub(" ", text)
    raw_tokens = text.split()
    tokens = [t for t in raw_tokens if t not in _FILLERS]
    return [_normalize_token(t) for t in tokens]


def _check_phrase(tokens: List[str], phrase_tokens: List[str], window: int = 5) -> bool:
    """Check if phrase tokens appear in order within a sliding window.

    The phrase matches if all phrase tokens are found in order within
    any window of `window` consecutive tokens in the input.
    """
    if not phrase_tokens:
        return False
    n = len(tokens)
    plen = len(phrase_tokens)
    for start in range(n):
        end = min(start + window, n)
        window_tokens = tokens[start:end]
        pidx = 0
        for t in window_tokens:
            if t == phrase_tokens[pidx]:
                pidx += 1
                if pidx == plen:
                    return True
        # Also try with normalized phrase tokens against raw
    return False


def matched_domains(utterance: str) -> Set[str]:
    """Compute which domains' keywords/phrases appear in the utterance.

    Returns a set of domain names (e.g. {"calendar", "email"}).
    Empty set means no extraction call needed.
    """
    register_plugin_domains()

    if not utterance or not utterance.strip():
        return set()

    tokens = _tokenize(utterance)
    if not tokens:
        return set()

    token_set = set(tokens)
    result: Set[str] = set()

    for domain, (keywords, phrases) in DOMAIN_REGISTRY.items():
        # Keyword match: any token in the keyword set
        if token_set & keywords:
            result.add(domain)
            continue
        # Phrase match: any phrase found in token sequence
        for phrase_tokens in phrases:
            if _check_phrase(tokens, phrase_tokens):
                result.add(domain)
                break

    return result
