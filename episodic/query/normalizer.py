"""
MQL Normalizer

Unicode and whitespace normalization for MQL input.
Returns (s_norm, NormalizationAudit) tuple.

NOTE: Does NOT lowercase - preserves quoted content fidelity.
Case-insensitive matching is handled at lexer via keyword map.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .types import NormalizationAudit


# Unicode punctuation normalization map
PUNCT_MAP = {
    '\u201c': '"',  # Left double quote "
    '\u201d': '"',  # Right double quote "
    '\u2018': "'",  # Left single quote '
    '\u2019': "'",  # Right single quote '
    '\u2014': '-',  # Em dash —
    '\u2013': '-',  # En dash –
    '\u00A0': ' ',  # Non-breaking space
    '\u2002': ' ',  # En space
    '\u2003': ' ',  # Em space
    '\u2009': ' ',  # Thin space
}


def normalize(s_raw: str) -> Tuple[str, NormalizationAudit]:
    """
    Apply Unicode and whitespace normalization.

    Returns (s_norm, audit_record).

    Transformations:
    1. Unicode punctuation normalization (smart quotes, dashes, special spaces)
    2. Whitespace collapse (multiple spaces -> single space)
    3. Trim leading/trailing whitespace

    Does NOT lowercase (preserves quoted content fidelity).
    """
    s = s_raw
    changes: List[str] = []

    # 1. Unicode punctuation normalization
    for old, new in PUNCT_MAP.items():
        if old in s:
            changes.append(f"replaced {repr(old)} with {repr(new)}")
            s = s.replace(old, new)

    # 2. Whitespace normalization (collapse multiple spaces to one)
    s_collapsed = re.sub(r'\s+', ' ', s)
    if s_collapsed != s:
        changes.append("collapsed whitespace")
    s = s_collapsed

    # 3. Trim leading/trailing whitespace
    s_trimmed = s.strip()
    if s_trimmed != s:
        changes.append("trimmed whitespace")
    s = s_trimmed

    return s, NormalizationAudit(
        raw=s_raw,
        normalized=s,
        changes=tuple(changes)
    )
