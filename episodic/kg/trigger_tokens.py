"""Trigger token lists for deterministic entailment checking.

Each predicate maps to a set of trigger phrases. The validator checks
that at least one trigger appears (case-insensitive) in the assertion span.
"""

TRIGGER_TOKENS: dict[str, list[str]] = {
    'uses': [
        'use', 'using', 'used', 'run', 'running',
        'rely on', 'my setup', 'work with', 'working with',
        'daily', 'regularly',
        'studies', 'studying', 'plays', 'playing',
        'drives', 'driving', 'reads', 'reading',
        'writes', 'writing',
    ],
    'wants': [
        'want', 'need', 'looking for', 'hoping',
        'wish', 'interested in', 'would like', 'plan to',
        'replace', 'upgrade', 'get into',
    ],
    'prefers': [
        'prefer', 'rather', 'instead of', 'better than',
        'favorite', 'go-to', 'over',
    ],
    'role': [
        "i'm a", 'i am a', 'my role', 'i work as',
        'my job', 'my position', 'by profession', 'my background',
        'background is',
    ],
    'has': [
        'have', 'has', 'had', 'own', 'owns', 'owned', 'got',
        'my', "i've got", 'i have', 'we have',
        'keeps', 'raised', 'raising', 'grew', 'growing',
        'built', 'building', 'called',
    ],
    'located_at': [
        'at', 'in', 'from', 'based in', 'located',
        'studies at', 'works at', 'enrolled at', 'lives in',
        'attends', 'attending', 'enrolled', 'lives', 'living',
        'based', 'moved to', 'grew up',
    ],
    'part_of': [
        'part of', 'member of', 'belongs to', 'in', 'on',
        'within', 'works for', 'employed by',
        'works at', 'joined', 'at',
    ],
    'related_to': [
        'wife', 'husband', 'partner', 'daughter', 'son',
        'brother', 'sister', 'friend', 'colleague',
        'mother', 'father', 'parent', 'child', 'married to',
        'my wife', 'my husband', 'my daughter', 'my son',
        'my brother', 'my sister', 'my friend',
        'my mom', 'my dad', 'my partner', 'named',
    ],
    'is_a': [
        'is a', 'is an', 'type of', 'kind of', 'which is',
        "it's a", "it's", "that's", 'called',
    ],
    'powered_by': [
        'runs on', 'powered by', 'uses', 'fueled by',
        'with', 'running',
    ],
}


def check_trigger(predicate: str, span_text: str) -> bool:
    """Return True if span_text contains at least one trigger token
    for the given predicate. Case-insensitive matching.

    The match is substring-based: 'using' matches 'I have been using Vim'.
    """
    triggers = TRIGGER_TOKENS.get(predicate, [])
    text_lower = span_text.lower()
    return any(trigger in text_lower for trigger in triggers)
