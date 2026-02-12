"""Token registry for managing lexer tokens from core and plugins.

Provides collision detection and freeze semantics to ensure token
stability after initialization.
"""

from typing import Dict, FrozenSet, List, Optional, Set

from ._protocol import TokenDefinition


# Core token kinds that plugins must not shadow
_CORE_TOKEN_KINDS: FrozenSet[str] = frozenset({
    # Query markers
    "QUERY",
    # Action verbs
    "ACTION_SET", "ACTION_CANCEL", "ACTION_PLAY", "ACTION_PAUSE",
    "ACTION_RESUME", "ACTION_STOP", "ACTION_SKIP", "ACTION_ADD",
    "ACTION_REMIND", "ACTION_ON", "ACTION_OFF", "ACTION_SEARCH",
    "ACTION_DRAFT", "ACTION_SEND", "ACTION_REPLY", "ACTION_FORWARD",
    "ACTION_RESCHEDULE",
    # Time
    "TIME_UNIT", "DURATION_WORD", "RELATIVE_DAY", "AMPM",
    "TIME_RANGE", "TIME_QUARTER_PAST", "TIME_QUARTER_TO", "TIME_HALF_PAST",
    # Core domain keywords
    "KW_TIMER", "KW_ALARM", "KW_REMIND", "KW_WEATHER",
    "KW_WEATHER_HIGH", "KW_WEATHER_LOW", "KW_NEWS", "KW_NOTE",
    "KW_TIME", "KW_DATE", "KW_RADIO", "KW_MUSIC", "KW_DND",
    # Structural
    "PREP_AT", "PREP_IN", "PREP_FOR", "PREP_ON", "PREP_TO", "PREP_FROM",
    "ARTICLE", "POLITENESS", "PRONOUN", "CONJ", "NAMED",
    "NUMBER", "WORD",
})


class TokenRegistry:
    """Manages token definitions from core and plugins.

    Core tokens are registered first and are immutable. Plugin tokens
    are added later. Collision detection prevents plugins from
    shadowing core words. After freeze(), no further registration
    is allowed.
    """

    def __init__(self) -> None:
        self._word_map: Dict[str, str] = {}       # word -> token_kind
        self._sources: Dict[str, str] = {}         # word -> source name
        self._frozen: bool = False
        self._core_words: Set[str] = set()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register_core(self, word_map: Dict[str, str]) -> None:
        """Register core tokens. Must be called before any plugin tokens."""
        if self._frozen:
            raise RuntimeError("TokenRegistry is frozen")
        for word, kind in word_map.items():
            self._word_map[word] = kind
            self._sources[word] = "__core__"
            self._core_words.add(word)

    def register_plugin(
        self,
        plugin_name: str,
        tokens: List[TokenDefinition],
    ) -> List[str]:
        """Register plugin tokens. Returns list of collision warnings.

        Collisions with core tokens are rejected (word is skipped).
        Collisions between plugins raise an error.
        """
        if self._frozen:
            raise RuntimeError("TokenRegistry is frozen")

        warnings: List[str] = []
        for td in tokens:
            word = td.word.lower()
            if word in self._core_words:
                warnings.append(
                    f"Plugin '{plugin_name}' token '{word}' shadows core "
                    f"token {self._word_map[word]} — skipped"
                )
                continue
            if word in self._word_map:
                existing_source = self._sources[word]
                if existing_source != plugin_name:
                    raise ValueError(
                        f"Token collision: '{word}' registered by "
                        f"'{existing_source}', cannot be re-registered by "
                        f"'{plugin_name}'"
                    )
                # Same plugin re-registering same word: update silently
            self._word_map[word] = td.token_kind
            self._sources[word] = plugin_name
        return warnings

    def freeze(self) -> None:
        """Freeze the registry. No further registration allowed."""
        self._frozen = True

    def get_word_map(self) -> Dict[str, str]:
        """Return the complete word -> token_kind map."""
        return dict(self._word_map)

    def get_plugin_tokens(self, plugin_name: str) -> Dict[str, str]:
        """Return tokens registered by a specific plugin."""
        return {
            word: kind
            for word, kind in self._word_map.items()
            if self._sources.get(word) == plugin_name
        }

    def has_word(self, word: str) -> bool:
        return word.lower() in self._word_map

    def source_of(self, word: str) -> Optional[str]:
        return self._sources.get(word.lower())
