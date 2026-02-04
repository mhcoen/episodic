"""
Grammar Parser for Voice Grammar.

Pattern matching engine that converts token streams into UtilityQuery candidates.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Set

from .lexer import Token, Lexer
from .label_extractor import LabelExtractor
from .time_normalizer import TimeNormalizer
from .confidence import ParseFeatures


@dataclass
class GrammarMatch:
    """Result of a grammar rule match."""
    command: str
    category: str
    args: Dict[str, Any]
    features: ParseFeatures
    consumed_tokens: int


@dataclass
class GrammarRule:
    """A grammar rule definition."""
    name: str
    category: str
    command: str
    patterns: List[List[str]]  # List of token kind patterns
    required_args: List[str]
    optional_args: List[str]
    is_exact_template: bool = False


class GrammarParser:
    """
    Grammar parser that matches token streams against rules.
    """

    def __init__(self) -> None:
        self.lexer = Lexer()
        self.label_extractor = LabelExtractor()
        self.time_normalizer = TimeNormalizer()
        self.rules = self._build_rules()

    def _build_rules(self) -> List[GrammarRule]:
        """Build the grammar rules."""
        return [
            # Time queries
            GrammarRule(
                name="time_query",
                category="time",
                command="time_now",
                patterns=[
                    ["QUERY", "KW_TIME"],                    # "what time"
                    ["QUERY", "KW_TIME", "PREP_AT"],         # "what time is it"
                    ["KW_TIME"],                             # "time" (single word handled elsewhere)
                ],
                required_args=[],
                optional_args=[],
                is_exact_template=True,
            ),
            GrammarRule(
                name="date_query",
                category="time",
                command="date_today",
                patterns=[
                    ["QUERY", "KW_DATE"],                    # "what date"
                    ["QUERY", "KW_DATE", "PRONOUN"],         # "what day is it"
                    ["RELATIVE_DAY"],                        # "today"
                ],
                required_args=[],
                optional_args=[],
                is_exact_template=True,
            ),

            # Timer commands
            GrammarRule(
                name="timer_set",
                category="timer",
                command="timer_set",
                patterns=[
                    ["ACTION_SET", "ARTICLE", "KW_TIMER", "PREP_FOR"],  # "set a timer for"
                    ["ACTION_SET", "KW_TIMER", "PREP_FOR"],             # "set timer for"
                    ["KW_TIMER", "PREP_FOR"],                           # "timer for"
                    ["KW_TIMER", "NUMBER"],                             # "timer 10"
                    ["NUMBER", "TIME_UNIT", "KW_TIMER"],                # "10 minute timer"
                ],
                required_args=["duration_s"],
                optional_args=["label"],
                is_exact_template=False,
            ),
            GrammarRule(
                name="timer_cancel",
                category="timer",
                command="timer_cancel",
                patterns=[
                    ["ACTION_CANCEL", "KW_TIMER"],           # "cancel timer"
                    ["ACTION_CANCEL", "ARTICLE", "KW_TIMER"],  # "cancel the timer"
                ],
                required_args=[],
                optional_args=["label"],
                is_exact_template=True,
            ),

            # Alarm commands
            GrammarRule(
                name="alarm_set",
                category="alarm",
                command="alarm_set",
                patterns=[
                    ["ACTION_SET", "ARTICLE", "KW_ALARM", "PREP_FOR"],  # "set an alarm for"
                    ["ACTION_SET", "ARTICLE", "KW_ALARM", "PREP_AT"],   # "set an alarm at"
                    ["ACTION_SET", "KW_ALARM", "PREP_FOR"],             # "set alarm for"
                    ["ACTION_SET", "KW_ALARM", "PREP_AT"],              # "set alarm at"
                    ["KW_ALARM", "PREP_FOR"],                           # "alarm for"
                    ["KW_ALARM", "PREP_AT"],                            # "alarm at"
                    ["KW_ALARM", "NUMBER"],                             # "wake me up"
                ],
                required_args=["time"],
                optional_args=["label"],
                is_exact_template=False,
            ),
            GrammarRule(
                name="alarm_cancel",
                category="alarm",
                command="alarm_cancel",
                patterns=[
                    ["ACTION_CANCEL", "KW_ALARM"],           # "cancel alarm"
                    ["ACTION_CANCEL", "ARTICLE", "KW_ALARM"],  # "cancel the alarm"
                ],
                required_args=[],
                optional_args=["label"],
                is_exact_template=True,
            ),

            # Weather queries
            GrammarRule(
                name="weather_query",
                category="weather",
                command="weather_now",
                patterns=[
                    ["QUERY", "KW_WEATHER"],                 # "what weather" / "how is the weather"
                    ["KW_WEATHER"],                          # "weather"
                ],
                required_args=[],
                optional_args=["place"],
                is_exact_template=True,
            ),
            GrammarRule(
                name="forecast_query",
                category="weather",
                command="weather_forecast",
                patterns=[
                    ["QUERY", "KW_WEATHER"],                 # "what forecast"
                ],
                required_args=[],
                optional_args=["place"],
                is_exact_template=True,
            ),

            # News queries
            GrammarRule(
                name="news_query",
                category="news",
                command="news_headlines",
                patterns=[
                    ["QUERY", "KW_NEWS"],                    # "what news"
                    ["KW_NEWS"],                             # "news"
                ],
                required_args=[],
                optional_args=["category"],
                is_exact_template=True,
            ),

            # Media commands
            GrammarRule(
                name="media_play",
                category="media",
                command="media_play",
                patterns=[
                    ["ACTION_PLAY"],                         # "play X"
                    ["ACTION_ON", "KW_RADIO"],               # "turn on radio" / "put on the radio"
                ],
                required_args=["query"],
                optional_args=[],
                is_exact_template=False,
            ),
            GrammarRule(
                name="media_stop",
                category="media",
                command="media_stop",
                patterns=[
                    ["ACTION_STOP", "KW_RADIO"],             # "stop radio"
                    ["ACTION_STOP", "KW_MUSIC"],             # "stop music"
                ],
                required_args=[],
                optional_args=[],
                is_exact_template=True,
            ),
            GrammarRule(
                name="media_pause",
                category="media",
                command="media_pause",
                patterns=[
                    ["ACTION_PAUSE"],                        # "pause"
                    ["ACTION_PAUSE", "KW_RADIO"],            # "pause radio"
                    ["ACTION_PAUSE", "KW_MUSIC"],            # "pause music"
                ],
                required_args=[],
                optional_args=[],
                is_exact_template=True,
            ),

            # Reminder commands
            GrammarRule(
                name="remind_set",
                category="reminder",
                command="remind_set",
                patterns=[
                    ["ACTION_REMIND", "PRONOUN"],            # "remind me"
                    ["ACTION_SET", "ARTICLE", "KW_REMIND"],  # "set a reminder"
                ],
                required_args=["text"],
                optional_args=["minutes", "at_time"],
                is_exact_template=False,
            ),

            # Note commands
            GrammarRule(
                name="note_add",
                category="note",
                command="note_add",
                patterns=[
                    ["ACTION_ADD", "ARTICLE", "KW_NOTE"],    # "add a note"
                    ["KW_NOTE"],                             # "note X"
                ],
                required_args=["text"],
                optional_args=[],
                is_exact_template=False,
            ),

        ]

    def parse(self, text: str, tokens: List[Token]) -> Optional[GrammarMatch]:
        """
        Parse tokens against grammar rules.
        Returns the best matching rule or None.
        """
        words = text.split()
        token_kinds = [t.kind for t in tokens]

        best_match: Optional[GrammarMatch] = None
        best_consumed = 0

        for rule in self.rules:
            match = self._try_rule(rule, tokens, token_kinds, words)
            if match and match.consumed_tokens > best_consumed:
                best_match = match
                best_consumed = match.consumed_tokens

        return best_match

    def _try_rule(
        self,
        rule: GrammarRule,
        tokens: List[Token],
        token_kinds: List[str],
        words: List[str]
    ) -> Optional[GrammarMatch]:
        """Try to match a single rule against tokens."""
        for pattern in rule.patterns:
            match_result = self._match_pattern(pattern, token_kinds)
            if match_result is not None:
                consumed, matched_positions = match_result

                # Extract arguments
                args, features = self._extract_args(
                    rule, tokens, words, consumed
                )

                # Check required args
                args_complete = all(
                    arg in args and args[arg] is not None
                    for arg in rule.required_args
                )
                args_partial = any(
                    arg in args and args[arg] is not None
                    for arg in rule.required_args
                )

                features.args_complete = args_complete
                features.args_partial = args_partial
                features.is_exact_template = rule.is_exact_template and args_complete

                # Check for domain keyword
                features.has_domain_keyword = self._has_domain_keyword(token_kinds)

                # Check for action/query markers
                features.has_action_marker = any(
                    k.startswith("ACTION_") for k in token_kinds[:consumed]
                )
                features.has_query_marker = any(
                    k == "QUERY" for k in token_kinds[:consumed]
                )

                features.word_count = len(words)

                return GrammarMatch(
                    command=rule.command,
                    category=rule.category,
                    args=args,
                    features=features,
                    consumed_tokens=consumed,
                )

        return None

    def _match_pattern(
        self,
        pattern: List[str],
        token_kinds: List[str]
    ) -> Optional[Tuple[int, List[int]]]:
        """
        Match a pattern against token kinds.
        Returns (consumed_count, matched_positions) or None.
        """
        if len(pattern) > len(token_kinds):
            return None

        matched_positions = []
        pattern_idx = 0
        token_idx = 0

        while pattern_idx < len(pattern) and token_idx < len(token_kinds):
            expected = pattern[pattern_idx]
            actual = token_kinds[token_idx]

            if actual == expected:
                matched_positions.append(token_idx)
                pattern_idx += 1
                token_idx += 1
            elif actual in ("ARTICLE", "POLITENESS", "PRONOUN"):
                # Skip filler tokens
                token_idx += 1
            else:
                # No match
                return None

        if pattern_idx == len(pattern):
            return (token_idx, matched_positions)
        return None

    def _extract_args(
        self,
        rule: GrammarRule,
        tokens: List[Token],
        words: List[str],
        consumed: int
    ) -> Tuple[Dict[str, Any], ParseFeatures]:
        """Extract arguments from tokens."""
        args: Dict[str, Any] = {}
        features = ParseFeatures()

        # For timer/alarm, we need to scan ALL tokens for duration/time
        # because the pattern may have consumed the NUMBER token
        remaining_tokens = tokens[consumed:]
        remaining_words = words[consumed:] if consumed < len(words) else []

        # Extract based on rule type
        if rule.command == "timer_set":
            # Pass all tokens for duration extraction
            args.update(self._extract_timer_args(tokens, words, rule.command))
        elif rule.command == "alarm_set":
            # Pass all tokens for time extraction
            args.update(self._extract_alarm_args(tokens, words, rule.command))
        elif rule.command == "media_play":
            args.update(self._extract_media_args(remaining_tokens, remaining_words, rule.command))
        elif rule.command == "remind_set":
            args.update(self._extract_reminder_args(remaining_tokens, remaining_words, rule.command))
        elif rule.command == "note_add":
            args.update(self._extract_note_args(remaining_tokens, remaining_words, rule.command))

        return args, features

    def _extract_timer_args(
        self,
        tokens: List[Token],
        words: List[str],
        rule: str
    ) -> Dict[str, Any]:
        """Extract timer arguments (duration, label)."""
        args: Dict[str, Any] = {"duration_s": None, "label": None}

        if not words:
            return args

        # Look for duration pattern: NUMBER TIME_UNIT
        duration_parts = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.kind == "NUMBER":
                duration_parts.append(token.value)
                if i + 1 < len(tokens) and tokens[i + 1].kind == "TIME_UNIT":
                    duration_parts.append(tokens[i + 1].value)
                    i += 2
                    continue
            elif token.kind == "DURATION_WORD":
                duration_parts.append(token.value)
            elif token.kind == "TIME_UNIT":
                duration_parts.append(token.value)
            i += 1

        if duration_parts:
            duration_str = " ".join(duration_parts)
            duration_s = self.time_normalizer.normalize_duration(duration_str)
            if duration_s:
                args["duration_s"] = duration_s

        # Look for label after "called" or "named"
        for i, token in enumerate(tokens):
            if token.kind == "NAMED" and i + 1 < len(tokens):
                label_words = [t.value for t in tokens[i + 1:]]
                if label_words:
                    args["label"] = " ".join(label_words)
                break

        return args

    def _extract_alarm_args(
        self,
        tokens: List[Token],
        words: List[str],
        rule: str
    ) -> Dict[str, Any]:
        """Extract alarm arguments (time, label)."""
        args: Dict[str, Any] = {"time": None, "label": None}

        if not words:
            return args

        # Look for time pattern
        time_parts = []
        for token in tokens:
            if token.kind in ("NUMBER", "AMPM", "TIME_QUARTER_PAST", "TIME_QUARTER_TO", "TIME_HALF_PAST"):
                time_parts.append(token.value)
            elif token.kind == "RELATIVE_DAY":
                time_parts.append(token.value)

        if time_parts:
            args["time"] = " ".join(time_parts)

        return args

    def _extract_media_args(
        self,
        tokens: List[Token],
        words: List[str],
        rule: str
    ) -> Dict[str, Any]:
        """Extract media arguments (query)."""
        args: Dict[str, Any] = {"query": None}

        if words:
            # Use label extractor to get the media query
            label, _ = self.label_extractor.extract(words, 0, rule)
            if label.text:
                args["query"] = label.text

        return args

    def _extract_reminder_args(
        self,
        tokens: List[Token],
        words: List[str],
        rule: str
    ) -> Dict[str, Any]:
        """Extract reminder arguments (text, time)."""
        args: Dict[str, Any] = {"text": None, "minutes": None, "at_time": None}

        if not words:
            return args

        # Look for "to X" pattern
        text_parts = []
        time_parts = []
        in_text = False
        in_time = False

        for i, token in enumerate(tokens):
            if token.kind == "PREP_TO" and not in_text:
                in_text = True
                continue
            elif token.kind == "PREP_IN" and in_text:
                in_text = False
                in_time = True
                continue
            elif token.kind == "PREP_AT" and in_text:
                in_text = False
                in_time = True
                continue

            if in_text:
                text_parts.append(token.value)
            elif in_time:
                time_parts.append(token.value)

        if text_parts:
            args["text"] = " ".join(text_parts)

        if time_parts:
            time_str = " ".join(time_parts)
            duration = self.time_normalizer.normalize_duration(time_str)
            if duration:
                args["minutes"] = duration // 60

        return args

    def _extract_note_args(
        self,
        tokens: List[Token],
        words: List[str],
        rule: str
    ) -> Dict[str, Any]:
        """Extract note arguments (text)."""
        args: Dict[str, Any] = {"text": None}

        if words:
            label, _ = self.label_extractor.extract(words, 0, rule)
            if label.text:
                args["text"] = label.text

        return args

    def _has_domain_keyword(self, token_kinds: List[str]) -> bool:
        """Check if tokens contain a domain keyword."""
        domain_keywords = {
            "KW_TIMER", "KW_ALARM", "KW_REMIND", "KW_WEATHER",
            "KW_NEWS", "KW_NOTE", "KW_TIME", "KW_DATE",
            "KW_RADIO", "KW_MUSIC", "KW_DND",
        }
        return any(k in domain_keywords for k in token_kinds)
