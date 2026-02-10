"""
Fuzzy Matcher for Voice Grammar.

Opt-in fuzzy matching with whitelisted corrections.
Pre-normalize letter sequences in Normalizer, then match single tokens.
All matches are logged for debugging.
"""

from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime


class FuzzyMatcher:
    """
    Opt-in fuzzy matching.
    Pre-normalize letter sequences in Normalizer, then match single tokens.
    """

    FUZZY_WHITELIST: Dict[str, Dict[str, List[str]]] = {
        "station_name": {
            "npr": ["inpr", "andpr", "mpr", "enpr"],
            "wbez": ["wbz", "wbes"],
            "bbc": ["bbc4"],
            "kexp": ["kex", "kxp"],
        },
        "domain_keyword": {
            "timer": ["timor"],
        },
        "time_unit": {
            "minutes": ["minits", "minuts", "minuets"],
            "seconds": ["secunds", "secondes"],
            "hours": ["ours", "houers"],
        }
    }

    # Context-dependent fuzzy
    CONTEXT_FUZZY: Dict[str, Dict] = {
        "weather": {
            "variants": ["whether"],
            "required_context": ["forecast", "temperature", "rain", "umbrella", "cold", "hot", "sunny"]
        }
    }

    # Never fuzzy match these
    FUZZY_BLACKLIST: Set[str] = {
        "set", "cancel", "stop", "delete", "remove", "play", "pause", "create", "add"
    }

    def __init__(self) -> None:
        self.log: List[Dict] = []

    def try_fuzzy_match(
        self,
        token: str,
        token_class: str,
        context_tokens: List[str]
    ) -> Tuple[Optional[str], bool]:
        """
        Attempt fuzzy match on a single token.
        Returns (corrected_token or None, fuzzy_was_used).
        """
        token_lower = token.lower()

        if token_lower in self.FUZZY_BLACKLIST:
            return (None, False)

        # Context-dependent fuzzy
        for canonical, config in self.CONTEXT_FUZZY.items():
            if token_lower in config["variants"]:
                context_lower = [t.lower() for t in context_tokens]
                if any(ctx in context_lower for ctx in config["required_context"]):
                    self._log_match(token, canonical, token_class, "context_fuzzy")
                    return (canonical, True)

        # Standard whitelist
        if token_class in self.FUZZY_WHITELIST:
            for canonical, variants in self.FUZZY_WHITELIST[token_class].items():
                if token_lower in variants:
                    self._log_match(token, canonical, token_class, "whitelist")
                    return (canonical, True)

        return (None, False)

    def _log_match(self, original: str, corrected: str, token_class: str, match_type: str) -> None:
        """Log a fuzzy match for debugging."""
        self.log.append({
            "original": original,
            "corrected": corrected,
            "class": token_class,
            "type": match_type,
            "timestamp": datetime.now().isoformat()
        })

    def get_log(self) -> List[Dict]:
        """Get the fuzzy match log."""
        return self.log.copy()

    def clear_log(self) -> None:
        """Clear the fuzzy match log."""
        self.log.clear()
