"""
Text Normalizer for Voice Grammar.

Prepares input text for lexer by:
1. Expanding contractions (boundary-aware)
2. Lowercasing
3. Stripping edge fillers
4. Normalizing punctuation
5. Joining letter sequences ("n p r" → "npr")
6. Converting number words ("twenty five" → "25")
"""

import re
from typing import List, Optional, Set


class NumericNormalizer:
    """
    Converts word numbers to digits.

    Supports:
        "twenty five" → "25"
        "one hundred" → "100"
        "one hundred and five" → "105"
        "two hundred thirty four" → "234"
        "one thousand five hundred" → "1500"
        "zero" → "0"

    Rejects (leaves unchanged):
        "thirty forty" (consecutive TENS)
        "five six" (consecutive UNITS without scale)
    """

    UNITS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19
    }

    TENS = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    SCALES = {"hundred": 100, "thousand": 1000}
    SKIP_TOKENS: Set[str] = {"and"}  # Allowed inside number sequences
    ALL_NUMBER_WORDS: Set[str] = set(UNITS.keys()) | set(TENS.keys()) | set(SCALES.keys()) | SKIP_TOKENS

    def normalize(self, text: str) -> str:
        """Replace word number sequences with digit equivalents."""
        tokens = text.split()
        result: List[str] = []
        i = 0

        while i < len(tokens):
            token_lower = tokens[i].lower()

            if token_lower in self.ALL_NUMBER_WORDS and token_lower not in self.SKIP_TOKENS:
                # Collect consecutive number words (including "and" as skip)
                num_tokens: List[str] = []
                j = i
                while j < len(tokens) and tokens[j].lower() in self.ALL_NUMBER_WORDS:
                    if tokens[j].lower() not in self.SKIP_TOKENS:
                        num_tokens.append(tokens[j].lower())
                    j += 1

                if not self._is_malformed(num_tokens):
                    value = self._parse_number_sequence(num_tokens)
                    if value is not None:
                        result.append(str(value))
                        i = j
                        continue

                # Malformed or parse failed: keep original
                result.append(tokens[i])
                i += 1
            else:
                result.append(tokens[i])
                i += 1

        return " ".join(result)

    def _is_malformed(self, tokens: List[str]) -> bool:
        """Reject clearly malformed sequences."""
        for k in range(len(tokens) - 1):
            curr = tokens[k]
            next_tok = tokens[k + 1]

            # Two consecutive TENS: "thirty forty"
            if curr in self.TENS and next_tok in self.TENS:
                return True

            # Two consecutive UNITS (not separated by scale): "five six"
            if curr in self.UNITS and next_tok in self.UNITS:
                return True

        return False

    def _parse_number_sequence(self, tokens: List[str]) -> Optional[int]:
        """Parse validated sequence."""
        if not tokens:
            return None

        total = 0
        current = 0

        for token in tokens:
            if token in self.UNITS:
                current += self.UNITS[token]
            elif token in self.TENS:
                current += self.TENS[token]
            elif token == "hundred":
                if current == 0:
                    current = 1
                current *= 100
            elif token == "thousand":
                if current == 0:
                    current = 1
                current *= 1000
                total += current
                current = 0

        total += current
        return total  # Returns 0 for "zero", not None


class LetterSequenceNormalizer:
    """
    Join single-letter runs that form known sequences.
    "n p r" → "npr"
    "b b c" → "bbc"
    """

    KNOWN_SEQUENCES: Set[str] = {"npr", "bbc", "wbez", "wfmt", "kexp", "kusc", "wbgo", "wnyc"}

    def normalize(self, text: str) -> str:
        tokens = text.split()
        result: List[str] = []
        i = 0

        while i < len(tokens):
            if len(tokens[i]) == 1 and tokens[i].isalpha():
                # Collect consecutive single letters
                letters = [tokens[i]]
                j = i + 1
                while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                    letters.append(tokens[j])
                    j += 1

                joined = "".join(letters).lower()
                if joined in self.KNOWN_SEQUENCES:
                    result.append(joined)
                    i = j
                    continue

            result.append(tokens[i])
            i += 1

        return " ".join(result)


class Normalizer:
    """
    Main normalizer. Order:
    1. Expand contractions (boundary-aware, before lowercase)
    2. Lowercase
    3. Strip edge fillers (loop until stable)
    4. Normalize punctuation
    5. Join letter sequences ("n p r" → "npr")
    6. Normalize numbers ("twenty five" → "25")
    """

    # Contractions as (pattern, replacement) tuples
    CONTRACTIONS = [
        (r"\bwhat's\b", "what is"),
        (r"\bWhat's\b", "what is"),
        (r"\bit's\b", "it is"),
        (r"\bIt's\b", "it is"),
        (r"\bthat's\b", "that is"),
        (r"\bThat's\b", "that is"),
        (r"\bdon't\b", "do not"),
        (r"\bDon't\b", "do not"),
        (r"\bcan't\b", "cannot"),
        (r"\bCan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"\bWon't\b", "will not"),
        (r"\bi'm\b", "i am"),
        (r"\bI'm\b", "i am"),
        (r"\bi'll\b", "i will"),
        (r"\bI'll\b", "i will"),
        (r"\blet's\b", "let us"),
        (r"\bLet's\b", "let us"),
        (r"\bdidn't\b", "did not"),
        (r"\bDidn't\b", "did not"),
        (r"\bdoesn't\b", "does not"),
        (r"\bDoesn't\b", "does not"),
        (r"\bisn't\b", "is not"),
        (r"\bIsn't\b", "is not"),
        (r"\baren't\b", "are not"),
        (r"\bAren't\b", "are not"),
        (r"\bwasn't\b", "was not"),
        (r"\bWasn't\b", "was not"),
        (r"\bweren't\b", "were not"),
        (r"\bWeren't\b", "were not"),
        (r"\bhow's\b", "how is"),
        (r"\bHow's\b", "how is"),
        (r"\bwhere's\b", "where is"),
        (r"\bWhere's\b", "where is"),
        (r"\bwho's\b", "who is"),
        (r"\bWho's\b", "who is"),
        (r"\bthere's\b", "there is"),
        (r"\bThere's\b", "there is"),
    ]

    # Fillers to strip from edges
    EDGE_FILLERS = [
        "um", "uh", "er", "ah", "like", "you know",
        "basically", "anyway", "so", "well", "okay", "ok",
        "hey", "hi", "yo"
    ]

    def __init__(self) -> None:
        self.numeric = NumericNormalizer()
        self.letter_seq = LetterSequenceNormalizer()

    def normalize(self, text: str) -> str:
        text = text.strip()
        text = self._expand_contractions(text)
        text = text.lower()
        text = self._strip_edge_fillers(text)
        text = self._normalize_punctuation(text)
        text = self.letter_seq.normalize(text)
        text = self.numeric.normalize(text)
        return text

    def _expand_contractions(self, text: str) -> str:
        """Boundary-aware contraction expansion."""
        for pattern, replacement in self.CONTRACTIONS:
            text = re.sub(pattern, replacement, text)
        return text

    def _strip_edge_fillers(self, text: str) -> str:
        """Strip fillers at edges, loop until stable."""
        changed = True
        while changed:
            changed = False
            text_before = text
            for filler in self.EDGE_FILLERS:
                # Use word boundary at start to strip leading fillers
                pattern = rf"^{re.escape(filler)}\b\s*,?\s*"
                text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
                # Use word boundary at end to strip trailing fillers
                pattern = rf"\s*,?\s*\b{re.escape(filler)}$"
                text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            text = text.strip()
            if text != text_before:
                changed = True

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation for lexer compatibility.
        - Replace dashes/hyphens with spaces
        - Expand dotted acronyms: "n.p.r." → "n p r"
        - Strip sentence-ending punctuation
        - Strip commas
        """
        # Replace dashes/hyphens with spaces
        text = re.sub(r'[-–—]', ' ', text)

        # Expand dotted acronyms: "n.p.r." → "n p r"
        text = re.sub(r'\b([a-z])\.(?=[a-z]\.|\s|$)', r'\1 ', text)

        # Strip sentence-ending punctuation
        text = re.sub(r'[.?!]+$', '', text)

        # Strip commas
        text = text.replace(',', ' ')

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
