"""
False Attribution Detection Harness for Episodic.

Category F Implementation:
- Detect attribution claims in model output (prior_convo, memory, tool, temporal)
- Check if claims are supported by context
- Calculate False Attribution Rate (FAR)
- Reporting with evidence pointers

A "false attribution" is when the model claims something was previously said,
stored in memory, agreed upon, or observed, when it was not present in any
allowed sources (recent turns, recalled memory blocks, or explicitly labeled anchors).
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set
from difflib import SequenceMatcher

from episodic.replay import ReplaySnapshot


class ClaimType(Enum):
    """Types of attribution claims."""
    PRIOR_CONVO = "prior_convo"  # "you said earlier", "we agreed", "as we discussed"
    MEMORY = "memory"           # "I have this in memory", "I remember that"
    TOOL = "tool"               # "I looked it up", "the search results show"
    TEMPORAL = "temporal"       # "today it is", "currently", "latest is"


@dataclass
class AttributionClaim:
    """
    An attribution claim extracted from model output.
    """
    type: ClaimType
    span: Tuple[int, int]  # (start, end) indices in original text
    text_snippet: str      # The extracted claim text
    claim_content: str     # The specific content being claimed (what was "said", etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "span": list(self.span),
            "text_snippet": self.text_snippet,
            "claim_content": self.claim_content,
        }


@dataclass
class SupportDecision:
    """
    Decision on whether a claim is supported by context.
    """
    claim: AttributionClaim
    supported: bool
    evidence_block: Optional[str] = None  # Which context block matched
    evidence_snippet: Optional[str] = None  # The matching text
    similarity_score: float = 0.0  # How similar the match is (0-1)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim.to_dict(),
            "supported": self.supported,
            "evidence_block": self.evidence_block,
            "evidence_snippet": self.evidence_snippet,
            "similarity_score": self.similarity_score,
            "reason": self.reason,
        }


@dataclass
class AttributionReport:
    """
    Full attribution analysis report for a response.
    """
    response_text: str
    claims: List[AttributionClaim]
    decisions: List[SupportDecision]
    false_attribution_rate: float  # FAR = unsupported / total
    unsupported_count: int
    total_claims: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_text": self.response_text[:500] + "..." if len(self.response_text) > 500 else self.response_text,
            "claims": [c.to_dict() for c in self.claims],
            "decisions": [d.to_dict() for d in self.decisions],
            "false_attribution_rate": self.false_attribution_rate,
            "unsupported_count": self.unsupported_count,
            "total_claims": self.total_claims,
        }


# =============================================================================
# Attribution Claim Patterns
# =============================================================================

# Prior conversation patterns
PRIOR_CONVO_PATTERNS = [
    # Direct references to past statements
    (r"you\s+(?:said|mentioned|told\s+me|stated|indicated|noted)\s+(?:earlier|before|previously|yesterday|that)", "prior_statement"),
    (r"(?:as\s+)?(?:we|you)\s+(?:discussed|talked\s+about|went\s+over|covered)", "prior_discussion"),
    (r"we\s+(?:agreed|decided|concluded|determined)", "prior_agreement"),
    (r"(?:as\s+)?(?:i\s+)?(?:mentioned|said|noted|stated)\s+(?:earlier|before|previously)", "self_reference"),
    (r"(?:you\s+)?(?:asked|wanted|requested)\s+(?:me\s+)?(?:earlier|before|previously)", "prior_request"),
    (r"last\s+time\s+(?:we|you|i)", "last_time"),
    (r"in\s+(?:our|the)\s+(?:previous|earlier|last)\s+(?:conversation|discussion|session)", "prior_session"),
    (r"(?:you\s+)?(?:already|previously)\s+(?:said|told\s+me|mentioned|indicated)", "already_said"),
    (r"earlier\s+you\s+(?:said|mentioned|told\s+me|stated)", "earlier_you_said"),
]

# Memory claims patterns
MEMORY_PATTERNS = [
    (r"i\s+(?:have|'ve)\s+(?:this|that|it)\s+in\s+(?:my\s+)?memory", "in_memory"),
    (r"i\s+remember\s+(?:that|you|when|how)", "remember"),
    (r"(?:from|in)\s+(?:my|our)\s+(?:memory|records|notes)", "from_memory"),
    (r"i\s+(?:recall|recollect)\s+(?:that|you|when)", "recall"),
    (r"(?:based\s+on|according\s+to)\s+(?:my|our)\s+(?:previous|stored|saved)\s+(?:information|data|memory)", "based_on_memory"),
    (r"i\s+(?:have|'ve)\s+(?:stored|saved|recorded)\s+(?:that|this|it)", "stored"),
]

# Tool observation patterns
TOOL_PATTERNS = [
    (r"i\s+(?:looked\s+(?:it\s+)?up|searched\s+for|found)", "looked_up"),
    (r"(?:the|my)\s+(?:search|lookup)\s+(?:results?|shows?|indicates?)", "search_results"),
    (r"according\s+to\s+(?:the|my)\s+(?:search|lookup|research)", "according_to_search"),
    (r"i\s+(?:checked|verified|confirmed)\s+(?:and|that)", "checked"),
    (r"(?:the|my)\s+(?:tool|function|api)\s+(?:returned|shows?|indicates?)", "tool_returned"),
]

# Temporal claims patterns (when no tool present)
TEMPORAL_PATTERNS = [
    (r"currently,?\s+(?:the\s+)?", "currently"),
    (r"(?:today|right\s+now|at\s+the\s+moment),?\s+(?:it\s+)?(?:is|the)?", "current_state"),
    (r"(?:the\s+)?(?:latest|current|most\s+recent)\s+(?:version|release|update|news|data)", "latest"),
    (r"as\s+of\s+(?:today|now|this\s+moment)", "as_of_now"),
    (r"(?:the\s+)?(?:current|today's)\s+(?:weather|temperature|price|rate|status)", "current_info"),
]


def _compile_patterns(patterns: List[Tuple[str, str]]) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns with case-insensitive flag."""
    return [(re.compile(p, re.IGNORECASE), name) for p, name in patterns]


COMPILED_PRIOR_CONVO = _compile_patterns(PRIOR_CONVO_PATTERNS)
COMPILED_MEMORY = _compile_patterns(MEMORY_PATTERNS)
COMPILED_TOOL = _compile_patterns(TOOL_PATTERNS)
COMPILED_TEMPORAL = _compile_patterns(TEMPORAL_PATTERNS)


def _extract_claim_content(text: str, match: re.Match, window: int = 100) -> str:
    """
    Extract the content being claimed (what follows the attribution phrase).
    """
    end = match.end()
    # Get text after the match up to sentence boundary or window limit
    after = text[end:end + window]

    # Find sentence boundary
    sentence_end = len(after)
    for punct in [". ", "! ", "? ", "\n"]:
        pos = after.find(punct)
        if pos != -1 and pos < sentence_end:
            sentence_end = pos

    claim_content = after[:sentence_end].strip()

    # Clean up leading punctuation/whitespace
    claim_content = re.sub(r'^[,:\s]+', '', claim_content)

    return claim_content


def detect_claims(text: str) -> List[AttributionClaim]:
    """
    Detect attribution claims in model output.

    Scans for patterns indicating the model is making claims about:
    - Prior conversation content
    - Memory/stored information
    - Tool/search results
    - Current temporal information

    Args:
        text: The model's response text

    Returns:
        List of AttributionClaim objects
    """
    claims = []
    seen_spans: Set[Tuple[int, int]] = set()

    def add_claims(patterns: List[Tuple[re.Pattern, str]], claim_type: ClaimType):
        for pattern, _ in patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())

                # Avoid duplicate overlapping claims
                overlap = False
                for seen_start, seen_end in seen_spans:
                    if (span[0] < seen_end and span[1] > seen_start):
                        overlap = True
                        break

                if not overlap:
                    seen_spans.add(span)
                    claim_content = _extract_claim_content(text, match)
                    claims.append(AttributionClaim(
                        type=claim_type,
                        span=span,
                        text_snippet=match.group(),
                        claim_content=claim_content,
                    ))

    add_claims(COMPILED_PRIOR_CONVO, ClaimType.PRIOR_CONVO)
    add_claims(COMPILED_MEMORY, ClaimType.MEMORY)
    add_claims(COMPILED_TOOL, ClaimType.TOOL)
    add_claims(COMPILED_TEMPORAL, ClaimType.TEMPORAL)

    # Sort by position in text
    claims.sort(key=lambda c: c.span[0])

    return claims


# =============================================================================
# Support Checking
# =============================================================================

def _extract_context_text(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract text from context messages by role/type.

    Returns dict mapping block name to text content.
    """
    blocks = {}

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Multimodal content - extract text parts
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        block_name = f"{role}_{i}"

        # Detect special block types
        if role == "system":
            if "## Summary" in content:
                block_name = "summary"
            elif "## Relevant Past Context" in content:
                block_name = "anchors"
            elif "Relevant context from knowledge base" in content:
                block_name = "rag_context"
            elif "[Memory]" in content:
                block_name = "memory_context"
            elif "search results" in content.lower():
                block_name = "web_context"
            else:
                block_name = f"system_{i}"

        blocks[block_name] = content

    return blocks


def _similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_support_in_context(
    claim_content: str,
    context_blocks: Dict[str, str],
    similarity_threshold: float = 0.6
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Search for supporting evidence in context blocks.

    Returns:
        Tuple of (block_name, matching_snippet, similarity_score)
    """
    if not claim_content or len(claim_content.strip()) < 3:
        return None, None, 0.0

    best_match = (None, None, 0.0)

    # Normalize claim content
    claim_lower = claim_content.lower().strip()

    # Extract key words (filter out common words)
    common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "that", "this", "it", "to", "for", "of", "in", "on", "with",
                    "you", "i", "me", "my", "your", "and", "or", "but"}
    claim_key_words = [w for w in claim_lower.split() if w not in common_words and len(w) > 2]

    if not claim_key_words:
        return None, None, 0.0

    for block_name, block_text in context_blocks.items():
        if not block_text or len(block_text.strip()) < 5:
            continue

        block_lower = block_text.lower()

        # Check for exact substring match
        if claim_lower in block_lower:
            # Find the matching portion
            idx = block_lower.find(claim_lower)
            snippet = block_text[idx:idx + len(claim_content)]
            return block_name, snippet, 1.0

        # Check if key words are present
        block_words_set = set(block_lower.split())
        matching_words = sum(1 for w in claim_key_words if w in block_words_set)
        key_word_ratio = matching_words / len(claim_key_words) if claim_key_words else 0

        # Require at least 60% of key words to match
        if key_word_ratio >= 0.6:
            # Check for partial matches using sliding window
            claim_words = claim_lower.split()
            if len(claim_words) >= 3:
                # Try to find runs of matching words
                block_words = block_lower.split()
                for i in range(max(0, len(block_words) - len(claim_words) + 1)):
                    window = " ".join(block_words[i:i + len(claim_words)])
                    sim = _similarity(claim_lower, window)
                    if sim > best_match[2] and sim >= similarity_threshold:
                        # Extract original case snippet
                        try:
                            word_pos = 0
                            for j in range(i):
                                pos = block_text.lower().find(block_words[j], word_pos)
                                if pos >= 0:
                                    word_pos = pos + len(block_words[j])
                            snippet_start = block_text.lower().find(block_words[i], word_pos)
                            if snippet_start >= 0:
                                snippet_end = min(snippet_start + len(window) + 10, len(block_text))
                                snippet = block_text[snippet_start:snippet_end]
                                best_match = (block_name, snippet, sim)
                        except:
                            pass

    return best_match


def check_claim_support(
    claim: AttributionClaim,
    context_blocks: Dict[str, str],
    has_tool_output: bool = False,
    has_web_context: bool = False,
) -> SupportDecision:
    """
    Check if a claim is supported by the context.

    Args:
        claim: The attribution claim to check
        context_blocks: Dict of block_name -> text content
        has_tool_output: Whether tool output is present in context
        has_web_context: Whether web search results are present

    Returns:
        SupportDecision indicating if the claim is supported
    """
    # For temporal claims, require tool/web output
    if claim.type == ClaimType.TEMPORAL:
        if not has_tool_output and not has_web_context:
            return SupportDecision(
                claim=claim,
                supported=False,
                reason="Temporal claim without tool/web context",
            )
        else:
            return SupportDecision(
                claim=claim,
                supported=True,
                evidence_block="web_context" if has_web_context else "tool_output",
                reason="Temporal claim with tool/web context present",
            )

    # For tool claims, require tool output or web context
    if claim.type == ClaimType.TOOL:
        if not has_tool_output and not has_web_context:
            return SupportDecision(
                claim=claim,
                supported=False,
                reason="Tool claim without tool/web context",
            )

    # For prior_convo and memory claims, search for supporting content
    block_name, snippet, sim = _find_support_in_context(
        claim.claim_content,
        context_blocks,
    )

    if block_name and sim >= 0.6:
        return SupportDecision(
            claim=claim,
            supported=True,
            evidence_block=block_name,
            evidence_snippet=snippet,
            similarity_score=sim,
            reason=f"Found matching content in {block_name}",
        )

    # Check if the claim contains uncertainty markers
    uncertainty_markers = [
        "i don't see that",
        "i don't have that",
        "not in the context",
        "i'm not sure if",
        "i don't recall",
        "i can't find",
    ]
    claim_lower = claim.text_snippet.lower() + " " + claim.claim_content.lower()
    for marker in uncertainty_markers:
        if marker in claim_lower:
            return SupportDecision(
                claim=claim,
                supported=True,
                reason="Claim contains uncertainty marker",
            )

    return SupportDecision(
        claim=claim,
        supported=False,
        reason="No supporting evidence found in context",
    )


def analyze_response(
    response_text: str,
    assembled_messages: List[Dict[str, Any]],
    has_tool_output: bool = False,
) -> AttributionReport:
    """
    Analyze a model response for false attributions.

    Args:
        response_text: The model's response
        assembled_messages: The context messages (from ReplaySnapshot)
        has_tool_output: Whether tool output was available

    Returns:
        AttributionReport with claims, decisions, and FAR score
    """
    # Detect claims
    claims = detect_claims(response_text)

    if not claims:
        return AttributionReport(
            response_text=response_text,
            claims=[],
            decisions=[],
            false_attribution_rate=0.0,
            unsupported_count=0,
            total_claims=0,
        )

    # Extract context blocks
    context_blocks = _extract_context_text(assembled_messages)

    # Check for web context
    has_web_context = any("web" in k.lower() or "search" in v.lower()
                          for k, v in context_blocks.items())

    # Check each claim
    decisions = []
    for claim in claims:
        decision = check_claim_support(
            claim,
            context_blocks,
            has_tool_output=has_tool_output,
            has_web_context=has_web_context,
        )
        decisions.append(decision)

    # Calculate FAR
    unsupported_count = sum(1 for d in decisions if not d.supported)
    total_claims = len(claims)
    far = unsupported_count / total_claims if total_claims > 0 else 0.0

    return AttributionReport(
        response_text=response_text,
        claims=claims,
        decisions=decisions,
        false_attribution_rate=far,
        unsupported_count=unsupported_count,
        total_claims=total_claims,
    )


def analyze_snapshot_response(
    snapshot: ReplaySnapshot,
    response_text: str,
) -> AttributionReport:
    """
    Analyze a response in the context of a ReplaySnapshot.

    Args:
        snapshot: The ReplaySnapshot with context
        response_text: The model's response

    Returns:
        AttributionReport
    """
    has_tool_output = bool(snapshot.inputs.web_context)
    return analyze_response(
        response_text=response_text,
        assembled_messages=snapshot.assembled_messages,
        has_tool_output=has_tool_output,
    )


# =============================================================================
# Mitigation Hook
# =============================================================================

ATTRIBUTION_MITIGATION_PROMPT = """
IMPORTANT: Attribution Accuracy Rules

When making claims about prior statements or information:
1. Only use phrases like "you said earlier" or "we discussed" if you can cite the EXACT quoted text from the conversation above
2. If you cannot find the exact text in the provided context, say "I don't see that in the provided context"
3. Never claim to "remember" or have information "in memory" unless it's explicitly present above
4. Do not make claims about current dates, weather, or real-time information unless search results are provided

If asked about something that should be in context but isn't, respond with:
"I don't see that information in our current conversation. Could you remind me of the details?"
"""


def get_mitigation_prompt() -> str:
    """
    Get the prompt addition for attribution accuracy mitigation.

    This can be added to system prompts during testing to verify
    the harness detects improvements.
    """
    return ATTRIBUTION_MITIGATION_PROMPT


def apply_mitigation_to_messages(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply mitigation prompt to a message list.

    Prepends the mitigation instructions to the first system message
    or creates a new system message if none exists.
    """
    if not messages:
        return [{"role": "system", "content": ATTRIBUTION_MITIGATION_PROMPT}]

    result = list(messages)

    # Find first system message
    system_idx = None
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Prepend to existing system message
        result[system_idx] = {
            "role": "system",
            "content": ATTRIBUTION_MITIGATION_PROMPT + "\n\n" + result[system_idx].get("content", "")
        }
    else:
        # Insert new system message at beginning
        result.insert(0, {"role": "system", "content": ATTRIBUTION_MITIGATION_PROMPT})

    return result
