"""
LLM topic-coverage verifier with batched calls and quote validation.

Design:
- K=50 initial candidates (no threshold)
- Batch size B=10 (so 5 batches max, but usually 1)
- Accept target A=3 verified hits (stop once you have A)
- Max batches per query = 2 (so ≤2 LLM calls/query)
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
CACHE_PATH = EXPERIMENT_DIR / "verifier_cache.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"


class Relation(Enum):
    SAME = "SAME"
    SUBSUMES = "SUBSUMES"
    SUBSUMED_BY = "SUBSUMED_BY"
    OVERLAP = "OVERLAP"
    UNRELATED = "UNRELATED"


@dataclass
class Quote:
    text: str
    start: int = -1  # Will be populated by validation
    end: int = -1


@dataclass
class VerifierResult:
    statement_id: int
    relation: Relation
    confidence: float
    quotes: list[Quote]
    rationale: str
    quote_check_passed: bool = False
    quote_check_errors: list[str] = field(default_factory=list)
    ambiguous_quotes: list[int] = field(default_factory=list)  # Indices of ambiguous quotes


# =============================================================================
# VERIFIER PROMPT (fixed, versioned)
# =============================================================================

VERIFIER_PROMPT = """You are a topic-coverage classifier. Given a QUERY TOPIC and a list of CANDIDATE MEMORY statements, determine whether each memory is genuinely about the query topic.

CRITICAL RULES:
1. You MUST ground every decision in DIRECT QUOTES from the candidate text.
2. Broad association is NOT evidence. "computer" does NOT match "python" unless Python is explicitly discussed.
3. Polysemy matters: "java" the language ≠ "java" the coffee. "apple" the company ≠ "apple" the fruit.
4. If you cannot quote specific evidence, you MUST output UNRELATED.

RELATION TYPES:
- SAME: The memory is directly and specifically about the query topic
- SUBSUMES: The query topic is broader; the memory discusses a subtopic (e.g., query="philosophy", memory about epistemology)
- SUBSUMED_BY: The memory is broader; it contains the query topic as a subtopic
- OVERLAP: Significant topical overlap but neither contains the other
- UNRELATED: No genuine topical connection (broad association doesn't count)

OUTPUT FORMAT (strict):
{
  "statement_id": <int>,
  "relation": "<SAME|SUBSUMES|SUBSUMED_BY|OVERLAP|UNRELATED>",
  "quotes": [{"text": "..."}] or [],
  "rationale": "<one sentence max>"
}

REQUIREMENTS:
- For UNRELATED: quotes array MUST be empty [], rationale explains why not related
- For any other relation: EXACTLY 2 quotes required
- Each quote: 8-15 words, case-insensitive exact substring from candidate
- Quotes must be distinct
- Rationale: ONE sentence maximum for entire response

Respond with a JSON array, one object per candidate, same order as input."""


def get_prompt_hash() -> str:
    """Get a hash of the verifier prompt for caching."""
    return hashlib.sha256(VERIFIER_PROMPT.encode()).hexdigest()[:16]


# =============================================================================
# CACHE
# =============================================================================

def init_cache():
    """Initialize the verifier cache database."""
    conn = sqlite3.connect(CACHE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            model_id TEXT,
            prompt_hash TEXT,
            query_normalized TEXT,
            statement_id INTEGER,
            statement_text_hash TEXT,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_id, prompt_hash, query_normalized, statement_id, statement_text_hash)
        )
    """)
    conn.commit()
    conn.close()


def get_cached_result(model_id: str, query: str, statement_id: int, statement_text: str) -> Optional[VerifierResult]:
    """Get a cached verifier result if available."""
    conn = sqlite3.connect(CACHE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT result_json FROM cache
           WHERE model_id=? AND prompt_hash=? AND query_normalized=?
           AND statement_id=? AND statement_text_hash=?""",
        (model_id, get_prompt_hash(), query.lower().strip(), statement_id,
         hashlib.sha256(statement_text.encode()).hexdigest()[:16])
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return parse_result_json(json.loads(row[0]))
    return None


def cache_result(model_id: str, query: str, statement_id: int, statement_text: str, result: VerifierResult):
    """Cache a verifier result."""
    conn = sqlite3.connect(CACHE_PATH)
    cursor = conn.cursor()
    result_json = json.dumps({
        "statement_id": result.statement_id,
        "relation": result.relation.value,
        "confidence": result.confidence,
        "quotes": [{"start": q.start, "end": q.end, "text": q.text} for q in result.quotes],
        "rationale": result.rationale,
        "quote_check_passed": result.quote_check_passed,
        "quote_check_errors": result.quote_check_errors,
        "ambiguous_quotes": result.ambiguous_quotes,
    })
    cursor.execute(
        """INSERT OR REPLACE INTO cache
           (model_id, prompt_hash, query_normalized, statement_id, statement_text_hash, result_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model_id, get_prompt_hash(), query.lower().strip(), statement_id,
         hashlib.sha256(statement_text.encode()).hexdigest()[:16], result_json)
    )
    conn.commit()
    conn.close()


def parse_result_json(data: dict) -> VerifierResult:
    """Parse a result JSON dict into a VerifierResult."""
    quotes = []
    for q in data.get("quotes", []):
        quotes.append(Quote(
            text=q.get("text", ""),
            start=q.get("start", -1),
            end=q.get("end", -1),
        ))
    return VerifierResult(
        statement_id=data["statement_id"],
        relation=Relation(data["relation"]),
        confidence=data["confidence"],
        quotes=quotes,
        rationale=data.get("rationale", ""),
        quote_check_passed=data.get("quote_check_passed", False),
        quote_check_errors=data.get("quote_check_errors", []),
        ambiguous_quotes=data.get("ambiguous_quotes", []),
    )


# =============================================================================
# QUOTE VALIDATION
# =============================================================================

@dataclass
class QuoteValidationResult:
    """Result of quote validation with ambiguity tracking."""
    passed: bool
    errors: list[str]
    validated_quotes: list[Quote]
    ambiguous_quotes: list[int] = field(default_factory=list)  # Indices of quotes with multiple occurrences


def _canonicalize(s: str) -> str:
    """
    Safe canonicalization for quote matching:
    - Unicode NFC normalization
    - Smart quotes → ASCII quotes
    - Em/en dashes → hyphen
    - Collapse whitespace runs to single space
    """
    import unicodedata
    import re

    # NFC normalization
    s = unicodedata.normalize('NFC', s)

    # Smart quotes → ASCII
    s = s.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

    # Em/en dashes → hyphen
    s = s.replace('—', '-').replace('–', '-')

    # Collapse whitespace (but preserve single spaces)
    s = re.sub(r'\s+', ' ', s)

    return s


def _word_count(s: str) -> int:
    """Count words after collapsing whitespace."""
    import re
    return len(re.sub(r'\s+', ' ', s.strip()).split())


def validate_quotes(quotes: list[Quote], text: str) -> QuoteValidationResult:
    """
    Validate quotes by searching for them in the candidate text.

    Validation rules:
    - At least 2 distinct quotes required for non-UNRELATED
    - Each quote must be case-insensitive exact substring (after canonicalization)
    - Each quote must be at least 5 words
    - Quotes must be distinct (no duplicates)
    - Ambiguous quotes (multiple occurrences) require disambiguation

    Returns QuoteValidationResult with:
    - passed: True if validation succeeded
    - errors: List of error messages
    - validated_quotes: Quotes with computed positions
    - ambiguous_quotes: Indices of quotes that occur multiple times in text
    """
    errors = []
    validated_quotes = []
    ambiguous_quotes = []
    seen_quotes = set()  # For duplicate detection

    # Must have at least 2 quotes
    if len(quotes) < 2:
        errors.append(f"Need at least 2 quotes, got {len(quotes)}")
        return QuoteValidationResult(False, errors, [], [])

    # Canonicalize and lowercase text for matching
    text_canon = _canonicalize(text).lower()

    for i, q in enumerate(quotes):
        quote_text = q.text.strip()
        if not quote_text:
            errors.append(f"Quote {i}: empty text")
            continue

        # Check word count (minimum 3 words)
        wc = _word_count(quote_text)
        if wc < 3:
            errors.append(f"Quote {i}: only {wc} words, need at least 3")
            continue

        # Check for duplicates
        quote_normalized = quote_text.lower()
        if quote_normalized in seen_quotes:
            errors.append(f"Quote {i}: duplicate of earlier quote")
            continue
        seen_quotes.add(quote_normalized)

        # Canonicalize and search (case-insensitive)
        quote_canon = _canonicalize(quote_text).lower()
        pos = text_canon.find(quote_canon)

        if pos == -1:
            errors.append(f"Quote {i}: '{quote_text[:50]}' not found in text")
            continue

        # Check for multiple occurrences (ambiguity)
        second_pos = text_canon.find(quote_canon, pos + 1)
        if second_pos != -1:
            ambiguous_quotes.append(i)

        # Create validated quote with actual position
        validated_quotes.append(Quote(
            text=quote_text,
            start=pos,
            end=pos + len(quote_canon)
        ))

    if len(validated_quotes) < 2:
        errors.append(f"Only {len(validated_quotes)} quotes validated, need at least 2")
        return QuoteValidationResult(False, errors, validated_quotes, ambiguous_quotes)

    # Ambiguity policy: if ALL quotes are ambiguous, reject unless we have 3+ quotes
    # (at least one unique quote needed for disambiguation)
    if len(ambiguous_quotes) == len(validated_quotes) and len(validated_quotes) < 3:
        errors.append("All quotes are ambiguous (appear multiple times); need at least one unique quote")
        return QuoteValidationResult(False, errors, validated_quotes, ambiguous_quotes)

    # Check non-overlapping
    sorted_quotes = sorted(validated_quotes, key=lambda q: q.start)
    for i in range(len(sorted_quotes) - 1):
        q1, q2 = sorted_quotes[i], sorted_quotes[i + 1]
        if q1.end > q2.start:
            errors.append(f"Quotes {i} and {i+1} overlap")

    return QuoteValidationResult(len(errors) == 0, errors, validated_quotes, ambiguous_quotes)


# =============================================================================
# LLM VERIFIER CALL
# =============================================================================

# Truncation settings for cost control
HEAD_CHARS = 300
TAIL_CHARS = 300


def _truncate_text(text: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> tuple[str, str]:
    """
    Truncate text using head+tail strategy to preserve both beginning and end.
    Returns (truncated_text_for_llm, full_text_for_validation).

    Head+tail preserves endings where definitions/answers often live.
    """
    if len(text) <= head + tail:
        return text, text

    truncated = text[:head] + "\n[...]\n" + text[-tail:]
    return truncated, text


def call_verifier_batch(
    query: str,
    candidates: list[tuple[int, str]],  # (statement_id, text)
    model_id: str = "gpt-4o-mini",
    truncate: bool = True,
) -> list[VerifierResult]:
    """
    Call the LLM verifier for a batch of candidates.
    Returns a VerifierResult for each candidate.

    Args:
        truncate: If True, use head+tail truncation for LLM input (cost saving).
                  Validation still runs against full text.
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm not available - install with: pip install litellm")

    # Build the user prompt with candidates (optionally truncated for LLM)
    # Note: validation runs against full text from candidates, not truncated
    candidates_text = ""
    for i, (stmt_id, text) in enumerate(candidates):
        display_text = _truncate_text(text)[0] if truncate else text
        candidates_text += f"\n--- CANDIDATE {i+1} (statement_id={stmt_id}) ---\n{display_text}\n"

    user_prompt = f"""QUERY TOPIC: "{query}"

CANDIDATES:{candidates_text}

Analyze each candidate and return a JSON array with one result object per candidate."""

    # Call the LLM
    response = litellm.completion(
        model=model_id,
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    # Parse response
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        # Handle both {"results": [...]} and direct array
        if isinstance(data, dict) and "results" in data:
            results_data = data["results"]
        elif isinstance(data, list):
            results_data = data
        else:
            # Try to find an array in the response
            results_data = []
            for key in data:
                if isinstance(data[key], list):
                    results_data = data[key]
                    break
    except json.JSONDecodeError as e:
        # Return UNRELATED for all if parsing fails
        return [
            VerifierResult(
                statement_id=stmt_id,
                relation=Relation.UNRELATED,
                confidence=0.0,
                quotes=[],
                rationale=f"JSON parse error: {e}",
                quote_check_passed=False,
                quote_check_errors=["JSON parse error"],
            )
            for stmt_id, _ in candidates
        ]

    # Match results to candidates
    results = []
    stmt_id_to_text = {stmt_id: text for stmt_id, text in candidates}

    for r in results_data:
        stmt_id = r.get("statement_id")
        if stmt_id not in stmt_id_to_text:
            continue

        # Parse quotes - handle both old format (with offsets) and new format (text only)
        quotes = []
        for q in r.get("quotes", []):
            if isinstance(q, dict):
                quotes.append(Quote(
                    text=q.get("text", ""),
                    start=q.get("start", -1),
                    end=q.get("end", -1),
                ))
            elif isinstance(q, str):
                quotes.append(Quote(text=q))

        result = VerifierResult(
            statement_id=stmt_id,
            relation=Relation(r.get("relation", "UNRELATED")),
            confidence=r.get("confidence", 0.0),
            quotes=quotes,
            rationale=r.get("rationale", ""),
        )

        # Validate quotes if relation is not UNRELATED
        if result.relation != Relation.UNRELATED:
            validation = validate_quotes(quotes, stmt_id_to_text[stmt_id])
            result.quote_check_passed = validation.passed
            result.quote_check_errors = validation.errors
            result.ambiguous_quotes = validation.ambiguous_quotes
            if validation.validated_quotes:
                result.quotes = validation.validated_quotes  # Use validated quotes with positions
            # Force UNRELATED if validation fails
            if not validation.passed:
                result.relation = Relation.UNRELATED
        else:
            result.quote_check_passed = True  # N/A for UNRELATED

        results.append(result)

    # Fill in any missing candidates
    returned_ids = {r.statement_id for r in results}
    for stmt_id, text in candidates:
        if stmt_id not in returned_ids:
            results.append(VerifierResult(
                statement_id=stmt_id,
                relation=Relation.UNRELATED,
                confidence=0.0,
                quotes=[],
                rationale="Not returned by verifier",
                quote_check_passed=False,
                quote_check_errors=["Missing from verifier response"],
            ))

    return results


# =============================================================================
# BATCHED VERIFIER WITH EARLY EXIT
# =============================================================================

@dataclass
class VerifierStats:
    """Statistics for a single query verification run."""
    query: str
    total_candidates: int
    batches_used: int
    llm_calls: int
    cache_hits: int
    accepted_count: int
    results: list[VerifierResult] = field(default_factory=list)


def verify_query(
    query: str,
    candidate_ids: list[int],
    model_id: str = "gpt-4o-mini",
    batch_size: int = 5,  # Production default: 5 (cost-efficient)
    accept_target: int = 3,  # Use 1 for "when did we discuss X?" queries
    max_batches: int = 1,  # Production default: 1 (rely on retrieval quality)
) -> VerifierStats:
    """
    Verify candidates for a query with batched LLM calls and early exit.

    Production recommendations:
    - batch_size=5, max_batches=1 for cost efficiency (~1k tokens/query)
    - accept_target=1 for single-match queries ("when did we discuss X?")
    - accept_target=3 for listing queries ("what have we discussed about X?")

    Algorithm:
    1. Take candidates in batches of batch_size
    2. Check cache first, call LLM for uncached
    3. Accept candidates with SAME/SUBSUMES/SUBSUMED_BY/OVERLAP that pass quote validation
    4. Stop when accept_target reached or max_batches used
    5. If batch 1 yields 0 and max_batches > 1, do fallback batch
    """
    # Load statements from DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    stmt_texts = {}
    for stmt_id in candidate_ids:
        cursor.execute("SELECT text FROM statements WHERE id=?", (stmt_id,))
        row = cursor.fetchone()
        if row:
            stmt_texts[stmt_id] = row[0]
    conn.close()

    stats = VerifierStats(
        query=query,
        total_candidates=len(candidate_ids),
        batches_used=0,
        llm_calls=0,
        cache_hits=0,
        accepted_count=0,
    )

    accepted = []
    all_results = []
    remaining_ids = list(candidate_ids)

    for batch_num in range(1, max_batches + 1):
        if not remaining_ids:
            break

        # Take next batch
        batch_ids = remaining_ids[:batch_size]
        remaining_ids = remaining_ids[batch_size:]
        stats.batches_used = batch_num

        # Check cache first
        uncached = []
        for stmt_id in batch_ids:
            if stmt_id not in stmt_texts:
                continue
            cached = get_cached_result(model_id, query, stmt_id, stmt_texts[stmt_id])
            if cached:
                stats.cache_hits += 1
                all_results.append(cached)
                if cached.relation != Relation.UNRELATED and cached.quote_check_passed:
                    accepted.append(cached)
            else:
                uncached.append((stmt_id, stmt_texts[stmt_id]))

        # Call LLM for uncached
        if uncached:
            stats.llm_calls += 1
            batch_results = call_verifier_batch(query, uncached, model_id)
            for result in batch_results:
                # Cache the result
                cache_result(model_id, query, result.statement_id, stmt_texts[result.statement_id], result)
                all_results.append(result)
                if result.relation != Relation.UNRELATED and result.quote_check_passed:
                    accepted.append(result)

        # Early exit conditions
        if len(accepted) >= accept_target:
            break

        # If batch 1 yields 0 and we have more, continue to batch 2
        # Otherwise stop
        if batch_num == 1 and len(accepted) == 0 and remaining_ids:
            continue  # Fallback to batch 2
        elif batch_num >= 1 and len(accepted) > 0:
            # Got some results, stop
            break

    stats.accepted_count = len(accepted)
    stats.results = all_results
    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_verification(
    model_id: str = "gpt-4o-mini",
    batch_size: int = 10,
    accept_target: int = 3,
    max_batches: int = 2,
) -> list[VerifierStats]:
    """Run verification on all query cases."""
    init_cache()

    with open(QUERY_CASES_PATH) as f:
        cases = json.load(f)

    all_stats = []
    for case in cases:
        query = case["query"]
        candidate_ids = case["candidates"]

        print(f"Verifying query: '{query}' ({len(candidate_ids)} candidates)...")
        stats = verify_query(
            query=query,
            candidate_ids=candidate_ids,
            model_id=model_id,
            batch_size=batch_size,
            accept_target=accept_target,
            max_batches=max_batches,
        )
        all_stats.append(stats)
        print(f"  -> {stats.accepted_count} accepted, {stats.llm_calls} LLM calls, {stats.cache_hits} cache hits")

    return all_stats


if __name__ == "__main__":
    # Quick test
    init_cache()
    print(f"Prompt hash: {get_prompt_hash()}")
    print(f"Cache path: {CACHE_PATH}")
