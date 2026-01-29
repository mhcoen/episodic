"""
Fixed benchmark scenarios with pre-computed embeddings for deterministic CI.

These fixtures enable reproducible testing without calling embedding models
during CI runs, ensuring tests are fast and deterministic.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class FixedResumeScenario:
    """A resume scenario with pre-computed embeddings for determinism."""

    scenario_id: str
    description: str
    category: str  # "short_gap", "medium_gap", "long_gap", "ambiguous"

    # Topic A (original)
    topic_a_name: str
    topic_a_exchanges: List[Dict[str, str]]  # [{user: ..., assistant: ...}, ...]

    # Topic B (intervening)
    topic_b_name: str
    topic_b_exchanges: List[Dict[str, str]]

    # Resume query
    resume_query: str

    # Expected outcomes
    expected_reactivation: str  # "topic_a", "topic_b", "continue", "disambiguate"

    # Fields with defaults (must come after non-default fields)
    topic_a_embedding: List[float] = field(default_factory=list)  # Pre-computed centroid
    topic_b_embedding: List[float] = field(default_factory=list)
    resume_query_embedding: List[float] = field(default_factory=list)  # Pre-computed
    expected_context_contains: List[str] = field(default_factory=list)  # MUST appear
    expected_context_excludes: List[str] = field(default_factory=list)  # Must NOT appear


def load_benchmark_fixtures() -> List[FixedResumeScenario]:
    """Load fixtures from JSON file."""
    fixtures_path = FIXTURES_DIR / "resume_scenarios.json"
    if not fixtures_path.exists():
        logger.warning(f"Fixtures file not found at {fixtures_path}, using defaults")
        return create_default_fixtures()

    with open(fixtures_path) as f:
        data = json.load(f)

    return [FixedResumeScenario(**s) for s in data["scenarios"]]


def create_default_fixtures() -> List[FixedResumeScenario]:
    """Create default fixture set (embeddings to be computed separately)."""
    scenarios = [
        # Short gap (5-10 turns)
        FixedResumeScenario(
            scenario_id="short_gap_python",
            description="Return to Python discussion after brief coffee tangent",
            category="short_gap",
            topic_a_name="python-debugging",
            topic_a_exchanges=[
                {
                    "user": "How do I fix this IndexError in Python?",
                    "assistant": "IndexError usually means you're accessing an index that doesn't exist in the list. Check your list length with len() before accessing.",
                },
                {
                    "user": "What about using try-except?",
                    "assistant": "Yes, you can wrap it in a try-except block to catch IndexError and handle it gracefully.",
                },
            ],
            topic_b_name="coffee-brewing",
            topic_b_exchanges=[
                {
                    "user": "What temperature for pour-over coffee?",
                    "assistant": "Ideal is 195-205F (90-96C). Water that's too hot will over-extract and taste bitter.",
                },
            ],
            resume_query="What was that Python fix again?",
            expected_reactivation="topic_a",
            expected_context_contains=["IndexError", "try-except"],
            expected_context_excludes=["coffee", "temperature", "pour-over"],
        ),
        # Medium gap (20-50 turns)
        FixedResumeScenario(
            scenario_id="medium_gap_database",
            description="Return to database schema after multiple intervening topics",
            category="medium_gap",
            topic_a_name="database-design",
            topic_a_exchanges=[
                {
                    "user": "Should I use UUID or integer for primary keys?",
                    "assistant": "UUIDs are better for distributed systems, integers are more efficient for single-database setups. Consider your scaling needs.",
                },
                {
                    "user": "What about foreign key indexing?",
                    "assistant": "Always index foreign keys for join performance. Without indexes, JOINs become table scans.",
                },
            ],
            topic_b_name="weekend-plans",
            topic_b_exchanges=[
                {
                    "user": "Any good hiking trails nearby?",
                    "assistant": "There are several great trails in the area. What difficulty level are you looking for?",
                },
                {
                    "user": "What should I pack?",
                    "assistant": "For a day hike: water, snacks, sunscreen, first aid kit, and appropriate footwear.",
                },
            ],
            resume_query="Back to the database question - should I add that index?",
            expected_reactivation="topic_a",
            expected_context_contains=["UUID", "foreign key", "index"],
            expected_context_excludes=["hiking", "trails", "pack"],
        ),
        # Long gap (100+ turns / simulated "year later")
        FixedResumeScenario(
            scenario_id="long_gap_ml_project",
            description="Year-later resume of ML project discussion",
            category="long_gap",
            topic_a_name="ml-model-training",
            topic_a_exchanges=[
                {
                    "user": "What learning rate should I use for BERT fine-tuning?",
                    "assistant": "Start with 2e-5 for BERT fine-tuning. This is the recommended rate from the original paper.",
                },
                {
                    "user": "How many epochs?",
                    "assistant": "For fine-tuning, 3-4 epochs is typically sufficient. More can lead to overfitting.",
                },
            ],
            topic_b_name="recipe-collection",
            topic_b_exchanges=[
                {
                    "user": "How do I make pasta carbonara?",
                    "assistant": "The authentic recipe uses guanciale, egg yolks, pecorino romano, and black pepper. No cream.",
                },
            ],
            resume_query="What was the learning rate you recommended for that model?",
            expected_reactivation="topic_a",
            expected_context_contains=["learning rate", "2e-5", "BERT"],
            expected_context_excludes=["pasta", "carbonara", "recipe"],
        ),
        # Ambiguous case
        FixedResumeScenario(
            scenario_id="ambiguous_java",
            description="Ambiguous 'java' - programming vs coffee",
            category="ambiguous",
            topic_a_name="java-programming",
            topic_a_exchanges=[
                {
                    "user": "How do I handle NullPointerException in Java?",
                    "assistant": "You can use Optional<T> to avoid nulls, or add null checks. Modern Java recommends Optional for APIs.",
                },
            ],
            topic_b_name="coffee-origins",
            topic_b_exchanges=[
                {
                    "user": "Tell me about Java coffee beans",
                    "assistant": "Java beans are from Indonesia, known for a full body and low acidity. Estate Java is highly regarded.",
                },
            ],
            resume_query="Tell me more about Java",
            expected_reactivation="disambiguate",
            expected_context_contains=[],  # Depends on disambiguation choice
            expected_context_excludes=[],
        ),
        # Cross-topic reference
        FixedResumeScenario(
            scenario_id="cross_topic_explicit",
            description="Explicit reference to earlier topic",
            category="short_gap",
            topic_a_name="git-workflow",
            topic_a_exchanges=[
                {
                    "user": "What's the best git branching strategy?",
                    "assistant": "For most teams, trunk-based development with short-lived feature branches works well. GitFlow is good for releases.",
                },
            ],
            topic_b_name="project-planning",
            topic_b_exchanges=[
                {
                    "user": "How should I structure my sprint planning?",
                    "assistant": "Start with backlog refinement, then planning poker for estimates, and finally capacity planning.",
                },
            ],
            resume_query="Going back to that git branching discussion, what about rebasing?",
            expected_reactivation="topic_a",
            expected_context_contains=["branching", "trunk-based", "GitFlow"],
            expected_context_excludes=["sprint", "backlog", "planning poker"],
        ),
    ]

    return scenarios


def compute_and_save_embeddings(
    scenarios: Optional[List[FixedResumeScenario]] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    Compute embeddings once and save to fixtures file.

    Run this once to generate the fixtures JSON with pre-computed embeddings:
        python -c "from episodic.evaluation.benchmark_fixtures import compute_and_save_embeddings; compute_and_save_embeddings()"
    """
    from episodic.rag_utils import SilentSentenceTransformerEmbeddingFunction

    if scenarios is None:
        scenarios = create_default_fixtures()

    embed_fn = SilentSentenceTransformerEmbeddingFunction(model_name=model_name)

    for scenario in scenarios:
        # Compute topic A centroid
        topic_a_texts = [
            f"{e['user']} {e['assistant']}" for e in scenario.topic_a_exchanges
        ]
        topic_a_embeddings = embed_fn(topic_a_texts)
        scenario.topic_a_embedding = np.mean(topic_a_embeddings, axis=0).tolist()

        # Compute topic B centroid
        topic_b_texts = [
            f"{e['user']} {e['assistant']}" for e in scenario.topic_b_exchanges
        ]
        topic_b_embeddings = embed_fn(topic_b_texts)
        scenario.topic_b_embedding = np.mean(topic_b_embeddings, axis=0).tolist()

        # Compute resume query embedding
        query_embeddings = embed_fn([scenario.resume_query])
        scenario.resume_query_embedding = query_embeddings[0].tolist()

    # Save to file
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures_path = FIXTURES_DIR / "resume_scenarios.json"

    with open(fixtures_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "embedding_model": model_name,
                "scenarios": [asdict(s) for s in scenarios],
            },
            f,
            indent=2,
        )

    print(f"Saved {len(scenarios)} scenarios with embeddings to {fixtures_path}")


def get_fixture_embedding_dimension() -> int:
    """Get the embedding dimension from fixtures, or default if not available."""
    scenarios = load_benchmark_fixtures()
    if scenarios and scenarios[0].topic_a_embedding:
        return len(scenarios[0].topic_a_embedding)
    # Default for all-MiniLM-L6-v2
    return 384


def validate_fixtures() -> Dict[str, Any]:
    """
    Validate that fixtures are properly formed and have embeddings.

    Returns:
        Dict with validation results and any issues found.
    """
    issues = []
    scenarios = load_benchmark_fixtures()

    if not scenarios:
        return {"valid": False, "issues": ["No scenarios found"], "scenario_count": 0}

    for scenario in scenarios:
        if not scenario.topic_a_embedding:
            issues.append(f"{scenario.scenario_id}: Missing topic_a_embedding")
        if not scenario.topic_b_embedding:
            issues.append(f"{scenario.scenario_id}: Missing topic_b_embedding")
        if not scenario.resume_query_embedding:
            issues.append(f"{scenario.scenario_id}: Missing resume_query_embedding")

        # Check embedding dimensions match
        dims = set()
        if scenario.topic_a_embedding:
            dims.add(len(scenario.topic_a_embedding))
        if scenario.topic_b_embedding:
            dims.add(len(scenario.topic_b_embedding))
        if scenario.resume_query_embedding:
            dims.add(len(scenario.resume_query_embedding))

        if len(dims) > 1:
            issues.append(f"{scenario.scenario_id}: Inconsistent embedding dimensions: {dims}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "scenario_count": len(scenarios),
        "categories": list(set(s.category for s in scenarios)),
    }
