"""KG eval-ablation dataclasses (EvalResult, EvalSummary).

Split out of eval_ablation.py; re-exported there.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    prompt_id: str
    condition: str  # "A", "B", "C"
    category: str
    # Correctness
    required_facts_found: int = 0
    required_facts_total: int = 0
    expected_contains_found: int = 0
    expected_contains_total: int = 0
    factual_score: float = 0.0
    # Cost
    kg_block_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # Latency
    context_build_ms: float = 0.0
    llm_response_ms: float = 0.0
    # Closure tracking
    derived_edges_count: int = 0
    closure_expected: bool = False
    closure_rule: str = ""
    # Derived relevance (condition C only, 0 for A/B)
    derived_max_relevance: float = 0.0
    derived_mean_relevance: float = 0.0
    derived_top_fact: str = ""
    derived_top_relevance: float = 0.0
    # Oracle (closure_expected items only)
    oracle_hit: bool = False
    oracle_fact: str = ""
    # Raw
    llm_response: str = ""
    kg_context_text: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)
    by_category: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    preload_stats: dict = field(default_factory=dict)
    closure_checks: dict = field(default_factory=dict)


CONDITIONS = {
    "A": {"kg_context": False, "kg_max_derived": 0},
    "B": {"kg_context": True, "kg_max_derived": 0},
    "C": {"kg_context": True, "kg_max_derived": 3},
}


