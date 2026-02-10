# Phase 2 Fix: Chain-Aware Closure Scoring

## Problem with Phase 2 v1
Lexical overlap scoring filters out closure-derived facts because closure targets entities NOT mentioned in the prompt. "Where does my daughter go to school?" has zero overlap with "Emma located_at MIT". Filtering on overlap==0 removes exactly the facts closure exists to produce.

## Architecture: Two separate scoring functions

### A. direct_score(edge, prompt_tokens, matched_ids)
For 1-hop neighborhood edges. Unchanged from Phase 2 v1 except: soft ranking, not hard filter.

```python
def direct_score(e, prompt_tokens, matched_ids):
    e_tokens = compute_prompt_tokens(f"{e.subj_name} {e.predicate} {e.obj_name}")
    overlap = len(prompt_tokens & e_tokens)
    touch = 2 if (e.subj_id in matched_ids or e.obj_id in matched_ids) else 0
    recency = 1  # placeholder, use node recency if available
    return 5 * overlap + 2 * touch + recency
```

**Gating**: If max(direct_score) across ALL direct edges == 0 AND no closure candidate has seed_bonus > 0, suppress entire block. Otherwise, rank direct edges by score, include top N.

### B. closure_score(derived, prompt_text, matched_ids, seeds)
For closure-derived edges. Scores the derivation chain, not the endpoint text.

```python
KINSHIP_CUES = frozenset({
    'daughter', 'son', 'wife', 'husband', 'partner', 'family',
    'kid', 'child', 'parent', 'mother', 'father', 'sister', 'brother',
    'spouse', 'sibling',
})
DEVICE_CUES = frozenset({
    'laptop', 'macbook', 'computer', 'machine', 'desktop', 'phone',
    'spec', 'specs', 'ram', 'gpu', 'cpu', 'memory', 'storage',
    'run', 'models', 'inference', 'keyboard', 'monitor', 'display',
    'device', 'setup', 'rig',
})

def closure_score(d: DerivedFact, prompt_tokens: set, matched_ids: set, seeds: list[int]):
    # 1. Seed bonus: does chain originate from a prompt seed?
    seed_bonus = 0
    if d.source_seed_id in seeds[:3]:  # top-K seeds
        seed_bonus = 3
    elif d.source_seed_id == USER_SELF_ID:
        seed_bonus = 2  # user:self via first-person cue
    
    # 2. Bridge bonus: does prompt intent match the closure rule?
    bridge_bonus = 0
    if d.closure_rule == 'KINSHIP_LOCATION' and (prompt_tokens & KINSHIP_CUES):
        bridge_bonus = 3
    elif d.closure_rule == 'DEVICE_SPEC' and (prompt_tokens & DEVICE_CUES):
        bridge_bonus = 3
    
    # 3. Lexical overlap: tiebreaker only
    d_tokens = compute_prompt_tokens(f"{d.subj_name} {d.predicate} {d.obj_name}")
    overlap = len(prompt_tokens & d_tokens)
    
    return seed_bonus + bridge_bonus + overlap
```

**No hard filter on closure edges by overlap.** Only constrained by:
- Total derived cap (kg_max_derived, default 3)
- Per-seed derived cap (kg_derived_per_seed, default 2)
- Must originate from a seed entity (enforced at generation, not scoring)

### C. DerivedFact must carry chain metadata

Add fields to DerivedFact (or the closure result struct):

```python
@dataclass
class DerivedFact:
    subj_name: str
    predicate: str
    obj_name: str
    closure_rule: str        # 'KINSHIP_LOCATION' | 'DEVICE_SPEC'
    source_seed_id: int      # the matched entity that started the chain
    intermediate_id: int     # the bridging entity (e.g., Emma)
    intermediate_name: str   # for /kg explain display
```

### D. Closure expansion: seeds only

Hard invariant: `apply_closure_rules()` ONLY expands from seed entities.

```python
seeds = matched_entity_ids[:kg_closure_seed_limit]  # top K by weight
if first_person_cue and USER_SELF_ID not in seeds:
    seeds.append(USER_SELF_ID)

for seed_id in seeds:
    seed_edges = get_1hop(seed_id, conn)
    for rule in CLOSURE_RULES:
        candidates = rule.derive(seed_id, seed_edges, conn)
        for c in candidates:
            c.source_seed_id = seed_id
            all_candidates.append(c)
```

Signal chain is NEVER a closure seed unless "signal chain" is explicitly mentioned in the prompt and matched by mention detection. This prevents DEVICE_SPEC firing on signal chain for a kinship prompt.

### E. Reworked per-entity caps

Replace fixed cap=4 with:

```python
def _budget_edges(direct_edges, derived_edges, matched_ids, budget):
    """Allocate edge budget across entities."""
    # 1. Guarantee: at least 2 edges per matched seed entity
    guaranteed = {}
    for eid in matched_ids:
        eid_edges = [e for e in direct_edges if _incident(e, eid)]
        guaranteed[eid] = eid_edges[:2]
    
    # 2. Fill remaining budget by global direct_score rank
    used = set(id(e) for g in guaranteed.values() for e in g)
    remaining = [e for e in direct_edges if id(e) not in used]
    remaining.sort(key=lambda e: e._score, reverse=True)
    
    fill_budget = budget - sum(len(g) for g in guaranteed.values())
    fill = remaining[:fill_budget]
    
    # 3. Append derived edges (already scored and capped)
    result = []
    for g in guaranteed.values():
        result.extend(g)
    result.extend(fill)
    result.extend(derived_edges)
    return result
```

No special-casing user:self cap. Instead, the guarantee-then-fill strategy naturally limits any single entity: each seed gets 2 guaranteed, the rest compete by score.

### F. Injection gating (revised)

```python
max_direct = max(direct_scores, default=0)
has_seeded_closure = any(c.source_seed_id in seeds for c in derived_edges)

if max_direct == 0 and not has_seeded_closure:
    # Suppress: no relevant direct edges AND no seed-originating closure
    return KGContextResult(suppressed=True, suppressed_reason="no_relevant_edges")
```

Closure can inject even when all direct edges have overlap==0, IF it was triggered from a matched seed AND bridge_bonus fired. This handles "Where does my daughter go to school?" where direct edges for user:self may have zero overlap, but KINSHIP_LOCATION closure from user:self produces Emma→MIT with bridge_bonus=3.

## Validation: Small subset first

### Step 1: Implement A-F

### Step 2: Run ONLY the 22 closure items (not all 58)

```bash
/kg eval --dry-run --filter closure_expected
```

Verify three structural invariants:

**Invariant 1**: For KINSHIP_LOCATION items, C's derived facts include the expected kinship closure (Emma→MIT etc.), NOT device specs.
**Invariant 2**: For DEVICE_SPEC items, C's derived facts include the expected device closure (MacBook→RAM etc.), NOT kinship closures.
**Invariant 3**: B has 0 derived facts for all 22 items (unchanged from Phase 2 v1).

Also verify:
- oracle_hit > 0 (targeting > 15/22)
- No suppressed blocks on closure items (kinship/device cues should fire)

### Step 3: If invariants hold, run full eval (all 58 items)

Compare against both baselines:
- Pre-Phase2: A=0.56, B=0.63, C=0.63
- Phase2 v1: A=0.54, B=0.62, C=0.60

Expected improvements:
- multi_hop C > B (kinship closure injects correct facts)
- relational_bridging B ≥ A (poisoning fixed by revised gating)
- alias/baseline/temporal: stable

## Config knobs

| Key | Default | Description |
|-----|---------|-------------|
| kg_closure_seed_limit | 3 | Max matched entities used as closure seeds |
| kg_derived_per_seed | 2 | Max derived edges per seed entity |
| kg_edges_per_entity | 4 | Guaranteed + fill cap per entity |
| kg_relevance_gate | true | Suppress block if no relevant edges and no seeded closure |

## Files to modify
- episodic/kg/context_source.py: scoring functions, closure expansion, gating, DerivedFact fields
- episodic/kg/eval_ablation.py: --filter flag for subset eval
- tests/kg/test_kg_readside.py: update Phase 2 tests for chain-aware scoring
