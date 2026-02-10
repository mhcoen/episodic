# KG Phase 2: Prompt-Aware Closure — Final Report

## Summary

Phase 2 adds prompt-aware closure scoring to the KG read-side context injection.
Closure-derived facts (KINSHIP_LOCATION, DEVICE_SPEC) are scored and routed based
on bridge cue sets in the user's prompt, giving the LLM grounded multi-hop facts
it couldn't otherwise access from direct edges alone.

**Result: +16pp C-B lift (0.61 → 0.77), no regressions.**

## Eval Progression

| Run | A (base) | B (KG) | C (KG+cl) | C-B lift |
|-----|----------|--------|-----------|----------|
| Pre-Phase2 | 0.56 | 0.63 | 0.63 | +0.00 |
| Phase2v1 (lexical overlap) | 0.54 | 0.62 | 0.60 | -0.02 |
| **Phase2v2 (chain-aware)** | **0.57** | **0.61** | **0.77** | **+0.16** |

## Per-Category Breakdown (Phase2v2)

| Category | A (base) | B (KG) | C (KG+cl) | C-B lift |
|----------|----------|--------|-----------|----------|
| multi_hop | 0.20 | 0.17 | 0.61 | **+44pp** |
| relational_bridging | 0.14 | 0.14 | 0.55 | **+41pp** |
| alias | 0.69 | 0.89 | 0.89 | 0 |
| baseline_factual | 0.86 | 0.87 | 0.88 | +1pp |
| temporal_negation | 0.90 | 0.90 | 0.90 | 0 |

## Closure-Specific Metrics

| Rule | B score | C score | Lift | Oracle |
|------|---------|---------|------|--------|
| KINSHIP_LOCATION | 0.00 | 0.44 | +44pp | 6/6 (100%) |
| DEVICE_SPEC | 0.21 | 0.62 | +41pp | 6/16 (38%) |
| ALL closure | 0.16 | 0.58 | +42pp | 12/22 (55%) |

DEVICE_SPEC oracle is 38% because many device-spec facts are already present as
direct edges (deduped before oracle check). The facts ARE in the context block,
just via the direct path rather than the closure path.

## Architecture: Phase2v2 Chain-Aware Scoring

### Two Scoring Functions

**`_score_direct_edges()`** — For ranking 1-hop neighborhood edges:
```
direct_score = 5 * overlap + 2 * touch + recency
```
- `overlap`: word-level intersection with prompt tokens
- `touch`: 2 if edge connects to a co-mentioned entity, 0 otherwise
- `recency`: source_node_id / 1M (tiebreaker)

**`closure_score()`** — For ranking closure-derived facts:
```
closure_score = seed_bonus + bridge_bonus + overlap
```
- `seed_bonus`: 3 if source_seed in top-3 matched entities, 2 if user:self
- `bridge_bonus`: 3 if rule matches prompt cues (KINSHIP_CUES or DEVICE_CUES)
- `overlap`: word-level intersection with prompt tokens

### Bridge Cue Routing

Two frozen cue sets determine which closure rule gets bridge_bonus:

- **KINSHIP_CUES**: daughter, son, wife, husband, partner, family, kid, child,
  parent, mother, father, sister, brother, spouse, sibling
- **DEVICE_CUES**: laptop, macbook, computer, machine, desktop, phone, spec,
  specs, ram, gpu, cpu, memory, storage, keyboard, monitor, display, device,
  setup, rig

When the prompt contains "daughter" (KINSHIP_CUES), KINSHIP_LOCATION gets +3
bridge_bonus, routing it above DEVICE_SPEC. Vice versa for device prompts.

### Seed-Only Expansion

Closure rules only fire from seed entities (top-K matched + user:self).
`apply_closure_rules()` queries the DB directly for all user:self `related_to`
and `has` edges, bypassing the neighborhood cap that previously starved
`related_to` edges.

### Budget Allocation

`_budget_edges()` replaces the flat per-entity cap:
- Guarantee 2 edges per seed entity
- Fill remaining budget by global rank_score
- Total budget = matched_entities * kg_edges_per_entity

### Revised Gating

Suppress KG context block only if:
- `max_overlap == 0` (no direct edge overlaps with prompt), AND
- NOT (`bool(derived) AND has_bridge_cue`)

Bridge-cued closure bypasses gating even with zero direct overlap.

### DerivedFact Chain Metadata

Each DerivedFact carries:
- `source_seed_id`: entity that seeded the closure expansion
- `intermediate_id`: entity in the middle of the 2-hop chain
- `intermediate_name`: human-readable name of intermediate

## Bugs Fixed

### 1. max_derived=0 Off-By-One

`apply_closure_rules` checked `len(capped) >= max_derived` AFTER appending the
first item. When `max_derived=0`: append → `len([item]) >= 0` → True → break.
Always returned 1 derived fact instead of 0.

**Fix**: Early return `if max_derived <= 0: return []`.

### 2. Neighborhood Cap Starving related_to

`retrieve_neighborhood()` returns max 5 edges, ranked by predicate priority.
With 6+ `has` edges (priority 0), `related_to` (priority 5) was always pushed
out. `apply_closure_rules` iterated these capped edges, so KINSHIP_LOCATION
never fired.

**Fix**: Closure queries DB directly for user:self `related_to` and `has` edges
with a dedicated SQL query, independent of the neighborhood cap.

## Config Knobs Added

| Key | Alias | Default | Description |
|-----|-------|---------|-------------|
| kg_closure_seed_limit | kg-closure-seed-limit | 3 | Max matched entities as closure seeds |
| kg_edges_per_entity | kg-edges-per-entity | 4 | Budget per entity (guarantee 2 + fill) |
| kg_derived_per_seed | kg-derived-per-seed | 2 | Max derived edges per seed entity |
| kg_relevance_gate | kg-relevance-gate | true | Suppress if no overlap and no bridge cue |

## Cost

Average KG context block: 83 tokens (C condition).
No LLM calls on the read path. All scoring is deterministic.

## Test Coverage

- 78 KG-specific tests passing
- Phase 2v2 tests: bridge cue routing, budget limits, closure metadata, suppression
- Eval harness: 58-item dataset, 3 conditions, dry-run and full LLM modes
- Closure subset validation: 22 items with invariant checks
