# Ablation Fix: Closure Differentiation

## Problem
B and C conditions produce identical results because all setup_context sentences establish direct edges. Closure rules (KINSHIP_LOCATION, DEVICE_SPEC) never fire — there's nothing to derive.

## Fix: Split facts across separate setup turns

### Target categories
- `multi_hop` (11 items)
- `relational_bridging` (11 items)
- Optionally add 3-4 `device_spec` items for independent DEVICE_SPEC measurement

### Leave unchanged
- `alias` (11 items) — tests mention detection, not closure
- `temporal_negation` (7 items) — tests TIME_PAST gating
- `baseline_factual` (12 items) — tests direct 1-hop retrieval

### Rewrite rules

Each multi_hop/relational_bridging item must be rewritten so:
1. Setup has 2-4 turns, each establishing ONE edge
2. Under B (no closure), only 1-hop incident edges on mentioned entities are available — insufficient to answer
3. Under C (closure on), exactly one derived fact bridges the gap

#### Pattern: KINSHIP_LOCATION
```json
{
  "id": "multi_hop_01",
  "prompt": "Where does my daughter go to school?",
  "setup_context": [
    "My daughter is Emma.",
    "She sounds lovely.",
    "Emma is a sophomore at MIT.",
    "MIT is a great school."
  ],
  "answer_key": {
    "required_facts": ["Emma", "MIT"],
    "expected_answer_contains": ["MIT"],
    "category": "multi_hop"
  },
  "closure_expected": true,
  "closure_rule": "KINSHIP_LOCATION",
  "closure_derived": "Emma located_at MIT"
}
```

Why B fails: prompt mentions "daughter" → matches user:self → retrieves `user related_to Emma`. But MIT is not incident on user:self. No MIT in context.
Why C succeeds: KINSHIP_LOCATION fires: `user related_to Emma` + `Emma located_at MIT` → derives `Emma located_at MIT`, injected into context.

#### Pattern: DEVICE_SPEC
```json
{
  "id": "rel_bridge_07",
  "prompt": "How much RAM does my main laptop have?",
  "setup_context": [
    "My main laptop is a MacBook Pro M3 Max.",
    "Good choice.",
    "The MacBook Pro M3 Max has 64 gigs of RAM.",
    "That's plenty for local inference."
  ],
  "answer_key": {
    "required_facts": ["MacBook Pro M3 Max", "64"],
    "expected_answer_contains": ["64"],
    "category": "relational_bridging"
  },
  "closure_expected": true,
  "closure_rule": "DEVICE_SPEC",
  "closure_derived": "MacBook Pro M3 Max has 64GB RAM"
}
```

Why B fails: prompt mentions "laptop" → matches MacBook Pro M3 Max (alias or canonical). Retrieves `user has MacBook Pro M3 Max`. But RAM is not incident on user:self, it's incident on MacBook. So B may get `user has MacBook` but NOT `MacBook has 64GB RAM`.
Why C succeeds: DEVICE_SPEC fires: `user has MacBook` + `MacBook has 64GB RAM` → derives `MacBook Pro M3 Max has 64GB RAM`.

**IMPORTANT**: For DEVICE_SPEC to work correctly, `retrieve_neighborhood()` is called for each matched entity. If "laptop" matches MacBook Pro M3 Max directly, then B WILL get edges incident on MacBook including `MacBook has 64GB RAM` — that's a 1-hop edge on the matched entity. In that case B and C would be identical again.

DEVICE_SPEC only differentiates B from C when:
- The prompt mentions a PARENT entity (user:self via "my") and the spec is on a CHILD entity
- The child entity is NOT separately mentioned in the prompt

So DEVICE_SPEC prompts should avoid mentioning the device name directly:
```json
{
  "prompt": "How much RAM does my daily setup have?",
  ...
}
```
Here "my daily setup" might not match MacBook Pro M3 Max. The user:self edges include `has MacBook`, but MacBook's own edges (has 64GB) require 2-hop. Only closure bridges this.

If alias detection maps "laptop" → MacBook Pro M3 Max, then MacBook IS a matched entity and its 1-hop edges include RAM. In that case, make the prompt even more indirect: "What are the specs of the machine I do inference on?"

### Concrete rewrite plan

For each of the 22 items (11 multi_hop + 11 relational_bridging):

1. Identify which closure rule applies (KINSHIP_LOCATION or DEVICE_SPEC)
2. Split setup into 2-4 turns, each with one edge
3. Write the prompt so it mentions only the ROOT entity (user:self side), not the leaf
4. Add `closure_expected: true`, `closure_rule`, `closure_derived` fields
5. Verify that the derived fact is the one needed to answer correctly

### Items that DON'T fit closure patterns

Some existing multi_hop/relational_bridging items establish facts that don't match either closure rule pattern (e.g., "What animals do we have at home?" — direct has-edges, no bridging needed). These should be:
- Recategorized to `baseline_factual` if they're 1-hop, OR
- Rewritten to fit a closure pattern, OR  
- Kept as-is with `closure_expected: false` (they still test B>A)

### Harness changes

1. Add `closure_expected` field support in eval scoring
2. Track `derived_edges_count` and `derived_tokens` per result
3. Add automated check: for `closure_expected: true` items:
   - C must produce `derived_edges_count >= 1`
   - B must produce `derived_edges_count == 0`
   - If violated, flag as dataset/harness failure (not a model failure)
4. Summary table should show B vs C delta separately for closure_expected items

### Expected outcome

After restructuring:
- A: low scores across all categories (no facts)
- B: high on baseline_factual and alias (direct 1-hop), LOW on closure_expected items
- C: high on everything — closure bridges the gap on multi_hop and relational_bridging
- B vs C delta: measurable ONLY on closure_expected items, zero elsewhere
