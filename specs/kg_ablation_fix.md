# Phase 1.3 Fix: Ablation Redesign

## Problem

The current harness includes `setup_context` as conversation history in ALL conditions, including A (no KG). Since the setup messages sit 1-2 turns before the prompt, the LLM can trivially answer from chat history alone. The eval cannot measure KG benefit because Condition A gets the same facts.

## Fix 1: Two broken dataset items

### temporal_03
- Prompt: "What editor did I switch away from?"
- Setup: "I switched to Neovim about three years ago from VS Code"
- BUG: answer_key has required_facts=["Neovim"], expected_answer_contains=["Neovim"]
- FIX: required_facts=["VS Code"], expected_answer_contains=["VS Code"]
  (The question asks what was switched AWAY FROM, which is VS Code)

### temporal_06
- Prompt: "What conferences have I previously published at?"
- Setup: "I want to publish at ACL, it's a top venue"
- BUG: Setup expresses aspiration, not past publication. Inconsistent with prompt.
- FIX: Change setup to: "I published a paper at ACL last year on dialogue segmentation"
  And assistant response to: "ACL is a top venue for that work"
  This makes the setup consistent with the "previously published" question.

## Fix 2: Ablation redesign (Option A)

### Architecture change

setup_context is NO LONGER included in the message list for any condition. Instead:

1. **Before the eval run**: For each prompt, extract KG triples from setup_context into an eval-specific DB (or the live DB). This is a one-time cost. Use the real extraction pipeline.

2. **During the eval run**: For each prompt × condition:
   - Messages contain ONLY the eval prompt (single user message). No setup_context.
   - Condition A: `kg_context=False`. LLM sees only the bare prompt. No facts available.
   - Condition B: `kg_context=True`, `kg_max_derived=0`. KG injects relevant edges as a system message.
   - Condition C: `kg_context=True`, `kg_max_derived=3`. KG injects edges + closure-derived facts.

3. **Scoring**: Same as before — substring matching on required_facts and expected_answer_contains.

### Implementation

#### eval_ablation.py changes

```python
def run_ablation(dataset_path, db_path=None, model=None, conditions=None, dry_run=False):
    # Phase 1: Preload KG from setup_context (once, before conditions)
    # - Create or use eval DB
    # - For each prompt: insert setup_context as nodes, run extraction
    # - This populates kg_entities, kg_edges, kg_assertions, kg_mentions
    
    # Phase 2: Evaluate (per prompt × condition)
    for prompt_item in dataset:
        for condition in conditions:
            # Configure condition
            if condition == "A":
                config.set('kg_context', False)
            elif condition == "B":
                config.set('kg_context', True)
                config.set('kg_max_derived', 0)
            elif condition == "C":
                config.set('kg_context', True)
                config.set('kg_max_derived', 3)
            
            # Build messages — NO setup_context
            messages = []
            
            # Add KG context if enabled (conditions B, C)
            if condition != "A":
                kg_result = get_kg_context(prompt_item['prompt'], conn)
                if kg_result:
                    messages.insert(0, {"role": "system", "content": kg_result.text})
            
            # Add ONLY the eval prompt
            messages.append({"role": "user", "content": prompt_item['prompt']})
            
            # Call LLM, score
            ...
```

#### DB strategy

Two options, pick whichever is simpler:

**Option 1 (eval DB)**: Create a fresh SQLite DB for the eval. Insert setup_context messages as nodes, run KG extraction on each. This isolates eval from production data but costs ~50 LLM extraction calls.

**Option 2 (live DB + verify)**: Use the live production DB. Before the run, verify that the entities referenced in each prompt's answer_key exist in the KG. Skip prompts where entities are missing. This costs zero extraction calls but limits coverage to entities already extracted from real conversations.

Recommendation: **Option 1** (eval DB). It's deterministic — the eval is self-contained and reproducible regardless of production DB state. The ~50 extraction calls are a one-time cost, and the setup_context strings are short (1-2 sentences each), so extraction is cheap.

#### Preload implementation

```python
def preload_kg_from_dataset(dataset, conn):
    """Extract KG triples from all setup_context strings. One-time cost."""
    from episodic.kg.extractor import extract_from_text  # or whatever the entry point is
    
    node_id_counter = 1
    for item in dataset:
        for i, ctx in enumerate(item['setup_context']):
            if i % 2 == 0:  # Only extract from user messages
                # Insert as a node
                node_id = f"eval_{node_id_counter}"
                conn.execute(
                    "INSERT INTO nodes (node_id, role, content) VALUES (?, 'user', ?)",
                    (node_id, ctx)
                )
                # Run extraction
                extract_from_text(ctx, node_id, conn)
                node_id_counter += 1
    conn.commit()
```

### Expected outcomes

- **Condition A**: Should score LOW on most prompts. The LLM has no facts — just a bare question like "Can she run local models on it?" with no antecedent for "she" or "it".
- **Condition B**: Should score HIGHER. KG injects edges like `Emma has MacBook Pro M3 Max`, `MacBook Pro M3 Max has 64GB RAM`.
- **Condition C**: Should score HIGHEST on multi_hop and relational_bridging. Closure derives facts like `Emma located_at MIT` from `user related_to Emma` + `Emma located_at MIT`.

This is a clean, auditable measurement of the KG's causal contribution to answer quality.

### Deliverables

1. Fix temporal_03 and temporal_06 in eval_dataset.json
2. Rewrite eval_ablation.py to use Option A (no setup_context in messages, preload KG into eval DB)
3. Update tests in test_kg_eval.py
4. Run --dry-run to verify: condition A sees no KG block, conditions B/C see appropriate blocks
5. Report: dry-run output showing the three conditions for a few representative prompts
