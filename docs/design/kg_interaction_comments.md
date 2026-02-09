# Comments on KG user interaction model (from earlier response)

This is mostly consistent with your spec, but it has two correctness problems and one missing interaction that will matter in practice.

## 1) “Primary interaction mode” is overstated

The “two channels” framing is correct (implicit context assembly + explicit CLI/visualization), but calling silent injection “the primary interaction mode and the one that justifies the KG’s existence” is too strong.

Phase 0 is explicitly audit-first (assertions with spans, CLI inspection, visualization). In demos, the explicit surfaces are often the point. Keep both as first-class.

## 2) “If the source conversation is correct, the KG should be correct” is false

Even if the conversation is correct, the extractor can still be wrong. Your design already acknowledges this (validation, skip list), but the text implies correctness follows from source correctness, which is not justified.

Replace with: the KG is intended to be a faithful projection of the conversation, but extraction errors can occur; explicit surfaces exist to audit and remediate them.

## 3) Remediation path is underspecified (and slightly wrong)

“Skip the bad node and let future context correct it, or rebuild” is incomplete.

Skipping a node prevents extraction from that turn. It does not correct any already-applied bad edges from that node unless you also purge them. You need a corrective remediation primitive that preserves replay determinism:

- `/kg retract <node_id>`: marks that node’s patch as retracted and deletes the KG rows derived from it, or forces a rebuild from that node onward. Pick one and specify it. Without this, “skip” is only preventative, not corrective.

## 4) “No user-facing edit interface” is fine, but alias/confirmation is still needed

Even in Phase 0, users will want to resolve obvious alias issues without editing edges. You can support this without creating a second source of truth:

- `/kg alias add <entity_id> "<alias>"`

Implement it as a logged event (a DAG node or a separate curation log that participates in replay). Otherwise you will rely on the LLM to propose aliases, which is worse.

## 5) Clarify who “user” is

Your system has an end-user who chats and a developer/operator who runs CLI commands. The interaction text mixes them. Split the model:

- End-user experience: implicit KG use; optionally visualization in a UI
- Operator/dev experience: CLI inspection and maintenance commands

## 6) Minor spec consistency points

- You mention Pyvis HTML explicitly; your spec says Plotly default with Pyvis optional. Either commit to Pyvis or keep it conditional.
- “Bridge entities across topics” is only meaningful once cross-topic entity resolution exists. In Phase 0, cross-topic resolution is disabled except canonical keys, so bridging is mostly limited to canonical-key entities. Note that limitation.
