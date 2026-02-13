# Security Architecture

Episodic includes a defense-in-depth security pipeline that protects
against prompt injection, data exfiltration, knowledge graph poisoning,
and cross-mode contamination. The pipeline applies to both MCP tool
integration and web search synthesis (muse mode).

## Defense Layers

| Layer | Name | Purpose |
|-------|------|---------|
| L0 | Input Sanitization | Sanitize HTML (including e.g. hidden elements, comments, data attributes); NFC normalize Unicode; strip zero-width characters |
| L1 | Content Isolation | Wrap untrusted content in `<untrusted_content>` tags with source attribution |
| L2 | Action Gate | Require user confirmation before tool calls; escalate when web-derived content is present |
| L3 | Source Gate | Filter KG extraction by provenance; quarantine triples from untrusted sources |
| L4 | Canary Detection | Inject per-session canary tokens; detect and discard responses that leak them |
| L5 | Tool Firewall | Capability-based tool authorization and allowlisting |
| L6 | Outbound DLP | Detect and block exfiltration attempts via overlap and similarity checks |
| L7 | Provenance Tracking | Tag content provenance; prevent reuse of suspicious content across trust boundaries |
| L8 | Rate Limiting | Per-client rate and cost limits to prevent abuse |
| L9 | Request Binding | Message hash binding to prevent replay attacks |

## Muse Mode (Web Synthesis) Hardening

When muse mode fetches and synthesizes web content, additional
protections activate:

- **Synthesis isolation** — The synthesis LLM call has no tool access,
  preventing injected instructions from executing actions.
- **Prompt structure** — System instructions and untrusted content are
  separated into distinct message roles with anti-injection fencing.
- **Canary injection** — A per-session canary is embedded in the system
  message. If the LLM leaks it in the response, the output is discarded.
- **Provenance tagging** — Muse DAG nodes carry `source_type=web_synthesis`
  so downstream systems (context assembly, KG, RAG) can distinguish
  web-derived content from user conversation.
- **Cross-mode boundary** — When MCP tools are active, web-derived content
  in conversation context triggers enhanced confirmation for all tool calls.
- **RAG filtering** — Web-indexed RAG chunks carry `trust_level=untrusted`
  and are excluded from tool-enabled context by default.
- **KG quarantine** — Knowledge graph triples extracted from web synthesis
  responses are quarantined rather than added to the active graph.
- **Prompt budget** — Per-source and total character caps prevent attackers
  from dominating the synthesis prompt with adversarial content.
- **Single-normalization invariant** — All text is NFC-normalized exactly
  once at the ingestion boundary, preventing normalization-based bypasses.

## MCP Server Security

When running as an MCP server, Episodic applies the full L0-L9 pipeline
to all incoming tool calls. Key server-mode protections:

- **Token authentication** — SHA-256 hash-only credential storage with
  rotation support. Plaintext is shown once on creation and never stored.
- **Localhost binding** — Server binds to `127.0.0.1` by default.
- **Cost limits** — Configurable per-client daily cost limits.
- **Trace redaction** — Parameters containing keys, tokens, secrets, or
  passwords are automatically redacted in trace storage.
- **Input validation** — Server-mode input passes through the full
  sanitization and isolation pipeline before processing.

## Configuration

All security features are enabled by default. Key configuration options:

| Setting | Default | Description |
|---------|---------|-------------|
| `muse_sanitize_html` | `true` | Apply L0 sanitization to fetched HTML |
| `muse_isolate_content` | `true` | Wrap untrusted content in isolation tags |
| `muse_escalation_gate` | `true` | Enhanced confirmation when web-derived context present |
| `muse_kg_quarantine` | `true` | Quarantine KG triples from web synthesis |
| `muse_rag_in_tool_context` | `false` | Include untrusted RAG chunks in tool-enabled context |
| `muse_max_chars_per_source` | *(bounded)* | Per-source character cap in synthesis prompt |
| `muse_max_chars_total` | *(bounded)* | Total character cap for all sources |

## Specifications

Detailed security specifications are in the `specs/` and `devel/` directories:

- `devel/MCP_SECURITY_HARDENING.md` — Parent MCP security spec (L0-L9)
- `specs/MUSE_SECURITY_HARDENING.md` — Muse-mode security spec
- `specs/MUSE_SECURITY_HARDENING_v1.1_AMENDMENTS.md` — Adversarial review amendments
- `specs/L3_SOURCE_GATE_IMPL.md` — Source gate implementation spec
