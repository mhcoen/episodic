# MCP Server Guide

Episodic includes a built-in [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that exposes conversation memory, knowledge base, and LLM capabilities to external AI clients such as Claude Desktop, Cursor, or custom agents.

## Prerequisites

The MCP server requires the `mcp` package:

```bash
pip install "mcp>=1.26.0,<2.0.0"
```

Episodic will tell you if the package is missing when you run `/mcp start`.

## Quick Start

```bash
# 1. Start the server
/mcp start

# 2. Create an auth token
/mcp token create my-agent

# 3. Copy the token — it is shown once and cannot be retrieved

# 4. Verify
/mcp status
```

The server starts in the background on `127.0.0.1:51983` by default.

## Server Management

### Starting

```bash
/mcp start                      # Background, default port (51983)
/mcp start --port 8080          # Custom port
/mcp start --host 0.0.0.0      # Bind to all interfaces (use with caution)
/mcp start --foreground         # Run in foreground (blocks CLI input)
```

You can also set defaults in configuration:

```bash
/set mcp_port 8080
/set mcp_host 127.0.0.1
```

Or via environment variables:

```bash
export EPISODIC_MCP_PORT=8080
export EPISODIC_MCP_HOST=127.0.0.1
```

### Stopping

```bash
/mcp stop
```

### Status

```bash
/mcp status
```

Shows whether the server is running, its PID, port, uptime, and conversation node count.

The server also exposes a `/health` endpoint:

```bash
curl http://127.0.0.1:51983/health
```

## Token Authentication

Every request to the MCP server (except `/health` and the SSE transport endpoints) requires a bearer token in the `Authorization` header.

### Creating Tokens

```bash
/mcp token create <client_id>
```

`client_id` is a label you choose to identify the client (e.g., `claude-desktop`, `my-script`). The command prints the token once:

```
Token created successfully.

  Token (save this — it will NOT be shown again):
  epk_v1_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345678901

  Token ID:  a1b2c3d4-e5f6-7890-abcd-ef1234567890
  Client:    claude-desktop
  Scopes:    all (unrestricted)
```

Save the token immediately. Only its SHA-256 hash is stored in the database; the plaintext cannot be recovered.

### Scoped Tokens

Restrict a token to specific tools:

```bash
/mcp token create readonly-agent --scopes get_topics,search_knowledge,search_memory
```

An empty scope list (the default) grants access to all tools.

### Listing Tokens

```bash
/mcp token list
```

Shows active (non-revoked, non-expired) tokens with their abbreviated ID, client, scopes, and creation time.

### Revoking Tokens

```bash
/mcp token revoke <token_id>
```

Revocation is immediate. The token ID is printed when the token is created and shown (abbreviated) by `/mcp token list`.

### Rotating Tokens

Replace a token without downtime:

```bash
# Immediate rotation (old token revoked instantly)
/mcp token rotate <token_id>

# Grace period — old token remains valid for 5 minutes
/mcp token rotate <token_id> --grace 300
```

The new token is printed; update your client configuration, then the old token expires.

## Connecting a Client

### Claude Desktop

Add to your Claude Desktop MCP configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "episodic": {
      "url": "http://127.0.0.1:51983/sse",
      "headers": {
        "Authorization": "Bearer epk_v1_your_token_here"
      }
    }
  }
}
```

### Generic MCP Client

Any MCP-compatible client can connect via SSE transport at `http://<host>:<port>/sse` with the `Authorization: Bearer <token>` header.

## Available Tools

The server exposes 9 tools, organized into three categories.

### Read-Only Tools

| Tool | Description |
|------|-------------|
| `mcp_get_model_info` | Current models and providers across all contexts (chat, detection, compression, synthesis, intent) |
| `mcp_get_runtime_state` | Curated runtime configuration — 9 safe keys, no secrets or API keys |
| `mcp_get_topics` | Conversation topics with name, node boundaries, and confidence. Accepts `limit` parameter (default 50) |

### Search Tools

| Tool | Description |
|------|-------------|
| `mcp_search_knowledge` | Search user-indexed documents via the RAG knowledge base. Requires RAG to be enabled (`/rag on`) |
| `mcp_search_memory` | Search conversation memory for relevant past exchanges |

### Stateful and Write Tools

| Tool | Description |
|------|-------------|
| `mcp_ask_llm_stateless` | One-shot LLM query with optional RAG and/or memory context. No conversation nodes are created |
| `mcp_create_thread` | Create a new conversation thread and receive a thread handle (shown once). Threads are stored in the conversation DAG |
| `mcp_ask_llm_stateful` | Send a message in a thread. Requires a valid thread handle with write permission. Appends user and assistant nodes to the thread's DAG |
| `mcp_index_document` | Index content into the RAG knowledge base with MCP provenance metadata. Accepts `content`, `source_name`, and `content_type` (text/markdown/code) |

### Stateful Conversations (Threads)

External clients can hold multi-turn conversations through the thread system:

```
1. Call mcp_create_thread → returns a thread_handle (save it)
2. Call mcp_ask_llm_stateful with the handle + your message
3. The server assembles context from the thread's history, calls the LLM,
   and appends both user and assistant nodes to the thread's DAG
4. Repeat step 2 for follow-up messages
```

Thread handles are cryptographic capabilities — they authenticate the caller and encode permissions. Like tokens, only SHA-256 hashes are stored.

## Traces (Audit Log)

Every tool call is recorded in the trace log with timing, status, and redacted parameters.

### Viewing Traces

```bash
/mcp traces                              # Last 20 traces
/mcp traces --limit 50                   # Last 50
/mcp traces --tool search_knowledge      # Filter by tool
```

Each trace records:
- Timestamp, duration, and status (ok/error)
- Tool name and client ID
- Redacted input parameters (keys matching `key`, `token`, `secret`, `password`, `credential`, `auth` are replaced with `[REDACTED]`)
- Input/output hashes and sizes
- Token counts and model information
- Error code and safe error message on failure

## Security Model

### Credential Storage

Tokens and thread handles use a hash-only storage model:

- Plaintext is generated using `secrets.token_bytes(32)` (cryptographically secure)
- Only the SHA-256 hash is written to the database
- Plaintext is displayed once on creation and never stored or logged

If the database is compromised, attackers get hashes, not usable credentials.

### Network Binding

The server binds to `127.0.0.1` (localhost only) by default. Binding to `0.0.0.0` exposes the server to the network — only do this behind a reverse proxy or firewall.

### Cost Limits

The auth middleware enforces a per-client daily cost limit (default $10/day). When a client exceeds the limit, requests return HTTP 429 until the next UTC day.

### Trace Redaction

Tool parameters are automatically redacted before trace storage. Any parameter key containing `key`, `token`, `secret`, `password`, `credential`, or `auth` is replaced with `[REDACTED]`. Nested dict values are recursively redacted.

### Authentication Middleware

All non-public endpoints require a valid bearer token. The middleware checks:

1. Presence of `Authorization: Bearer <token>` header
2. Token exists in the database (via hash lookup)
3. Token is not revoked
4. Token is not expired
5. Client's daily cost is within limits

Failed checks return 401 (missing header), 403 (invalid/revoked/expired), or 429 (cost limit).

## Client Mode (Consuming External MCP Servers)

Episodic can also act as an MCP *client*, connecting to external MCP servers to use their tools. This enables integration with filesystem servers, calendar servers, or any other MCP-compatible tool provider.

### Configuring External Servers

Add external servers to your configuration:

```bash
/set mcp_servers {"filesystem": {"command": "npx", "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed"], "env": {}, "lifecycle": "on-demand"}, "calendar": {"command": "python", "args": ["-m", "mcp_server_calendar"], "env": {"CALENDAR_TOKEN": "..."}, "lifecycle": "on-demand"}}
```

Or edit `~/.episodic/config.json` directly:

```json
{
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed"],
      "env": {},
      "lifecycle": "on-demand"
    },
    "calendar": {
      "command": "python",
      "args": ["-m", "mcp_server_calendar"],
      "env": {"CALENDAR_TOKEN": "..."},
      "lifecycle": "on-demand"
    }
  }
}
```

Each server entry has:

| Key | Description |
|-----|-------------|
| `command` | Executable to run (e.g., `npx`, `python`, `node`) |
| `args` | Command-line arguments |
| `env` | Extra environment variables (merged with current environment) |
| `lifecycle` | `on-demand` (connect when needed) or `persistent` (manual connect/disconnect) |

### Managing External Servers

```bash
/mcp servers                    # List configured servers and their status
/mcp connect <server_id>        # Connect to a server
/mcp disconnect <server_id>     # Disconnect from a server
/mcp tools                      # List all discovered tools from connected servers
/mcp tools <server_id>          # List tools from a specific server
```

### Tool Namespacing

Tools from external servers are namespaced as `server_id.tool_name` to prevent collisions:

```
filesystem.read_file
filesystem.write_file
filesystem.list_directory
calendar.list_events
calendar.create_event
```

### On-Demand Connections

Servers configured with `"lifecycle": "on-demand"` are automatically connected when their tools are needed. You don't need to manually connect before calling a tool.

### Tracing

External tool calls are recorded in the trace log with `direction: client_tool_call` (as opposed to `server_tool_call` for incoming requests). View them with:

```bash
/mcp traces                     # Shows both server and client traces
```

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `mcp_port` | `51983` | Server port |
| `mcp_host` | `127.0.0.1` | Server bind address |
| `mcp_servers` | `{}` | External MCP servers to connect to (see Client Mode above) |
| `EPISODIC_MCP_PORT` | `51983` | Environment variable override for port |
| `EPISODIC_MCP_HOST` | `127.0.0.1` | Environment variable override for host |

### Database Tables

The MCP server creates three tables in the Episodic database:

| Table | Purpose |
|-------|---------|
| `mcp_tokens` | Token hashes, client IDs, scopes, expiration, revocation |
| `mcp_thread_handles` | Thread handle hashes, thread IDs, permissions |
| `mcp_traces` | Tool call audit log |
| `mcp_cost_accounting` | Daily per-client cost tracking |

Tables are auto-created on first use. No manual migration is needed.

## Troubleshooting

### Server won't start

- **Port in use**: Try `/mcp start --port 8080` or check what's using port 51983
- **MCP package missing**: Install with `pip install "mcp>=1.26.0,<2.0.0"`
- **Stale pidfile**: If the server crashed, `/mcp status` will detect the stale pidfile and clean it up

### Token rejected

- Tokens are shown once on creation — if lost, create a new one
- Check `/mcp token list` to verify the token hasn't been revoked
- Ensure the `Authorization: Bearer <token>` header is set correctly (no extra whitespace)

### Tools return errors

- **RAG not available**: Enable RAG with `/rag on` before using `search_knowledge` or `index_document`
- **Memory not available**: The memory system initializes automatically but may not be ready immediately after startup
- **Thread handle invalid**: Handles are single-use credentials — verify you saved the handle from `create_thread`

### External server won't connect

- **Server not found**: Verify the server is listed in `mcp_servers` config — check with `/mcp servers`
- **Command not found**: Ensure the command (`npx`, `python`, etc.) is in your PATH
- **Connection timeout**: The external server may not be starting properly — check the command and args
- **MCP package missing**: The `mcp` package is required for client mode — install with `pip install "mcp>=1.26.0,<2.0.0"`

### Checking server health

```bash
curl -s http://127.0.0.1:51983/health | python -m json.tool
```

Returns server status, version, uptime, and node count. This endpoint requires no authentication.
