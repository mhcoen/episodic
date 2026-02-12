"""Google Workspace MCP server manifest."""

from episodic.mcp.plugins._protocol import ServerManifest

GSUITE_MANIFEST = ServerManifest(
    server_id="mcp-gsuite",
    display_name="Google Workspace",
    command="npx",
    args=["-y", "@anthropic/mcp-gsuite"],
    env_vars=["GSUITE_CREDENTIALS", "GSUITE_TOKEN"],
    connect_policy="manual",
)
