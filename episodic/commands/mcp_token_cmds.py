"""MCP token management subcommands.

Split out of mcp_cmd.py. Re-imported there so mcp_command dispatch (mcp_token)
and mcp_traces (which uses _get_db_connection) resolve. Tests that patch the
token sub-dispatch or _get_db_connection must target this module, since
mcp_token calls its siblings in this namespace.
"""

import typer
from typing import List

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color,
)


def _get_db_connection():
    """Get a database connection for token operations."""
    import sqlite3
    from episodic.db_connection import get_db_path
    conn = sqlite3.connect(get_db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def mcp_token(args: List[str]) -> None:
    """Handle /mcp token subcommands."""
    if not args:
        mcp_token_list()
        return

    sub = args[0]
    sub_args = args[1:]

    if sub == "create":
        mcp_token_create(sub_args)
    elif sub == "revoke":
        mcp_token_revoke(sub_args)
    elif sub == "rotate":
        mcp_token_rotate(sub_args)
    elif sub == "list":
        mcp_token_list()
    else:
        typer.secho(f"Unknown token action: {sub}", fg=get_error_color())
        typer.secho(
            "Usage: /mcp token [create|revoke|rotate|list]",
            fg=get_text_color(),
        )


def mcp_token_create(args: List[str]) -> None:
    """Create a new MCP token."""
    if not args:
        typer.secho(
            "Usage: /mcp token create <client_id> [--scopes tool1,tool2]",
            fg=get_error_color(),
        )
        return

    client_id = args[0]
    scopes = None

    # Parse --scopes
    i = 1
    while i < len(args):
        if args[i] == "--scopes" and i + 1 < len(args):
            i += 1
            scopes = [s.strip() for s in args[i].split(",") if s.strip()]
        i += 1

    from episodic.mcp.auth import create_token

    conn = _get_db_connection()
    try:
        plaintext, token_id = create_token(conn, client_id, scopes)
    finally:
        conn.close()

    typer.secho("Token created successfully.", fg=get_success_color(), bold=True)
    typer.echo()
    typer.secho("  Token (save this — it will NOT be shown again):", fg=get_warning_color())
    typer.secho(f"  {plaintext}", fg=get_text_color(), bold=True)
    typer.echo()
    typer.secho(f"  Token ID:  {token_id}", fg=get_text_color())
    typer.secho(f"  Client:    {client_id}", fg=get_text_color())
    if scopes:
        typer.secho(f"  Scopes:    {', '.join(scopes)}", fg=get_text_color())
    else:
        typer.secho("  Scopes:    all (unrestricted)", fg=get_text_color())


def mcp_token_revoke(args: List[str]) -> None:
    """Revoke an MCP token."""
    if not args:
        typer.secho(
            "Usage: /mcp token revoke <token_id>",
            fg=get_error_color(),
        )
        return

    token_id = args[0]

    from episodic.mcp.auth import revoke_token

    conn = _get_db_connection()
    try:
        success = revoke_token(conn, token_id)
    finally:
        conn.close()

    if success:
        typer.secho(f"Token {token_id[:8]}... revoked.", fg=get_success_color())
    else:
        typer.secho(
            f"Token not found or already revoked: {token_id[:8]}...",
            fg=get_error_color(),
        )


def mcp_token_rotate(args: List[str]) -> None:
    """Rotate an MCP token."""
    if not args:
        typer.secho(
            "Usage: /mcp token rotate <token_id> [--grace <seconds>]",
            fg=get_error_color(),
        )
        return

    token_id = args[0]
    grace = 0

    # Parse --grace
    i = 1
    while i < len(args):
        if args[i] == "--grace" and i + 1 < len(args):
            i += 1
            try:
                grace = int(args[i])
            except ValueError:
                typer.secho(f"Invalid grace period: {args[i]}", fg=get_error_color())
                return
        i += 1

    from episodic.mcp.auth import rotate_token

    conn = _get_db_connection()
    try:
        result = rotate_token(conn, token_id, grace_seconds=grace)
    finally:
        conn.close()

    if result is None:
        typer.secho(
            f"Token not found or already revoked: {token_id[:8]}...",
            fg=get_error_color(),
        )
        return

    new_plaintext, new_token_id = result

    typer.secho("Token rotated successfully.", fg=get_success_color(), bold=True)
    typer.echo()
    typer.secho("  New token (save this — it will NOT be shown again):", fg=get_warning_color())
    typer.secho(f"  {new_plaintext}", fg=get_text_color(), bold=True)
    typer.echo()
    typer.secho(f"  New Token ID:  {new_token_id}", fg=get_text_color())
    if grace > 0:
        typer.secho(
            f"  Old token will remain valid for {grace}s.",
            fg=get_text_color(), dim=True,
        )
    else:
        typer.secho("  Old token revoked immediately.", fg=get_text_color(), dim=True)


def mcp_token_list() -> None:
    """List active MCP tokens."""
    from episodic.mcp.auth import list_tokens

    conn = _get_db_connection()
    try:
        tokens = list_tokens(conn)
    finally:
        conn.close()

    if not tokens:
        typer.secho("No active MCP tokens.", fg=get_text_color())
        typer.secho(
            "Create one with: /mcp token create <client_id>",
            fg=get_text_color(), dim=True,
        )
        return

    typer.secho(f"Active MCP Tokens ({len(tokens)}):", fg=get_heading_color(), bold=True)
    for tok in tokens:
        tid = tok["token_id"][:8]
        scopes = ", ".join(tok["scopes"]) if tok["scopes"] else "all"
        typer.secho(
            f"  {tid}...  client={tok['client_id']}  "
            f"scopes={scopes}  created={tok['created_at']}",
            fg=get_text_color(),
        )


