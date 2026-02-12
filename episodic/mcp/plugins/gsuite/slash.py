"""Slash command definitions for the gsuite plugin.

Only /cal and /email survive the refactor. All other calendar/email
slash commands are removed.
"""

from episodic.mcp.plugins._protocol import SlashCommand

GSUITE_SLASH_COMMANDS = [
    SlashCommand(
        name="/cal",
        aliases=["/calendar"],
        category="Calendar & Email",
        description="Calendar commands via natural language",
        domain="calendar",
        completions=["today", "tomorrow", "this week", "next week"],
    ),
    SlashCommand(
        name="/email",
        aliases=["/mail", "/gmail"],
        category="Calendar & Email",
        description="Email commands via natural language",
        domain="email",
        completions=["unread", "from", "about", "recent"],
    ),
]
