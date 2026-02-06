"""
News Command Handlers.

Handles news-related utility commands.
"""

import sqlite3
from typing import Optional

from ..types import UtilityQuery, UtilityResult
from ..providers.news import NewsProvider


# Global provider instance (initialized on first use)
_news_provider: Optional[NewsProvider] = None


def get_news_provider() -> NewsProvider:
    """Get or create the news provider."""
    global _news_provider
    if _news_provider is None:
        _news_provider = NewsProvider()
    return _news_provider


def handle_news_headlines(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle news_headlines command.

    Gets top headlines.

    Args in query:
        category: News category (optional, defaults to "general")
        count: Number of headlines (optional, defaults to 5)
    """
    provider = get_news_provider()

    # Configure provider
    provider.configure({})

    category = query.args.get("category", "general")
    count = query.args.get("count", 5)

    result = provider.get("news_headlines", {"category": category, "count": count})

    if result.status == "error":
        return UtilityResult.error(
            result.payload.get("error", "news_error"),
            result.speech_text,
        )

    return UtilityResult.ok(
        display=result.display_text,
        speech=result.speech_text,
        **result.payload,
    )


def handle_news_topic(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle news_topic command.

    Gets headlines for a specific topic/category.

    Args in query:
        category: News category (required)
        count: Number of headlines (optional, defaults to 5)
    """
    provider = get_news_provider()

    category = query.args.get("category")
    if not category:
        return UtilityResult.error(
            "missing_category",
            "Which news category? Try: general, tech, business, science, health, politics, world"
        )

    count = query.args.get("count", 5)

    result = provider.get("news_topic", {"category": category, "count": count})

    if result.status == "error":
        return UtilityResult.error(
            result.payload.get("error", "news_error"),
            result.speech_text,
        )

    return UtilityResult.ok(
        display=result.display_text,
        speech=result.speech_text,
        **result.payload,
    )


# Command routing
NEWS_HANDLERS = {
    "news_headlines": handle_news_headlines,
    "news_topic": handle_news_topic,
}


def dispatch_news_command(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """Dispatch a news category command to the appropriate handler."""
    handler = NEWS_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown news command: {query.command}"
        )

    return handler(query, conn, user_tz)
