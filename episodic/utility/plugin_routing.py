"""Plugin slash-command routing for utility commands.

Split out of cli_integration.py. Builds a UtilityQuery from a plugin's slash
command; re-imported into cli_integration so handle_utility_command resolves it.
"""

from typing import Optional

from .types import UtilityQuery
from .dispatcher import create_utility_query


def _handle_plugin_slash_command(sc, args_str: str, cmd: str) -> Optional[UtilityQuery]:
    """Build a UtilityQuery for a plugin slash command.

    Uses the extraction pipeline's matched_domains + domain scoping
    to create a query that will be dispatched via async MCP path.

    Since the user explicitly typed a slash command, we always produce
    a query — extraction refines it, but on failure or null intent we
    fall back to a simple passthrough query for the domain.
    """
    default_command = (
        f"{sc.domain}.query" if sc.domain == "calendar"
        else f"{sc.domain}.search"
    )

    if not args_str.strip():
        # No arguments — default to a basic query for the domain
        default_args: dict = {}
        if sc.domain == "email":
            default_args = {"unread_only": True}
        return create_utility_query(
            sc.domain, default_command,
            args=default_args, source="cli",
            raw_input=f"/{cmd}",
        )

    # Use extraction to parse the natural language args
    try:
        import asyncio
        from episodic.mcp.extraction import (
            extract_intent,
            check_dispatchability,
        )
        from episodic.mcp.extraction.prompt import get_intents_for_domains

        # Scope extraction to the slash command's domain
        domains = {sc.domain}
        intents = get_intents_for_domains(domains)

        async def _extract():
            return await extract_intent(
                args_str,
                matched_domains=domains,
                contacts={},
            )

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _extract())
                result = future.result()
        except RuntimeError:
            result = asyncio.run(_extract())

        verdict = check_dispatchability(result, intents)

        if verdict.dispatchable and verdict.intent:
            return create_utility_query(
                sc.domain, verdict.intent,
                args=verdict.args,
                source="cli",
                confidence=1.0,
                raw_input=f"/{cmd} {args_str}",
            )
        elif verdict.missing_required_args:
            missing = ", ".join(verdict.missing_required_args)
            from ..color_utils import secho_color
            secho_color(f"Missing required info: {missing}", fg="yellow")
            return None
        elif verdict.is_unknown_command:
            from ..color_utils import secho_color
            secho_color(
                f"Not sure what you mean by that. "
                f"Try: /{cmd} {', '.join(sc.completions[:3])}",
                fg="yellow",
            )
            return None
        # Null intent or other non-dispatchable: fall through to default
    except Exception:
        pass  # Extraction failure: fall through to default

    # Fallback: user typed /cmd <text>, pass text as raw query
    return create_utility_query(
        sc.domain, default_command,
        args={"query": args_str},
        source="cli",
        raw_input=f"/{cmd} {args_str}",
    )


