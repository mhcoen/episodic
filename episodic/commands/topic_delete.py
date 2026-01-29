"""
Topic deletion command for Episodic.

Provides /topic delete subcommand to remove topics by:
- Exact name
- Pattern match
- Time range

Usage:
    /topic delete python-retry-mechanisms     # By exact name
    /topic delete --pattern "sourdough"       # By pattern match
    /topic delete --time "since yesterday"    # By time range
    /topic delete --time "between 10am and 2pm today"
"""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import (
    get_system_color, get_text_color, get_heading_color,
    get_warning_color, get_success_color
)
from episodic.db_topic_delete import (
    get_topics_by_name,
    get_topics_by_pattern,
    get_topics_by_time_range,
    delete_topics_batch,
    check_tables_exist
)
from episodic.utils.date_parser import parse_time_range, format_time_range


def topic_delete(
    name: Optional[str] = None,
    pattern: Optional[str] = None,
    time: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Delete topics matching the given criteria.

    Args:
        name: Exact topic name to delete
        pattern: Pattern to match topic names (case-insensitive)
        time: Natural language time expression (e.g., "since yesterday")
        force: Skip confirmation prompt
        dry_run: Show what would be deleted without deleting

    Returns:
        True if deletion was successful, False otherwise
    """
    # Check that at least one filter is provided
    if not name and not pattern and not time:
        typer.secho("Please specify what to delete:", fg=get_warning_color())
        typer.secho("  /topic delete <name>           Delete by exact name", fg=get_text_color())
        typer.secho("  /topic delete --pattern <pat>  Delete by pattern match", fg=get_text_color())
        typer.secho("  /topic delete --time <expr>    Delete by time range", fg=get_text_color())
        typer.echo()
        typer.secho("Examples:", fg=get_heading_color())
        typer.secho('  /topic delete python-retry-mechanisms', fg=get_text_color())
        typer.secho('  /topic delete --pattern "sourdough"', fg=get_text_color())
        typer.secho('  /topic delete --time "since yesterday"', fg=get_text_color())
        typer.secho('  /topic delete --time "between 10am and 2pm today"', fg=get_text_color())
        return False

    # Check required tables exist
    tables = check_tables_exist()
    if not tables.get('topics', False):
        typer.secho("No topics table found in database.", fg=get_warning_color())
        return False

    # Find topics to delete
    topics = []
    filter_desc = ""

    if name:
        topics = get_topics_by_name(name)
        filter_desc = f"name = '{name}'"
    elif pattern:
        topics = get_topics_by_pattern(pattern)
        filter_desc = f"pattern = '{pattern}'"
    elif time:
        user_tz = config.get("timezone", "America/Chicago")
        time_range = parse_time_range(time, user_tz=user_tz)
        if not time_range:
            typer.secho(f"Could not parse time expression: {time}", fg=get_warning_color())
            typer.secho("Examples: 'since yesterday', 'before last week', 'between 10am and 2pm today'", fg=get_text_color())
            return False

        start_utc, end_utc = time_range
        topics = get_topics_by_time_range(start_utc, end_utc)
        filter_desc = format_time_range(start_utc, end_utc, user_tz)

    # Report what was found
    if not topics:
        typer.secho(f"No topics found matching: {filter_desc}", fg=get_system_color())
        return True

    # Display topics that would be deleted
    typer.secho(f"\n{'Would delete' if dry_run else 'Topics to delete'} ({len(topics)} total):", fg=get_heading_color())
    typer.secho("=" * 60, fg=get_heading_color())

    for i, topic in enumerate(topics[:10]):  # Show max 10
        status = "" if topic.get('end_node_id') else " (ongoing)"
        created = topic.get('created_at', 'unknown')[:16] if topic.get('created_at') else 'unknown'
        typer.secho(f"  {i + 1}. {topic['name']}{status}", fg=get_text_color())
        typer.secho(f"      Created: {created}", fg=get_text_color(), dim=True)

    if len(topics) > 10:
        typer.secho(f"  ... and {len(topics) - 10} more", fg=get_text_color(), dim=True)

    typer.echo()

    if dry_run:
        typer.secho("(Dry run - no changes made)", fg=get_system_color())
        return True

    # Confirm deletion
    if not force:
        typer.secho(f"This will permanently delete {len(topics)} topic(s) and their associated data.", fg=get_warning_color())
        typer.secho("Conversation nodes will be preserved.", fg=get_text_color(), dim=True)
        confirm = typer.confirm("Proceed with deletion?")
        if not confirm:
            typer.secho("Cancelled.", fg=get_text_color())
            return False

    # Perform deletion
    typer.secho("Deleting topics...", fg=get_system_color())

    count, totals = delete_topics_batch(topics, delete_embeddings=True)

    # Report results
    typer.secho(f"\n{get_success_color()}Deleted {count} topic(s):", fg=get_success_color())
    typer.secho(f"  Topics table:        {totals['topics']}", fg=get_text_color())
    typer.secho(f"  Topic centroids:     {totals['centroids']}", fg=get_text_color())
    typer.secho(f"  Topic nodes:         {totals['topic_nodes']}", fg=get_text_color())
    typer.secho(f"  Working set entries: {totals['working_set']}", fg=get_text_color())
    typer.secho(f"  Embeddings:          {totals['embeddings']}", fg=get_text_color())

    return True


def handle_topic_delete(args_str: str) -> bool:
    """
    Handle /topic delete command with argument parsing.

    Args:
        args_str: Raw argument string from command

    Returns:
        True if successful
    """
    import shlex

    # Parse arguments
    try:
        args = shlex.split(args_str) if args_str else []
    except ValueError as e:
        typer.secho(f"Error parsing arguments: {e}", fg="red")
        return False

    # Extract options
    name = None
    pattern = None
    time = None
    force = False
    dry_run = False

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == '--pattern' and i + 1 < len(args):
            pattern = args[i + 1]
            i += 2
        elif arg == '--time' and i + 1 < len(args):
            time = args[i + 1]
            i += 2
        elif arg == '--force' or arg == '-f':
            force = True
            i += 1
        elif arg == '--dry-run' or arg == '-n':
            dry_run = True
            i += 1
        elif not arg.startswith('-') and name is None:
            name = arg
            i += 1
        else:
            typer.secho(f"Unknown argument: {arg}", fg="red")
            return False

    return topic_delete(name=name, pattern=pattern, time=time, force=force, dry_run=dry_run)
