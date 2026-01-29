"""
Maintenance commands for Episodic CLI.

Provides commands for offline/background maintenance tasks like
topic summarization, index rebuilding, etc.
"""

import typer
from typing import Optional

from episodic.configuration import get_heading_color, get_text_color, get_system_color
from episodic.color_utils import secho_color

app = typer.Typer(help="Maintenance commands for database upkeep")


@app.command("summarize")
def summarize_command(
    force: bool = typer.Option(False, "--force", "-f", help="Re-summarize all topics even if not stale"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Only summarize topics matching this name"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show what would be summarized without doing it"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use for summarization"),
):
    """
    Generate summaries for stale topics.

    Topics are considered "stale" if:
    - They have never been summarized, OR
    - They have grown by 4+ exchanges since last summary

    Examples:
        /maintenance summarize              # Summarize all stale topics
        /maintenance summarize --force      # Re-summarize all topics
        /maintenance summarize -t python    # Only topics containing "python"
    """
    from episodic.maintenance.summarization import (
        get_stale_topics,
        summarize_stale_topics,
    )

    # First show what's stale
    stale = get_stale_topics()

    if topic:
        stale = [t for t in stale if topic.lower() in t['name'].lower()]

    if not stale and not force:
        secho_color("No stale topics found. Use --force to re-summarize all.", fg=get_system_color())
        return

    if dry_run:
        secho_color(f"\n{'Would summarize' if not force else 'Would re-summarize'} {len(stale)} topics:", fg=get_heading_color())
        typer.echo("")

        for t in stale:
            name = t['name']
            node_count = t.get('node_count', 0)
            new_ex = t.get('new_exchanges', 0)
            has_summary = bool(t.get('existing_summary'))

            status = "has summary" if has_summary else "no summary"
            secho_color(f"  • {name}", fg=get_text_color(), bold=True)
            typer.echo(f"    {node_count} nodes, {new_ex} new exchanges, {status}")

        typer.echo("")
        secho_color("Use without --dry-run to execute.", fg=get_system_color())
        return

    # Actually summarize
    secho_color(f"\n📝 Summarizing {len(stale) if not force else 'all'} topics...", fg=get_heading_color())
    typer.echo("")

    results = summarize_stale_topics(
        force=force,
        model=model,
        topic_name_filter=topic
    )

    # Report results
    success_count = sum(1 for r in results if r.success and r.summary_md)
    skip_count = sum(1 for r in results if r.success and not r.summary_md)
    error_count = sum(1 for r in results if not r.success)

    for r in results:
        if r.success and r.summary_md:
            secho_color(f"✅ {r.topic_name}", fg="green")
            # Show first line of summary
            first_line = r.summary_md.split('\n')[0][:80]
            typer.echo(f"   {first_line}...")
        elif r.success:
            secho_color(f"⏭️  {r.topic_name} (skipped: {r.error})", fg="yellow")
        else:
            secho_color(f"❌ {r.topic_name}: {r.error}", fg="red")

    typer.echo("")
    secho_color(f"Summary: {success_count} summarized, {skip_count} skipped, {error_count} errors", fg=get_system_color())


@app.command("status")
def status_command():
    """
    Show maintenance status and what needs attention.
    """
    from episodic.maintenance.summarization import get_stale_topics
    from episodic.db_topics import get_all_topics

    secho_color("\n📊 Maintenance Status", fg=get_heading_color(), bold=True)
    typer.echo("=" * 40)

    # Topic summaries
    all_topics = get_all_topics()
    stale = get_stale_topics()

    typer.echo(f"\n📝 Topic Summaries:")
    typer.echo(f"   Total topics: {len(all_topics)}")
    typer.echo(f"   Stale topics: {len(stale)}")

    if stale:
        typer.echo(f"\n   Stale topics needing summarization:")
        for t in stale[:5]:  # Show first 5
            typer.echo(f"   • {t['name']} ({t.get('new_exchanges', 0)} new exchanges)")
        if len(stale) > 5:
            typer.echo(f"   ... and {len(stale) - 5} more")

    # Recommendations
    typer.echo("")
    if stale:
        secho_color("💡 Recommendation: Run '/maintenance summarize' to update topic summaries", fg=get_system_color())
    else:
        secho_color("✅ All topics are up to date", fg="green")


def maintenance_command(args: str = ""):
    """
    Main entry point for /maintenance command.

    Subcommands:
        summarize   - Generate summaries for stale topics
        status      - Show maintenance status
    """
    args_list = args.strip().split() if args.strip() else []

    if not args_list or args_list[0] in ["-h", "--help", "help"]:
        _show_help()
        return

    subcommand = args_list[0]
    remaining_args = args_list[1:]

    if subcommand == "summarize":
        # Parse arguments manually since we're not using typer's CLI
        force = "--force" in remaining_args or "-f" in remaining_args
        dry_run = "--dry-run" in remaining_args or "-d" in remaining_args

        # Extract --topic value
        topic = None
        for i, arg in enumerate(remaining_args):
            if arg in ("--topic", "-t") and i + 1 < len(remaining_args):
                topic = remaining_args[i + 1]
                break

        # Extract --model value
        model = None
        for i, arg in enumerate(remaining_args):
            if arg in ("--model", "-m") and i + 1 < len(remaining_args):
                model = remaining_args[i + 1]
                break

        summarize_command(force=force, topic=topic, dry_run=dry_run, model=model)

    elif subcommand == "status":
        status_command()

    else:
        typer.secho(f"Unknown subcommand: {subcommand}", fg="red")
        _show_help()


def _show_help():
    """Show help for maintenance command."""
    secho_color("\n🔧 Maintenance Commands", fg=get_heading_color(), bold=True)
    typer.echo("")
    typer.echo("Usage: /maintenance <subcommand> [options]")
    typer.echo("")
    typer.echo("Subcommands:")
    typer.echo("  summarize   Generate summaries for stale topics")
    typer.echo("  status      Show maintenance status")
    typer.echo("")
    typer.echo("Examples:")
    typer.echo("  /maintenance summarize           # Summarize stale topics")
    typer.echo("  /maintenance summarize --force   # Re-summarize all")
    typer.echo("  /maintenance summarize -t python # Only python topics")
    typer.echo("  /maintenance summarize --dry-run # Preview what would be done")
    typer.echo("  /maintenance status              # Show what needs attention")
