"""CLI command handlers for /kg promote and /kg quarantined."""
import time as _time

import typer

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color, get_system_color,
)


def _get_quarantined_assertions(conn) -> list:
    """Query all quarantined assertions with edge details."""
    try:
        return conn.execute("""
            SELECT a.assertion_id, s.canonical_name, e.predicate,
                   o.canonical_name, a.source_origin, a.source_node_id
            FROM kg_assertions a
            JOIN kg_edges e ON e.assertion_id = a.assertion_id
            JOIN kg_entities s ON e.subj_entity_id = s.entity_id
            JOIN kg_entities o ON e.obj_entity_id = o.entity_id
            WHERE a.quarantined = 1
            ORDER BY a.assertion_id
        """).fetchall()
    except Exception:
        return []


def kg_quarantined() -> None:
    """List quarantined KG assertions (without promotion prompt)."""
    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    with _use_conn() as conn:
        rows = _get_quarantined_assertions(conn)

    if not rows:
        typer.secho("No quarantined facts.", fg=get_success_color())
        return

    # Group by source_origin
    sources: dict[str, list] = {}
    for row in rows:
        origin = row[4] or 'unknown'
        sources.setdefault(origin, []).append(row)

    for origin, items in sources.items():
        typer.secho(
            f"\nQuarantined facts ({len(items)} items, source: {origin}):",
            fg=get_heading_color(), bold=True,
        )
        for i, (aid, subj, pred, obj, _origin, node_id) in enumerate(items, 1):
            typer.secho(
                f"  [{i}] ({subj}, {pred}, {obj})",
                fg=get_text_color(),
            )
            typer.secho(
                f"      Source: {_origin}  Node: {node_id}",
                fg=get_text_color(), dim=True,
            )


def kg_promote(args: list[str]) -> None:
    """Promote quarantined assertions to trusted status.

    Usage:
        /kg promote        - interactive: list and choose
        /kg promote all    - promote all quarantined assertions
        /kg promote <id>   - promote specific assertion ID
    """
    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    with _use_conn() as conn:
        rows = _get_quarantined_assertions(conn)

        if not rows:
            typer.secho("No quarantined facts to promote.", fg=get_success_color())
            return

        # Handle "promote all" argument (requires confirmation)
        if args and args[0].lower() == 'all':
            typer.secho(
                f"About to promote {len(rows)} quarantined assertion(s).",
                fg=get_warning_color(),
            )
            try:
                if input("Confirm? [y/N] ").strip().lower() != 'y':
                    typer.secho("Cancelled.", fg=get_text_color())
                    return
            except (EOFError, KeyboardInterrupt):
                typer.secho("\nCancelled.", fg=get_text_color())
                return
            _do_promote(conn, [r[0] for r in rows], rows)
            return

        # Handle specific assertion ID
        if args:
            try:
                target_aid = int(args[0])
            except ValueError:
                typer.secho(f"Invalid assertion ID: {args[0]}", fg=get_error_color())
                return
            matching = [r for r in rows if r[0] == target_aid]
            if not matching:
                typer.secho(
                    f"Assertion {target_aid} not found or not quarantined.",
                    fg=get_error_color(),
                )
                return
            _do_promote(conn, [target_aid], matching)
            return

        # Interactive mode: show list and ask
        typer.secho(
            f"\nQuarantined facts ({len(rows)} items):",
            fg=get_heading_color(), bold=True,
        )
        for i, (aid, subj, pred, obj, origin, node_id) in enumerate(rows, 1):
            typer.secho(
                f"  [{i}] ({subj}, {pred}, {obj})",
                fg=get_text_color(),
            )
            typer.secho(
                f"      Source: {origin}",
                fg=get_text_color(), dim=True,
            )

        typer.secho(
            "\nPromote: [a]ll  [1-N] individual  [n]one",
            fg=get_system_color(),
        )
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            typer.secho("\nCancelled.", fg=get_text_color())
            return

        if choice in ('n', 'none', ''):
            typer.secho("No changes made.", fg=get_text_color())
            return

        if choice in ('a', 'all'):
            _do_promote(conn, [r[0] for r in rows], rows)
            return

        # Parse individual selections (e.g., "1", "1,3", "1-3")
        selected_indices = _parse_selection(choice, len(rows))
        if not selected_indices:
            typer.secho("Invalid selection.", fg=get_error_color())
            return

        selected_aids = [rows[i][0] for i in selected_indices]
        selected_rows = [rows[i] for i in selected_indices]
        _do_promote(conn, selected_aids, selected_rows)


def _parse_selection(choice: str, max_n: int) -> list[int]:
    """Parse user selection like '1', '1,3', '1-3' into 0-based indices."""
    indices = []
    for part in choice.replace(' ', '').split(','):
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                for i in range(int(start), int(end) + 1):
                    if 1 <= i <= max_n:
                        indices.append(i - 1)
            except ValueError:
                return []
        else:
            try:
                i = int(part)
                if 1 <= i <= max_n:
                    indices.append(i - 1)
            except ValueError:
                return []
    return sorted(set(indices))


def _do_promote(conn, assertion_ids: list[int], rows: list) -> None:
    """Execute promotion for given assertion IDs."""
    now = _time.time()

    for aid in assertion_ids:
        old_origin = conn.execute(
            "SELECT source_origin FROM kg_assertions WHERE assertion_id = ?",
            (aid,)
        ).fetchone()
        old_origin_val = old_origin[0] if old_origin else ''

        conn.execute(
            "UPDATE kg_assertions SET quarantined = 0, "
            "source_origin = 'user_promoted_from_' || COALESCE(source_origin, '') "
            "WHERE assertion_id = ?",
            (aid,)
        )
        conn.execute(
            "INSERT INTO kg_promotions "
            "(assertion_id, promoted_at, promoted_by, source_origin) "
            "VALUES (?, ?, 'cli_user', ?)",
            (aid, now, old_origin_val)
        )

    conn.commit()
    typer.secho(
        f"Promoted {len(assertion_ids)} assertion(s).",
        fg=get_success_color(),
    )
