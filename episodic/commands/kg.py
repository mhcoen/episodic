"""CLI command handler for /kg and subcommands."""

import os
import typer
from typing import List

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color, get_system_color,
)


def kg_command(action: str = None, *args: str) -> None:
    """Handle /kg command routing."""
    if action is None or action == 'status':
        kg_status()
    elif action == 'visualize':
        kg_visualize(list(args))
    elif action == 'entities':
        kg_entities()
    elif action == 'entity':
        kg_entity(list(args))
    elif action == 'edges':
        kg_edges(list(args))
    elif action == 'search':
        kg_search(list(args))
    elif action == 'update':
        kg_update(list(args))
    elif action == 'rebuild':
        kg_rebuild(list(args))
    elif action == 'skip':
        kg_skip(list(args))
    elif action == 'patch':
        kg_patch(list(args))
    elif action == 'stats':
        kg_stats()
    else:
        typer.secho(f"Unknown KG action: {action}", fg=get_error_color())
        typer.secho(
            "Usage: /kg [status|visualize|entities|entity|edges|search|"
            "update|rebuild|skip|patch|stats]",
            fg=get_text_color(),
        )


def kg_status() -> None:
    """Show KG status: table existence, entity count, edge count."""
    from episodic.kg.db_kg import kg_tables_exist

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        typer.secho(
            "The KG extraction pipeline has not been run yet.",
            fg=get_text_color(), dim=True,
        )
        return

    from episodic.kg.db_kg import get_all_entities, get_all_edges, get_node_id_range

    entities = get_all_entities()
    edges = get_all_edges()
    node_range = get_node_id_range()

    typer.secho("Knowledge Graph Status", fg=get_heading_color(), bold=True)
    typer.secho(f"  Entities: {len(entities)}", fg=get_text_color())
    typer.secho(f"  Edges:    {len(edges)}", fg=get_text_color())
    if node_range != (0, 0):
        typer.secho(
            f"  Node range: {node_range[0]} - {node_range[1]}",
            fg=get_text_color(),
        )


def kg_visualize(args: List[str]) -> None:
    """Launch KG visualization with optional flags."""
    import argparse

    parser = argparse.ArgumentParser(prog='/kg visualize', add_help=False)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument(
        '--layout', type=str, default='cose',
        choices=['cose', 'concentric', 'grid'],
    )
    parser.add_argument('--type', type=str, default=None, dest='entity_type')
    parser.add_argument('--relation', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)

    try:
        opts = parser.parse_args(args)
    except SystemExit:
        return  # argparse prints usage on error

    from episodic.kg.db_kg import kg_tables_exist
    from episodic.kg.visualize import visualize_kg

    if not kg_tables_exist():
        typer.secho(
            "Knowledge graph tables not found. Run /kg update first.",
            fg=get_warning_color(),
        )
        return

    entity_types = [opts.entity_type] if opts.entity_type else None
    predicates = [opts.relation] if opts.relation else None
    tags = [opts.tag] if opts.tag else None

    # Default save path if --save given without a value
    save_path = opts.save
    if save_path == '':
        import time
        save_dir = os.path.expanduser('~/.episodic/exports')
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        save_path = os.path.join(save_dir, f'kg-{ts}.html')

    path = visualize_kg(
        save_path=save_path,
        layout=opts.layout,
        entity_types=entity_types,
        predicates=predicates,
        tags=tags,
    )

    if save_path:
        typer.secho(f"Saved to: {path}", fg=get_success_color())
    else:
        typer.secho(f"Visualization opened ({path})", fg=get_system_color())


def kg_entities() -> None:
    """List all entities in the KG."""
    from episodic.kg.db_kg import kg_tables_exist, get_all_entities

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    entities = get_all_entities()
    if not entities:
        typer.secho("No entities in knowledge graph.", fg=get_text_color())
        return

    typer.secho(
        f"KG Entities ({len(entities)}):", fg=get_heading_color(), bold=True,
    )
    for ent in entities:
        etype = ent['entity_type']
        name = ent['canonical_name']
        eid = ent['entity_id']
        key = ent.get('canonical_key') or ''
        key_str = f" [{key}]" if key else ''
        typer.secho(
            f"  e{eid}: {name} ({etype}){key_str}", fg=get_text_color(),
        )


def kg_entity(args: List[str]) -> None:
    """Show detail for a single entity."""
    if not args:
        typer.secho("Usage: /kg entity <id>", fg=get_text_color())
        return

    from episodic.kg.db_kg import (
        kg_tables_exist, get_all_entities, get_entity_aliases, get_entity_degree,
    )

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    try:
        entity_id = int(args[0].lstrip('e'))
    except ValueError:
        typer.secho(f"Invalid entity ID: {args[0]}", fg=get_error_color())
        return

    entities = get_all_entities()
    ent = next((e for e in entities if e['entity_id'] == entity_id), None)
    if not ent:
        typer.secho(f"Entity {entity_id} not found.", fg=get_error_color())
        return

    aliases = get_entity_aliases(entity_id)
    degree = get_entity_degree(entity_id)

    typer.secho(
        f"Entity e{entity_id}: {ent['canonical_name']}",
        fg=get_heading_color(), bold=True,
    )
    typer.secho(f"  Type:    {ent['entity_type']}", fg=get_text_color())
    if ent.get('canonical_key'):
        typer.secho(f"  Key:     {ent['canonical_key']}", fg=get_text_color())
    typer.secho(
        f"  Aliases: {', '.join(aliases) if aliases else 'none'}",
        fg=get_text_color(),
    )
    typer.secho(f"  Degree:  {degree}", fg=get_text_color())


def kg_edges(args: List[str]) -> None:
    """List edges, optionally filtered by entity ID."""
    from episodic.kg.db_kg import kg_tables_exist, get_all_edges

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    edges = get_all_edges()
    if not edges:
        typer.secho("No edges in knowledge graph.", fg=get_text_color())
        return

    # Optional entity filter
    entity_id = None
    if args:
        try:
            entity_id = int(args[0].lstrip('e'))
        except ValueError:
            pass

    if entity_id is not None:
        edges = [
            e for e in edges
            if e['subj_entity_id'] == entity_id or e['obj_entity_id'] == entity_id
        ]

    typer.secho(f"KG Edges ({len(edges)}):", fg=get_heading_color(), bold=True)
    for edge in edges:
        src = edge['subj_entity_id']
        pred = edge['predicate']
        obj = edge['obj_entity_id']
        typer.secho(
            f"  e{src} --{pred}--> e{obj}", fg=get_text_color(),
        )


def kg_search(args: List[str]) -> None:
    """Search entities by name or alias."""
    if not args:
        typer.secho("Usage: /kg search <query>", fg=get_text_color())
        return

    from episodic.kg.db_kg import kg_tables_exist, get_all_entities, get_entity_aliases

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    query = ' '.join(args).lower()
    entities = get_all_entities()
    matches = []

    for ent in entities:
        name = (ent.get('canonical_name') or '').lower()
        key = (ent.get('canonical_key') or '').lower()
        aliases = [a.lower() for a in get_entity_aliases(ent['entity_id'])]

        if (query in name or query in key
                or any(query in a for a in aliases)):
            matches.append(ent)

    if not matches:
        typer.secho(f"No entities matching '{query}'.", fg=get_text_color())
        return

    typer.secho(
        f"Search results ({len(matches)}):",
        fg=get_heading_color(), bold=True,
    )
    for ent in matches:
        etype = ent['entity_type']
        name = ent['canonical_name']
        eid = ent['entity_id']
        typer.secho(f"  e{eid}: {name} ({etype})", fg=get_text_color())


def kg_update(args: List[str]) -> None:
    """Run batch extraction from current high-water mark."""
    import argparse

    parser = argparse.ArgumentParser(prog='/kg update', add_help=False)
    parser.add_argument('--max', type=int, default=None, dest='max_nodes')
    parser.add_argument('--lookback', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')

    try:
        opts = parser.parse_args(args)
    except SystemExit:
        return

    from episodic.kg.batch import run_batch, get_high_water_mark
    from episodic.kg.schema import ensure_kg_schema
    from episodic.kg.db_kg import _use_conn

    with _use_conn() as conn:
        ensure_kg_schema(conn)
        hwm = get_high_water_mark(conn)

        # Check if up to date
        try:
            max_row = conn.execute(
                "SELECT MAX(rowid) FROM nodes WHERE role = 'user'"
            ).fetchone()
            max_node_id = max_row[0] if max_row and max_row[0] else 0
        except Exception:
            max_node_id = 0

        if hwm >= max_node_id and max_node_id > 0:
            typer.secho("KG is up to date.", fg=get_success_color())
            return

        typer.secho(
            f"Starting extraction from HWM={hwm}"
            + (f" (max {opts.max_nodes} nodes)" if opts.max_nodes else "")
            + (" [dry run]" if opts.dry_run else ""),
            fg=get_heading_color(),
        )

        def progress(node_id, index, total):
            typer.secho(
                f"  Processing node {node_id} ({index}/{total})...",
                fg=get_system_color(),
            )

        result = run_batch(
            lookback=opts.lookback,
            max_nodes=opts.max_nodes,
            conn=conn,
            progress_callback=progress,
            dry_run=opts.dry_run,
        )

        typer.secho(
            f"\nExtraction complete:", fg=get_heading_color(), bold=True,
        )
        typer.secho(
            f"  Nodes processed: {result['nodes_processed']}",
            fg=get_text_color(),
        )
        typer.secho(
            f"  Patches applied: {result['patches_applied']}",
            fg=get_success_color() if result['patches_applied'] else get_text_color(),
        )
        typer.secho(
            f"  Patches rejected: {result['patches_rejected']}",
            fg=get_warning_color() if result['patches_rejected'] else get_text_color(),
        )
        typer.secho(
            f"  HWM: {result['hwm_before']} -> {result['hwm_after']}",
            fg=get_text_color(),
        )

        if result['errors']:
            typer.secho("\nRejection reasons:", fg=get_warning_color())
            for err in result['errors'][:10]:
                reason = err['reason']
                if len(reason) > 80:
                    reason = reason[:77] + '...'
                typer.secho(
                    f"  node {err['node_id']}: {reason}",
                    fg=get_text_color(), dim=True,
                )


def kg_rebuild(args: List[str]) -> None:
    """Full rebuild: drop all KG data and reprocess from scratch."""
    typer.secho(
        "This will delete all KG data and reprocess from scratch.",
        fg=get_warning_color(), bold=True,
    )

    try:
        confirm = input("Continue? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        typer.secho("\nCancelled.", fg=get_text_color())
        return

    if confirm.lower() != 'y':
        typer.secho("Cancelled.", fg=get_text_color())
        return

    from episodic.kg.batch import run_rebuild

    typer.secho("Rebuilding KG...", fg=get_heading_color())

    def progress(node_id, index, total):
        typer.secho(
            f"  Processing node {node_id} ({index}/{total})...",
            fg=get_system_color(),
        )

    result = run_rebuild(progress_callback=progress)

    typer.secho(
        f"\nRebuild complete:", fg=get_heading_color(), bold=True,
    )
    typer.secho(
        f"  Nodes processed: {result['nodes_processed']}",
        fg=get_text_color(),
    )
    typer.secho(
        f"  Patches applied: {result['patches_applied']}",
        fg=get_success_color() if result['patches_applied'] else get_text_color(),
    )
    typer.secho(
        f"  Patches rejected: {result['patches_rejected']}",
        fg=get_warning_color() if result['patches_rejected'] else get_text_color(),
    )


def kg_skip(args: List[str]) -> None:
    """Add node to skip list and advance HWM if stuck."""
    if not args:
        typer.secho("Usage: /kg skip <node_id> [--reason ...]", fg=get_text_color())
        return

    try:
        node_id = int(args[0])
    except ValueError:
        typer.secho(f"Invalid node ID: {args[0]}", fg=get_error_color())
        return

    reason = ''
    if '--reason' in args:
        idx = args.index('--reason')
        reason = ' '.join(args[idx + 1:])

    from episodic.kg.batch import add_to_skiplist

    add_to_skiplist(node_id, reason)
    typer.secho(
        f"Node {node_id} added to skip list.", fg=get_success_color(),
    )


def kg_patch(args: List[str]) -> None:
    """Show the patch record for a specific node."""
    if not args:
        typer.secho("Usage: /kg patch <node_id>", fg=get_text_color())
        return

    try:
        node_id = int(args[0])
    except ValueError:
        typer.secho(f"Invalid node ID: {args[0]}", fg=get_error_color())
        return

    import json
    from episodic.kg.db_kg import kg_tables_exist, _use_conn

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    with _use_conn() as conn:
        row = conn.execute(
            "SELECT patch_json, applied, rejection_reason, model_id, "
            "extraction_time_ms FROM kg_patches WHERE node_id = ?",
            (node_id,)
        ).fetchone()

    if not row:
        typer.secho(f"No patch found for node {node_id}.", fg=get_text_color())
        return

    patch_json, applied, rejection, model_id, time_ms = row

    typer.secho(
        f"Patch for node {node_id}:", fg=get_heading_color(), bold=True,
    )
    typer.secho(f"  Applied:   {'yes' if applied else 'no'}", fg=get_text_color())
    if rejection:
        typer.secho(f"  Rejected:  {rejection}", fg=get_warning_color())
    if model_id:
        typer.secho(f"  Model:     {model_id}", fg=get_text_color())
    if time_ms:
        typer.secho(f"  Time:      {time_ms}ms", fg=get_text_color())

    if patch_json and patch_json.strip():
        try:
            parsed = json.loads(patch_json)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            typer.secho(f"\n{pretty}", fg=get_system_color())
        except json.JSONDecodeError:
            typer.secho(f"\n{patch_json}", fg=get_system_color())


def kg_stats() -> None:
    """Show comprehensive KG statistics."""
    from episodic.kg.db_kg import kg_tables_exist, _use_conn

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    with _use_conn() as conn:
        # Entity counts by type
        typer.secho("KG Statistics", fg=get_heading_color(), bold=True)

        try:
            rows = conn.execute(
                "SELECT entity_type, COUNT(*) FROM kg_entities "
                "GROUP BY entity_type ORDER BY entity_type"
            ).fetchall()
            typer.secho("\n  Entities by type:", fg=get_text_color())
            total_ent = 0
            for row in rows:
                typer.secho(f"    {row[0]}: {row[1]}", fg=get_text_color())
                total_ent += row[1]
            typer.secho(f"    total: {total_ent}", fg=get_text_color(), dim=True)
        except Exception:
            pass

        # Edge counts by predicate
        try:
            rows = conn.execute(
                "SELECT predicate, COUNT(*) FROM kg_edges "
                "GROUP BY predicate ORDER BY predicate"
            ).fetchall()
            typer.secho("\n  Edges by predicate:", fg=get_text_color())
            total_edge = 0
            for row in rows:
                typer.secho(f"    {row[0]}: {row[1]}", fg=get_text_color())
                total_edge += row[1]
            typer.secho(
                f"    total: {total_edge}", fg=get_text_color(), dim=True,
            )
        except Exception:
            pass

        # Assertions
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM kg_assertions"
            ).fetchone()
            typer.secho(
                f"\n  Assertions: {row[0] if row else 0}", fg=get_text_color(),
            )
        except Exception:
            pass

        # Patches
        try:
            row = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN applied=1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN applied=0 THEN 1 ELSE 0 END) "
                "FROM kg_patches"
            ).fetchone()
            if row:
                typer.secho(
                    f"  Patches: {row[0]} total "
                    f"({row[1] or 0} applied, {row[2] or 0} rejected)",
                    fg=get_text_color(),
                )
        except Exception:
            pass

        # HWM vs max node_id
        try:
            hwm_row = conn.execute(
                "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
            ).fetchone()
            hwm = int(hwm_row[0]) if hwm_row else 0

            max_row = conn.execute(
                "SELECT MAX(rowid) FROM nodes WHERE role = 'user'"
            ).fetchone()
            max_nid = max_row[0] if max_row and max_row[0] else 0

            staleness = max(0, max_nid - hwm)
            typer.secho(
                f"\n  High-water mark: {hwm}", fg=get_text_color(),
            )
            typer.secho(
                f"  Max user node:   {max_nid}", fg=get_text_color(),
            )
            if staleness > 0:
                typer.secho(
                    f"  Staleness:       {staleness} nodes behind",
                    fg=get_warning_color(),
                )
            else:
                typer.secho(
                    "  Staleness:       up to date",
                    fg=get_success_color(),
                )
        except Exception:
            pass

        # Skip list
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM kg_skiplist"
            ).fetchone()
            skip_count = row[0] if row else 0
            if skip_count:
                typer.secho(
                    f"  Skip list:       {skip_count} nodes",
                    fg=get_text_color(),
                )
        except Exception:
            pass
