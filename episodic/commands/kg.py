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
    elif action == 'probe':
        kg_probe(list(args))
    else:
        typer.secho(f"Unknown KG action: {action}", fg=get_error_color())
        typer.secho(
            "Usage: /kg [status|visualize|entities|entity|edges|search|"
            "update|rebuild|skip|patch|stats|probe]",
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
        typer.secho("Knowledge graph tables not found. Run /kg update first.",
                     fg=get_warning_color())
        return

    entity_types = [opts.entity_type] if opts.entity_type else None
    predicates = [opts.relation] if opts.relation else None
    tags = [opts.tag] if opts.tag else None
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
            typer.secho(f"  Processing node {node_id} ({index}/{total})...",
                         fg=get_system_color())

        r = run_batch(lookback=opts.lookback, max_nodes=opts.max_nodes,
                      conn=conn, progress_callback=progress, dry_run=opts.dry_run)
        typer.secho("\nExtraction complete:", fg=get_heading_color(), bold=True)
        typer.secho(f"  Nodes processed: {r['nodes_processed']}", fg=get_text_color())
        typer.secho(f"  Patches applied: {r['patches_applied']}",
                     fg=get_success_color() if r['patches_applied'] else get_text_color())
        typer.secho(f"  Patches rejected: {r['patches_rejected']}",
                     fg=get_warning_color() if r['patches_rejected'] else get_text_color())
        typer.secho(f"  HWM: {r['hwm_before']} -> {r['hwm_after']}", fg=get_text_color())

        if r['errors']:
            typer.secho("\nRejection reasons:", fg=get_warning_color())
            for err in r['errors'][:10]:
                reason = err['reason'][:77] + '...' if len(err['reason']) > 80 else err['reason']
                typer.secho(f"  node {err['node_id']}: {reason}",
                             fg=get_text_color(), dim=True)


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

    r = run_rebuild(progress_callback=progress)
    typer.secho(f"\nRebuild complete:", fg=get_heading_color(), bold=True)
    typer.secho(f"  Nodes processed: {r['nodes_processed']}", fg=get_text_color())
    typer.secho(f"  Patches applied: {r['patches_applied']}",
                 fg=get_success_color() if r['patches_applied'] else get_text_color())
    typer.secho(f"  Patches rejected: {r['patches_rejected']}",
                 fg=get_warning_color() if r['patches_rejected'] else get_text_color())


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
    tc = get_text_color()
    typer.secho(f"Patch for node {node_id}:", fg=get_heading_color(), bold=True)
    typer.secho(f"  Applied:   {'yes' if applied else 'no'}", fg=tc)
    if rejection:
        typer.secho(f"  Rejected:  {rejection}", fg=get_warning_color())
    if model_id:
        typer.secho(f"  Model:     {model_id}", fg=tc)
    if time_ms:
        typer.secho(f"  Time:      {time_ms}ms", fg=tc)
    if patch_json and patch_json.strip():
        try:
            pretty = json.dumps(json.loads(patch_json), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pretty = patch_json
        typer.secho(f"\n{pretty}", fg=get_system_color())


def _safe_query(conn, sql, default=None):
    """Run a query, returning default on error."""
    try:
        return conn.execute(sql).fetchall()
    except Exception:
        return default if default is not None else []


def kg_stats() -> None:
    """Show comprehensive KG statistics."""
    from episodic.kg.db_kg import kg_tables_exist, _use_conn

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    tc = get_text_color()
    with _use_conn() as conn:
        typer.secho("KG Statistics", fg=get_heading_color(), bold=True)

        rows = _safe_query(conn,
            "SELECT entity_type, COUNT(*) FROM kg_entities "
            "GROUP BY entity_type ORDER BY entity_type")
        if rows:
            typer.secho("\n  Entities by type:", fg=tc)
            total = sum(r[1] for r in rows)
            for r in rows:
                typer.secho(f"    {r[0]}: {r[1]}", fg=tc)
            typer.secho(f"    total: {total}", fg=tc, dim=True)

        rows = _safe_query(conn,
            "SELECT predicate, COUNT(*) FROM kg_edges "
            "GROUP BY predicate ORDER BY predicate")
        if rows:
            typer.secho("\n  Edges by predicate:", fg=tc)
            total = sum(r[1] for r in rows)
            for r in rows:
                typer.secho(f"    {r[0]}: {r[1]}", fg=tc)
            typer.secho(f"    total: {total}", fg=tc, dim=True)

        rows = _safe_query(conn, "SELECT COUNT(*) FROM kg_assertions")
        if rows:
            typer.secho(f"\n  Assertions: {rows[0][0]}", fg=tc)

        rows = _safe_query(conn,
            "SELECT COUNT(*), SUM(CASE WHEN applied=1 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN applied=0 THEN 1 ELSE 0 END) FROM kg_patches")
        if rows and rows[0]:
            r = rows[0]
            typer.secho(f"  Patches: {r[0]} total "
                         f"({r[1] or 0} applied, {r[2] or 0} rejected)", fg=tc)

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
            typer.secho(f"\n  High-water mark: {hwm}", fg=tc)
            typer.secho(f"  Max user node:   {max_nid}", fg=tc)
            stale_fg = get_warning_color() if staleness else get_success_color()
            stale_str = f"{staleness} nodes behind" if staleness else "up to date"
            typer.secho(f"  Staleness:       {stale_str}", fg=stale_fg)
        except Exception:
            pass

        rows = _safe_query(conn, "SELECT COUNT(*) FROM kg_skiplist")
        if rows and rows[0][0]:
            typer.secho(f"  Skip list:       {rows[0][0]} nodes", fg=tc)


def _fetch_assertion_spans(conn, edges) -> dict:
    """Look up assertion span text for EdgeFacts. Returns {assertion_id: text}."""
    aids = {ef.assertion_id for ef in edges if ef.assertion_id is not None}
    if not aids:
        return {}
    cache = {}
    for aid in aids:
        try:
            row = conn.execute(
                "SELECT a.span_start, a.span_end, n.content "
                "FROM kg_assertions a JOIN nodes n ON n.rowid = a.source_node_id "
                "WHERE a.assertion_id = ?", (aid,)).fetchone()
            cache[aid] = row[2][row[0]:row[1]] if row and row[2] else "[no span]"
        except Exception:
            cache[aid] = "[no span]"
    return cache


def kg_probe(args: List[str]) -> None:
    """Dry-run get_kg_context() against the live DB and print diagnostics."""
    if not args:
        typer.secho("Usage: /kg probe <text>", fg=get_text_color())
        return

    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    from episodic.kg.context_source import get_kg_context

    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return

    with _use_conn() as conn:
        result = get_kg_context(' '.join(args), conn)

        if result is None:
            typer.secho("No KG context produced (no entity mentions detected).",
                         fg=get_text_color())
            return

        # Pre-fetch assertion spans for edges
        span_cache = _fetch_assertion_spans(conn, result.edges)

    hc, tc, sc = get_heading_color(), get_text_color(), get_system_color()

    typer.secho("Matched entities:", fg=hc, bold=True)
    for m in result.matched_entities:
        typer.secho(f"  e{m['entity_id']}: \"{m['surface_form']}\" "
                     f"(w={m['weight']:.1f})", fg=tc)

    if result.edges:
        typer.secho(f"\nEdges ({len(result.edges)}):", fg=hc, bold=True)
        for ef in result.edges:
            tags = f" [{', '.join(ef.tags)}]" if ef.tags else ""
            span = span_cache.get(ef.assertion_id, "[no span]")
            typer.secho(f"  {ef.subj_name} --{ef.predicate}--> {ef.obj_name}  "
                         f"rank={ef.rank_score:.3f}  node:{ef.source_node_id}"
                         f"{tags}", fg=tc)
            typer.secho(f"    \"{span}\"", fg=tc, dim=True)

    if result.derived:
        typer.secho(f"\nDerived ({len(result.derived)}):", fg=hc, bold=True)
        for df in result.derived:
            nodes = ", ".join(str(n) for n in df.source_node_ids)
            typer.secho(f"  ({df.rule}) {df.subj_name} --{df.predicate}--> "
                         f"{df.obj_name}  from nodes:[{nodes}]", fg=tc)

    typer.secho(f"\nInjected block ({result.budget_used}/{result.budget_total} "
                 f"tokens, cache: {result.cache_status}):", fg=hc, bold=True)
    typer.secho(result.text, fg=sc)
