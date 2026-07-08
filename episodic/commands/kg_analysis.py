"""KG analysis subcommands: merge, dupes, deadlines, eval.

Split out of commands/kg.py to keep it under the size limit. Re-imported there
so kg_command dispatch resolves. These have no kg.py-local dependencies.
"""

import os
from typing import List

import typer

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color, get_system_color,
)


def kg_merge(args: List[str]) -> None:
    """Merge two entities. Lower ID becomes survivor by default."""
    if len(args) < 2:
        typer.secho("Usage: /kg merge <id1> <id2> [--survivor=<id>]",
                     fg=get_text_color())
        return
    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    from episodic.kg.merge import merge_entities
    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return
    try:
        id1, id2 = int(args[0].lstrip('e')), int(args[1].lstrip('e'))
    except ValueError:
        typer.secho("Invalid entity IDs.", fg=get_error_color())
        return
    # Default: lower ID survives (older = canonical)
    survivor, merged = (id1, id2) if id1 < id2 else (id2, id1)
    for a in args[2:]:
        if a.startswith('--survivor='):
            try:
                s = int(a.split('=')[1].lstrip('e'))
                if s == id1:
                    survivor, merged = id1, id2
                elif s == id2:
                    survivor, merged = id2, id1
            except ValueError:
                pass
    tc = get_text_color()
    with _use_conn() as conn:
        # Show what we're merging
        for eid, label in [(survivor, "Survivor"), (merged, "Merged")]:
            row = conn.execute(
                "SELECT canonical_name, entity_type FROM kg_entities "
                "WHERE entity_id = ?", (eid,)).fetchone()
            if not row:
                typer.secho(f"Entity {eid} not found.", fg=get_error_color())
                return
            typer.secho(f"  {label}: e{eid} {row[0]} ({row[1]})", fg=tc)
        try:
            confirm = input("Merge? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            typer.secho("\nCancelled.", fg=tc)
            return
        if confirm.lower() != 'y':
            typer.secho("Cancelled.", fg=tc)
            return
        try:
            result = merge_entities(survivor, merged, "manual_merge", conn)
        except ValueError as e:
            typer.secho(f"Error: {e}", fg=get_error_color())
            return
    typer.secho("Merge complete:", fg=get_heading_color(), bold=True)
    for k in ('moved_edges', 'dropped_edges', 'moved_aliases',
              'dropped_aliases', 'moved_mentions'):
        typer.secho(f"  {k}: {result[k]}", fg=tc)


def kg_dupes() -> None:
    """Find duplicate entities (same canonical_name + type)."""
    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return
    with _use_conn() as conn:
        rows = conn.execute(
            "SELECT e1.entity_id, e2.entity_id, e1.canonical_name, e1.entity_type "
            "FROM kg_entities e1 "
            "JOIN kg_entities e2 ON e1.entity_id < e2.entity_id "
            "  AND LOWER(e1.canonical_name) = LOWER(e2.canonical_name) "
            "  AND e1.entity_type = e2.entity_type "
            "WHERE e1.merged_into_entity_id IS NULL "
            "  AND e2.merged_into_entity_id IS NULL "
            "ORDER BY e1.canonical_name"
        ).fetchall()
    if not rows:
        typer.secho("No duplicate entities found.", fg=get_success_color())
        return
    typer.secho(f"Duplicate entities ({len(rows)}):", fg=get_heading_color(), bold=True)
    for r in rows:
        typer.secho(f"  e{r[0]} + e{r[1]}: {r[2]} ({r[3]})", fg=get_text_color())


def kg_deadlines() -> None:
    """Show temporal edges (deadline, scheduled_for, starts_at, ends_at, recurring)."""
    from episodic.kg.db_kg import kg_tables_exist, _use_conn
    if not kg_tables_exist():
        typer.secho("Knowledge graph tables not found.", fg=get_warning_color())
        return
    with _use_conn() as conn:
        rows = conn.execute("""
            SELECT s.canonical_name, e.predicate, o.canonical_name, a.source_node_id
            FROM kg_edges e
            JOIN kg_entities s ON e.subj_entity_id = s.entity_id
            JOIN kg_entities o ON e.obj_entity_id = o.entity_id
            JOIN kg_assertions a ON e.assertion_id = a.assertion_id
            WHERE e.predicate IN ('deadline','scheduled_for','starts_at','ends_at','recurring')
              AND a.status = 'active'
              AND (a.quarantined = 0 OR a.quarantined IS NULL)
              AND s.merged_into_entity_id IS NULL
              AND o.merged_into_entity_id IS NULL
            ORDER BY e.predicate, s.canonical_name
        """).fetchall()
    if not rows:
        typer.secho("No temporal edges found.", fg=get_system_color())
        return
    typer.secho("Temporal edges:", fg=get_heading_color())
    for subj, pred, obj, node_id in rows:
        typer.secho(
            f"  {subj} {pred} {obj}  (node {node_id})", fg=get_text_color(),
        )



def kg_eval(args: List[str]) -> None:
    """Run KG ablation evaluation."""
    import argparse
    parser = argparse.ArgumentParser(prog='/kg eval', add_help=False)
    parser.add_argument('dataset', nargs='?', default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--conditions', type=str, default='A,B,C')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-preload', action='store_true')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter dataset (e.g., closure_expected)')
    try:
        opts = parser.parse_args(args)
    except SystemExit:
        return
    from episodic.kg.eval_ablation import (
        run_ablation, format_summary_table, save_results,
    )
    conds = [c.strip() for c in opts.conditions.split(',')]
    filter_closure = opts.filter == 'closure_expected'
    typer.secho("KG Ablation Evaluation" + (" [dry run]" if opts.dry_run else ""),
                fg=get_heading_color(), bold=True)
    summary = run_ablation(
        dataset_path=opts.dataset, model=opts.model,
        conditions=conds, dry_run=opts.dry_run,
        skip_preload=opts.skip_preload,
        filter_closure=filter_closure,
    )
    table = format_summary_table(summary, conds)
    typer.secho(f"\n{table}", fg=get_text_color())
    if not opts.dry_run:
        path = save_results(summary)
        typer.secho(f"\nFull results saved to: {path}", fg=get_success_color())
