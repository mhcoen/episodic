"""KG explain and blame subcommands."""

from typing import List

import typer

from episodic.configuration import (
    get_text_color, get_heading_color, get_warning_color, get_system_color,
)


def kg_explain_last() -> None:
    """Show what happened on the most recent KG context injection."""
    from episodic.kg.context_source import get_last_kg_result

    result = get_last_kg_result()
    if result is None:
        typer.secho(
            "No KG context injection recorded this session.",
            fg=get_warning_color(),
        )
        return

    hc, tc, sc = get_heading_color(), get_text_color(), get_system_color()

    typer.secho("KG Context Injection Report", fg=hc, bold=True)
    typer.secho("-" * 50, fg=tc)

    # Matched entities
    typer.secho("Matched entities:", fg=hc)
    for i, m in enumerate(result.matched_entities, 1):
        typer.secho(
            f"  {i}. e{m['entity_id']} \"{m['surface_form']}\" "
            f"(w={m['weight']:.1f})",
            fg=tc,
        )

    # Selected edges with rank scores
    if result.edges:
        typer.secho(f"\nSelected edges ({len(result.edges)}):", fg=hc)
        for ef in result.edges:
            tags = f" [{', '.join(ef.tags)}]" if ef.tags else ""
            typer.secho(
                f"  + {ef.subj_name} --{ef.predicate}--> {ef.obj_name}  "
                f"rank={ef.rank_score:.3f}  [node:{ef.source_node_id}]{tags}",
                fg=tc,
            )

    # Derived facts
    if result.derived:
        typer.secho(f"\nDerived rules fired ({len(result.derived)}):", fg=hc)
        for df in result.derived:
            nodes = ", ".join(str(n) for n in df.source_node_ids)
            typer.secho(
                f"  + ({df.rule}) {df.subj_name} --{df.predicate}--> "
                f"{df.obj_name}  [from nodes:{nodes}]",
                fg=tc,
            )
    else:
        typer.secho("\nDerived rules fired (0): (none)", fg=tc, dim=True)

    # Budget
    typer.secho(
        f"\nBudget: {result.budget_used}/{result.budget_total} tokens  "
        f"Cache: {result.cache_status}",
        fg=sc,
    )

    # Dropped edges
    if result.dropped_edges:
        typer.secho(f"\nDropped edges ({len(result.dropped_edges)}):", fg=hc)
        for ef in result.dropped_edges:
            typer.secho(
                f"  - {ef.subj_name} --{ef.predicate}--> {ef.obj_name}  "
                f"rank={ef.rank_score:.3f}",
                fg=tc, dim=True,
            )
    else:
        typer.secho("\nDropped edges: (none — all fit within budget)",
                     fg=tc, dim=True)

    if result.dropped_derived:
        typer.secho(f"\nDropped derived ({len(result.dropped_derived)}):", fg=hc)
        for df in result.dropped_derived:
            typer.secho(
                f"  - ({df.rule}) {df.subj_name} --{df.predicate}--> "
                f"{df.obj_name}",
                fg=tc, dim=True,
            )


def kg_blame(args: List[str]) -> None:
    """Show provenance for an edge from the last KG injection."""
    if not args:
        typer.secho("Usage: /kg blame <text>", fg=get_text_color())
        return

    from episodic.kg.context_source import get_last_kg_result

    result = get_last_kg_result()
    if result is None:
        typer.secho("No KG context injection recorded this session.",
                     fg=get_warning_color())
        return

    query = ' '.join(args).lower()
    hc, tc = get_heading_color(), get_text_color()

    # Search in edges
    for ef in result.edges:
        line = f"{ef.subj_name} {ef.predicate} {ef.obj_name}".lower()
        if query in line:
            _blame_edge(ef, hc, tc)
            return

    # Search in derived
    for df in result.derived:
        line = f"{df.subj_name} {df.predicate} {df.obj_name}".lower()
        if query in line:
            _blame_derived(df, hc, tc)
            return

    typer.secho(f"No matching edge for \"{' '.join(args)}\" in last injection.",
                 fg=get_warning_color())
    typer.secho("Hint: use part of the edge text, e.g. /kg blame Emma located_at",
                 fg=get_text_color(), dim=True)


def _blame_edge(ef, hc, tc) -> None:
    """Show provenance for a direct edge."""
    from episodic.kg.db_kg import _use_conn

    typer.secho(f"Edge: {ef.subj_name} {ef.predicate} {ef.obj_name}",
                 fg=hc, bold=True)
    typer.secho("-" * 50, fg=tc)
    typer.secho("Type: direct", fg=tc)
    typer.secho(f"Rank: {ef.rank_score:.3f}", fg=tc)
    if ef.tags:
        typer.secho(f"Tags: {ef.tags}", fg=tc)

    if ef.assertion_id is None:
        typer.secho("Assertion: (not recorded)", fg=tc, dim=True)
        return

    with _use_conn() as conn:
        row = conn.execute(
            "SELECT a.assertion_id, a.source_node_id, a.span_start, a.span_end, "
            "a.polarity, a.certainty, a.tags, n.content "
            "FROM kg_assertions a JOIN nodes n ON n.rowid = a.source_node_id "
            "WHERE a.assertion_id = ?",
            (ef.assertion_id,),
        ).fetchone()

    if not row:
        typer.secho(f"Assertion {ef.assertion_id}: (not found in DB)",
                     fg=tc, dim=True)
        return

    aid, nid, sp_start, sp_end, polarity, certainty, tags, content = row
    span_text = content[sp_start:sp_end] if content else "[no content]"

    typer.secho(f"Source: node {nid}, assertion {aid}", fg=tc)
    typer.secho(f"Polarity: {polarity}  Certainty: {certainty}", fg=tc)
    typer.secho(f"Span [{sp_start}:{sp_end}]: \"{span_text}\"",
                 fg=get_system_color())

    # Show mention details for subject and object
    with _use_conn() as conn:
        for label, name in [("Subject", ef.subj_name), ("Object", ef.obj_name)]:
            eid_row = conn.execute(
                "SELECT entity_id FROM kg_entities WHERE canonical_name = ?",
                (name,),
            ).fetchone()
            if not eid_row:
                continue
            mention = conn.execute(
                "SELECT span_start, span_end, surface_text, confidence "
                "FROM kg_mentions WHERE node_id = ? AND entity_id = ?",
                (nid, eid_row[0]),
            ).fetchone()
            if mention:
                typer.secho(
                    f"{label} mention: \"{mention[2]}\" "
                    f"[{mention[0]}:{mention[1]}] conf={mention[3]:.2f}",
                    fg=tc,
                )


def _blame_derived(df, hc, tc) -> None:
    """Show provenance for a derived edge."""
    from episodic.kg.db_kg import _use_conn

    typer.secho(f"Edge: {df.subj_name} {df.predicate} {df.obj_name}",
                 fg=hc, bold=True)
    typer.secho("-" * 50, fg=tc)
    typer.secho(f"Type: derived (rule: {df.rule})", fg=tc)

    if not df.source_node_ids:
        typer.secho("Source edges: (none recorded)", fg=tc, dim=True)
        return

    with _use_conn() as conn:
        for i, nid in enumerate(df.source_node_ids, 1):
            row = conn.execute(
                "SELECT content FROM nodes WHERE rowid = ?", (nid,)
            ).fetchone()
            content = row[0][:100] if row and row[0] else "[no content]"
            typer.secho(f"  Source {i}: node {nid}", fg=tc)
            typer.secho(f"    \"{content}\"", fg=tc, dim=True)
