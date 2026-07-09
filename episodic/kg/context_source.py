"""KG read-side context injection. No LLM calls on the read path."""

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from episodic.config import config
from episodic.debug_utils import debug_print

from episodic.kg.context_common import *  # noqa: F401,F403  (re-exported)
from episodic.kg.context_common import (  # underscore names
    _normalize_surface, _parse_tags, _get_cache_key, _PUNCT_RE,
)
from episodic.kg.mention_dict import MentionDictionary  # noqa: F401

def retrieve_neighborhood(
    entity_id: int, conn: sqlite3.Connection, user_text: str = "",
    co_mentioned_ids: Optional[set[int]] = None, max_edges: int = 5,
) -> list[EdgeFact]:
    include_past = config.get('kg_include_past', False) or bool(
        PAST_TENSE_CUES.search(user_text)
    )
    co_mentioned = co_mentioned_ids or set()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT e.edge_id, e.subj_entity_id, e.predicate, e.obj_entity_id,
                   e.assertion_id, a.source_node_id, a.tags,
                   s.canonical_name AS subj_name, o.canonical_name AS obj_name
            FROM kg_edges e
            JOIN kg_assertions a ON e.assertion_id = a.assertion_id
            JOIN kg_entities s ON e.subj_entity_id = s.entity_id
            JOIN kg_entities o ON e.obj_entity_id = o.entity_id
            WHERE (e.subj_entity_id = ? OR e.obj_entity_id = ?)
              AND a.status = 'active'
              AND (a.quarantined = 0 OR a.quarantined IS NULL)
              AND s.merged_into_entity_id IS NULL
              AND o.merged_into_entity_id IS NULL
            ORDER BY a.source_node_id DESC""",
            (entity_id, entity_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.row_factory = None
    seen_triples: dict[tuple, tuple] = {}
    for row in rows:
        tags = _parse_tags(row['tags'])
        if 'TIME_PAST' in tags and not include_past:
            continue
        triple = (row['subj_entity_id'], row['predicate'], row['obj_entity_id'])
        if triple not in seen_triples:
            seen_triples[triple] = (row, tags)
    facts: list[EdgeFact] = []
    for row, tags in seen_triples.values():
        other_id = (row['obj_entity_id'] if row['subj_entity_id'] == entity_id
                    else row['subj_entity_id'])
        co_bonus = 10.0 if other_id in co_mentioned else 0.0
        recency = row['source_node_id'] / 1_000_000.0
        pred_pri = PREDICATE_PRIORITY.get(row['predicate'], 5)
        facts.append(EdgeFact(
            subj_name=row['subj_name'], predicate=row['predicate'],
            obj_name=row['obj_name'], source_node_id=row['source_node_id'],
            rank_score=co_bonus + recency - pred_pri * 0.01,
            tags=tags, assertion_id=row['assertion_id'],
            subj_entity_id=row['subj_entity_id'],
            obj_entity_id=row['obj_entity_id'],
        ))
    facts.sort(key=lambda f: f.rank_score, reverse=True)
    return facts[:max_edges]


def _entity_id_by_name(name: str, conn: sqlite3.Connection) -> Optional[int]:
    try:
        row = conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_name = ?", (name,),
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def _query_edges_for(
    entity_id: int, predicate: str, conn: sqlite3.Connection
) -> list[dict]:
    try:
        return [
            {'obj_name': r[4], 'source_node_id': r[3]}
            for r in conn.execute(
                """SELECT e.subj_entity_id, e.predicate, e.obj_entity_id,
                       a.source_node_id, o.canonical_name AS obj_name
                FROM kg_edges e
                JOIN kg_assertions a ON e.assertion_id = a.assertion_id
                JOIN kg_entities o ON e.obj_entity_id = o.entity_id
                WHERE e.subj_entity_id = ? AND e.predicate = ?
                  AND a.status = 'active'
                  AND (a.quarantined = 0 OR a.quarantined IS NULL)""",
                (entity_id, predicate),
            ).fetchall()
        ]
    except sqlite3.OperationalError:
        return []


def _score_direct_edges(
    edges: list[EdgeFact], prompt_tokens: set[str], matched_id_set: set[int],
) -> int:
    """Recompute rank_score as direct_score on all edges. Returns max overlap."""
    max_overlap = 0
    for e in edges:
        e_tokens = compute_prompt_tokens(
            f"{e.subj_name} {e.predicate} {e.obj_name}")
        overlap = len(prompt_tokens & e_tokens)
        if overlap > max_overlap:
            max_overlap = overlap
        touch = 2 if (e.subj_entity_id in matched_id_set
                      or e.obj_entity_id in matched_id_set) else 0
        e.rank_score = 5 * overlap + 2 * touch + e.source_node_id / 1_000_000.0
    edges.sort(key=lambda e: e.rank_score, reverse=True)
    return max_overlap


def closure_score(
    d: DerivedFact, prompt_tokens: set[str], seed_ids: list[int],
    user_self_id: int,
) -> float:
    """Score closure-derived fact: seed_bonus + bridge_bonus + overlap."""
    seed_bonus = 0
    if d.source_seed_id in seed_ids[:3]:
        seed_bonus = 3
    elif d.source_seed_id == user_self_id:
        seed_bonus = 2
    bridge_bonus = 0
    if d.rule == 'KINSHIP_LOCATION' and (prompt_tokens & KINSHIP_CUES):
        bridge_bonus = 3
    elif d.rule == 'DEVICE_SPEC' and (prompt_tokens & DEVICE_CUES):
        bridge_bonus = 3
    d_tokens = compute_prompt_tokens(f"{d.subj_name} {d.predicate} {d.obj_name}")
    overlap = len(prompt_tokens & d_tokens)
    return seed_bonus + bridge_bonus + overlap


def apply_closure_rules(
    seed_ids: list[int], edges: list[EdgeFact],
    conn: sqlite3.Connection, max_derived: int = 3,
    prompt_tokens: set[str] | None = None,
    user_self_id: int = 0, derived_per_seed: int = 2,
) -> list[DerivedFact]:
    """Apply KINSHIP_LOCATION and DEVICE_SPEC from seed entities only."""
    if max_derived <= 0:
        return []
    if not user_self_id:
        try:
            row = conn.execute(
                "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
            ).fetchone()
            if not row:
                return []
            user_self_id = row[0]
        except sqlite3.OperationalError:
            return []

    candidates: list[DerivedFact] = []
    try:
        user_edges = conn.execute(
            """SELECT e.predicate, e.obj_entity_id, o.canonical_name, a.source_node_id
               FROM kg_edges e
               JOIN kg_assertions a ON e.assertion_id = a.assertion_id
               JOIN kg_entities o ON e.obj_entity_id = o.entity_id
               WHERE e.subj_entity_id = ? AND e.predicate IN ('related_to','has')
                 AND a.status='active'
                 AND (a.quarantined = 0 OR a.quarantined IS NULL)
                 AND o.merged_into_entity_id IS NULL""",
            (user_self_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        user_edges = []
    for pred, obj_id, obj_name, src_node in user_edges:
        if pred == 'related_to':
            for loc in _query_edges_for(obj_id, 'located_at', conn):
                candidates.append(DerivedFact(
                    subj_name=obj_name, predicate='located_at',
                    obj_name=loc['obj_name'], rule='KINSHIP_LOCATION',
                    source_node_ids=[src_node, loc['source_node_id']],
                    source_seed_id=user_self_id,
                    intermediate_id=obj_id, intermediate_name=obj_name,
                ))
        elif pred == 'has':
            for spec in _query_edges_for(obj_id, 'has', conn):
                candidates.append(DerivedFact(
                    subj_name=obj_name, predicate='has',
                    obj_name=spec['obj_name'], rule='DEVICE_SPEC',
                    source_node_ids=[src_node, spec['source_node_id']],
                    source_seed_id=user_self_id,
                    intermediate_id=obj_id, intermediate_name=obj_name,
                ))
    seen: set[tuple[str, str, str]] = set()
    unique = [c for c in candidates
              if (t := (c.subj_name, c.predicate, c.obj_name)) not in seen
              and not seen.add(t)]
    p_tokens = prompt_tokens or set()
    for c in unique:
        c.relevance_score = closure_score(c, p_tokens, seed_ids, user_self_id)
    unique.sort(key=lambda c: (c.relevance_score, max(c.source_node_ids, default=0)),
                reverse=True)
    per_seed: dict[int, int] = {}
    capped: list[DerivedFact] = []
    for c in unique:
        cnt = per_seed.get(c.source_seed_id, 0)
        if cnt >= derived_per_seed:
            continue
        per_seed[c.source_seed_id] = cnt + 1
        capped.append(c)
        if len(capped) >= max_derived:
            break
    return capped


def _budget_edges(
    direct_edges: list[EdgeFact], matched_ids: list[int], budget: int = 12,
) -> list[EdgeFact]:
    """Guarantee 2 edges per seed, fill rest by global rank_score."""
    guaranteed: dict[int, list[EdgeFact]] = {}
    for eid in matched_ids:
        guaranteed[eid] = [e for e in direct_edges
                           if e.subj_entity_id == eid or e.obj_entity_id == eid][:2]
    used = {id(e) for g in guaranteed.values() for e in g}
    remaining = [e for e in direct_edges if id(e) not in used]
    fill_n = budget - sum(len(g) for g in guaranteed.values())
    result = [e for g in guaranteed.values() for e in g]
    result.extend(remaining[:max(0, fill_n)])
    result.sort(key=lambda e: e.rank_score, reverse=True)
    return result


def format_kg_context(
    facts: list[EdgeFact], derived_facts: list[DerivedFact],
    budget_tokens: int = 500,
) -> tuple[str, list[EdgeFact], list[DerivedFact]]:
    """Format facts within token budget. Returns (text, dropped, dropped_derived)."""
    if not facts and not derived_facts:
        return "", [], []
    entries: list[tuple[float, str, EdgeFact | DerivedFact]] = []
    for f in facts:
        line = f"- {f.subj_name} {f.predicate} {f.obj_name} [node:{f.source_node_id}]"
        entries.append((f.rank_score, line, f))
    for d in derived_facts:
        nodes_str = " + ".join(f"node:{nid}" for nid in d.source_node_ids)
        line = f"- (derived:{d.rule}) {d.subj_name} {d.predicate} {d.obj_name} [from {nodes_str}]"
        entries.append((999.0, line, d))
    entries.sort(key=lambda x: x[0], reverse=True)
    header = "Known facts about entities mentioned:"
    tokens_used = len(header) // 4
    result_lines: list[str] = []
    dropped_edges: list[EdgeFact] = []
    dropped_derived: list[DerivedFact] = []
    for _rank, line, source in entries:
        line_tokens = len(line) // 4
        if tokens_used + line_tokens > budget_tokens:
            (dropped_edges if isinstance(source, EdgeFact)
             else dropped_derived).append(source)
            continue
        result_lines.append(line)
        tokens_used += line_tokens
    if not result_lines:
        return "", list(facts), list(derived_facts)
    return header + "\n" + "\n".join(result_lines), dropped_edges, dropped_derived


_mention_dict = MentionDictionary()
_last_kg_result: Optional[KGContextResult] = None


def get_last_kg_result() -> Optional[KGContextResult]:
    """Return the most recent KGContextResult."""
    return _last_kg_result


def get_kg_context(
    user_text: str, conn: sqlite3.Connection,
) -> Optional[KGContextResult]:
    """Main entry: detect mentions, retrieve edges, apply closure, format."""
    try:
        if conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
        ).fetchone() is None:
            debug_print("KG tables not found, skipping", category='kg')
            return None
    except sqlite3.OperationalError:
        return None

    max_entities = config.get('kg_max_entities', 5)
    max_edges = config.get('kg_max_edges', 5)
    max_derived = config.get('kg_max_derived', 3)
    budget = config.get('kg_budget', 500)
    edges_per_entity = config.get('kg_edges_per_entity', 4)
    relevance_gate = config.get('kg_relevance_gate', True)
    seed_limit = config.get('kg_closure_seed_limit', 3)
    derived_per_seed = config.get('kg_derived_per_seed', 2)

    cache_status = _mention_dict.rebuild(conn)
    matches = _mention_dict.detect_mentions(user_text, max_entities=max_entities)

    has_first_person = bool(FIRST_PERSON_CUES.search(user_text))
    matched_eids = {eid for eid, _, _ in matches}
    user_self_id = _entity_id_by_name('<user>', conn) or 0
    if has_first_person and user_self_id and user_self_id not in matched_eids:
        matches.insert(0, (user_self_id, '<user>', 1.0))

    if not matches:
        debug_print("No entity mentions detected", category='kg')
        return None

    debug_print(
        f"Matched {len(matches)} entities: "
        + ", ".join(f"{form}(id={eid})" for eid, form, _ in matches),
        category='kg',
    )

    matched_ids = [eid for eid, _, _ in matches]
    matched_id_set = set(matched_ids)

    seed_ids = matched_ids[:seed_limit]
    if has_first_person and user_self_id and user_self_id not in seed_ids:
        seed_ids.append(user_self_id)

    co_mentioned = set(matched_ids)
    all_facts: list[EdgeFact] = []
    for entity_id, _form, _weight in matches:
        all_facts.extend(retrieve_neighborhood(
            entity_id, conn, user_text=user_text,
            co_mentioned_ids=co_mentioned - {entity_id},
            max_edges=max_edges,
        ))

    seen_triples: set[tuple[str, str, str]] = set()
    deduped: list[EdgeFact] = []
    for f in all_facts:
        triple = (f.subj_name, f.predicate, f.obj_name)
        if triple not in seen_triples:
            seen_triples.add(triple)
            deduped.append(f)

    # Compute prompt tokens once
    prompt_tokens = compute_prompt_tokens(user_text)

    # Score direct edges and get max overlap for gating
    max_overlap = _score_direct_edges(deduped, prompt_tokens, matched_id_set)

    # Budget allocation: guarantee 2 per seed, fill by score
    total_budget = len(matched_ids) * edges_per_entity
    all_facts = _budget_edges(deduped, matched_ids, budget=total_budget)

    # Closure from seeds only
    derived = apply_closure_rules(
        seed_ids, all_facts, conn, max_derived=max_derived,
        prompt_tokens=prompt_tokens, user_self_id=user_self_id,
        derived_per_seed=derived_per_seed,
    )
    derived = [d for d in derived
               if (d.subj_name, d.predicate, d.obj_name) not in seen_triples]

    debug_print(
        f"Retrieved {len(all_facts)} edges, {len(derived)} derived facts",
        category='kg',
    )

    # Revised gating: suppress if max_overlap==0 AND no bridge-cued closure
    global _last_kg_result
    has_bridge_cue = bool(prompt_tokens & (KINSHIP_CUES | DEVICE_CUES))
    has_seeded_closure = bool(derived) and has_bridge_cue
    if relevance_gate and max_overlap == 0 and not has_seeded_closure:
        debug_print("KG block suppressed: no relevant edges", category='kg')
        result = KGContextResult(
            text="", matched_entities=[
                {'entity_id': eid, 'surface_form': form, 'weight': w}
                for eid, form, w in matches],
            edge_count=len(all_facts), derived_count=len(derived),
            budget_used=0, budget_total=budget, cache_status=cache_status,
            edges=all_facts, derived=derived,
            suppressed=True, suppressed_reason="no_relevant_edges",
        )
        _last_kg_result = result
        return result

    text, dropped_edges, dropped_derived = format_kg_context(
        all_facts, derived, budget_tokens=budget
    )
    if not text:
        return None

    result = KGContextResult(
        text=text,
        matched_entities=[
            {'entity_id': eid, 'surface_form': form, 'weight': w}
            for eid, form, w in matches],
        edge_count=len(all_facts), derived_count=len(derived),
        budget_used=len(text) // 4, budget_total=budget,
        cache_status=cache_status, edges=all_facts, derived=derived,
        dropped_edges=dropped_edges, dropped_derived=dropped_derived,
    )
    _last_kg_result = result
    return result
