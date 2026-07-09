"""MentionDictionary: surface-form -> entity-id lookup with incremental refresh.

Split out of context_source.py; re-exported there. Uses no config.
"""

import sqlite3
from typing import Optional

from episodic.kg.context_common import _normalize_surface, _get_cache_key


class MentionDictionary:
    """Maps normalized surface forms to entity IDs.

    Kept fresh incrementally: the common per-turn case (kg_realtime advancing
    the high-water mark by adding new entities/aliases) only fetches rows newer
    than what's already loaded, instead of re-scanning every KG row each turn.
    A full rebuild is forced only when membership can shrink or shift — a merge
    (merge_epoch bump), a rollback/reset (high-water mark not advancing), or a
    new curation.
    """
    def __init__(self):
        self._dict: dict[str, list[tuple[int, float, str]]] = {}
        self._hwm: str = ""                 # last cache key ("hwm:merge_epoch")
        self._hwm_int: int = -1             # numeric high-water mark component
        self._merge_epoch: str = ""
        self._max_entity_id: int = 0
        self._max_alias_id: int = 0
        self._max_curation_id: int = 0
        self._sorted_forms: Optional[list[str]] = None  # cached, invalidated on change

    def rebuild(self, conn: sqlite3.Connection) -> str:
        cache_key = _get_cache_key(conn)
        if cache_key == self._hwm:
            return "hit"

        parts = cache_key.split(':', 1)
        try:
            new_hwm_int = int(parts[0])
        except (ValueError, IndexError):
            new_hwm_int = -1
        merge_epoch = parts[1] if len(parts) > 1 else ""

        # Incremental is only safe when the graph strictly grew: the HWM
        # advanced, no merge happened, and no curation was added. Otherwise
        # entities/aliases may have been removed or reassigned — rebuild fully.
        need_full = (
            not self._hwm
            or new_hwm_int < 0
            or new_hwm_int <= self._hwm_int
            or merge_epoch != self._merge_epoch
            or self._has_new_curations(conn)
        )

        if need_full:
            self._full_rebuild(conn)
        else:
            self._incremental(conn)

        self._hwm = cache_key
        self._hwm_int = new_hwm_int
        self._merge_epoch = merge_epoch
        self._sorted_forms = None  # membership changed; recompute on next detect
        return "rebuilt"

    def _has_new_curations(self, conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute(
                "SELECT MAX(curation_id) FROM kg_curations"
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        max_id = row[0] if row and row[0] is not None else 0
        return max_id > self._max_curation_id

    def _full_rebuild(self, conn: sqlite3.Connection) -> None:
        new_dict: dict[str, list[tuple[int, float, str]]] = {}
        self._max_entity_id = 0
        self._max_alias_id = 0
        self._max_curation_id = 0
        try:
            for eid, name in conn.execute(
                "SELECT entity_id, canonical_name FROM kg_entities "
                "WHERE merged_into_entity_id IS NULL"
            ).fetchall():
                key = _normalize_surface(name)
                if key:
                    new_dict.setdefault(key, []).append((eid, 1.0, "canonical"))
                if eid > self._max_entity_id:
                    self._max_entity_id = eid
        except sqlite3.OperationalError:
            pass
        try:
            for aid, eid, alias in conn.execute(
                "SELECT alias_id, entity_id, alias FROM kg_entity_aliases"
            ).fetchall():
                key = _normalize_surface(alias)
                if key:
                    new_dict.setdefault(key, []).append((eid, 0.8, "alias"))
                if aid > self._max_alias_id:
                    self._max_alias_id = aid
        except sqlite3.OperationalError:
            pass
        self._apply_curations(conn, new_dict)
        self._dict = new_dict

    def _incremental(self, conn: sqlite3.Connection) -> None:
        try:
            for eid, name in conn.execute(
                "SELECT entity_id, canonical_name FROM kg_entities "
                "WHERE merged_into_entity_id IS NULL AND entity_id > ?",
                (self._max_entity_id,),
            ).fetchall():
                key = _normalize_surface(name)
                if key:
                    self._dict.setdefault(key, []).append((eid, 1.0, "canonical"))
                if eid > self._max_entity_id:
                    self._max_entity_id = eid
        except sqlite3.OperationalError:
            pass
        try:
            for aid, eid, alias in conn.execute(
                "SELECT alias_id, entity_id, alias FROM kg_entity_aliases "
                "WHERE alias_id > ?",
                (self._max_alias_id,),
            ).fetchall():
                key = _normalize_surface(alias)
                if key:
                    self._dict.setdefault(key, []).append((eid, 0.8, "alias"))
                if aid > self._max_alias_id:
                    self._max_alias_id = aid
        except sqlite3.OperationalError:
            pass
        # Curations are handled by the full-rebuild trigger (_has_new_curations),
        # so an incremental pass never needs to touch them.

    def _apply_curations(
        self, conn: sqlite3.Connection,
        target: dict[str, list[tuple[int, float, str]]],
    ) -> None:
        try:
            adds: dict[int, set[str]] = {}
            removes: dict[int, set[str]] = {}
            for cid, eid, ctype, value in conn.execute(
                "SELECT curation_id, entity_id, curation_type, value FROM kg_curations"
            ).fetchall():
                if cid > self._max_curation_id:
                    self._max_curation_id = cid
                if ctype == 'alias_add':
                    adds.setdefault(eid, set()).add(value)
                elif ctype == 'alias_remove':
                    removes.setdefault(eid, set()).add(value)
            for eid, alias_set in adds.items():
                for alias in alias_set - removes.get(eid, set()):
                    key = _normalize_surface(alias)
                    if key:
                        target.setdefault(key, []).append((eid, 0.8, "alias"))
        except sqlite3.OperationalError:
            pass

    def detect_mentions(
        self, user_text: str, max_entities: int = 5
    ) -> list[tuple[int, str, float]]:
        normalized_text = _normalize_surface(user_text)
        if not normalized_text:
            return []
        if self._sorted_forms is None:
            self._sorted_forms = sorted(self._dict.keys(), key=len, reverse=True)
        sorted_forms = self._sorted_forms
        matches: list[tuple[int, str, float]] = []
        seen_ids: set[int] = set()
        consumed: list[tuple[int, int]] = []
        for form in sorted_forms:
            if len(matches) >= max_entities:
                break
            start = 0
            while True:
                idx = normalized_text.find(form, start)
                if idx == -1:
                    break
                end = idx + len(form)
                if idx > 0 and normalized_text[idx - 1].isalnum():
                    start = end
                    continue
                if end < len(normalized_text) and normalized_text[end].isalnum():
                    start = end
                    continue
                if any(not (end <= cs or idx >= ce) for cs, ce in consumed):
                    start = end
                    continue
                for eid, weight, kind in self._dict[form]:
                    if eid not in seen_ids:
                        matches.append((eid, form, weight))
                        seen_ids.add(eid)
                        consumed.append((idx, end))
                        break
                start = end
        return matches[:max_entities]
