"""NetworkX graph construction from KG query results + Cytoscape JSON export."""

import networkx as nx
from typing import Optional

from .db_kg import (
    kg_tables_exist, get_all_entities, get_all_edges,
    get_entity_aliases, get_assertion_span_text,
)

# Edge color mapping by predicate
PREDICATE_COLORS = {
    'uses':       '#888888',
    'wants':      '#D4A03E',
    'prefers':    '#5BA0C9',
    'role':       '#C75A8A',
    'has':        '#6CC3A1',
    'located_at': '#E6B333',
    'part_of':    '#CC6699',
    'related_to': '#FF6666',
    'is_a':       '#99CCFF',
    'powered_by': '#FF9933',
}

DEFAULT_EDGE_COLOR = '#888888'


def build_kg_graph(
    entity_types: Optional[list[str]] = None,
    predicates: Optional[list[str]] = None,
    node_id_range: Optional[tuple[int, int]] = None,
    tags: Optional[list[str]] = None,
    conn=None,
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from KG data with optional filters."""
    G = nx.DiGraph()

    if not kg_tables_exist(conn):
        return G

    # Step 1: Add entity nodes
    entities = get_all_entities(conn)
    for ent in entities:
        etype = ent['entity_type']
        if entity_types and etype not in entity_types:
            continue

        entity_id = ent['entity_id']
        canonical_key = ent.get('canonical_key') or ''

        G.add_node(f"e{entity_id}", **{
            'entity_id': entity_id,
            'entity_type': etype,
            'canonical_name': ent['canonical_name'],
            'canonical_key': canonical_key,
            'aliases': get_entity_aliases(entity_id, conn),
            'created_node_id': ent.get('created_node_id'),
            'degree': 0,
            'is_user_self': canonical_key == 'user:self',
        })

    # Step 2: Add edges
    edges = get_all_edges(conn)
    for edge in edges:
        predicate = edge['predicate']
        if predicates and predicate not in predicates:
            continue

        a_node_id = edge.get('node_id')
        if node_id_range and a_node_id is not None:
            if a_node_id < node_id_range[0] or a_node_id > node_id_range[1]:
                continue

        edge_tags = edge.get('tags') or []
        if tags and not set(edge_tags).intersection(set(tags)):
            continue

        src = f"e{edge['subj_entity_id']}"
        tgt = f"e{edge['obj_entity_id']}"

        # Both endpoints must be in the graph
        if src not in G or tgt not in G:
            continue

        # Resolve span text
        span_text = None
        if a_node_id is not None:
            span_text = get_assertion_span_text(
                a_node_id, edge.get('span_start', 0), edge.get('span_end', 0), conn
            )

        G.add_edge(src, tgt, **{
            'edge_id': edge['edge_id'],
            'predicate': predicate,
            'assertion_id': edge['assertion_id'],
            'node_id': a_node_id,
            'span_text': span_text,
            'polarity': edge.get('polarity', 'affirm'),
            'tags': edge_tags,
            'has_time_past': 'TIME_PAST' in edge_tags,
            'has_sentiment_neg': 'SENTIMENT_NEG' in edge_tags,
        })

    # Step 3: Compute degree from visible edges
    for n in G.nodes():
        G.nodes[n]['degree'] = G.degree(n)

    # Step 4: Prune isolates (degree 0), keep user:self
    isolates = [
        n for n in list(G.nodes())
        if G.degree(n) == 0 and not G.nodes[n].get('is_user_self')
    ]
    G.remove_nodes_from(isolates)

    return G


def graph_to_cytoscape_json(G: nx.DiGraph) -> dict:
    """Convert NetworkX DiGraph to Cytoscape.js JSON format."""
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            'data': {
                'id': node_id,
                'entity_type': data.get('entity_type', ''),
                'canonical_name': data.get('canonical_name', ''),
                'canonical_key': data.get('canonical_key', ''),
                'aliases': data.get('aliases', []),
                'created_node_id': data.get('created_node_id'),
                'degree': data.get('degree', 0),
                'is_user_self': data.get('is_user_self', False),
            }
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        predicate = data.get('predicate', '')
        edges.append({
            'data': {
                'id': f"edge_{data.get('edge_id', '')}",
                'source': src,
                'target': tgt,
                'predicate': predicate,
                'assertion_id': data.get('assertion_id'),
                'node_id': data.get('node_id'),
                'span_text': data.get('span_text'),
                'polarity': data.get('polarity', 'affirm'),
                'tags': data.get('tags', []),
                'has_time_past': data.get('has_time_past', False),
                'has_sentiment_neg': data.get('has_sentiment_neg', False),
                'edgeColor': PREDICATE_COLORS.get(predicate, DEFAULT_EDGE_COLOR),
            }
        })

    return {'nodes': nodes, 'edges': edges}
