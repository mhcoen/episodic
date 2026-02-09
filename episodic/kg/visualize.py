"""Template rendering, file output, and display launch for KG visualization."""

import json
import os
import tempfile
import webbrowser
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from .graph_builder import build_kg_graph, graph_to_cytoscape_json, PREDICATE_COLORS
from .db_kg import get_node_id_range

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')

# Type color mapping for template rendering
TYPE_COLORS = {
    'person': '#4A90D9',
    'artifact': '#E8833A',
    'topic': '#7BC67E',
    'org': '#B07CD8',
}


def render_kg_html(
    entity_types: Optional[list[str]] = None,
    predicates: Optional[list[str]] = None,
    node_id_range: Optional[tuple[int, int]] = None,
    tags: Optional[list[str]] = None,
    layout: str = 'cose',
    conn=None,
) -> str:
    """Build the graph, export to JSON, render the Jinja2 template.

    Returns the complete HTML string.
    """
    G = build_kg_graph(entity_types, predicates, node_id_range, tags, conn)
    cy_json = graph_to_cytoscape_json(G)

    max_degree = max(
        (n['data']['degree'] for n in cy_json['nodes']), default=1
    )
    # Ensure max_degree is at least 1 to avoid mapData(0,0,...) issues
    if max_degree < 1:
        max_degree = 1

    full_range = get_node_id_range(conn)

    # Collect unique entity types and predicates from the actual data
    found_types = sorted({n['data']['entity_type'] for n in cy_json['nodes']})
    found_preds = sorted({e['data']['predicate'] for e in cy_json['edges']})
    # Use found types/preds if data exists, otherwise use defaults
    display_types = found_types if found_types else ['person', 'artifact', 'topic', 'org']
    display_preds = found_preds if found_preds else ['uses', 'wants', 'prefers', 'role', 'has', 'located_at', 'part_of', 'related_to', 'is_a', 'powered_by']

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    template = env.get_template('kg_graph.html')

    html = template.render(
        graph_json=json.dumps(cy_json),
        max_degree=max_degree,
        node_id_min=full_range[0],
        node_id_max=full_range[1],
        initial_layout=layout,
        entity_types=display_types,
        predicates=display_preds,
        type_colors=TYPE_COLORS,
        pred_colors=PREDICATE_COLORS,
    )
    return html


def visualize_kg(
    save_path: Optional[str] = None,
    layout: str = 'cose',
    entity_types: Optional[list[str]] = None,
    predicates: Optional[list[str]] = None,
    node_id_range: Optional[tuple[int, int]] = None,
    tags: Optional[list[str]] = None,
    conn=None,
) -> str:
    """Main entry point. Renders HTML and saves/displays it.

    Returns the path to the HTML file.
    """
    html = render_kg_html(entity_types, predicates, node_id_range, tags, layout, conn)

    if save_path:
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return save_path

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='episodic-kg-', delete=False,
        mode='w', encoding='utf-8',
    )
    tmp.write(html)
    tmp.close()

    webbrowser.open(f'file://{tmp.name}')

    return tmp.name
