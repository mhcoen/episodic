"""
This module provides visualization capabilities for the Episodic conversation DAG
using NetworkX and PyVis.
"""

import os
import networkx as nx
from pyvis.network import Network
import tempfile
from episodic.db import get_connection

def get_all_nodes():
    """
    Retrieve all nodes from the database.

    Returns:
        List of dictionaries containing node data (id, short_id, content, parent_id)
    """
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT id, short_id, content, parent_id FROM nodes")

            # Get column names from cursor description
            columns = [desc[0] for desc in c.description]
            rows = c.fetchall()

        # Create a list of dictionaries with column names as keys
        result = []
        for row in rows:
            node = {}
            for i, column in enumerate(columns):
                node[column] = row[i]
            result.append(node)
        return result
    except Exception as e:
        print(f"Error retrieving nodes from database: {str(e)}")
        return []

def visualize_dag(output_path=None, height="800px", width="100%"):
    """
    Create an interactive visualization of the conversation DAG.

    Args:
        output_path: Path to save the HTML file (default: temporary file)
        height: Height of the visualization (default: 800px)
        width: Width of the visualization (default: 100%)

    Returns:
        Path to the generated HTML file
    """
    # Get all nodes from the database
    nodes = get_all_nodes()

    if not nodes:
        print("No nodes found in the database. Add some messages first.")
        return None

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for node in nodes:
        # Ensure content is a string and truncate for display
        content_str = str(node["content"])
        display_content = content_str[:50] + "..." if len(content_str) > 50 else content_str
        # Include short ID in parentheses before the content
        display_content = f"({node['short_id']}) {display_content}"
        G.add_node(node["id"], title=content_str, label=display_content)
        if node["parent_id"]:
            G.add_edge(node["parent_id"], node["id"])

    # Create interactive visualization
    net = Network(height=height, width=width, directed=True, notebook=False)
    try:
        net.from_nx(G)
    except Exception as e:
        print(f"Error converting NetworkX graph to PyVis network: {str(e)}")
        raise

    # Add physics and interaction options
    options = {
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.01,
                "nodeDistance": 120
            },
            "solver": "hierarchicalRepulsion"
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed"
            }
        },
        "interaction": {
            "navigationButtons": True,
            "keyboard": True
        }
    }

    # Convert options to a JSON string for PyVis
    # PyVis expects a string representation of the options
    import json
    options_str = json.dumps(options)
    net.set_options(options_str)

    # Save to file
    if output_path is None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            output_path = tmp.name

    net.save_graph(output_path)
    return output_path
