"""
This module provides visualization capabilities for the Episodic conversation DAG
using NetworkX and PyVis.
"""

import os
import networkx as nx
from pyvis.network import Network
import tempfile
from episodic.db import get_connection, get_head

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

def visualize_dag(output_path=None, height="800px", width="100%", interactive=False, server_url=None):
    """
    Create an interactive visualization of the conversation DAG.

    Args:
        output_path: Path to save the HTML file (default: temporary file)
        height: Height of the visualization (default: 800px)
        width: Width of the visualization (default: 100%)
        interactive: Whether to include interactive features like double-click to set current node
        server_url: URL of the server for interactive features (default: http://localhost:5000)

    Returns:
        Path to the generated HTML file
    """
    # Set default server URL if not provided
    if interactive and server_url is None:
        server_url = "http://localhost:5000"
    # Get all nodes from the database
    nodes = get_all_nodes()

    if not nodes:
        print("No nodes found in the database. Add some messages first.")
        return None

    # Get the current head node
    current_node_id = get_head()

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for node in nodes:
        # Ensure content is a string and truncate for display
        content_str = str(node["content"])
        display_content = content_str[:50] + "..." if len(content_str) > 50 else content_str
        # Include short ID in parentheses before the content
        display_content = f"({node['short_id']}) {display_content}"

        # Set different color and border for current node
        if node["id"] == current_node_id:
            # For current node, use orange color and thicker border
            G.add_node(node["id"], title=content_str, label=display_content, 
                      # Use a more explicit color format
                      color={
                          "background": "#FFA500",  # Orange background
                          "border": "#FF8C00",      # Darker orange border
                          "highlight": {
                              "background": "#FFA500",
                              "border": "#FF8C00"
                          },
                          "hover": {
                              "background": "#FFA500",
                              "border": "#FF8C00"
                          }
                      },
                      borderWidth=3, borderWidthSelected=5,
                      # Add a custom attribute to identify this as the current node
                      is_current=True)
        else:
            # For non-current nodes, use default color
            G.add_node(node["id"], title=content_str, label=display_content, 
                      # Use a more explicit color format
                      color={
                          "background": "#97c2fc",  # Light blue background
                          "border": "#7c9fc9",      # Darker blue border
                          "highlight": {
                              "background": "#97c2fc",
                              "border": "#7c9fc9"
                          },
                          "hover": {
                              "background": "#97c2fc",
                              "border": "#7c9fc9"
                          }
                      },
                      is_current=False)

    # Add edges after all nodes are created
    # Keep track of root nodes (nodes without parents)
    root_nodes = []

    for node in nodes:
        if node["parent_id"]:
            G.add_edge(node["parent_id"], node["id"])
        else:
            # This is a root node
            root_nodes.append(node["id"])

    # If there are multiple root nodes, connect them to a virtual root
    # This ensures all nodes are connected in the visualization
    if len(root_nodes) > 1:
        # Create a virtual root node
        virtual_root_id = "virtual_root"
        G.add_node(virtual_root_id, label="(root)", title="Virtual root node", shape="dot", size=10, 
                  color={
                      "background": "#CCCCCC",  # Light gray background
                      "border": "#AAAAAA",      # Darker gray border
                      "highlight": {
                          "background": "#CCCCCC",
                          "border": "#AAAAAA"
                      },
                      "hover": {
                          "background": "#CCCCCC",
                          "border": "#AAAAAA"
                      }
                  })

        # Connect all root nodes to the virtual root
        for root_id in root_nodes:
            G.add_edge(virtual_root_id, root_id)

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
            "keyboard": True,
            "hover": True,  # Enable hover interactions
            "tooltipDelay": 200  # Show tooltips after 200ms hover
        },
        "nodes": {
            "font": {
                "size": 14,  # Larger font size for better readability
                "face": "arial"
            },
            "shape": "box",  # Box shape to better display text
            "margin": 10     # Margin around text
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.5  # Smaller arrows
                }
            }
        }
    }

    # Convert options to a JSON string for PyVis
    # PyVis expects a string representation of the options
    import json
    options_str = json.dumps(options)
    net.set_options(options_str)

    # Add custom JavaScript for interactive features if requested
    if interactive:
        # JavaScript for double-click to set current node
        custom_js = """
        // Add double-click event handler
        network.on("doubleClick", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];

                // Skip the virtual root node
                if (nodeId === "virtual_root") {
                    return;
                }

                // Show loading indicator
                document.body.style.cursor = 'wait';

                // Make AJAX call to set current node
                fetch('""" + server_url + """/set_current_node?id=' + nodeId, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    // Show success message
                    alert(data.message || 'Current node updated');

                    // Instead of reloading the page, update the node color directly
                    // First, reset all nodes to their default color
                    var allNodes = nodes.get({ returnType: "Object" });
                    for (var id in allNodes) {
                        if (allNodes[id].is_current) {
                            // Reset previously current node
                            allNodes[id].color = {
                                background: "#97c2fc",
                                border: "#7c9fc9",
                                highlight: {
                                    background: "#97c2fc",
                                    border: "#7c9fc9"
                                },
                                hover: {
                                    background: "#97c2fc",
                                    border: "#7c9fc9"
                                }
                            };
                            allNodes[id].is_current = false;
                        }
                    }

                    // Set the new current node color
                    var currentNode = allNodes[nodeId];
                    if (currentNode) {
                        currentNode.color = {
                            background: "#FFA500",
                            border: "#FF8C00",
                            highlight: {
                                background: "#FFA500",
                                border: "#FF8C00"
                            },
                            hover: {
                                background: "#FFA500",
                                border: "#FF8C00"
                            }
                        };
                        currentNode.is_current = true;

                        // Update the nodes in the network
                        nodes.update(Object.values(allNodes));

                        console.log("Updated current node color: " + nodeId);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error setting current node: ' + error);
                })
                .finally(() => {
                    // Reset cursor
                    document.body.style.cursor = 'default';
                });
            }
        });

        // Add title to the page and ensure current node is highlighted
        document.addEventListener('DOMContentLoaded', function() {
            var header = document.createElement('h1');
            header.textContent = 'Episodic Conversation Visualization';
            header.style.textAlign = 'center';
            header.style.marginBottom = '20px';

            var instructions = document.createElement('p');
            instructions.innerHTML = '<strong>Instructions:</strong> Double-click on a node to make it the current node. The current node is highlighted in orange.';
            instructions.style.textAlign = 'center';
            instructions.style.marginBottom = '20px';

            document.body.insertBefore(instructions, document.body.firstChild);
            document.body.insertBefore(header, document.body.firstChild);

            // Ensure current node is highlighted
            setTimeout(function() {
                // Find the current node (the one with is_current=true)
                var allNodes = nodes.get({ returnType: "Object" });
                for (var nodeId in allNodes) {
                    if (allNodes[nodeId].is_current) {
                        // Force the color to be orange
                        allNodes[nodeId].color = "#FFA500";
                        // Update the node in the network
                        nodes.update([allNodes[nodeId]]);
                        console.log("Highlighted current node: " + nodeId);
                        break;
                    }
                }
            }, 500);
        });
        """

        # Add the custom JavaScript to the visualization
        # Instead of trying to update options, we'll use a different approach
        # to add the custom JavaScript directly to the HTML
        net.html += f"<script>{custom_js}</script>"

    # Save to file
    if output_path is None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            output_path = tmp.name

    net.save_graph(output_path)
    return output_path
