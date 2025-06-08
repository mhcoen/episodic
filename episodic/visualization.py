"""
This module provides visualization capabilities for the Episodic conversation DAG
using NetworkX and Plotly.
"""

import os
import networkx as nx
import tempfile
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from episodic.db import get_connection, get_head

def custom_simple_layout(G):
    """
    A simple layout algorithm that doesn't require external dependencies.

    This function creates a basic tree layout for a directed graph.
    It's used as a fallback when both pygraphviz and numpy are not available.

    Args:
        G: A NetworkX DiGraph object

    Returns:
        A dictionary mapping node IDs to (x, y) positions
    """
    # Initialize positions dictionary
    pos = {}

    # Find root nodes (nodes with no incoming edges)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # If there are no root nodes, use the first node as root
    if not root_nodes and G.nodes():
        root_nodes = [list(G.nodes())[0]]

    # Initialize a queue for BFS traversal
    queue = [(node, 0, i) for i, node in enumerate(root_nodes)]
    visited = set(root_nodes)

    # Assign positions using BFS
    while queue:
        node, level, position = queue.pop(0)

        # Assign position: x is horizontal position, y is negative level (to go downward)
        pos[node] = (position, -level)

        # Get children (outgoing edges)
        children = list(G.successors(node))

        # Add children to queue
        for i, child in enumerate(children):
            if child not in visited:
                # Position children relative to parent
                queue.append((child, level + 1, position - (len(children) - 1) / 2 + i))
                visited.add(child)

    # If there are nodes not visited (disconnected components)
    for node in G.nodes():
        if node not in pos:
            # Assign a default position
            pos[node] = (0, 0)

    return pos

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

def visualize_dag(output_path=None, height="800px", width="100%", interactive=False, server_url=None, websocket=True):
    """
    Create an interactive visualization of the conversation DAG using Plotly.

    Args:
        output_path: Path to save the HTML file (default: temporary file)
        height: Height of the visualization (default: 800px)
        width: Width of the visualization (default: 100%)
        interactive: Whether to include interactive features like double-click to set current node
        server_url: URL of the server for interactive features (default: http://localhost:5000)
        websocket: Whether to enable WebSocket for real-time updates (default: True)

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
    node_labels = {}
    node_colors = []
    node_hover_texts = []
    node_ids = []
    is_current = []

    # Add nodes to the graph
    for node in nodes:
        # Ensure content is a string and truncate for display
        content_str = str(node["content"])
        display_content = content_str[:50] + "..." if len(content_str) > 50 else content_str
        # Include short ID in parentheses before the content
        display_content = f"({node['short_id']}) {display_content}"

        # Add node to the graph
        G.add_node(node["id"], 
                  title=content_str, 
                  label=display_content,
                  is_current=(node["id"] == current_node_id))

        # Store node information for Plotly
        node_ids.append(node["id"])
        node_labels[node["id"]] = display_content
        node_hover_texts.append(content_str)

        # Set color based on whether this is the current node
        if node["id"] == current_node_id:
            node_colors.append("#FFA500")  # Orange for current node
            is_current.append(True)
        else:
            node_colors.append("#97c2fc")  # Light blue for other nodes
            is_current.append(False)

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
        G.add_node(virtual_root_id, 
                  label="(root)", 
                  title="Virtual root node")

        # Add virtual root to our lists
        node_ids.append(virtual_root_id)
        node_labels[virtual_root_id] = "(root)"
        node_hover_texts.append("Virtual root node")
        node_colors.append("#CCCCCC")  # Light gray
        is_current.append(False)

        # Connect all root nodes to the virtual root
        for root_id in root_nodes:
            G.add_edge(virtual_root_id, root_id)

    # Use our custom layout algorithm that doesn't require external dependencies
    # This provides a consistent layout regardless of what packages are installed
    pos = custom_simple_layout(G)

    # Extract node positions
    x_nodes = [pos[node_id][0] for node_id in node_ids]
    y_nodes = [pos[node_id][1] for node_id in node_ids]

    # Extract edge positions
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Add None to create a break in the line
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create node trace
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        text=[node_labels[node_id] for node_id in node_ids],
        textposition="bottom center",
        hovertext=node_hover_texts,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=20,
            line=dict(width=2, color='#888')
        ),
        customdata=list(zip(node_ids, is_current)),
        showlegend=False
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        title='Episodic Conversation Visualization',
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=int(height.replace('px', '')),
        width=1000 if width == '100%' else int(width.replace('px', '')),
        plot_bgcolor='rgba(255,255,255,1)'
    )

    # Add interactive features if requested
    if interactive:
        # Add JavaScript for interactive features
        # This will handle double-click to set current node and right-click for context menu
        fig.update_layout(
            clickmode='event+select',
            # Add custom JavaScript for interactivity
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Instructions',
                            method='relayout',
                            args=['annotations[0].text', 
                                  'Double-click: Set as current node<br>Right-click: Delete node and descendants']
                        )
                    ],
                    x=0.5,
                    y=1.1,
                    xanchor='center',
                    yanchor='top'
                )
            ],
            annotations=[
                dict(
                    text='Double-click: Set as current node<br>Right-click: Delete node and descendants',
                    showarrow=False,
                    x=0.5,
                    y=1.05,
                    xanchor='center',
                    yanchor='bottom'
                )
            ]
        )

        # Add custom JavaScript for WebSocket, double-click, and right-click functionality
        custom_js = """
        <script>
        // Wait for the plot to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Connect to WebSocket server if enabled
            var socket = null;
            if (""" + str(websocket).lower() + """) {
                // Extract the server URL from the current page
                var serverUrl = '""" + server_url + """';

                // Connect to the Socket.IO server
                socket = io(serverUrl);

                // Handle connection event
                socket.on('connect', function() {
                    console.log('Connected to WebSocket server');
                });

                // Handle graph update event
                socket.on('graph_update', function(data) {
                    console.log('Received graph update:', data);
                    updateVisualization(data);
                });

                // Handle disconnection event
                socket.on('disconnect', function() {
                    console.log('Disconnected from WebSocket server');
                });
            }

            // Function to update the visualization with new graph data
            function updateVisualization(data) {
                // Get the plot div
                var plotDiv = document.querySelector('.plotly-graph-div');
                if (!plotDiv) {
                    console.error('Plot div not found');
                    return;
                }

                // Get the node trace (index 1 in the data array)
                var nodeTrace = plotDiv._fullData[1];
                if (!nodeTrace) {
                    console.error('Node trace not found');
                    return;
                }

                // Update node colors based on current node
                var colors = [];
                var isCurrentArray = [];

                for (var i = 0; i < data.nodes.length; i++) {
                    var node = data.nodes[i];
                    if (node.is_current) {
                        colors.push("#FFA500");  // Orange for current node
                        isCurrentArray.push(true);
                    } else {
                        colors.push("#97c2fc");  // Light blue for other nodes
                        isCurrentArray.push(false);
                    }
                }

                // Update the node trace
                Plotly.restyle(plotDiv, {
                    'marker.color': [colors],
                    'customdata': [data.nodes.map(function(node, i) {
                        return [node.id, isCurrentArray[i]];
                    })]
                }, [1]);

                console.log('Visualization updated');
            }
            // Get the plot div
            var plotDiv = document.querySelector('.plotly-graph-div');

            // Store the currently selected node
            var selectedNode = null;

            // Create a context menu for right-click
            var contextMenu = document.createElement("div");
            contextMenu.id = "context-menu";
            contextMenu.style.position = "absolute";
            contextMenu.style.display = "none";
            contextMenu.style.backgroundColor = "white";
            contextMenu.style.border = "1px solid #ccc";
            contextMenu.style.boxShadow = "2px 2px 5px rgba(0,0,0,0.2)";
            contextMenu.style.padding = "5px 0";
            contextMenu.style.zIndex = "1000";
            document.body.appendChild(contextMenu);

            // Add delete option to context menu
            var deleteOption = document.createElement("div");
            deleteOption.textContent = "Delete node and descendants";
            deleteOption.style.padding = "5px 10px";
            deleteOption.style.cursor = "pointer";
            deleteOption.addEventListener("mouseover", function() {
                this.style.backgroundColor = "#f0f0f0";
            });
            deleteOption.addEventListener("mouseout", function() {
                this.style.backgroundColor = "white";
            });
            contextMenu.appendChild(deleteOption);

            // Add click event listener to the plot
            plotDiv.addEventListener('click', function(evt) {
                var clickType = evt.detail === 2 ? 'double' : 'single';
                console.log("Click type: " + clickType);

                // Get the event data from Plotly
                var eventData = plotDiv._fullData[1];
                if (!eventData || !eventData.customdata) {
                    console.log("No event data or customdata available");
                    return;
                }

                // Find the closest point
                var pointIndex = getClosestPoint(evt, plotDiv);
                if (pointIndex === -1) {
                    console.log("No point found near click location");
                    return;
                }

                // Get the node ID from customdata
                var nodeId = eventData.customdata[pointIndex][0];
                console.log("Clicked on node: " + nodeId);

                // Skip the virtual root node
                if (nodeId === "virtual_root") {
                    console.log("Skipping virtual root node");
                    return;
                }

                // Handle double-click to set current node
                if (clickType === 'double') {
                    console.log("Double-click detected, setting current node: " + nodeId);

                    // Show loading indicator
                    document.body.style.cursor = 'wait';

                    // Make AJAX call to set current node
                    fetch('""" + server_url + """/set_current_node?id=' + nodeId, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => {
                        console.log("Response received");
                        return response.json();
                    })
                    .then(data => {
                        console.log("Success: ", data);
                        // Show success message
                        alert(data.message || 'Current node updated');

                        // If WebSocket is not enabled, reload the page to refresh the visualization
                        if (!""" + str(websocket).lower() + """) {
                            window.location.reload();
                        }
                        // Otherwise, the visualization will be updated via WebSocket
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

            // Add right-click event listener to the plot
            plotDiv.addEventListener('contextmenu', function(evt) {
                evt.preventDefault();
                evt.stopPropagation();  // Stop event propagation

                // Get the event data from Plotly
                var eventData = plotDiv._fullData[1];
                if (!eventData || !eventData.customdata) {
                    console.log("No event data or customdata available");
                    return;
                }

                // Find the closest point
                var pointIndex = getClosestPoint(evt, plotDiv);
                if (pointIndex === -1) {
                    console.log("No point found near click location");
                    return;
                }

                // Get the node ID from customdata
                var nodeId = eventData.customdata[pointIndex][0];
                console.log("Right-clicked on node: " + nodeId);

                // Skip the virtual root node
                if (nodeId === "virtual_root") {
                    console.log("Skipping virtual root node");
                    return;
                }

                // Store the selected node
                selectedNode = nodeId;

                // Position and show the context menu
                contextMenu.style.left = evt.clientX + "px";
                contextMenu.style.top = evt.clientY + "px";
                contextMenu.style.display = "block";
                console.log("Context menu displayed at: " + evt.clientX + ", " + evt.clientY);
            });

            // Add click handler for delete option
            deleteOption.addEventListener("click", function() {
                if (selectedNode) {
                    console.log("Delete option clicked for node: " + selectedNode);

                    if (confirm("Are you sure you want to delete this node and all its descendants?")) {
                        console.log("Deletion confirmed for node: " + selectedNode);

                        // Show loading indicator
                        document.body.style.cursor = 'wait';

                        // Make AJAX call to delete node
                        fetch('""" + server_url + """/delete_node?id=' + selectedNode, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        })
                        .then(response => {
                            console.log("Response received from delete_node endpoint");
                            return response.json();
                        })
                        .then(data => {
                            console.log("Success: ", data);
                            // Show success message
                            alert(data.message || 'Node deleted');

                            // If WebSocket is not enabled, reload the page to refresh the visualization
                            if (!""" + str(websocket).lower() + """) {
                                window.location.reload();
                            }
                            // Otherwise, the visualization will be updated via WebSocket
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('Error deleting node: ' + error);
                        })
                        .finally(() => {
                            // Reset cursor and hide context menu
                            document.body.style.cursor = 'default';
                            contextMenu.style.display = "none";
                        });
                    } else {
                        console.log("Deletion canceled for node: " + selectedNode);
                        // Hide context menu if delete is canceled
                        contextMenu.style.display = "none";
                    }
                } else {
                    console.log("No node selected for deletion");
                }
            });

            // Hide context menu when clicking elsewhere
            document.addEventListener("click", function(event) {
                // Only hide if the click is outside the context menu
                if (!contextMenu.contains(event.target)) {
                    if (contextMenu.style.display === "block") {
                        console.log("Hiding context menu due to click outside");
                        contextMenu.style.display = "none";
                    }
                } else {
                    console.log("Click inside context menu, keeping it open");
                }
            });

            // Helper function to find the closest point to a click event
            function getClosestPoint(evt, plotDiv) {
                try {
                    // Get the axis objects from the plot layout
                    var xaxis = plotDiv._fullLayout.xaxis;
                    var yaxis = plotDiv._fullLayout.yaxis;

                    if (!xaxis || !yaxis) {
                        console.error("Could not get axis objects from plot layout");
                        return -1;
                    }

                    // Get the plot's position on the page
                    var plotRect = plotDiv.getBoundingClientRect();

                    // Convert from screen coordinates to data coordinates
                    var x = xaxis.p2d(evt.clientX - plotRect.left);
                    var y = yaxis.p2d(evt.clientY - plotRect.top);

                    console.log("Click position in data coordinates: (" + x + ", " + y + ")");

                    // Get the node data
                    var eventData = plotDiv._fullData[1];
                    if (!eventData || !eventData.x || !eventData.y) {
                        console.error("Node data not available");
                        return -1;
                    }

                    // Find the closest node
                    var minDistance = Infinity;
                    var closestPoint = -1;

                    for (var i = 0; i < eventData.x.length; i++) {
                        var dx = eventData.x[i] - x;
                        var dy = eventData.y[i] - y;
                        var distance = Math.sqrt(dx*dx + dy*dy);

                        if (distance < minDistance) {
                            minDistance = distance;
                            closestPoint = i;
                        }
                    }

                    console.log("Closest point: " + closestPoint + " at distance: " + minDistance);

                    // Only return a point if it's close enough (within a threshold)
                    var threshold = 50;  // Adjust this value as needed
                    if (minDistance < threshold) {
                        return closestPoint;
                    } else {
                        console.log("No point within threshold distance");
                        return -1;
                    }
                } catch (error) {
                    console.error("Error in getClosestPoint: " + error);
                    return -1;
                }
            }
        });
        </script>
        """

    # Save to file
    if output_path is None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            output_path = tmp.name

    # Write the figure to HTML with the custom JavaScript
    with open(output_path, 'w') as f:
        html_content = pio.to_html(
            fig, 
            full_html=True,
            include_plotlyjs=True,
            config={'responsive': True}
        )

        # Add Socket.IO client library and custom JavaScript if interactive
        if interactive:
            # Add Socket.IO client library if WebSocket is enabled
            socketio_script = ""
            if websocket:
                socketio_script = """
                <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
                """

            # Insert Socket.IO client library and custom JavaScript before the closing body tag
            html_content = html_content.replace('</body>', f'{socketio_script}{custom_js}</body>')

        f.write(html_content)

    return output_path
