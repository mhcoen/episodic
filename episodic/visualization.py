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
from episodic.configuration import (
    VISUALIZATION_VERTICAL_SPACING, VISUALIZATION_DEFAULT_WIDTH, VISUALIZATION_NODE_SIZE,
    VISUALIZATION_CONTENT_TRUNCATE_LENGTH, CURRENT_NODE_COLOR, DEFAULT_NODE_COLOR,
    VIRTUAL_ROOT_COLOR, HTTP_POLLING_INTERVAL,
    CLICK_THRESHOLD_TOUCH, CLICK_THRESHOLD_MOUSE
)

def custom_simple_layout(G):
    """
    A vertical tree layout algorithm that places the root node at the top center.

    This function creates a balanced tree layout for a directed graph by:
    1. Placing the root node at the top center
    2. Positioning children vertically below their parents
    3. Distributing children horizontally centered under their parent

    Args:
        G: A NetworkX DiGraph object

    Returns:
        A dictionary mapping node IDs to (x, y) positions
    """
    pos = {}

    # Find root nodes (nodes with no incoming edges)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # If there are no root nodes, use the first node as root
    if not root_nodes and G.nodes():
        root_nodes = [list(G.nodes())[0]]

    # Sort root nodes to ensure consistent ordering
    root_nodes = sorted(root_nodes)

    # Calculate the maximum width needed at each level
    level_widths = {}
    nodes_at_level = {}

    def calculate_level_info(node, level=0):
        # Initialize this level if not seen before
        if level not in nodes_at_level:
            nodes_at_level[level] = []
            level_widths[level] = 0

        # Add this node to its level
        nodes_at_level[level].append(node)

        # Update the width needed for this level
        level_widths[level] += 1

        # Process children recursively
        # Sort children to ensure consistent ordering
        children = sorted(list(G.successors(node)))
        for child in children:
            calculate_level_info(child, level+1)

    # Calculate level information for each root node
    for root in root_nodes:
        calculate_level_info(root)

    # Calculate the maximum width across all levels
    max_width = max(level_widths.values()) if level_widths else 1

    # Position nodes level by level
    for level in sorted(nodes_at_level.keys()):
        # Sort nodes at this level to ensure consistent ordering
        nodes = sorted(nodes_at_level[level])
        # Calculate horizontal spacing
        spacing = max(1.0, max_width / (len(nodes) + 1))

        # Position nodes at this level
        for i, node in enumerate(nodes):
            # Center the nodes horizontally
            x_pos = (i + 1) * spacing - max_width / 2
            # Position vertically based on level (negative to go downward)
            y_pos = -level * VISUALIZATION_VERTICAL_SPACING
            pos[node] = (x_pos, y_pos)

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
            c.execute("SELECT id, short_id, content, parent_id FROM nodes ORDER BY ROWID")

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
    Create an interactive visualization of the conversation DAG using Plotly.

    Args:
        output_path: Path to save the HTML file (default: temporary file)
        height: Height of the visualization (default: 800px)
        width: Width of the visualization (default: 100%)
        interactive: Whether to include interactive features like double-click to set current node
        server_url: URL of the server for interactive features (default: http://127.0.0.1:5000)

    Returns:
        Path to the generated HTML file
    """
    # Initialize custom_js as an empty string
    custom_js = ""

    # Set default server URL if not provided
    if interactive and server_url is None:
        server_url = "http://127.0.0.1:5000"

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
        display_content = content_str[:VISUALIZATION_CONTENT_TRUNCATE_LENGTH] + "..." if len(content_str) > VISUALIZATION_CONTENT_TRUNCATE_LENGTH else content_str
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
            node_colors.append(CURRENT_NODE_COLOR)  # Orange for current node
            is_current.append(True)
        else:
            node_colors.append(DEFAULT_NODE_COLOR)  # Light blue for other nodes
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
        node_colors.append(VIRTUAL_ROOT_COLOR)  # Light gray
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
            size=VISUALIZATION_NODE_SIZE,
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
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            automargin=True,  # Auto-adjust margins
            autorange=True    # Auto-range to fit all points
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            automargin=True,  # Auto-adjust margins
            autorange=True    # Auto-range to fit all points
        ),
        height=int(height.replace('px', '')),
        width=VISUALIZATION_DEFAULT_WIDTH if width == '100%' else int(width.replace('px', '')),
        plot_bgcolor='rgba(255,255,255,1)',
        autosize=True        # Enable auto-sizing
    )

    # Add interactive features if requested
    if interactive:
        # Add JavaScript for interactive features
        # This will handle double-click to set current node and right-click for context menu
        fig.update_layout(
#            clickmode='event+select',
            clickmode='event',
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
        <script>\n        // Color constants from configuration\n        var currentNodeColor = '""" + CURRENT_NODE_COLOR + """';\n        var defaultNodeColor = '""" + DEFAULT_NODE_COLOR + """';\n        var virtualRootColor = '""" + VIRTUAL_ROOT_COLOR + """';
        // Wait for the plot to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Auto-zoom to fit all nodes on initial load
            setTimeout(function() {
                fitGraphToViewport();
            }, 500);  // Small delay to ensure the plot is fully rendered

            // Connect to WebSocket server if enabled, otherwise use polling
            var socket = null;
            var pollingInterval = null;
            var serverUrl = '""" + server_url + """';
            var cleanUrl = serverUrl.endsWith('/') ? serverUrl.slice(0, -1) : serverUrl;

            console.log('Server URL:', serverUrl);
            console.log('Cleaned server URL:', cleanUrl);

            // Primary polling mechanism for when WebSockets are disabled
            function startPolling() {
                console.log('Initializing HTTP polling mechanism');

                // Show a notification to the user
                var notification = document.createElement('div');
                notification.style.position = 'fixed';
                notification.style.bottom = '10px';
                notification.style.right = '10px';
                notification.style.backgroundColor = '#2196F3';
                notification.style.color = 'white';
                notification.style.padding = '10px';
                notification.style.borderRadius = '5px';
                notification.style.zIndex = '1000';
                notification.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                notification.textContent = 'Using HTTP polling for live updates';
                document.body.appendChild(notification);

                // Remove the notification after 3 seconds
                setTimeout(function() {
                    notification.style.opacity = '0';
                    notification.style.transition = 'opacity 0.5s';

                    setTimeout(function() {
                        notification.remove();
                    }, 500);
                }, 3000);

                // Set up polling interval (every 5 seconds)
                pollingInterval = setInterval(function() {
                    console.log('Polling for updates...');

                    // Blink the indicator when polling starts
                    blinkUpdateIndicator();

                    // Make a request to get the current graph data
                    fetch(cleanUrl + '/get_graph_data')
                        .then(response => response.json())
                        .then(data => {
                            console.log('Received update from polling:', data);

                            // Update the visualization with the new data
                            try {
                                updateVisualization(data);
                            } catch (error) {
                                console.error('Error updating visualization from polling:', error);
                            }
                        })
                        .catch(error => {
                            console.error('Error polling for updates:', error);
                        });
                }, """ + str(HTTP_POLLING_INTERVAL) + """);
            }

            // Start HTTP polling for updates
            console.log('Using HTTP polling for updates');
            startPolling();

            // Function to create and blink a small green circle in the upper left corner
            function blinkUpdateIndicator() {
                // Check if indicator already exists
                var indicator = document.getElementById('update-indicator');

                // Create indicator if it doesn't exist
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.id = 'update-indicator';
                    indicator.style.position = 'fixed';
                    indicator.style.top = '10px';
                    indicator.style.left = '10px';
                    indicator.style.width = '12px';
                    indicator.style.height = '12px';
                    indicator.style.backgroundColor = '#4CAF50';
                    indicator.style.borderRadius = '50%';
                    indicator.style.zIndex = '1000';
                    indicator.style.opacity = '0';
                    indicator.style.transition = 'opacity 0.3s';
                    document.body.appendChild(indicator);
                }

                // Blink the indicator
                indicator.style.opacity = '1';
                setTimeout(function() {
                    indicator.style.opacity = '0';
                }, 500);
            }

            // Function to automatically fit the graph to the viewport
            function fitGraphToViewport() {
                var plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv && plotDiv.layout) {
                    Plotly.relayout(plotDiv, {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                }
            }

            // Function to update the visualization with new graph data
            function updateVisualization(data) {
                console.log('%c Updating visualization with data', 'background: #4CAF50; color: white; padding: 2px 5px; border-radius: 2px;');
                console.log('Data contains', data.nodes.length, 'nodes and', data.edges.length, 'edges');

                // Get the plot div
                var plotDiv = document.querySelector('.plotly-graph-div');
                if (!plotDiv) {
                    console.error('Plot div not found');
                    return;
                }

                try {
                    // Get the node trace (index 1 in the data array)
                    var nodeTrace = plotDiv._fullData[1];
                    if (!nodeTrace) {
                        console.error('Node trace not found in plot data');
                        return;
                    }

                    console.log('Current node trace has', nodeTrace.x.length, 'nodes');
                    console.log('New data has', data.nodes.length, 'nodes');

                    // If the node count is different, completely redraw the visualization
                    if (nodeTrace.x.length !== data.nodes.length) {
                        console.warn('Node count mismatch. Completely redrawing the visualization...');
                        console.log('Current node trace has', nodeTrace.x.length, 'nodes, but data has', data.nodes.length, 'nodes');

                        // Completely redraw the visualization with the new data
                        redrawVisualization(data, plotDiv);
                        return;
                    }

                    // If node count is the same, just update properties
                    // Create a map of node IDs to their indices in the data
                    var nodeIdToIndex = {};
                    for (var i = 0; i < data.nodes.length; i++) {
                        nodeIdToIndex[data.nodes[i].id] = i;
                    }

                    // Create arrays for the new colors and customdata
                    var newColors = [];
                    var newCustomdata = [];
                    var updatedNodes = 0;

                    // Update the colors based on whether each node is current
                    for (var i = 0; i < nodeTrace.customdata.length; i++) {
                        var nodeId = nodeTrace.customdata[i][0];
                        var dataIndex = nodeIdToIndex[nodeId];

                        if (dataIndex !== undefined) {
                            var node = data.nodes[dataIndex];
                            var isCurrentNode = node.is_current;
                            var wasCurrentNode = nodeTrace.customdata[i][1];

                            // Set color based on current status
                            var newColor = isCurrentNode ? "" + currentNodeColor + "" : "" + defaultNodeColor + "";
                            newColors.push(newColor);

                            // Update customdata with current status
                            newCustomdata.push([nodeId, isCurrentNode]);

                            // Log if the node's current status changed
                            if (isCurrentNode !== wasCurrentNode) {
                                console.log('Node', nodeId, 'current status changed from', wasCurrentNode, 'to', isCurrentNode);
                                updatedNodes++;
                            }
                        } else {
                            console.warn('Node', nodeId, 'not found in new data');
                            // This shouldn't happen if node counts match, but handle it just in case
                            newColors.push(nodeTrace.marker.color[i]);
                            newCustomdata.push(nodeTrace.customdata[i]);
                        }
                    }

                    console.log('Updating', updatedNodes, 'nodes with new properties');

                    // Update the node trace with new colors and customdata
                    Plotly.restyle(plotDiv, {
                        'marker.color': [newColors],
                        'customdata': [newCustomdata]
                    }, [1]);

                    console.log('%c Visualization updated successfully', 'background: #2196F3; color: white; padding: 2px 5px; border-radius: 2px;');

                    // Blink the update indicator
                    blinkUpdateIndicator();

                    // Auto-zoom to fit all nodes
                    fitGraphToViewport();
                } catch (error) {
                    console.error('Error updating visualization:', error);
                    console.error('Error stack:', error.stack);

                    // Show error notification
                    showErrorNotification('Error updating visualization: ' + error.message);
                }

                // Function to completely redraw the visualization with new data
                function redrawVisualization(data, plotDiv) {
                    console.log('Completely redrawing visualization with new data');

                    try {
                        // Extract node and edge data from the provided data
                        var nodes = data.nodes;
                        var edges = data.edges;
                        var currentNodeId = data.current_node_id;

                        // Prepare node data for Plotly
                        var nodeIds = [];
                        var nodeLabels = {};
                        var nodeHoverTexts = [];
                        var nodeColors = [];
                        var isCurrentNode = [];
                        var nodeX = [];
                        var nodeY = [];

                        // Find root nodes (nodes without parents)
                        var root_nodes = [];
                        var nodeIdToNode = {};

                        // Create a map of node IDs to nodes for quick lookup
                        for (var i = 0; i < nodes.length; i++) {
                            nodeIdToNode[nodes[i].id] = nodes[i];
                        }

                        // Find nodes that have no parents (root nodes)
                        for (var i = 0; i < nodes.length; i++) {
                            var isRoot = true;
                            for (var j = 0; j < edges.length; j++) {
                                var targetId = edges[j].target || edges[j].to;
                                if (targetId === nodes[i].id) {
                                    isRoot = false;
                                    break;
                                }
                            }
                            if (isRoot) {
                                root_nodes.push(nodes[i].id);
                            }
                        }

                        // If there are multiple root nodes, add a virtual root node
                        if (root_nodes.length > 1) {
                            // Create a virtual root node
                            var virtual_root_id = "virtual_root";

                            // Add virtual root to our lists
                            nodeIds.push(virtual_root_id);
                            nodeLabels[virtual_root_id] = "(root)";
                            nodeHoverTexts.push("Virtual root node");
                            nodeColors.push(virtualRootColor);  // Light gray
                            isCurrentNode.push(false);

                            // Position the virtual root node
                            nodeX.push(0);  // Center position
                            nodeY.push(0);  // Top position

                            // Add edges from virtual root to all root nodes
                            for (var i = 0; i < root_nodes.length; i++) {
                                var root_id = root_nodes[i];
                                // We'll add the edges later when processing all edges
                                edges.push({
                                    source: virtual_root_id,
                                    target: root_id
                                });
                            }
                        }

                        // Process nodes
                        for (var i = 0; i < nodes.length; i++) {
                            var node = nodes[i];

                            // Skip if this node ID is already in the nodeIds array (avoid duplicates)
                            if (nodeIds.indexOf(node.id) !== -1) {
                                continue;
                            }

                            // Add node ID to the array
                            nodeIds.push(node.id);

                            // Format the display content like in the original visualization
                            // Ensure we use node.content even if it's empty, only fall back to node.id if content is undefined
                            var content_str = node.content !== undefined ? node.content : (node.title || node.id);
                            // Convert content to string to handle non-string content
                            content_str = String(content_str);
                            var display_content = content_str.length > """ + str(VISUALIZATION_CONTENT_TRUNCATE_LENGTH) + """ ? content_str.substring(0, """ + str(VISUALIZATION_CONTENT_TRUNCATE_LENGTH) + """) + "..." : content_str;

                            // Use short_id if available, otherwise use a shortened version of the full ID
                            var short_id = node.short_id;
                            // First convert to string to handle numeric short_ids like "01"
                            if (short_id !== undefined && short_id !== null && short_id !== "") {
                                short_id = String(short_id);
                            } else if (node.id) {
                                // Create a shortened version of the ID if short_id is not available
                                short_id = node.id.substring(0, 8);
                            } else {
                                short_id = "unknown";
                            }

                            display_content = "(" + short_id + ") " + display_content;

                            nodeLabels[node.id] = display_content;
                            // Ensure we use node.content even if it's empty, only fall back to node.id if content is undefined
                            var hoverText = node.content !== undefined ? node.content : (node.title || node.id);
                            // Convert to string to handle non-string content
                            nodeHoverTexts.push(String(hoverText));

                            // Set color based on whether this is the current node
                            if (node.is_current || node.id === currentNodeId) {
                                nodeColors.push(currentNodeColor);  // Orange for current node
                                isCurrentNode.push(true);
                            } else {
                                nodeColors.push(defaultNodeColor);  // Light blue for other nodes
                                isCurrentNode.push(false);
                            }

                            // Use the node's position if available, otherwise use a default
                            if (node.x !== undefined && node.y !== undefined) {
                                nodeX.push(node.x);
                                nodeY.push(node.y);
                            } else {
                                // Get position from existing visualization if possible
                                var found = false;
                                if (plotDiv._fullData && plotDiv._fullData[1]) {
                                    var existingNodeTrace = plotDiv._fullData[1];
                                    for (var j = 0; j < existingNodeTrace.customdata.length; j++) {
                                        if (existingNodeTrace.customdata[j][0] === node.id) {
                                            nodeX.push(existingNodeTrace.x[j]);
                                            nodeY.push(existingNodeTrace.y[j]);
                                            found = true;
                                            break;
                                        }
                                    }
                                }

                                if (!found) {
                                    // Use a vertical layout algorithm
                                    // Position nodes in a vertical column, with the first node at the top
                                    nodeX.push(0);  // All nodes centered horizontally
                                    nodeY.push(-i); // Nodes stacked vertically, with increasing depth
                                }
                            }
                        }

                        // Prepare edge data for Plotly
                        var edgeX = [];
                        var edgeY = [];

                        // Process edges
                        for (var i = 0; i < edges.length; i++) {
                            var edge = edges[i];
                            // Handle both source/target and from/to edge formats
                            var sourceId = edge.source || edge.from;
                            var targetId = edge.target || edge.to;
                            var sourceIndex = nodeIds.indexOf(sourceId);
                            var targetIndex = nodeIds.indexOf(targetId);

                            if (sourceIndex !== -1 && targetIndex !== -1) {
                                var x0 = nodeX[sourceIndex];
                                var y0 = nodeY[sourceIndex];
                                var x1 = nodeX[targetIndex];
                                var y1 = nodeY[targetIndex];

                                // Add None to create a break in the line
                                edgeX.push(x0, x1, null);
                                edgeY.push(y0, y1, null);
                            }
                        }

                        // Create edge trace
                        var edgeTrace = {
                            x: edgeX,
                            y: edgeY,
                            mode: 'lines',
                            line: {
                                width: 1,
                                color: '#888'
                            },
                            hoverinfo: 'none',
                            showlegend: false
                        };

                        // Create node trace
                        var nodeTrace = {
                            x: nodeX,
                            y: nodeY,
                            mode: 'markers+text',
                            text: nodeIds.map(id => nodeLabels[id]),
                            textposition: "bottom center",
                            hovertext: nodeHoverTexts,
                            hoverinfo: 'text',
                            marker: {
                                showscale: false,
                                color: nodeColors,
                                size: 20,
                                line: {
                                    width: 2,
                                    color: '#888'
                                }
                            },
                            customdata: nodeIds.map((id, index) => [id, isCurrentNode[index]]),
                            showlegend: false
                        };

                        // Create the new data array
                        var newData = [edgeTrace, nodeTrace];

                        // Update the plot with the new data
                        Plotly.react(plotDiv, newData, plotDiv.layout);

                        console.log('%c Visualization completely redrawn successfully', 'background: #2196F3; color: white; padding: 2px 5px; border-radius: 2px;');

                        // Blink the update indicator
                        blinkUpdateIndicator();

                        // Auto-zoom to fit all nodes
                        fitGraphToViewport();
                    } catch (error) {
                        console.error('Error redrawing visualization:', error);
                        console.error('Error stack:', error.stack);

                        // Show error notification
                        showErrorNotification('Error redrawing visualization: ' + error.message);
                    }
                }


                // Helper function to show error notification
                function showErrorNotification(message) {
                    var errorNotification = document.createElement('div');
                    errorNotification.style.position = 'fixed';
                    errorNotification.style.top = '10px';
                    errorNotification.style.right = '10px';
                    errorNotification.style.backgroundColor = '#F44336';
                    errorNotification.style.color = 'white';
                    errorNotification.style.padding = '10px';
                    errorNotification.style.borderRadius = '5px';
                    errorNotification.style.zIndex = '1000';
                    errorNotification.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                    errorNotification.textContent = message;
                    document.body.appendChild(errorNotification);

                    // Remove the notification after a delay
                    setTimeout(function() {
                        errorNotification.style.opacity = '0';
                        errorNotification.style.transition = 'opacity 0.5s';

                        setTimeout(function() {
                            errorNotification.remove();
                        }, 500);
                    }, """ + str(HTTP_POLLING_INTERVAL) + """);
                }
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
                        method: 'POST'
                        // No Content-Type header since we're not sending a request body
                    })
                    .then(response => {
                        console.log("Response received");
                        // Check if response is OK before trying to parse JSON
                        if (!response.ok) {
                            throw new Error('Server returned ' + response.status + ': ' + response.statusText);
                        }
                        // Check if response has content before trying to parse JSON
                        if (response.headers.get('content-length') === '0') {
                            throw new Error('Empty response from server');
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log("Success: ", data);
                        // Show success message

                        // Poll for updates immediately
                        console.log('Polling for updates immediately');

                            // Blink the indicator when polling starts
                            blinkUpdateIndicator();

                            // Make a request to get the current graph data
                            fetch(cleanUrl + '/get_graph_data')
                                .then(response => response.json())
                                .then(data => {
                                    console.log('Received update from immediate polling:', data);

                                    // Update the visualization with the new data
                                    try {
                                        updateVisualization(data);
                                    } catch (error) {
                                        console.error('Error updating visualization from immediate polling:', error);
                                        // If update fails, fall back to page reload
                                        window.location.reload();
                                    }
                                })
                                .catch(error => {
                                    console.error('Error polling for updates:', error);
                                    // If polling fails, fall back to page reload
                                    window.location.reload();
                                });
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

                console.log("Right-click event detected");

                // Get the event data from Plotly
                var eventData = plotDiv._fullData[1];
                if (!eventData || !eventData.customdata) {
                    console.error("No event data or customdata available");
                    return;
                }

                // Find the closest point
                var pointIndex = getClosestPoint(evt, plotDiv);
                if (pointIndex === -1) {
                    console.log("No point found near right-click location");
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
                // Ensure the menu is within the viewport
                var menuX = evt.clientX;
                var menuY = evt.clientY;

                // Calculate menu dimensions
                var menuWidth = 200; // Approximate width
                var menuHeight = 30; // Approximate height

                // Adjust position if menu would go off-screen
                if (menuX + menuWidth > window.innerWidth) {
                    menuX = window.innerWidth - menuWidth;
                }
                if (menuY + menuHeight > window.innerHeight) {
                    menuY = window.innerHeight - menuHeight;
                }

                contextMenu.style.left = menuX + "px";
                contextMenu.style.top = menuY + "px";
                contextMenu.style.display = "block";
                console.log("Context menu displayed at: " + menuX + ", " + menuY);
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
                            method: 'POST'
                            // No Content-Type header since we're not sending a request body
                        })
                        .then(response => {
                            console.log("Response received from delete_node endpoint");
                            // Check if response is OK before trying to parse JSON
                            if (!response.ok) {
                                throw new Error('Server returned ' + response.status + ': ' + response.statusText);
                            }
                            // Check if response has content before trying to parse JSON
                            if (response.headers.get('content-length') === '0') {
                                throw new Error('Empty response from server');
                            }
                            return response.json();
                        })
                        .then(data => {
                            console.log("Success: ", data);
                            // Show success message

                            // Poll for updates immediately
                            console.log('Polling for updates immediately');

                                // Blink the indicator when polling starts
                                blinkUpdateIndicator();

                                // Make a request to get the current graph data
                                fetch(cleanUrl + '/get_graph_data')
                                    .then(response => response.json())
                                    .then(data => {
                                        console.log('Received update from immediate polling:', data);

                                        // Update the visualization with the new data
                                        try {
                                            updateVisualization(data);
                                        } catch (error) {
                                            console.error('Error updating visualization from immediate polling:', error);
                                            // If update fails, fall back to page reload
                                            window.location.reload();
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error polling for updates:', error);
                                        // If polling fails, fall back to page reload
                                        window.location.reload();
                                    });
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
                // Skip the data coordinate conversion and use screen coordinates directly
                       return findClosestPointByScreenCoordinates(evt, plotDiv, plotDiv._fullData[1]);
                } catch (error) {
                console.error("Error in getClosestPoint: " + error);
                return -1;
                }

                try {
                    console.log("Finding closest point to click at: " + evt.clientX + ", " + evt.clientY);

                    // First, check if we can use the Plotly event data directly
                    var eventData = plotDiv._fullData[1];
                    if (!eventData) {
                        console.error("No event data available");
                        return -1;
                    }

                    // Get the plot's position on the page
                    var plotRect = plotDiv.getBoundingClientRect();
                    console.log("Plot rect: " + JSON.stringify({
                        left: plotRect.left,
                        top: plotRect.top,
                        width: plotRect.width,
                        height: plotRect.height
                    }));

                    // Check if click is within the plot area
                    if (evt.clientX < plotRect.left || evt.clientX > plotRect.right ||
                        evt.clientY < plotRect.top || evt.clientY > plotRect.bottom) {
                        console.log("Click outside plot area");
                        return -1;
                    }

                    // Get the axis objects from the plot layout
                    var xaxis = plotDiv._fullLayout.xaxis;
                    var yaxis = plotDiv._fullLayout.yaxis;

                    if (!xaxis || !yaxis) {
                        console.error("Could not get axis objects from plot layout");
                        // Fall back to screen coordinates if we can't get data coordinates
                        return findClosestPointByScreenCoordinates(evt, plotDiv, eventData);
                    }

                    // Convert from screen coordinates to data coordinates
                    var x = xaxis.p2d(evt.clientX - plotRect.left);
                    var verticalOffset = 20; // This value may need adjustment based on testing
                    var y = yaxis.p2d(evt.clientY - plotRect.top - verticalOffset);                            

                    console.log("Click position in data coordinates: (" + x + ", " + y + ")");

                    // Check if we have node positions
                    if (!eventData.x || !eventData.y) {
                        console.error("Node position data not available");
                        return -1;
                    }

                    // Find the closest node
                    var minDistance = Infinity;
                    var closestPoint = -1;

                    for (var i = 0; i < eventData.x.length; i++) {
                        var dx = eventData.x[i] - x;
                        var dy = eventData.y[i] - y;
                        var distance = Math.sqrt(dx*dx + dy*dy);

                        console.log("Node " + i + " distance: " + distance);

                        if (distance < minDistance) {
                            minDistance = distance;
                            closestPoint = i;
                        }
                    }

                    console.log("Closest point: " + closestPoint + " at distance: " + minDistance);

                    // Only return a point if it's close enough (within a threshold)
                    // Use a larger threshold for touch devices
                    var threshold = ('ontouchstart' in window) ? """ + str(CLICK_THRESHOLD_TOUCH) + """ : """ + str(CLICK_THRESHOLD_MOUSE) + """;
                    if (minDistance < threshold) {
                        return closestPoint;
                    } else {
                        console.log("No point within threshold distance");
                        return -1;
                    }
                } catch (error) {
                    console.error("Error in getClosestPoint: " + error);
                    console.error("Stack trace: " + error.stack);
                    // Fall back to screen coordinates method
                    return findClosestPointByScreenCoordinates(evt, plotDiv, eventData);
                }
            }

            // Fallback function to find closest point using screen coordinates
            function findClosestPointByScreenCoordinates(evt, plotDiv, eventData) {
                try {
                    console.log("Using screen coordinates fallback method");

                    if (!eventData || !eventData.customdata) {
                        console.error("No event data or customdata available for fallback method");
                        return -1;
                    }

                    // Get node positions in screen coordinates
                    var nodePositions = [];
                    var plotRect = plotDiv.getBoundingClientRect();

                    // Find all node markers in the DOM
                    var nodeElements = plotDiv.querySelectorAll('.point');

                    if (nodeElements.length === 0) {
                        console.error("No node elements found in DOM");
                        return -1;
                    }

                    console.log("Found " + nodeElements.length + " node elements");

                    // Calculate distances to each node
                    var minDistance = Infinity;
                    var closestPoint = -1;

                    for (var i = 0; i < nodeElements.length; i++) {
                        var rect = nodeElements[i].getBoundingClientRect();
                        var centerX = rect.left + rect.width / 2;
                        var centerY = rect.top + rect.height / 2;

                        var dx = centerX - evt.clientX;
                        var dy = centerY - evt.clientY;
                        var distance = Math.sqrt(dx*dx + dy*dy);

                        if (distance < minDistance) {
                            minDistance = distance;
                            closestPoint = i;
                        }
                    }

                    console.log("Closest point by screen coordinates: " + closestPoint + " at distance: " + minDistance);

                    // Only return a point if it's close enough
                    var threshold = ('ontouchstart' in window) ? """ + str(CLICK_THRESHOLD_TOUCH) + """ : """ + str(CLICK_THRESHOLD_MOUSE) + """;
                    if (minDistance < threshold) {
                        return closestPoint;
                    } else {
                        console.log("No point within threshold distance (screen coordinates)");
                        return -1;
                    }
                } catch (error) {
                    console.error("Error in findClosestPointByScreenCoordinates: " + error);
                    console.error("Stack trace: " + error.stack);
                    return -1;
                }
            }
        });
        </script>
        """

    # Save to file
    try:
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
                config={
                    'responsive': True,
                    'scrollZoom': True,     # Enable scroll to zoom
                    'displayModeBar': True, # Show the mode bar with zoom controls
                    'modeBarButtonsToAdd': ['resetScale']  # Add reset scale button
                }
            )

            # Add DOCTYPE declaration if it's not already present
            if not html_content.startswith('<!DOCTYPE html>'):
                html_content = '<!DOCTYPE html>\n' + html_content

            # Add custom JavaScript if interactive
            if interactive:

                # Insert custom JavaScript before the closing body tag
                html_content = html_content.replace('</body>', f'{custom_js}</body>')

            f.write(html_content)
    except Exception as e:
        # If an exception occurs, ensure the temporary file is deleted if we created one
        if output_path is None and 'tmp' in locals() and os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise

    return output_path
