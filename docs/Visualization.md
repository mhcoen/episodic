# Visualization Guide

Episodic includes powerful visualization tools to explore the conversation DAG (Directed Acyclic Graph).

## Browser-Based Visualization

The default visualization opens in your web browser:

```bash
# Generate and open an interactive visualization in your browser
episodic visualize

# Save the visualization to a specific file
episodic visualize --output conversation.html

# Generate the visualization without opening it in a browser
episodic visualize --no-browser
```

## Native Window Visualization

Episodic also supports displaying the visualization in a native window instead of a web browser:

```bash
# Open the visualization in a native window
episodic visualize --native

# Customize the window size
episodic visualize --native --width 1200 --height 900
```

The native window visualization provides the same features as the browser-based visualization but in a standalone application window, which can be more convenient for some users.

## Visualization Features

Both visualization methods allow you to:
- See the entire conversation structure as a directed graph
- Hover over nodes to see the full content
- Zoom in/out and pan around the graph
- Move nodes to explore different layouts
- See the hierarchical structure of conversations
- Double-click on a node to make it the current node
- Right-click on a node to delete it and all its descendants

## Using CLI and Visualization Together

To use the CLI while having the visualization open, you need to run two separate processes:

### Step 1: Start the Visualization in One Terminal

```bash
# Open the browser-based visualization
episodic visualize

# Or open the native visualization window
episodic visualize --native
```

This will open the visualization and block that terminal until the visualization is closed or you press Ctrl+C.

### Step 2: Run the Interactive Shell in Another Terminal

Open a new terminal window and run:

```bash
# If installed with pip
episodic-shell

# Or using the Python module syntax
python -m episodic.cli
```

### How They Work Together

Both the visualization and the CLI access the same database, so:

1. Changes made in the CLI (adding nodes, changing the current node) will be reflected in the visualization
2. Changes made in the visualization (setting current node, deleting nodes) will be reflected in the CLI
3. The visualization automatically polls for updates, so you'll see your changes appear in real-time
4. You can keep both open simultaneously for a more interactive workflow

This approach gives you the best of both worlds: visual exploration of your conversation graph and powerful command-line control.

## Technical Notes

- **Visualization Engine**: The visualization uses Plotly, a powerful and interactive data visualization library, providing a robust and feature-rich experience.
- **Native Window Support**: The native window visualization uses PyWebView to embed the web-based visualization in a standalone application window.
- **Layout Algorithm**: The visualization uses a custom hierarchical layout algorithm that doesn't require external dependencies, ensuring consistent visualization across different environments.
- **Real-time Updates**: The visualization supports real-time updates:
  - Changes to the graph (setting current node, deleting nodes) are pushed to all connected clients
  - The visualization updates automatically without page reloads
  - Multiple users can view the same visualization and see changes in real-time
  - Uses HTTP polling for reliable updates
- **Interactive Features**: The visualization includes enhanced interactive features:
  - Double-click on a node to make it the current node
  - Right-click on a node to access a context menu for node deletion
  - Hover over nodes to see the full content
  - Pan and zoom to explore large conversation graphs
- **Robust Connectivity**: The visualization includes several features to ensure reliable operation:
  - Automatic reconnection if the connection is lost
  - HTTP polling for reliable real-time updates
  - Visual notifications of connection status and updates
  - Detailed error handling and logging for troubleshooting
- **Server Port**: If you encounter an "Address already in use" error (common on macOS where AirPlay uses port 5000), you can specify a different port:
  ```bash
  episodic visualize --port 5001
  ```