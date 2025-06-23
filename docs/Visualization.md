# Visualization Guide

Episodic includes powerful visualization tools to explore the conversation DAG (Directed Acyclic Graph).

## Browser-Based Visualization

The default visualization opens in your web browser:

```bash
# Generate and open an interactive visualization in your browser
> /visualize

# Save the visualization to a specific file
> /visualize --output conversation.html

# Generate the visualization without opening it in a browser
> /visualize --no-browser
```

## Native Window Visualization

Episodic also supports displaying the visualization in a native window instead of a web browser:

```bash
# Open the visualization in a native window
> /visualize --native

# Customize the window size
> /visualize --native --width 1200 --height 900
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

## Using Visualization with Talk Mode

The visualization can be used directly from the talk mode interface:

```bash
# Start the application
python -m episodic

# In the talk mode, start the visualization
> /visualize
```

The visualization will open in your browser, and you can continue using the talk mode interface in the same terminal. This allows you to:

1. Chat with the LLM and see the conversation graph update in real-time
2. Use commands like `/head` to navigate to different nodes
3. See the effects of your actions immediately in the visualization
4. Interact with the visualization (e.g., double-click on nodes) and see the changes reflected in the talk mode

This integrated approach provides a seamless experience where you can both converse with the LLM and visually explore the conversation structure at the same time.

### Multiple Visualization Windows

You can also open the visualization in a separate terminal if you prefer:

```bash
# In a separate terminal
python -m episodic

# Then immediately start the visualization
> /visualize
```

This allows you to have multiple visualization windows open at the same time, each showing the same conversation graph but potentially focused on different parts of it.

## Technical Notes

- **Visualization Engine**: The visualization uses Plotly, a powerful and interactive data visualization library, providing a robust and feature-rich experience.
- **Native Window Support**: The native window visualization uses PyWebView to embed the web-based visualization in a standalone application window.
- **Layout Algorithm**: The visualization uses a custom hierarchical layout algorithm that doesn't require external dependencies, ensuring consistent visualization across different environments.
- **Real-time Updates**: The visualization supports real-time updates via HTTP polling:
  - Changes to the graph (setting current node, deleting nodes) are reflected across all browser windows
  - The visualization updates automatically without page reloads through periodic HTTP requests
  - Multiple users can view the same visualization and see changes updated periodically
  - Uses HTTP polling every 5 seconds for reliable updates
- **Interactive Features**: The visualization includes enhanced interactive features:
  - Double-click on a node to make it the current node
  - Right-click on a node to access a context menu for node deletion
  - Hover over nodes to see the full content
  - Pan and zoom to explore large conversation graphs
- **Robust Connectivity**: The visualization includes several features to ensure reliable operation:
  - HTTP polling for reliable periodic updates
  - Visual notifications of successful updates
  - Detailed error handling and logging for troubleshooting
  - Fallback to page reload if polling fails
- **Server Port**: If you encounter an "Address already in use" error (common on macOS where AirPlay uses port 5000), you can specify a different port:
  ```bash
  > /visualize --port 5001
  ```
