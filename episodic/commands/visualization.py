"""
Visualization command for the Episodic CLI.
"""

import typer
from typing import Optional
from episodic.server import start_server, stop_server
from episodic.visualization import visualize_dag
from episodic.configuration import (
    DEFAULT_VISUALIZATION_PORT, get_system_color, get_heading_color, get_text_color
)


def visualize(
    output: Optional[str] = None, 
    no_browser: bool = False, 
    port: int = DEFAULT_VISUALIZATION_PORT
):
    """Generate and optionally display a visualization of the conversation graph."""
    import webbrowser
    import os
    import time
    
    try:
        if not no_browser:
            # Start server first, then generate interactive visualization
            server_url = start_server(server_port=port)
            typer.secho(f"\n🌐 Starting visualization server at {server_url}", fg=get_heading_color())
            
            # Generate interactive visualization that connects to the server
            output_path = visualize_dag(output, interactive=True, server_url=server_url)
            
            typer.secho("💡 Press Ctrl+C when done to stop the server.", fg=get_system_color())
            
            # Give the server a moment to fully start before opening browser
            time.sleep(1)
            
            typer.secho(f"📊 Opening browser to: {server_url}", fg=get_system_color())
            webbrowser.open(server_url)
            
            # Keep the server running until the user presses Ctrl+C
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                typer.secho("\n🛑 Stopping server...", fg=get_system_color())
                stop_server()
                typer.secho("✅ Server stopped.", fg=get_system_color())
        else:
            # Generate static visualization without interactive features
            output_path = visualize_dag(output, interactive=False)
            if output_path:
                typer.secho(f"✅ Visualization saved to: {output_path}", fg=get_system_color())
                typer.secho(f"📊 Opening visualization in browser: {output_path}", fg=get_system_color())
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except ImportError as e:
        typer.secho(f"Error: Missing required dependencies for visualization: {str(e)}", fg="red")
        typer.secho("Install visualization dependencies with: pip install networkx plotly", fg=get_text_color())
    except Exception as e:
        typer.secho(f"Error starting visualization: {str(e)}", fg="red", err=True)