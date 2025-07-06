"""
Settings and configuration commands for the Episodic CLI.

Handles parameter configuration, verification, and cost tracking.
"""

import typer
from typing import Optional
from episodic.config import config
from episodic.configuration import (
    DEFAULT_COLOR_MODE, get_text_color, get_system_color, get_heading_color
)
from episodic.conversation import conversation_manager
from episodic.llm import enable_cache, disable_cache
from episodic.param_mappings import normalize_param_name, get_display_name


# Global variables for settings
default_context_depth = 5
default_semdepth = 2


def set(param: Optional[str] = None, value: Optional[str] = None):
    """Configure various parameters."""
    global default_context_depth, default_semdepth

    # If no parameter is provided, show all parameters and their current values
    if not param:
        typer.secho("Current parameters:", fg=get_heading_color(), bold=True)
        typer.secho("(Changes are temporary - resets on restart)", fg=get_text_color())
        
        # Core conversation settings
        typer.secho("\nConversation:", fg=get_heading_color())
        typer.secho("  depth: ", nl=False, fg=get_text_color())
        typer.secho(f"{default_context_depth}", fg=get_system_color())
        typer.secho("  cache: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('use_context_cache', True)}", fg=get_system_color())
        
        # Display settings
        typer.secho("\nDisplay:", fg=get_heading_color())
        typer.secho("  color: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('color_mode', DEFAULT_COLOR_MODE)}", fg=get_system_color())
        typer.secho("  wrap: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('text_wrap', True)}", fg=get_system_color())
        typer.secho("  stream: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_responses', True)}", fg=get_system_color())
        typer.secho("  stream-rate: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_rate', 15)} words/sec", fg=get_system_color())
        typer.secho("  stream-constant-rate: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_constant_rate', False)}", fg=get_system_color())
        typer.secho("  stream-natural-rhythm: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_natural_rhythm', False)}", fg=get_system_color())
        typer.secho("  stream-char-mode: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_char_mode', False)}", fg=get_system_color())
        typer.secho("  stream-char-rate: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_char_rate', 1000)} chars/sec", fg=get_system_color())
        typer.secho("  stream-line-delay: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('stream_line_delay', 0.1)}s", fg=get_system_color())
        typer.secho("  cost: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_cost', False)}", fg=get_system_color())
        typer.secho("  topics: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_topics', False)}", fg=get_system_color())
        typer.secho("  hybrid-topics: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('use_hybrid_topic_detection', False)}", fg=get_system_color())
        
        # Analysis features
        typer.secho("\nAnalysis:", fg=get_heading_color())
        typer.secho("  drift: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_drift', True)}", fg=get_system_color())
        typer.secho("  semdepth: ", nl=False, fg=get_text_color())
        typer.secho(f"{default_semdepth}", fg=get_system_color())
        
        # Topic detection & compression
        typer.secho("\nTopic Management:", fg=get_heading_color())
        typer.secho("  topic-auto: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('automatic_topic_detection', True)}", fg=get_system_color())
        typer.secho("  topic-model: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('topic_detection_model', 'ollama/llama3')}", fg=get_system_color())
        typer.secho("  comp-auto: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('auto_compress_topics', True)}", fg=get_system_color())
        typer.secho("  comp-model: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('compression_model', 'ollama/llama3')}", fg=get_system_color())
        typer.secho("  comp-min: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('compression_min_nodes', 10)}", fg=get_system_color())
        typer.secho("  comp-notify: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_compression_notifications', True)}", fg=get_system_color())
        typer.secho("  topic-min: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('min_messages_before_topic_change', 8)}", fg=get_system_color())
        
        # Performance monitoring
        typer.secho("\nPerformance:", fg=get_heading_color())
        typer.secho("  benchmark: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('benchmark', False)}", fg=get_system_color())
        typer.secho("  benchmark-display: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('benchmark_display', False)}", fg=get_system_color())
        
        # Debug
        typer.secho("\nDebugging:", fg=get_heading_color())
        typer.secho("  debug: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('debug', False)}", fg=get_system_color())
        
        # Model Parameters
        typer.secho("\nModel Parameters:", fg=get_heading_color())
        typer.secho("  Use /model-params to view parameter details", fg=get_text_color())
        typer.secho("  main.temp, topic.temp, comp.temp, etc.", fg=get_system_color())
        return

    # Normalize parameter name (convert dashes to underscores, expand aliases)
    normalized_param = normalize_param_name(param.lower())
    
    # Handle the 'cost' parameter
    if normalized_param == "show_cost":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("show_cost", False)
            config.set("show_cost", not current)
            typer.echo(f"Cost display: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("show_cost", val)
            typer.echo(f"Cost display: {'ON' if val else 'OFF'} (this session only)")

    # Handle the 'cache' parameter
    elif normalized_param == "use_context_cache":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("use_context_cache", True)
            if current:
                disable_cache()
            else:
                enable_cache()
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            if val:
                enable_cache()
            else:
                disable_cache()

    # Handle the 'topics' parameter
    elif normalized_param == "show_topics":
        if not value:
            current = config.get("show_topics", False)
            typer.echo(f"Current topics display: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("show_topics", val)
            typer.echo(f"Topics display set to {val}")
    
    # Handle hybrid topic detection
    elif normalized_param == "use_hybrid_topic_detection":
        if not value:
            current = config.get("use_hybrid_topic_detection", False)
            typer.echo(f"Hybrid topic detection: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("use_hybrid_topic_detection", val)
            typer.echo(f"Hybrid topic detection: {'ON' if val else 'OFF'}")
            if val:
                typer.echo("Note: Requires sentence-transformers for embeddings")

    # Handle the 'stream' parameter for streaming responses
    elif normalized_param == "stream_responses":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("stream_responses", True)
            config.set("stream_responses", not current)
            typer.echo(f"Response streaming: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("stream_responses", val)
            typer.echo(f"Response streaming: {'ON' if val else 'OFF'}")

    # Handle 'stream_rate' parameter
    elif normalized_param == "stream_rate":
        if not value:
            typer.echo(f"Current stream rate: {config.get('stream_rate', 15)} words/sec")
        else:
            try:
                rate = int(value)
                if rate < 1:
                    typer.echo("Stream rate must be at least 1 word/sec")
                elif rate > 100:
                    typer.echo("Stream rate must be at most 100 words/sec")
                else:
                    config.set("stream_rate", rate)
                    typer.echo(f"Stream rate set to {rate} words/sec")
            except ValueError:
                typer.echo("Stream rate must be a number")

    # Handle 'stream_constant_rate' parameter
    elif normalized_param == "stream_constant_rate":
        if not value:
            current = config.get("stream_constant_rate", False)
            typer.echo(f"Current constant rate streaming: {current}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("stream_constant_rate", val)
            typer.echo(f"Constant rate streaming: {'ON' if val else 'OFF'}")

    # Handle 'stream_natural_rhythm' parameter
    elif normalized_param == "stream_natural_rhythm":
        if not value:
            current = config.get("stream_natural_rhythm", False)
            typer.echo(f"Current natural rhythm streaming: {current}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("stream_natural_rhythm", val)
            typer.echo(f"Natural rhythm streaming: {'ON' if val else 'OFF'}")

    # Handle 'stream_char_mode' parameter
    elif normalized_param == "stream_char_mode":
        if not value:
            current = config.get("stream_char_mode", False)
            typer.echo(f"Current character streaming mode: {current}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("stream_char_mode", val)
            typer.echo(f"Character streaming mode: {'ON' if val else 'OFF'}")

    # Handle 'stream_char_rate' parameter
    elif normalized_param == "stream_char_rate":
        if not value:
            typer.echo(f"Current character rate: {config.get('stream_char_rate', 1000)} chars/sec")
        else:
            try:
                rate = int(value)
                if rate < 100:
                    typer.echo("Character rate must be at least 100 chars/sec")
                elif rate > 10000:
                    typer.echo("Character rate must be at most 10000 chars/sec")
                else:
                    config.set("stream_char_rate", rate)
                    typer.echo(f"Character rate set to {rate} chars/sec")
            except ValueError:
                typer.echo("Character rate must be a number")

    # Handle 'stream_line_delay' parameter
    elif normalized_param == "stream_line_delay":
        if not value:
            typer.echo(f"Current line delay: {config.get('stream_line_delay', 0.1)}s")
        else:
            try:
                delay = float(value)
                if delay < 0:
                    typer.echo("Line delay must be non-negative")
                elif delay > 1.0:
                    typer.echo("Line delay must be at most 1 second")
                else:
                    config.set("stream_line_delay", delay)
                    typer.echo(f"Line delay set to {delay}s")
            except ValueError:
                typer.echo("Line delay must be a number")

    # Handle the 'depth' parameter for context depth
    elif normalized_param == "depth":
        if not value:
            typer.echo(f"Current context depth: {default_context_depth}")
        else:
            try:
                depth = int(value)
                if depth < 0:
                    typer.echo("Context depth must be non-negative")
                else:
                    default_context_depth = depth
                    typer.echo(f"Context depth set to {depth}")
            except ValueError:
                typer.echo("Context depth must be a number")

    # Handle the 'drift' parameter
    elif normalized_param == "show_drift":
        if not value:
            current = config.get("show_drift", True)
            typer.echo(f"Current drift display: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("show_drift", val)
            typer.echo(f"Drift display set to {val}")

    # Handle the 'semdepth' parameter for semantic depth
    elif normalized_param == "semdepth":
        if not value:
            typer.echo(f"Current semantic depth: {default_semdepth}")
        else:
            try:
                depth = int(value)
                if depth < 0:
                    typer.echo("Semantic depth must be non-negative")
                else:
                    default_semdepth = depth
                    typer.echo(f"Semantic depth set to {depth}")
            except ValueError:
                typer.echo("Semantic depth must be a number")

    # Handle debug parameter
    elif normalized_param == "debug":
        if not value:
            current = config.get("debug", False)
            typer.echo(f"Current debug mode: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("debug", val)
            typer.echo(f"Debug mode set to {val}")

    # Handle color parameter
    elif normalized_param == "color_mode":
        if not value:
            current = config.get("color_mode", DEFAULT_COLOR_MODE)
            typer.echo(f"Current color mode: {current}")
        else:
            if value.lower() in ["none", "basic", "full"]:
                config.set("color_mode", value.lower())
                typer.echo(f"Color mode set to: {value.lower()}")
                typer.echo("Note: Restart the session to see color changes in the prompt")
            else:
                typer.echo("Invalid color mode. Available options: none, basic, full")

    # Handle the 'wrap' parameter
    elif normalized_param == "text_wrap":
        if not value:
            current = config.get("text_wrap", True)
            typer.echo(f"Current text wrapping: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("text_wrap", val)
            typer.echo(f"Text wrapping: {'ON' if val else 'OFF'}")

    # Handle the 'automatic_topic_detection' parameter
    elif normalized_param == "automatic_topic_detection":
        if not value:
            current = config.get("automatic_topic_detection", True)
            typer.echo(f"Current automatic topic detection: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("automatic_topic_detection", val)
            typer.echo(f"Automatic topic detection: {'ON' if val else 'OFF'}")
            if not val:
                typer.echo("Use '/index <n>' to manually detect topics")

    # Handle the 'auto_compress_topics' parameter
    elif normalized_param == "auto_compress_topics":
        if not value:
            current = config.get("auto_compress_topics", True)
            typer.echo(f"Current auto compression: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("auto_compress_topics", val)
            typer.echo(f"Auto compression: {'ON' if val else 'OFF'}")
            # Restart compression manager if needed
            if val:
                from episodic.compression import start_auto_compression
                start_auto_compression()
            else:
                from episodic.compression import stop_auto_compression
                stop_auto_compression()

    # Handle the 'show_compression_notifications' parameter
    elif normalized_param == "show_compression_notifications":
        if not value:
            current = config.get("show_compression_notifications", True)
            typer.echo(f"Current compression notifications: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("show_compression_notifications", val)
            typer.echo(f"Compression notifications: {'ON' if val else 'OFF'}")

    # Handle the 'compression_min_nodes' parameter
    elif normalized_param == "compression_min_nodes":
        if not value:
            typer.echo(f"Current compression minimum nodes: {config.get('compression_min_nodes', 10)}")
        else:
            try:
                min_nodes = int(value)
                if min_nodes < 3:
                    typer.echo("Compression minimum nodes must be at least 3")
                else:
                    config.set("compression_min_nodes", min_nodes)
                    typer.echo(f"Compression minimum nodes set to {min_nodes}")
            except ValueError:
                typer.echo("Compression minimum nodes must be a number")

    # Handle the 'compression_model' parameter
    elif normalized_param == "compression_model":
        if not value:
            typer.echo(f"Current compression model: {config.get('compression_model', 'ollama/llama3')}")
        else:
            config.set("compression_model", value)
            typer.echo(f"Compression model set to {value}")

    # Handle the 'topic_detection_model' parameter
    elif normalized_param == "topic_detection_model":
        if not value:
            typer.echo(f"Current topic detection model: {config.get('topic_detection_model', 'ollama/llama3')}")
        else:
            config.set("topic_detection_model", value)
            typer.echo(f"Topic detection model set to {value}")

    # Handle the 'min_messages_before_topic_change' parameter
    elif normalized_param == "min_messages_before_topic_change":
        if not value:
            typer.echo(f"Current minimum messages: {config.get('min_messages_before_topic_change', 8)}")
        else:
            try:
                min_messages = int(value)
                if min_messages < 1:
                    typer.echo("Minimum messages must be at least 1")
                else:
                    config.set("min_messages_before_topic_change", min_messages)
                    typer.echo(f"Minimum messages before topic change set to {min_messages}")
            except ValueError:
                typer.echo("Minimum messages must be a number")

    # Handle benchmark parameter
    elif normalized_param == "benchmark":
        if not value:
            current = config.get("benchmark", False)
            typer.echo(f"Current benchmark mode: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("benchmark", val)
            typer.echo(f"Benchmark mode set to {val}")
            if val:
                typer.echo("Use '/benchmark' to see performance summary")

    # Handle benchmark_display parameter
    elif normalized_param == "benchmark_display":
        if not value:
            current = config.get("benchmark_display", False)
            typer.echo(f"Current benchmark display: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("benchmark_display", val)
            typer.echo(f"Benchmark display set to {val}")

    # Handle model parameter syntax (main.temp, topic.max, etc.)
    elif '.' in normalized_param:
        try:
            config.set(normalized_param, value)
            typer.echo(f"Set {param} to {value} (this session only)")
        except ValueError as e:
            typer.secho(f"Error: {e}", fg="red")
        except Exception as e:
            typer.secho(f"Failed to set {param}: {e}", fg="red")

    else:
        # Check if value is None (user just typed /set <param> without a value)
        if value is None:
            # Try to show the current value instead of setting to None
            # Check if normalized param exists in config (even if value is None)
            if normalized_param in config.config:
                current_value = config.get(normalized_param)
                typer.secho(f"Current value of {param}: ", nl=False, fg=get_text_color())
                if current_value is None:
                    # Check if we have defaults for this parameter
                    from episodic.config_defaults import DEFAULT_CONFIG
                    if normalized_param in DEFAULT_CONFIG:
                        default_value = DEFAULT_CONFIG[normalized_param]
                        typer.secho("None (using defaults)", fg=get_system_color())
                        if isinstance(default_value, dict):
                            typer.secho("  Default values:", fg=get_text_color())
                            for k, v in default_value.items():
                                typer.secho(f"    {k}: {v}", fg=get_system_color())
                    else:
                        typer.secho("None", fg=get_system_color())
                elif isinstance(current_value, dict):
                    typer.secho("<dict>", fg=get_system_color())
                    # Show dict contents
                    for k, v in current_value.items():
                        typer.secho(f"  {k}: {v}", fg=get_system_color())
                else:
                    typer.secho(f"{current_value}", fg=get_system_color())
                # Show documentation if available
                doc = config.get_doc(normalized_param)
                if doc != "No documentation available":
                    typer.secho(f"  ({doc})", fg=get_text_color())
            else:
                typer.echo(f"Parameter '{param}' does not exist")
        else:
            # Try to set it as a general config parameter
            try:
                config.set(normalized_param, value)
                typer.echo(f"Set {param} to {value} (this session only)")
                # Show documentation if available
                doc = config.get_doc(normalized_param)
                if doc != "No documentation available":
                    typer.secho(f"  ({doc})", fg=get_text_color())
            except:
                typer.echo(f"Unknown parameter: {param}")
                typer.secho("\nAvailable parameters:", fg=get_heading_color())
                typer.secho("  Conversation: ", nl=False, fg=get_heading_color())
                typer.secho("depth, cache", fg=get_system_color())
                typer.secho("  Display: ", nl=False, fg=get_heading_color())
                typer.secho("color, wrap, stream, stream-rate, stream-constant, cost, topics", fg=get_system_color())
                typer.secho("  Analysis: ", nl=False, fg=get_heading_color())
                typer.secho("drift, semdepth", fg=get_system_color())
                typer.secho("  Topic Management: ", nl=False, fg=get_heading_color())
                typer.secho("topic-auto, topic-model, topic-min, comp-auto, comp-model, comp-min", fg=get_system_color())
                typer.secho("  Performance: ", nl=False, fg=get_heading_color())
                typer.secho("benchmark, benchmark-display", fg=get_system_color())
                typer.secho("  Model Parameters: ", nl=False, fg=get_heading_color())
                typer.secho("main.temp, topic.temp, comp.temp (see /model-params)", fg=get_system_color())
                typer.secho("  Debugging: ", nl=False, fg=get_heading_color())
                typer.secho("debug", fg=get_system_color())
                typer.echo("Use 'set' without arguments to see all parameters and their current values")


def verify():
    """Verify database and configuration integrity."""
    from episodic.db import verify_database_integrity
    from episodic.benchmark import benchmark_operation
    import os
    
    typer.secho("\n🔍 Verifying Episodic Configuration", fg=get_heading_color(), bold=True)
    typer.secho("=" * 50, fg=get_heading_color())
    
    # Check database
    typer.secho("\n📊 Database:", fg=get_heading_color())
    with benchmark_operation("Database verification"):
        db_path = os.environ.get("EPISODIC_DB_PATH", os.path.expanduser("~/.episodic/episodic.db"))
        typer.echo(f"  Path: {db_path}")
        
        if os.path.exists(db_path):
            typer.echo(f"  Status: ✅ Found")
            # Get file size
            size = os.path.getsize(db_path)
            if size < 1024:
                typer.echo(f"  Size: {size} bytes")
            elif size < 1024 * 1024:
                typer.echo(f"  Size: {size/1024:.1f} KB")
            else:
                typer.echo(f"  Size: {size/(1024*1024):.1f} MB")
            
            # Run integrity check
            issues = verify_database_integrity()
            if issues:
                typer.secho("  Integrity: ❌ Issues found:", fg="red")
                for issue in issues:
                    typer.secho(f"    - {issue}", fg="red")
            else:
                typer.echo("  Integrity: ✅ No issues found")
        else:
            typer.secho("  Status: ❌ Not found (run 'init' to create)", fg="yellow")
    
    # Check configuration
    typer.secho("\n⚙️  Configuration:", fg=get_heading_color())
    typer.echo(f"  Active prompt: {config.get('active_prompt', 'default')}")
    typer.echo(f"  Model: {config.get('model', 'Not set')}")
    typer.echo(f"  Context caching: {config.get('use_context_cache', True)}")
    typer.echo(f"  Auto compression: {config.get('auto_compress_topics', True)}")
    
    # Check environment
    typer.secho("\n🌍 Environment:", fg=get_heading_color())
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic", 
        "GROQ_API_KEY": "Groq",
        "TOGETHER_API_KEY": "Together AI"
    }
    
    found_keys = []
    for key, provider in api_keys.items():
        if os.environ.get(key):
            found_keys.append(provider)
    
    if found_keys:
        typer.echo(f"  API Keys: ✅ {', '.join(found_keys)}")
    else:
        typer.secho("  API Keys: ⚠️  None found", fg="yellow")
    
    # Check if Ollama is available
    try:
        import subprocess
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            typer.echo("  Ollama: ✅ Installed")
        else:
            typer.echo("  Ollama: ❌ Not available")
    except:
        typer.echo("  Ollama: ❌ Not found")
    
    typer.secho("\n" + "=" * 50, fg=get_heading_color())


def model_params(param_set: Optional[str] = None):
    """Display model parameters for different contexts."""
    
    def format_param_value(value):
        """Format parameter value for display."""
        if value is None:
            return "default"
        elif isinstance(value, list):
            if not value:
                return "[]"
            return f"[{', '.join(repr(v) for v in value)}]"
        elif isinstance(value, float):
            return f"{value:.1f}" if value == int(value) else f"{value}"
        else:
            return str(value)
    
    def display_param_set(name: str, title: str):
        """Display a single parameter set."""
        params = config.get(name, {})
        typer.secho(f"\n{title}:", fg=get_heading_color(), bold=True)
        
        typer.secho("  temperature: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('temperature')), fg=get_system_color())
        
        typer.secho("  max_tokens: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('max_tokens')), fg=get_system_color())
        
        typer.secho("  top_p: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('top_p')), fg=get_system_color())
        
        typer.secho("  presence_penalty: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('presence_penalty')), fg=get_system_color())
        
        typer.secho("  frequency_penalty: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('frequency_penalty')), fg=get_system_color())
        
        typer.secho("  stop: ", nl=False, fg=get_text_color())
        typer.secho(format_param_value(params.get('stop')), fg=get_system_color())
    
    if param_set:
        # Display specific parameter set
        param_set_map = {
            'main': ('main_params', 'Main Conversation Parameters'),
            'topic': ('topic_params', 'Topic Detection Parameters'),
            'comp': ('compression_params', 'Compression Parameters'),
            'compression': ('compression_params', 'Compression Parameters')
        }
        
        if param_set.lower() in param_set_map:
            name, title = param_set_map[param_set.lower()]
            display_param_set(name, title)
        else:
            typer.secho(f"Unknown parameter set: {param_set}", fg="red")
            typer.echo("Available sets: main, topic, compression")
    else:
        # Display all parameter sets
        typer.secho("🎛️  Model Parameters", fg=get_heading_color(), bold=True)
        typer.secho("=" * 50, fg=get_heading_color())
        
        display_param_set('main_params', 'Main Conversation')
        display_param_set('topic_params', 'Topic Detection')
        display_param_set('compression_params', 'Compression')
        
        typer.secho("\n" + "─" * 50, fg=get_heading_color())
        typer.secho("Usage examples:", fg=get_heading_color())
        typer.secho("  /set main.temp 0.8", nl=False, fg=get_system_color())
        typer.secho("         # Set main temperature", fg=get_text_color())
        typer.secho("  /set topic.max 100", nl=False, fg=get_system_color())
        typer.secho("         # Set topic max_tokens", fg=get_text_color())
        typer.secho("  /set comp.presence 0.2", nl=False, fg=get_system_color())
        typer.secho("     # Set compression presence penalty", fg=get_text_color())
        typer.secho("  /set main.reset", nl=False, fg=get_system_color())
        typer.secho("            # Reset main parameters to defaults", fg=get_text_color())
        typer.secho("  /model-params main", nl=False, fg=get_system_color())
        typer.secho("         # View only main parameters", fg=get_text_color())


def config_docs():
    """Show documentation for all configuration parameters."""
    typer.secho("\n📚 Configuration Documentation", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    
    # Group parameters by category
    categories = {
        "Core Settings": ["active_prompt", "debug", "show_cost", "show_drift"],
        "Topic Detection": [
            "automatic_topic_detection", "auto_compress_topics", 
            "min_messages_before_topic_change", "running_topic_guess",
            "show_topics", "analyze_topic_boundaries", "use_llm_boundary_analysis",
            "manual_index_window_size", "manual_index_threshold"
        ],
        "Streaming": [
            "stream_responses", "stream_rate", "stream_constant_rate"
        ],
        "Model Parameters": ["main_params", "topic_params", "compression_params"]
    }
    
    for category, params in categories.items():
        typer.secho(f"\n{category}:", fg=get_heading_color(), bold=True)
        for param in params:
            doc = config.get_doc(param)
            value = config.get(param, "not set")
            if isinstance(value, dict):
                value_str = "<dict>"
            elif isinstance(value, list):
                value_str = "<list>"
            else:
                value_str = str(value)
            
            typer.secho(f"  {param}: ", nl=False, fg=get_text_color())
            typer.secho(f"{value_str}", fg=get_system_color())
            if doc != "No documentation available":
                typer.secho(f"    → {doc}", fg=get_text_color())
    
    typer.secho("\nUse '/set <parameter> <value>' to change settings", fg=get_text_color())


def cost():
    """Show cost information for the current session."""
    
    costs = conversation_manager.get_session_costs()
    
    typer.secho("\n💰 Session Cost Summary", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.secho("Input tokens:  ", nl=False, fg=get_text_color(), bold=True)
    typer.secho(f"{costs['total_input_tokens']:,}", fg=typer.colors.BRIGHT_CYAN, bold=True)
    
    typer.secho("Output tokens: ", nl=False, fg=get_text_color(), bold=True)
    typer.secho(f"{costs['total_output_tokens']:,}", fg=typer.colors.BRIGHT_CYAN, bold=True)
    
    typer.secho("Total tokens:  ", nl=False, fg=get_text_color(), bold=True)
    typer.secho(f"{costs['total_tokens']:,}", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    
    if costs.get('cache_read_tokens', 0) > 0:
        typer.secho("Cache reads:   ", nl=False, fg=get_text_color(), bold=True)
        typer.secho(f"{costs['cache_read_tokens']:,}", fg=typer.colors.BRIGHT_GREEN, bold=True)
        
        typer.secho("Cache writes:  ", nl=False, fg=get_text_color(), bold=True)
        typer.secho(f"{costs.get('cache_write_tokens', 0):,}", fg=typer.colors.BRIGHT_GREEN, bold=True)
    
    typer.secho("─" * 40, fg=get_heading_color())
    typer.secho("Total cost: ", nl=False, fg=get_text_color(), bold=True)
    typer.secho(f"${costs['total_cost_usd']:.4f}", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    
    if costs.get('cache_savings_usd', 0) > 0:
        typer.secho("Cache savings: ", nl=False, fg=get_text_color(), bold=True)
        typer.secho(f"${costs['cache_savings_usd']:.4f}", fg=typer.colors.BRIGHT_GREEN, bold=True)