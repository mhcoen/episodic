"""
Mode switching commands for Episodic (muse vs chat mode, local vs cloud mode).
"""

from typing import Optional

import typer
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def muse_command(enable: bool = True):
    """Enable or disable muse mode (all input becomes web searches)."""
    if enable:
        config.set("muse_mode", True)
        # Also enable web search if not already enabled
        if not config.get("web_search_enabled", False):
            config.set("web_search_enabled", True)
            typer.secho("✓ Web search enabled automatically", fg="green")
        typer.secho("🎭 Muse mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("All input will be treated as web search queries", fg=get_text_color())
        typer.secho("(Like Perplexity - synthesized answers from web search)", fg=typer.colors.WHITE, dim=True)
    else:
        config.set("muse_mode", False)
        typer.secho("💬 Chat mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("Input will be sent to the LLM", fg=get_text_color())


def chat_command(enable: bool = True):
    """Enable or disable chat mode (normal LLM conversation)."""
    if enable:
        # Chat on means muse off
        config.set("muse_mode", False)
        typer.secho("💬 Chat mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("Input will be sent to the LLM", fg=get_text_color())
    else:
        # Chat off means muse on
        config.set("muse_mode", True)
        # Also enable web search if not already enabled
        if not config.get("web_search_enabled", False):
            config.set("web_search_enabled", True)
            typer.secho("✓ Web search enabled automatically", fg="green")
        typer.secho("🎭 Muse mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("All input will be treated as web search queries", fg=get_text_color())
        typer.secho("(Like Perplexity - synthesized answers from web search)", fg=typer.colors.WHITE, dim=True)


def handle_muse(args: list):
    """Handle /muse command - now directly enables muse mode."""
    # Always enable muse mode when /muse is called
    muse_command(True)


def handle_chat(args: list):
    """Handle /chat command - now directly enables chat mode."""
    # Always enable chat mode when /chat is called
    chat_command(True)


# ============================================================
# Local/Cloud Mode Switching
# ============================================================

def mode_command(mode_name: Optional[str] = None):
    """Switch between local and cloud provider modes."""
    if mode_name is None:
        show_current_mode()
    elif mode_name == "local":
        switch_to_local()
    elif mode_name == "cloud":
        switch_to_cloud()
    else:
        typer.secho(f"Unknown mode: {mode_name}", fg="red")
        typer.secho("Available modes: local, cloud", fg=get_text_color())


def switch_to_local():
    """Switch all providers to local mode."""
    local_model = config.get("local_model", "ollama/llama3.3:70b-instruct-q4_K_M")
    local_stt = config.get("local_stt_provider", "local_whisper")
    local_tts = config.get("local_tts_provider", "local_piper")

    config.set("mode", "local")
    config.set("model", local_model)
    config.set("voice_stt_provider", local_stt)
    config.set("voice_tts_provider", local_tts)

    typer.secho("🏠 Local mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)
    typer.secho(f"  Model: {local_model}", fg=get_text_color())
    typer.secho(f"  STT: {local_stt}", fg=get_text_color())
    typer.secho(f"  TTS: {local_tts}", fg=get_text_color())


def switch_to_cloud():
    """Switch all providers to cloud mode."""
    cloud_model = config.get("cloud_model", "gpt-4o-mini")
    cloud_stt = config.get("cloud_stt_provider", "openai_whisper")
    cloud_tts = config.get("cloud_tts_provider", "openai_tts")

    config.set("mode", "cloud")
    config.set("model", cloud_model)
    config.set("voice_stt_provider", cloud_stt)
    config.set("voice_tts_provider", cloud_tts)

    typer.secho("☁️  Cloud mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)
    typer.secho(f"  Model: {cloud_model}", fg=get_text_color())
    typer.secho(f"  STT: {cloud_stt}", fg=get_text_color())
    typer.secho(f"  TTS: {cloud_tts}", fg=get_text_color())


def show_current_mode():
    """Display current mode and provider settings."""
    mode = config.get("mode", "cloud")
    model = config.get("model", "gpt-4o-mini")
    stt = config.get("voice_stt_provider", "local_whisper")
    tts = config.get("voice_tts_provider", "local_piper")

    icon = "🏠" if mode == "local" else "☁️"
    typer.secho(f"{icon} Current mode: ", nl=False, fg=get_system_color())
    typer.secho(mode.upper(), fg="bright_cyan", bold=True)
    typer.secho(f"  Model: {model}", fg=get_text_color())
    typer.secho(f"  STT: {stt}", fg=get_text_color())
    typer.secho(f"  TTS: {tts}", fg=get_text_color())
    typer.echo()
    typer.secho("Switch with: /mode local  or  /mode cloud", fg=get_text_color(), dim=True)