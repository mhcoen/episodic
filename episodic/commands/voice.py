"""
Voice mode command handlers for Episodic.

This module provides command handlers for voice mode - speech input and TTS output.
"""

from typing import Optional
import typer

from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def voice(action: Optional[str] = None):
    """
    Toggle or configure voice mode.

    Args:
        action: Subcommand (on/off/status/stt/tts) or None for toggle
    """
    if action is None:
        # Toggle
        if config.get("voice_mode", False):
            voice_off()
        else:
            voice_on()
    elif action == "on":
        voice_on()
    elif action == "off":
        voice_off()
    elif action == "status":
        voice_status()
    elif action == "stt":
        voice_stt_menu()
    elif action == "tts":
        voice_tts_menu()
    else:
        typer.secho(f"Unknown voice action: {action}", fg="yellow")
        typer.secho("Usage: /voice [on|off|status|stt|tts]", fg=get_text_color())


def voice_on():
    """Enable voice mode."""
    # Check dependencies first
    missing = _check_dependencies()
    if missing:
        typer.secho("Missing dependencies for voice mode:", fg="red")
        for dep in missing:
            typer.secho(f"  - {dep}", fg="yellow")
        typer.secho("\nInstall with: pip install sounddevice webrtcvad numpy", fg=get_text_color())
        return

    from episodic.voice import get_voice_manager, VoiceState

    manager = get_voice_manager()

    def on_state_change(state: VoiceState):
        """Display state changes."""
        if state == VoiceState.LISTENING:
            typer.secho("Listening...", fg="cyan", dim=True)
        elif state == VoiceState.PROCESSING:
            typer.secho("Processing...", fg="cyan", dim=True)

    manager.start(on_state_change=on_state_change)
    config.set("voice_mode", True)

    typer.secho("Voice mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)

    stt = config.get("voice_stt_provider", "local_whisper")
    tts = config.get("voice_tts_provider", "local_piper")
    typer.secho(f"STT: {stt}  |  TTS: {tts}", fg=get_text_color())
    typer.secho("Say 'exit voice' to disable, or use /voice off", fg=typer.colors.WHITE, dim=True)


def voice_off():
    """Disable voice mode."""
    from episodic.voice import get_voice_manager

    manager = get_voice_manager()
    manager.stop()
    config.set("voice_mode", False)

    typer.secho("Voice mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("DISABLED", fg="bright_red", bold=True)


def voice_status():
    """Show voice mode status and configuration."""
    typer.secho("Voice Mode Status", fg=get_system_color(), bold=True)
    typer.secho("-" * 30, fg=get_text_color())

    enabled = config.get("voice_mode", False)
    status_color = "bright_green" if enabled else "bright_red"
    typer.secho(f"Status: ", nl=False, fg=get_text_color())
    typer.secho("ENABLED" if enabled else "DISABLED", fg=status_color, bold=True)

    typer.echo()
    typer.secho("Providers:", fg=get_system_color())
    typer.secho(f"  STT: {config.get('voice_stt_provider', 'local_whisper')}", fg=get_text_color())
    typer.secho(f"  TTS: {config.get('voice_tts_provider', 'local_piper')}", fg=get_text_color())

    typer.echo()
    typer.secho("Settings:", fg=get_system_color())
    typer.secho(f"  TTS enabled: {config.get('voice_tts_enabled', True)}", fg=get_text_color())
    typer.secho(f"  Audio cues: {config.get('voice_audio_cues', True)}", fg=get_text_color())
    typer.secho(f"  Silence threshold: {config.get('voice_silence_threshold_ms', 1000)}ms", fg=get_text_color())
    typer.secho(f"  VAD aggressiveness: {config.get('voice_vad_aggressiveness', 2)}", fg=get_text_color())


def voice_stt_menu():
    """Show and configure STT provider."""
    typer.secho("Speech-to-Text Providers", fg=get_system_color(), bold=True)
    typer.secho("-" * 30, fg=get_text_color())

    current = config.get("voice_stt_provider", "local_whisper")

    providers = [
        ("local_whisper", "Local (faster-whisper)", "Free, runs locally"),
        ("openai_whisper", "OpenAI Whisper API", "~$0.006/min, excellent accuracy"),
        ("deepgram", "Deepgram API", "~$0.008/min, real-time streaming"),
    ]

    for i, (key, name, desc) in enumerate(providers, 1):
        marker = " *" if key == current else ""
        typer.secho(f"  {i}. {name}{marker}", fg=get_text_color())
        typer.secho(f"     {desc}", fg=typer.colors.WHITE, dim=True)

    typer.echo()
    typer.secho("Set with: /set voice_stt_provider <name>", fg=typer.colors.WHITE, dim=True)


def voice_tts_menu():
    """Show and configure TTS provider."""
    typer.secho("Text-to-Speech Providers", fg=get_system_color(), bold=True)
    typer.secho("-" * 30, fg=get_text_color())

    current = config.get("voice_tts_provider", "local_piper")

    providers = [
        ("local_piper", "Local Piper", "Free, fast, lower quality"),
        ("local_xtts", "Local XTTS", "Free, high quality, slow first load (~18s)"),
        ("openai_tts", "OpenAI TTS", "~$0.015/min, good quality"),
        ("elevenlabs", "ElevenLabs", "~$0.20/1k chars, highest quality"),
    ]

    for i, (key, name, desc) in enumerate(providers, 1):
        marker = " *" if key == current else ""
        typer.secho(f"  {i}. {name}{marker}", fg=get_text_color())
        typer.secho(f"     {desc}", fg=typer.colors.WHITE, dim=True)

    typer.echo()
    typer.secho("Set with: /set voice_tts_provider <name>", fg=typer.colors.WHITE, dim=True)


def _check_dependencies() -> list:
    """Check for required voice dependencies."""
    missing = []

    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        import webrtcvad
    except ImportError:
        missing.append("webrtcvad")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    return missing
