"""
Voice mode command handlers for Episodic.

This module provides command handlers for voice mode - speech input and TTS output.
"""

from typing import Optional
import typer

from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def voice(action: Optional[str] = None, arg: Optional[str] = None):
    """
    Toggle or configure voice mode.

    Args:
        action: Subcommand (on/off/status/stt/tts/info) or None for toggle
        arg: Optional argument (e.g., provider number for stt/tts)
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
        voice_stt_menu(arg)
    elif action == "tts":
        voice_tts_menu(arg)
    elif action == "info":
        voice_info()
    else:
        typer.secho(f"Unknown voice action: {action}", fg="yellow")
        typer.secho("Usage: /voice [on|off|status|stt|tts|info]", fg=get_text_color())


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

    # No visual status callback - audio cues handle feedback
    # Visual status indicators caused display corruption issues
    manager.start()
    config.set("voice_mode", True)

    typer.secho("Voice mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)

    stt = config.get("voice_stt_provider", "openai_whisper")
    tts = config.get("voice_tts_provider", "openai_tts")
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
    typer.secho(f"  STT: {config.get('voice_stt_provider', 'openai_whisper')}", fg=get_text_color())
    typer.secho(f"  TTS: {config.get('voice_tts_provider', 'openai_tts')}", fg=get_text_color())

    typer.echo()
    typer.secho("Settings:", fg=get_system_color())
    typer.secho(f"  TTS enabled: {config.get('voice_tts_enabled', True)}", fg=get_text_color())
    typer.secho(f"  Audio cues: {config.get('voice_audio_cues', True)}", fg=get_text_color())
    typer.secho(f"  Silence threshold: {config.get('voice_silence_threshold_ms', 1000)}ms", fg=get_text_color())
    typer.secho(f"  VAD aggressiveness: {config.get('voice_vad_aggressiveness', 2)}", fg=get_text_color())


def voice_stt_menu(selection: Optional[str] = None):
    """Show and configure STT provider."""
    from episodic.voice_pricing import get_stt_cost_per_minute

    # Build provider list with dynamic pricing
    whisper_price = get_stt_cost_per_minute("openai_whisper")
    deepgram_price = get_stt_cost_per_minute("deepgram")

    providers = [
        ("local_whisper", "Local (faster-whisper)", "Free, runs locally"),
        ("openai_whisper", "OpenAI Whisper API", f"~${whisper_price}/min, excellent accuracy"),
        ("deepgram", "Deepgram API", f"~${deepgram_price}/min, real-time streaming"),
    ]

    # Handle selection by number
    if selection is not None:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(providers):
                provider_key = providers[idx][0]
                config.set("voice_stt_provider", provider_key)
                typer.secho(f"STT provider set to: {providers[idx][1]}", fg="bright_green")
                return
            else:
                typer.secho(f"Invalid selection: {selection}. Choose 1-{len(providers)}", fg="yellow")
                return
        except ValueError:
            typer.secho(f"Invalid selection: {selection}. Use a number 1-{len(providers)}", fg="yellow")
            return

    # Show menu
    typer.secho("Speech-to-Text Providers", fg=get_system_color(), bold=True)
    typer.secho("-" * 30, fg=get_text_color())

    current = config.get("voice_stt_provider", "openai_whisper")

    for i, (key, name, desc) in enumerate(providers, 1):
        marker = " *" if key == current else ""
        typer.secho(f"  {i}. {name}{marker}", fg=get_text_color())
        typer.secho(f"     {desc}", fg=typer.colors.WHITE, dim=True)

    typer.echo()
    typer.secho("Set with: /voice stt <number>", fg=typer.colors.WHITE, dim=True)


def voice_tts_menu(selection: Optional[str] = None):
    """Show and configure TTS provider."""
    from episodic.voice_pricing import get_tts_cost_per_1k_chars

    # Build provider list with dynamic pricing
    openai_price = get_tts_cost_per_1k_chars("openai_tts", "tts-1")
    elevenlabs_price = get_tts_cost_per_1k_chars("elevenlabs")

    providers = [
        ("local_piper", "Local Piper", "Free, fast, lower quality"),
        ("local_xtts", "Local XTTS", "Free, high quality, slow first load (~18s)"),
        ("openai_tts", "OpenAI TTS", f"~${openai_price}/1k chars, good quality"),
        ("elevenlabs", "ElevenLabs", f"~${elevenlabs_price}/1k chars, highest quality"),
    ]

    # Handle selection by number
    if selection is not None:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(providers):
                provider_key = providers[idx][0]
                config.set("voice_tts_provider", provider_key)
                typer.secho(f"TTS provider set to: {providers[idx][1]}", fg="bright_green")
                return
            else:
                typer.secho(f"Invalid selection: {selection}. Choose 1-{len(providers)}", fg="yellow")
                return
        except ValueError:
            typer.secho(f"Invalid selection: {selection}. Use a number 1-{len(providers)}", fg="yellow")
            return

    # Show menu
    typer.secho("Text-to-Speech Providers", fg=get_system_color(), bold=True)
    typer.secho("-" * 30, fg=get_text_color())

    current = config.get("voice_tts_provider", "openai_tts")

    for i, (key, name, desc) in enumerate(providers, 1):
        marker = " *" if key == current else ""
        typer.secho(f"  {i}. {name}{marker}", fg=get_text_color())
        typer.secho(f"     {desc}", fg=typer.colors.WHITE, dim=True)

    typer.echo()
    typer.secho("Set with: /voice tts <number>", fg=typer.colors.WHITE, dim=True)


def voice_info():
    """Show audio device information and test microphone access."""
    typer.secho("Audio Device Information", fg=get_system_color(), bold=True)
    typer.secho("=" * 50, fg=get_text_color())

    # Check dependencies first
    missing = _check_dependencies()
    if missing:
        typer.secho("Missing dependencies:", fg="red")
        for dep in missing:
            typer.secho(f"  - {dep}", fg="yellow")
        typer.secho("\nInstall with: pip install sounddevice webrtcvad numpy", fg=get_text_color())
        return

    try:
        import sounddevice as sd
        import numpy as np

        devices = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        # Input devices
        typer.echo()
        typer.secho("INPUT Devices (Microphones)", fg=get_system_color(), bold=True)
        typer.secho("-" * 40, fg=get_text_color())

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                if i == default_input:
                    typer.secho(f"  [{i}] {dev['name']}", fg="bright_green", bold=True)
                    typer.secho(f"      {dev['max_input_channels']} ch, {int(dev['default_samplerate'])} Hz  <-- DEFAULT", fg="bright_green")
                else:
                    typer.secho(f"  [{i}] {dev['name']}", fg=get_text_color())
                    typer.secho(f"      {dev['max_input_channels']} ch, {int(dev['default_samplerate'])} Hz", fg=typer.colors.WHITE, dim=True)

        # Output devices
        typer.echo()
        typer.secho("OUTPUT Devices (Speakers)", fg=get_system_color(), bold=True)
        typer.secho("-" * 40, fg=get_text_color())

        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                if i == default_output:
                    typer.secho(f"  [{i}] {dev['name']}", fg="bright_green", bold=True)
                    typer.secho(f"      {dev['max_output_channels']} ch, {int(dev['default_samplerate'])} Hz  <-- DEFAULT", fg="bright_green")
                else:
                    typer.secho(f"  [{i}] {dev['name']}", fg=get_text_color())
                    typer.secho(f"      {dev['max_output_channels']} ch, {int(dev['default_samplerate'])} Hz", fg=typer.colors.WHITE, dim=True)

        # Test microphone access
        typer.echo()
        typer.secho("Microphone Test", fg=get_system_color(), bold=True)
        typer.secho("-" * 40, fg=get_text_color())

        try:
            input_info = sd.query_devices(default_input)
            device_sr = int(input_info['default_samplerate'])
            typer.secho(f"  Testing device: {input_info['name']}", fg=get_text_color())
            typer.secho(f"  Native sample rate: {device_sr} Hz", fg=get_text_color())

            # Try recording a brief sample
            test_duration = 0.1
            _ = sd.rec(int(test_duration * device_sr), samplerate=device_sr, channels=1, dtype='int16')
            sd.wait()

            typer.secho("  Microphone access: ", nl=False, fg=get_text_color())
            typer.secho("OK", fg="bright_green", bold=True)

        except Exception as e:
            typer.secho("  Microphone access: ", nl=False, fg=get_text_color())
            typer.secho("FAILED", fg="bright_red", bold=True)
            typer.secho(f"  Error: {e}", fg="red")
            typer.echo()
            typer.secho("Troubleshooting:", fg="yellow", bold=True)
            typer.secho("  1. Check System Settings > Privacy & Security > Microphone", fg=get_text_color())
            typer.secho("     Ensure your terminal app has microphone access", fg=get_text_color())
            typer.secho("  2. Restart your terminal after granting permission", fg=get_text_color())

    except Exception as e:
        typer.secho(f"Error querying audio devices: {e}", fg="red")


def _check_dependencies() -> list:
    """Check for required voice dependencies."""
    import warnings
    missing = []

    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
            import webrtcvad
    except ImportError:
        missing.append("webrtcvad")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    return missing
