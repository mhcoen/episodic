"""Voice infrastructure mixin: lazy provider getters and idle timer.

Split out of voice_mode.py; VoiceModeManager inherits it, so these methods run
on the instance (self._stt_provider, self._idle_timer, self._state, ...).
"""

import threading
import time
from typing import Optional

import typer

from episodic.config import config
from episodic.voice.voice_state import VoiceState


class _VoiceInfraMixin:
    """Lazy STT/TTS/audio/wake-word providers and the idle-timeout timer."""

    def _get_stt_provider(self):
        """Get or create STT provider based on config."""
        if self._stt_provider is None:
            from episodic.voice.stt_providers import get_stt_provider

            provider_name = config.get("voice_stt_provider", "openai_whisper")
            kwargs = {}

            if provider_name == "local_whisper":
                kwargs["model_size"] = config.get("voice_local_whisper_model", "base")

            self._stt_provider = get_stt_provider(provider_name, **kwargs)

        return self._stt_provider

    def _get_tts_provider(self):
        """Get or create TTS provider based on config."""
        from episodic.voice.tts_providers import get_tts_provider

        provider_name = config.get("voice_tts_provider", "openai_tts")
        kwargs = {}

        # Speed setting applies to providers that support it
        if provider_name in ("local_piper", "openai_tts"):
            kwargs["speed"] = config.get("voice_tts_speed", 1.0)

        if provider_name == "local_piper":
            kwargs["voice"] = config.get("voice_local_piper_voice", "en_US-lessac-medium")
        elif provider_name == "local_xtts":
            kwargs["speaker"] = config.get("voice_local_xtts_speaker", "Claribel Dervla")
        elif provider_name == "openai_tts":
            kwargs["voice"] = config.get("voice_openai_tts_voice", "alloy")
        elif provider_name == "azure_neural":
            kwargs["voice"] = config.get("voice_azure_neural_voice", "en-US-Ava:DragonHDLatestNeural")

        # Create cache key to detect config changes
        cache_key = f"{provider_name}:{sorted(kwargs.items())}"

        # For Azure, include credentials in cache key so provider is recreated when they change
        if provider_name == "azure_neural":
            import os
            azure_key = config.get("azure_speech_key") or os.environ.get("AZURE_SPEECH_KEY", "")
            azure_region = config.get("azure_speech_region") or os.environ.get("AZURE_SPEECH_REGION", "")
            cache_key += f":{azure_key[:8] if azure_key else ''}:{azure_region}"

        # Recreate provider if config changed or provider was cleared
        if self._tts_provider is None or not hasattr(self, '_tts_cache_key') or self._tts_cache_key != cache_key:
            self._tts_provider = get_tts_provider(provider_name, **kwargs)
            self._tts_cache_key = cache_key

        return self._tts_provider

    def _get_audio_capture(self):
        """Get or create audio capture."""
        if self._audio_capture is None:
            from episodic.voice.audio_capture import AudioCapture

            self._audio_capture = AudioCapture(
                target_sample_rate=16000,
                vad_aggressiveness=config.get("voice_vad_aggressiveness", 2),
                silence_threshold_ms=config.get("voice_silence_threshold_ms", 1000),
            )

        return self._audio_capture

    def _get_audio_playback(self):
        """Get or create audio playback."""
        if self._audio_playback is None:
            from episodic.voice.audio_playback import AudioPlayback

            self._audio_playback = AudioPlayback(
                on_start=self._on_playback_start,
                on_stop=self._on_playback_stop,
            )

        return self._audio_playback

    def _get_wake_word_detector(self):
        """Get or create Porcupine wake word detector."""
        from episodic.voice.wake_word import PorcupineWakeWordDetector

        keyword = config.get("voice_wake_word", "computer").lower()
        sensitivity = config.get("voice_wake_word_sensitivity", 0.5)

        # Check if settings changed or detector was cleared - recreate if so
        cache_key = f"{keyword}:{sensitivity}"
        if self._wake_word_detector is None or not hasattr(self, '_wake_word_cache_key') or self._wake_word_cache_key != cache_key:
            # Cleanup old detector
            if self._wake_word_detector is not None:
                self._wake_word_detector.cleanup()
                self._wake_word_detector = None

            self._wake_word_detector = PorcupineWakeWordDetector(
                keyword=keyword,
                sensitivity=sensitivity,
                on_wake_word=self._on_wake_word_detected,
            )
            self._wake_word_cache_key = cache_key

        return self._wake_word_detector

    def _start_idle_timer(self, timeout: Optional[float] = None):
        """Start timer to transition to IDLE state after inactivity."""
        self._cancel_idle_timer()

        if timeout is None:
            timeout = config.get("voice_idle_timeout", 15)
        if timeout > 0 and config.get("voice_wake_word_enabled", True):
            self._idle_timer = threading.Timer(timeout, self._enter_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()
            self._idle_timer_started_at = time.time()

    def _cancel_idle_timer(self):
        """Cancel pending idle timer."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        # Don't clear _idle_timer_started_at - we need it for resume

    def _resume_idle_timer(self):
        """Resume idle timer with remaining time (min 2 seconds)."""
        if self._idle_timer_started_at is None:
            self._start_idle_timer()
            return

        elapsed = time.time() - self._idle_timer_started_at
        full_timeout = config.get("voice_idle_timeout", 15)
        remaining = max(2.0, full_timeout - elapsed)  # At least 2 seconds
        self._start_idle_timer(remaining)

    def pause_idle_timer(self):
        """Pause the idle timer (call when processing LLM request)."""
        self._cancel_idle_timer()

    def resume_idle_timer(self):
        """Resume the idle timer (call after LLM response completes)."""
        # Don't start timer if TTS is still playing — _on_playback_stop
        # will start it when the last chunk finishes
        if self._state not in (VoiceState.OFF, VoiceState.IDLE, VoiceState.SPEAKING):
            self._start_idle_timer()

    def _enter_idle(self):
        """Transition to IDLE state (wake word listening only)."""
        import sys

        # If SPEAKING but playback is done, transition to LISTENING first
        if self._state == VoiceState.SPEAKING:
            if self._audio_playback and not self._audio_playback.is_playing:
                # Playback finished, unmute mic if needed
                if self._audio_capture and not config.get("voice_wake_word_interrupts_speech", False):
                    self._audio_capture.unmute()
                self._set_state(VoiceState.LISTENING)
            else:
                # Still speaking, try again later
                self._start_idle_timer()
                return

        # Transition from LISTENING to IDLE
        if self._state == VoiceState.LISTENING:
            wake_word = config.get("voice_wake_word", "computer")
            if config.get("voice_show_transcription", True):
                typer.secho(f"💤 Idle - say \"{wake_word}\" to wake", fg="yellow")
                sys.stdout.flush()
            from episodic.voice.audio_playback import play_sleep_chime
            play_sleep_chime()
            self._set_state(VoiceState.IDLE)
        elif self._state == VoiceState.PROCESSING:
            # Busy processing STT, try again later
            self._start_idle_timer()

