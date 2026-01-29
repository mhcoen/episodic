"""
Speech-to-Text (STT) provider abstraction for Episodic voice mode.

Supports multiple providers:
- local_whisper: faster-whisper (free, runs locally)
- openai_whisper: OpenAI Whisper API (cloud)
- deepgram: Deepgram API (cloud, real-time streaming)

Pricing is loaded from voice_pricing.json for cost tracking.
"""

import io
import os
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseSTTProvider(ABC):
    """Base class for STT providers."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio samples as numpy array (int16)
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text or None on failure
        """
        pass

    def cleanup(self) -> None:
        """Clean up any resources (e.g., loaded models)."""
        pass


def _audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())
    return wav_buffer.getvalue()


def _save_wav_temp(audio_data: np.ndarray, sample_rate: int) -> str:
    """Save audio to temp WAV file, return path."""
    wav_bytes = _audio_to_wav_bytes(audio_data, sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        return f.name


def _is_hallucination(text: str) -> bool:
    """Check if transcribed text is likely a Whisper hallucination."""
    if not text:
        return True

    text_lower = text.lower().strip()
    # Also create a version without trailing punctuation for matching
    text_no_punct = text_lower.rstrip('.,!?;:')

    # Common exact hallucinations (checked with and without trailing punctuation)
    hallucinations = {
        "bye", "bye bye", "bye-bye", "goodbye",
        "thank you", "thanks", "thank you very much",
        "you", "the", "a", "i", "it", "so", "and", "um", "uh",
        "...", ".", "", " ",
        "thanks for watching", "thank you for watching",
        "thanks for listening", "thank you for listening",
        "please subscribe", "subscribe", "like and subscribe",
        "don't forget to subscribe", "hit the subscribe button",
        "see you next time", "see you", "see you later", "see you soon",
        "see you in the next video", "see you in the next one",
        "take care", "have a nice day", "have a good day",
        "i'll see you in the next video", "until next time",
        "peace", "peace out", "later", "cheers",
    }
    if text_lower in hallucinations or text_no_punct in hallucinations:
        return True

    # Patterns that indicate hallucinations (URLs, promotional content)
    hallucination_patterns = [
        "www.", "http", ".com", ".org", ".net",
        "subscribe", "channel", "video",
        "engvid", "learn english",
        "copyright", "all rights reserved",
        "music", "♪", "♫",
        "[music]", "[applause]", "[laughter]",
        "transcribed by", "subtitles by",
    ]
    for pattern in hallucination_patterns:
        if pattern in text_lower:
            return True

    # Very short single words are often hallucinations
    if len(text_lower) <= 2:
        return True

    # Repeated characters or words
    if len(set(text_lower.replace(" ", ""))) <= 2:
        return True

    return False


class LocalWhisperProvider(BaseSTTProvider):
    """
    Local speech-to-text using faster-whisper.

    Free, runs entirely locally. Good accuracy, especially on Apple Silicon.
    Model sizes: tiny, base, small, medium, large-v2, large-v3
    """

    name = "local_whisper"

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            # Auto-detect best device/compute type
            self._model = WhisperModel(
                self.model_size,
                device="auto",
                compute_type="auto"
            )
        return self._model

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe using local faster-whisper."""
        try:
            # Check for minimum audio length (at least 0.1 seconds)
            min_samples = int(sample_rate * 0.1)
            if len(audio_data) < min_samples:
                return None

            # Check for silence (avoid hallucinations on quiet audio)
            peak = np.max(np.abs(audio_data))
            if peak < 500:  # Very quiet, likely silence
                return None

            model = self._load_model()
            temp_path = _save_wav_temp(audio_data, sample_rate)

            try:
                segments, info = model.transcribe(
                    temp_path,
                    beam_size=5,
                    language="en",  # Force English
                    condition_on_previous_text=False,  # Reduce hallucinations
                    no_speech_threshold=0.6,  # Higher threshold to filter non-speech
                )
                text = " ".join([segment.text for segment in segments]).strip()

                # Filter hallucinations
                if _is_hallucination(text):
                    return None

                return text if text else None
            finally:
                os.unlink(temp_path)

        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
        except Exception as e:
            print(f"Local Whisper error: {e}")
            return None

    def cleanup(self) -> None:
        """Release the model from memory."""
        self._model = None


class OpenAIWhisperProvider(BaseSTTProvider):
    """
    Cloud speech-to-text using OpenAI Whisper API.

    Excellent accuracy, no local compute needed.
    Pricing loaded from voice_pricing.json.
    """

    name = "openai_whisper"

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe using OpenAI Whisper API."""
        try:
            # Check for minimum audio length (at least 0.1 seconds)
            min_samples = int(sample_rate * 0.1)
            if len(audio_data) < min_samples:
                return None

            # Check for silence (avoid hallucinations on quiet audio)
            peak = np.max(np.abs(audio_data))
            if peak < 500:  # Very quiet, likely silence
                return None

            # Calculate audio duration for cost tracking
            duration_seconds = len(audio_data) / sample_rate

            client = self._get_client()
            temp_path = _save_wav_temp(audio_data, sample_rate)

            try:
                with open(temp_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en",  # Force English to prevent wrong language detection
                    )
                text = transcript.text.strip()

                # Track cost (pricing loaded from voice_pricing.json)
                from episodic.voice_pricing import get_stt_cost_per_minute
                cost_per_minute = get_stt_cost_per_minute("openai_whisper")
                cost_usd = (duration_seconds / 60.0) * cost_per_minute
                from episodic.llm_manager import llm_manager
                llm_manager.record_voice_stt(duration_seconds, cost_usd)

                # Filter hallucinations
                if _is_hallucination(text):
                    return None

                return text if text else None
            finally:
                os.unlink(temp_path)

        except ImportError:
            raise RuntimeError(
                "openai not installed. Run: pip install openai"
            )
        except Exception as e:
            print(f"OpenAI Whisper error: {e}")
            return None


class DeepgramProvider(BaseSTTProvider):
    """
    Cloud speech-to-text using Deepgram API.

    Real-time streaming support, very fast.
    Pricing loaded from voice_pricing.json.
    """

    name = "deepgram"

    def __init__(self, model: str = "nova-2"):
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load Deepgram client."""
        if self._client is None:
            from deepgram import DeepgramClient
            self._client = DeepgramClient()
        return self._client

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe using Deepgram API."""
        try:
            from deepgram import PrerecordedOptions

            # Calculate audio duration for cost tracking
            duration_seconds = len(audio_data) / sample_rate

            client = self._get_client()
            wav_bytes = _audio_to_wav_bytes(audio_data, sample_rate)

            options = PrerecordedOptions(
                model=self.model,
                smart_format=True,
            )

            response = client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": wav_bytes, "mimetype": "audio/wav"},
                options
            )

            text = response.results.channels[0].alternatives[0].transcript

            # Track cost (pricing loaded from voice_pricing.json)
            from episodic.voice_pricing import get_stt_cost_per_minute
            cost_per_minute = get_stt_cost_per_minute("deepgram")
            cost_usd = (duration_seconds / 60.0) * cost_per_minute
            from episodic.llm_manager import llm_manager
            llm_manager.record_voice_stt(duration_seconds, cost_usd)

            return text.strip() if text else None

        except ImportError:
            raise RuntimeError(
                "deepgram-sdk not installed. Run: pip install deepgram-sdk"
            )
        except Exception as e:
            print(f"Deepgram error: {e}")
            return None


# Provider registry
_STT_PROVIDERS = {
    "local_whisper": LocalWhisperProvider,
    "openai_whisper": OpenAIWhisperProvider,
    "deepgram": DeepgramProvider,
}

# Singleton instances (for model caching)
_provider_instances: dict[str, BaseSTTProvider] = {}


def get_stt_provider(
    provider_name: str = "local_whisper",
    **kwargs
) -> BaseSTTProvider:
    """
    Get an STT provider instance.

    Args:
        provider_name: One of 'local_whisper', 'openai_whisper', 'deepgram'
        **kwargs: Provider-specific options (e.g., model_size for local_whisper)

    Returns:
        STT provider instance (cached for model reuse)
    """
    # Create cache key from provider name and kwargs
    cache_key = f"{provider_name}:{sorted(kwargs.items())}"

    if cache_key not in _provider_instances:
        if provider_name not in _STT_PROVIDERS:
            raise ValueError(
                f"Unknown STT provider: {provider_name}. "
                f"Available: {list(_STT_PROVIDERS.keys())}"
            )
        _provider_instances[cache_key] = _STT_PROVIDERS[provider_name](**kwargs)

    return _provider_instances[cache_key]


def cleanup_stt_providers() -> None:
    """Clean up all cached STT provider instances."""
    for provider in _provider_instances.values():
        provider.cleanup()
    _provider_instances.clear()
