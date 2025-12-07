"""
Text-to-Speech (TTS) provider abstraction for Episodic voice mode.

Supports multiple providers:
- local_piper: Piper TTS (free, fast, runs locally)
- local_xtts: Coqui XTTS v2 (free, high quality, slower)
- openai_tts: OpenAI TTS API (~$0.015/min)
- elevenlabs: ElevenLabs API (~$0.20/1k chars, highest quality)
"""

import io
import os
import wave
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseTTSProvider(ABC):
    """Base class for TTS providers."""

    name: str = "base"

    @abstractmethod
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text to audio.

        Args:
            text: Text to speak

        Returns:
            Tuple of (audio_data as float32 array, sample_rate)
        """
        pass

    def cleanup(self) -> None:
        """Clean up any resources (e.g., loaded models)."""
        pass


class LocalPiperProvider(BaseTTSProvider):
    """
    Local text-to-speech using Piper.

    Free, very fast, runs locally. Lower quality than XTTS/cloud.
    Good for real-time interactive use.
    """

    name = "local_piper"

    # Common voice directories
    VOICE_DIRS = [
        os.path.expanduser("~/.local/share/piper-voices"),
        os.path.expanduser("~/piper-voices"),
        ".",
    ]

    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0):
        self.voice = voice
        self.speed = speed  # 1.0 = normal, >1.0 = faster, <1.0 = slower
        self._piper_voice = None

    def _find_voice_model(self) -> Optional[str]:
        """Find the voice model file."""
        for voice_dir in self.VOICE_DIRS:
            candidate = os.path.join(voice_dir, f"{self.voice}.onnx")
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_voice(self):
        """Lazy load the Piper voice."""
        if self._piper_voice is None:
            from piper import PiperVoice

            model_path = self._find_voice_model()
            if not model_path:
                raise RuntimeError(
                    f"Piper voice model '{self.voice}' not found.\n"
                    f"Download with:\n"
                    f"  mkdir -p ~/.local/share/piper-voices\n"
                    f"  cd ~/.local/share/piper-voices\n"
                    f"  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{self.voice}.onnx\n"
                    f"  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{self.voice}.onnx.json"
                )
            self._piper_voice = PiperVoice.load(model_path)
        return self._piper_voice

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using local Piper."""
        try:
            voice = self._load_voice()

            audio_floats = []
            sample_rate = None

            # Piper uses length_scale: <1.0 = faster, >1.0 = slower
            # We invert our speed so >1.0 = faster (more intuitive)
            length_scale = 1.0 / self.speed if self.speed > 0 else 1.0

            for chunk in voice.synthesize(text, length_scale=length_scale):
                audio_floats.append(chunk.audio_float_array)
                if sample_rate is None:
                    sample_rate = chunk.sample_rate

            if not audio_floats:
                return np.array([], dtype=np.float32), 22050

            audio = np.concatenate(audio_floats).astype(np.float32)
            return audio, sample_rate or 22050

        except ImportError:
            raise RuntimeError(
                "piper-tts not installed. Run: pip install piper-tts"
            )

    def cleanup(self) -> None:
        """Release the voice from memory."""
        self._piper_voice = None


class LocalXTTSProvider(BaseTTSProvider):
    """
    Local text-to-speech using Coqui XTTS v2.

    Free, high quality, runs locally. Slower than Piper.
    Model loads once (~18s), then generation is fast (~2.5s).
    """

    name = "local_xtts"

    # Available built-in speakers
    SPEAKERS = [
        "Claribel Dervla", "Daisy Studious", "Gracie Wise",
        "Tammie Ema", "Alison Dietlinde", "Ana Florence",
        "Annmarie Nele", "Asya Anara", "Brenda Stern",
        "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
        "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
        "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler",
        "Royston Min", "Viktor Eka", "Abrahan Mack",
        "Adde Michal", "Baldur Sansen", "Craig Gutsy",
        "Damien Black", "Gilberto Mathias", "Ilkin Urbano",
        "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim",
        "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios",
        "Nova Hogarth", "Maja Ruoho", "Uta Obando",
        "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
        "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
        "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa",
        "Alma María", "Rosemary Okafor", "Ige Behringer",
        "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro",
        "Aaron Dreschner", "Kumar Dahl", "Eugenio Matarac",
        "Ferran Sansen", "Xavier Hayasaka", "Luis Moray",
        "Marcos Rudaski",
    ]

    def __init__(self, speaker: str = "Claribel Dervla"):
        self.speaker = speaker
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy load the XTTS model."""
        if self._model is None:
            import torch
            from TTS.api import TTS

            # Detect best device
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self._device)
        return self._model

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using local XTTS."""
        try:
            model = self._load_model()

            wav = model.tts(text=text, speaker=self.speaker, language="en")
            audio = np.array(wav, dtype=np.float32)

            # XTTS outputs at 24kHz
            return audio, 24000

        except ImportError:
            raise RuntimeError(
                "coqui-tts not installed. Run: pip install coqui-tts"
            )

    def cleanup(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._device = None


class OpenAITTSProvider(BaseTTSProvider):
    """
    Cloud text-to-speech using OpenAI TTS API.

    Cost: $0.015/1K characters (tts-1) or $0.030/1K characters (tts-1-hd)
    Good quality, fast, no local compute needed.
    """

    name = "openai_tts"

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    # OpenAI TTS pricing per 1K characters
    COST_PER_1K_CHARS = {
        "tts-1": 0.015,
        "tts-1-hd": 0.030,
    }

    def __init__(self, voice: str = "alloy", model: str = "tts-1", speed: float = 1.0):
        self.voice = voice
        self.model = model  # tts-1 (fast) or tts-1-hd (higher quality)
        self.speed = max(0.25, min(4.0, speed))  # OpenAI supports 0.25 to 4.0
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using OpenAI TTS API."""
        try:
            client = self._get_client()

            response = client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=self.speed,
                response_format="wav"
            )

            audio_bytes = response.content

            # Track cost
            char_count = len(text)
            cost_per_1k = self.COST_PER_1K_CHARS.get(self.model, 0.015)
            cost_usd = (char_count / 1000.0) * cost_per_1k
            from episodic.llm_manager import llm_manager
            llm_manager.record_voice_tts(char_count, cost_usd)

            # Parse WAV to get audio data
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                sample_rate = wav.getframerate()

            audio_array = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            return audio_float, sample_rate

        except ImportError:
            raise RuntimeError(
                "openai not installed. Run: pip install openai"
            )

    def cleanup(self) -> None:
        """Nothing to clean up for cloud provider."""
        self._client = None


class ElevenLabsProvider(BaseTTSProvider):
    """
    Cloud text-to-speech using ElevenLabs API.

    Cost: ~$0.30/1000 characters (expensive but highest quality)
    Best quality, most natural sounding.
    """

    name = "elevenlabs"

    # ElevenLabs pricing: $0.30 per 1K characters (Creator plan)
    COST_PER_1K_CHARS = 0.30

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):  # Rachel
        self.voice_id = voice_id
        self._client = None

    def _get_client(self):
        """Lazy load ElevenLabs client."""
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs()
        return self._client

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using ElevenLabs API."""
        try:
            client = self._get_client()

            audio_generator = client.generate(
                text=text,
                voice=self.voice_id,
                model="eleven_monolingual_v1"
            )

            # Collect all audio chunks
            audio_bytes = b"".join(audio_generator)

            # Track cost
            char_count = len(text)
            cost_usd = (char_count / 1000.0) * self.COST_PER_1K_CHARS
            from episodic.llm_manager import llm_manager
            llm_manager.record_voice_tts(char_count, cost_usd)

            # ElevenLabs returns MP3 by default, convert to numpy
            # Using pydub for MP3 decoding
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

            # Convert to numpy float32
            samples = np.array(audio_segment.get_array_of_samples())
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)  # Mono mix

            audio_float = samples.astype(np.float32) / 32768.0
            return audio_float, audio_segment.frame_rate

        except ImportError as e:
            if "elevenlabs" in str(e):
                raise RuntimeError(
                    "elevenlabs not installed. Run: pip install elevenlabs"
                )
            raise RuntimeError(
                "pydub not installed (needed for MP3 decoding). Run: pip install pydub"
            )


# Provider registry
_TTS_PROVIDERS = {
    "local_piper": LocalPiperProvider,
    "local_xtts": LocalXTTSProvider,
    "openai_tts": OpenAITTSProvider,
    "elevenlabs": ElevenLabsProvider,
}

# Singleton instances (for model caching)
_provider_instances: dict[str, BaseTTSProvider] = {}


def get_tts_provider(
    provider_name: str = "local_piper",
    **kwargs
) -> BaseTTSProvider:
    """
    Get a TTS provider instance.

    Args:
        provider_name: One of 'local_piper', 'local_xtts', 'openai_tts', 'elevenlabs'
        **kwargs: Provider-specific options (e.g., voice for openai_tts)

    Returns:
        TTS provider instance (cached for model reuse)
    """
    # Create cache key from provider name and kwargs
    cache_key = f"{provider_name}:{sorted(kwargs.items())}"

    if cache_key not in _provider_instances:
        if provider_name not in _TTS_PROVIDERS:
            raise ValueError(
                f"Unknown TTS provider: {provider_name}. "
                f"Available: {list(_TTS_PROVIDERS.keys())}"
            )
        _provider_instances[cache_key] = _TTS_PROVIDERS[provider_name](**kwargs)

    return _provider_instances[cache_key]


def cleanup_tts_providers() -> None:
    """Clean up all cached TTS provider instances."""
    for provider in _provider_instances.values():
        provider.cleanup()
    _provider_instances.clear()
