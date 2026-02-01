"""
Text-to-Speech (TTS) provider abstraction for Episodic voice mode.

Supports multiple providers:
- local_piper: Piper TTS (free, fast, runs locally)
- local_xtts: Coqui XTTS v2 (free, high quality, slower)
- openai_tts: OpenAI TTS API (cloud)
- elevenlabs: ElevenLabs API (cloud, highest quality)
- azure_neural: Azure Neural TTS (cloud, DragonHD voices)

Pricing is loaded from voice_pricing.json for cost tracking.
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
            from piper.config import SynthesisConfig
            voice = self._load_voice()

            audio_floats = []
            sample_rate = None

            # Piper uses length_scale: <1.0 = faster, >1.0 = slower
            # We invert our speed so >1.0 = faster (more intuitive)
            length_scale = 1.0 / self.speed if self.speed > 0 else 1.0
            syn_config = SynthesisConfig(length_scale=length_scale)

            for chunk in voice.synthesize(text, syn_config=syn_config):
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

    Good quality, fast, no local compute needed.
    Pricing loaded from voice_pricing.json.
    """

    name = "openai_tts"

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

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

            # Track cost (pricing loaded from voice_pricing.json)
            char_count = len(text)
            from episodic.voice_pricing import get_tts_cost_per_1k_chars
            cost_per_1k = get_tts_cost_per_1k_chars("openai_tts", model=self.model)
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

    Highest quality, most natural sounding.
    Pricing loaded from voice_pricing.json.
    """

    name = "elevenlabs"

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

            # Track cost (pricing loaded from voice_pricing.json)
            char_count = len(text)
            from episodic.voice_pricing import get_tts_cost_per_1k_chars
            cost_per_1k = get_tts_cost_per_1k_chars("elevenlabs")
            cost_usd = (char_count / 1000.0) * cost_per_1k
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


class AzureNeuralProvider(BaseTTSProvider):
    """
    Cloud text-to-speech using Azure Neural TTS (including HD voices).

    High quality neural voices with DragonHD option for premium quality.
    Supports 150+ languages with multiple voices per locale.
    """

    name = "azure_neural"

    # Popular HD voices (DragonHD = highest quality)
    HD_VOICES = [
        "en-US-Ava:DragonHDLatestNeural",
        "en-US-Andrew:DragonHDLatestNeural",
        "en-US-Emma:DragonHDLatestNeural",
        "en-US-Brian:DragonHDLatestNeural",
        "en-GB-Sonia:DragonHDLatestNeural",
        "en-GB-Ryan:DragonHDLatestNeural",
    ]

    # Standard neural voices (still high quality, lower cost)
    STANDARD_VOICES = [
        "en-US-JennyNeural",
        "en-US-GuyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
    ]

    def __init__(
        self,
        voice: str = "en-US-JennyNeural",
        speech_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.voice = voice
        self.speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION", "eastus")
        self._synthesizer = None

        if not self.speech_key:
            raise ValueError(
                "Azure Speech key required. Set AZURE_SPEECH_KEY env var or pass speech_key parameter."
            )

    def _get_synthesizer(self):
        """Lazy load the Azure Speech synthesizer."""
        if self._synthesizer is None:
            import azure.cognitiveservices.speech as speechsdk

            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.region,
            )
            speech_config.speech_synthesis_voice_name = self.voice
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
            )

            # Output to memory stream instead of speaker
            self._synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None,  # No audio output, we'll capture the data
            )
        return self._synthesizer

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Azure Neural TTS."""
        try:
            import azure.cognitiveservices.speech as speechsdk

            synthesizer = self._get_synthesizer()
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get audio data from result
                audio_data = result.audio_data

                # Skip WAV header (44 bytes) and convert to numpy
                audio_bytes = audio_data[44:]
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Track cost
                char_count = len(text)
                from episodic.voice_pricing import get_tts_cost_per_1k_chars
                cost_per_1k = get_tts_cost_per_1k_chars("azure_neural")
                cost_usd = (char_count / 1000.0) * cost_per_1k
                from episodic.llm_manager import llm_manager
                llm_manager.record_voice_tts(char_count, cost_usd)

                return audio_float, 24000

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                raise RuntimeError(
                    f"Azure TTS canceled: {cancellation.reason}. "
                    f"Error: {cancellation.error_details}"
                )
            else:
                raise RuntimeError(f"Azure TTS failed with reason: {result.reason}")

        except ImportError:
            raise RuntimeError(
                "azure-cognitiveservices-speech not installed. "
                "Run: pip install azure-cognitiveservices-speech"
            )

    def cleanup(self) -> None:
        """Release synthesizer resources."""
        self._synthesizer = None


# Provider registry
_TTS_PROVIDERS = {
    "local_piper": LocalPiperProvider,
    "local_xtts": LocalXTTSProvider,
    "openai_tts": OpenAITTSProvider,
    "elevenlabs": ElevenLabsProvider,
    "azure_neural": AzureNeuralProvider,
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
        provider_name: One of 'local_piper', 'local_xtts', 'openai_tts',
                       'elevenlabs', 'azure_neural'
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
