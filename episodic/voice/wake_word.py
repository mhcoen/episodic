"""
Wake word detection for Episodic voice mode.

Uses Porcupine for efficient, low-latency wake word detection.
"""

from typing import Callable, Optional
import numpy as np


class PorcupineWakeWordDetector:
    """
    Wake word detector using Picovoice Porcupine.

    Processes audio frames in real-time with minimal CPU overhead.
    Built-in keywords include: "computer", "jarvis", "alexa", etc.

    Requires a free access key from https://console.picovoice.ai/
    Set via PICOVOICE_ACCESS_KEY environment variable or porcupine_access_key config.
    """

    # Porcupine built-in keywords (free tier)
    BUILTIN_KEYWORDS = [
        "alexa", "americano", "blueberry", "bumblebee", "computer",
        "grapefruit", "grasshopper", "hey google", "hey siri", "jarvis",
        "ok google", "picovoice", "porcupine", "terminator"
    ]

    def __init__(
        self,
        keyword: str = "computer",
        sensitivity: float = 0.5,
        access_key: Optional[str] = None,
        on_wake_word: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize Porcupine wake word detector.

        Args:
            keyword: Wake word to detect (must be a built-in keyword)
            sensitivity: Detection sensitivity 0.0-1.0 (higher = more sensitive)
            access_key: Picovoice access key (or set PICOVOICE_ACCESS_KEY env var)
            on_wake_word: Callback when wake word is detected
        """
        self.keyword = keyword.lower()
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.access_key = access_key
        self.on_wake_word = on_wake_word

        self._porcupine = None
        self._frame_length = 512  # Porcupine requires 512 samples at 16kHz
        self._sample_rate = 16000
        self._audio_buffer = np.array([], dtype=np.int16)

    def _get_access_key(self) -> str:
        """Get Porcupine access key from config or environment."""
        import os
        from episodic.config import config

        # Priority: explicit > config > environment
        if self.access_key:
            return self.access_key

        key = config.get("porcupine_access_key", "")
        if key:
            return key

        key = os.environ.get("PICOVOICE_ACCESS_KEY", "")
        if key:
            return key

        raise ValueError(
            "Porcupine access key required. Get a free key at https://console.picovoice.ai/\n"
            "Set via: /set porcupine_access_key YOUR_KEY\n"
            "Or set PICOVOICE_ACCESS_KEY environment variable"
        )

    def _init_porcupine(self):
        """Initialize Porcupine engine (lazy load)."""
        if self._porcupine is None:
            import pvporcupine

            if self.keyword not in self.BUILTIN_KEYWORDS:
                raise ValueError(
                    f"Unknown keyword '{self.keyword}'. "
                    f"Available: {', '.join(self.BUILTIN_KEYWORDS)}"
                )

            access_key = self._get_access_key()

            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[self.keyword],
                sensitivities=[self.sensitivity],
            )
            self._frame_length = self._porcupine.frame_length
            self._sample_rate = self._porcupine.sample_rate

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Process audio samples and check for wake word.

        Args:
            audio: Audio samples as int16 numpy array
            sample_rate: Sample rate of audio (will resample if not 16kHz)

        Returns:
            True if wake word was detected, False otherwise
        """
        self._init_porcupine()

        # Resample if needed
        if sample_rate != self._sample_rate:
            from episodic.voice.audio_capture import _resample_audio
            audio = _resample_audio(audio, sample_rate, self._sample_rate)

        # Add to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])

        # Process complete frames
        detected = False
        while len(self._audio_buffer) >= self._frame_length:
            frame = self._audio_buffer[:self._frame_length]
            self._audio_buffer = self._audio_buffer[self._frame_length:]

            keyword_index = self._porcupine.process(frame.tolist())
            if keyword_index >= 0:
                detected = True
                if self.on_wake_word:
                    self.on_wake_word()

        return detected

    def reset(self):
        """Clear audio buffer."""
        self._audio_buffer = np.array([], dtype=np.int16)

    def cleanup(self):
        """Release Porcupine resources."""
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        self._audio_buffer = np.array([], dtype=np.int16)

    @property
    def frame_length(self) -> int:
        """Get required frame length in samples."""
        self._init_porcupine()
        return self._frame_length

    @property
    def sample_rate(self) -> int:
        """Get required sample rate."""
        return self._sample_rate

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
