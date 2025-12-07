"""
Audio capture with Voice Activity Detection (VAD) for Episodic voice mode.

Handles microphone input and detects when speech starts/stops.
"""

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np


def _get_device_sample_rate() -> int:
    """Get the default input device's native sample rate."""
    import sounddevice as sd
    try:
        default_input = sd.default.device[0]
        input_info = sd.query_devices(default_input)
        return int(input_info['default_samplerate'])
    except Exception:
        return 48000  # Common fallback


def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio

    try:
        from scipy import signal
        num_samples = int(len(audio) * to_rate / from_rate)
        return signal.resample(audio, num_samples).astype(audio.dtype)
    except ImportError:
        # Fallback: simple linear interpolation (less accurate but no scipy needed)
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * to_rate / from_rate))
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float32)).astype(audio.dtype)


class AudioCapture:
    """
    Captures audio from microphone with voice activity detection.

    Uses webrtcvad for speech detection and sounddevice for audio input.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        vad_aggressiveness: int = 1,
        silence_threshold_ms: int = 1000,
        pre_speech_buffer_ms: int = 500,
    ):
        """
        Initialize audio capture.

        Args:
            target_sample_rate: Target sample rate for STT (16000 recommended)
            vad_aggressiveness: VAD sensitivity 0-3 (higher = more aggressive filtering)
            silence_threshold_ms: Silence duration to end speech detection
            pre_speech_buffer_ms: Audio to keep before speech detected
        """
        self.target_sample_rate = target_sample_rate
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_threshold_ms = silence_threshold_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms

        # Device sample rate (set on start)
        self._device_sample_rate: int = 16000
        # VAD always works at 16kHz
        self._vad_sample_rate: int = 16000

        self._vad = None
        self._stream = None
        self._is_recording = False
        self._is_muted = False

        # Audio buffers (stored at device sample rate, resampled on output)
        self._pre_speech_buffer: deque = deque()
        self._speech_buffer: list = []

        # State tracking
        self._speech_started = False
        self._silence_start: Optional[float] = None

        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable[[np.ndarray], None]] = None

        # Thread safety
        self._lock = threading.Lock()

    def _init_vad(self):
        """Initialize VAD if not already done."""
        if self._vad is None:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
                import webrtcvad
            self._vad = webrtcvad.Vad(self.vad_aggressiveness)

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}")

        if self._is_muted:
            return

        # Convert to int16
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

        with self._lock:
            self._process_audio(audio_int16)

    def _process_audio(self, audio_int16: np.ndarray):
        """Process audio chunk with VAD."""
        # VAD requires 10, 20, or 30ms frames at 8000, 16000, or 32000 Hz
        # We resample to 16kHz for VAD processing
        frame_duration_ms = 30
        vad_frame_size = int(self._vad_sample_rate * frame_duration_ms / 1000)  # 480 samples at 16kHz
        device_frame_size = int(self._device_sample_rate * frame_duration_ms / 1000)

        # Resample chunk for VAD if needed
        if self._device_sample_rate != self._vad_sample_rate:
            audio_for_vad = _resample_audio(audio_int16, self._device_sample_rate, self._vad_sample_rate)
        else:
            audio_for_vad = audio_int16

        # Process in VAD-compatible frames
        # We use device-rate frames for storage but VAD-rate frames for detection
        num_frames = len(audio_for_vad) // vad_frame_size
        for i in range(num_frames):
            vad_frame = audio_for_vad[i * vad_frame_size:(i + 1) * vad_frame_size]
            device_frame = audio_int16[i * device_frame_size:(i + 1) * device_frame_size]

            frame_bytes = vad_frame.tobytes()

            try:
                is_speech = self._vad.is_speech(frame_bytes, self._vad_sample_rate)
            except Exception:
                # VAD can fail on certain audio, treat as non-speech
                is_speech = False

            if is_speech:
                if not self._speech_started:
                    # Speech just started
                    self._speech_started = True
                    self._silence_start = None

                    # Include pre-speech buffer
                    self._speech_buffer = list(self._pre_speech_buffer)

                    if self._on_speech_start:
                        self._on_speech_start()

                self._speech_buffer.append(device_frame)

            else:
                if self._speech_started:
                    # In speech but silence detected
                    self._speech_buffer.append(device_frame)

                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif (time.time() - self._silence_start) * 1000 > self.silence_threshold_ms:
                        # Silence threshold exceeded, speech ended
                        self._finalize_speech()
                else:
                    # Not in speech, maintain pre-speech buffer
                    self._pre_speech_buffer.append(device_frame)
                    # Limit buffer size
                    max_frames = int(self.pre_speech_buffer_ms / frame_duration_ms)
                    while len(self._pre_speech_buffer) > max_frames:
                        self._pre_speech_buffer.popleft()

    def _finalize_speech(self):
        """Finalize captured speech and trigger callback."""
        if self._speech_buffer:
            # Concatenate all frames (at device sample rate)
            audio = np.concatenate(self._speech_buffer)

            # Resample to target sample rate for STT
            if self._device_sample_rate != self.target_sample_rate:
                audio = _resample_audio(audio, self._device_sample_rate, self.target_sample_rate)

            # Reset state
            self._speech_started = False
            self._silence_start = None
            self._speech_buffer = []
            self._pre_speech_buffer.clear()

            # Trigger callback
            if self._on_speech_end:
                self._on_speech_end(audio)

    def start(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Start capturing audio.

        Args:
            on_speech_start: Called when speech is detected
            on_speech_end: Called when speech ends, receives audio as int16 numpy array
        """
        import sounddevice as sd

        self._init_vad()

        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

        self._is_recording = True
        self._speech_started = False
        self._silence_start = None
        self._speech_buffer = []
        self._pre_speech_buffer.clear()

        # Get device's native sample rate to avoid format errors
        self._device_sample_rate = _get_device_sample_rate()

        # Start audio stream at device's native rate
        self._stream = sd.InputStream(
            samplerate=self._device_sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=int(self._device_sample_rate * 0.03),  # 30ms blocks
        )
        self._stream.start()

    def stop(self) -> Optional[np.ndarray]:
        """
        Stop capturing audio.

        Returns:
            Any remaining speech audio (at target sample rate), or None
        """
        self._is_recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Return any remaining speech
        with self._lock:
            if self._speech_buffer:
                audio = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                # Resample to target sample rate
                if self._device_sample_rate != self.target_sample_rate:
                    audio = _resample_audio(audio, self._device_sample_rate, self.target_sample_rate)
                return audio

        return None

    def mute(self):
        """Temporarily mute the microphone (for TTS playback)."""
        self._is_muted = True

    def unmute(self):
        """Resume listening after mute."""
        self._is_muted = False
        # Clear any audio that accumulated during mute
        with self._lock:
            self._pre_speech_buffer.clear()

    @property
    def is_muted(self) -> bool:
        """Check if microphone is muted."""
        return self._is_muted

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording


def record_until_silence(
    target_sample_rate: int = 16000,
    silence_threshold_ms: int = 1000,
    vad_aggressiveness: int = 2,
    max_duration_s: float = 30.0,
    on_start: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    Record audio until silence is detected.

    Convenience function for simple recording without callbacks.

    Args:
        target_sample_rate: Target sample rate for output (16000 recommended for STT)
        silence_threshold_ms: Silence duration to stop recording
        vad_aggressiveness: VAD sensitivity 0-3
        max_duration_s: Maximum recording duration
        on_start: Called when speech is first detected

    Returns:
        Recorded audio as int16 numpy array at target sample rate, or None if no speech
    """
    result = None
    done = threading.Event()

    def on_speech_end(audio):
        nonlocal result
        result = audio
        done.set()

    capture = AudioCapture(
        target_sample_rate=target_sample_rate,
        silence_threshold_ms=silence_threshold_ms,
        vad_aggressiveness=vad_aggressiveness,
    )

    capture.start(on_speech_start=on_start, on_speech_end=on_speech_end)

    # Wait for speech or timeout
    done.wait(timeout=max_duration_s)

    # Stop and get any remaining audio
    remaining = capture.stop()
    if remaining is not None and result is None:
        result = remaining

    return result
