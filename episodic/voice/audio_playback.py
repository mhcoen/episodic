"""
Audio playback queue for Episodic voice mode.

Handles non-blocking audio playback with queuing for streaming TTS.
Uses a continuous output stream to avoid gaps between audio chunks.
"""

import collections
import threading
from typing import Callable, Optional

import numpy as np


def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio

    try:
        from scipy import signal
        num_samples = int(len(audio) * to_rate / from_rate)
        return signal.resample(audio, num_samples).astype(np.float32)
    except ImportError:
        # Fallback: simple linear interpolation
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * to_rate / from_rate))
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class AudioPlayback:
    """
    Non-blocking audio playback with queuing.

    Uses a continuous output stream to avoid gaps between audio chunks.
    Supports streaming TTS by queueing audio samples for sequential playback.
    """

    # Standard output sample rate (resample all audio to this)
    OUTPUT_SAMPLE_RATE = 22050

    # Fade out duration in samples (~10ms) to prevent clicks at end
    FADE_OUT_SAMPLES = 220

    def __init__(self, on_start: Optional[Callable] = None, on_stop: Optional[Callable] = None):
        """
        Initialize audio playback.

        Args:
            on_start: Called when playback starts (first samples enqueued)
            on_stop: Called when playback queue is empty and silence begins
        """
        # Sample buffer (deque of float32 samples)
        self._sample_buffer: collections.deque = collections.deque()
        self._buffer_lock = threading.Lock()

        # Output stream
        self._stream = None
        self._stream_active = False

        # State tracking
        self._is_playing = False
        self._samples_played = 0
        self._silence_samples = 0

        # Callbacks
        self._on_start = on_start
        self._on_stop = on_stop

        # Wait event for synchronization
        self._empty_event = threading.Event()
        self._empty_event.set()  # Initially empty

    def _audio_callback(self, outdata, frames, time_info, status):
        """Fill output buffer from sample queue."""
        if status:
            print(f"Playback status: {status}")

        with self._buffer_lock:
            available = len(self._sample_buffer)

            if available >= frames:
                # Enough samples - fill the buffer
                for i in range(frames):
                    outdata[i, 0] = self._sample_buffer.popleft()
                self._samples_played += frames
                self._silence_samples = 0

                # Trigger on_start callback when we start playing after silence
                if not self._is_playing:
                    self._is_playing = True
                    self._empty_event.clear()
                    if self._on_start:
                        # Call in separate thread to avoid blocking audio callback
                        threading.Thread(target=self._on_start, daemon=True).start()

            elif available > 0:
                # Partial buffer - use what we have, apply fade-out, fill rest with silence
                for i in range(available):
                    sample = self._sample_buffer.popleft()
                    # Apply fade-out to last samples to prevent click
                    if i >= available - self.FADE_OUT_SAMPLES:
                        fade_pos = available - i
                        fade_factor = fade_pos / self.FADE_OUT_SAMPLES
                        sample *= fade_factor
                    outdata[i, 0] = sample
                outdata[available:, 0] = 0
                self._samples_played += available
                self._silence_samples += frames - available

            else:
                # No samples - output silence
                outdata[:, 0] = 0
                self._silence_samples += frames

                # After ~100ms of silence, trigger on_stop
                if self._is_playing and self._silence_samples > self.OUTPUT_SAMPLE_RATE * 0.1:
                    self._is_playing = False
                    self._empty_event.set()
                    if self._on_stop:
                        threading.Thread(target=self._on_stop, daemon=True).start()

    def start(self):
        """Start the playback stream."""
        if self._stream is not None and self._stream_active:
            return

        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=1024,  # ~46ms at 22050 Hz
        )
        self._stream.start()
        self._stream_active = True
        self._silence_samples = 0
        self._is_playing = False

    def stop(self):
        """Stop playback and clear queue."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._stream_active = False

        with self._buffer_lock:
            self._sample_buffer.clear()

        self._is_playing = False
        self._empty_event.set()

    def enqueue(self, audio: np.ndarray, sample_rate: int):
        """
        Add audio to the playback queue.

        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Sample rate of the audio in Hz
        """
        # Resample to match output stream if needed
        if sample_rate != self.OUTPUT_SAMPLE_RATE:
            audio = _resample_audio(audio, sample_rate, self.OUTPUT_SAMPLE_RATE)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Add to buffer
        with self._buffer_lock:
            self._sample_buffer.extend(audio)
            self._empty_event.clear()

    def _apply_fade_out_to_buffer(self):
        """Apply fade-out to the end of the current buffer to prevent clicks."""
        with self._buffer_lock:
            buf_len = len(self._sample_buffer)
            if buf_len < self.FADE_OUT_SAMPLES:
                return

            # Convert last N samples to list, apply fade, put back
            fade_samples = []
            for _ in range(self.FADE_OUT_SAMPLES):
                fade_samples.append(self._sample_buffer.pop())
            fade_samples.reverse()

            for i, sample in enumerate(fade_samples):
                fade_factor = 1.0 - (i / self.FADE_OUT_SAMPLES)
                self._sample_buffer.append(sample * fade_factor)

    def clear_queue(self):
        """
        Clear the playback queue with fade-out to prevent clicks.

        Used for speech interruption (e.g., user presses a key to stop TTS).
        """
        self._apply_fade_out_to_buffer()
        with self._buffer_lock:
            # Keep only the fade-out portion, discard the rest
            fade_samples = min(len(self._sample_buffer), self.FADE_OUT_SAMPLES * 2)
            # Convert to list, keep only fade portion, put back
            remaining = []
            for _ in range(fade_samples):
                if self._sample_buffer:
                    remaining.append(self._sample_buffer.popleft())
            self._sample_buffer.clear()
            self._sample_buffer.extend(remaining)

    def play_immediate(self, audio: np.ndarray, sample_rate: int):
        """
        Play audio immediately, clearing the queue.

        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Sample rate in Hz
        """
        # Clear existing samples
        with self._buffer_lock:
            self._sample_buffer.clear()

        # Enqueue the new audio
        self.enqueue(audio, sample_rate)

    def finish(self):
        """Signal that no more audio will be added and apply fade-out."""
        self._apply_fade_out_to_buffer()

    def wait(self):
        """Wait for all queued audio to finish playing."""
        # Wait for empty event (triggered after ~100ms of silence)
        self._empty_event.wait()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing

    @property
    def queue_size(self) -> int:
        """Get number of samples in the queue."""
        with self._buffer_lock:
            return len(self._sample_buffer)

    @property
    def queue_duration(self) -> float:
        """Get duration of queued audio in seconds."""
        return self.queue_size / self.OUTPUT_SAMPLE_RATE


def play_audio(audio: np.ndarray, sample_rate: int):
    """
    Simple blocking audio playback.

    Args:
        audio: Audio data as float32 numpy array
        sample_rate: Sample rate in Hz
    """
    import sounddevice as sd
    sd.play(audio, sample_rate)
    sd.wait()


def play_beep(frequency: float = 440.0, duration: float = 0.1, sample_rate: int = 44100):
    """
    Play a simple beep sound.

    Args:
        frequency: Beep frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    import sounddevice as sd

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate sine wave with fade in/out
    wave = np.sin(2 * np.pi * frequency * t)

    # Apply envelope (fade in/out)
    fade_samples = int(sample_rate * 0.01)  # 10ms fade
    envelope = np.ones_like(wave)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    wave *= envelope

    audio = (wave * 0.3).astype(np.float32)  # Reduce volume
    sd.play(audio, sample_rate)
    sd.wait()


def play_chime(sample_rate: int = 44100):
    """Play a pleasant two-tone chime."""
    import sounddevice as sd

    duration = 0.15
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Two notes
    wave1 = np.sin(2 * np.pi * 523.25 * t)  # C5
    wave2 = np.sin(2 * np.pi * 659.25 * t)  # E5

    # Apply envelope
    fade_samples = int(sample_rate * 0.02)
    envelope = np.ones_like(wave1)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    # Combine with second note delayed
    combined = np.zeros(int(sample_rate * 0.3))
    combined[:len(wave1)] += wave1 * envelope
    combined[int(sample_rate * 0.1):int(sample_rate * 0.1) + len(wave2)] += wave2 * envelope

    audio = (combined * 0.25).astype(np.float32)
    sd.play(audio, sample_rate)
    sd.wait()
