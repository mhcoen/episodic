"""
Audio playback queue for Episodic voice mode.

Handles non-blocking audio playback with queuing for streaming TTS.
"""

import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np


class AudioPlayback:
    """
    Non-blocking audio playback with queuing.

    Supports streaming TTS by queueing audio chunks for sequential playback.
    """

    def __init__(self, on_start: Optional[Callable] = None, on_stop: Optional[Callable] = None):
        """
        Initialize audio playback.

        Args:
            on_start: Called when playback starts
            on_stop: Called when playback queue is empty
        """
        self._queue: queue.Queue[Tuple[np.ndarray, int]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = False
        self._on_start = on_start
        self._on_stop = on_stop

    def _playback_thread(self):
        """Background thread for audio playback."""
        import sounddevice as sd

        first_chunk = True

        while not self._stop_event.is_set():
            try:
                # Get next audio chunk (with timeout to check stop event)
                audio, sample_rate = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._is_playing:
                    self._is_playing = False
                    if self._on_stop:
                        self._on_stop()
                continue

            if not self._is_playing:
                self._is_playing = True
                if self._on_start:
                    self._on_start()

            first_chunk = False

            # Play audio (blocking)
            try:
                sd.play(audio, sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")

            self._queue.task_done()

    def start(self):
        """Start the playback thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._playback_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop playback and clear queue."""
        import sounddevice as sd

        self._stop_event.set()

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Stop any current playback
        try:
            sd.stop()
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._is_playing = False

    def enqueue(self, audio: np.ndarray, sample_rate: int):
        """
        Add audio to the playback queue.

        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Sample rate in Hz
        """
        self._queue.put((audio, sample_rate))

    def play_immediate(self, audio: np.ndarray, sample_rate: int):
        """
        Play audio immediately, clearing the queue.

        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Sample rate in Hz
        """
        import sounddevice as sd

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Stop current playback
        try:
            sd.stop()
        except Exception:
            pass

        # Play immediately
        if self._on_start:
            self._on_start()

        try:
            sd.play(audio, sample_rate)
            sd.wait()
        finally:
            if self._on_stop:
                self._on_stop()

    def wait(self):
        """Wait for all queued audio to finish playing."""
        self._queue.join()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing

    @property
    def queue_size(self) -> int:
        """Get number of items in the queue."""
        return self._queue.qsize()


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
