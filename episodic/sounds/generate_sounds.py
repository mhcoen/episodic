"""
Generate default sound WAV files using only stdlib (wave + math).

All files: 22050 Hz, mono, 16-bit PCM WAV.

Run: python -m episodic.sounds.generate_sounds
"""

import math
import struct
import wave
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 22050
MAX_AMP = 32767  # 16-bit signed max


def _sine_samples(freq: float, duration_s: float, amplitude: float = 1.0,
                  fade_out_s: float = 0.0) -> list[int]:
    """Generate sine wave samples with optional fade-out."""
    n_samples = int(SAMPLE_RATE * duration_s)
    fade_samples = int(SAMPLE_RATE * fade_out_s)
    samples = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        val = math.sin(2 * math.pi * freq * t) * amplitude
        # Apply fade-out envelope
        if fade_out_s > 0 and i >= n_samples - fade_samples:
            remaining = n_samples - i
            val *= remaining / fade_samples
        samples.append(int(val * MAX_AMP))
    return samples


def _silence(duration_s: float) -> list[int]:
    """Generate silence."""
    return [0] * int(SAMPLE_RATE * duration_s)


def _sweep_samples(freq_start: float, freq_end: float, duration_s: float,
                   amplitude: float = 1.0, fade_out_s: float = 0.0) -> list[int]:
    """Generate a linear frequency sweep."""
    n_samples = int(SAMPLE_RATE * duration_s)
    fade_samples = int(SAMPLE_RATE * fade_out_s)
    samples = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        frac = i / n_samples
        freq = freq_start + (freq_end - freq_start) * frac
        val = math.sin(2 * math.pi * freq * t) * amplitude
        if fade_out_s > 0 and i >= n_samples - fade_samples:
            remaining = n_samples - i
            val *= remaining / fade_samples
        samples.append(int(val * MAX_AMP))
    return samples


def _write_wav(path: Path, samples: list[int]) -> None:
    """Write samples to a WAV file."""
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        data = struct.pack(f'<{len(samples)}h', *samples)
        wf.writeframes(data)


def generate_alarm(output_dir: Path) -> Path:
    """Three short 880 Hz beeps with gaps (~1.5s total, designed to loop)."""
    samples: list[int] = []
    for i in range(3):
        samples.extend(_sine_samples(880, 0.2, amplitude=0.8, fade_out_s=0.03))
        if i < 2:
            samples.extend(_silence(0.15))
    # Trailing silence for loop gap
    samples.extend(_silence(0.35))
    path = output_dir / "alarm_default.wav"
    _write_wav(path, samples)
    return path


def generate_timer(output_dir: Path) -> Path:
    """Ascending two-tone chime C5 (523Hz) -> E5 (659Hz), ~800ms."""
    samples: list[int] = []
    samples.extend(_sine_samples(523.25, 0.3, amplitude=0.7, fade_out_s=0.08))
    samples.extend(_silence(0.05))
    samples.extend(_sine_samples(659.25, 0.4, amplitude=0.8, fade_out_s=0.15))
    path = output_dir / "timer_default.wav"
    _write_wav(path, samples)
    return path


def generate_notification(output_dir: Path) -> Path:
    """Single 1047 Hz ping with decay (~150ms)."""
    samples = _sine_samples(1046.50, 0.15, amplitude=0.6, fade_out_s=0.12)
    path = output_dir / "notification.wav"
    _write_wav(path, samples)
    return path


def generate_error(output_dir: Path) -> Path:
    """Descending sweep 440 -> 220 Hz (~500ms)."""
    samples = _sweep_samples(440, 220, 0.5, amplitude=0.7, fade_out_s=0.1)
    path = output_dir / "error.wav"
    _write_wav(path, samples)
    return path


def generate_success(output_dir: Path) -> Path:
    """Ascending C5 -> E5 -> G5 (~600ms)."""
    samples: list[int] = []
    samples.extend(_sine_samples(523.25, 0.15, amplitude=0.6, fade_out_s=0.03))
    samples.extend(_silence(0.02))
    samples.extend(_sine_samples(659.25, 0.15, amplitude=0.7, fade_out_s=0.03))
    samples.extend(_silence(0.02))
    samples.extend(_sine_samples(783.99, 0.2, amplitude=0.8, fade_out_s=0.08))
    path = output_dir / "success.wav"
    _write_wav(path, samples)
    return path


def generate_all(output_dir: Optional[Path] = None) -> list[Path]:
    """Generate all default sound files. Returns list of created paths."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        generate_alarm(output_dir),
        generate_timer(output_dir),
        generate_notification(output_dir),
        generate_error(output_dir),
        generate_success(output_dir),
    ]
    return paths


if __name__ == "__main__":
    paths = generate_all()
    for p in paths:
        size = p.stat().st_size
        print(f"  {p.name}: {size:,} bytes")
    print(f"Generated {len(paths)} sound files")
