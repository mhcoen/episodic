#!/usr/bin/env python3
"""
Simple voice test script for Episodic.

Tests:
1. Microphone recording
2. Speech-to-text (OpenAI Whisper API or local faster-whisper)
3. Text-to-speech (OpenAI TTS API or local Piper)
4. Audio playback

Usage:
    python scripts/test_voice.py
"""

import sys
import io
import wave
import tempfile
import time
import numpy as np

def check_dependencies():
    """Check required packages are installed."""
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)

    return True

def test_microphone(duration=3):
    """Test microphone recording."""
    import sounddevice as sd

    print(f"\n=== Recording {duration} seconds ===")
    print("Speak now!")

    sample_rate = 16000

    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                      channels=1, dtype='int16')
        sd.wait()
        print(f"✓ Recorded {len(audio)} samples")

        max_amplitude = np.max(np.abs(audio))
        print(f"  Max amplitude: {max_amplitude}")
        if max_amplitude < 100:
            print("  ⚠ Audio level very low - check microphone")
        else:
            print("  ✓ Audio level looks good")

        return audio, sample_rate
    except Exception as e:
        print(f"✗ Microphone error: {e}")
        return None, None

def audio_to_wav_bytes(audio_data, sample_rate):
    """Convert numpy audio to WAV bytes."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())
    return wav_buffer.getvalue()

def save_wav_temp(audio_data, sample_rate):
    """Save audio to temp WAV file, return path."""
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        return f.name

# ============ STT FUNCTIONS ============

def stt_openai(audio_data, sample_rate):
    """Speech-to-text with OpenAI Whisper API."""
    from openai import OpenAI
    import os

    print("\n=== STT: OpenAI Whisper API ===")

    try:
        client = OpenAI()
        temp_path = save_wav_temp(audio_data, sample_rate)

        start = time.time()
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        elapsed = time.time() - start

        text = transcript.text.strip()
        print(f"✓ Transcribed in {elapsed:.2f}s: \"{text}\"")

        os.unlink(temp_path)
        return text
    except Exception as e:
        print(f"✗ OpenAI STT error: {e}")
        return None

def stt_local_whisper(audio_data, sample_rate, model_size="base"):
    """Speech-to-text with local faster-whisper."""
    import os

    print(f"\n=== STT: Local Whisper ({model_size}) ===")

    try:
        from faster_whisper import WhisperModel

        print(f"Loading model '{model_size}' (first time downloads ~150MB)...")
        start_load = time.time()
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        load_time = time.time() - start_load
        print(f"  Model loaded in {load_time:.2f}s")

        temp_path = save_wav_temp(audio_data, sample_rate)

        start = time.time()
        segments, info = model.transcribe(temp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        elapsed = time.time() - start

        print(f"✓ Transcribed in {elapsed:.2f}s: \"{text}\"")
        print(f"  Detected language: {info.language} ({info.language_probability:.0%})")

        os.unlink(temp_path)
        return text
    except ImportError:
        print("✗ faster-whisper not installed. Run: pip install faster-whisper")
        return None
    except Exception as e:
        print(f"✗ Local Whisper error: {e}")
        return None

# ============ TTS FUNCTIONS ============

def tts_openai(text, voice="alloy"):
    """Text-to-speech with OpenAI TTS API."""
    from openai import OpenAI
    import sounddevice as sd

    print(f"\n=== TTS: OpenAI ({voice}) ===")

    try:
        client = OpenAI()

        display_text = f"\"{text[:50]}...\"" if len(text) > 50 else f"\"{text}\""
        print(f"Converting: {display_text}")

        start = time.time()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="wav"
        )
        gen_time = time.time() - start

        audio_bytes = response.content
        print(f"✓ Generated in {gen_time:.2f}s ({len(audio_bytes)} bytes)")

        # Play
        print("Playing...")
        wav_buffer = io.BytesIO(audio_bytes)
        with wave.open(wav_buffer, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            rate = wav.getframerate()

        audio_array = np.frombuffer(frames, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        sd.play(audio_float, rate)
        sd.wait()
        print("✓ Playback complete")
        return True
    except Exception as e:
        print(f"✗ OpenAI TTS error: {e}")
        return False

# Global cache for XTTS model (expensive to load)
_xtts_model = None
_xtts_device = None

def tts_coqui_xtts(text):
    """Text-to-speech with Coqui XTTS v2 - high quality local TTS."""
    global _xtts_model, _xtts_device
    import sounddevice as sd

    print("\n=== TTS: Coqui XTTS v2 (high quality local) ===")

    try:
        import torch
        from TTS.api import TTS

        display_text = f"\"{text[:50]}...\"" if len(text) > 50 else f"\"{text}\""
        print(f"Converting: {display_text}")

        # Load model if not cached
        if _xtts_model is None:
            # Detect device - prefer MPS on Apple Silicon, then CUDA, then CPU
            if torch.backends.mps.is_available():
                _xtts_device = "mps"
            elif torch.cuda.is_available():
                _xtts_device = "cuda"
            else:
                _xtts_device = "cpu"
            print(f"Using device: {_xtts_device}")

            print("Loading XTTS v2 model (first time downloads ~1.5GB)...")
            start_load = time.time()
            _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_xtts_device)
            load_time = time.time() - start_load
            print(f"  Model loaded in {load_time:.2f}s")
        else:
            print(f"Using cached model (device: {_xtts_device})")

        # Synthesize - XTTS needs a speaker reference
        # Use a built-in speaker (or you could use speaker_wav for voice cloning)
        speaker = "Claribel Dervla"  # One of the built-in voices
        print(f"  Using speaker: {speaker}")

        start = time.time()
        wav = _xtts_model.tts(text=text, speaker=speaker, language="en")
        gen_time = time.time() - start
        print(f"✓ Generated in {gen_time:.2f}s ({len(wav)} samples)")

        # Play
        print("Playing...")
        audio_float = np.array(wav, dtype=np.float32)
        # XTTS outputs at 24kHz
        sd.play(audio_float, 24000)
        sd.wait()
        print("✓ Playback complete")
        return True
    except ImportError:
        print("✗ coqui-tts not installed. Run: pip install coqui-tts")
        return False
    except Exception as e:
        print(f"✗ XTTS error: {e}")
        import traceback
        traceback.print_exc()
        return False

def tts_local_piper(text, voice="en_US-lessac-medium"):
    """Text-to-speech with local Piper."""
    import sounddevice as sd
    import os

    print(f"\n=== TTS: Local Piper ({voice}) ===")

    try:
        from piper import PiperVoice

        display_text = f"\"{text[:50]}...\"" if len(text) > 50 else f"\"{text}\""
        print(f"Converting: {display_text}")

        # Find voice model - check common locations
        voice_dirs = [
            os.path.expanduser("~/.local/share/piper-voices"),
            os.path.expanduser("~/piper-voices"),
            ".",
        ]

        model_path = None
        for voice_dir in voice_dirs:
            candidate = os.path.join(voice_dir, f"{voice}.onnx")
            if os.path.exists(candidate):
                model_path = candidate
                break

        if not model_path:
            print(f"✗ Voice model not found. Download with:")
            print(f"  mkdir -p ~/.local/share/piper-voices")
            print(f"  cd ~/.local/share/piper-voices")
            print(f"  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice}.onnx")
            print(f"  curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice}.onnx.json")
            return False

        print(f"Loading voice from {model_path}...")
        start_load = time.time()
        piper_voice = PiperVoice.load(model_path)
        load_time = time.time() - start_load
        print(f"  Voice loaded in {load_time:.2f}s")

        # Synthesize - Piper returns AudioChunk objects per sentence
        start = time.time()
        audio_floats = []
        sample_rate = None
        for chunk in piper_voice.synthesize(text):
            audio_floats.append(chunk.audio_float_array)
            if sample_rate is None:
                sample_rate = chunk.sample_rate

        audio_float = np.concatenate(audio_floats)
        gen_time = time.time() - start
        print(f"✓ Generated in {gen_time:.2f}s ({len(audio_float)} samples)")

        # Play
        print("Playing...")
        sd.play(audio_float, sample_rate)
        sd.wait()
        print("✓ Playback complete")
        return True
    except ImportError:
        print("✗ piper-tts not installed. Run: pip install piper-tts")
        return False
    except Exception as e:
        print(f"✗ Local Piper error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============ TEST MENUS ============

def compare_stt():
    """Compare OpenAI vs Local Whisper STT."""
    print("\n" + "="*50)
    print("STT COMPARISON: OpenAI vs Local Whisper")
    print("="*50)
    print("\nPress Enter to record 5 seconds of speech...")
    input()

    audio, sample_rate = test_microphone(duration=5)
    if audio is None:
        return

    print("\n--- Testing both STT providers ---")

    # OpenAI
    text_openai = stt_openai(audio, sample_rate)

    # Local
    text_local = stt_local_whisper(audio, sample_rate, model_size="base")

    print("\n" + "="*50)
    print("RESULTS:")
    print(f"  OpenAI:  \"{text_openai}\"")
    print(f"  Local:   \"{text_local}\"")
    print("="*50)

def compare_tts():
    """Compare OpenAI vs Local Piper vs XTTS."""
    print("\n" + "="*50)
    print("TTS COMPARISON: OpenAI vs Piper vs XTTS")
    print("="*50)

    text = "Hello! This is a test of the text to speech system. The quick brown fox jumps over the lazy dog."

    print(f"\nText: \"{text}\"")
    print("\nPress Enter to hear OpenAI TTS...")
    input()
    tts_openai(text, voice="nova")

    print("\nPress Enter to hear Local Piper TTS (fast, lower quality)...")
    input()
    tts_local_piper(text)

    print("\nPress Enter to hear Coqui XTTS (slower, higher quality)...")
    input()
    tts_coqui_xtts(text)

    print("\n" + "="*50)
    print("Quality ranking (typical):")
    print("  1. OpenAI TTS - best overall")
    print("  2. XTTS - close to OpenAI, free & local")
    print("  3. Piper - faster but more robotic")
    print("="*50)

def test_local_roundtrip():
    """Full local test: record -> local whisper -> local piper."""
    print("\n" + "="*50)
    print("LOCAL ROUNDTRIP (no API calls)")
    print("="*50)
    print("\nThis uses only local models (free, private).")
    print("Press Enter to record 3 seconds...")
    input()

    audio, sample_rate = test_microphone(duration=3)
    if audio is None:
        return

    text = stt_local_whisper(audio, sample_rate)
    if not text:
        text = "I could not transcribe that."

    tts_local_piper(f"You said: {text}")

    print("\n✓ Complete - no API calls made!")

def main():
    print("="*50)
    print("EPISODIC VOICE TEST - Extended")
    print("="*50)

    check_dependencies()

    while True:
        print("\nOptions:")
        print("1. Quick TTS test (OpenAI)")
        print("2. Quick TTS test (Local Piper - fast, lower quality)")
        print("3. Quick TTS test (Coqui XTTS - slow first load, high quality)")
        print("4. Compare STT (OpenAI vs Local Whisper)")
        print("5. Compare TTS (OpenAI vs Piper vs XTTS)")
        print("6. Full local roundtrip (Whisper + Piper, no API)")
        print("7. Full OpenAI roundtrip")
        print("q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "1":
            tts_openai("Hello! This is OpenAI text to speech. How does it sound?")
        elif choice == "2":
            tts_local_piper("Hello! This is local Piper text to speech. How does it sound?")
        elif choice == "3":
            tts_coqui_xtts("Hello! This is Coqui XTTS text to speech. How does it sound?")
        elif choice == "4":
            compare_stt()
        elif choice == "5":
            compare_tts()
        elif choice == "6":
            test_local_roundtrip()
        elif choice == "7":
            print("\nPress Enter to record 3 seconds...")
            input()
            audio, sr = test_microphone(3)
            if audio is not None:
                text = stt_openai(audio, sr)
                if text:
                    tts_openai(f"You said: {text}")
        elif choice == "q":
            print("Bye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
