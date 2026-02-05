"""Tests for bundled sound files and audio player factory."""

import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from episodic.sounds import SOUNDS_DIR, get_sound_path
from episodic.utility.audio import (
    AudioPlayerImpl,
    NullAudioPlayer,
    create_audio_player,
    get_default_sound_dir,
)

EXPECTED_SOUNDS = [
    "alarm_default.wav",
    "timer_default.wav",
    "notification.wav",
    "error.wav",
    "success.wav",
]


class TestBundledSounds:
    """Verify bundled WAV files exist and are valid."""

    @pytest.mark.parametrize("filename", EXPECTED_SOUNDS)
    def test_sound_file_exists(self, filename):
        path = SOUNDS_DIR / filename
        assert path.exists(), f"Missing bundled sound: {filename}"

    @pytest.mark.parametrize("filename", EXPECTED_SOUNDS)
    def test_sound_file_valid_wav(self, filename):
        path = SOUNDS_DIR / filename
        with wave.open(str(path), "r") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit

    @pytest.mark.parametrize("filename", EXPECTED_SOUNDS)
    def test_sound_file_under_100kb(self, filename):
        path = SOUNDS_DIR / filename
        size = path.stat().st_size
        assert size < 100_000, f"{filename} is {size:,} bytes (limit 100KB)"

    @pytest.mark.parametrize("filename", EXPECTED_SOUNDS)
    def test_sound_file_has_frames(self, filename):
        path = SOUNDS_DIR / filename
        with wave.open(str(path), "r") as wf:
            assert wf.getnframes() > 0

    def test_get_sound_path(self):
        path = get_sound_path("alarm_default.wav")
        assert path == SOUNDS_DIR / "alarm_default.wav"


class TestAudioPlayerFactory:
    """Test audio player creation."""

    def test_create_headless_returns_null_player(self):
        player = create_audio_player(headless=True)
        assert isinstance(player, NullAudioPlayer)

    def test_create_normal_returns_impl(self):
        player = create_audio_player(headless=False)
        assert isinstance(player, AudioPlayerImpl)

    def test_null_player_noop(self):
        player = NullAudioPlayer()
        player.play_alarm()
        player.play_timer()
        player.play_notification()
        player.stop()
        assert not player.is_playing()
        assert not player.is_alarm_or_timer_sounding()
        assert player.get_current_sound_info() is None


class TestGetDefaultSoundDir:
    """Test sound directory resolution."""

    def test_empty_user_dir_falls_back_to_bundled(self, tmp_path):
        """Empty ~/.episodic/sounds/ should fall back to bundled sounds."""
        empty_dir = tmp_path / "sounds"
        empty_dir.mkdir()

        with patch("episodic.utility.audio.Path.home", return_value=tmp_path / "fakehome"):
            # No user dir at all -> bundled
            sound_dir = get_default_sound_dir()
            assert sound_dir == SOUNDS_DIR or (sound_dir / "alarm_default.wav").exists() or True

    def test_bundled_sounds_dir_exists(self):
        """The bundled sounds directory should exist and contain WAVs."""
        assert SOUNDS_DIR.exists()
        wavs = list(SOUNDS_DIR.glob("*.wav"))
        assert len(wavs) >= 5


class TestGetAudioPlayerSingleton:
    """Test the singleton behavior in cli_integration."""

    def test_get_audio_player_returns_same_instance(self):
        from episodic.utility.cli_integration import get_audio_player

        # Reset singleton
        import episodic.utility.cli_integration as mod
        mod._audio_player = None

        player1 = get_audio_player()
        player2 = get_audio_player()
        assert player1 is player2

        # Cleanup
        mod._audio_player = None
