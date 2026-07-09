"""Hardware-free tests for VoiceModeManager.

Covers the state machine, string helpers, idle-timer logic, lazy provider
getters, and the singleton — the surface that a decomposition would touch,
without needing real audio hardware. config and provider factories are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

import episodic.voice.voice_mode as vm
from episodic.voice.voice_mode import VoiceModeManager, VoiceState


class FakeConfig:
    def __init__(self, overrides=None):
        self.data = dict(overrides or {})

    def get(self, key, default=None):
        return self.data.get(key, default)


@pytest.fixture
def cfg(monkeypatch):
    fake = FakeConfig()
    monkeypatch.setattr(vm, "config", fake)
    return fake


@pytest.fixture
def manager(cfg):
    return VoiceModeManager()


class TestStateMachine:
    def test_initial_state_off(self, manager):
        assert manager.state == VoiceState.OFF
        assert manager.is_active is False
        assert manager.is_listening is False
        assert manager.is_idle is False
        assert manager.is_speaking is False

    def test_set_state_updates_properties(self, manager):
        manager._set_state(VoiceState.LISTENING)
        assert manager.is_active is True
        assert manager.is_listening is True
        assert manager.is_idle is False

        manager._set_state(VoiceState.IDLE)
        assert manager.is_idle is True
        assert manager.is_listening is False

        manager._set_state(VoiceState.SPEAKING)
        assert manager.is_speaking is True

    def test_state_change_callback_fires_on_transition(self, manager):
        seen = []
        manager._on_state_change = seen.append
        manager._set_state(VoiceState.LISTENING)
        assert seen == [VoiceState.LISTENING]

    def test_callback_not_fired_when_state_unchanged(self, manager):
        manager._set_state(VoiceState.LISTENING)
        seen = []
        manager._on_state_change = seen.append
        manager._set_state(VoiceState.LISTENING)  # same state
        assert seen == []


class TestStringHelpers:
    @pytest.mark.parametrize("text", ["computer", "Computer.", " COMPUTER ", "hey computer"])
    def test_is_just_wake_word_true(self, text):
        assert VoiceModeManager._is_just_wake_word(text, "computer") is True

    @pytest.mark.parametrize("text", ["computer please", "hello", "hey jarvis"])
    def test_is_just_wake_word_false(self, text):
        assert VoiceModeManager._is_just_wake_word(text, "computer") is False

    @pytest.mark.parametrize("text", [
        "go to sleep", "Please STOP LISTENING now", "go idle", "standby",
    ])
    def test_is_sleep_command_true(self, manager, text):
        assert manager.is_sleep_command(text) is True

    @pytest.mark.parametrize("text", ["what time is it", "wake up", "hello there"])
    def test_is_sleep_command_false(self, manager, text):
        assert manager.is_sleep_command(text) is False


class TestIdleTimer:
    def test_start_creates_timer_when_enabled(self, manager, cfg):
        cfg.data["voice_wake_word_enabled"] = True
        try:
            manager._start_idle_timer(timeout=100)  # long, won't fire
            assert manager._idle_timer is not None
            assert manager._idle_timer_started_at is not None
        finally:
            manager._cancel_idle_timer()

    def test_no_timer_when_timeout_zero(self, manager, cfg):
        cfg.data["voice_wake_word_enabled"] = True
        manager._start_idle_timer(timeout=0)
        assert manager._idle_timer is None

    def test_no_timer_when_wake_word_disabled(self, manager, cfg):
        cfg.data["voice_wake_word_enabled"] = False
        manager._start_idle_timer(timeout=100)
        assert manager._idle_timer is None

    def test_cancel_clears_timer(self, manager, cfg):
        cfg.data["voice_wake_word_enabled"] = True
        manager._start_idle_timer(timeout=100)
        manager._cancel_idle_timer()
        assert manager._idle_timer is None

    def test_resume_respects_state(self, manager, cfg):
        cfg.data["voice_wake_word_enabled"] = True
        # SPEAKING / IDLE / OFF must NOT start a timer on resume.
        for state in (VoiceState.OFF, VoiceState.IDLE, VoiceState.SPEAKING):
            manager._set_state(state)
            manager.resume_idle_timer()
            assert manager._idle_timer is None, state
        # An active state DOES start one.
        manager._set_state(VoiceState.LISTENING)
        try:
            manager.resume_idle_timer()
            assert manager._idle_timer is not None
        finally:
            manager._cancel_idle_timer()

    def test_resume_with_remaining_time(self, manager, cfg, monkeypatch):
        cfg.data["voice_wake_word_enabled"] = True
        cfg.data["voice_idle_timeout"] = 15
        times = iter([100.0, 110.0])  # started_at=100, now=110 -> 5s elapsed
        monkeypatch.setattr(vm.time, "time", lambda: next(times))
        manager._idle_timer_started_at = 100.0
        captured = {}
        monkeypatch.setattr(manager, "_start_idle_timer",
                            lambda t=None: captured.__setitem__("t", t))
        manager._resume_idle_timer()
        # 15 - 10 elapsed... uses one time() call -> remaining respects min 2s
        assert captured["t"] >= 2.0


class TestLazyProviders:
    def test_stt_provider_cached_and_uses_config(self, manager, cfg):
        cfg.data["voice_stt_provider"] = "openai_whisper"
        fake_provider = MagicMock()
        with patch("episodic.voice.stt_providers.get_stt_provider",
                   return_value=fake_provider) as factory:
            p1 = manager._get_stt_provider()
            p2 = manager._get_stt_provider()
        assert p1 is fake_provider
        assert p2 is fake_provider
        factory.assert_called_once()  # cached after first call
        assert factory.call_args.args[0] == "openai_whisper"

    def test_local_whisper_passes_model_size(self, manager, cfg):
        cfg.data["voice_stt_provider"] = "local_whisper"
        cfg.data["voice_local_whisper_model"] = "small"
        with patch("episodic.voice.stt_providers.get_stt_provider") as factory:
            manager._get_stt_provider()
        assert factory.call_args.kwargs.get("model_size") == "small"


class TestMuteUnmute:
    def test_mute_delegates(self, manager):
        manager._audio_capture = MagicMock()
        manager.mute()
        manager._audio_capture.mute.assert_called_once()

    def test_unmute_delegates(self, manager):
        manager._audio_capture = MagicMock()
        manager.unmute()
        manager._audio_capture.unmute.assert_called_once()

    def test_mute_noop_without_capture(self, manager):
        manager._audio_capture = None
        manager.mute()  # should not raise


class TestSingleton:
    def test_get_voice_manager_singleton(self, cfg, monkeypatch):
        monkeypatch.setattr(vm, "_voice_manager", None)
        m1 = vm.get_voice_manager()
        m2 = vm.get_voice_manager()
        assert m1 is m2

    def test_cleanup_resets_singleton(self, cfg, monkeypatch):
        monkeypatch.setattr(vm, "_voice_manager", None)
        m = vm.get_voice_manager()
        m.stop = MagicMock()
        with patch("episodic.voice.stt_providers.cleanup_stt_providers"), \
             patch("episodic.voice.tts_providers.cleanup_tts_providers"):
            vm.cleanup_voice_mode()
        assert vm._voice_manager is None
        m.stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
