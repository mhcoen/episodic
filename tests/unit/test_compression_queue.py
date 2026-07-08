"""queue_topic_for_compression must report success and start the worker."""

import pytest

from episodic.compression import AsyncCompressionManager


def test_queue_compression_returns_true_and_starts_worker(monkeypatch):
    mgr = AsyncCompressionManager()
    # Don't actually compress anything: stub the segment processor.
    monkeypatch.setattr(mgr, "_compress_topic_segment", lambda job: True)

    assert mgr.running is False
    ok = mgr.queue_compression("start", "end", "topic-a")
    try:
        assert ok is True
        # The worker is started lazily so the job is actually processed
        # instead of sitting in the queue forever.
        assert mgr.running is True
    finally:
        mgr.stop()
    assert mgr.running is False


def test_queue_topic_respects_disabled_flag(monkeypatch):
    import episodic.compression as compression_mod

    monkeypatch.setattr(compression_mod.config, "get",
                        lambda k, d=None: False if k == "auto_compress_topics" else d)

    result = compression_mod.queue_topic_for_compression("s", "e", "t")
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
