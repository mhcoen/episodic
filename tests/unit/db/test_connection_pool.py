"""Tests for the SQLite ConnectionPool accounting and thread-safety."""

import os
import threading
import tempfile

import pytest

from episodic.db_connection import ConnectionPool


@pytest.fixture
def pool():
    db_path = os.path.join(tempfile.mkdtemp(), "pool.db")
    p = ConnectionPool(db_path, pool_size=3)
    # Seed a table so SELECTs work.
    c = p.get_connection()
    c.execute("CREATE TABLE t (x INTEGER)")
    c.commit()
    p.return_connection(c)
    yield p
    p.close_all()


def test_checkout_return_does_not_saturate(pool):
    # Repeated serial checkout/return must not exhaust the pool. The old
    # qsize()+len(connection_info) double-count would saturate and time out.
    for _ in range(50):
        conn = pool.get_connection(timeout=2)
        conn.execute("SELECT 1")
        pool.return_connection(conn)
    # Never exceeds the configured size.
    assert pool._total <= pool.pool_size


def test_concurrent_checkout_within_limit(pool):
    # Hold up to pool_size connections concurrently, then release.
    held = [pool.get_connection(timeout=2) for _ in range(pool.pool_size)]
    assert pool._total == pool.pool_size
    for c in held:
        pool.return_connection(c)
    # A subsequent checkout still works and count stays bounded.
    c = pool.get_connection(timeout=2)
    assert pool._total <= pool.pool_size
    pool.return_connection(c)


def test_connection_usable_from_another_thread(pool):
    # check_same_thread=False: a pooled connection works off-thread.
    conn = pool.get_connection(timeout=2)
    result = {}

    def worker():
        try:
            conn.execute("INSERT INTO t (x) VALUES (1)")
            conn.commit()
            result["ok"] = True
        except Exception as e:  # pragma: no cover
            result["err"] = repr(e)

    th = threading.Thread(target=worker)
    th.start()
    th.join(timeout=5)
    pool.return_connection(conn)
    assert result.get("ok") is True, result


def test_parallel_threads_do_not_deadlock(pool):
    errors = []

    def worker():
        try:
            for _ in range(20):
                c = pool.get_connection(timeout=5)
                c.execute("SELECT 1")
                pool.return_connection(c)
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert pool._total <= pool.pool_size


def test_close_all_resets_total(pool):
    c = pool.get_connection(timeout=2)
    pool.return_connection(c)
    pool.close_all()
    assert pool._total == 0
    assert pool.connection_info == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
