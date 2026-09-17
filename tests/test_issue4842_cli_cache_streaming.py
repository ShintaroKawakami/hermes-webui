from __future__ import annotations

import threading
import time

import api.models as models


class _DelayedAcquireLock:
    """Gate one lock acquisition so an invalidation can win the race."""

    def __init__(self):
        self._lock = threading.Lock()
        self._skip = 0
        self._delay = False
        self.before_delayed_acquire = threading.Event()
        self.allow_delayed_acquire = threading.Event()

    def arm_after(self, acquisitions):
        self._skip = acquisitions
        self._delay = True

    def __enter__(self):
        if self._delay:
            if self._skip:
                self._skip -= 1
            else:
                self._delay = False
                self.before_delayed_acquire.set()
                assert self.allow_delayed_acquire.wait(1.0)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def _cache_context(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    db_path = hermes_home / "state.db"
    cache_key = ("cli-cache-test",)
    monkeypatch.setattr(
        models,
        "_resolve_cli_sessions_context",
        lambda: (hermes_home, db_path, "default", cache_key),
    )
    return hermes_home, db_path, cache_key


def test_get_cli_sessions_follower_reuses_stale_rows_during_slow_rebuild(monkeypatch, tmp_path):
    """A slow refresh must not keep sidebar followers behind the cache lock."""
    _hermes_home, _db_path, cache_key = _cache_context(monkeypatch, tmp_path)
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 60.0)
    cache_stamp = models._cli_sessions_cache_invalidation_stamp()
    with models._CLI_SESSIONS_CACHE_LOCK:
        models._CLI_SESSIONS_CACHE[cache_key] = (
            time.monotonic() - 1.0,
            cache_stamp,
            [{"session_id": "stale", "title": "stale-row"}],
        )

    started = threading.Event()
    owner_block = threading.Event()
    results = {}

    def blocking_loader(*_args, **_kwargs):
        started.set()
        owner_block.wait()
        return [{"session_id": "fresh", "title": "fresh-row"}]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", blocking_loader)
    owner = threading.Thread(
        target=lambda: results.setdefault("owner", models.get_cli_sessions()),
        daemon=True,
    )
    follower = threading.Thread(
        target=lambda: results.setdefault("follower", models.get_cli_sessions()),
        daemon=True,
    )

    try:
        owner.start()
        assert started.wait(1.0), "owner did not start"
        follower.start()
        follower.join(1.0)

        assert not follower.is_alive()
        assert results["follower"] == [{"session_id": "stale", "title": "stale-row"}]
    finally:
        owner_block.set()
        owner.join(1.0)
        models.clear_cli_sessions_cache()

    assert not owner.is_alive()
    assert results["owner"] == [{"session_id": "fresh", "title": "fresh-row"}]


def test_get_cli_sessions_caches_a_deep_copied_uncached_result(monkeypatch, tmp_path):
    """Cache hits keep the established deep-copy isolation and read behavior."""
    _cache_context(monkeypatch, tmp_path)
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 60.0)
    loads = []

    def loader(*_args, **_kwargs):
        loads.append(True)
        return [{"session_id": "one", "details": {"title": "original"}}]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", loader)
    try:
        first = models.get_cli_sessions()
        first[0]["details"]["title"] = "mutated"
        second = models.get_cli_sessions()
    finally:
        models.clear_cli_sessions_cache()

    assert loads == [True]
    assert second == [{"session_id": "one", "details": {"title": "original"}}]


def test_clear_during_rebuild_does_not_restore_invalidated_rows(monkeypatch, tmp_path):
    """An explicit invalidation must win over a rebuild already in flight."""
    _hermes_home, _db_path, cache_key = _cache_context(monkeypatch, tmp_path)
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 60.0)
    started = threading.Event()
    release = threading.Event()
    results = {}

    def blocking_loader(*_args, **_kwargs):
        started.set()
        release.wait()
        return [{"session_id": "stale-after-clear"}]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", blocking_loader)
    owner = threading.Thread(
        target=lambda: results.setdefault("owner", models.get_cli_sessions()),
        daemon=True,
    )
    owner.start()
    assert started.wait(1.0), "owner did not start"

    models.clear_cli_sessions_cache()
    release.set()
    owner.join(1.0)

    assert not owner.is_alive()
    assert results["owner"] == [{"session_id": "stale-after-clear"}]
    with models._CLI_SESSIONS_CACHE_LOCK:
        assert cache_key not in models._CLI_SESSIONS_CACHE

    monkeypatch.setattr(
        models,
        "_load_cli_sessions_uncached",
        lambda *_args, **_kwargs: [{"session_id": "recovered"}],
    )
    assert models.get_cli_sessions() == [{"session_id": "recovered"}]


def test_cold_follower_does_not_start_a_second_rebuild(monkeypatch, tmp_path):
    """A cold follower returns promptly instead of duplicating a slow query."""
    _cache_context(monkeypatch, tmp_path)
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 60.0)
    started = threading.Event()
    release = threading.Event()
    calls = []
    results = {}

    def blocking_loader(*_args, **_kwargs):
        calls.append(True)
        started.set()
        release.wait()
        return [{"session_id": "fresh"}]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", blocking_loader)
    owner = threading.Thread(
        target=lambda: results.setdefault("owner", models.get_cli_sessions()),
        daemon=True,
    )
    follower = threading.Thread(
        target=lambda: results.setdefault("follower", models.get_cli_sessions()),
        daemon=True,
    )
    owner.start()
    assert started.wait(1.0), "owner did not start"
    follower.start()
    follower.join(1.0)
    try:
        assert not follower.is_alive()
        assert results["follower"] == []
        assert calls == [True]
    finally:
        release.set()
        owner.join(1.0)
        models.clear_cli_sessions_cache()

    assert results["owner"] == [{"session_id": "fresh"}]


def test_clear_between_invalidation_check_and_store_wins(monkeypatch, tmp_path):
    """A clear in the final write window must prevent stale cache repopulation."""
    _hermes_home, _db_path, cache_key = _cache_context(monkeypatch, tmp_path)
    gate = _DelayedAcquireLock()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_LOCK", gate)
    models.clear_cli_sessions_cache()
    monkeypatch.setattr(models, "_CLI_SESSIONS_CACHE_TTL_SECONDS", 60.0)
    started = threading.Event()
    release = threading.Event()
    results = {}

    def blocking_loader(*_args, **_kwargs):
        started.set()
        release.wait()
        return [{"session_id": "invalidated"}]

    monkeypatch.setattr(models, "_load_cli_sessions_uncached", blocking_loader)
    owner = threading.Thread(
        target=lambda: results.setdefault("owner", models.get_cli_sessions()),
        daemon=True,
    )
    owner.start()
    assert started.wait(1.0), "owner did not start"
    # The first lock acquisition after the loader is the stamp read; delay
    # the following cache-write acquisition to open the exact race window.
    gate.arm_after(1)
    release.set()
    assert gate.before_delayed_acquire.wait(1.0), "cache write was not gated"
    models.clear_cli_sessions_cache()
    gate.allow_delayed_acquire.set()
    owner.join(1.0)

    assert not owner.is_alive()
    assert results["owner"] == [{"session_id": "invalidated"}]
    with gate:
        assert cache_key not in models._CLI_SESSIONS_CACHE
