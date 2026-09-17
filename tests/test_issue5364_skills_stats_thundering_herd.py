import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api import profiles


def test_profile_skill_stats_single_flight_but_profiles_are_independent(monkeypatch, tmp_path):
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    (profile_a / "skills").mkdir(parents=True)
    (profile_b / "skills").mkdir(parents=True)

    active = 0
    max_active = 0
    scans = {}
    state_lock = threading.Lock()
    original_iter = __import__(
        "agent.skill_utils", fromlist=["iter_skill_index_files"]
    ).iter_skill_index_files

    def counting_iter(skills_dir, filename):
        nonlocal active, max_active
        profile = Path(skills_dir).parent.name
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            scans[profile] = scans.get(profile, 0) + 1
        time.sleep(0.05)
        try:
            yield from original_iter(skills_dir, filename)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr("agent.skill_utils.iter_skill_index_files", counting_iter)
    profiles._SKILLS_STATS_CACHE.clear()
    profiles._SKILLS_STATS_LOCKS.clear()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(profiles._get_profile_skills_stats, [profile_a] * 3 + [profile_b]))

    assert results == [(0, 0)] * 4
    assert scans == {"a": 1, "b": 1}
    assert max_active >= 2


def test_profile_list_single_flight_stamps_ttl_after_build(monkeypatch):
    profiles._invalidate_list_profiles_cache()
    build_started = threading.Event()
    release_build = threading.Event()
    build_calls = 0
    build_finished = 0.0
    call_lock = threading.Lock()

    def slow_build():
        nonlocal build_calls, build_finished
        with call_lock:
            build_calls += 1
        build_started.set()
        assert release_build.wait(timeout=2)
        build_finished = time.time()
        return [{"name": "default", "is_default": True}]

    monkeypatch.setattr(profiles, "_build_profile_rows_fast", slow_build)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(profiles.list_profiles_api)
        assert build_started.wait(timeout=2)
        second = pool.submit(profiles.list_profiles_api)
        time.sleep(0.05)
        assert not second.done()
        release_build.set()
        assert first.result() == second.result()

    assert build_calls == 1
    assert profiles._LIST_PROFILES_CACHE[1] >= build_finished
    profiles._invalidate_list_profiles_cache()
