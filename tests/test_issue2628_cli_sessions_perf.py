"""Regression coverage for capped CLI/agent session sidebar scans (#2628)."""

import pathlib
import sqlite3
import time

import api.agent_sessions as agent_sessions

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make_state_db(path, *, sessions=80, messages_per_session=3):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            session_source TEXT,
            title TEXT,
            model TEXT,
            started_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0,
            parent_session_id TEXT,
            ended_at REAL,
            end_reason TEXT
        );
        CREATE INDEX idx_sessions_started ON sessions(started_at);
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
        """
    )
    base = time.time() - sessions
    for i in range(sessions):
        sid = f"cli_perf_{i:04d}"
        started = base + i
        conn.execute(
            """
            INSERT INTO sessions
            (id, source, session_source, title, model, started_at, message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, 'cli', 'cli', ?, 'openai/gpt-5', ?, ?, NULL, NULL, NULL)
            """,
            (sid, sid, started, messages_per_session),
        )
        for j in range(messages_per_session):
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, 'hello', ?)",
                (f"msg_{i:04d}_{j:02d}", sid, "user" if j == 0 else "assistant", started + j / 10),
            )
    conn.commit()
    conn.close()


def _make_modern_state_db(path, *, sessions=80):
    """Create the modern state.db shape used by the indexed fast candidate path."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            session_source TEXT,
            title TEXT,
            model TEXT,
            started_at REAL NOT NULL,
            last_activity_at REAL,
            message_count INTEGER DEFAULT 0,
            parent_session_id TEXT,
            ended_at REAL,
            end_reason TEXT
        );
        CREATE INDEX idx_sessions_effective_activity
            ON sessions(COALESCE(last_activity_at, started_at) DESC, started_at DESC);
        CREATE INDEX idx_sessions_started ON sessions(started_at DESC);
        CREATE INDEX idx_sessions_source_id ON sessions(source, id);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        );
        CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
        """
    )
    base = time.time() - sessions
    for i in range(sessions):
        sid = f"modern_{i:04d}"
        started = base + i
        conn.execute(
            """
            INSERT INTO sessions
            (id, source, session_source, title, model, started_at,
             last_activity_at, message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, 'cli', 'cli', ?, 'openai/gpt-5', ?, ?, 1, NULL, NULL, NULL)
            """,
            (sid, sid, started, started),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, 'user', 'hello', ?)",
            (sid, started),
        )

    # A resumed old session must still win by its latest message timestamp.
    old_started = time.time() - 60 * 60 * 24 * 30
    recent_activity = time.time() + 60
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_resumed_old', 'cli', 'cli', 'Old resumed session',
                'openai/gpt-5', ?, ?, 2, NULL, NULL, NULL)
        """,
        (old_started, old_started),
    )
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('modern_resumed_old', 'user', 'old hello', ?)",
        (old_started,),
    )
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('modern_resumed_old', 'assistant', 'recent reply', ?)",
        (recent_activity,),
    )
    conn.commit()
    conn.close()


def test_importable_agent_rows_push_sidebar_limit_into_sql(tmp_path):
    """A capped sidebar scan should not aggregate the entire state.db first."""
    db = tmp_path / "state.db"
    _make_state_db(db, sessions=120, messages_per_session=5)

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=20, exclude_sources=("webui",))

    assert len(rows) == 20
    assert [row["id"] for row in rows][:3] == ["cli_perf_0119", "cli_perf_0118", "cli_perf_0117"]
    assert {row["actual_message_count"] for row in rows} == {5}

    src = (REPO_ROOT / "api" / "agent_sessions.py").read_text()
    assert "WITH candidates AS" in src
    assert "CROSS JOIN sessions s ON s.id = c.id" in src
    assert "SELECT MAX(mx.timestamp) FROM messages mx WHERE mx.session_id = s.id" in src
    assert "candidate_limit = max(result_limit * 8, result_limit)" in src


def test_importable_agent_rows_limit_includes_resumed_old_session(tmp_path):
    """The capped candidate window must not hide old sessions resumed recently."""
    db = tmp_path / "state.db"
    _make_state_db(db, sessions=200, messages_per_session=1)

    old_started = time.time() - 60 * 60 * 24 * 30
    recent_activity = time.time() + 60
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('cli_resumed_old', 'cli', 'cli', 'Old resumed session', 'openai/gpt-5', ?, 2, NULL, NULL, NULL)
        """,
        (old_started,),
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('old_msg_1', 'cli_resumed_old', 'user', 'old hello', ?)",
        (old_started,),
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('old_msg_2', 'cli_resumed_old', 'assistant', 'recent reply', ?)",
        (recent_activity,),
    )
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=20, exclude_sources=("webui",))

    assert rows[0]["id"] == "cli_resumed_old"
    assert rows[0]["actual_message_count"] == 2


def test_modern_candidate_projection_keeps_resumed_old_session(tmp_path):
    """The indexed modern path must retain exact latest-message ordering."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    root_started = time.time() - 900
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_chain_root', 'cli', 'cli', 'Modern chain',
                'openai/gpt-5', ?, ?, 0, NULL, ?, 'compression')
        """,
        (root_started, root_started, root_started + 10),
    )
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_chain_empty', 'cli', 'cli', 'Modern chain #2',
                'openai/gpt-5', ?, ?, 1, 'modern_chain_root', ?, NULL)
        """,
        (root_started + 11, root_started + 11, root_started + 20),
    )
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('modern_chain_empty', 'user', 'chain', ?)",
        (root_started + 12,),
    )
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=200, exclude_sources=("webui",))

    assert rows[0]["id"] == "modern_resumed_old"
    assert rows[0]["actual_message_count"] == 2
    chain = next(row for row in rows if row["id"] == "modern_chain_empty")
    assert chain["_lineage_root_id"] == "modern_chain_root"
    assert chain["_compression_segment_count"] == 2
    src = (REPO_ROOT / "api" / "agent_sessions.py").read_text()
    assert "latest_messages AS" in src
    assert "FROM latest_messages lm" in src
    assert "message_candidates_raw" in src
    assert "CROSS JOIN sessions s ON s.id = mc.id" in src


def test_modern_candidate_projection_filters_excluded_sources_before_limit(tmp_path):
    """Excluded recent rows must not consume the bounded candidate window."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    base = time.time() + 1000
    for i in range(20):
        sid = f"modern_webui_{i:02d}"
        started = base + i
        conn.execute(
            """
            INSERT INTO sessions
            (id, source, session_source, title, model, started_at,
             last_activity_at, message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, 'webui', 'webui', ?, 'openai/gpt-5', ?, ?, 1, NULL, NULL, NULL)
            """,
            (sid, sid, started, started),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, 'user', 'hidden', ?)",
            (sid, started),
        )
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(
        db, limit=1, exclude_sources=("webui",)
    )

    assert rows[0]["id"] == "modern_resumed_old"


def test_modern_empty_candidates_filter_excluded_sources_before_limit(monkeypatch, tmp_path):
    """Excluded empty rows must not consume the bounded empty candidate window."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    base = time.time() + 1000
    for i in range(70):
        sid = f"modern_empty_webui_{i:02d}"
        started = base + i
        conn.execute(
            """
            INSERT INTO sessions
            (id, source, session_source, title, model, started_at,
             last_activity_at, message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, 'webui', 'webui', ?, 'openai/gpt-5', ?, ?, 0, NULL, NULL, NULL)
            """,
            (sid, sid, started, started),
        )
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_empty_cli_after_hidden', 'cli', 'cli',
                'Recovered empty session', 'openai/gpt-5', ?, ?, 0,
                NULL, NULL, NULL)
        """,
        (base - 1, base - 1),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(agent_sessions, "_project_agent_session_rows", lambda rows: rows)
    monkeypatch.setattr(agent_sessions, "is_cli_session_row_visible", lambda row: True)
    rows = agent_sessions.read_importable_agent_session_rows(
        db, limit=1, exclude_sources=("webui",)
    )

    assert rows[0]["id"] == "modern_empty_cli_after_hidden"


def test_modern_candidate_projection_keeps_message_count_mismatch_candidate(monkeypatch, tmp_path):
    """A stale positive count must not hide a session with no message rows."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    started = time.time() + 120
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_message_count_mismatch', 'cli', 'cli',
                'Useful recovered conversation', 'openai/gpt-5', ?, ?,
                3, NULL, NULL, NULL)
        """,
        (started, started),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(agent_sessions, "_project_agent_session_rows", lambda rows: rows)
    monkeypatch.setattr(agent_sessions, "is_cli_session_row_visible", lambda row: True)
    rows = agent_sessions.read_importable_agent_session_rows(
        db, limit=1, exclude_sources=("webui",)
    )

    assert rows[0]["id"] == "modern_message_count_mismatch"


def test_modern_candidate_projection_limits_before_wide_session_lookup(tmp_path):
    """The modern candidate plan must fetch wide session rows by candidate ID."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    plan = conn.execute(
        """
        EXPLAIN QUERY PLAN
        WITH latest_messages AS (
            SELECT mx.session_id, MAX(mx.timestamp) AS last_message_at
            FROM messages mx
            GROUP BY mx.session_id
        ), message_candidates_raw AS (
            SELECT lm.session_id AS id, lm.last_message_at
            FROM latest_messages lm
            WHERE lm.last_message_at IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM sessions sf
                  WHERE sf.id = lm.session_id
                    AND sf.source IS NOT NULL
              )
            ORDER BY lm.last_message_at DESC
            LIMIT ?
        ), message_candidates AS (
            SELECT mc.id, mc.last_message_at, s.started_at
            FROM (
                SELECT id, last_message_at FROM message_candidates_raw
            ) mc
            CROSS JOIN sessions s ON s.id = mc.id
            WHERE s.source IS NOT NULL
            ORDER BY COALESCE(mc.last_message_at, s.started_at) DESC,
                     s.started_at DESC
            LIMIT ?
        )
        SELECT s.id
        FROM message_candidates c
        CROSS JOIN sessions s ON s.id = c.id
        """,
        (160, 160),
    ).fetchall()
    conn.close()

    details = [row[3] for row in plan]
    assert any("SEARCH s USING INDEX sqlite_autoindex_sessions_1 (id=?)" in detail for detail in details)
    assert not any(detail == "SCAN s" for detail in details)


def test_modern_candidate_projection_full_query_has_no_wide_session_scan(monkeypatch, tmp_path):
    """The exact production CTE must keep empty candidates index-bounded."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    captured = []
    real_connect = sqlite3.connect

    class CapturingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, sql, params=()):
            if "latest_messages AS" in sql:
                captured.append((sql, tuple(params)))
            self._cursor.execute(sql, params)
            return self

        def fetchall(self):
            return self._cursor.fetchall()

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class CapturingConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            return CapturingCursor(super().cursor(*args, **kwargs))

    def capturing_connect(*args, **kwargs):
        kwargs["factory"] = CapturingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", capturing_connect)
    agent_sessions.read_importable_agent_session_rows(
        db, limit=20, exclude_sources=("webui",)
    )
    assert captured
    sql, params = captured[-1]

    conn = real_connect(str(db))
    details = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]
    conn.close()

    assert any(
        "SCAN s USING" in detail and "idx_sessions_started" in detail
        for detail in details
    )
    assert any("SEARCH s USING INTEGER PRIMARY KEY" in detail for detail in details)
    assert any("COVERING INDEX idx_sessions_source_id" in detail for detail in details)
    assert not any(detail == "SCAN s" for detail in details)


def test_modern_candidate_projection_does_not_require_effective_activity_index(tmp_path):
    """The modern CTE uses message/source/started indexes, not this legacy sort index."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    conn.execute("DROP INDEX idx_sessions_effective_activity")
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(
        db, limit=20, exclude_sources=("webui",)
    )

    assert rows
    assert rows[0]["id"] == "modern_resumed_old"


def test_modern_candidate_projection_falls_back_to_started_at_for_null_timestamp(tmp_path):
    """A message with a NULL timestamp still follows the sidebar recency contract."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    newest = time.time() + 120
    conn.execute(
        """
        INSERT INTO sessions
        (id, source, session_source, title, model, started_at,
         last_activity_at, message_count, parent_session_id, ended_at, end_reason)
        VALUES ('modern_null_timestamp', 'cli', 'cli', 'Null timestamp',
                'openai/gpt-5', ?, ?, 1, NULL, NULL, NULL)
        """,
        (newest, newest),
    )
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('modern_null_timestamp', 'user', 'hello', NULL)"
    )
    conn.commit()
    conn.close()

    rows = agent_sessions.read_importable_agent_session_rows(db, limit=1, exclude_sources=("webui",))

    assert rows[0]["id"] == "modern_null_timestamp"


def test_modern_empty_candidate_projection_orders_by_started_at(monkeypatch, tmp_path):
    """Empty rows use the same started_at order as the final projection."""
    db = tmp_path / "state.db"
    _make_modern_state_db(db)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE sessions SET source = 'webui'")
    base = time.time()
    for i in range(10):
        conn.execute(
            """
            INSERT INTO sessions
            (id, source, session_source, title, model, started_at,
             last_activity_at, message_count, parent_session_id, ended_at, end_reason)
            VALUES (?, 'cli', 'cli', ?, 'openai/gpt-5', ?, ?, 0, NULL, NULL, NULL)
            """,
            (f"modern_empty_{i:02d}", f"Empty {i}", base + i, base - i),
        )
    conn.commit()
    conn.close()

    # Keep the raw candidate rows visible so this test isolates SQL ordering;
    # normal projection intentionally hides empty standalone sessions.
    monkeypatch.setattr(agent_sessions, "_project_agent_session_rows", lambda rows: rows)
    monkeypatch.setattr(agent_sessions, "is_cli_session_row_visible", lambda row: True)
    rows = agent_sessions.read_importable_agent_session_rows(db, limit=1, exclude_sources=("webui",))

    assert rows[0]["id"] == "modern_empty_09"


def test_importable_agent_rows_zero_limit_skips_query_work(tmp_path):
    db = tmp_path / "state.db"
    _make_state_db(db, sessions=5, messages_per_session=1)

    assert agent_sessions.read_importable_agent_session_rows(db, limit=0, exclude_sources=("webui",)) == []
