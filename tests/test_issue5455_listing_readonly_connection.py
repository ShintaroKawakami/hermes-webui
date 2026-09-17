"""Regression tests for the read-only #5455 session-listing connection."""

import sqlite3

import pytest

import api.agent_sessions as agent_sessions


def _make_db(path, *, indexed):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, title TEXT, model TEXT, message_count INTEGER,
            started_at REAL, source TEXT, session_source TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL
        );
        INSERT INTO sessions VALUES
            ('cli-1', 'Hello', 'gpt', 2, 1000.0, 'cli', 'cli');
        INSERT INTO messages (session_id, role, timestamp) VALUES
            ('cli-1', 'user', 1001.0), ('cli-1', 'assistant', 1002.0);
        """
    )
    if indexed:
        conn.execute(
            "CREATE INDEX idx_messages_session ON messages(session_id, timestamp)"
        )
    conn.commit()
    conn.close()


class _NoCommitConnection:
    """Proxy that makes a request-path commit an explicit test failure."""

    def __init__(self, connection):
        self._connection = connection

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._connection.row_factory = value

    def __getattr__(self, name):
        if name == "commit":
            raise AssertionError("session listing must not commit")
        return getattr(self._connection, name)


@pytest.mark.parametrize("indexed", [False, True])
def test_listing_reads_indexed_and_unindexed_databases_without_writes(
    tmp_path, monkeypatch, indexed
):
    db = tmp_path / ("indexed.db" if indexed else "unindexed.db")
    _make_db(db, indexed=indexed)
    calls = []
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        calls.append((database, args, kwargs))
        return _NoCommitConnection(real_connect(database, *args, **kwargs))

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", connect)

    rows = agent_sessions.read_importable_agent_session_rows(
        db, exclude_sources=None
    )

    assert [row["id"] for row in rows] == ["cli-1"]
    assert calls == [
        (
            db.resolve().as_uri() + "?mode=ro",
            (),
            {
                "uri": True,
                "timeout": agent_sessions.AGENT_STATE_DB_READ_TIMEOUT_SECONDS,
            },
        )
    ]
    read_only_conn = real_connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            read_only_conn.execute(
                "CREATE INDEX should_not_be_created ON messages(session_id)"
            )
    finally:
        read_only_conn.close()


def test_listing_encodes_url_significant_database_path(tmp_path, monkeypatch):
    db = tmp_path / "state #5455%2F?readonly.db"
    _make_db(db, indexed=False)
    calls = []
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        calls.append(database)
        return _NoCommitConnection(real_connect(database, *args, **kwargs))

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", connect)

    assert agent_sessions.read_importable_agent_session_rows(
        db, exclude_sources=None
    )[0]["id"] == "cli-1"
    assert calls == [db.resolve().as_uri() + "?mode=ro"]


def test_sidebar_metadata_and_orphan_probe_use_bounded_timeout(tmp_path, monkeypatch):
    """All other /api/sessions SQLite readers must share the same bound."""
    db = tmp_path / "state.db"
    _make_db(db, indexed=False)
    calls = []
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        calls.append((database, args, kwargs))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", connect)
    agent_sessions.read_session_lineage_metadata(db, ["cli-1"])

    import api.models as models

    monkeypatch.setattr(models, "_agent_state_db_path", lambda **_kwargs: db)
    assert models.agent_session_rows_existing(["cli-1"]) == frozenset({"cli-1"})

    assert len(calls) == 2
    assert all(
        call[2]["timeout"] == agent_sessions.AGENT_STATE_DB_READ_TIMEOUT_SECONDS
        for call in calls
    )


def test_listing_installs_and_clears_query_progress_deadline(tmp_path, monkeypatch):
    """The query itself must be bounded in addition to SQLite busy waiting."""
    db = tmp_path / "state.db"
    _make_db(db, indexed=True)
    events = []
    real_connect = sqlite3.connect

    class CapturingConnection(sqlite3.Connection):
        def set_progress_handler(self, callback, instruction_count):
            events.append((callback is not None, instruction_count))
            return super().set_progress_handler(callback, instruction_count)

    def connect(database, *args, **kwargs):
        kwargs["factory"] = CapturingConnection
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", connect)
    assert agent_sessions.read_importable_agent_session_rows(
        db, limit=20, exclude_sources=None
    )
    assert events[-2:] == [(True, 1000), (False, 0)]


def test_listing_clears_query_deadline_when_fetchall_is_interrupted(
    tmp_path, monkeypatch
):
    """The deadline must cover row production, not only cursor.execute()."""
    db = tmp_path / "state.db"
    _make_db(db, indexed=True)
    real_connect = sqlite3.connect

    class FetchallInterruptCursor:
        def __init__(self, cursor, connection):
            self._cursor = cursor
            self._connection = connection

        def execute(self, *args, **kwargs):
            self._cursor.execute(*args, **kwargs)
            return self

        def fetchall(self):
            callback = self._connection.progress_callback
            if callback is not None and callback():
                raise sqlite3.OperationalError("interrupted")
            return self._cursor.fetchall()

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class FetchallInterruptConnection:
        def __init__(self, connection):
            self._connection = connection
            self.progress_callback = None
            self.events = []

        @property
        def row_factory(self):
            return self._connection.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self._connection.row_factory = value

        def set_progress_handler(self, callback, instruction_count):
            self.progress_callback = callback
            self.events.append((callback is not None, instruction_count))

        def cursor(self):
            return FetchallInterruptCursor(self._connection.cursor(), self)

        def close(self):
            self._connection.close()

        def __getattr__(self, name):
            return getattr(self._connection, name)

    holder = []

    def connect(database, *args, **kwargs):
        connection = FetchallInterruptConnection(real_connect(database, *args, **kwargs))
        holder.append(connection)
        return connection

    monkeypatch.setattr(agent_sessions.sqlite3, "connect", connect)
    clock = iter((100.0, 100.3))
    monkeypatch.setattr(agent_sessions.time, "monotonic", lambda: next(clock))
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        agent_sessions.read_importable_agent_session_rows(
            db, limit=20, exclude_sources=None
        )

    assert holder[0].events[-2:] == [(True, 1000), (False, 0)]
