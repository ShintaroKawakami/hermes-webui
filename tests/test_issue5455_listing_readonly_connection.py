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
    assert calls == [(db.resolve().as_uri() + "?mode=ro", (), {"uri": True})]
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
