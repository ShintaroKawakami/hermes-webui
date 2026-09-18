"""Regression coverage for deleted sessions being resurrected by title work."""

import threading
from unittest.mock import MagicMock, patch


def test_background_title_update_skips_after_session_was_deleted():
    """A title request finishing after deletion must not save its stale object."""
    from api.streaming import _run_background_title_update

    session = MagicMock()
    session.session_id = "deleted-title-race"
    session.title = "Ping"
    session.llm_title_generated = False
    session.messages = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]

    events = []
    with (
        patch("api.streaming._aux_title_configured", return_value=True),
        patch(
            "api.streaming._generate_llm_session_title_via_aux",
            return_value=("Ping and Pong", "llm_aux", "Ping and Pong"),
        ),
        patch("api.streaming.get_session", side_effect=[session, KeyError("deleted")]),
        patch("api.streaming.SESSIONS", {}),
        patch("api.streaming.LOCK", threading.Lock()),
    ):
        _run_background_title_update(
            session_id=session.session_id,
            user_text="ping",
            assistant_text="pong",
            placeholder_title="Ping",
            put_event=lambda event, payload: events.append((event, payload)),
            agent=None,
        )

    session.save.assert_not_called()
    status_events = [payload for event, payload in events if event == "title_status"]
    assert status_events
    assert status_events[-1]["reason"] == "missing_session"


def test_delete_route_uses_per_session_lock_for_background_writers():
    """The delete route must share the title/stream writer lock."""
    import inspect
    import api.routes as routes

    source = inspect.getsource(routes._delete_session_for_webui)
    assert "with _get_session_agent_lock(sid):" in source
    assert "SESSIONS.pop(sid, None)" in source
