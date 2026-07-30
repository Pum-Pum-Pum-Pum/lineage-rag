import json
import logging
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import conversations as conversation_route
from app.conversation.store import SqliteConversationStore


def _client(store: SqliteConversationStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[
        conversation_route.get_conversation_store
    ] = lambda: store
    return TestClient(app)


def test_request_id_and_defensive_headers_are_returned(
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        response = _client(store).get(
            "/conversations",
            headers={"X-Request-ID": "client-request-123"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "client-request-123"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )


def test_invalid_request_id_is_replaced_before_logging(
    caplog,
    tmp_path: Path,
) -> None:
    caplog.set_level(logging.INFO, logger="api_audit")
    unsafe = "bad\nrequest secret-token"
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        response = _client(store).get(
            "/conversations",
            headers={"X-Request-ID": unsafe},
        )

    request_id = response.headers["X-Request-ID"]
    assert request_id != unsafe
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        request_id,
    )
    assert unsafe not in caplog.text
    assert "secret-token" not in caplog.text


def test_audit_event_uses_route_template_and_excludes_request_body(
    caplog,
    tmp_path: Path,
) -> None:
    caplog.set_level(logging.INFO, logger="api_audit")
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        client = _client(store)
        created = client.post(
            "/conversations",
            json={"title": "private-title-must-not-be-logged"},
            headers={"X-Request-ID": "audit-123"},
        ).json()
        caplog.clear()

        response = client.get(
            f"/conversations/{created['conversation_id']}",
            headers={"X-Request-ID": "audit-456"},
        )

    assert response.status_code == 200
    event = json.loads(caplog.records[-1].message)
    assert event["event"] == "api_request_completed"
    assert event["request_id"] == "audit-456"
    assert event["method"] == "GET"
    assert event["route"] == "/conversations/{conversation_id}"
    assert event["status_code"] == 200
    assert event["duration_ms"] >= 0
    assert created["conversation_id"] not in caplog.text
    assert "private-title-must-not-be-logged" not in caplog.text
