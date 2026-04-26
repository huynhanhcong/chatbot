from __future__ import annotations

from fastapi.testclient import TestClient

from Flow_code import api


def test_voice_route_serves_voice_app() -> None:
    client = TestClient(api.app)

    response = client.get("/voice")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Medical Voice Chat" in response.text


def test_voice_static_serves_assets() -> None:
    client = TestClient(api.app)

    response = client.get("/voice/static/app.js")

    assert response.status_code == 200
    assert "sendVoiceTurn" in response.text

