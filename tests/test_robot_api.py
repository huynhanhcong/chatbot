from __future__ import annotations

from fastapi.testclient import TestClient

from Flow_code import api
from Flow_code.arduino_service import ArduinoDispenseService


def test_robot_serves_new_web_app() -> None:
    client = TestClient(api.app)

    response = client.get("/robot")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Mobile robot y tế" in response.text
    assert "robot-modern-2" in response.text
    assert 'id="homeScreen"' in response.text
    assert 'id="guideButton"' in response.text
    assert 'id="mapButton"' in response.text
    assert 'id="mapScreen"' in response.text
    assert 'id="doctorButton"' in response.text
    assert 'id="doctorSettingsButton"' in response.text
    assert 'id="excelUploadButton"' in response.text
    assert 'id="doorToggleButton"' in response.text
    assert 'id="rotateTrayButton"' in response.text
    assert 'id="doctorMedicineRows"' in response.text
    assert 'id="settingsPanel"' in response.text
    assert 'id="apiProviderSelect"' in response.text
    assert 'id="apiModelSelect"' in response.text
    assert 'id="voiceOverlay"' in response.text
    assert "/robot/static/app.js" in response.text


def test_robot_static_script_loads() -> None:
    client = TestClient(api.app)

    response = client.get("/robot/static/app.js")

    assert response.status_code == 200
    assert "showScreen" in response.text
    assert 'document.querySelector("#homeScreen")' in response.text
    assert 'document.querySelector("#guideButton")' in response.text
    assert 'document.querySelector("#mapButton")' in response.text
    assert 'document.querySelector("#doctorSettingsButton")' in response.text
    assert 'document.querySelector("#apiProviderSelect")' in response.text
    assert "API_MODEL_OPTIONS" in response.text
    assert 'document.querySelector("#doorToggleButton")' in response.text
    assert "robotDoctorSettings" in response.text
    assert "voiceOverlayEl.hidden = false" in response.text


def test_robot_static_styles_include_hub_and_settings_layout() -> None:
    client = TestClient(api.app)

    response = client.get("/robot/static/styles.css")

    assert response.status_code == 200
    assert ".robot-frame" in response.text
    assert ".medicine-cta" in response.text
    assert ".doctor-dashboard" in response.text
    assert ".doctor-medicine-row" in response.text
    assert ".settings-panel" in response.text
    assert ".voice-overlay" in response.text


def test_robot_legacy_route_serves_old_web_app() -> None:
    client = TestClient(api.app)

    response = client.get("/robot-legacy")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Mobile robot" in response.text
    assert "/robot-legacy/static/app.js" in response.text
    assert 'id="homeScreen"' in response.text
    assert 'id="doctorButton"' in response.text


def test_robot_dispense_manual_uses_simulated_service(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_arduino_service",
        ArduinoDispenseService(port=None, simulate_delay_seconds=0),
    )
    client = TestClient(api.app)

    response = client.post(
        "/robot/api/medicine/dispense",
        json={"code": "123456", "source": "manual"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "done",
        "mode": "simulated",
        "code": "123456",
        "source": "manual",
    }


def test_robot_dispense_qr_uses_simulated_service(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_arduino_service",
        ArduinoDispenseService(port=None, simulate_delay_seconds=0),
    )
    client = TestClient(api.app)

    response = client.post(
        "/robot/api/medicine/dispense",
        json={"code": "QR-ORDER-7", "source": "qr"},
    )

    assert response.status_code == 200
    assert response.json()["source"] == "qr"
    assert response.json()["code"] == "QR-ORDER-7"
