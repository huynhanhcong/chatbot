from __future__ import annotations

import threading

import pytest

from Flow_code.arduino_service import (
    ArduinoBusyError,
    ArduinoDispenseError,
    ArduinoDispenseService,
    ArduinoTimeoutError,
)


class FakeSerial:
    def __init__(self, lines: list[bytes | str]) -> None:
        self.lines = list(lines)
        self.writes: list[bytes] = []
        self.closed = False
        self.flushed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        self.flushed = True

    def readline(self) -> bytes | str:
        if not self.lines:
            return b""
        return self.lines.pop(0)

    def close(self) -> None:
        self.closed = True


def test_simulated_dispense_without_port_returns_done() -> None:
    service = ArduinoDispenseService(port=None, simulate_delay_seconds=0)

    result = service.dispense(code=" 12345 ", source="manual")

    assert result == {
        "status": "done",
        "mode": "simulated",
        "code": "12345",
        "source": "manual",
    }


def test_serial_dispense_sends_command_and_waits_for_done() -> None:
    fake = FakeSerial([b"READY\n", b"DONE\n"])
    service = ArduinoDispenseService(
        port="COM3",
        timeout_seconds=0.2,
        serial_factory=lambda port, baud, timeout: fake,
    )

    result = service.dispense(code="ABC123", source="qr")

    assert result["status"] == "done"
    assert result["mode"] == "serial"
    assert fake.writes == [b"DISPENSE:ABC123\n"]
    assert fake.flushed is True
    assert fake.closed is True


def test_serial_dispense_raises_arduino_error_line() -> None:
    fake = FakeSerial([b"ERROR:door blocked\n"])
    service = ArduinoDispenseService(
        port="COM3",
        timeout_seconds=0.2,
        serial_factory=lambda port, baud, timeout: fake,
    )

    with pytest.raises(ArduinoDispenseError, match="door blocked"):
        service.dispense(code="ABC123", source="manual")

    assert fake.closed is True


def test_serial_dispense_times_out_without_done() -> None:
    fake = FakeSerial([])
    service = ArduinoDispenseService(
        port="COM3",
        timeout_seconds=0.03,
        serial_factory=lambda port, baud, timeout: fake,
    )

    with pytest.raises(ArduinoTimeoutError, match="Timed out"):
        service.dispense(code="ABC123", source="manual")

    assert fake.closed is True


def test_dispense_rejects_concurrent_request() -> None:
    started = threading.Event()
    release = threading.Event()
    fake = FakeSerial([])

    def serial_factory(port: str, baud: int, timeout: float) -> FakeSerial:
        return fake

    def blocking_readline() -> bytes:
        started.set()
        release.wait(timeout=1)
        return b"DONE\n"

    fake.readline = blocking_readline  # type: ignore[method-assign]
    service = ArduinoDispenseService(
        port="COM3",
        timeout_seconds=1,
        serial_factory=serial_factory,
    )
    result: dict[str, str] = {}

    thread = threading.Thread(
        target=lambda: result.update(service.dispense(code="FIRST", source="manual")),
    )
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(ArduinoBusyError):
        service.dispense(code="SECOND", source="manual")

    release.set()
    thread.join(timeout=1)
    assert result["status"] == "done"
