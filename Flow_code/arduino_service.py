from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class SerialConnection(Protocol):
    def write(self, data: bytes) -> Any:
        ...

    def readline(self) -> bytes | str:
        ...

    def close(self) -> Any:
        ...


SerialFactory = Callable[[str, int, float], SerialConnection]


class ArduinoDispenseError(RuntimeError):
    """Raised when the Arduino reports or causes a dispense failure."""


class ArduinoBusyError(ArduinoDispenseError):
    """Raised when another dispense request is already running."""


class ArduinoTimeoutError(ArduinoDispenseError):
    """Raised when the Arduino does not report DONE before timeout."""


@dataclass(frozen=True)
class ArduinoDispenseResult:
    status: str
    mode: str
    code: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "mode": self.mode,
            "code": self.code,
            "source": self.source,
        }


class ArduinoDispenseService:
    def __init__(
        self,
        *,
        port: str | None = None,
        baud_rate: int = 9600,
        timeout_seconds: float = 60,
        serial_factory: SerialFactory | None = None,
        simulate_delay_seconds: float = 0.05,
    ) -> None:
        self.port = port.strip() if port else None
        self.baud_rate = baud_rate
        self.timeout_seconds = timeout_seconds
        self.serial_factory = serial_factory
        self.simulate_delay_seconds = simulate_delay_seconds
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "ArduinoDispenseService":
        return cls(
            port=os.getenv("ARDUINO_PORT"),
            baud_rate=int(os.getenv("ARDUINO_BAUD_RATE", "9600")),
            timeout_seconds=float(os.getenv("ARDUINO_TIMEOUT_SECONDS", "60")),
        )

    def dispense(self, *, code: str, source: str) -> dict[str, str]:
        normalized_code = code.strip()
        if not normalized_code:
            raise ValueError("Medicine code is required.")

        if not self._lock.acquire(blocking=False):
            raise ArduinoBusyError("Arduino is already dispensing medicine.")

        try:
            if not self.port:
                return self._simulate_dispense(normalized_code, source)
            return self._serial_dispense(normalized_code, source)
        finally:
            self._lock.release()

    def _simulate_dispense(self, code: str, source: str) -> dict[str, str]:
        if self.simulate_delay_seconds > 0:
            time.sleep(self.simulate_delay_seconds)
        return ArduinoDispenseResult(
            status="done",
            mode="simulated",
            code=code,
            source=source,
        ).as_dict()

    def _serial_dispense(self, code: str, source: str) -> dict[str, str]:
        connection = self._open_serial()
        try:
            connection.write(f"DISPENSE:{code}\n".encode("utf-8"))
            self._flush_if_supported(connection)

            deadline = time.monotonic() + self.timeout_seconds
            last_line = ""
            while time.monotonic() < deadline:
                raw_line = connection.readline()
                line = self._decode_line(raw_line)
                if not line:
                    time.sleep(0.02)
                    continue
                last_line = line
                if line == "DONE":
                    return ArduinoDispenseResult(
                        status="done",
                        mode="serial",
                        code=code,
                        source=source,
                    ).as_dict()
                if line.startswith("ERROR:"):
                    message = line.removeprefix("ERROR:").strip() or "Arduino returned an error."
                    raise ArduinoDispenseError(message)

            suffix = f" Last line: {last_line}" if last_line else ""
            raise ArduinoTimeoutError(f"Timed out waiting for DONE from Arduino.{suffix}")
        finally:
            connection.close()

    def _open_serial(self) -> SerialConnection:
        if not self.port:
            raise ArduinoDispenseError("Arduino port is not configured.")
        if self.serial_factory:
            return self.serial_factory(self.port, self.baud_rate, 1)

        try:
            import serial
        except ImportError as exc:
            raise ArduinoDispenseError("pyserial is required when ARDUINO_PORT is configured.") from exc

        return serial.Serial(self.port, self.baud_rate, timeout=1)

    @staticmethod
    def _decode_line(raw_line: bytes | str) -> str:
        if isinstance(raw_line, bytes):
            return raw_line.decode("utf-8", errors="replace").strip()
        return str(raw_line).strip()

    @staticmethod
    def _flush_if_supported(connection: SerialConnection) -> None:
        flush = getattr(connection, "flush", None)
        if callable(flush):
            flush()
