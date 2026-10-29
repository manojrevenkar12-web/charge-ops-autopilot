"""The central system (CSMS) side of OCPP 2.0.1 over WebSockets.

Stations connect to ws://host:port/<station id> with the `ocpp2.0.1` subprotocol. Every call a
station sends is recorded exactly as received (payload and flat row) before it is answered, and
every command the autopilot sends is recorded with the shield's verdict and the station's answer.
Payloads are validated against the official OCPP 2.0.1 JSON schemas by the `ocpp` library in both
directions."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import websockets
from ocpp.routing import on
from ocpp.v201 import ChargePoint, call, call_result
from ocpp.v201.enums import (
    Action,
    AuthorizationStatusEnumType,
    LogEnumType,
    OperationalStatusEnumType,
    RegistrationStatusEnumType,
    ResetEnumType,
)

from .. import ocpp_flat
from ..ops.interface import Command
from .store import MemoryStore, Store


class Clock:
    """Simulated time that runs `speed` times faster than real time (1 in production)."""

    def __init__(self, t0: int, speed: float = 1.0) -> None:
        self.t0, self.speed, self.real0 = t0, speed, time.monotonic()

    def now(self) -> int:
        return int(self.t0 + (time.monotonic() - self.real0) * self.speed)

    def iso(self) -> str:
        return datetime.fromtimestamp(self.now(), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    async def sleep_sim(self, seconds: float) -> None:
        await asyncio.sleep(seconds / self.speed)


class StationLink(ChargePoint):
    """One connected station, seen from the CSMS."""

    def __init__(self, sid: str, conn: Any, csms: CSMS) -> None:
        super().__init__(sid, conn, response_timeout=10)
        self.csms = csms

    async def route_message(self, raw_msg: str) -> None:
        msg = json.loads(raw_msg)
        if msg[0] == 2:  # a call from the station: record it before answering
            await self.csms.record(self.id, msg[2], msg[3], msg[1])
        await super().route_message(raw_msg)

    @on(Action.boot_notification)
    async def on_boot(self, **_: Any) -> call_result.BootNotification:
        return call_result.BootNotification(
            current_time=self.csms.clock.iso(),
            interval=self.csms.heartbeat_s,
            status=RegistrationStatusEnumType.accepted,
        )

    @on(Action.heartbeat)
    async def on_heartbeat(self, **_: Any) -> call_result.Heartbeat:
        return call_result.Heartbeat(current_time=self.csms.clock.iso())

    @on(Action.status_notification)
    async def on_status(self, **_: Any) -> call_result.StatusNotification:
        return call_result.StatusNotification()

    @on(Action.authorize)
    async def on_authorize(self, **_: Any) -> call_result.Authorize:
        return call_result.Authorize(id_token_info={"status": AuthorizationStatusEnumType.accepted})

    @on(Action.transaction_event)
    async def on_tx(self, **_: Any) -> call_result.TransactionEvent:
        return call_result.TransactionEvent()

    @on(Action.notify_event)
    async def on_event(self, **_: Any) -> call_result.NotifyEvent:
        return call_result.NotifyEvent()

    @on(Action.security_event_notification)
    async def on_security(self, **_: Any) -> call_result.SecurityEventNotification:
        return call_result.SecurityEventNotification()

    @on(Action.firmware_status_notification)
    async def on_fw(self, **_: Any) -> call_result.FirmwareStatusNotification:
        return call_result.FirmwareStatusNotification()

    @on(Action.log_status_notification)
    async def on_log(self, **_: Any) -> call_result.LogStatusNotification:
        return call_result.LogStatusNotification()


class CSMS:
    def __init__(self, clock: Clock, store: Store | None = None, heartbeat_s: int = 3600) -> None:
        self.clock = clock
        self.store = store or MemoryStore()
        self.heartbeat_s = heartbeat_s
        self.links: dict[str, StationLink] = {}
        self.server: Any = None

    async def record(
        self, sid: str, action: str, payload: dict[str, Any], message_id: str, direction: str = "cp"
    ) -> None:
        rows = ocpp_flat.from_payload(
            action,
            payload,
            charger_id=sid,
            received=self.clock.now(),
            direction=direction,
            message_id=message_id,
        )
        await self.store.append(rows, payload)

    async def _handler(self, ws: Any) -> None:
        if ws.subprotocol != "ocpp2.0.1":
            await ws.close(1002, "ocpp2.0.1 required")
            return
        sid = ws.request.path.strip("/")
        link = StationLink(sid, ws, self)
        self.links[sid] = link
        try:
            await link.start()
        except websockets.ConnectionClosed:
            pass
        finally:
            if self.links.get(sid) is link:
                del self.links[sid]

    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self.server = await websockets.serve(self._handler, host, port, subprotocols=["ocpp2.0.1"])
        return self.server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def send(self, cmd: Command, verdict: dict[str, Any]) -> str:
        """Send one autopilot command as an OCPP call; record it, the verdict, and the station's answer."""
        now = self.clock.now()
        rec = {
            "ts": now,
            "charger_id": cmd.charger_id,
            "action": cmd.action,
            "evse": cmd.evse,
            "params": cmd.params,
            "decided_by": cmd.decided_by,
            "incident": cmd.incident,
            "reason": cmd.reason,
            "shield": verdict,
            "result": "",
        }
        link = self.links.get(cmd.charger_id)
        payload = self._payload(cmd)
        if payload is None:
            rec["result"] = "NotAnOcppAction"
        elif link is None:
            rec["result"] = "NotConnected"
        else:
            action = type(payload).__name__
            await self.record(cmd.charger_id, action, ocpp_flat.to_payload(_flat_of(cmd, now))[1], f"c{now}", "cs")
            try:
                res = await link.call(payload, suppress=False)
                rec["result"] = str(getattr(res, "status", "Accepted"))
            except Exception as e:  # timeout or CallError: recorded, not retried blindly
                rec["result"] = f"Error:{type(e).__name__}"
            await self.store.append(
                [
                    {
                        **_flat_of(cmd, self.clock.now()),
                        "direction": "res",
                        "action": action,
                        "status": rec["result"],
                    }
                ],
                {"status": rec["result"]},
            )
        await self.store.command(rec)
        return rec["result"]

    @staticmethod
    def _payload(cmd: Command) -> Any:
        a, p = cmd.action, cmd.params
        kind = ResetEnumType.on_idle if p.get("type") == "OnIdle" else ResetEnumType.immediate
        if a == "evse_reset":
            return call.Reset(type=kind, evse_id=cmd.evse or 1)
        if a == "station_reset":
            return call.Reset(type=kind)
        if a == "unlock":
            return call.UnlockConnector(evse_id=cmd.evse or 1, connector_id=1)
        if a in ("set_inoperative", "set_operative"):
            st = (
                OperationalStatusEnumType.inoperative if a == "set_inoperative" else OperationalStatusEnumType.operative
            )
            return call.ChangeAvailability(operational_status=st, evse={"id": cmd.evse} if cmd.evse else None)
        if a == "get_log":
            return call.GetLog(
                log={"remote_location": "https://logs.example/upload"},
                log_type=LogEnumType.diagnostics_log,
                request_id=1,
            )
        if a == "firmware":
            return call.UpdateFirmware(
                request_id=1,
                firmware={
                    "location": f"https://firmware.example/{p.get('version')}",
                    "retrieve_date_time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
        return None


def _flat_of(cmd: Command, now: int) -> dict[str, Any]:
    action = {
        "evse_reset": "Reset",
        "station_reset": "Reset",
        "unlock": "UnlockConnector",
        "set_inoperative": "ChangeAvailability",
        "set_operative": "ChangeAvailability",
        "get_log": "GetLog",
        "firmware": "UpdateFirmware",
    }.get(cmd.action, cmd.action)
    st = {"set_inoperative": "Inoperative", "set_operative": "Operative"}.get(cmd.action)
    return {c: None for c in ocpp_flat.COLUMNS} | {
        "ts": now,
        "received": now,
        "charger_id": cmd.charger_id,
        "direction": "cs",
        "action": action,
        "evse": cmd.evse if cmd.action in ("evse_reset", "unlock", "set_inoperative", "set_operative") else 0,
        "status": st,
        "firmware": cmd.params.get("version"),
        "message_id": f"c{now}",
    }
