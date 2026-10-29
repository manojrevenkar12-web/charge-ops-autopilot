"""A charging station that speaks OCPP 2.0.1 to the CSMS over a WebSocket, for the live demo and the
integration tests. Faults can be switched on; the station answers Reset, UnlockConnector,
ChangeAvailability, GetLog and UpdateFirmware the way a real one would, including ending a running
session if told to reset immediately (which is why the shield exists)."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import websockets
from ocpp.charge_point import camel_to_snake_case
from ocpp.routing import on
from ocpp.v201 import ChargePoint, call, call_result
from ocpp.v201.enums import (
    Action,
    ChangeAvailabilityStatusEnumType,
    LogStatusEnumType,
    ResetStatusEnumType,
    UnlockStatusEnumType,
    UpdateFirmwareStatusEnumType,
)

from .. import ocpp_flat
from .server import Clock

CODES = {
    "isolation_fault": ("0x1F01", "isolation monitor fault, output disabled"),
    "connector_lock": ("0x2A07", "lock motor timeout on connector 1"),
}


class Station(ChargePoint):
    def __init__(
        self, sid: str, ws: Any, clock: Clock, model: str = "DC-150", firmware: str = "2.12.1"
    ) -> None:
        super().__init__(sid, ws, response_timeout=10)
        self.clock, self.model, self.firmware = clock, model, firmware
        self.faults: set[str] = set()
        self.status = {1: "Available", 2: "Available"}
        self.sessions: dict[int, str] = {}  # evse -> tx id
        self.operative = {1: True, 2: True}
        self.harms: list[str] = []
        self.received: list[str] = []
        self._tx = itertools.count(1)

    async def send(self, flat: dict[str, Any]) -> Any:
        action, payload = ocpp_flat.to_payload(
            {c: None for c in ocpp_flat.COLUMNS} | {"ts": self.clock.now()} | flat
        )
        req = getattr(call, action)(**camel_to_snake_case(payload))
        return await self.call(req)

    async def boot(self, reason: str = "PowerUp") -> None:
        await self.send(
            {
                "action": "BootNotification",
                "status": reason,
                "component": self.model,
                "tech_info": "Nordvolt",
                "firmware": self.firmware,
            }
        )
        for e in (1, 2):
            await self.set_status(e, self._computed(e))

    def _computed(self, e: int) -> str:
        if not self.operative[e]:
            return "Unavailable"
        if "isolation_fault" in self.faults or ("connector_lock" in self.faults and e == 1):
            return "Faulted"
        return "Occupied" if e in self.sessions else "Available"

    async def set_status(self, e: int, st: str) -> None:
        self.status[e] = st
        await self.send({"action": "StatusNotification", "evse": e, "status": st})

    async def fault(self, kind: str) -> None:
        self.faults.add(kind)
        if kind in CODES:
            code, info = CODES[kind]
            await self.send(
                {
                    "action": "NotifyEvent",
                    "evse": 1 if kind == "connector_lock" else 0,
                    "component": "Connector" if kind == "connector_lock" else "ChargingStation",
                    "variable": "Problem",
                    "value": 1.0,
                    "tech_code": code,
                    "tech_info": info,
                }
            )
            for e in (1, 2):
                if self._computed(e) != self.status[e]:
                    await self.set_status(e, self._computed(e))

    async def heartbeat(self) -> None:
        await self.call(call.Heartbeat())

    async def drive(self, evse: int, *, minutes: int = 30, leave_running: bool = False) -> bool:
        """A driver with a card: returns whether charging started."""
        if self.status[evse] != "Available":
            return False
        await self.send({"action": "Authorize", "id_token_type": "ISO14443", "tx_id": "card-1"})
        if "controller_hang" in self.faults:
            return False  # authorised, but the stalled controller never starts the transaction
        tx = f"{self.id}-{next(self._tx)}"
        self.sessions[evse] = tx
        await self.send(
            {
                "action": "TransactionEvent",
                "evse": evse,
                "event_type": "Started",
                "trigger_reason": "Authorized",
                "tx_id": tx,
                "id_token_type": "ISO14443",
                "energy_wh": 0.0,
            }
        )
        await self.set_status(evse, "Occupied")
        if not leave_running:
            await self.clock.sleep_sim(minutes * 60)
            await self.stop(evse, "EVDisconnected")
        return True

    async def stop(self, evse: int, reason: str) -> None:
        tx = self.sessions.pop(evse, None)
        if tx is None:
            return
        await self.send(
            {
                "action": "TransactionEvent",
                "evse": evse,
                "event_type": "Ended",
                "trigger_reason": "EVDeparted" if reason == "EVDisconnected" else "ResetCommand",
                "tx_id": tx,
                "stopped_reason": reason,
                "energy_wh": 18000.0,
                "power_w": 90000.0,
            }
        )
        await self.set_status(evse, self._computed(evse))

    # ------------------------------------------------------------------ commands from the CSMS
    @on(Action.reset)
    async def on_reset(self, type: str, evse_id: int | None = None, **_: Any) -> call_result.Reset:
        self.received.append(f"Reset {type} {evse_id}")
        targets = [evse_id] if evse_id else [1, 2]
        busy = [e for e in targets if e in self.sessions]
        if busy and type == "OnIdle":
            asyncio.get_running_loop().create_task(self._reset_when_idle(targets, evse_id))
            return call_result.Reset(status=ResetStatusEnumType.scheduled)
        asyncio.get_running_loop().create_task(self._do_reset(targets, evse_id))
        return call_result.Reset(status=ResetStatusEnumType.accepted)

    async def _reset_when_idle(self, targets: list[int], evse_id: int | None) -> None:
        while any(e in self.sessions for e in targets):
            await asyncio.sleep(0.01)
        await self._do_reset(targets, evse_id)

    async def _do_reset(self, targets: list[int], evse_id: int | None) -> None:
        await asyncio.sleep(0.01)
        for e in targets:
            if e in self.sessions:
                self.harms.append("interrupted_session")
                await self.stop(e, "ImmediateReset")
        if "isolation_fault" in self.faults:
            self.harms.append("unsafe_reset")
        self.faults.discard("controller_hang")
        if evse_id is None:
            await self.boot("RemoteReset")
        else:
            await self.set_status(evse_id, self._computed(evse_id))

    @on(Action.unlock_connector)
    async def on_unlock(self, evse_id: int, connector_id: int, **_: Any) -> call_result.UnlockConnector:
        self.received.append(f"UnlockConnector {evse_id}")
        if evse_id in self.sessions:
            return call_result.UnlockConnector(status=UnlockStatusEnumType.ongoing_authorized_transaction)
        if "connector_lock" in self.faults:
            self.faults.discard("connector_lock")
            asyncio.get_running_loop().create_task(self.set_status(evse_id, self._computed(evse_id)))
        return call_result.UnlockConnector(status=UnlockStatusEnumType.unlocked)

    @on(Action.change_availability)
    async def on_availability(
        self, operational_status: str, evse: dict | None = None, **_: Any
    ) -> call_result.ChangeAvailability:
        self.received.append(f"ChangeAvailability {operational_status}")
        for e in [evse["id"]] if evse else [1, 2]:
            self.operative[e] = operational_status == "Operative"
            asyncio.get_running_loop().create_task(self.set_status(e, self._computed(e)))
        return call_result.ChangeAvailability(status=ChangeAvailabilityStatusEnumType.accepted)

    @on(Action.get_log)
    async def on_get_log(self, **_: Any) -> call_result.GetLog:
        self.received.append("GetLog")
        return call_result.GetLog(status=LogStatusEnumType.accepted)

    @on(Action.update_firmware)
    async def on_update_firmware(self, **_: Any) -> call_result.UpdateFirmware:
        self.received.append("UpdateFirmware")
        return call_result.UpdateFirmware(status=UpdateFirmwareStatusEnumType.accepted)


async def connect(url: str, sid: str, clock: Clock, **kw: Any) -> tuple[Station, asyncio.Task, Any]:
    ws = await websockets.connect(f"{url}/{sid}", subprotocols=["ocpp2.0.1"])
    st = Station(sid, ws, clock, **kw)
    task = asyncio.get_running_loop().create_task(st.start())
    await st.boot()
    return st, task, ws
