"""Fast-mode rows are real OCPP 2.0.1: every action the simulator writes encodes to a payload that the
official JSON schema accepts, and decoding it gives the row back."""

import asyncio
import json

import pytest
from ocpp.messages import Call, validate_payload

from cpa import ocpp_flat

ROWS = [
    {
        "action": "BootNotification",
        "status": "RemoteReset",
        "component": "DC-150",
        "tech_info": "Nordvolt",
        "firmware": "2.12.1",
    },
    {"action": "Heartbeat"},
    {"action": "StatusNotification", "evse": 2, "status": "Faulted"},
    {"action": "Authorize", "id_token_type": "ISO14443", "tx_id": "card"},
    {
        "action": "TransactionEvent",
        "evse": 1,
        "event_type": "Started",
        "trigger_reason": "Authorized",
        "tx_id": "t1",
        "id_token_type": "ISO14443",
        "energy_wh": 0.0,
    },
    {
        "action": "TransactionEvent",
        "evse": 1,
        "event_type": "Ended",
        "trigger_reason": "EVCommunicationLost",
        "tx_id": "t1",
        "stopped_reason": "Other",
        "energy_wh": 12.0,
    },
    {
        "action": "NotifyEvent",
        "evse": 1,
        "component": "Connector",
        "variable": "Problem",
        "value": 1.0,
        "tech_code": "0x2A07",
        "tech_info": "lock motor timeout on connector 1",
    },
    {"action": "NotifyEvent", "component": "PowerModule", "variable": "Temperature", "value": 41.5},
    {"action": "FirmwareStatusNotification", "status": "Installed"},
    {"action": "Reset", "direction": "cs", "evse": 1},
    {"action": "UnlockConnector", "direction": "cs", "evse": 2},
    {"action": "ChangeAvailability", "direction": "cs", "evse": 0, "status": "Inoperative"},
    {"action": "GetLog", "direction": "cs"},
    {"action": "RequestStartTransaction", "direction": "cs", "evse": 1},
]


def full(r):
    return (
        {c: None for c in ocpp_flat.COLUMNS}
        | {
            "ts": 1_780_000_000,
            "received": 1_780_000_000,
            "charger_id": "DE-0001",
            "direction": "cp",
            "evse": 0,
        }
        | r
    )


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['action']}-{i}" for i, r in enumerate(ROWS)])
def test_payload_passes_the_official_schema(row):
    action, payload = ocpp_flat.to_payload(full(row))
    asyncio.run(validate_payload(Call("1", action, payload), "2.0.1"))


@pytest.mark.parametrize(
    "row", [r for r in ROWS if r.get("direction", "cp") == "cp"], ids=lambda r: r["action"]
)
def test_round_trip(row):
    r = full(row)
    action, payload = ocpp_flat.to_payload(r)
    back = ocpp_flat.from_payload(
        action, json.loads(json.dumps(payload)), charger_id="DE-0001", received=r["received"]
    )[0]
    for k in (
        "action",
        "evse",
        "status",
        "event_type",
        "trigger_reason",
        "stopped_reason",
        "tech_code",
        "tech_info",
        "component",
        "variable",
        "value",
        "id_token_type",
        "firmware",
    ):
        if r.get(k) is not None:
            assert back[k] == r[k], (k, back[k], r[k])
