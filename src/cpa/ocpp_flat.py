"""One flat row per OCPP 2.0.1 message, shared by the simulator's fast mode and the live central
system. The live CSMS decodes real JSON payloads into these rows; the fast mode writes the rows
directly; `to_payload` turns a row back into the OCPP payload so tests can validate fast-mode rows
against the official JSON schemas and check the round trip.

Times are epoch seconds (UTC). `ts` is when the event happened at the charger, `received` when the
central system got it: a charger that lost its connection sends its queue later, and features must
only use what had been received at the time of a decision."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

SCHEMA: dict[str, Any] = {
    "ts": pl.Int64,
    "received": pl.Int64,
    "charger_id": pl.Utf8,
    "direction": pl.Utf8,  # "cp" charger to CSMS, "cs" CSMS to charger, "res" a response to a cs call
    "action": pl.Utf8,
    "evse": pl.Int8,  # 0 = the whole station
    "status": pl.Utf8,  # connector status, response status, firmware status, boot reason
    "id_token_type": pl.Utf8,
    "event_type": pl.Utf8,
    "trigger_reason": pl.Utf8,
    "stopped_reason": pl.Utf8,
    "tx_id": pl.Utf8,
    "energy_wh": pl.Float64,
    "power_w": pl.Float64,
    "component": pl.Utf8,
    "variable": pl.Utf8,
    "value": pl.Float64,
    "tech_code": pl.Utf8,
    "tech_info": pl.Utf8,
    "firmware": pl.Utf8,
    "message_id": pl.Utf8,
}
COLUMNS = list(SCHEMA)

CP_ACTIONS = {
    "BootNotification",
    "Heartbeat",
    "StatusNotification",
    "Authorize",
    "TransactionEvent",
    "NotifyEvent",
    "SecurityEventNotification",
    "FirmwareStatusNotification",
}
CS_ACTIONS = {
    "RequestStartTransaction",
    "Reset",
    "UnlockConnector",
    "ChangeAvailability",
    "UpdateFirmware",
    "GetLog",
    "TriggerMessage",
}


def empty() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def frame(rows: dict[str, list[Any]]) -> pl.DataFrame:
    n = len(rows["ts"])
    cols = {c: rows.get(c, [None] * n) for c in COLUMNS}
    return pl.DataFrame(cols, schema=SCHEMA)


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_payload(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The OCPP 2.0.1 request payload for a flat row (charger-to-CSMS and CSMS-to-charger calls)."""
    a = row["action"]
    t = _iso(row["ts"])
    if a == "BootNotification":
        return a, {
            "reason": row["status"] or "PowerUp",
            "chargingStation": {
                "model": row["component"] or "unknown",
                "vendorName": row["tech_info"] or "unknown",
                "firmwareVersion": row["firmware"] or "",
            },
        }
    if a == "Heartbeat":
        return a, {}
    if a == "StatusNotification":
        return a, {
            "timestamp": t,
            "connectorStatus": row["status"],
            "evseId": int(row["evse"]),
            "connectorId": 1,
        }
    if a == "Authorize":
        return a, {"idToken": {"idToken": row["tx_id"] or "token", "type": row["id_token_type"]}}
    if a == "TransactionEvent":
        p: dict[str, Any] = {
            "eventType": row["event_type"],
            "timestamp": t,
            "triggerReason": row["trigger_reason"],
            "seqNo": 0,
            "transactionInfo": {"transactionId": row["tx_id"]},
            "evse": {"id": int(row["evse"]), "connectorId": 1},
        }
        if row["stopped_reason"]:
            p["transactionInfo"]["stoppedReason"] = row["stopped_reason"]
        if row["id_token_type"]:
            p["idToken"] = {"idToken": "token", "type": row["id_token_type"]}
        if row["energy_wh"] is not None:
            sv = [{"value": row["energy_wh"], "measurand": "Energy.Active.Import.Register"}]
            if row["power_w"] is not None:
                sv.append({"value": row["power_w"], "measurand": "Power.Active.Import"})
            p["meterValue"] = [{"timestamp": t, "sampledValue": sv}]
        return a, p
    if a == "NotifyEvent":
        ev: dict[str, Any] = {
            "eventId": 1,
            "timestamp": t,
            "trigger": "Alerting" if row["tech_code"] else "Periodic",
            "actualValue": "" if row["value"] is None else f"{row['value']:g}",
            "eventNotificationType": "HardWiredNotification",
            "component": {"name": row["component"]} | ({"evse": {"id": int(row["evse"])}} if row["evse"] else {}),
            "variable": {"name": row["variable"]},
        }
        if row["tech_code"]:
            ev["techCode"] = row["tech_code"][:50]
        if row["tech_info"]:
            ev["techInfo"] = row["tech_info"][:500]
        return a, {"generatedAt": t, "seqNo": 0, "eventData": [ev]}
    if a == "SecurityEventNotification":
        return a, {"type": row["status"], "timestamp": t} | ({"techInfo": row["tech_info"]} if row["tech_info"] else {})
    if a == "FirmwareStatusNotification":
        return a, {"status": row["status"]}
    if a == "RequestStartTransaction":
        return a, {
            "idToken": {"idToken": "app-token", "type": "eMAID"},
            "remoteStartId": 1,
            "evseId": int(row["evse"]),
        }
    if a == "Reset":
        return a, {"type": "Immediate"} | ({"evseId": int(row["evse"])} if row["evse"] else {})
    if a == "UnlockConnector":
        return a, {"evseId": int(row["evse"]), "connectorId": 1}
    if a == "ChangeAvailability":
        return a, {"operationalStatus": row["status"]} | ({"evse": {"id": int(row["evse"])}} if row["evse"] else {})
    if a == "UpdateFirmware":
        return a, {
            "requestId": 1,
            "firmware": {"location": f"https://firmware.example/{row['firmware']}", "retrieveDateTime": t},
        }
    if a == "GetLog":
        return a, {
            "logType": "DiagnosticsLog",
            "requestId": 1,
            "log": {"remoteLocation": "https://logs.example/up"},
        }
    if a == "TriggerMessage":
        return a, {"requestedMessage": row["status"] or "StatusNotification"}
    raise ValueError(f"no payload for {a}")


def from_payload(
    action: str,
    payload: dict[str, Any],
    *,
    charger_id: str,
    received: int,
    direction: str = "cp",
    message_id: str | None = None,
) -> list[dict[str, Any]]:
    """Flat rows for a real OCPP 2.0.1 call (the live CSMS). NotifyEvent may carry several events."""
    base = {c: None for c in COLUMNS} | {
        "charger_id": charger_id,
        "received": received,
        "ts": received,
        "direction": direction,
        "action": action,
        "evse": 0,
        "message_id": message_id,
    }

    def ts(s: str | None) -> int:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()) if s else received

    if action == "BootNotification":
        cs = payload.get("chargingStation", {})
        return [
            base
            | {
                "status": payload.get("reason"),
                "component": cs.get("model"),
                "tech_info": cs.get("vendorName"),
                "firmware": cs.get("firmwareVersion"),
            }
        ]
    if action == "Heartbeat":
        return [base]
    if action == "StatusNotification":
        return [
            base
            | {
                "ts": ts(payload.get("timestamp")),
                "status": payload["connectorStatus"],
                "evse": payload["evseId"],
            }
        ]
    if action == "Authorize":
        return [base | {"id_token_type": payload["idToken"]["type"], "tx_id": payload["idToken"]["idToken"]}]
    if action == "TransactionEvent":
        info = payload.get("transactionInfo", {})
        row = base | {
            "ts": ts(payload.get("timestamp")),
            "event_type": payload["eventType"],
            "trigger_reason": payload.get("triggerReason"),
            "tx_id": info.get("transactionId"),
            "stopped_reason": info.get("stoppedReason"),
            "evse": (payload.get("evse") or {}).get("id", 0),
            "id_token_type": (payload.get("idToken") or {}).get("type"),
        }
        for mv in payload.get("meterValue", []):
            for sv in mv.get("sampledValue", []):
                if sv.get("measurand", "Energy.Active.Import.Register") == "Energy.Active.Import.Register":
                    row["energy_wh"] = float(sv["value"])
                elif sv.get("measurand") == "Power.Active.Import":
                    row["power_w"] = float(sv["value"])
        return [row]
    if action == "NotifyEvent":
        out = []
        for ev in payload.get("eventData", []):
            v = ev.get("actualValue", "")
            try:
                num = float(v)
            except ValueError:
                num = None
            out.append(
                base
                | {
                    "ts": ts(ev.get("timestamp")),
                    "component": ev["component"]["name"],
                    "variable": ev["variable"]["name"],
                    "value": num,
                    "evse": (ev["component"].get("evse") or {}).get("id", 0),
                    "tech_code": ev.get("techCode"),
                    "tech_info": ev.get("techInfo"),
                }
            )
        return out
    if action == "SecurityEventNotification":
        return [
            base
            | {
                "ts": ts(payload.get("timestamp")),
                "status": payload["type"],
                "tech_info": payload.get("techInfo"),
            }
        ]
    if action == "FirmwareStatusNotification":
        return [base | {"status": payload["status"]}]
    return [base]
