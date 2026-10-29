"""The fleet as the central system sees it, kept up to date from OCPP messages: connector status,
open transactions, last contact, firmware, and per-charger counters of what went wrong.

The shield checks preconditions against this view (never against the simulator), so a command is
only as safe as what the CSMS knows: a transaction the charger has not reported yet is invisible."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

HOUR = 3600


@dataclass
class ChargerView:
    charger_id: str
    model: str
    kind: str
    country: str
    site_id: str
    firmware: str
    status: dict[int, tuple[str, int]] = field(default_factory=lambda: {1: ("Available", 0), 2: ("Available", 0)})
    open_tx: dict[str, tuple[int, int]] = field(default_factory=dict)  # tx -> (evse, start)
    last_seen: int = 0
    last_boot: int = 0
    codes: deque[tuple[int, str, str, int]] = field(default_factory=lambda: deque(maxlen=20))  # ts, code, info, evse
    failed_starts: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    evcomm_fails: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=50))  # ts, evse
    good_sessions: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    rfid_auths: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    contactless_auths: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    starts: deque[int] = field(default_factory=lambda: deque(maxlen=400))
    watchdog: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    temps: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=30))
    operative: dict[int, bool] = field(default_factory=lambda: {0: True, 1: True, 2: True})


class FleetView:
    def __init__(self, reference: dict[str, pl.DataFrame]) -> None:
        ch = reference["chargers"]
        self.chargers: dict[str, ChargerView] = {
            r["charger_id"]: ChargerView(r["charger_id"], r["model"], r["kind"], r["country"], r["site_id"],
                                         r["firmware"]) for r in ch.iter_rows(named=True)}
        self.sites = {r["site_id"]: r for r in reference["sites"].iter_rows(named=True)}
        self.site_chargers: dict[str, list[str]] = defaultdict(list)
        for c in self.chargers.values():
            self.site_chargers[c.site_id].append(c.charger_id)
        self.now = 0
        self._pending: dict[str, list[tuple[int, int]]] = defaultdict(list)  # auth/remote starts awaiting a start

    def update(self, msgs: pl.DataFrame, now: int) -> None:
        self.now = now
        if msgs.is_empty():
            return
        seen = msgs.group_by("charger_id").agg(pl.col("received").max())
        for cid, t in seen.iter_rows():
            v = self.chargers.get(cid)
            if v:
                v.last_seen = max(v.last_seen, t)
        rest = msgs.filter(pl.col("action") != "Heartbeat")
        for r in rest.iter_rows(named=True):
            v = self.chargers.get(r["charger_id"])
            if v is None:
                continue
            a, d = r["action"], r["direction"]
            if a == "StatusNotification":
                v.status[r["evse"]] = (r["status"], r["ts"])
            elif a == "TransactionEvent":
                if r["event_type"] == "Started":
                    v.open_tx[r["tx_id"]] = (r["evse"], r["ts"])
                    v.starts.append(r["ts"])
                    pend = self._pending.get(v.charger_id)
                    if pend:
                        pend.pop(0)
                elif r["event_type"] == "Ended":
                    v.open_tx.pop(r["tx_id"], None)
                    e = r["energy_wh"] or 0.0
                    if r["trigger_reason"] == "EVCommunicationLost" or (r["stopped_reason"] == "Other" and e < 100):
                        v.evcomm_fails.append((r["ts"], r["evse"]))
                    elif r["stopped_reason"] in ("EVDisconnected", "Local", "Remote"):
                        v.good_sessions.append(r["ts"])
            elif a == "Authorize":
                self._pending[v.charger_id].append((r["ts"], 0))
                if r["id_token_type"] == "ISO14443":
                    v.rfid_auths.append(r["ts"])
                elif r["id_token_type"] == "Central":
                    v.contactless_auths.append(r["ts"])
            elif a == "RequestStartTransaction" and d == "res" and r["status"] == "Accepted":
                self._pending[v.charger_id].append((r["ts"], 0))
            elif a == "NotifyEvent":
                if r["component"] == "PowerModule" and r["variable"] == "Temperature":
                    v.temps.append((r["ts"], r["value"]))
                elif r["component"] == "Controller" and r["variable"] == "Watchdog":
                    v.watchdog.append(r["ts"])
                if r["tech_code"] or r["variable"] == "Problem":
                    v.codes.append((r["ts"], r["tech_code"] or "", r["tech_info"] or "", r["evse"]))
            elif a == "BootNotification":
                v.last_boot = r["ts"]
                v.open_tx.clear()
                if r["firmware"]:
                    v.firmware = r["firmware"]
            elif a == "ChangeAvailability" and d == "cs":
                v.operative[r["evse"]] = r["status"] == "Operative"
        # an authorisation or remote start with no transaction within ten minutes failed to start
        for cid, pend in self._pending.items():
            v = self.chargers[cid]
            while pend and pend[0][0] < now - 600:
                v.failed_starts.append(pend.pop(0)[0])

    # ---------------------------------------------------------------- queries
    def online(self, cid: str, now: int | None = None) -> bool:
        return (now or self.now) - self.chargers[cid].last_seen < 1.6 * HOUR

    def faulted(self, cid: str) -> list[int]:
        return [e for e, (s, _) in self.chargers[cid].status.items() if s == "Faulted"]

    def busy(self, cid: str, evse: int = 0) -> bool:
        """An open transaction on the EVSE (or on any EVSE when evse == 0), as far as the CSMS knows."""
        v = self.chargers[cid]
        return any(evse in (0, e) for e, _ in v.open_tx.values())

    def recent(self, dq: Any, hours: float, now: int | None = None) -> int:
        t = (now or self.now) - hours * HOUR
        return sum(1 for x in dq if (x[0] if isinstance(x, tuple) else x) >= t)

    def site_offline_share(self, site_id: str) -> float:
        ids = self.site_chargers[site_id]
        return float(np.mean([not self.online(c) for c in ids])) if ids else 0.0
