"""Today's process, as a policy: the NOC opens a ticket when a connector has been Faulted for a
while, a charger has been silent, or a driver complains; an operator works it during staffed hours
(usually a station reset first); if that does not help, the charger joins the dispatch queue and
technicians visit the oldest tickets first. Silent failures surface only through complaints.
Firmware problems reach the vendor once enough complaints pile up on one model."""

from __future__ import annotations

import math
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .. import settings
from .interface import Command, Observation, Stop
from .state import HOUR, FleetView

DAY = 86400


def unit(*key: Any) -> float:
    """A deterministic uniform number for a policy's own randomness (operator delays and the like)."""
    return zlib.crc32("|".join(map(str, key)).encode()) / 2**32


@dataclass
class Ticket:
    id: str
    charger_id: str
    opened: int
    reason: str
    state: str = "new"  # new -> worked -> recheck -> dispatch -> part -> closed
    due: int = 0
    actions: list[str] = field(default_factory=list)
    part: str | None = None
    part_eta: int | None = None
    visits: int = 0
    closed: int | None = None


class TicketBook:
    def __init__(self) -> None:
        self.open: dict[str, Ticket] = {}
        self.closed: list[Ticket] = []
        self.n = 0

    def get_or_open(self, cid: str, now: int, reason: str) -> Ticket:
        t = self.open.get(cid)
        if t is None:
            self.n += 1
            t = Ticket(f"INC-{self.n:05d}", cid, now, reason)
            self.open[cid] = t
        return t

    def close(self, cid: str, now: int) -> None:
        t = self.open.pop(cid, None)
        if t:
            t.state, t.closed = "closed", now
            self.closed.append(t)

    def all(self) -> list[Ticket]:
        return [*self.closed, *self.open.values()]


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def fifo_routes(
    view: FleetView,
    queue: list[Ticket],
    techs: pl.DataFrame,
    per_tech: int,
    parts: dict[str, list[str]] | None = None,
) -> dict[str, list[Stop]]:
    """Oldest tickets first, each to the nearest technician of the charger's country who has room."""
    by_country: dict[str, list[str]] = defaultdict(list)
    base: dict[str, tuple[float, float]] = {}
    for r in techs.iter_rows(named=True):
        by_country[r["country"]].append(r["tech_id"])
        base[r["tech_id"]] = (r["lat"], r["lon"])
    plan: dict[str, list[Stop]] = defaultdict(list)
    for t in sorted(queue, key=lambda t: t.opened):
        country = view.chargers[t.charger_id].country
        options = [x for x in by_country[country] if len(plan[x]) < per_tech]
        if not options:
            continue
        site = view.sites[view.chargers[t.charger_id].site_id]
        tech = min(options, key=lambda x: km((site["lat"], site["lon"]), base[x]))
        p = list((parts or {}).get(t.charger_id, []))
        if t.part and t.part_eta is not None:
            p.append(t.part)
        plan[tech].append(Stop(t.charger_id, p, t.id))
    return dict(plan)


class BaselinePolicy:
    name = "baseline"

    def __init__(self) -> None:
        self.cfg = settings.ops()["noc"]
        self.tickets = TicketBook()
        self.complaints: dict[str, list[int]] = defaultdict(list)
        self.escalations: dict[tuple[str, str], int] = {}
        self.rolled_back: set[tuple[str, str]] = set()
        self.decisions: list[dict[str, Any]] = []

    def start(self, reference: dict[str, pl.DataFrame], t0: int) -> None:
        self.view = FleetView(reference)
        self.techs = reference["technicians"]
        for v in self.view.chargers.values():
            v.last_seen = t0
        self.versions: dict[str, set[str]] = defaultdict(set)

    def staffed(self, t: int) -> bool:
        a, b = self.cfg["staffed_utc"]
        return a <= (t // HOUR) % 24 < b

    def on_hour(self, obs: Observation) -> list[Command]:
        now = obs.now
        v = self.view
        v.update(obs.messages, now)
        cmds: list[Command] = []
        # tickets from what the NOC sees
        for cid, cv in v.chargers.items():
            if cid in self.tickets.open:
                continue
            faulted = [
                e
                for e, (s, ts) in cv.status.items()
                if s == "Faulted" and now - ts >= self.cfg["faulted_ticket_after_hours"] * HOUR
            ]
            if faulted:
                self._open(cid, now, "faulted")
            elif now - cv.last_seen >= self.cfg["offline_ticket_after_hours"] * HOUR:
                self._open(cid, now, "offline")
        for r in obs.complaints.iter_rows(named=True):
            self.complaints[r["charger_id"]].append(r["ts"])
            if r["charger_id"] in v.chargers and r["charger_id"] not in self.tickets.open:
                self._open(r["charger_id"], now, "complaint")
        for r in obs.visits.iter_rows(named=True):
            self._visit(r, now)
        self._vendor(now, cmds)
        if not self.staffed(now):
            return cmds
        for t in list(self.tickets.open.values()):
            if t.state in ("new", "recheck") and now >= t.due:
                cmds.extend(self._work(t, now))
        return cmds

    def _open(self, cid: str, now: int, reason: str) -> None:
        t = self.tickets.get_or_open(cid, now, reason)
        a, b = self.cfg["pickup_delay_hours"]
        t.due = now + int((a + (b - a) * unit(cid, now, "pickup")) * HOUR)

    def _work(self, t: Ticket, now: int) -> list[Command]:
        v = self.view
        cid = t.charger_id
        if t.state == "recheck":
            if self._looks_fine(cid, since=t.due - self.cfg["recheck_after_hours"] * HOUR):
                self.tickets.close(cid, now)
            else:
                t.state = "dispatch"
            return []
        t.state = "worked"
        if not v.online(cid):
            t.state = "dispatch"
            return []
        codes = " ".join(f"{c} {i}" for _, c, i, _ in v.active_codes(cid))
        safety = any(k in codes for k in ("ISO", "1F01", "GROUND", "F03", "IMD", "insulation"))
        if safety and unit(cid, now, "spot") < self.cfg["recognise_safety_code"]:
            t.state = "dispatch"
            return []
        t.actions.append("station_reset")
        t.state = "recheck"
        t.due = now + self.cfg["recheck_after_hours"] * HOUR
        return [
            Command(
                cid,
                "station_reset",
                params={"type": "OnIdle"},
                reason=f"NOC first response to {t.reason}",
                incident=t.id,
                decided_by="operator",
            )
        ]

    def _looks_fine(self, cid: str, since: int) -> bool:
        if not self.view.online(cid) or self.view.faulted(cid):
            return False
        return not any(ts >= since for ts in self.complaints.get(cid, []))

    def _visit(self, r: dict[str, Any], now: int) -> None:
        t = self.tickets.open.get(r["charger_id"])
        if t is None:
            return
        t.visits += 1
        if r["outcome"] == "part_needed":
            t.state, t.part, t.part_eta = "part", r["part_needed"], r["part_eta"]
        elif r["outcome"] == "not_reached":
            t.state = "dispatch"
        else:
            self.tickets.close(r["charger_id"], now)

    def _vendor(self, now: int, cmds: list[Command]) -> None:
        cfg = self.cfg["vendor_escalation"]
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for cid, ts in self.complaints.items():
            cv = self.view.chargers.get(cid)
            # the NOC groups complaints about sessions that stop early (it sees those chargers' error stops)
            if cv and self.view.recent(cv.evcomm_fails, 72, now) >= 2:
                counts[(cv.model, cv.firmware)] += sum(1 for x in ts if x >= now - 7 * DAY)
        for key, n in counts.items():
            if n >= cfg["complaints_in_7_days"] and key not in self.escalations:
                self.escalations[key] = now + cfg["vendor_turnaround_days"] * DAY
        for (model, version), due in list(self.escalations.items()):
            if now >= due and (model, version) not in self.rolled_back and self.staffed(now):
                self.rolled_back.add((model, version))
                cmds.append(
                    Command(
                        "*",
                        "halt_rollout",
                        params={"model": model, "version": version},
                        reason="vendor advised",
                        decided_by="operator",
                    )
                )
                for cid, cv in self.view.chargers.items():
                    if cv.model == model and cv.firmware == version:
                        prev = self._previous(model, version)
                        if prev:
                            cmds.append(
                                Command(
                                    cid,
                                    "firmware",
                                    params={"version": prev},
                                    reason="vendor advised rollback",
                                    decided_by="operator",
                                )
                            )

    def _previous(self, model: str, version: str) -> str | None:
        seen = sorted({cv.firmware for cv in self.view.chargers.values() if cv.model == model} - {version})
        return seen[-1] if seen else None

    def plan_day(self, day_start: int) -> dict[str, list[Stop]]:
        queue = [
            t
            for t in self.tickets.open.values()
            if t.state == "dispatch" or (t.state == "part" and (t.part_eta or 0) <= day_start + 8 * HOUR)
        ]
        return fifo_routes(self.view, queue, self.techs, self.cfg["stops_per_technician"])

    def artifacts(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "incidents": [
                {
                    "id": t.id,
                    "charger_id": t.charger_id,
                    "opened": t.opened,
                    "closed": t.closed,
                    "state": t.state,
                    "resolution": t.reason,
                    "actions": t.actions,
                    "visits": t.visits,
                }
                for t in self.tickets.all()
            ]
        }

    def report(self) -> dict[str, Any]:
        return {
            "tickets": len(self.tickets.all()),
            "open": len(self.tickets.open),
            "escalations": {f"{m} {v}": d for (m, v), d in self.escalations.items()},
        }
