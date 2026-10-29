"""The shield: code that has the last word on every action, whoever proposed it (the planner, a
model, an operator approving a proposal). Rules and their sources live in config/ops.yaml; the
shield checks them against the CSMS's view of the fleet and its own record of what was done.

A verdict names every rule that blocked. Nothing here reads the simulator."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .. import settings
from .interface import Command
from .state import HOUR, FleetView

DAY = 86400


@dataclass
class Verdict:
    allowed: bool
    blocked_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Shield:
    def __init__(self, view: FleetView, enabled: bool = True) -> None:
        self.view = view
        self.enabled = enabled
        cfg = settings.ops()
        self.actions = set(cfg["actions"])
        self.rules = {r["id"]: r for r in cfg["shield"]["rules"]}
        self.history: dict[tuple[str, str], deque[int]] = defaultdict(lambda: deque(maxlen=20))
        self.network: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=500))
        self.site_busy: dict[str, list[tuple[int, int]]] = defaultdict(list)  # site -> (until, ...) of resets
        self.log: list[dict[str, Any]] = []

    def check(self, cmd: Command, now: int) -> Verdict:
        v = Verdict(True)
        if not self.enabled:
            return v
        a = cmd.action
        if a not in self.actions:
            v.blocked_by.append("action-allowlist")
            v.allowed = False
            return self._log(cmd, now, v)
        cv = self.view.chargers.get(cmd.charger_id)
        if cv is None and a != "halt_rollout":
            v.blocked_by.append("action-allowlist")
            v.allowed = False
            return self._log(cmd, now, v)
        for rid, r in self.rules.items():
            if a not in r["applies_to"] and "*" not in r["applies_to"]:
                continue
            ok = getattr(self, "_" + rid.replace("-", "_"))(cmd, now, r)
            if not ok:
                v.blocked_by.append(rid)
        v.allowed = not v.blocked_by
        return self._log(cmd, now, v)

    def record(self, cmd: Command, now: int) -> None:
        """An allowed command was sent."""
        self.history[(cmd.charger_id, cmd.action)].append(now)
        self.network[cmd.action].append(now)
        if cmd.action in ("evse_reset", "station_reset", "firmware"):
            cv = self.view.chargers.get(cmd.charger_id)
            if cv:
                self.site_busy[cv.site_id].append((now + 15 * 60, 0))

    def _log(self, cmd: Command, now: int, v: Verdict) -> Verdict:
        self.log.append(
            {
                "ts": now,
                "charger_id": cmd.charger_id,
                "action": cmd.action,
                "allowed": v.allowed,
                "blocked_by": ",".join(v.blocked_by),
                "decided_by": cmd.decided_by,
                "incident": cmd.incident,
            }
        )
        return v

    # ---------------------------------------------------------------- rules
    def _no_reset_on_insulation_fault(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        cv = self.view.chargers[cmd.charger_id]
        for _, code, info, _ in cv.codes:
            if code in r["block_if_codes"] or any(w.lower() in (info or "").lower() for w in r["block_if_text"]):
                return False
        return not cmd.params.get("safety_concern", False)

    def _no_reset_during_transaction(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        if cmd.params.get("type") == "OnIdle":
            return True  # the charger itself waits for running sessions to end
        evse = cmd.evse if cmd.action == "evse_reset" else 0
        return not self.view.busy(cmd.charger_id, evse)

    def _cooldown(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        h = self.history[(cmd.charger_id, cmd.action)]
        hours = r["hours"].get(cmd.action, 0)
        if h and now - h[-1] < hours * HOUR:
            return False
        return sum(1 for t in h if t >= now - DAY) < r["max_per_day"].get(cmd.action, 99)

    def _escalate_reset(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        if r.get("unless_station_level") and cmd.params.get("station_level"):
            return True
        return any(t >= now - DAY for t in self.history[(cmd.charger_id, r["requires_prior"])])

    def _site_blast_radius(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        cv = self.view.chargers[cmd.charger_id]
        busy = [u for u, _ in self.site_busy[cv.site_id] if u > now]
        self.site_busy[cv.site_id] = [(u, 0) for u in busy]
        return len(busy) < r["max_concurrent_per_site"]

    def _network_rate(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        lim = r["max_per_hour"].get(cmd.action)
        return lim is None or sum(1 for t in self.network[cmd.action] if t > now - HOUR) < lim

    def _firmware_staged(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        return bool(cmd.params.get("stage_approved"))  # the rollout plan (cohort.py) stages and approves

    def _action_allowlist(self, cmd: Command, now: int, r: dict[str, Any]) -> bool:
        return cmd.action in self.actions
