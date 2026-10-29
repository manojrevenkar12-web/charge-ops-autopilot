"""Runs a policy against the fleet, hour by hour, and records what operations could observe and
what really happened.

Within each hour, events (faults starting and ending, drivers arriving, sessions ending, commands
arriving, technicians arriving, reboots finishing) are processed in time order. At the end of the
hour the policy receives the OCPP messages, complaints, visit reports and logs that arrived, and
returns commands for the next hour; once a day it returns the technicians' routes."""

from __future__ import annotations

import copy
import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .. import ocpp_flat
from ..ops.interface import Command, Observation, Policy, Stop
from . import texts
from .world import DAY, HOUR, Episode, World, distance_km

REBOOT_S = {"evse_reset": 120, "station_reset": 300}
FAULTED_KINDS = {"isolation_fault"}  # whole station Faulted
CONNECTOR_FAULT_KINDS = {"connector_lock"}
SILENT = {"controller_hang", "rfid_reader", "payment_terminal", "cable_wear"}


@dataclass(eq=False)
class SessionState:
    tx: str
    evse: int
    start: int
    end: int
    energy_wh: float
    power_w: float
    auth: str


class Engine:
    def __init__(
        self, world: World, policy: Policy, start: int, end: int, run_id: str, *, shield_audit: bool = True
    ) -> None:
        self.w, self.policy, self.start, self.end, self.run_id = world, policy, start, end, run_id
        cfg = world.cfg
        self.cfg = cfg
        ch = world.chargers
        self.n = len(ch)
        self.cid = ch["charger_id"].to_list()
        self.index = {c: i for i, c in enumerate(self.cid)}
        self.model = ch["model"].to_list()
        self.kind = ch["kind"].to_list()
        self.kw = ch["kw"].to_numpy()
        self.site = ch["site"].to_numpy()
        self.country = ch["country"].to_list()
        countries = cfg["fleet"]["countries"]
        self.utc = np.array([countries[c]["utc"] for c in self.country])
        self.lang = [countries[c]["lang"] for c in self.country]
        sites = world.sites
        self.site_ll = list(zip(sites["lat"].to_list(), sites["lon"].to_list(), strict=True))
        self.charger_ll = [self.site_ll[s] for s in self.site]
        self.firmware = ch["firmware"].to_list()
        self.fw_history: list[list[str]] = [[f] for f in self.firmware]
        self.halted: set[tuple[str, str]] = set()
        self.bugs = {(r["model"], r["version"]): r["bug"] for r in cfg["firmware_rollouts"] if r.get("bug")}

        self.active: list[list[Episode]] = [[] for _ in range(self.n)]
        self.online = np.ones(self.n, bool)
        self.still_charges = np.ones(self.n, bool)
        self.operative = np.ones((self.n, 3), bool)
        self.reboot_until = np.zeros(self.n, np.int64)
        self.sessions: list[list[SessionState]] = [[] for _ in range(self.n)]
        self.status = [["", "Available", "Available"] for _ in range(self.n)]
        self.queue: list[list[dict[str, Any]]] = [[] for _ in range(self.n)]
        self.site_out = np.zeros(len(sites), bool)
        self.part_orders: dict[tuple[int, str], int] = {}
        self.events: list[tuple[int, int, str, Any]] = []
        self.seq = itertools.count()
        self.ep_ptr = 0
        self.out_ptr = 0
        self.outages = sorted(world.outages, key=lambda o: o[1])
        self.tx_counter = itertools.count(1)
        self.msg_counter = itertools.count(1)
        self.visit_counter = itertools.count(1)

        # rows for the current hour (columns), and everything else that operations could observe
        self.rows: dict[str, list[Any]] = {c: [] for c in ocpp_flat.COLUMNS}
        self.frames: list[pl.DataFrame] = []
        self.complaints: list[dict[str, Any]] = []
        self.visits: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.hour_complaints: list[dict[str, Any]] = []
        self.hour_visits: list[dict[str, Any]] = []
        self.hour_logs: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []

        # truth
        days = (end - start) // DAY + 1
        self.day0 = start
        self.usable_h = np.zeros((days, self.n))
        self.total_h = np.zeros((days, self.n))
        self.reported_h = np.zeros((days, self.n))
        self.usable = np.ones((self.n, 3))
        self.excluded = np.zeros(self.n, bool)
        self.sessions_truth: list[tuple] = []
        self.harms: list[dict[str, Any]] = []
        self.visit_truth: list[dict[str, Any]] = []
        self.episodes_seen: list[Episode] = []
        self.firmware_events: list[dict[str, Any]] = []
        self.pending_cmds: list[Command] = []
        self.carry: pl.DataFrame | None = None
        self.plans: dict[str, list[Stop]] = {}
        fs = cfg["field_service"]
        self.fs = fs
        self.techs = world.technicians.to_dicts()
        rev = cfg["drivers"]["revenue_per_kwh_eur"]
        self.rev_per_kwh = rev
        # chargers whose episodes started before this run began are carried over by fast-forwarding
        self._fast_forward()

    # ---------------------------------------------------------------- emission
    def emit(self, c: int, ts: int, action: str, direction: str = "cp", **kw: Any) -> None:
        row = {
            "ts": int(ts),
            "received": int(ts),
            "charger_id": self.cid[c],
            "direction": direction,
            "action": action,
            "evse": kw.pop("evse", 0),
            "message_id": f"m{next(self.msg_counter)}",
            **kw,
        }
        if direction == "cp" and not self.online[c]:
            if action == "TransactionEvent":
                self.queue[c].append(row)
            return
        for k in ocpp_flat.COLUMNS:
            self.rows[k].append(row.get(k))

    def push(self, t: float, kind: str, payload: Any) -> None:
        heapq.heappush(self.events, (int(t), next(self.seq), kind, payload))

    # ---------------------------------------------------------------- state helpers
    def _status_of(self, c: int, e: int, t: int) -> str:
        if not self.operative[c, e] or not self.operative[c, 0]:
            return "Unavailable"
        if self.reboot_until[c] > t:
            return "Unavailable"
        for ep in self.active[c]:
            if ep.kind in FAULTED_KINDS:
                return "Faulted"
            if ep.kind in CONNECTOR_FAULT_KINDS and ep.evse == e:
                return "Faulted"
            if ep.kind == "power_module" and t >= ep.onset_end and ep.relief_until <= t:
                return "Faulted"
        if any(s.evse == e and s.start <= t < s.end for s in self.sessions[c]):
            return "Occupied"
        return "Available"

    def _refresh(self, c: int, t: int) -> None:
        for e in (1, 2):
            st = self._status_of(c, e, t)
            if st != self.status[c][e]:
                self.status[c][e] = st
                if self.online[c]:
                    self.emit(c, t, "StatusNotification", evse=e, status=st)
        self._recompute_usable(c, t)

    def _bug(self, c: int) -> dict[str, Any] | None:
        return self.bugs.get((self.model[c], self.firmware[c]))

    def _recompute_usable(self, c: int, t: int) -> None:
        d = self.cfg["drivers"]["auth_mix"]
        faults = self.cfg["faults"]
        self.excluded[c] = bool(self.site_out[self.site[c]])
        for e in (1, 2):
            st = self._status_of(c, e, t)
            if st in ("Faulted", "Unavailable") or (not self.online[c] and not self.still_charges[c]):
                self.usable[c, e] = 0.0
                continue
            p = 1.0
            kinds = {ep.kind: ep for ep in self.active[c]}
            if "controller_hang" in kinds:
                p *= 1 - faults["controller_hang"]["effect"]["start_fails"]
            auth = 1.0
            if "rfid_reader" in kinds:
                auth -= d["rfid"]
            if "payment_terminal" in kinds and self.kind[c] != "AC":
                auth -= d["contactless"]
            p *= max(auth, 0.0)
            if "cable_wear" in kinds and kinds["cable_wear"].evse == e:
                p *= 1 - faults["cable_wear"]["effect"]["ev_comm_fails"]
            bug = self._bug(c)
            if bug:
                p *= 1 - bug["ev_comm_fails"] * bug["affected_ev_share"]
            self.usable[c, e] = p

    # ---------------------------------------------------------------- run
    def _fast_forward(self) -> None:
        """Bring the fleet to the run's start: firmware installed before it, episodes still open at it."""
        day_idx = (self.start - self.w.t0) // DAY
        for (m, v), days in self.w.rollout_day.items():
            for c in range(self.n):
                if days[c] < day_idx and self.model[c] == m:
                    self.fw_history[c].append(v)
                    self.firmware[c] = v
        while self.ep_ptr < len(self.w.episodes) and self.w.episodes[self.ep_ptr].start < self.start:
            self.ep_ptr += 1
        while self.out_ptr < len(self.outages) and self.outages[self.out_ptr][1] < self.start:
            self.out_ptr += 1

    def run(self) -> None:
        w = self.w
        ref = {
            "chargers": w.chargers.select(
                [
                    "charger_id",
                    "site_id",
                    "country",
                    "model",
                    "vendor",
                    "kind",
                    "kw",
                    "connectors",
                    "firmware",
                ]
            ),
            "sites": w.sites.select(["site_id", "country", "lat", "lon", "utc_offset", "motorway"]),
            "technicians": w.technicians,
        }
        self.policy.start(ref, self.start)
        for c in range(self.n):
            self._recompute_usable(c, self.start)
        for hs in range(self.start, self.end, HOUR):
            self.step(hs)

    def step(self, hs: int) -> None:
        he = hs + HOUR
        h = (hs - self.w.t0) // HOUR
        if (hs // HOUR) % 24 == 5:  # 05:00 UTC: the day's routes
            self._start_day(hs - 5 * HOUR)
        # commands decided last hour
        for cmd in self.pending_cmds:
            self.push(hs + cmd.offset_s, "cmd", cmd)
        self.pending_cmds = []
        # episodes and outages that start this hour
        eps = self.w.episodes
        while self.ep_ptr < len(eps) and eps[self.ep_ptr].start < he:
            ep = copy.copy(eps[self.ep_ptr])
            self.push(ep.start, "ep_start", ep)
            self.ep_ptr += 1
        while self.out_ptr < len(self.outages) and self.outages[self.out_ptr][1] < he:
            s, a, b = self.outages[self.out_ptr]
            self.push(a, "out_start", s)
            self.push(b, "out_end", s)
            self.out_ptr += 1
        self._firmware_installs(hs, h)
        # drivers
        counts, u = self.w.session_draws(h)
        for c in np.nonzero(counts)[0]:
            for k in range(counts[c]):
                self.push(hs + int(u[c, k, 0] * HOUR), "attempt", (int(c), u[c, k]))
        while self.events and self.events[0][0] < he:
            t, _, kind, payload = heapq.heappop(self.events)
            getattr(self, f"_on_{kind}")(t, payload)
        self._periodic(hs, h)
        # truth: an hour of connector availability
        di = (hs - self.day0) // DAY
        ok = ~self.excluded
        self.usable_h[di] += np.where(ok, self.usable[:, 1] + self.usable[:, 2], 0)
        self.total_h[di] += np.where(ok, 2, 0)
        rep = np.array(
            [
                (self.online[c] and self.status[c][1] in ("Available", "Occupied"))
                + (self.online[c] and self.status[c][2] in ("Available", "Occupied"))
                for c in range(self.n)
            ]
        )
        self.reported_h[di] += np.where(ok, rep, 0)
        # what operations sees: only what the CSMS has received by the end of the hour (a message stamped
        # just before the hour ends may arrive in the next one)
        frame = ocpp_flat.frame(self.rows)
        self.rows = {c: [] for c in ocpp_flat.COLUMNS}
        if self.carry is not None:
            frame = pl.concat([self.carry, frame]).sort("received")
        self.carry = frame.filter(pl.col("received") >= he)
        frame = frame.filter(pl.col("received") < he)
        self.frames.append(frame)
        obs = Observation(
            now=he,
            messages=frame,
            complaints=pl.DataFrame(self.hour_complaints, schema=COMPLAINT_SCHEMA),
            visits=pl.DataFrame(self.hour_visits, schema=VISIT_SCHEMA),
            logs=pl.DataFrame(self.hour_logs, schema=LOG_SCHEMA),
        )
        self.hour_complaints, self.hour_visits, self.hour_logs = [], [], []
        self.pending_cmds = list(self.policy.on_hour(obs))

    # ---------------------------------------------------------------- periodic messages
    def _periodic(self, hs: int, h: int) -> None:
        rows = []
        for c in np.nonzero(self.online)[0]:
            t = hs + (int(c) * 37) % HOUR
            rows.append((t, self.cid[c]))
        hb = pl.DataFrame(
            {"ts": [r[0] for r in rows], "charger_id": [r[1] for r in rows]},
            schema={"ts": pl.Int64, "charger_id": pl.Utf8},
        )
        hb = hb.with_columns(
            received=pl.col("ts"), direction=pl.lit("cp"), action=pl.lit("Heartbeat"), evse=pl.lit(0, pl.Int8)
        )
        tele_c = [c for c in np.nonzero(self.online)[0] if self.kind[c] != "AC" and (h + c) % 4 == 0]
        temps = []
        local = (h + self.utc) % 24
        g = self.w.hour_uniforms(h, self.n)
        for c in tele_c:
            base = 33 + 7 * math.sin((local[c] - 9) / 24 * 2 * math.pi) + 3 * (g[c, 0] - 0.5)
            for ep in self.active[c]:
                if ep.kind == "power_module":
                    prog = min(1.0, (hs - ep.start) / max(ep.onset_end - ep.start, 1))
                    base += self.cfg["faults"]["power_module"]["signals"]["temperature_rise_c"] * prog
            temps.append(round(base, 1))
        tele = pl.DataFrame(
            {
                "ts": [hs + 1800 + int(c) % 600 for c in tele_c],
                "charger_id": [self.cid[c] for c in tele_c],
                "value": temps,
            },
            schema={"ts": pl.Int64, "charger_id": pl.Utf8, "value": pl.Float64},
        )
        tele = tele.with_columns(
            received=pl.col("ts"),
            direction=pl.lit("cp"),
            action=pl.lit("NotifyEvent"),
            evse=pl.lit(0, pl.Int8),
            component=pl.lit("PowerModule"),
            variable=pl.lit("Temperature"),
        )
        for c in range(self.n):  # silent faults that leave traces now and then
            for ep in self.active[c]:
                if not self.online[c]:
                    continue
                sig = self.cfg["faults"][ep.kind].get("signals", {})
                if ep.kind == "controller_hang" and g[c, 1] < sig.get("watchdog_event", 0):
                    self.emit(
                        c,
                        hs + int(g[c, 2] * HOUR),
                        "NotifyEvent",
                        component="Controller",
                        variable="Watchdog",
                        value=1.0,
                        tech_code=None,
                        tech_info=self._vendor_info(ep, g[c, 3]),
                    )
                if ep.kind == "payment_terminal" and g[c, 4] < sig.get("terminal_event", 0):
                    self.emit(
                        c,
                        hs + int(g[c, 5] * HOUR),
                        "NotifyEvent",
                        component="PaymentTerminal",
                        variable="Problem",
                        value=1.0,
                        tech_code=self._code(ep, c),
                        tech_info=self._vendor_info(ep, g[c, 6]),
                    )
        extra = [hb, tele]
        cur = ocpp_flat.frame(self.rows) if self.rows["ts"] else ocpp_flat.empty()
        self.rows = {k: [] for k in ocpp_flat.COLUMNS}
        merged = pl.concat(
            [
                cur,
                *[
                    x.select(
                        [pl.col(k) if k in x.columns else pl.lit(None, v).alias(k) for k, v in ocpp_flat.SCHEMA.items()]
                    )
                    for x in extra
                ],
            ]
        )
        merged = merged.sort("received")
        for k in ocpp_flat.COLUMNS:
            self.rows[k] = merged[k].to_list()

    def _code(self, ep: Episode, c: int) -> str | None:
        return self.cfg["faults"][ep.kind].get("tech_codes", {}).get(self.model[c])

    def _vendor_info(self, ep: Episode, u: float) -> str:
        if ep.injection and u < 0.5:
            return texts.INJECTIONS[ep.id % len(texts.INJECTIONS)]
        opts = texts.VENDOR_INFO.get(ep.kind)
        return opts[int(u * len(opts))].format(evse=ep.evse) if opts else ""

    # ---------------------------------------------------------------- faults
    def _on_ep_start(self, t: int, ep: Episode) -> None:
        c = ep.charger
        if any(a.kind == ep.kind for a in self.active[c]):
            return  # already has this fault
        self.active[c].append(ep)
        self.episodes_seen.append(ep)
        f = self.cfg["faults"][ep.kind]
        if math.isfinite(ep.natural_end):
            self.push(ep.natural_end, "ep_end", (ep, "natural"))
        if ep.kind == "power_module":
            self.push(ep.onset_end, "pm_trip", ep)
        elif ep.kind in ("connector_lock", "isolation_fault"):
            self.emit(
                c,
                t + 1,
                "NotifyEvent",
                evse=ep.evse,
                component="Connector" if ep.evse else "ChargingStation",
                variable="Problem",
                value=1.0,
                tech_code=self._code(ep, c),
                tech_info=self._vendor_info(ep, ep.remedy_u[-2]),
            )
        elif ep.kind == "comm_loss":
            self.still_charges[c] = ep.remedy_u[-1] < f["effect"]["still_charges"]
            self._go_offline(c, t)
        self._refresh(c, t)

    def _on_pm_trip(self, t: int, ep: Episode) -> None:
        if ep not in self.active[ep.charger]:
            return
        c = ep.charger
        self.emit(
            c,
            t,
            "NotifyEvent",
            component="PowerModule",
            variable="Problem",
            value=1.0,
            tech_code=self._code(ep, c),
            tech_info=self._vendor_info(ep, ep.remedy_u[-2]),
        )
        self._interrupt(c, 0, t, "PowerLoss", cause="power_module")
        self._refresh(c, t)

    def _on_relief_end(self, t: int, ep: Episode) -> None:
        if ep in self.active[ep.charger] and ep.relief_until <= t:
            c = ep.charger
            self.emit(
                c,
                t,
                "NotifyEvent",
                component="PowerModule",
                variable="Problem",
                value=1.0,
                tech_code=self._code(ep, c),
                tech_info=self._vendor_info(ep, ep.remedy_u[-3]),
            )
            self._interrupt(c, 0, t, "PowerLoss", cause="power_module")
            self._refresh(c, t)

    def _on_ep_end(self, t: int, payload: tuple[Episode, str]) -> None:
        ep, reason = payload
        c = ep.charger
        if ep not in self.active[c]:
            return
        self.active[c].remove(ep)
        ep.end, ep.end_reason = t, reason
        if ep.kind == "comm_loss":
            self._go_online(c, t, boot=False)
        self._refresh(c, t)

    def _go_offline(self, c: int, t: int) -> None:
        self.online[c] = False

    def _go_online(self, c: int, t: int, boot: bool) -> None:
        if self.site_out[self.site[c]] or any(ep.kind == "comm_loss" for ep in self.active[c]):
            return
        self.online[c] = True
        self.still_charges[c] = True
        if boot:
            self.emit(
                c,
                t,
                "BootNotification",
                status="PowerUp",
                component=self.model[c],
                tech_info=self.w.chargers["vendor"][c],
                firmware=self.firmware[c],
            )
        for row in self.queue[c]:  # the offline queue arrives now, with the original timestamps
            row["received"] = t
            for k in ocpp_flat.COLUMNS:
                self.rows[k].append(row.get(k))
        self.queue[c] = []
        for e in (1, 2):
            st = self._status_of(c, e, t)
            self.status[c][e] = st
            self.emit(c, t + 1, "StatusNotification", evse=e, status=st)

    def _on_out_start(self, t: int, s: int) -> None:
        self.site_out[s] = True
        for c in np.nonzero(self.site == s)[0]:
            self._interrupt(int(c), 0, t, "PowerLoss", cause="grid_outage")
            self.online[c] = False
            self.still_charges[c] = False
            self._recompute_usable(int(c), t)

    def _on_out_end(self, t: int, s: int) -> None:
        self.site_out[s] = False
        for c in np.nonzero(self.site == s)[0]:
            self._go_online(int(c), t, boot=True)
            self._recompute_usable(int(c), t)

    # ---------------------------------------------------------------- drivers
    def _on_attempt(self, t: int, payload: tuple[int, np.ndarray], retry: bool = False) -> None:
        c, u = payload
        dcfg = self.cfg["drivers"]
        # the app shows Faulted/Unavailable connectors; drivers go to one that looks free
        free = [
            e
            for e in (1, 2)
            if self.status[c][e] == "Available"
            and not any(s.evse == e and s.start <= t < s.end for s in self.sessions[c])
        ]
        shown_down = [e for e in (1, 2) if self.status[c][e] in ("Faulted", "Unavailable")]
        if self.site_out[self.site[c]]:
            self._truth_session(t, c, 0, "", "outage", "grid_outage", 0.0)
            return
        if not free:
            outcome = "unavailable_seen" if shown_down else "busy"
            self._truth_session(t, c, 0, "", outcome, self._down_cause(c) if shown_down else "", 0.0)
            return
        e = free[int(u[11] * len(free))]
        mix = dcfg["auth_mix"]
        if u[1] < mix["rfid"]:
            auth = "rfid"
        elif u[1] < mix["rfid"] + mix["app"] or self.kind[c] == "AC":
            auth = "app"
        else:
            auth = "contactless"
        kinds = {ep.kind: ep for ep in self.active[c]}
        token = {"rfid": "ISO14443", "app": "eMAID", "contactless": "Central"}[auth]
        fail = None
        if not self.online[c] and (not self.still_charges[c] or auth == "app"):
            fail = ("fail_offline", "comm_loss")
        elif self.reboot_until[c] > t:
            fail = ("fail_reboot", "reboot")
        elif auth == "rfid" and "rfid_reader" in kinds:
            fail = ("fail_auth", "rfid_reader")
        elif auth == "contactless" and "payment_terminal" in kinds:
            fail = ("fail_auth", "payment_terminal")
        if fail is None:
            if auth == "app":
                self.emit(c, t, "RequestStartTransaction", direction="cs", evse=e, id_token_type=token)
                self.emit(c, t + 2, "RequestStartTransaction", direction="res", evse=e, status="Accepted")
            else:
                self.emit(c, t, "Authorize", id_token_type=token, status="Accepted")
            if "controller_hang" in kinds and u[5] < self.cfg["faults"]["controller_hang"]["effect"]["start_fails"]:
                fail = ("fail_start", "controller_hang")
        if fail:
            self._truth_session(t, c, e, auth, fail[0], fail[1], 0.0)
            self._after_failure(t, c, u, fail, auth, retry)
            return
        tx = f"{self.run_id[:2]}{next(self.tx_counter)}"
        self.emit(
            c,
            t + 20,
            "TransactionEvent",
            evse=e,
            event_type="Started",
            trigger_reason="RemoteStart" if auth == "app" else "Authorized",
            tx_id=tx,
            id_token_type=token,
            energy_wh=0.0,
        )
        bug = self._bug(c)
        ev_fail = (
            "cable_wear" in kinds
            and kinds["cable_wear"].evse == e
            and u[6] < self.cfg["faults"]["cable_wear"]["effect"]["ev_comm_fails"]
        )
        cause = "cable_wear"
        if not ev_fail and bug and u[2] < bug["affected_ev_share"] and u[7] < bug["ev_comm_fails"]:
            ev_fail, cause = True, "firmware_bug"
        if ev_fail:
            end = t + 60 + int(u[4] * 120)
            s = SessionState(tx, e, t + 20, end, float(int(u[3] * 40)), 0.0, auth)
            self.sessions[c].append(s)
            self.push(end, "sess_end", (c, s, "Other", "EVCommunicationLost"))
            self._refresh(c, t + 20)
            self._truth_session(t, c, e, auth, "fail_evcomm", cause, 0.0)
            self._after_failure(t, c, u, ("fail_evcomm", cause), auth, retry)
            return
        kwh = self.cfg["drivers"]["kwh_per_session"][self.kind[c]] * (0.5 + u[3])
        derate = 1.0
        if "power_module" in kinds:
            derate = (
                self.cfg["faults"]["power_module"]["effect"]["derate_to"] if t >= kinds["power_module"].start else 1
            )
        car_kw = 11.0 if self.kind[c] == "AC" else min(self.kw[c] * derate, 60 + 140 * u[4])
        power = car_kw * 1000
        dur = int(kwh / car_kw * HOUR * (0.9 + 0.2 * u[4]))
        s = SessionState(tx, e, t + 20, t + 20 + dur, kwh * 1000, power, auth)
        self.sessions[c].append(s)
        self.push(s.end, "sess_end", (c, s, "EVDisconnected", "EVDeparted"))
        self._refresh(c, t + 20)
        self._truth_session(t, c, e, auth, "ok", "", kwh)

    def _down_cause(self, c: int) -> str:
        for ep in self.active[c]:
            if ep.kind in ("isolation_fault", "connector_lock", "power_module"):
                return ep.kind
        return "unavailable"

    def _after_failure(self, t: int, c: int, u: np.ndarray, fail: tuple[str, str], auth: str, retry: bool) -> None:
        d = self.cfg["drivers"]
        if u[9] < d["complaint_per_failure"]:
            kind = {
                "rfid_reader": "rfid",
                "payment_terminal": "contactless",
                "controller_hang": "start",
                "cable_wear": "stops",
                "firmware_bug": "stops",
                "comm_loss": "offline",
            }.get(fail[1], "start")
            lang = self.lang[c] if u[10] > 0.3 else "en"
            row = {
                "ts": t + 600,
                "charger_id": self.cid[c],
                "lang": lang,
                "text": texts.complaint(kind, lang, self.cid[c], float(u[10])),
            }
            self.complaints.append(row)
            self.hour_complaints.append(row)
        if not retry and u[8] < d["retry_after_failure"]:
            u2 = np.roll(u, 3)
            u2[0], u2[5], u2[6], u2[7], u2[8] = u[0], u[8] / d["retry_after_failure"], u[9], u[10], 1.0
            self.push(t + 120, "retry", (c, u2))

    def _on_retry(self, t: int, payload: tuple[int, np.ndarray]) -> None:
        self._on_attempt(t, payload, retry=True)

    def _truth_session(self, t: int, c: int, e: int, auth: str, outcome: str, cause: str, kwh: float) -> None:
        self.sessions_truth.append((t, c, e, auth, outcome, cause, kwh))

    def _on_sess_end(self, t: int, payload: tuple[int, SessionState, str, str]) -> None:
        c, s, reason, trig = payload
        if s not in self.sessions[c] or s.end != t:
            return
        self.sessions[c].remove(s)
        self.emit(
            c,
            t,
            "TransactionEvent",
            evse=s.evse,
            event_type="Ended",
            trigger_reason=trig,
            tx_id=s.tx,
            stopped_reason=reason,
            energy_wh=round(s.energy_wh, 1),
            power_w=round(s.power_w, 0) if s.power_w else None,
        )
        self._refresh(c, t)

    def _interrupt(self, c: int, evse: int, t: int, reason: str, cause: str, by: str = "") -> int:
        n = 0
        for s in list(self.sessions[c]):
            if (evse == 0 or s.evse == evse) and s.start <= t < s.end:
                frac = (t - s.start) / max(s.end - s.start, 1)
                lost_kwh = s.energy_wh / 1000 * (1 - frac)
                s.energy_wh *= frac
                s.end = t
                self.sessions[c].remove(s)
                self.emit(
                    c,
                    t,
                    "TransactionEvent",
                    evse=s.evse,
                    event_type="Ended",
                    trigger_reason="AbnormalCondition" if reason != "ImmediateReset" else "ResetCommand",
                    tx_id=s.tx,
                    stopped_reason=reason,
                    energy_wh=round(s.energy_wh, 1),
                )
                if by:
                    self.harms.append(
                        {
                            "ts": t,
                            "charger_id": self.cid[c],
                            "harm": "interrupted_session",
                            "action": cause,
                            "decided_by": by,
                            "lost_kwh": lost_kwh,
                        }
                    )
                n += 1
        return n

    # ---------------------------------------------------------------- commands
    def _on_cmd(self, t: int, cmd: Command) -> None:
        c = self.index.get(cmd.charger_id)
        rec = {
            "ts": t,
            "charger_id": cmd.charger_id,
            "action": cmd.action,
            "evse": cmd.evse,
            "reason": cmd.reason,
            "incident": cmd.incident,
            "decided_by": cmd.decided_by,
            "params": str(cmd.params),
            "result": "",
        }
        self.commands.append(rec)
        if cmd.action == "halt_rollout":
            self.halted.add((cmd.params["model"], cmd.params["version"]))
            rec["result"] = "Halted"
            return
        if c is None:
            rec["result"] = "UnknownCharger"
            return
        ocpp_action = {
            "evse_reset": "Reset",
            "station_reset": "Reset",
            "unlock": "UnlockConnector",
            "set_inoperative": "ChangeAvailability",
            "set_operative": "ChangeAvailability",
            "firmware": "UpdateFirmware",
            "get_log": "GetLog",
        }[cmd.action]
        evse = cmd.evse if cmd.action in ("evse_reset", "unlock", "set_inoperative", "set_operative") else 0
        st = {"set_inoperative": "Inoperative", "set_operative": "Operative"}.get(cmd.action)
        self.emit(
            c,
            t,
            ocpp_action,
            direction="cs",
            evse=evse,
            status=st,
            firmware=cmd.params.get("version"),
            message_id=f"c{len(self.commands)}",
        )
        if not self.online[c]:
            rec["result"] = "NotConnected"
            self.emit(c, t + 30, ocpp_action, direction="res", evse=evse, status="NotConnected")
            return
        getattr(self, f"_do_{cmd.action}")(t, c, evse, cmd, rec)
        self.emit(c, t + 2, ocpp_action, direction="res", evse=evse, status=rec["result"])
        self._refresh(c, t + 3)

    def _remedy(self, t: int, c: int, action: str, evse: int, cmd: Command) -> list[str]:
        ended = []
        for ep in list(self.active[c]):
            f = self.cfg["faults"][ep.kind]
            if action in ("evse_reset", "unlock") and ep.evse not in (0, evse):
                continue
            if action == "unlock" and ep.kind != "connector_lock":
                continue
            p = f["remedies"].get(action, 0.0)
            u = ep.remedy_u[min(ep.attempts, len(ep.remedy_u) - 4)]
            ep.attempts += 1
            if u < p:
                self.push(t + REBOOT_S.get(action, 30), "ep_end", (ep, action))
                ended.append(ep.kind)
            elif ep.kind == "power_module" and t >= ep.onset_end and u < f["remedies"].get("relief", 0):
                ep.relief_until = t + int(f["remedies"]["relief_hours"] * HOUR)
                self.push(ep.relief_until, "relief_end", ep)
        return ended

    def _reset(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any], action: str) -> None:
        by = cmd.decided_by or self.policy.name
        if cmd.params.get("type") == "OnIdle":
            busy = [s.end for s in self.sessions[c] if (evse == 0 or s.evse == evse) and s.start <= t < s.end]
            if busy:  # OCPP OnIdle: the charger resets when the running sessions have ended
                later = Command(
                    cmd.charger_id,
                    cmd.action,
                    cmd.evse,
                    {"type": "Immediate"},
                    reason=cmd.reason,
                    incident=cmd.incident,
                    decided_by=by,
                )
                self.push(max(busy) + 30, "idle_reset", (c, evse, later, action))
                rec["result"] = "Scheduled"
                return
        self._interrupt(c, evse, t, "ImmediateReset", cause=action, by=by)
        for ep in self.active[c]:
            if self.cfg["faults"][ep.kind].get("unsafe_to_reset"):
                self.harms.append(
                    {
                        "ts": t,
                        "charger_id": self.cid[c],
                        "harm": "unsafe_reset",
                        "action": action,
                        "decided_by": by,
                        "lost_kwh": 0.0,
                    }
                )
        self._remedy(t, c, action, evse, cmd)
        rec["result"] = "Accepted"
        if action == "station_reset":
            self.reboot_until[c] = t + REBOOT_S[action]
            self.push(t + REBOOT_S[action], "boot", (c, "RemoteReset"))
        else:
            self.operative[c, evse] = False
            self.push(t + REBOOT_S[action], "evse_back", (c, evse))

    def _on_idle_reset(self, t: int, payload: tuple[int, int, Command, str]) -> None:
        c, evse, cmd, action = payload
        if not self.online[c]:
            return
        if any((evse == 0 or s.evse == evse) and s.start <= t < s.end for s in self.sessions[c]):
            self.push(max(s.end for s in self.sessions[c]) + 30, "idle_reset", payload)
            return
        self._reset(t, c, evse, cmd, {"result": ""}, action)
        self._refresh(c, t + 3)

    def _do_evse_reset(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        self._reset(t, c, evse or 1, cmd, rec, "evse_reset")

    def _do_station_reset(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        self._reset(t, c, 0, cmd, rec, "station_reset")

    def _on_evse_back(self, t: int, payload: tuple[int, int]) -> None:
        c, e = payload
        self.operative[c, e] = True
        self._refresh(c, t)

    def _on_boot(self, t: int, payload: tuple[int, str]) -> None:
        c, reason = payload
        self.emit(
            c,
            t,
            "BootNotification",
            status=reason,
            component=self.model[c],
            tech_info=self.w.chargers["vendor"][c],
            firmware=self.firmware[c],
        )
        for e in (1, 2):
            self.status[c][e] = ""
        self._refresh(c, t + 5)

    def _do_unlock(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        if any(s.evse == evse and s.start <= t < s.end for s in self.sessions[c]):
            rec["result"] = "OngoingAuthorizedTransaction"
            return
        had = any(ep.kind == "connector_lock" and ep.evse == evse for ep in self.active[c])
        ended = self._remedy(t, c, "unlock", evse, cmd)
        rec["result"] = "Unlocked" if (ended or not had) else "UnlockFailed"

    def _do_set_inoperative(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        self.operative[c, evse] = False
        rec["result"] = "Accepted"

    def _do_set_operative(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        self.operative[c, evse] = True
        rec["result"] = "Accepted"

    def _do_firmware(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        version = cmd.params["version"]
        busy_until = max([s.end for s in self.sessions[c]] + [t])
        rec["result"] = "Accepted"
        self.push(busy_until + 600, "fw_install", (c, version, "operator"))

    def _do_get_log(self, t: int, c: int, evse: int, cmd: Command, rec: dict[str, Any]) -> None:
        rec["result"] = "Accepted"
        self.push(t + 900, "log_upload", c)

    def _on_log_upload(self, t: int, c: int) -> None:
        kinds = [ep.kind for ep in self.active[c]]
        g = np.random.Generator(np.random.Philox(key=[self.w.seed, 3_000_003 * 10**10 + (t // 60) * 1000 + c]))
        row = {"ts": t, "charger_id": self.cid[c], "text": texts.diagnostic_log(kinds, g.random(8))}
        self.logs.append(row)
        self.hour_logs.append(row)

    # ---------------------------------------------------------------- firmware
    def _firmware_installs(self, hs: int, h: int) -> None:
        day = (hs - self.w.t0) // DAY
        local = (h + self.utc) % 24
        for (m, v), days in self.w.rollout_day.items():
            if (m, v) in self.halted:
                continue
            for c in np.nonzero((days == day) & (local == 2))[0]:
                c = int(c)
                if self.firmware[c] != v and v not in self.fw_history[c]:
                    self.push(hs + 300, "fw_install", (c, v, "rollout"))

    def _on_fw_install(self, t: int, payload: tuple[int, str, str]) -> None:
        c, version, by = payload
        if not self.online[c]:
            self.push(t + HOUR, "fw_install", payload)
            return
        if any(s.start <= t < s.end for s in self.sessions[c]):
            self.push(max(s.end for s in self.sessions[c]) + 60, "fw_install", payload)
            return
        for st in ("Downloading", "Downloaded", "Installing"):
            self.emit(c, t, "FirmwareStatusNotification", status=st)
        ok = (
            by == "rollout"
            or np.random.Generator(np.random.Philox(key=[self.w.seed, 4_000_003 * 10**10 + t * 1000 + c])).random()
            < self.cfg["rollback_remedies"]["firmware_rollback"]
        )
        if ok:
            self.firmware[c] = version
            if version not in self.fw_history[c]:
                self.fw_history[c].append(version)
        self.emit(c, t + 400, "FirmwareStatusNotification", status="Installed" if ok else "InstallationFailed")
        self.firmware_events.append({"ts": t, "charger_id": self.cid[c], "version": version, "by": by, "ok": ok})
        self.reboot_until[c] = t + 480
        self.push(t + 480, "boot", (c, "FirmwareUpdate"))
        self._refresh(c, t)

    # ---------------------------------------------------------------- field service
    def _start_day(self, day_start: int) -> None:
        fs = self.fs
        dow = ((day_start - self.w.t0) // DAY) % 7
        if dow not in fs["workdays"]:
            return  # technicians work Monday to Friday; nobody asks for a plan at the weekend
        plan = self.policy.plan_day(day_start)
        for tech in self.techs:
            stops = plan.get(tech["tech_id"], [])
            if not stops:
                continue
            t = day_start + (fs["workday_local"][0] - tech["utc_offset"]) * HOUR
            t_end = day_start + (fs["workday_local"][1] - tech["utc_offset"]) * HOUR
            pos = (tech["lat"], tech["lon"])
            km_total = 0.0
            for s in stops:
                c = self.index.get(s.charger_id)
                if c is None:
                    continue
                km = distance_km(pos, self.charger_ll[c]) * 1.25
                arrive = t + km / fs["speed_kmh"] * HOUR
                back = distance_km(self.charger_ll[c], (tech["lat"], tech["lon"])) * 1.25 / fs["speed_kmh"] * HOUR
                if arrive + fs["visit_hours"]["diagnose"] * HOUR + back > t_end:
                    self._visit_report(int(t_end), tech, s, "not_reached", None, None, "", 0.0)
                    continue
                km_total += km
                self.push(arrive, "visit", (tech, s, km))
                t = arrive + (fs["visit_hours"]["diagnose"] + fs["visit_hours"]["repair"] * 0.6) * HOUR
                pos = self.charger_ll[c]

    def _on_visit(self, t: int, payload: tuple[dict[str, Any], Stop, float]) -> None:
        tech, stop, km = payload
        c = self.index[stop.charger_id]
        fs = self.fs
        g = np.random.Generator(np.random.Philox(key=[self.w.seed, (5_000_003 + c) * 100_000 + t // DAY]))
        u = g.random(16)
        cost = fs["visit_cost_eur"] + km * fs["km_cost_eur"]
        if self.site_out[self.site[c]]:
            self._visit_report(t, tech, stop, "site_without_power", None, None, "site_without_power", cost)
            self.visit_truth.append(
                {
                    "ts": t,
                    "charger_id": stop.charger_id,
                    "necessary": False,
                    "fixed": "",
                    "wasted": "grid_outage",
                    "cost_eur": cost,
                    "incident": stop.incident,
                }
            )
            return
        eps = [ep for ep in self.active[c] if ep.start <= t]
        fixed, needed = [], []
        onsite = {"controller_hang": 0.99, "payment_terminal": 0.6, "rfid_reader": 0.45, "comm_loss": 0.6}
        for i, ep in enumerate(eps):
            part = self.cfg["faults"][ep.kind].get("part")
            if ep.kind in onsite and u[i] < onsite[ep.kind]:
                fixed.append(ep)
                continue
            if part is None:
                fixed.append(ep)
                continue
            ordered = self.part_orders.get((c, part))
            available = (
                (ordered is not None and ordered <= t)
                or (part in stop.parts and u[8 + i % 4] < 0.95)
                or u[4 + i % 4] < fs["van_stock"][part]
            )
            if available:
                fixed.append(ep)
                self.part_orders.pop((c, part), None)
            else:
                needed.append((ep, part))
                if (c, part) not in self.part_orders:
                    self.part_orders[(c, part)] = t + fs["part_lead_days"][part] * DAY
        repair_s = int((fs["visit_hours"]["diagnose"] + (fs["visit_hours"]["repair"] if fixed else 0)) * HOUR)
        for ep in fixed:
            self.push(t + repair_s, "ep_end", (ep, "repair"))
        if fixed and self.reboot_until[c] < t + repair_s:  # the technician power-cycles when done
            self.reboot_until[c] = t + repair_s
            self.push(t + repair_s, "boot", (c, "PowerUp"))
        primary = (fixed or [e for e, _ in needed] or [None])[0]
        cause = primary.kind if primary else "no_fault_found"
        if primary and u[12] < fs["note_cause_wrong"]:  # the technician writes down the wrong cause
            others = [k for k in texts.NOTES if k not in (cause, "site_without_power")]
            cause = others[int(u[13] * len(others))]
        part_needed = needed[0][1] if needed else None
        eta = self.part_orders.get((c, part_needed)) if part_needed else None
        outcome = "part_needed" if needed else ("fixed" if fixed else "no_fault_found")
        part_used = next(
            (self.cfg["faults"][ep.kind].get("part") for ep in fixed if self.cfg["faults"][ep.kind].get("part")),
            None,
        )
        if u[15] >= fs["structured_part_field"]:
            part_used = None  # the structured field is often left empty; the note still says what was done
        self._visit_report(
            t + repair_s,
            tech,
            stop,
            outcome,
            part_used,
            part_needed,
            cause,
            cost,
            eta=eta,
            evse=primary.evse if primary else 1,
            u=float(u[14]),
        )
        self.visit_truth.append(
            {
                "ts": t,
                "charger_id": stop.charger_id,
                "necessary": bool(eps),
                "fixed": ",".join(sorted(ep.kind for ep in fixed)),
                "wasted": "" if eps else "no_fault",
                "cost_eur": cost,
                "incident": stop.incident,
                "needed": ",".join(sorted(ep.kind for ep, _ in needed)),
            }
        )

    def _visit_report(
        self,
        t: int,
        tech: dict[str, Any],
        stop: Stop,
        outcome: str,
        part_used: str | None,
        part_needed: str | None,
        cause: str,
        cost: float,
        eta: int | None = None,
        evse: int = 1,
        u: float = 1.0,
    ) -> None:
        lang = tech["lang"] if u < self.fs["note_language_local"] else "en"
        note = texts.note(cause, lang, evse or 1, part_needed) if cause else ""
        row = {
            "visit_id": f"V{next(self.visit_counter)}",
            "tech_id": tech["tech_id"],
            "charger_id": stop.charger_id,
            "ts": int(t),
            "outcome": outcome,
            "part_used": part_used,
            "part_needed": part_needed,
            "part_eta": eta,
            "lang": lang,
            "note": note,
            "cost_eur": round(cost, 2),
            "incident": stop.incident,
        }
        self.visits.append(row)
        self.hour_visits.append(row)

    # ---------------------------------------------------------------- output
    def write(self, observed: Path, truth: Path) -> None:
        observed.mkdir(parents=True, exist_ok=True)
        truth.mkdir(parents=True, exist_ok=True)
        frames = [*self.frames, self.carry] if self.carry is not None else self.frames
        pl.concat(frames).write_parquet(observed / "messages.parquet", compression="zstd")
        pl.DataFrame(self.complaints, schema=COMPLAINT_SCHEMA).write_parquet(observed / "complaints.parquet")
        pl.DataFrame(self.visits, schema=VISIT_SCHEMA).write_parquet(observed / "visits.parquet")
        pl.DataFrame(self.logs, schema=LOG_SCHEMA).write_parquet(observed / "logs.parquet")
        pl.DataFrame(self.commands, schema=COMMAND_SCHEMA).write_parquet(observed / "commands.parquet")
        st = self.sessions_truth
        pl.DataFrame(
            {
                "ts": [s[0] for s in st],
                "charger_id": [self.cid[s[1]] for s in st],
                "evse": [s[2] for s in st],
                "auth": [s[3] for s in st],
                "outcome": [s[4] for s in st],
                "cause": [s[5] for s in st],
                "kwh": [s[6] for s in st],
            },
            schema={
                "ts": pl.Int64,
                "charger_id": pl.Utf8,
                "evse": pl.Int8,
                "auth": pl.Utf8,
                "outcome": pl.Utf8,
                "cause": pl.Utf8,
                "kwh": pl.Float64,
            },
        ).write_parquet(truth / "sessions.parquet")
        pl.DataFrame(
            [
                {
                    "episode": e.id,
                    "charger_id": self.cid[e.charger],
                    "kind": e.kind,
                    "evse": e.evse,
                    "start": e.start,
                    "onset_end": e.onset_end or None,
                    "end": None if math.isinf(e.end) else int(e.end),
                    "end_reason": e.end_reason,
                    "attempts": e.attempts,
                    "injection": e.injection,
                }
                for e in self.episodes_seen
            ],
            schema=EPISODE_SCHEMA,
        ).write_parquet(truth / "episodes.parquet")
        pl.DataFrame(self.harms, schema=HARM_SCHEMA).write_parquet(truth / "harms.parquet")
        pl.DataFrame(self.visit_truth, schema=VISIT_TRUTH_SCHEMA).write_parquet(truth / "visits.parquet")
        pl.DataFrame(
            self.firmware_events,
            schema={
                "ts": pl.Int64,
                "charger_id": pl.Utf8,
                "version": pl.Utf8,
                "by": pl.Utf8,
                "ok": pl.Boolean,
            },
        ).write_parquet(truth / "firmware.parquet")
        days = self.usable_h.shape[0]
        pl.DataFrame(
            {
                "day": np.repeat(np.arange(days), self.n),
                "charger_id": self.cid * days,
                "usable_h": self.usable_h.ravel(),
                "total_h": self.total_h.ravel(),
                "reported_h": self.reported_h.ravel(),
            }
        ).filter(pl.col("total_h") > 0).write_parquet(truth / "availability.parquet")


COMPLAINT_SCHEMA = {"ts": pl.Int64, "charger_id": pl.Utf8, "lang": pl.Utf8, "text": pl.Utf8}
VISIT_SCHEMA = {
    "visit_id": pl.Utf8,
    "tech_id": pl.Utf8,
    "charger_id": pl.Utf8,
    "ts": pl.Int64,
    "outcome": pl.Utf8,
    "part_used": pl.Utf8,
    "part_needed": pl.Utf8,
    "part_eta": pl.Int64,
    "lang": pl.Utf8,
    "note": pl.Utf8,
    "cost_eur": pl.Float64,
    "incident": pl.Utf8,
}
LOG_SCHEMA = {"ts": pl.Int64, "charger_id": pl.Utf8, "text": pl.Utf8}
COMMAND_SCHEMA = {
    "ts": pl.Int64,
    "charger_id": pl.Utf8,
    "action": pl.Utf8,
    "evse": pl.Int64,
    "reason": pl.Utf8,
    "incident": pl.Utf8,
    "decided_by": pl.Utf8,
    "params": pl.Utf8,
    "result": pl.Utf8,
}
EPISODE_SCHEMA = {
    "episode": pl.Int64,
    "charger_id": pl.Utf8,
    "kind": pl.Utf8,
    "evse": pl.Int64,
    "start": pl.Int64,
    "onset_end": pl.Int64,
    "end": pl.Int64,
    "end_reason": pl.Utf8,
    "attempts": pl.Int64,
    "injection": pl.Boolean,
}
HARM_SCHEMA = {
    "ts": pl.Int64,
    "charger_id": pl.Utf8,
    "harm": pl.Utf8,
    "action": pl.Utf8,
    "decided_by": pl.Utf8,
    "lost_kwh": pl.Float64,
}
VISIT_TRUTH_SCHEMA = {
    "ts": pl.Int64,
    "charger_id": pl.Utf8,
    "necessary": pl.Boolean,
    "fixed": pl.Utf8,
    "wasted": pl.Utf8,
    "cost_eur": pl.Float64,
    "incident": pl.Utf8,
    "needed": pl.Utf8,
}
_ = defaultdict
