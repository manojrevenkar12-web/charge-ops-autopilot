"""The autopilot: an operations policy that runs the fleet from what the CSMS sees.

Every hour it updates the fleet view and the hourly signals, computes features for every charger,
opens incidents where symptoms appear (including silent failures nobody reports), and decides the
next step for each open incident: the diagnosis model and an LLM reading of the text evidence give
the likely remedy; the planner picks the remote action most likely to work, or a visit with the
right parts; the shield checks it; the autonomy ladder says whether it runs now or waits for an
operator. Once a day it plans technicians' routes, adds preventive visits from the failure-risk
model, and looks for firmware cohorts that fail more than their peers. Once a week it retrains,
and a new model only decides after passing the offline gate and a shadow period.

Flags switch parts off for the comparisons (assist = every action waits for an operator, and so on)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .. import kb, settings
from ..gateway.core import Gateway, GatewayError
from ..ml import features, signals
from ..ml.incidents import QUIET_HOURS
from .autonomy import Ladder
from .baseline import fifo_routes, unit
from .cohort import CohortAlert
from .cohort import detect as detect_cohorts
from .dispatch import Job, plan_country
from .interface import Command, Observation, Stop
from .shield import Shield
from .state import HOUR, FleetView

DAY = 86400
REMOTE = ("unlock", "evse_reset", "station_reset")


@dataclass
class Incident:
    id: str
    charger_id: str
    opened: int
    onset_hour: int
    last_symptom_hour: int
    symptoms: set[str] = field(default_factory=set)
    state: str = "open"  # open -> waiting -> dispatch -> part -> resolved
    tried: list[tuple[str, int, int]] = field(default_factory=list)  # action, at, evse
    pending: dict[str, Any] | None = None  # an action awaiting its outcome
    proposal: dict[str, Any] | None = None  # waiting for an operator
    probs: dict[str, float] = field(default_factory=dict)
    shadow: dict[str, float] = field(default_factory=dict)
    reading: dict[str, Any] | None = None
    log_requested: bool = False
    log_text: str | None = None
    parts: list[str] = field(default_factory=list)
    part_eta: int | None = None
    inoperative: bool = False
    value_per_hour: float = 0.0
    wait_until: int = 0
    closed: int | None = None
    resolution: str = ""
    decisions: list[dict[str, Any]] = field(default_factory=list)
    preventive: bool = False
    good_since_symptom: bool = False
    first_probs: dict[str, float] = field(default_factory=dict)
    part_used: str | None = None
    watch_since: int = 0


class Autopilot:
    def __init__(self, name: str = "autopilot", *, autonomy: bool = True, vrp: bool = True, predictive: bool = True,
                 cohorts: bool = True, shield: bool = True, guards: bool = True, reader: bool = True,
                 models_from: Any = None, retrain: bool = True) -> None:
        self.name = name
        self.flags = {"autonomy": autonomy, "vrp": vrp, "predictive": predictive, "cohorts": cohorts,
                      "shield": shield, "guards": guards, "reader": reader, "retrain": retrain}
        self.ops = settings.ops()
        self.pcfg = self.ops["planner"]
        self.noc = self.ops["noc"]
        self.gateway = Gateway(guards=guards)
        self.ladder = Ladder()
        self.models_from = models_from  # a ModelOps instance (mlops/lifecycle.py)
        self.incidents: dict[str, Incident] = {}
        self.closed: list[Incident] = []
        self.n = 0
        self.beliefs: dict[tuple[str, str], list[float]] = {}
        self.alerts: list[CohortAlert] = []
        self.approvals: list[tuple[int, Command, str]] = []  # due, command, incident
        self.events: list[dict[str, Any]] = []
        self.sig: pl.DataFrame | None = None
        self.complaints: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.logs: dict[str, str] = {}
        self.feature_log: list[pl.DataFrame] = []
        self.keep_features = False
        self.preventive_today = 0
        self.pending_operative: list[str] = []

    # ---------------------------------------------------------------- set-up
    def start(self, reference: dict[str, pl.DataFrame], t0: int) -> None:
        self.view = FleetView(reference)
        for v in self.view.chargers.values():
            v.last_seen = t0
        self.shield = Shield(self.view, enabled=self.flags["shield"])
        self.ref = reference["chargers"]
        self.sites = reference["sites"]
        self.techs = reference["technicians"]
        self.t0 = t0
        if self.models_from is not None:
            self.models_from.attach(self)

    # ---------------------------------------------------------------- hourly loop
    def on_hour(self, obs: Observation) -> list[Command]:
        now = obs.now
        hour = now // HOUR - 1  # the hour that just ended
        self.view.update(obs.messages, now)
        self.gateway.tick(now)
        for r in obs.complaints.iter_rows(named=True):
            self.complaints[r["charger_id"]].append((r["ts"], r["text"]))
        for r in obs.logs.iter_rows(named=True):
            self.logs[r["charger_id"]] = r["text"]
        for r in obs.visits.iter_rows(named=True):
            self._visit(r, now)
        new = signals.hourly(obs.messages)
        self.sig = new if self.sig is None else pl.concat([self.sig.filter(pl.col("hour") > hour - 175), new],
                                                          how="diagonal_relaxed")
        f = self._features(hour)
        if self.keep_features:
            self.feature_log.append(f.select("charger_id", "hour", *features.FEATURES))
        sym = features.symptoms(f, self.ops["detection"]).select(
            "charger_id", "symptom", "sym_faulted", "sym_offline", "sym_starts", "sym_evcomm", "sym_rfid",
            "sym_central", "tx_good")
        cmds: list[Command] = []
        frows = {r["charger_id"]: r for r in f.iter_rows(named=True)}
        for r in sym.iter_rows(named=True):
            self._track(r, hour, now)
        if self.models_from is not None:
            self.models_from.observe(obs.messages, obs.visits)
            self.models_from.on_hour(now)
        self._outcomes(now)
        cmds.extend(self._reopen(now))
        decide = [i for i in self.incidents.values() if i.state in ("open", "waiting") and i.pending is None
                  and i.proposal is None and now >= i.wait_until]
        if decide:
            df = pl.DataFrame([frows[i.charger_id] for i in decide]).with_columns(
                age=pl.Series([float(hour - i.onset_hour) for i in decide]))
            probs = self._diagnose(df)
            for inc, p, row in zip(decide, probs, df.iter_rows(named=True), strict=True):
                if not inc.first_probs:
                    inc.first_probs = dict(p)
                cmds.extend(self._decide(inc, p, row, now))
        cmds.extend(self._approved(now))
        if self.flags["cohorts"] and (now // HOUR) % 24 == 4:
            cmds.extend(self._cohorts(now))
        cmds.extend(self._rollout_stages(now))
        return [c for c in cmds if c is not None]

    def _features(self, hour: int) -> pl.DataFrame:
        status = pl.DataFrame(
            [{"charger_id": cid, "evse": e, "status": s, "received": hour * HOUR}
             for cid, cv in self.view.chargers.items() for e, (s, _) in cv.status.items() if e in (1, 2)],
            schema={"charger_id": pl.Utf8, "evse": pl.Int8, "status": pl.Utf8, "received": pl.Int64})
        ref = self.ref.with_columns(firmware=pl.Series([self.view.chargers[c].firmware for c in self.ref["charger_id"]]))
        return features.online(self.sig, status, ref, hour)

    # ---------------------------------------------------------------- incidents
    def _track(self, r: dict[str, Any], hour: int, now: int) -> None:
        cid = r["charger_id"]
        inc = self.incidents.get(cid)
        if r["symptom"]:
            names = {k[4:] for k in ("sym_faulted", "sym_offline", "sym_starts", "sym_evcomm", "sym_rfid",
                                     "sym_central") if r[k]}
            if inc is None:
                self.n += 1
                inc = Incident(f"AP-{self.n:05d}", cid, now, hour, hour, names)
                self.incidents[cid] = inc
                self.events.append({"ts": now, "incident": inc.id, "charger_id": cid, "event": "opened",
                                    "detail": ",".join(sorted(names))})
            else:
                inc.last_symptom_hour = hour
                inc.symptoms |= names
                if inc.state == "watch" and now - inc.watch_since > 2 * HOUR:
                    inc.state = "open"  # it came back: decide again
                    inc.resolution = ""
            inc.good_since_symptom = False
        elif inc is not None:
            if r["tx_good"] > 0:
                inc.good_since_symptom = True
            if hour - inc.last_symptom_hour > QUIET_HOURS and getattr(inc, "good_since_symptom", False) \
                    and inc.state not in ("part",):
                self._close(inc, now, "recovered")

    def _close(self, inc: Incident, now: int, how: str) -> None:
        inc.state, inc.closed = "resolved", now
        inc.resolution = inc.resolution or how
        self.incidents.pop(inc.charger_id, None)
        self.closed.append(inc)
        self.events.append({"ts": now, "incident": inc.id, "charger_id": inc.charger_id, "event": "closed",
                            "detail": inc.resolution})
        if self.models_from is not None:
            self.models_from.resolved(inc)

    def _outcomes(self, now: int) -> None:
        """Did the last remote action work? Observable: back online, not Faulted, a good session after it and no
        failed start since."""
        win = self.ops["autonomy"]["success_window_hours"] * HOUR
        for inc in list(self.incidents.values()):
            p = inc.pending
            if p is None:
                continue
            cv = self.view.chargers[inc.charger_id]
            good = any(t > p["at"] for t in cv.good_sessions)
            bad = any(t > p["at"] for t in cv.failed_starts) or any(t > p["at"] for t, _ in cv.evcomm_fails)
            faulted = bool(self.view.faulted(inc.charger_id)) and p["action"] in REMOTE
            done = now - p["at"] >= win or (good and not bad and not faulted) or bad
            if not done:
                continue
            success = good and not bad and not faulted and self.view.online(inc.charger_id)
            if p["action"] in REMOTE:
                self.ladder.observe(p["action"], cv.model, success, now)
                self._learn(p["action"], p["cls"], success)
            inc.pending = None
            inc.decisions.append({"ts": now, "outcome": "worked" if success else "did not work", "action": p["action"]})
            if success:
                inc.resolution = p["action"]
                inc.state, inc.watch_since = "watch", now  # wait for the quiet day that closes it
            else:
                inc.state = "open"

    # ---------------------------------------------------------------- diagnosis and planning
    def _diagnose(self, df: pl.DataFrame) -> list[dict[str, float]]:
        mo = self.models_from
        if mo is None or mo.diagnosis is None:
            return [{"other": 1.0} for _ in range(df.height)]
        p = mo.diagnosis.predict(df)
        mo.decision_rows.append(df.select([c for c in features.NUMERIC if c in df.columns]))
        out = [dict(zip(mo.diagnosis.classes, map(float, row), strict=True)) for row in p]
        if mo.challenger is not None:
            q = mo.challenger.predict(df)
            for d, row, cid in zip(out, q, df["charger_id"], strict=True):
                inc = self.incidents.get(cid)
                if inc:
                    inc.shadow = dict(zip(mo.challenger.classes, map(float, row), strict=True))
            del d
        return out

    def _belief(self, action: str, cls: str) -> tuple[float, float]:
        k = (action, cls)
        if k not in self.beliefs:
            pri = self.pcfg["remedy_priors"].get(cls, self.pcfg["remedy_priors"]["other"])
            self.beliefs[k] = list(map(float, pri.get(action, [1, 9])))
        return tuple(self.beliefs[k])  # type: ignore[return-value]

    def _learn(self, action: str, cls: str, success: bool) -> None:
        a, b = self._belief(action, cls)
        self.beliefs[(action, cls)] = [a + success, b + (not success)]

    def _read(self, inc: Incident, row: dict[str, Any], now: int, probs: dict[str, float]) -> dict[str, Any] | None:
        if not self.flags["reader"]:
            return None
        cv = self.view.chargers[inc.charger_id]
        code = next((c for _, c, _, _ in reversed(cv.codes) if c), "")
        info = next((i for _, _, i, _ in reversed(cv.codes) if i), "")
        complaints = [t for ts, t in self.complaints.get(inc.charger_id, []) if ts >= inc.opened - DAY][-3:]
        log = self.logs.get(inc.charger_id) if inc.log_requested else None
        if not (code or info or complaints or log):
            return None
        q = " ".join([code, info, " ".join(sorted(inc.symptoms)), *complaints])
        passages = [p.text for p in kb.search(q, cv.model, 4)]
        fields = {"model": cv.model, "firmware": cv.firmware, "symptoms": sorted(inc.symptoms),
                  "diagnosis": {k: round(v, 2) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4]},
                  "tech_code": code, "tech_info": info, "log": log, "complaints": complaints, "passages": passages}
        try:
            r = self.gateway.call("incident-reader", fields, now=now, caller=inc.id)
        except GatewayError as e:
            inc.decisions.append({"ts": now, "note": f"reader unavailable ({e.kind}); deciding from the model"})
            return None
        out = dict(r.output)
        out["_quarantined"] = len(r.guard.quarantined)
        return out

    def _combine(self, probs: dict[str, float], reading: dict[str, Any] | None) -> dict[str, float]:
        if not reading or not reading.get("suspected"):
            return probs
        w = self.pcfg["reader_weight"] * float(reading.get("confidence", 0.5))
        boost = {s["remedy"]: float(s["p"]) for s in reading["suspected"] if isinstance(s, dict)}
        out = {k: v * (1 + w * boost.get(k, 0.0) * 3) for k, v in probs.items()}
        for k, b in boost.items():
            if k not in out and k == "unlock":
                out[k] = 0.05 * w * b * 3
        z = sum(out.values()) or 1.0
        return {k: v / z for k, v in out.items()}

    def _value_per_hour(self, row: dict[str, Any]) -> float:
        cv = self.view.chargers[row["charger_id"]]
        b = self.ops["business"]
        sessions_h = max(row.get("tx_good_168", 0) or 0, row.get("tx_started_168", 0) or 0) / 168
        sessions_h = max(sessions_h, 0.05)
        kwh = self.pcfg["kwh_per_session"][cv.kind]
        return sessions_h * (kwh * b["revenue_per_kwh_eur"] + b["failed_attempt_cost_eur"])

    def _usability(self, row: dict[str, Any]) -> float:
        good = row.get("tx_good_24", 0) or 0
        bad = (row.get("failed_starts_24", 0) or 0) + (row.get("evcomm_fail_24", 0) or 0)
        if row.get("faulted_now", 0):
            return 0.0
        return 1.0 if good + bad == 0 else good / (good + bad)

    def _decide(self, inc: Incident, probs: dict[str, float], row: dict[str, Any], now: int) -> list[Command]:
        cid = inc.charger_id
        cv = self.view.chargers[cid]
        if inc.reading is None or (inc.log_requested and inc.log_text is None and cid in self.logs):
            inc.reading = self._read(inc, row, now, probs) or {}
            if inc.log_requested and cid in self.logs:
                inc.log_text = self.logs[cid]
        reading = inc.reading or {}
        combined = self._combine(probs, reading)
        inc.probs = combined
        inc.value_per_hour = self._value_per_hour(row)
        safety = bool(reading.get("safety_concern"))
        top = max(combined, key=combined.get) if combined else "other"
        cmds: list[Command] = []
        note = {"ts": now, "top": top, "p": round(combined.get(top, 0), 3), "safety": safety,
                "reader": reading.get("first_action"), "quarantined": reading.get("_quarantined", 0)}
        online = self.view.online(cid)
        # 1. a site where everything went quiet at once: a grid or network outage, nothing to repair
        if not online and (row.get("site_silent") or 0) >= 0.6 and (row.get("site_n") or 0) >= 2:
            inc.wait_until = now + 3 * HOUR
            inc.decisions.append(note | {"decision": "wait: whole site silent"})
            return []
        # 2. waiting may be the cheapest fix; and a quiet charger that only *looks* odd is watched, not visited
        ws = self.pcfg["wait_if_self_resolving"]
        likely_parts = [k for k, v in combined.items() if k.startswith("part:") and v >= self.pcfg["part_probability"]]
        failing = (row.get("failed_starts_24") or 0) + (row.get("evcomm_fail_24") or 0) > 0 or \
            (row.get("faulted_now") or 0) > 0 or not online
        if combined.get("self_resolved", 0) >= ws["probability"]:
            if now - inc.opened < ws["hours"] * HOUR:
                inc.wait_until = inc.opened + ws["hours"] * HOUR
                inc.decisions.append(note | {"decision": f"wait {ws['hours']} h: likely to clear on its own"})
                return []
            if not likely_parts and not failing and now - inc.opened < ws["max_watch_hours"] * HOUR:
                inc.wait_until = now + 12 * HOUR
                inc.decisions.append(note | {"decision": "keep watching: nothing is failing"})
                return []
        # 3. a firmware cohort problem is handled fleet-wide, not charger by charger
        if any(a.model == cv.model and a.version == cv.firmware for a in self.alerts) and combined.get("firmware", 0) + \
                combined.get("part:cable", 0) > 0.3:
            inc.wait_until = now + 12 * HOUR
            inc.decisions.append(note | {"decision": "wait: firmware cohort rollback in progress"})
            return []
        # 4. remote actions, most likely to work first
        if online and not safety:
            tried = {a for a, _, _ in inc.tried}
            options = []
            faulted = self.view.faulted(cid)
            station_level = "offline" in inc.symptoms or len(faulted) == 2 or "starts" in inc.symptoms
            for a in REMOTE:
                if a in tried:
                    continue
                if a == "unlock" and not faulted:
                    continue
                if a == "station_reset" and "evse_reset" not in tried and not station_level:
                    continue  # the runbook (and the shield) want the EVSE first
                p = sum(pc * (lambda ab: ab[0] / (ab[0] + ab[1]))(self._belief(a, c)) for c, pc in combined.items())
                options.append((p, a))
            options.sort(reverse=True)
            if options and options[0][0] >= self.pcfg["min_success_to_try"]:
                p, a = options[0]
                evse = (faulted[0] if faulted else 1) if a in ("unlock", "evse_reset") else 0
                params = {"type": "OnIdle"} if a != "unlock" else {}
                params |= {"station_level": station_level, "safety_concern": safety}
                cmd = Command(cid, a, evse, params, reason=f"P(works)={p:.2f}; likely {top}", incident=inc.id)
                out = self._issue(inc, cmd, now, cls=top)
                inc.decisions.append(note | {"decision": f"{a} (expected success {p:.2f})", "sent": bool(out)})
                return out
            if not inc.log_requested and float(reading.get("confidence", 0)) < self.pcfg["get_log_below_confidence"]:
                inc.log_requested = True
                cmd = Command(cid, "get_log", reason="reading unsure; fetch diagnostic log", incident=inc.id,
                              decided_by="autonomous")
                if self.shield.check(cmd, now).allowed:
                    self.shield.record(cmd, now)
                    inc.wait_until = now + HOUR
                    inc.decisions.append(note | {"decision": "get_log"})
                    return [cmd]
        # 5. a visit, with the parts the diagnosis expects; keep drivers away if it mostly fails meanwhile
        if inc.state != "dispatch":
            parts = sorted({k.split(":", 1)[1] for k, v in combined.items()
                            if k.startswith("part:") and v >= self.pcfg["part_probability"]})
            if safety:
                parts = sorted(set(parts) | {"cable"})
            inc.parts = parts
            inc.state = "dispatch"
            inc.decisions.append(note | {"decision": f"dispatch with {parts or 'no parts'}"})
            if (safety or self._usability(row) < self.pcfg["inoperative_below_usability"]) and not inc.inoperative \
                    and online:
                cmd = Command(cid, "set_inoperative", 0, {}, reason="mostly failing until repaired", incident=inc.id)
                cmds.extend(self._issue(inc, cmd, now, cls=top, track=False))
        return cmds

    def _issue(self, inc: Incident, cmd: Command, now: int, cls: str, track: bool = True) -> list[Command]:
        v = self.shield.check(cmd, now)
        if not v.allowed:
            inc.decisions.append({"ts": now, "blocked": cmd.action, "rules": v.blocked_by})
            if cmd.action in REMOTE:
                inc.tried.append((cmd.action, now, cmd.evse))  # do not ask again; move on
            return []
        cv = self.view.chargers[cmd.charger_id]
        level = self.ladder.level(cmd.action, cv.model) if self.flags["autonomy"] else "propose"
        if level == "act":
            cmd.decided_by = "autonomous"
            return self._send(inc, cmd, now, cls, track)
        due = self._noc_time(now, cmd.charger_id)
        inc.proposal = {"cmd": cmd, "due": due, "cls": cls, "track": track, "at": now}
        self.approvals.append((due, cmd, inc.id))
        return []

    def _send(self, inc: Incident, cmd: Command, now: int, cls: str, track: bool) -> list[Command]:
        self.shield.record(cmd, now)
        if cmd.action in REMOTE:
            inc.tried.append((cmd.action, now, cmd.evse))
            if track:
                inc.pending = {"action": cmd.action, "at": now, "cls": cls}
                inc.state = "waiting"
        if cmd.action == "set_inoperative":
            inc.inoperative = True
        self.events.append({"ts": now, "incident": inc.id, "charger_id": inc.charger_id, "event": cmd.action,
                            "detail": cmd.decided_by})
        return [cmd]

    def _noc_time(self, now: int, key: str) -> int:
        a, b = self.noc["staffed_utc"]
        t = now
        while not (a <= (t // HOUR) % 24 < b):
            t += HOUR
        lo, hi = self.noc["pickup_delay_hours"]
        return t + int((lo + (hi - lo) * unit(key, now, "approve")) * HOUR)

    def _approved(self, now: int) -> list[Command]:
        out, keep = [], []
        for due, cmd, iid in self.approvals:
            if due > now:
                keep.append((due, cmd, iid))
                continue
            inc = next((i for i in self.incidents.values() if i.id == iid), None)
            if cmd.action in ("halt_rollout", "firmware"):
                cmd.decided_by = "operator"
                out.append(cmd)
                continue
            if inc is None or inc.proposal is None:
                continue
            prop = inc.proposal
            inc.proposal = None
            v = self.shield.check(cmd, now)  # the world may have changed since it was proposed
            if not v.allowed:
                inc.decisions.append({"ts": now, "blocked": cmd.action, "rules": v.blocked_by, "after": "approval"})
                if cmd.action in REMOTE:
                    inc.tried.append((cmd.action, now, cmd.evse))
                continue
            cmd.decided_by = "operator"
            out.extend(self._send(inc, cmd, now, prop["cls"], prop["track"]))
        self.approvals = keep
        return out

    # ---------------------------------------------------------------- visits
    def _visit(self, r: dict[str, Any], now: int) -> None:
        inc = self.incidents.get(r["charger_id"])
        if inc is None:
            return
        inc.decisions.append({"ts": now, "visit": r["outcome"], "part_used": r["part_used"],
                              "part_needed": r["part_needed"]})
        if r["outcome"] == "part_needed":
            inc.state, inc.parts, inc.part_eta = "part", sorted({*inc.parts, r["part_needed"]}), r["part_eta"]
        elif r["outcome"] == "not_reached":
            inc.state = "dispatch"
        elif r["outcome"] in ("fixed", "no_fault_found", "site_without_power"):
            inc.resolution = {"fixed": "visit", "no_fault_found": "no_fault_found",
                              "site_without_power": "grid"}[r["outcome"]]
            inc.part_used = r["part_used"]
            if r["outcome"] == "fixed" and not r["part_used"] and r["note"]:
                try:  # the structured field was left empty: read the note
                    res = self.gateway.call("note-labeler", {"note": r["note"], "lang": r["lang"]}, now=now,
                                            caller=inc.id)
                    inc.part_used = res.output.get("part")
                except GatewayError:
                    pass
            if inc.inoperative:
                self.pending_operative.append(inc.charger_id)
                inc.inoperative = False
            self._close(inc, now, inc.resolution)

    # ---------------------------------------------------------------- daily: routes, preventive visits
    def plan_day(self, day_start: int) -> dict[str, list[Stop]]:
        now = day_start + 5 * HOUR
        jobs: list[Job] = []
        sites = {r["site_id"]: r for r in self.sites.iter_rows(named=True)}
        hour = now // HOUR - 1
        for inc in self.incidents.values():
            if inc.state == "dispatch" or (inc.state == "part" and (inc.part_eta or 0) <= day_start + 8 * HOUR):
                cv = self.view.chargers[inc.charger_id]
                quiet = hour - inc.last_symptom_hour >= 6 and self.view.online(inc.charger_id) \
                    and not self.view.faulted(inc.charger_id) and not inc.inoperative and not inc.preventive
                if quiet and inc.state == "dispatch":
                    inc.decisions.append({"ts": now, "decision": "visit held: no symptom for 6 h"})
                    continue
                s = sites[cv.site_id]
                age_h = (now - inc.opened) / HOUR
                value = inc.value_per_hour * 24 * (1 + min(age_h / 72, 2))  # a day of downtime, more for old ones
                jobs.append(Job(inc.id, inc.charger_id, s["lat"], s["lon"], max(value, 30.0),
                                self.pcfg["dispatch_service_minutes"], inc.parts))
        if self.flags["predictive"] and self.models_from is not None and self.models_from.risk is not None:
            jobs.extend(self._preventive(now, sites))
        if not self.flags["vrp"]:
            return self._fifo(jobs)
        fs_speed, out = 70.0, {}
        by_country: dict[str, list[Job]] = defaultdict(list)
        for j in jobs:
            by_country[self.view.chargers[j.charger_id].country].append(j)
        techs = self.techs.to_dicts()
        for country, js in by_country.items():
            ts = [t for t in techs if t["country"] == country]
            routes = plan_country(js, ts, workday_min=9 * 60, speed_kmh=fs_speed,
                                  km_cost=self.ops["business"]["km_cost_eur"],
                                  max_stops=self.ops["dispatch"]["stops_per_technician"],
                                  seconds=self.ops["dispatch"]["solver_seconds"] / max(len(by_country), 1))
            for tech, route in routes.items():
                out[tech] = [Stop(j.charger_id, j.parts, j.key) for j in route]
        return out

    def _fifo(self, jobs: list[Job]) -> dict[str, list[Stop]]:
        class T:  # adapter to the baseline's queue
            def __init__(self, j: Job, opened: int) -> None:
                self.charger_id, self.opened, self.id, self.part, self.part_eta = j.charger_id, opened, j.key, None, None

        opened = {i.id: i.opened for i in self.incidents.values()}
        q = [T(j, opened.get(j.key, 0)) for j in jobs]
        parts = {j.charger_id: j.parts for j in jobs}
        return fifo_routes(self.view, q, self.techs, self.ops["dispatch"]["stops_per_technician"], parts)

    def _preventive(self, now: int, sites: dict[str, dict[str, Any]]) -> list[Job]:
        cfg = self.pcfg["preventive"]
        hour = now // HOUR - 1
        f = self._features(hour)
        f = f.filter(pl.col("faulted_now") == 0)
        p = self.models_from.risk.predict(f)[:, 1]
        f = f.with_columns(risk=pl.Series(p)).filter((pl.col("risk") >= cfg["risk_threshold"])
                                                    & (pl.col("temp_trend") >= cfg["temperature_trend_c"])
                                                    & (pl.col("kind") != "AC")).sort("risk", descending=True)
        jobs = []
        for r in f.head(cfg["max_per_day"]).iter_rows(named=True):
            cid = r["charger_id"]
            if cid in self.incidents:
                continue
            self.n += 1
            inc = Incident(f"AP-{self.n:05d}", cid, now, hour, hour, {"risk"}, state="dispatch", preventive=True)
            inc.parts = ["power_module"]
            inc.value_per_hour = self._value_per_hour(r)
            self.incidents[cid] = inc
            self.events.append({"ts": now, "incident": inc.id, "charger_id": cid, "event": "preventive",
                                "detail": f"risk {r['risk']:.2f}, module +{r['temp_trend']:.1f} C"})
            s = sites[self.view.chargers[cid].site_id]
            jobs.append(Job(inc.id, cid, s["lat"], s["lon"], inc.value_per_hour * 48 * float(r["risk"]),
                            self.pcfg["dispatch_service_minutes"], ["power_module"]))
        return jobs

    # ---------------------------------------------------------------- firmware cohorts
    def _cohorts(self, now: int) -> list[Command]:
        for a in detect_cohorts(self.view, now):
            if any(x.model == a.model and x.version == a.version for x in self.alerts):
                continue
            self.alerts.append(a)
            self.events.append({"ts": now, "incident": None, "charger_id": None, "event": "cohort_alert",
                                "detail": f"{a.model} {a.version}: {a.rate:.3f} vs {a.peer_rate:.3f} (p={a.p_value:.1e})"})
            a.approved = self._noc_time(now, f"{a.model}{a.version}")
        return []

    def _rollout_stages(self, now: int) -> list[Command]:
        out: list[Command] = []
        cfg = next(r for r in self.ops["shield"]["rules"] if r["id"] == "firmware-staged")
        for a in self.alerts:
            if a.approved is None or now < a.approved:
                continue
            if not a.stages:
                out.append(Command("*", "halt_rollout", params={"model": a.model, "version": a.version},
                                   reason="cohort alert approved", decided_by="operator"))
                a.stages.append({"stage": 0, "at": now, "share": 0.0, "chargers": []})
            last = a.stages[-1]
            if now - last["at"] < cfg["stage_hours"] * HOUR and last["stage"] > 0:
                continue
            k = len(a.stages)
            if k > len(cfg["stages"]):
                continue
            share = cfg["stages"][k - 1]
            cohort = sorted([c for c in self.view.chargers.values() if c.model == a.model and c.firmware == a.version],
                            key=lambda c: -len(c.evcomm_fails))
            done = {cid for s in a.stages for cid in s["chargers"]}
            total = a.chargers
            n = max(1, math.ceil(share * total) - len(done))
            pick = [c.charger_id for c in cohort if c.charger_id not in done][:n]
            for cid in pick:
                cmd = Command(cid, "firmware", 0, {"version": a.previous, "stage_approved": True},
                              reason=f"staged rollback {k}/{len(cfg['stages'])}", decided_by="operator")
                if self.shield.check(cmd, now).allowed:
                    self.shield.record(cmd, now)
                    out.append(cmd)
            a.stages.append({"stage": k, "at": now, "share": share, "chargers": pick})
        return out

    # ---------------------------------------------------------------- report
    def artifacts(self) -> dict[str, list[dict[str, Any]]]:
        """Everything the operations room shows: incidents with their decisions, the shield's verdicts, the
        gateway's ledger and spans, the event stream."""
        incs = []
        for i in [*self.closed, *self.incidents.values()]:
            incs.append({"id": i.id, "charger_id": i.charger_id, "opened": i.opened, "closed": i.closed,
                         "state": i.state, "resolution": i.resolution, "symptoms": sorted(i.symptoms),
                         "preventive": i.preventive, "parts": i.parts, "first_probs": i.first_probs,
                         "probs": i.probs, "reading": {k: v for k, v in (i.reading or {}).items()},
                         "tried": [list(t) for t in i.tried], "decisions": i.decisions,
                         "value_per_hour": round(i.value_per_hour, 2), "part_used": i.part_used})
        return {"incidents": incs, "shield": self.shield.log, "ledger": self.gateway.ledger,
                "spans": self.gateway.spans(), "events": self.events}

    def report(self) -> dict[str, Any]:
        return {"incidents": len(self.closed) + len(self.incidents), "open": len(self.incidents),
                "ladder": self.ladder.snapshot(),
                "beliefs": {f"{a}|{c}": [round(x, 1) for x in v] for (a, c), v in self.beliefs.items()},
                "alerts": [{"model": a.model, "version": a.version, "raised": a.raised, "approved": a.approved,
                            "p_value": a.p_value, "rate": a.rate, "peer_rate": a.peer_rate,
                            "stages": [{k: v for k, v in s.items() if k != "chargers"} | {"n": len(s["chargers"])}
                                       for s in a.stages]} for a in self.alerts],
                "flags": self.flags}

    def _reopen(self, now: int) -> list[Command]:
        """Chargers taken out of service while waiting for a technician go back into service once repaired."""
        out = []
        for cid in self.pending_operative:
            cmd = Command(cid, "set_operative", 0, {}, reason="repaired", decided_by="autonomous")
            if self.shield.check(cmd, now).allowed:
                self.shield.record(cmd, now)
                out.append(cmd)
        self.pending_operative = []
        return out


_ = np
