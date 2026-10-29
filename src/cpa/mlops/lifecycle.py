"""The models' life inside a run: which version decides, weekly retraining on history plus what the
autopilot has seen since, the offline gate, a shadow period against incidents as they resolve, then
promotion; and daily drift checks of the features the diagnosis model sees.

Everything uses observable data. A resolved incident's label is what fixed it (the remote action
that worked, the part the technician used, or nothing), the same rule as in training."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

from .. import settings
from ..llm import lexicon
from ..ml import dataset, features, models, signals
from . import gate
from .registry import Registry, ServedModel

HOUR = 3600
DAY = 86400


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 30:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


class ModelOps:
    def __init__(
        self, registry: Registry, history: dataset.Prepared, chargers: pl.DataFrame, *, retrain: bool = True
    ) -> None:
        self.reg = registry
        self.history = history
        self.chargers = chargers
        self.cfg = settings.models()
        self.retrain_enabled = retrain
        self.diagnosis = self._load("diagnosis", "champion")
        self.risk = self._load("risk", "champion")
        self.challenger: ServedModel | None = None
        self.challenger_risk: ServedModel | None = None
        self.shadow: list[dict[str, Any]] = []
        self.history_log: list[dict[str, Any]] = []
        self.drift: list[dict[str, Any]] = []
        self.policy: Any = None
        self.run_parts: dict[str, list[pl.DataFrame]] = {
            "hourly": [],
            "status": [],
            "actions": [],
            "visits": [],
        }
        self.last_retrain = 0
        self.decision_rows: list[pl.DataFrame] = []
        self.serving_ref: pl.DataFrame | None = None
        self.serving_from = 0

    def _load(self, name: str, alias: str) -> ServedModel | None:
        v = self.reg.alias(name, alias)
        return self.reg.load(name, v) if v else None

    def attach(self, policy: Any) -> None:
        self.policy = policy
        self._reference()
        self.history_log.append(
            {
                "ts": policy.t0,
                "event": "start",
                "diagnosis": self.diagnosis.version if self.diagnosis else None,
                "risk": self.risk.version if self.risk else None,
            }
        )

    def _reference(self) -> None:
        if self.diagnosis is None:
            return
        meta = self.diagnosis.meta
        self.ref_window = (meta["card"]["trained_on"]["from_hour"], meta["card"]["trained_on"]["to_hour"])

    # ---------------------------------------------------------------- data the run produces
    def observe(self, obs_messages: pl.DataFrame, visits: pl.DataFrame) -> None:
        if obs_messages.height:
            self.run_parts["status"].append(signals.status_events(obs_messages))
            self.run_parts["actions"].append(dataset.actions_of(obs_messages))
        if visits.height:
            self.run_parts["visits"].append(visits)

    def on_hour(self, now: int) -> None:
        p = self.policy
        if p.sig is not None:
            h = now // HOUR - 1
            self.run_parts["hourly"].append(p.sig.filter(pl.col("hour") == h))
        if (now // HOUR) % 24 == 3:
            self._drift(now)
            dow = ((now // DAY) + 3) % 7  # 1970-01-01 was a Thursday
            if self.retrain_enabled and dow == self.cfg["retraining"]["weekday"] and now - self.last_retrain > 3 * DAY:
                self._retrain(now)

    # ---------------------------------------------------------------- retraining and promotion
    def _run_prepared(self) -> dataset.Prepared:
        parts = {k: pl.concat(v, how="diagonal_relaxed") if v else None for k, v in self.run_parts.items()}
        hist = self.history
        start = int(parts["hourly"]["hour"].min()) if parts["hourly"] is not None else 10**12
        return dataset.Prepared(
            pl.concat([hist.hourly.filter(pl.col("hour") < start), parts["hourly"]], how="diagonal_relaxed")
            if parts["hourly"] is not None
            else hist.hourly,
            pl.concat(
                [hist.status.filter(pl.col("received") < start * HOUR), parts["status"]],
                how="diagonal_relaxed",
            )
            if parts["status"] is not None
            else hist.status,
            pl.concat([hist.actions.filter(pl.col("ts") < start * HOUR), parts["actions"]], how="diagonal_relaxed")
            if parts["actions"] is not None
            else hist.actions,
            pl.concat([hist.visits.filter(pl.col("ts") < start * HOUR), parts["visits"]], how="diagonal_relaxed")
            if parts["visits"] is not None
            else hist.visits,
        )

    def _retrain(self, now: int) -> None:
        t = time.perf_counter()
        self.last_retrain = now
        prep = self._run_prepared()
        ds = dataset.build(prep, self.chargers, lexicon.read, until=now)
        rec: dict[str, Any] = {"ts": now, "event": "retrain", "examples": ds["diagnosis"].height}
        cand = models.train_diagnosis(ds["diagnosis"], self.cfg["diagnosis"])
        tr, va = models._split(ds["diagnosis"], "onset", self.cfg["diagnosis"]["validation_days"])
        onx = models.to_onnx(cand)
        checks = models.onnx_check(cand, onx, va.tail(min(2000, va.height)), iterations=40)
        verdict = self._fair_gate(cand, self.diagnosis, va, "onset", checks)
        rec["diagnosis_gate"] = {
            "passed": verdict["passed"],
            "reasons": verdict["reasons"],
            "candidate_macro_f1": verdict["candidate"].get("macro_f1"),
            "champion_macro_f1": verdict["champion"].get("macro_f1"),
        }
        if verdict["passed"]:
            v = self.reg.register(
                cand, onx, checks, {"trained_at": str(now), "run": self.policy.name, "stage": "challenger"}
            )
            self.reg.set_alias("diagnosis", "challenger", v)
            self.challenger = self.reg.load("diagnosis", v)
            self.challenger_rows = tr.select([c for c in features.NUMERIC if c in tr.columns])
            self.shadow = []
            rec["challenger"] = v
        rcand = models.train_risk(ds["risk"], self.cfg["risk"])
        _, rva = models._split(ds["risk"], "hour", self.cfg["risk"]["validation_days"])
        ronx = models.to_onnx(rcand)
        rchecks = models.onnx_check(rcand, ronx, rva.tail(min(2000, rva.height)), iterations=40)
        rverdict = self._fair_gate(rcand, self.risk, rva, "hour", rchecks)
        rec["risk_gate"] = {
            "passed": rverdict["passed"],
            "reasons": rverdict["reasons"],
            "candidate_pr_auc": rverdict["candidate"].get("pr_auc"),
            "champion_pr_auc": rverdict["champion"].get("pr_auc"),
        }
        if rverdict["passed"]:  # the risk model only ranks preventive visits: it is promoted after the offline gate
            v = self.reg.register(rcand, ronx, rchecks, {"trained_at": str(now), "run": self.policy.name})
            old = self.reg.alias("risk", "champion")
            if old:
                self.reg.set_alias("risk", "previous", old)
            self.reg.set_alias("risk", "champion", v)
            self.risk = self.reg.load("risk", v)
            rec["risk_promoted"] = v
        rec["seconds"] = round(time.perf_counter() - t, 1)
        self.history_log.append(rec)

    def _fair_gate(
        self, cand: Any, champion: Any, va: pl.DataFrame, col: str, checks: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare on the validation rows neither model trained on: the champion was trained up to some hour, and
        rows before it would flatter the champion. Until there are enough such rows the challenger waits."""
        if champion is not None:
            cut = int(champion.meta["card"]["trained_on"]["to_hour"])
            va = va.filter(pl.col(col) > cut)
            need = self.cfg["promotion"]["offline"]["min_unseen_rows"][cand.kind]
            if va.height < need:
                return {
                    "passed": False,
                    "candidate": {},
                    "champion": {},
                    "reasons": [f"only {va.height} rows the champion has not seen (needs {need})"],
                }
        return gate.offline(cand, champion, va, checks)

    def resolved(self, inc: Any) -> None:
        """Shadow scoring: when an incident resolves, compare what champion and challenger predicted first."""
        if self.challenger is None or not inc.shadow or not getattr(inc, "first_probs", None):
            return
        label = self.label_of(inc)
        if label is None:
            return
        champ = max(inc.first_probs, key=inc.first_probs.get)
        chal = max(inc.shadow, key=inc.shadow.get)
        self.shadow.append({"incident": inc.id, "label": label, "champion": champ, "challenger": chal})
        s = self.cfg["promotion"]["shadow"]
        if len(self.shadow) >= s["min_incidents"]:
            c1 = sum(r["label"] == r["champion"] for r in self.shadow)
            c2 = sum(r["label"] == r["challenger"] for r in self.shadow)
            rec = {
                "ts": inc.closed,
                "event": "shadow_verdict",
                "n": len(self.shadow),
                "champion_correct": c1,
                "challenger_correct": c2,
                "version": self.challenger.version,
            }
            if c2 >= c1:
                old = self.reg.alias("diagnosis", "champion")
                if old:
                    self.reg.set_alias("diagnosis", "previous", old)
                self.reg.set_alias("diagnosis", "champion", self.challenger.version)
                self.diagnosis = self.challenger
                self.serving_ref, self.serving_from = None, len(self.decision_rows)  # a new baseline for drift
                rec["promoted"] = True
            else:
                rec["promoted"] = False
            self.reg.client.delete_registered_model_alias("diagnosis", "challenger")
            self.challenger = None
            self.shadow = []
            self.history_log.append(rec)

    @staticmethod
    def label_of(inc: Any) -> str | None:
        r = inc.resolution
        if r in ("evse_reset", "station_reset"):
            return "reset"
        if r == "unlock":
            return "unlock"
        if r == "visit":
            part = getattr(inc, "part_used", None)
            return f"part:{part}" if part else "onsite_restart"
        if r in ("recovered", "no_fault_found", "grid"):
            return "self_resolved"
        return None

    # ---------------------------------------------------------------- drift
    def _drift(self, now: int) -> None:
        """PSI of each numeric feature: the rows the champion scored over the last days, against the first rows it
        scored in service. Training rows are sampled differently (fixed incident ages, other policies' incidents), so
        they would make every day look like drift; the champion's own early weeks are the like-for-like baseline."""
        if self.diagnosis is None or not self.decision_rows:
            return
        rows = self.decision_rows[self.serving_from :]
        if self.serving_ref is None:
            n = sum(r.height for r in rows)
            if n >= self.cfg["drift"]["reference_rows"]:
                self.serving_ref = pl.concat(rows, how="diagonal_relaxed")
                self.serving_from = len(self.decision_rows)
            return
        cur = pl.concat(self.decision_rows[-24 * self.cfg["drift"]["window_days"] :], how="diagonal_relaxed")
        if cur.height < 30:
            return
        ref = self.serving_ref
        scores = {
            c: psi(ref[c].to_numpy().astype(float), cur[c].to_numpy().astype(float))
            for c in features.NUMERIC
            if c in cur.columns and c in ref.columns
        }
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        alert = any(v > self.cfg["drift"]["psi_alert"] for _, v in top)
        self.drift.append(
            {
                "ts": now,
                "rows": cur.height,
                "top": [{"feature": k, "psi": round(v, 3)} for k, v in top],
                "alert": alert,
                "champion": self.diagnosis.version,
            }
        )

    def set_reference_rows(self, rows: pl.DataFrame) -> None:
        self._ref_rows = rows

    def report(self) -> dict[str, Any]:
        return {
            "events": self.history_log,
            "drift": self.drift[-60:],
            "drift_alerts": sum(1 for d in self.drift if d["alert"]),
            "champion": {
                "diagnosis": self.diagnosis.version if self.diagnosis else None,
                "risk": self.risk.version if self.risk else None,
            },
            "shadow_pending": len(self.shadow),
        }
