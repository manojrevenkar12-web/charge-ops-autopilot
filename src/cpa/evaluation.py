"""Judges each policy's run against the simulator's truth. With src/cpa/sim, the only code that may
read build/truth or config/fleet.yaml.

Availability is per connector-hour: the chance a driver arriving then could charge (0 when Faulted,
Unavailable, or broken in a way that fails every start; partial when, say, only cards fail).
Grid outages are excluded, as uptime definitions in tenders exclude them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .settings import RUNS, TRUTH
from .sim.world import truth_config

FAILS = ["fail_start", "fail_auth", "fail_evcomm", "fail_offline", "fail_reboot"]
KIND_TO_CLASS = {
    "controller_hang": {"reset", "onsite_restart"},
    "connector_lock": {"unlock", "part:lock_actuator", "reset"},
    "power_module": {"part:power_module"},
    "rfid_reader": {"part:rfid_reader", "reset"},
    "payment_terminal": {"part:payment_terminal", "reset"},
    "cable_wear": {"part:cable"},
    "isolation_fault": {"part:cable"},
    "comm_loss": {"self_resolved", "part:modem"},
    "firmware_bug": {"firmware"},
}


def _load(run: Path) -> dict[str, Any]:
    meta = json.loads((run / "meta.json").read_text())
    t, o = run / "truth", run / "observed"
    d = {
        k: pl.read_parquet(t / f"{k}.parquet")
        for k in ("sessions", "episodes", "harms", "visits", "availability", "firmware")
    }
    d["commands"] = pl.read_parquet(o / "commands.parquet")
    d["obs_visits"] = pl.read_parquet(o / "visits.parquet")
    d["meta"] = meta
    ops = run / "ops"
    d["ops"] = {f.stem: json.loads(f.read_text()) for f in ops.glob("*.json")} if ops.exists() else {}
    return d


def evaluate_run(run: Path) -> dict[str, Any]:
    cfg = truth_config()
    d = _load(run)
    s0, s1 = d["meta"]["scored_from"], d["meta"]["end"]
    day0 = (s0 - d["meta"]["start"]) // 86400
    av = d["availability"].filter(pl.col("day") >= day0)
    sess = d["sessions"].filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    out: dict[str, Any] = {"policy": d["meta"]["policy"], "days": (s1 - s0) // 86400}
    usable, total, reported = av["usable_h"].sum(), av["total_h"].sum(), av["reported_h"].sum()
    out["availability"] = {
        "true": usable / total,
        "reported": reported / total,
        "gap": (reported - usable) / total,
    }
    ch = pl.read_parquet(run.parent.parent / "observed" / "reference" / "chargers.parquet")
    per = (
        av.group_by("charger_id")
        .agg(a=pl.col("usable_h").sum() / pl.col("total_h").sum())
        .join(ch.select("charger_id", "kind", "country"), on="charger_id")
    )
    rapid = per.filter(pl.col("kind") != "AC")
    out["availability"]["rapid_mean"] = float(rapid["a"].mean())
    out["availability"]["rapid_share_99"] = float((rapid["a"] >= 0.99).mean())
    out["availability"]["rapid_share_97"] = float((rapid["a"] >= 0.97).mean())
    out["availability"]["by_country"] = {
        r["country"]: round(r["a"], 4)
        for r in per.group_by("country").agg(pl.col("a").mean()).sort("country").iter_rows(named=True)
    }
    counts = dict(sess.group_by("outcome").len().iter_rows())
    ok = counts.get("ok", 0)
    failed = sum(counts.get(k, 0) for k in FAILS)
    unseen = counts.get("unavailable_seen", 0)
    kwh = float(sess.filter(pl.col("outcome") == "ok")["kwh"].mean() or 0)
    price = cfg["drivers"]["revenue_per_kwh_eur"]
    out["drivers"] = {
        "sessions_ok": ok,
        "failed_attempts": failed,
        "turned_away_unavailable": unseen,
        "charging_success_rate": ok / max(ok + failed, 1),
        "lost_revenue_eur": (failed * (1 - cfg["drivers"]["retry_after_failure"]) + unseen) * kwh * price,
        "failures_by_cause": dict(sess.filter(pl.col("outcome").is_in(FAILS)).group_by("cause").len().iter_rows()),
    }
    vt = d["visits"].filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    out["field"] = {
        "visits": vt.height,
        "cost_eur": float(vt["cost_eur"].sum()),
        "necessary_share": float(vt["necessary"].mean()) if vt.height else None,
        "no_fault_found": int((vt["wasted"] == "no_fault").sum()),
        "grid_wasted": int((vt["wasted"] == "grid_outage").sum()),
        "part_missing": int((vt["needed"].fill_null("") != "").sum()),
    }
    ep = d["episodes"].filter((pl.col("start") >= s0) & (pl.col("start") < s1))
    ep = ep.with_columns(hours=((pl.col("end").fill_null(s1).clip(upper_bound=s1) - pl.col("start")) / 3600))
    out["repair"] = {
        "episodes": ep.height,
        "median_hours_by_kind": {
            r["kind"]: round(r["hours"], 1)
            for r in ep.group_by("kind").agg(pl.col("hours").median()).sort("kind").iter_rows(named=True)
        },
        "ended_by": dict(ep.group_by("end_reason").len().iter_rows()),
    }
    hm = d["harms"].filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    cm_all = d["commands"].filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    out["harms"] = {
        "interrupted_sessions": int((hm["harm"] == "interrupted_session").sum()),
        "unsafe_resets": int((hm["harm"] == "unsafe_reset").sum()),
        "injected_actions": int(cm_all["reason"].fill_null("").str.contains("injected").sum()),
        "by": dict(hm.group_by("decided_by").len().iter_rows()) if hm.height else {},
    }
    cm = d["commands"].filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    out["commands"] = {
        f"{a}|{b}": n for a, b, n in cm.group_by("action", "decided_by").len().sort("action").iter_rows()
    }
    fw = d["firmware"].filter(pl.col("by") == "rollout")
    buggy = [(r["model"], r["version"]) for r in cfg["firmware_rollouts"] if r.get("bug")]
    out["firmware"] = {
        "buggy_installs": int(fw.filter(pl.col("version").is_in([v for _, v in buggy])).height),
        "rollbacks": int(d["firmware"].filter((pl.col("by") == "operator") & pl.col("ok")).height),
        "bug_failures": int(sess.filter(pl.col("cause") == "firmware_bug").height),
    }
    out["value"] = {
        "lost_revenue_eur": out["drivers"]["lost_revenue_eur"],
        "field_cost_eur": out["field"]["cost_eur"],
    }
    out |= _incident_quality(d, s0, s1)
    out |= _llm(d, s0, s1)
    return out


def _incident_quality(d: dict[str, Any], s0: int, s1: int) -> dict[str, Any]:
    incs = d["ops"].get("incidents")
    if not incs or "first_probs" not in incs[0]:
        return {}
    ep = d["episodes"]
    by_c = ep.partition_by("charger_id", as_dict=True)
    detected, delays, correct, total, false_inc = 0, [], 0, 0, 0
    scored = [i for i in incs if s0 <= i["opened"] < s1 and not i["preventive"]]
    for i in scored:
        e = by_c.get((i["charger_id"],))
        hit = None
        if e is not None:
            m = e.filter((pl.col("start") <= i["opened"]) & (pl.col("end").fill_null(10**12) >= i["opened"] - 3600))
            if m.height:
                hit = m.sort("start").row(-1, named=True)
        if hit is None:
            false_inc += 1
            continue
        if i["first_probs"]:
            total += 1
            top = max(i["first_probs"], key=i["first_probs"].get)
            correct += top in KIND_TO_CLASS.get(hit["kind"], set())
    # silent faults: were they found, and how fast?
    silent = ep.filter(
        pl.col("kind").is_in(["controller_hang", "rfid_reader", "payment_terminal", "cable_wear"])
        & (pl.col("start") >= s0)
        & (pl.col("start") < s1 - 86400)
    )
    opened = pl.DataFrame(
        [{"charger_id": i["charger_id"], "opened": i["opened"]} for i in incs],
        schema={"charger_id": pl.Utf8, "opened": pl.Int64},
    )
    for r in silent.iter_rows(named=True):
        o = opened.filter(
            (pl.col("charger_id") == r["charger_id"])
            & (pl.col("opened") >= r["start"] - 3600)
            & (pl.col("opened") <= (r["end"] or s1))
        )
        if o.height:
            detected += 1
            delays.append((o["opened"].min() - r["start"]) / 3600)
    prev = [i for i in incs if i["preventive"] and s0 <= i["opened"] < s1]
    pm_hits = 0
    for i in prev:
        e = by_c.get((i["charger_id"],))
        if (
            e is not None
            and e.filter(
                (pl.col("kind") == "power_module")
                & (pl.col("start") <= i["opened"])
                & (pl.col("end").fill_null(10**12) >= i["opened"])
            ).height
        ):
            pm_hits += 1
    return {
        "incidents": {
            "opened": len(scored),
            "false": false_inc,
            "diagnosed": total,
            "diagnosis_top1_matches_truth": correct / total if total else None,
            "silent_faults": silent.height,
            "silent_detected": detected / max(silent.height, 1),
            "silent_detection_median_hours": float(np.median(delays)) if delays else None,
            "preventive_visits": len(prev),
            "preventive_with_failing_module": pm_hits,
        }
    }


def _llm(d: dict[str, Any], s0: int, s1: int) -> dict[str, Any]:
    ledger = d["ops"].get("ledger")
    if not ledger:
        return {}
    lg = pl.DataFrame(ledger).filter((pl.col("ts") >= s0) & (pl.col("ts") < s1))
    days = (s1 - s0) / 86400
    inj = d["episodes"].filter(pl.col("injection") & (pl.col("start") < s1))
    shield = pl.DataFrame(d["ops"].get("shield") or [], infer_schema_length=None)
    return {
        "llm": {
            "calls": lg.height,
            "cost_eur_per_day": float(lg["cost_eur"].sum()) / days,
            "by_use_case": {
                r["use_case"]: {"calls": r["calls"], "cost_eur": round(r["cost"], 4)}
                for r in lg.group_by("use_case")
                .agg(calls=pl.len(), cost=pl.col("cost_eur").sum())
                .iter_rows(named=True)
            },
            "quarantined_fields": int(lg["quarantined"].sum()),
            "masked_values": int(lg["masked"].sum()),
            "episodes_with_injected_text": inj.height,
            "outcomes": dict(lg.group_by("outcome").len().iter_rows()),
        },
        "shield": {
            "checked": shield.height,
            "blocked": int((~shield["allowed"]).sum()) if shield.height else 0,
            "blocked_by": dict(shield.filter(~pl.col("allowed")).group_by("blocked_by").len().iter_rows())
            if shield.height
            else {},
        },
    }


def evaluate_all(names: list[str]) -> dict[str, Any]:
    return {n: evaluate_run(RUNS / n) for n in names if (RUNS / n / "meta.json").exists()}


_ = TRUTH
