"""Writes what the operations room shows (web/public/data) from the evaluation and the autopilot's run.
Reporting code: like the evaluation, it may read the truth, to show the simulator's answer next to
each incident."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from . import settings

LABELS = {
    "baseline": ("Today's process", "NOC tickets, a reset first, technicians oldest ticket first"),
    "assist": ("Assistant", "the same brain as the autopilot, every action approved by an operator"),
    "autopilot": ("Autopilot", "acts alone where it has earned it, within the shield"),
    "autopilot-fifo": ("Autopilot, oldest-first visits", "the autopilot without route optimisation"),
    "autopilot-unshielded": ("Autopilot without shield", "planner alone, no shield, no guardrails"),
    "llm-decides": ("LLM decides (control)", "the model's recommended action executed as it stands"),
}
ACTION = re.compile(r"^(evse_reset|station_reset|unlock|get_log|set_inoperative)\b")
SILENT = {"starts", "rfid", "central", "evcomm"}


def _read(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def overview() -> dict[str, Any]:
    latest = _read(settings.REPORTS / "latest.json") or {"policies": {}}
    gate = _read(settings.REPORTS / "gate.json") or {"passed": False, "checks": []}
    pols = {}
    for k, r in latest["policies"].items():
        label, desc = LABELS.get(k, (k, ""))
        pols[k] = r | {"label": label, "description": desc}
    days = next((r["days"] for r in pols.values()), 0)
    return {
        "generated": "cpa export-web",
        "days": days,
        "order": [k for k in LABELS if k in pols],
        "policies": pols,
        "gate": gate,
    }


def incidents(run: str = "autopilot", limit: int = 400) -> list[dict[str, Any]]:
    d = settings.RUNS / run
    incs = _read(d / "ops" / "incidents.json") or []
    meta = _read(d / "meta.json") or {}
    s0 = meta.get("scored_from", 0)
    shield = _read(d / "ops" / "shield.json") or []
    cmds = (
        pl.read_parquet(d / "observed" / "commands.parquet") if (d / "observed" / "commands.parquet").exists() else None
    )
    visits = (
        pl.read_parquet(d / "observed" / "visits.parquet") if (d / "observed" / "visits.parquet").exists() else None
    )
    ref = pl.read_parquet(settings.OBSERVED / "reference" / "chargers.parquet")
    info = {r["charger_id"]: r for r in ref.iter_rows(named=True)}
    episodes = (
        pl.read_parquet(d / "truth" / "episodes.parquet") if (d / "truth" / "episodes.parquet").exists() else None
    )
    by_inc_cmd: dict[str, list[dict[str, Any]]] = {}
    if cmds is not None:
        for r in cmds.filter(pl.col("incident").is_not_null()).iter_rows(named=True):
            by_inc_cmd.setdefault(r["incident"], []).append(r)
    by_inc_shield: dict[str, list[dict[str, Any]]] = {}
    for r in shield:
        if r.get("incident"):
            by_inc_shield.setdefault(r["incident"], []).append(r)
    out = []
    for i in incs:
        if i["opened"] < s0:
            continue
        cycles = _cycles(i, by_inc_cmd.get(i["id"], []), by_inc_shield.get(i["id"], []))
        tags = set()
        if any(c["shield"] and not c["shield"]["allowed"] for c in cycles):
            tags.add("blocked")
        if any(c["command"] and c["command"]["decided_by"] == "autonomous" for c in cycles):
            tags.add("autonomous")
        if SILENT & set(i["symptoms"]):
            tags.add("silent")
        if any(c["safety"] for c in cycles):
            tags.add("safety")
        if any(c["quarantined"] for c in cycles):
            tags.add("quarantined")
        top = max(i["first_probs"], key=i["first_probs"].get) if i.get("first_probs") else None
        vs = []
        if visits is not None:
            for v in visits.filter(
                (pl.col("charger_id") == i["charger_id"])
                & (pl.col("ts") >= i["opened"])
                & (pl.col("ts") <= (i["closed"] or 10**12) + 3600)
            ).iter_rows(named=True):
                vs.append(
                    {
                        "ts": v["ts"],
                        "outcome": v["outcome"],
                        "part_used": v["part_used"],
                        "part_needed": v["part_needed"],
                    }
                )
        c = info.get(i["charger_id"], {})
        out.append(
            {
                "id": i["id"],
                "charger_id": i["charger_id"],
                "model": c.get("model", ""),
                "country": c.get("country", ""),
                "opened": i["opened"],
                "closed": i["closed"],
                "resolution": i["resolution"],
                "symptoms": i["symptoms"],
                "preventive": i["preventive"],
                "top": {"cls": top, "p": round(i["first_probs"][top], 3)} if top else None,
                "reading": {
                    k: i["reading"].get(k) for k in ("first_action", "safety_concern", "confidence", "_quarantined")
                }
                if i.get("reading")
                else None,
                "cycles": cycles,
                "visits": vs,
                "tags": sorted(tags),
                "truth": _truth(episodes, i["charger_id"], i["opened"]),
            }
        )
    rank = {"blocked": 0, "safety": 1, "quarantined": 2, "autonomous": 3, "silent": 4}
    out.sort(
        key=lambda r: (
            min([rank.get(t, 9) for t in r["tags"]] + [9 if not r["preventive"] else 5]),
            r["opened"],
        )
    )
    return out[:limit]


def _cycles(i: dict[str, Any], cmds: list[dict[str, Any]], shield: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []
    for d in i["decisions"]:
        if "decision" in d:
            m = ACTION.match(d["decision"])
            cycles.append(
                {
                    "ts": d["ts"],
                    "diagnosis": d.get("top"),
                    "p": d.get("p"),
                    "reading": d.get("reader"),
                    "safety": bool(d.get("safety")),
                    "quarantined": int(d.get("quarantined") or 0),
                    "plan": d["decision"],
                    "action": m.group(1) if m else None,
                    "shield": None,
                    "route": None,
                    "command": None,
                    "outcome": None,
                }
            )
        elif "blocked" in d and cycles and cycles[-1]["action"] == d["blocked"]:
            cycles[-1]["shield"] = {"allowed": False, "rules": d["rules"]}
        elif "outcome" in d and cycles:
            for c in reversed(cycles):
                if c["action"] == d.get("action"):
                    c["outcome"] = d["outcome"]
                    break
    used = set()
    for c in cycles:
        if not c["action"] or c["shield"]:
            continue
        sh = next((s for s in shield if s["action"] == c["action"] and s["ts"] >= c["ts"] - 1), None)
        if sh is not None:
            c["shield"] = {
                "allowed": sh["allowed"],
                "rules": [r for r in (sh["blocked_by"] or "").split(",") if r],
            }
        cm = next(
            (k for k, x in enumerate(cmds) if k not in used and x["action"] == c["action"] and x["ts"] >= c["ts"]),
            None,
        )
        if cm is not None:
            used.add(cm)
            x = cmds[cm]
            c["command"] = {"result": x["result"], "decided_by": x["decided_by"]}
            c["route"] = "act" if x["decided_by"] == "autonomous" else "propose"
        elif c["shield"] and c["shield"]["allowed"]:
            c["route"] = "propose"  # proposed; the incident ended before an operator got to it
    return [c for c in cycles if c["action"] or c["plan"].startswith(("dispatch", "wait", "keep"))][-6:]


def _truth(episodes: pl.DataFrame | None, cid: str, at: int) -> str | None:
    if episodes is None:
        return None
    e = episodes.filter(
        (pl.col("charger_id") == cid) & (pl.col("start") <= at) & (pl.col("end").fill_null(10**12) >= at - 3600)
    ).sort("start")
    return e["kind"][-1] if e.height else "no fault"


def autonomy(run: str = "autopilot") -> dict[str, Any]:
    meta = _read(settings.RUNS / run / "meta.json") or {"report": {}}
    a = settings.ops()["autonomy"]
    return {
        "ladder": meta["report"].get("ladder", []),
        "thresholds": {
            "min_outcomes": a["promote"]["min_outcomes"],
            "wilson_lower": a["promote"]["wilson_lower"],
        },
        "never_autonomous": a["never_autonomous"],
    }


def models(run: str = "autopilot") -> dict[str, Any]:
    boot = _read(settings.REPORTS / "models" / "bootstrap.json") or {}
    life = _read(settings.RUNS / run / "ops" / "mlops.json") or {"events": [], "drift": [], "champion": {}}
    return {
        "bootstrap": {k: boot[k] for k in ("diagnosis", "risk") if k in boot},
        "events": life["events"],
        "drift": life["drift"],
        "champion": life["champion"],
        "psi_alert": settings.models()["drift"]["psi_alert"],
    }


def gateway(run: str = "autopilot") -> dict[str, Any]:
    cfg = settings.gateway()
    meta = _read(settings.RUNS / run / "meta.json") or {}
    s0, s1 = meta.get("scored_from", 0), meta.get("end", 1)
    ledger = _read(settings.RUNS / run / "ops" / "ledger.json") or []
    lg = pl.DataFrame(ledger, infer_schema_length=None).filter(pl.col("ts") >= s0) if ledger else None
    days = max((s1 - s0) / 86400, 1)
    out = []
    for name, uc in cfg["use_cases"].items():
        rows = lg.filter(pl.col("use_case") == name) if lg is not None else None
        n = rows.height if rows is not None else 0
        out.append(
            {
                "name": name,
                "owner": uc["owner"],
                "purpose": uc["purpose"],
                "route": uc["route"],
                "daily_budget_eur": uc["daily_budget_eur"],
                "tokens_per_minute": uc["tokens_per_minute"],
                "calls": n,
                "cache_hits": int((rows["outcome"] == "cache_hit").sum()) if n else 0,
                "cost_eur_per_day": float(rows["cost_eur"].sum()) / days if n else 0.0,
                "tokens_in": int(rows["tokens_in"].sum()) if n else 0,
                "tokens_out": int(rows["tokens_out"].sum()) if n else 0,
                "quarantined": int(rows["quarantined"].sum()) if n else 0,
                "masked": int(rows["masked"].sum()) if n else 0,
                "refused": int(
                    rows["outcome"].is_in(["over_budget", "rate_limited", "failed", "output_rejected"]).sum()
                )
                if n
                else 0,
            }
        )
    return {
        "provider": cfg["providers"]["default"],
        "use_cases": out,
        "note": "Here the provider is a deterministic stand-in (no model key in this environment), so token counts "
        "are estimates and costs are what those tokens would cost at list prices.",
    }


def export(dest: Path | None = None) -> list[str]:
    dest = dest or settings.ROOT / "web" / "public" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fn in (
        ("overview", overview),
        ("incidents", incidents),
        ("autonomy", autonomy),
        ("models", models),
        ("gateway", gateway),
    ):
        (dest / f"{name}.json").write_text(json.dumps(fn(), default=str))
        files.append(name)
    return files
