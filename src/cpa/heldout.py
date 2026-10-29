"""Runs a held-out set (heldout/<set>/fleets/*.yaml): each fleet in its own process and build folder,
with today's process and the autopilot over the same test weeks, judged by the same gate (the checks
that need other runs are left out). The first run of a set is written to
reports/heldout/<set>/first_run.json and never rewritten; later runs go to latest.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

from .settings import ROOT

RUNS = ["baseline", "autopilot"]


def run_world() -> dict[str, Any]:
    """Inside the fleet's own process (CPA_FLEET, CPA_BUILD and CPA_REPORTS are set)."""
    from . import pipeline, settings
    from .gate import gate

    pipeline.world(log=lambda *_: None)
    pipeline.train(log=lambda *_: None)
    pipeline.runs(RUNS, workers=1, log=lambda *_: None)
    res = pipeline.evaluate(RUNS)
    cfg = dict(settings.release())
    cfg["checks"] = [c for c in cfg["checks"] if c.get("run", cfg["compare"]["candidate"]) in RUNS]
    verdict = gate(res, cfg)
    out = {"evaluation": res, "gate": verdict}
    (settings.REPORTS / "heldout_world.json").write_text(json.dumps(out, default=str))
    return out


def run_set(name: str, log: Any = print) -> dict[str, Any]:
    fleets = sorted((ROOT / "heldout" / name / "fleets").glob("*.yaml"))
    if not fleets:
        raise SystemExit(f"no fleets in heldout/{name}/fleets")
    out: dict[str, Any] = {
        "set": name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fleets": {},
    }
    for f in fleets:
        build = ROOT / "build" / "heldout" / name / f.stem
        record = build / "record.json"
        if record.exists():  # resumable: a fleet already run keeps its record
            out["fleets"][f.stem] = json.loads(record.read_text())
            continue
        env = {
            **os.environ,
            "CPA_FLEET": str(f),
            "CPA_BUILD": str(build),
            "CPA_REPORTS": str(build / "reports"),
        }
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-c", "from cpa.heldout import run_world; run_world()"],
            env=env,
            capture_output=True,
            text=True,
        )
        rec: dict[str, Any] = {"seconds": round(time.perf_counter() - t0), "exit": p.returncode}
        res = build / "reports" / "heldout_world.json"
        if res.exists():
            rec |= json.loads(res.read_text())
        else:
            rec["error"] = (p.stderr or p.stdout)[-3000:]
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(rec, default=str))
        out["fleets"][f.stem] = rec
        state = "error" if "gate" not in rec else ("gate pass" if rec["gate"]["passed"] else "gate fail")
        log(f"{f.stem}: {state} ({rec['seconds']} s)")
    out["summary"] = summarise(out["fleets"])
    dest = ROOT / "reports" / "heldout" / name
    dest.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1, default=str) + "\n"
    (dest / "latest.json").write_text(text)
    if not (dest / "first_run.json").exists():
        (dest / "first_run.json").write_text(text)
    return out


def summarise(fleets: dict[str, Any]) -> dict[str, Any]:
    ok = {k: v for k, v in fleets.items() if "evaluation" in v}

    def pair(f: Any) -> list[dict[str, Any]]:
        rows = []
        for k, v in ok.items():
            b, a = v["evaluation"].get("baseline"), v["evaluation"].get("autopilot")
            if b and a:
                rows.append({"fleet": k, "baseline": f(b), "autopilot": f(a)})
        return rows

    return {
        "fleets": len(fleets),
        "completed": len(ok),
        "gate_passed": sum(1 for v in ok.values() if v["gate"]["passed"]),
        "failed_checks": {k: [c["id"] for c in v["gate"]["checks"] if not c["passed"]] for k, v in ok.items()},
        "true_availability": pair(lambda r: round(r["availability"]["true"], 4)),
        "charging_success": pair(lambda r: round(r["drivers"]["charging_success_rate"], 4)),
        "field_cost_eur": pair(lambda r: round(r["field"]["cost_eur"])),
        "harms": pair(lambda r: r["harms"]["total"]),
        "buggy_installs": pair(lambda r: r["firmware"]["buggy_installs"]),
    }
