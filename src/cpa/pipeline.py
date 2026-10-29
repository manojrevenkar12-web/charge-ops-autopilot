"""The whole run: world and history, the first champions, every policy over the same test weeks,
the evaluation against the truth, and the release gate."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl

from . import settings

POLICIES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "assist": {"autonomy": False},                       # same brain, every action waits for an operator
    "autopilot": {},
    "autopilot-fifo": {"vrp": False},                    # today's oldest-first dispatch instead of routing
    "autopilot-unshielded": {"shield": False, "guards": False},  # the planner alone, without shield or guardrails
    "llm-decides": {"shield": False, "guards": False, "llm_decides": True, "retrain": False},  # positive control
}


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def world(log: Any = print) -> dict[str, Any]:
    from .ops.baseline import BaselinePolicy
    from .sim import runner
    from .sim import world as W

    w = W.build()
    for d in (settings.OBSERVED, settings.TRUTH):
        if d.exists():
            shutil.rmtree(d)
    return runner.history(w, BaselinePolicy(), log=log)


def train(log: Any = print) -> dict[str, Any]:
    from .ml import dataset
    from .mlops.bootstrap import train_champions
    from .mlops.registry import Registry

    if settings.MODELS.exists():
        shutil.rmtree(settings.MODELS)
    reg = Registry(settings.MODELS)
    prep = dataset.prepare(settings.OBSERVED / "history")
    ch = pl.read_parquet(settings.OBSERVED / "reference" / "chargers.parquet")
    out = train_champions(reg, prep, ch, log=log)
    _json(settings.REPORTS / "models" / "bootstrap.json", out)
    return out


def run_policy(name: str) -> dict[str, Any]:
    """One policy over the test weeks, in its own process with its own registry seeded with the champions."""
    from .ml import dataset
    from .mlops.lifecycle import ModelOps
    from .mlops.registry import Registry
    from .ops.autopilot import Autopilot
    from .ops.baseline import BaselinePolicy
    from .sim import runner
    from .sim import world as W

    w = W.build()
    if name == "baseline":
        policy: Any = BaselinePolicy()
    else:
        root = settings.RUNS / name / "models"
        if root.exists():
            shutil.rmtree(root)
        reg = Registry(root)
        for kind in ("diagnosis", "risk"):
            v = reg.register_files(kind, settings.MODELS / "bootstrap" / kind, {"stage": "champion", "trained_on": "history"})
            reg.set_alias(kind, "champion", v)
        ch = pl.read_parquet(settings.OBSERVED / "reference" / "chargers.parquet")
        mo = ModelOps(reg, dataset.prepare(settings.OBSERVED / "history"), ch)
        mo.set_reference_rows(pl.read_parquet(settings.MODELS / "bootstrap" / "reference_rows.parquet"))
        policy = Autopilot(name, models_from=mo, **POLICIES[name])
    meta = runner.test_run(w, policy, name, log=lambda *_: None)
    if hasattr(policy, "models_from") and policy.models_from is not None:
        _json(settings.RUNS / name / "ops" / "mlops.json", policy.models_from.report())
    return meta


def runs(names: list[str] | None = None, workers: int = 2, log: Any = print) -> dict[str, Any]:
    names = names or list(POLICIES)
    out = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        futs = {n: ex.submit(run_policy, n) for n in names}
        for n, f in futs.items():
            out[n] = f.result()
            log(f"{n}: {out[n]['seconds']} s")
    return out


def evaluate(names: list[str] | None = None, log: Any = print) -> dict[str, Any]:
    from .evaluation import evaluate_all

    res = evaluate_all(names or list(POLICIES))
    for r in res.values():
        r["harms"]["total"] = r["harms"]["interrupted_sessions"] + r["harms"]["unsafe_resets"] + \
            r["harms"].get("injected_actions", 0)
    _json(settings.REPORTS / "latest.json", {"generated_by": "cpa run", "policies": res})
    return res


def run(log: Any = print) -> dict[str, Any]:
    from .gate import gate

    t = time.perf_counter()
    log("world and history")
    world(log)
    log("first champions")
    train(log)
    log("policies over the test weeks")
    runs(log=log)
    log("evaluation")
    res = evaluate(log=log)
    verdict = gate(res)
    _json(settings.REPORTS / "gate.json", verdict)
    log(f"gate: {'pass' if verdict['passed'] else 'FAIL'} ({round(time.perf_counter() - t)} s)")
    for c in verdict["checks"]:
        log(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['id']}: {c['detail']}")
    return {"evaluation": res, "gate": verdict}
