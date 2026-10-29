"""History and test-period runs. The history is the fleet under today's process (the baseline);
the models learn from what it left behind. Each policy then runs the test period from the same
starting point with the same random draws; a warm-up of `warmup_days` before the test lets each
policy build up its own backlog, and only the test days are scored."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..ops.interface import Policy
from ..settings import OBSERVED, RUNS, TRUTH
from . import world as W
from .engine import Engine

WARMUP_DAYS = 14


def write_reference(w: W.World, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    w.chargers.select(["charger_id", "site_id", "country", "model", "vendor", "kind", "kw", "connectors",
                       "firmware"]).write_parquet(dest / "chargers.parquet")
    w.sites.select(["site_id", "country", "lat", "lon", "utc_offset", "motorway"]).write_parquet(dest / "sites.parquet")
    w.technicians.write_parquet(dest / "technicians.parquet")


def history(w: W.World, policy: Policy, log: Any = print) -> dict[str, Any]:
    t = time.perf_counter()
    e = Engine(w, policy, w.t0, w.history_end, "history")
    e.run()
    write_reference(w, OBSERVED / "reference")
    e.write(OBSERVED / "history", TRUTH / "history")
    meta = {"start": w.t0, "end": w.history_end, "seconds": round(time.perf_counter() - t, 1)}
    (OBSERVED / "history" / "meta.json").write_text(json.dumps(meta))
    log(f"history: {meta['seconds']} s")
    return meta


def test_run(w: W.World, policy: Policy, name: str, log: Any = print) -> dict[str, Any]:
    t = time.perf_counter()
    start = w.history_end - WARMUP_DAYS * W.DAY
    e = Engine(w, policy, start, w.end, name)
    e.run()
    out = RUNS / name
    e.write(out / "observed", out / "truth")
    meta = {"policy": name, "start": start, "scored_from": w.history_end, "end": w.end,
            "seconds": round(time.perf_counter() - t, 1), "report": policy.report()}
    (out / "meta.json").write_text(json.dumps(meta, default=str))
    log(f"{name}: {meta['seconds']} s")
    return meta
