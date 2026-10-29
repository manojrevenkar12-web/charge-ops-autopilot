"""The release gate over the evaluation (config/release.yaml)."""

from __future__ import annotations

import operator
from typing import Any

from . import settings

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}


def _get(d: dict[str, Any], path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def gate(results: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or settings.release()
    cand, ref = cfg["compare"]["candidate"], cfg["compare"]["reference"]
    checks = []
    for c in cfg["checks"]:
        run = c.get("run", cand)
        value = _get(results.get(run, {}), c["metric"])
        reference = _get(results.get(ref, {}), c["metric"])
        if value is None:
            checks.append({"id": c["id"], "passed": False, "detail": f"{run}: {c['metric']} missing"})
            continue
        if "vs_reference" in c:
            target = None if reference is None else reference + c["vs_reference"]
        elif "vs_reference_ratio" in c:
            target = None if reference is None else reference * c["vs_reference_ratio"]
        else:
            target = c["value"]
        ok = target is not None and OPS[c["op"]](value, target)
        fmt = (lambda x: f"{x:.4g}") if isinstance(value, float) else str
        checks.append(
            {
                "id": c["id"],
                "passed": bool(ok),
                "value": value,
                "target": target,
                "detail": f"{run} {c['metric']} = {fmt(value)} {c['op']} {fmt(target) if target is not None else '?'}",
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
