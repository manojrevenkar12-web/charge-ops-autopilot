"""The first champions: trained on the history, validated on its last four weeks, checked in ONNX,
registered with their model cards. Version 1 of each model is what the autopilot starts with."""

from __future__ import annotations

from typing import Any

import polars as pl

from .. import settings
from ..llm import lexicon
from ..ml import dataset, models
from .registry import Registry


def train_champions(registry: Registry, prep: dataset.Prepared, chargers: pl.DataFrame,
                    log: Any = print) -> dict[str, Any]:
    cfg = settings.models()
    ds = dataset.build(prep, chargers, lexicon.read)
    out: dict[str, Any] = {"incidents": ds["incidents"].height,
                           "labels": dict(ds["incidents"]["label"].value_counts().iter_rows())}
    for kind, train, col in (("diagnosis", models.train_diagnosis, "onset"), ("risk", models.train_risk, "hour")):
        data = ds[kind]
        m = train(data, cfg[kind])
        _, va = models._split(data, col, cfg[kind]["validation_days"])
        onx = models.to_onnx(m)
        checks = models.onnx_check(m, onx, va.tail(min(2000, va.height)))
        v = registry.register(m, onx, checks, {"stage": "champion", "trained_on": "history"})
        registry.set_alias(kind, "champion", v)
        out[kind] = {"version": v, "metrics": m.metrics, "checks": checks, "card": m.card}
        log(f"{kind} v{v}: {({k: round(x, 3) for k, x in m.metrics.items() if isinstance(x, float)})}")
    out["reference_rows"] = ds["diagnosis"]
    return out
