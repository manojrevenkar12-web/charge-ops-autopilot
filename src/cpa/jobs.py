"""Azure ML job entry points (azure/ml/retrain-pipeline.yml): the same steps the autopilot runs in
process each Monday, as separate jobs over the OCPP record exported from PostgreSQL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from . import settings
from .llm import lexicon
from .ml import dataset, models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "train", "gate"])
    ap.add_argument("--observed")
    ap.add_argument("--prepared")
    ap.add_argument("--candidates")
    ap.add_argument("--out")
    ap.add_argument("--register")
    a = ap.parse_args()
    if a.step == "prepare":
        p = dataset.prepare(Path(a.observed))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for k in ("hourly", "status", "actions", "visits"):
            getattr(p, k).write_parquet(out / f"{k}.parquet")
        shutil.copy(Path(a.observed) / "reference" / "chargers.parquet", out / "chargers.parquet")
    elif a.step == "train":
        src = Path(a.prepared)
        p = dataset.Prepared(
            *(pl.read_parquet(src / f"{k}.parquet") for k in ("hourly", "status", "actions", "visits"))
        )
        ds = dataset.build(p, pl.read_parquet(src / "chargers.parquet"), lexicon.read)
        out = Path(a.out)
        cfg = settings.models()
        for kind, train in (("diagnosis", models.train_diagnosis), ("risk", models.train_risk)):
            m = train(ds[kind], cfg[kind])
            (out / kind).mkdir(parents=True, exist_ok=True)
            m.booster.save_model(str(out / kind / "model.txt"))
            (out / kind / "model.onnx").write_bytes(models.to_onnx(m))
            (out / kind / "meta.json").write_text(models.save_meta(m))
    elif a.step == "gate":
        from .mlops.registry import Registry

        reg = Registry()
        for kind in ("diagnosis", "risk"):
            folder = Path(a.candidates) / kind
            meta = json.loads((folder / "meta.json").read_text())
            v = reg.register_files(kind, folder, {"stage": a.register or "challenger"})
            reg.set_alias(kind, a.register or "challenger", v)
            print(kind, v, meta["metrics"])


if __name__ == "__main__":
    main()
