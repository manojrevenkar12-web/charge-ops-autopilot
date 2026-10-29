"""Azure ML managed online endpoint entry point (azure/ml/deployment.yml): the registered model's
folder holds model.onnx and meta.json, as the MLflow registry stores them."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import onnxruntime as ort
import polars as pl

from ..ml.models import Encoder

_model: dict[str, Any] = {}


def init() -> None:
    root = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    folder = next(p.parent for p in root.rglob("model.onnx"))
    meta = json.loads((folder / "meta.json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    _model.update(
        session=ort.InferenceSession((folder / "model.onnx").read_bytes(), so),
        encoder=Encoder.from_json(meta["encoder"]),
        classes=meta["classes"],
    )


def run(raw: str) -> dict[str, Any]:
    rows = json.loads(raw)["rows"]
    x = _model["encoder"].transform(pl.DataFrame(rows, infer_schema_length=None))
    out = _model["session"].run(None, {"x": x})
    p = out[1] if len(out) > 1 else out[0]
    return {"classes": _model["classes"], "probabilities": p.tolist()}
