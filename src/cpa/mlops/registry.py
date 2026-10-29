"""Model registry on MLflow (SQLite locally; Azure ML's registry in production, see azure/ml).

A version carries the LightGBM model, its ONNX export, the encoder, metrics, the model card and the
results of its checks. Aliases say which version decides (`champion`), which is being watched
(`challenger`) and which decided before (`previous`, for rollback)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from .. import settings
from ..ml.models import Encoder, TrainedModel

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@dataclass
class ServedModel:
    """A registered version as the autopilot uses it: ONNX Runtime with the encoder."""

    name: str
    version: str
    classes: list[str]
    encoder: Encoder
    session: Any
    booster: lgb.Booster
    meta: dict[str, Any]

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if df.height == 0:
            return np.zeros((0, len(self.classes)))
        x = self.encoder.transform(df)
        out = self.session.run(None, {"x": x})
        p = out[1] if len(out) > 1 else out[0]
        return p if p.ndim == 2 and p.shape[1] == len(self.classes) else np.column_stack([1 - p, p])


class Registry:
    def __init__(self, root: Path | None = None) -> None:
        import mlflow

        self.root = root or settings.MODELS
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = f"sqlite:///{self.root / 'registry.db'}"
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_registry_uri(self.uri)
        self.mlflow = mlflow
        self.client = mlflow.MlflowClient(self.uri, self.uri)
        exp = settings.models()["registry"]["experiment"]
        e = self.client.get_experiment_by_name(exp)
        self.experiment = (
            e.experiment_id
            if e
            else self.client.create_experiment(exp, artifact_location=(self.root / "artifacts").as_uri())
        )

    def register(
        self, m: TrainedModel, onnx_bytes: bytes, checks: dict[str, Any], tags: dict[str, str]
    ) -> str:
        name = m.kind
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            m.booster.save_model(str(d / "model.txt"))
            (d / "model.onnx").write_bytes(onnx_bytes)
            meta = {
                "kind": m.kind,
                "classes": m.classes,
                "encoder": m.encoder.to_json(),
                "metrics": m.metrics,
                "card": m.card,
                "checks": checks,
            }
            (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
            run = self.client.create_run(self.experiment, tags={"model": name, **tags})
            rid = run.info.run_id
            for k, v in _flat(m.metrics).items():
                self.client.log_metric(rid, k, v)
            for f in d.iterdir():
                self.client.log_artifact(rid, str(f), "model")
            self.client.set_terminated(rid)
        try:
            self.client.get_registered_model(name)
        except Exception:
            self.client.create_registered_model(name, description=f"charge-ops {name} model")
        mv = self.client.create_model_version(name, f"runs:/{rid}/model", rid, tags=tags)
        return str(mv.version)

    def register_files(self, kind: str, folder: Path, tags: dict[str, str]) -> str:
        """Register a version from a saved folder (model.txt, model.onnx, meta.json)."""
        meta = json.loads((folder / "meta.json").read_text())
        run = self.client.create_run(self.experiment, tags={"model": kind, **tags})
        rid = run.info.run_id
        for k, v in _flat(meta.get("metrics", {})).items():
            self.client.log_metric(rid, k, v)
        for f in folder.iterdir():
            self.client.log_artifact(rid, str(f), "model")
        self.client.set_terminated(rid)
        try:
            self.client.get_registered_model(kind)
        except Exception:
            self.client.create_registered_model(kind, description=f"charge-ops {kind} model")
        return str(self.client.create_model_version(kind, f"runs:/{rid}/model", rid, tags=tags).version)

    def set_alias(self, name: str, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(name, alias, version)

    def alias(self, name: str, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(name, alias).version)
        except Exception:
            return None

    def load(self, name: str, version: str) -> ServedModel:
        import onnxruntime as ort

        mv = self.client.get_model_version(name, version)
        local = Path(
            self.mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="model", tracking_uri=self.uri
            )
        )
        meta = json.loads((local / "meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        sess = ort.InferenceSession(
            (local / "model.onnx").read_bytes(), so, providers=["CPUExecutionProvider"]
        )
        booster = lgb.Booster(model_file=str(local / "model.txt"))
        return ServedModel(
            name, version, meta["classes"], Encoder.from_json(meta["encoder"]), sess, booster, meta
        )

    def versions(self, name: str) -> list[dict[str, Any]]:
        out = []
        for mv in self.client.search_model_versions(f"name='{name}'"):
            out.append(
                {
                    "version": mv.version,
                    "tags": dict(mv.tags),
                    "aliases": list(mv.aliases),
                    "created": mv.creation_timestamp,
                }
            )
        return sorted(out, key=lambda r: int(r["version"]))


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, int | float) and v is not None:
            out[f"{prefix}{k}"] = float(v)
    return out
