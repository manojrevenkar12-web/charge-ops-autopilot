"""The two production models: diagnosis (incident -> what will fix it) and failure risk (charger ->
a connector goes Faulted within 48 h). LightGBM, encoded to a fixed numeric matrix so the same
model runs natively and in ONNX Runtime."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from .features import CATEGORICAL, NUMERIC

ONEHOT_MAX = 24


@dataclass
class Encoder:
    numeric: list[str]
    vocab: dict[str, list[str]]

    @classmethod
    def fit(cls, df: pl.DataFrame, extra: list[str] | None = None) -> Encoder:
        vocab = {}
        for c in CATEGORICAL:
            counts = df[c].fill_null("∅").value_counts(sort=True)
            vocab[c] = counts[c].head(ONEHOT_MAX).to_list()
        return cls(NUMERIC + (extra or []), vocab)

    @property
    def columns(self) -> list[str]:
        cols = list(self.numeric)
        for c, vs in self.vocab.items():
            cols += [f"{c}={v}" for v in vs] + [f"{c}=other"]
        return cols

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        parts = [df.select([pl.col(c).cast(pl.Float32).fill_null(np.nan) for c in self.numeric]).to_numpy()]
        for c, vs in self.vocab.items():
            col = df[c].fill_null("∅").to_numpy()
            oh = np.zeros((len(col), len(vs) + 1), np.float32)
            idx = {v: i for i, v in enumerate(vs)}
            for r, v in enumerate(col):
                oh[r, idx.get(v, len(vs))] = 1.0
            parts.append(oh)
        return np.hstack(parts).astype(np.float32)

    def to_json(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "vocab": self.vocab}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Encoder:
        return cls(d["numeric"], d["vocab"])


@dataclass
class TrainedModel:
    kind: str  # diagnosis | risk
    booster: lgb.Booster
    encoder: Encoder
    classes: list[str]
    metrics: dict[str, Any]
    card: dict[str, Any]

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        p = self.booster.predict(self.encoder.transform(df))
        return p if p.ndim == 2 else np.column_stack([1 - p, p])


def _split(df: pl.DataFrame, time_col: str, validation_days: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    cut = df[time_col].max() - validation_days * 24
    return df.filter(pl.col(time_col) < cut), df.filter(pl.col(time_col) >= cut)


def diagnosis_classes(labels: pl.Series, min_examples: int) -> list[str]:
    vc = labels.value_counts()
    keep = sorted(vc.filter(pl.col("count") >= min_examples)[labels.name].to_list())
    return [*keep, "other"]


def _fold(labels: pl.Series, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx.get(v, idx["other"]) for v in labels.to_list()])


def train_diagnosis(
    diag: pl.DataFrame,
    cfg: dict[str, Any],
    *,
    encoder: Encoder | None = None,
    classes: list[str] | None = None,
    validate: bool = True,
) -> TrainedModel:
    classes = classes or diagnosis_classes(diag["label"], cfg["min_class_examples"])
    tr, va = _split(diag, "onset", cfg["validation_days"]) if validate else (diag, diag.head(0))
    enc = encoder or Encoder.fit(tr, extra=["age"])
    params = dict(cfg["lightgbm"])
    weight = None
    if params.pop("class_weight", None) == "balanced":
        y = _fold(tr["label"], classes)
        freq = np.bincount(y, minlength=len(classes)).astype(float)
        weight = (len(y) / (len(classes) * np.maximum(freq, 1)))[y]
    n_est = params.pop("n_estimators")
    ds = lgb.Dataset(enc.transform(tr), _fold(tr["label"], classes), weight=weight, free_raw_data=False)
    booster = lgb.train(
        {
            **params,
            "objective": "multiclass",
            "num_class": len(classes),
            "seed": 7,
            "deterministic": True,
            "num_threads": 2,
        },
        ds,
        num_boost_round=n_est,
    )
    m = TrainedModel("diagnosis", booster, enc, classes, {}, {})
    m.metrics = evaluate_diagnosis(m, va) if va.height else {}
    m.card = {
        "trained_on": {
            "incidents": int(tr["onset"].n_unique()),
            "examples": tr.height,
            "from_hour": int(tr["onset"].min()),
            "to_hour": int(tr["onset"].max()),
            "labels": dict(tr["label"].value_counts().iter_rows()),
        },
        "validated_on": {"examples": va.height},
    }
    return m


def evaluate_diagnosis(m: TrainedModel, va: pl.DataFrame) -> dict[str, Any]:
    p = m.predict(va)
    y = _fold(va["label"], m.classes)
    pred = p.argmax(1)
    top2 = np.argsort(-p, 1)[:, :2]
    out = {
        "examples": len(y),
        "accuracy": float((pred == y).mean()),
        "top2": float(np.mean([y[i] in top2[i] for i in range(len(y))])),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1), labels=list(range(len(m.classes))))),
        "by_model": {},
    }
    for mdl in sorted(va["model"].unique().to_list()):
        sel = (va["model"] == mdl).to_numpy()
        if sel.sum() >= 15:
            out["by_model"][mdl] = float(
                f1_score(y[sel], pred[sel], average="macro", labels=np.unique(y[sel]))
            )
    return out


def train_risk(
    risk: pl.DataFrame, cfg: dict[str, Any], *, encoder: Encoder | None = None, validate: bool = True
) -> TrainedModel:
    tr, va = _split(risk, "hour", cfg["validation_days"]) if validate else (risk, risk.head(0))
    enc = encoder or Encoder.fit(tr)
    params = dict(cfg["lightgbm"])
    n_est = params.pop("n_estimators")
    ds = lgb.Dataset(enc.transform(tr), tr["y"].to_numpy(), free_raw_data=False)
    booster = lgb.train(
        {**params, "objective": "binary", "seed": 7, "deterministic": True, "num_threads": 2},
        ds,
        num_boost_round=n_est,
    )
    m = TrainedModel("risk", booster, enc, ["ok", "faults_within_48h"], {}, {})
    m.metrics = evaluate_risk(m, va) if va.height else {}
    m.card = {
        "trained_on": {
            "rows": tr.height,
            "positives": int(tr["y"].sum()),
            "from_hour": int(tr["hour"].min()),
            "to_hour": int(tr["hour"].max()),
        },
        "validated_on": {"rows": va.height},
    }
    return m


def evaluate_risk(m: TrainedModel, va: pl.DataFrame) -> dict[str, Any]:
    p = m.predict(va)[:, 1]
    y = va["y"].to_numpy()
    k = max(1, int(round(len(y) / max(va["hour"].n_unique(), 1) * 0.02)))  # top 2% of chargers each day
    days = va.with_columns(p=pl.Series(p)).with_columns(
        rank=pl.col("p").rank("ordinal", descending=True).over("hour")
    )
    top = days.filter(pl.col("rank") <= k)
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "base_rate": float(y.mean()),
        "precision_top2pct": float(top["y"].mean()),
        "top_k_per_day": k,
    }


def to_onnx(m: TrainedModel) -> bytes:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    n = len(m.encoder.columns)
    onx = convert_lightgbm(
        m.booster, initial_types=[("x", FloatTensorType([None, n]))], zipmap=False, target_opset=15
    )
    return onx.SerializeToString()


def onnx_check(
    m: TrainedModel, onnx_bytes: bytes, sample: pl.DataFrame, batch: int = 64, iterations: int = 200
) -> dict[str, float]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_bytes, so, providers=["CPUExecutionProvider"])
    x = m.encoder.transform(sample)
    native = m.predict(sample)
    out = sess.run(None, {"x": x})
    probs = out[1] if len(out) > 1 else out[0]
    probs = (
        probs
        if probs.ndim == 2 and probs.shape[1] == native.shape[1]
        else np.column_stack([1 - probs, probs])
    )
    absdiff = np.abs(probs - native).max(1)
    diff = float(absdiff.max())
    agree = float((probs.argmax(1) == native.argmax(1)).mean())
    p99 = float(np.percentile(absdiff, 99))
    xb = x[:batch] if len(x) >= batch else np.resize(x, (batch, x.shape[1]))
    lat_onnx, lat_native = [], []
    for _ in range(iterations):
        t = time.perf_counter()
        sess.run(None, {"x": xb})
        lat_onnx.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        m.booster.predict(xb, num_threads=1)
        lat_native.append((time.perf_counter() - t) * 1000)
    return {
        "max_abs_diff": diff,
        "p99_abs_diff": p99,
        "argmax_agreement": agree,
        "rows": len(x),
        "p50_ms_onnx": float(np.percentile(lat_onnx, 50)),
        "p95_ms_onnx": float(np.percentile(lat_onnx, 95)),
        "p50_ms_native": float(np.percentile(lat_native, 50)),
        "p95_ms_native": float(np.percentile(lat_native, 95)),
        "batch": batch,
    }


def save_meta(m: TrainedModel) -> str:
    return json.dumps(
        {
            "kind": m.kind,
            "classes": m.classes,
            "encoder": m.encoder.to_json(),
            "metrics": m.metrics,
            "card": m.card,
        }
    )
