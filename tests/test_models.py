"""Model encoding and the ONNX export, on a small synthetic table."""

import numpy as np
import polars as pl

from cpa.ml import features, models


def table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    d = {c: rng.poisson(1.0, n).astype(float) for c in features.NUMERIC}
    d |= {
        "model": rng.choice(["DC-150", "HPC-350"], n),
        "kind": rng.choice(["DC", "HPC"], n),
        "country": rng.choice(["DE", "IT"], n),
        "firmware": rng.choice(["1", "2"], n),
        "code": rng.choice(["0x2A07", None, "E-PWR-12"], n),
        "age": rng.choice([0.0, 2.0], n),
    }
    df = pl.DataFrame(d)
    lab = np.where(
        df["failed_starts_24"].to_numpy() > 1,
        "reset",
        np.where(df["code"].fill_null("").to_numpy() == "0x2A07", "part:lock_actuator", "self_resolved"),
    )
    return df.with_columns(label=pl.Series(lab), onset=pl.Series(np.arange(n) * 3))


def test_encoder_handles_unseen_categories():
    df = table()
    enc = models.Encoder.fit(df, extra=["age"])
    x = enc.transform(df.with_columns(model=pl.lit("NEW-MODEL")))
    assert x.shape[1] == len(enc.columns)
    assert x[:, enc.columns.index("model=other")].sum() == df.height


def test_diagnosis_learns_and_onnx_agrees():
    df = table()
    cfg = {
        "min_class_examples": 5,
        "validation_days": 5,
        "lightgbm": {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 40, "verbose": -1},
    }
    m = models.train_diagnosis(df, cfg)
    assert m.metrics["accuracy"] > 0.8
    chk = models.onnx_check(m, models.to_onnx(m), df.tail(200), iterations=5)
    assert chk["argmax_agreement"] >= 0.995 and chk["p99_abs_diff"] < 1e-3
