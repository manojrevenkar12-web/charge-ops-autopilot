"""What a new model version must show before it may decide anything (config/models.yaml promotion).

Offline: on the same most recent weeks, not worse than the champion overall and no hardware model
much worse. Export: ONNX gives the same answers and fits the latency budget. Online (in the
autopilot): shadow scoring against resolved incidents, then a canary with automatic rollback."""

from __future__ import annotations

from typing import Any

import polars as pl

from .. import settings
from ..ml import models as M


def offline(
    candidate: M.TrainedModel, champion: Any | None, validation: pl.DataFrame, checks: dict[str, Any]
) -> dict[str, Any]:
    cfg = settings.models()["promotion"]
    reasons: list[str] = []
    out: dict[str, Any] = {"candidate": {}, "champion": {}}
    if candidate.kind == "diagnosis":
        cand = M.evaluate_diagnosis(candidate, validation)
        out["candidate"] = cand
        if champion is not None:
            champ = _eval_served_diagnosis(champion, validation)
            out["champion"] = champ
            gain = cand["macro_f1"] - champ["macro_f1"]
            if gain < cfg["offline"]["diagnosis_min_gain_macro_f1"]:
                reasons.append(f"macro-F1 {cand['macro_f1']:.3f} below champion {champ['macro_f1']:.3f}")
            for mdl, f1 in champ["by_model"].items():
                c = cand["by_model"].get(mdl)
                if c is not None and f1 - c > cfg["offline"]["diagnosis_max_slice_drop"]:
                    reasons.append(f"{mdl}: macro-F1 drops {f1 - c:.3f}")
    else:
        cand = M.evaluate_risk(candidate, validation)
        out["candidate"] = cand
        if champion is not None:
            p = champion.predict(validation)[:, 1]
            from sklearn.metrics import average_precision_score

            champ = {"pr_auc": float(average_precision_score(validation["y"].to_numpy(), p))}
            out["champion"] = champ
            if cand["pr_auc"] - champ["pr_auc"] < cfg["offline"]["risk_min_gain_pr_auc"]:
                reasons.append(f"PR-AUC {cand['pr_auc']:.3f} below champion {champ['pr_auc']:.3f}")
    o = cfg["onnx"]
    if checks["argmax_agreement"] < o["min_argmax_agreement"]:
        reasons.append(f"ONNX agrees on {checks['argmax_agreement']:.4f} of rows")
    if checks["p99_abs_diff"] > o["max_p99_abs_diff"]:
        reasons.append(f"ONNX p99 difference {checks['p99_abs_diff']:.2e}")
    if checks["p95_ms_onnx"] > cfg["latency_p95_ms"]:
        reasons.append(f"p95 latency {checks['p95_ms_onnx']:.1f} ms over budget")
    out["passed"] = not reasons
    out["reasons"] = reasons
    return out


def _eval_served_diagnosis(champion: Any, va: pl.DataFrame) -> dict[str, Any]:
    """Score the champion's classes on the candidate's validation rows (labels folded into its classes)."""
    tm = M.TrainedModel("diagnosis", champion.booster, champion.encoder, champion.classes, {}, {})
    return M.evaluate_diagnosis(tm, va)
