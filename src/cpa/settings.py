"""Paths and configuration. Operations code reads config/ops.yaml, models.yaml, gateway.yaml and
release.yaml; only the simulator and the evaluation read config/fleet.yaml (the truth)."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
BUILD = Path(os.environ.get("CPA_BUILD", ROOT / "build"))
REPORTS = Path(os.environ.get("CPA_REPORTS", ROOT / "reports"))
OBSERVED = BUILD / "observed"  # what operations can see: reference tables and OCPP message logs
TRUTH = BUILD / "truth"  # only the simulator writes and only the evaluation reads
MODELS = BUILD / "models"  # model registry (MLflow) and exported artefacts
RUNS = BUILD / "runs"  # one directory per policy run over the test period


def _overrides() -> dict[str, Any]:
    raw = os.environ.get("CPA_SETTINGS_OVERRIDE")
    return yaml.safe_load(raw) if raw else {}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@cache
def load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG / f"{name}.yaml").read_text())
    return _merge(data, _overrides().get(name, {}))


def ops() -> dict[str, Any]:
    return load("ops")


def models() -> dict[str, Any]:
    return load("models")


def gateway() -> dict[str, Any]:
    return load("gateway")


def release() -> dict[str, Any]:
    return load("release")
