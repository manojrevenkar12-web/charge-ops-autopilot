"""Guardrails around every model call: personal data is masked before a prompt leaves, text from
chargers and drivers that carries instructions is quarantined, and outputs are checked against the
use case's schema and the action allowlist before anything downstream sees them."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import settings

SCHEMAS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "incident_reading": {
        "suspected": list,
        "safety_concern": bool,
        "confidence": (int, float),
        "evidence": list,
        "first_action": str,
    },
    "note_label": {"part": (str, type(None)), "cause": str, "confidence": (int, float)},
    "brief": {"text": str, "lang": str},
}


@dataclass
class GuardReport:
    masked: dict[str, int] = field(default_factory=dict)
    quarantined: list[str] = field(default_factory=list)
    output_errors: list[str] = field(default_factory=list)


def _cfg() -> dict[str, Any]:
    return settings.gateway()["guardrails"]


def mask_pii(text: str, report: GuardReport) -> str:
    pats = _cfg()["pii_patterns"]
    for name in ("email", "phone"):
        text, n = re.subn(pats[name], f"[{name}]", text)
        if n:
            report.masked[name] = report.masked.get(name, 0) + n
    if pats.get("name_hint"):
        # a capitalised first and last name right before masked contact details
        text, n = re.subn(r"\b[A-ZÀ-Ž][a-zà-ž]+ [A-ZÀ-Ž][a-zà-ž]+(?=,? \[(?:phone|email)\])", "[name]", text)
        text, n2 = re.subn(r"\b[a-zà-ž]+\.[a-zà-ž]+(?=@)", "[name]", text)
        if n + n2:
            report.masked["name"] = report.masked.get("name", 0) + n + n2
    return text


def injection(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _cfg()["injection"]["markers"])


def sanitise(fields: dict[str, Any], untrusted: list[str], report: GuardReport) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, str):
            v = mask_pii(v, report)
            if k in untrusted and injection(v):
                report.quarantined.append(k)
                v = "[quarantined: instruction-like text from an untrusted source]"
        elif isinstance(v, list):
            vv = []
            for i, x in enumerate(v):
                if isinstance(x, str):
                    x = mask_pii(x, report)
                    if k in untrusted and injection(x):
                        report.quarantined.append(f"{k}[{i}]")
                        x = "[quarantined: instruction-like text from an untrusted source]"
                vv.append(x)
            v = vv
        out[k] = v
    return out


def check_output(schema: str, out: Any, report: GuardReport) -> bool:
    spec = SCHEMAS[schema]
    if not isinstance(out, dict):
        report.output_errors.append("not an object")
        return False
    for k, t in spec.items():
        if k not in out:
            report.output_errors.append(f"missing {k}")
        elif not isinstance(out[k], t):
            report.output_errors.append(f"{k} has type {type(out[k]).__name__}")
    if _cfg()["output"]["reject_unknown_fields"]:
        extra = set(out) - set(spec)
        if extra:
            report.output_errors.append(f"unknown fields {sorted(extra)}")
    if schema == "incident_reading":
        allowed = set(_cfg()["output"]["allowed_actions"])
        if out.get("first_action") not in allowed:
            report.output_errors.append(f"action {out.get('first_action')!r} not allowed")
        for s in out.get("suspected", []):
            if not (isinstance(s, dict) and isinstance(s.get("remedy"), str) and isinstance(s.get("p"), int | float)):
                report.output_errors.append("bad suspected entry")
                break
    return not report.output_errors
