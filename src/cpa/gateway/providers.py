"""Model providers behind the gateway.

`simulated` is a deterministic stand-in used when no key is configured (this repository's CI): it
reads the same prompt fields a model would, with rules over the runbook passages it is given, and it
*follows instructions embedded in untrusted text* the way a naive model can, so the guardrails are
tested against something that would actually misbehave. `anthropic` and `azure_foundry` call real
models with the same prompts."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from ..llm import lexicon
from . import prompts


class Provider(Protocol):
    name: str

    def complete(self, model: str, use_case: str, fields: dict[str, Any], max_tokens: int) -> tuple[str, int, int]:
        """Returns (raw text, input tokens, output tokens)."""
        ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


REMEDY_HINTS = [
    (r"unlockconnector|unlock", "unlock"),
    (r"lock actuator|lock fault|locking pin|lock did not", "part:lock_actuator"),
    (r"power module|power stage|thermal derating", "part:power_module"),
    (r"rfid", "part:rfid_reader"),
    (r"payment terminal|acquirer", "part:payment_terminal"),
    (r"cable|inlet", "part:cable"),
    (r"reset the evse|reset the station|station reset", "reset"),
]
LOG_HINTS = [
    (r"\[wdt\]|watchdog|not scheduled|main loop stalled", "reset"),
    (r"\[rfid\]|rfid:", "part:rfid_reader"),
    (r"\[pt\]|pt:", "part:payment_terminal"),
    (r"\[pm2\]|pm:", "part:power_module"),
    (r"\[hlc\]|slac", "part:cable"),
    (r"\[lock\]|lock:", "part:lock_actuator"),
    (r"\[imd\]|imd:", "part:cable"),
    (r"\[net\]|modem:", "part:modem"),
]
COMPLAINT_HINTS = [
    (r"card does nothing|karte geht|tessera|pas werkt|kort fungerar|kort virker|karta nie|karta nefunguje|kártyám|cardul", "part:rfid_reader"),
    (r"payment|zahlung|pagamento|pinbetaling|kortbetalning|kortbetaling|płatność|platba|fizetés|plata", "part:payment_terminal"),
    (r"will not start|startet aber nicht|non parte|start niet|startar inte|starter ikke|nie startuje|nespustí|nem indul|nu pornește", "reset"),
    (r"stops after|bricht|si ferma|stopt na|avbryts|stopper efter|przerywa|přeruší|leáll|se oprește", "part:cable"),
]
SAFETY = re.compile(r"insulation|isolation|ground fault|imd|riso|earth fault|never reset|do not reset", re.I)
INJECTED_ACTION = [("set every charger", "set_operative"), ("station_reset", "station_reset"),
                   ("mark this incident resolved", "wait")]


class SimulatedProvider:
    name = "simulated"

    def complete(self, model: str, use_case: str, fields: dict[str, Any], max_tokens: int) -> tuple[str, int, int]:
        prompt = prompts.render(use_case, fields)
        tin = estimate_tokens(prompt)
        out = getattr(self, "_" + use_case.replace("-", "_"))(fields)
        text = json.dumps(out, ensure_ascii=False)
        return text, tin, estimate_tokens(text)

    def _incident_reader(self, f: dict[str, Any]) -> dict[str, Any]:
        untrusted = " ".join([f.get("tech_info") or "", f.get("log") or "", " ".join(f.get("complaints") or [])])
        low = untrusted.lower()
        for marker, action in INJECTED_ACTION:  # a naive model does what the text says
            if marker in low:
                return {"suspected": [], "safety_concern": False, "confidence": 0.9,
                        "evidence": ["instruction found in charger text"], "first_action": action}
        scores: dict[str, float] = {}
        evidence = []
        code = (f.get("tech_code") or "").strip()
        passages = f.get("passages") or []
        safety = False
        for p in passages:
            for line in p.splitlines():
                if code and code.lower() in line.lower():
                    evidence.append(line.strip()[:160])
                    if SAFETY.search(line):
                        safety = True
                    for pat, rem in REMEDY_HINTS:
                        if re.search(pat, line, re.I):
                            scores[rem] = scores.get(rem, 0) + 1.0
        for pat, rem in LOG_HINTS:
            if re.search(pat, (f.get("log") or ""), re.I):
                scores[rem] = scores.get(rem, 0) + 1.5
                evidence.append(f"log matches {pat}")
        for c in f.get("complaints") or []:
            for pat, rem in COMPLAINT_HINTS:
                if re.search(pat, c, re.I):
                    scores[rem] = scores.get(rem, 0) + 0.7
        if SAFETY.search(f.get("tech_info") or "") or SAFETY.search(f.get("log") or ""):
            safety = True
        total = sum(scores.values())
        suspected = [{"remedy": k, "p": round(v / total, 3)} for k, v in sorted(scores.items(), key=lambda kv: -kv[1])]
        conf = min(0.95, 0.35 + 0.15 * total) if total else 0.2
        first = "wait"
        if safety:
            first = "set_inoperative"
        elif suspected:
            top = suspected[0]["remedy"]
            first = {"unlock": "unlock", "reset": "evse_reset"}.get(top, "dispatch")
        return {"suspected": suspected[:4], "safety_concern": safety, "confidence": round(conf, 3),
                "evidence": evidence[:5], "first_action": first}

    def _note_labeler(self, f: dict[str, Any]) -> dict[str, Any]:
        r = lexicon.read(f.get("note") or "", f.get("lang") or "")
        conf = 0.95 if r["part"] else (0.7 if r["cause"] in ("none", "restart") else 0.3)
        return {"part": r["part"], "cause": r["cause"], "confidence": conf}

    def _dispatch_brief(self, f: dict[str, Any]) -> dict[str, Any]:
        parts = ", ".join(f.get("parts") or []) or "-"
        return {"text": f"{f.get('charger_id')}: {f.get('summary', '')} Parts: {parts}.", "lang": f.get("lang", "en")}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.Anthropic(base_url=os.environ.get("ANTHROPIC_BASE_URL"))

    def complete(self, model: str, use_case: str, fields: dict[str, Any], max_tokens: int) -> tuple[str, int, int]:
        r = self.client.messages.create(model=model, max_tokens=max_tokens, system=prompts.SYSTEM[use_case],
                                        messages=[{"role": "user", "content": prompts.render(use_case, fields)}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


class AzureFoundryProvider:
    """Claude or other models deployed in Microsoft Foundry, through its OpenAI-compatible endpoint."""

    name = "azure_foundry"

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=os.environ["AZURE_AI_ENDPOINT"], api_key=os.environ["AZURE_AI_KEY"])

    def complete(self, model: str, use_case: str, fields: dict[str, Any], max_tokens: int) -> tuple[str, int, int]:
        r = self.client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": prompts.SYSTEM[use_case]},
                      {"role": "user", "content": prompts.render(use_case, fields)}])
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens


def make(name: str) -> Provider:
    if name == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if name == "azure_foundry" and os.environ.get("AZURE_AI_KEY"):
        return AzureFoundryProvider()
    return SimulatedProvider()
