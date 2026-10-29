"""The LLM gateway. Every model call in the system goes through `Gateway.call(use_case, fields)`.

For each call: the use case must exist; personal data is masked and instruction-like untrusted text
is quarantined; the prompt is cut to the use case's input limit; the use case's tokens-per-minute
limit and daily budget are checked (the budget reserves the worst case before the call and settles
the actual cost after it); the answer may come from the cache; otherwise the route is tried in order,
skipping models whose circuit is open; the output must parse and pass the schema and action
allowlist. Each call is a span with OpenTelemetry GenAI attributes and a line in the usage ledger.

Clock: the gateway takes `now` from its caller, so the simulator's time drives budgets and limits.
`limits.RedisLimits` replaces the in-process counters when several gateway replicas share them."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .. import settings
from . import guard, prompts
from .providers import Provider, estimate_tokens, make

DAY = 86400
LATENCY_MODEL = {  # modelled (not measured) latency of real providers, for the simulated stand-in
    "claude-haiku-5-5": (0.45, 0.006), "claude-sonnet-5-5": (1.1, 0.018), "claude-opus-5-5": (1.8, 0.03),
    "distilled-notes": (0.002, 0.0),
}


class GatewayError(Exception):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass
class Result:
    output: dict[str, Any]
    model: str
    tokens_in: int
    tokens_out: int
    cost_eur: float
    latency_ms: float
    cached: bool
    guard: guard.GuardReport
    trace_id: str


@dataclass
class Limits:
    """In-process token-per-minute windows and daily budgets."""

    tpm: dict[str, deque[tuple[float, int]]] = field(default_factory=lambda: defaultdict(deque))
    spent: dict[tuple[str, int], float] = field(default_factory=lambda: defaultdict(float))
    reserved: dict[tuple[str, int], float] = field(default_factory=lambda: defaultdict(float))

    def take_tokens(self, uc: str, now: float, tokens: int, limit: int) -> bool:
        q = self.tpm[uc]
        while q and q[0][0] <= now - 60:
            q.popleft()
        if sum(t for _, t in q) + tokens > limit:
            return False
        q.append((now, tokens))
        return True

    def reserve(self, uc: str, now: float, amount: float, budget: float) -> bool:
        k = (uc, int(now // DAY))
        if self.spent[k] + self.reserved[k] + amount > budget:
            return False
        self.reserved[k] += amount
        return True

    def settle(self, uc: str, now: float, reserved: float, actual: float) -> None:
        k = (uc, int(now // DAY))
        self.reserved[k] -= reserved
        self.spent[k] += actual


class Gateway:
    def __init__(self, provider: str | None = None, limits: Limits | None = None, *, guards: bool = True,
                 fail: dict[str, float] | None = None) -> None:
        self.cfg = settings.gateway()
        self.provider_name = provider or self.cfg["providers"]["default"]
        self.providers: dict[str, Provider] = {}
        self.limits = limits or Limits()
        self.guards = guards
        self.cache: dict[str, tuple[float, dict[str, Any], int, int]] = {}
        self.ledger: list[dict[str, Any]] = []
        self.breaker: dict[str, tuple[int, float]] = {}  # model -> (failures, open until)
        self.fail = fail or {}  # test hook: probability a model call fails
        self._n = 0
        self.exporter = InMemorySpanExporter()
        tp = TracerProvider()
        tp.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = tp.get_tracer("cpa.gateway")
        self.local_models: dict[str, Any] = {}  # e.g. the distilled note classifier

    def provider_for(self, model: str) -> Provider:
        spec = self.cfg["models"][model]
        if spec["provider"] == "local":
            return self.local_models[model]
        name = self.provider_name
        if name not in self.providers:
            self.providers[name] = make(name)
        return self.providers[name]

    def price(self, model: str, tin: int, tout: int) -> float:
        m = self.cfg["models"][model]
        return (tin * m["input_usd"] + tout * m["output_usd"]) / 1e6 * self.cfg["usd_to_eur"]

    def call(self, use_case: str, fields: dict[str, Any], *, now: float, caller: str = "") -> Result:
        uc = self.cfg["use_cases"].get(use_case)
        if uc is None:
            raise GatewayError("unknown_use_case", use_case)
        rep = guard.GuardReport()
        self._clock = now
        self._n += 1
        trace_id = f"g{self._n:07d}"
        with self.tracer.start_as_current_span(f"chat {use_case}") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("cpa.use_case", use_case)
            span.set_attribute("cpa.caller", caller)
            span.set_attribute("cpa.prompt_version", prompts.VERSION)
            clean = guard.sanitise(fields, uc["untrusted_fields"], rep) if self.guards else dict(fields)
            span.set_attribute("cpa.guard.quarantined", len(rep.quarantined))
            span.set_attribute("cpa.guard.masked", sum(rep.masked.values()))
            prompt_tokens = estimate_tokens(prompts.render(use_case, clean))
            if prompt_tokens > uc["max_input_tokens"]:
                clean = self._shrink(use_case, clean, uc["max_input_tokens"])
                prompt_tokens = estimate_tokens(prompts.render(use_case, clean))
            key = hashlib.sha256(json.dumps([use_case, clean], sort_keys=True, default=str).encode()).hexdigest()
            if uc.get("cache") and key in self.cache:
                _, out, tin, tout = self.cache[key]
                res = Result(out, "cache", 0, 0, 0.0, 1.0, True, rep, trace_id)
                self._record(use_case, caller, now, res, "cache_hit")
                span.set_attribute("cpa.cached", True)
                return res
            if not self.limits.take_tokens(use_case, now, prompt_tokens + uc["max_output_tokens"],
                                           uc["tokens_per_minute"]):
                self._record(use_case, caller, now, None, "rate_limited")
                raise GatewayError("rate_limited", use_case)
            last_err = ""
            for model in self._route(use_case, uc):
                worst = self.price(model, prompt_tokens, uc["max_output_tokens"])
                if not self.limits.reserve(use_case, now, worst, uc["daily_budget_eur"]):
                    self._record(use_case, caller, now, None, "over_budget")
                    raise GatewayError("over_budget", use_case)
                t0 = time.perf_counter()
                try:
                    if self.fail.get(model, 0) and _unit(trace_id, model) < self.fail[model]:
                        raise RuntimeError("provider unavailable")
                    text, tin, tout = self.provider_for(model).complete(model, use_case, clean, uc["max_output_tokens"])
                except Exception as e:  # provider error: count it, try the next model
                    self.limits.settle(use_case, now, worst, 0.0)
                    self._trip(model, now)
                    last_err = str(e)
                    continue
                cost = self.price(model, tin, tout)
                self.limits.settle(use_case, now, worst, cost)
                base, per = LATENCY_MODEL.get(model, (0.5, 0.01))
                latency = (time.perf_counter() - t0) * 1000 + (
                    (base + per * tout) * 1000 if self.provider_name == "simulated" else 0)
                try:
                    out = json.loads(text)
                except json.JSONDecodeError:
                    out = None
                ok = guard.check_output(uc["output_schema"], out, rep) if self.guards else isinstance(out, dict)
                span.set_attribute("gen_ai.provider.name", self.provider_for(model).name)
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.response.model", model)
                span.set_attribute("gen_ai.usage.input_tokens", tin)
                span.set_attribute("gen_ai.usage.output_tokens", tout)
                span.set_attribute("cpa.cost_eur", cost)
                if not ok:
                    self._record(use_case, caller, now, Result({}, model, tin, tout, cost, latency, False, rep,
                                                               trace_id), "output_rejected")
                    last_err = "; ".join(rep.output_errors)
                    rep.output_errors = []
                    continue
                res = Result(out, model, tin, tout, cost, latency, False, rep, trace_id)
                if uc.get("cache"):
                    self.cache[key] = (now, out, tin, tout)
                self._record(use_case, caller, now, res, "ok")
                return res
            self._record(use_case, caller, now, None, "failed")
            raise GatewayError("all_routes_failed", last_err)

    def _route(self, use_case: str, uc: dict[str, Any]) -> list[str]:
        return [m for m in uc["route"] if self.breaker.get(m, (0, 0.0))[1] <= self._clock]

    _clock = 0.0

    def tick(self, now: float) -> None:
        self._clock = now

    def _trip(self, model: str, now: float) -> None:
        n, _ = self.breaker.get(model, (0, 0.0))
        n += 1
        self.breaker[model] = (0, now + 120) if n >= 3 else (n, 0.0)

    def _shrink(self, use_case: str, fields: dict[str, Any], limit: int) -> dict[str, Any]:
        """Cut the least important context first (later runbook passages, older complaints), then long text."""
        out = dict(fields)
        for k in ("passages", "complaints"):
            while estimate_tokens(prompts.render(use_case, out)) > limit and out.get(k):
                out[k] = out[k][:-1]
        for k in ("log", "tech_info", "note"):
            if isinstance(out.get(k), str) and estimate_tokens(prompts.render(use_case, out)) > limit:
                out[k] = out[k][: max(200, limit * 2)]
        return out

    def _record(self, uc: str, caller: str, now: float, r: Result | None, outcome: str) -> None:
        self.ledger.append({
            "ts": int(now), "use_case": uc, "caller": caller, "outcome": outcome,
            "model": r.model if r else None, "tokens_in": r.tokens_in if r else 0, "tokens_out": r.tokens_out if r else 0,
            "cost_eur": round(r.cost_eur, 6) if r else 0.0, "latency_ms": round(r.latency_ms, 1) if r else None,
            "quarantined": len(r.guard.quarantined) if r else 0,
            "masked": sum(r.guard.masked.values()) if r else 0, "trace_id": r.trace_id if r else None})

    def spans(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "attributes": dict(s.attributes or {}),
                 "duration_ms": (s.end_time - s.start_time) / 1e6 if s.end_time else None}
                for s in self.exporter.get_finished_spans()]


def _unit(*key: Any) -> float:
    return int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16) / 2**32
