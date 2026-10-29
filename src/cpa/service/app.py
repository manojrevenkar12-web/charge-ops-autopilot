"""The AI unit's service API for other teams (full-stack and AI deployment engineers): model scoring
and the LLM gateway behind one authenticated, metered HTTP surface.

- GET  /health                 versions of the models in use and the gateway provider
- POST /v1/diagnose            incident feature rows -> remedy probabilities (champion, ONNX Runtime)
- POST /v1/risk                charger feature rows -> probability a connector fails within 48 h
- GET  /v1/use-cases           the gateway's use cases, owners, budgets and limits
- POST /v1/ai/{use_case}       a model call through the gateway (masking, quarantine, budgets, limits)
- GET  /metrics                Prometheus text format
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from typing import Any

import polars as pl
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import settings
from ..gateway.core import Gateway, GatewayError

MAX_ROWS = 512


class Rows(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=MAX_ROWS)


class AiCall(BaseModel):
    fields: dict[str, Any]


def create_app(registry: Any = None, gateway: Gateway | None = None) -> FastAPI:
    app = FastAPI(title="charge-ops AI services", version="1.0")
    state: dict[str, Any] = {"models": {}, "gateway": gateway or Gateway()}
    counts: Counter[str] = Counter()
    latency: dict[str, list[float]] = defaultdict(list)
    token = os.environ.get("CPA_API_TOKEN")

    def models() -> dict[str, Any]:
        if not state["models"] and registry is not None:
            for kind in ("diagnosis", "risk"):
                v = registry.alias(kind, "champion")
                if v:
                    state["models"][kind] = registry.load(kind, v)
        return state["models"]

    @app.middleware("http")
    async def auth_and_meter(request: Request, call_next: Any) -> Any:
        if (
            token
            and request.url.path.startswith("/v1")
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return PlainTextResponse("missing or wrong bearer token", status_code=401)
        t = time.perf_counter()
        resp = await call_next(request)
        key = f"{request.method} {request.scope.get('route').path if request.scope.get('route') else request.url.path}"
        counts[f"{key} {resp.status_code}"] += 1
        latency[key].append((time.perf_counter() - t) * 1000)
        return resp

    @app.get("/health")
    def health() -> dict[str, Any]:
        m = models()
        return {
            "status": "ok",
            "models": {k: v.version for k, v in m.items()},
            "gateway_provider": state["gateway"].provider_name,
        }

    def score(kind: str, body: Rows) -> dict[str, Any]:
        m = models().get(kind)
        if m is None:
            raise HTTPException(503, f"no {kind} champion registered")
        df = pl.DataFrame(body.rows, infer_schema_length=None)
        missing = [c for c in m.encoder.numeric if c not in df.columns]
        if missing:
            raise HTTPException(422, f"missing features: {missing[:10]}")
        for c in m.encoder.vocab:
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, pl.Utf8).alias(c))
        p = m.predict(df)
        return {
            "model": kind,
            "version": m.version,
            "classes": m.classes,
            "probabilities": [[round(float(x), 5) for x in row] for row in p],
        }

    @app.post("/v1/diagnose")
    def diagnose(body: Rows) -> dict[str, Any]:
        return score("diagnosis", body)

    @app.post("/v1/risk")
    def risk(body: Rows) -> dict[str, Any]:
        return score("risk", body)

    @app.get("/v1/use-cases")
    def use_cases() -> list[dict[str, Any]]:
        cfg = settings.gateway()["use_cases"]
        return [
            {
                "name": k,
                "owner": v["owner"],
                "purpose": v["purpose"],
                "route": v["route"],
                "daily_budget_eur": v["daily_budget_eur"],
                "tokens_per_minute": v["tokens_per_minute"],
            }
            for k, v in cfg.items()
        ]

    @app.post("/v1/ai/{use_case}")
    def ai(use_case: str, body: AiCall, x_caller: str = Header("unknown")) -> dict[str, Any]:
        g: Gateway = state["gateway"]
        try:
            r = g.call(use_case, body.fields, now=time.time(), caller=x_caller)
        except GatewayError as e:
            code = {"unknown_use_case": 404, "rate_limited": 429, "over_budget": 429}.get(e.kind, 502)
            raise HTTPException(code, e.kind) from e
        return {
            "output": r.output,
            "model": r.model,
            "cached": r.cached,
            "cost_eur": r.cost_eur,
            "tokens": {"input": r.tokens_in, "output": r.tokens_out},
            "guard": {"masked": r.guard.masked, "quarantined": r.guard.quarantined},
            "trace_id": r.trace_id,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = ["# TYPE cpa_requests_total counter"]
        for k, n in sorted(counts.items()):
            method, path, status = k.split(" ")
            lines.append(f'cpa_requests_total{{method="{method}",path="{path}",status="{status}"}} {n}')
        lines.append("# TYPE cpa_request_ms summary")
        for k, xs in sorted(latency.items()):
            method, path = k.split(" ")
            xs = sorted(xs)
            lines.append(
                f'cpa_request_ms{{method="{method}",path="{path}",quantile="0.95"}} {xs[int(0.95 * (len(xs) - 1))]:.2f}'
            )
        lines.append("# TYPE cpa_gateway_cost_eur_total counter")
        cost: dict[str, float] = defaultdict(float)
        for r in state["gateway"].ledger:
            cost[r["use_case"]] += r["cost_eur"]
        for uc, c in sorted(cost.items()):
            lines.append(f'cpa_gateway_cost_eur_total{{use_case="{uc}"}} {c:.6f}')
        return "\n".join(lines) + "\n"

    return app
