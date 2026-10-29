# 3. One LLM gateway, its settings generated as Azure API Management policies

Status: accepted

## Context
The AI unit owns the LLM gateway: routing, token limits, cost per use case, guardrails and logging.
EDRI runs on Azure, where API Management's AI gateway enforces token limits, token metrics and
semantic caching at the edge.

## Decision
Use cases are the unit of ownership (config/gateway.yaml): owner, route with fallback, token limit,
daily budget, PII masking, untrusted fields, output schema, cache. The in-process gateway enforces
them for every call (budgets reserve the worst case before a call and settle after it; a circuit
breaker skips failing models; spans carry OpenTelemetry GenAI attributes). `cpa apim` generates
the same settings as APIM policies (`llm-token-limit`, `llm-emit-token-metric`,
`llm-semantic-cache-*`, content safety), and CI fails if the committed policies are stale.

## Consequences
One source of truth for limits; the local gateway and APIM cannot drift apart silently. Budgets in
EUR per day are enforced in process and alerted on in Azure (cost budgets per use-case tag).
