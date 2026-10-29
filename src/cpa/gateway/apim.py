"""The gateway's use cases as Azure API Management AI-gateway policies (one API operation per use
case): token limits (`llm-token-limit`), token metrics with use-case dimensions
(`llm-emit-token-metric`), semantic caching where the use case allows it, content safety on the
prompt, and a backend pool in the route's priority order. Generated from config/gateway.yaml, so
the local gateway and APIM enforce the same numbers (tests/test_gateway.py checks it)."""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from .. import settings


def policy(use_case: str) -> str:
    cfg = settings.gateway()
    uc = cfg["use_cases"][use_case]
    root = ET.Element("policies")
    inbound = ET.SubElement(root, "inbound")
    ET.SubElement(inbound, "base")
    ET.SubElement(inbound, "set-header", {"name": "x-use-case", "exists-action": "override"}).append(_value(use_case))
    ET.SubElement(inbound, "llm-content-safety", {"backend-id": "content-safety", "shield-prompt": "true"})
    ET.SubElement(
        inbound,
        "llm-token-limit",
        {
            "counter-key": f'@("{use_case}")',
            "tokens-per-minute": str(uc["tokens_per_minute"]),
            "estimate-prompt-tokens": "true",
            "remaining-tokens-variable-name": "remainingTokens",
        },
    )
    metric = ET.SubElement(inbound, "llm-emit-token-metric", {"namespace": "charge-ops"})
    ET.SubElement(metric, "dimension", {"name": "use-case", "value": use_case})
    ET.SubElement(metric, "dimension", {"name": "owner", "value": uc["owner"]})
    ET.SubElement(
        metric,
        "dimension",
        {"name": "caller", "value": '@(context.Request.Headers.GetValueOrDefault("x-caller", "unknown"))'},
    )
    if uc.get("cache"):
        ET.SubElement(
            inbound,
            "llm-semantic-cache-lookup",
            {"score-threshold": "0.05", "embeddings-backend-id": "embeddings"},
        )
    remote = [m for m in uc["route"] if cfg["models"][m]["provider"] != "local"]
    ET.SubElement(inbound, "set-backend-service", {"backend-id": f"pool-{use_case}"})
    backend = ET.SubElement(root, "backend")
    ET.SubElement(backend, "forward-request", {"timeout": "30", "buffer-response": "true"})
    outbound = ET.SubElement(root, "outbound")
    ET.SubElement(outbound, "base")
    if uc.get("cache"):
        ET.SubElement(outbound, "llm-semantic-cache-store", {"duration": "3600"})
    on_error = ET.SubElement(root, "on-error")
    ET.SubElement(on_error, "base")
    root.insert(
        0,
        ET.Comment(
            f" generated from config/gateway.yaml use case {use_case}; backends in priority order: "
            f"{', '.join(remote)}; daily budget EUR {uc['daily_budget_eur']} is enforced by the "
            f"cost alert in azure/infra/main.bicep "
        ),
    )
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def _value(text: str) -> ET.Element:
    v = ET.Element("value")
    v.text = text
    return v


def export(out: str | Path) -> list[str]:
    d = Path(out)
    if not d.is_absolute():
        d = settings.ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for uc in settings.gateway()["use_cases"]:
        f = d / f"{uc}.xml"
        f.write_text(policy(uc))
        files.append(str(f.relative_to(settings.ROOT)))
    return files
