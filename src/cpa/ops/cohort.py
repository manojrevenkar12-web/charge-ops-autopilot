"""Fleet-level problems: a hardware model on one firmware version failing more than the same model
on other versions. Per-charger diagnosis cannot see these; resetting chargers one by one does not fix
them. The detector compares EV-communication failures per started session between a cohort and its
peers (an exact conditional binomial test) and, when one stands out, proposes halting the rollout
and rolling back in stages, which always needs a person's approval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import binomtest

from .. import settings
from .state import HOUR, FleetView


@dataclass
class CohortAlert:
    model: str
    version: str
    previous: str
    chargers: int
    rate: float
    peer_rate: float
    p_value: float
    raised: int
    approved: int | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)


def detect(view: FleetView, now: int) -> list[CohortAlert]:
    cfg = settings.ops()["detection"]["cohort"]
    since = now - cfg["window_days"] * 24 * HOUR
    fails: dict[tuple[str, str], int] = defaultdict(int)
    starts: dict[tuple[str, str], int] = defaultdict(int)
    members: dict[tuple[str, str], int] = defaultdict(int)
    for cv in view.chargers.values():
        k = (cv.model, cv.firmware)
        members[k] += 1
        fails[k] += sum(1 for t, _ in cv.evcomm_fails if t >= since)
        starts[k] += sum(1 for t in cv.starts if t >= since)
    out = []
    for (model, version), n in members.items():
        if n < cfg["min_chargers"] or starts[(model, version)] < 30:
            continue
        peers = [k for k in members if k[0] == model and k[1] != version]
        pf = sum(fails[k] for k in peers)
        ps = sum(starts[k] for k in peers)
        if ps < 100:
            continue
        f, s = fails[(model, version)], starts[(model, version)]
        rate, peer = f / s, (pf + 1) / (ps + 1)
        if rate < cfg["rate_ratio"] * peer:
            continue
        # given f+pf failures, how surprising is it that f of them fell on this cohort's share of sessions?
        p = binomtest(f, f + pf, s / (s + ps), alternative="greater").pvalue
        if p < cfg["alpha"]:
            prev = max((k[1] for k in peers), key=lambda v: members[(model, v)])
            out.append(CohortAlert(model, version, prev, n, rate, peer, float(p), now))
    return out
