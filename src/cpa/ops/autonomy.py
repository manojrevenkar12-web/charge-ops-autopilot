"""The autonomy ladder: per action and hardware model, the autopilot starts by proposing (an
operator approves) and earns the right to act alone only from outcomes it observed itself. An
outcome is a success when the charger is back (online, not Faulted) with a good session within the
success window and no new symptom; this is observable, not the simulator's truth.

Promotion needs at least `min_outcomes` and a Wilson lower bound above `wilson_lower`; demotion
happens when the recent window's upper bound falls below `wilson_upper`. Some actions are never
autonomous."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .. import settings


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, min(c - h, p)), min(1.0, max(c + h, p))  # clamp rounding at k = 0 and k = n


@dataclass
class Rung:
    level: str = "propose"
    outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=400))
    changes: list[tuple[int, str]] = field(default_factory=list)


class Ladder:
    def __init__(self) -> None:
        self.cfg = settings.ops()["autonomy"]
        self.rungs: dict[tuple[str, str], Rung] = defaultdict(lambda: Rung(self.cfg["start_level"]))

    def level(self, action: str, model: str) -> str:
        if action in self.cfg["never_autonomous"]:
            return "propose"
        return self.rungs[(action, model)].level

    def observe(self, action: str, model: str, success: bool, now: int) -> None:
        r = self.rungs[(action, model)]
        r.outcomes.append(success)
        n, k = len(r.outcomes), sum(r.outcomes)
        if (
            r.level == "propose"
            and n >= self.cfg["promote"]["min_outcomes"]
            and action not in self.cfg["never_autonomous"]
        ):
            lo, _ = wilson(k, n)
            if lo >= self.cfg["promote"]["wilson_lower"]:
                r.level = "act"
                r.changes.append((now, "act"))
        elif r.level == "act":
            recent = list(r.outcomes)[-self.cfg["demote"]["window"] :]
            _, hi = wilson(sum(recent), len(recent))
            if len(recent) >= self.cfg["demote"]["window"] and hi < self.cfg["demote"]["wilson_upper"]:
                r.level = "propose"
                r.changes.append((now, "propose"))

    def snapshot(self) -> list[dict[str, Any]]:
        out = []
        for (a, m), r in sorted(self.rungs.items()):
            n, k = len(r.outcomes), sum(r.outcomes)
            lo, hi = wilson(k, n)
            out.append(
                {
                    "action": a,
                    "model": m,
                    "level": r.level,
                    "outcomes": n,
                    "successes": k,
                    "wilson_lower": round(lo, 3),
                    "wilson_upper": round(hi, 3),
                    "changes": [{"ts": t, "to": lv} for t, lv in r.changes],
                }
            )
        return out
