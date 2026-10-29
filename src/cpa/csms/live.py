"""The autopilot on a live CSMS: every (simulated) hour, the messages the CSMS recorded become the
autopilot's observation, and its commands go out as real OCPP calls."""

from __future__ import annotations

import asyncio
from typing import Any

import polars as pl

from ..ops.interface import Observation
from .server import CSMS, Clock

HOUR = 3600
EMPTY = {
    "complaints": {"ts": pl.Int64, "charger_id": pl.Utf8, "lang": pl.Utf8, "text": pl.Utf8},
    "visits": {
        "visit_id": pl.Utf8,
        "tech_id": pl.Utf8,
        "charger_id": pl.Utf8,
        "ts": pl.Int64,
        "outcome": pl.Utf8,
        "part_used": pl.Utf8,
        "part_needed": pl.Utf8,
        "part_eta": pl.Int64,
        "lang": pl.Utf8,
        "note": pl.Utf8,
        "cost_eur": pl.Float64,
        "incident": pl.Utf8,
    },
    "logs": {"ts": pl.Int64, "charger_id": pl.Utf8, "text": pl.Utf8},
}


def reference(stations: list[dict[str, Any]]) -> dict[str, pl.DataFrame]:
    chargers = pl.DataFrame(
        [
            {
                "charger_id": s["id"],
                "site_id": s.get("site", "S-live"),
                "country": "DE",
                "model": s.get("model", "DC-150"),
                "vendor": "Nordvolt",
                "kind": "DC",
                "kw": 150,
                "connectors": 2,
                "firmware": s.get("firmware", "2.12.1"),
            }
            for s in stations
        ]
    )
    sites = pl.DataFrame(
        [
            {"site_id": sid, "country": "DE", "lat": 52.52, "lon": 13.40, "utc_offset": 1, "motorway": False}
            for sid in chargers["site_id"].unique().to_list()
        ]
    )
    techs = pl.DataFrame(
        [{"tech_id": "T-DE-1", "country": "DE", "lat": 52.52, "lon": 13.40, "utc_offset": 1, "lang": "de"}]
    )
    return {"chargers": chargers, "sites": sites, "technicians": techs}


class LiveLoop:
    def __init__(self, csms: CSMS, policy: Any, clock: Clock) -> None:
        self.csms, self.policy, self.clock = csms, policy, clock
        self.hour0 = clock.t0
        self.sent: list[tuple[str, str, str]] = []

    async def hour(self, h: int) -> None:
        lo, hi = self.hour0 + h * HOUR, self.hour0 + (h + 1) * HOUR
        while self.clock.now() < hi:
            await asyncio.sleep(0.005)
        msgs = await self.csms.store.between(lo, hi)
        obs = Observation(now=hi, messages=msgs, **{k: pl.DataFrame(schema=v) for k, v in EMPTY.items()})
        for cmd in self.policy.on_hour(obs):
            verdict = (
                self.policy.shield.log[-1] if getattr(self.policy, "shield", None) and self.policy.shield.log else {}
            )
            result = await self.csms.send(cmd, verdict)
            self.sent.append((cmd.charger_id, cmd.action, result))
