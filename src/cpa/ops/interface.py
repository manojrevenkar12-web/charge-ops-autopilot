"""What an operations policy sees and what it may do. The simulator calls these; policies never
import the simulator. Everything a policy receives is something EDRI could observe: OCPP messages,
hotline complaints, technicians' visit reports and diagnostic logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import polars as pl


@dataclass
class Observation:
    now: int  # end of the hour just simulated (epoch seconds)
    messages: pl.DataFrame  # OCPP rows received in the hour (ocpp_flat.SCHEMA)
    complaints: pl.DataFrame  # ts, charger_id, lang, text
    visits: pl.DataFrame  # technicians' reports: visit_id, tech_id, charger_id, ts, outcome, part_used, part_needed, part_eta, lang, note, cost_eur
    logs: pl.DataFrame  # diagnostic logs uploaded after GetLog: ts, charger_id, text


@dataclass
class Command:
    charger_id: str
    action: str  # evse_reset, station_reset, unlock, set_inoperative, set_operative, firmware, get_log, halt_rollout
    evse: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    offset_s: int = 60  # seconds after the hour starts
    reason: str = ""
    incident: str | None = None
    decided_by: str = ""  # "autonomous", "operator", ...


@dataclass
class Stop:
    charger_id: str
    parts: list[str] = field(default_factory=list)
    incident: str | None = None


class Policy(Protocol):
    name: str

    def start(self, reference: dict[str, pl.DataFrame], t0: int) -> None: ...

    def on_hour(self, obs: Observation) -> list[Command]: ...

    def plan_day(self, day_start: int) -> dict[str, list[Stop]]: ...

    def report(self) -> dict[str, Any]: ...
