"""The fleet and everything random about it, drawn once.

Every policy runs the test period against the same draws (common random numbers): the same
faults arrive at the same chargers at the same minutes, the same drivers turn up, and the k-th
remote action on an episode succeeds or fails on the same uniform number. Differences between
policies are then differences in what they did, not in luck."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import yaml

from ..settings import CONFIG

HOUR = 3600
DAY = 86400
MAX_PER_HOUR = 6  # session arrivals per charger and hour are capped here (rarely binding)
SESSION_FIELDS = 12  # uniforms per arrival: minute, auth, ev make, energy, duration, failure, ...
REMEDY_DRAWS = 12

CITIES: dict[str, list[tuple[float, float]]] = {
    "DE": [(52.52, 13.40), (48.14, 11.58), (50.94, 6.96), (53.55, 9.99)],
    "GB": [(51.51, -0.13), (52.49, -1.89), (53.48, -2.24)],
    "IT": [(45.46, 9.19), (41.90, 12.50), (44.49, 11.34)],
    "NL": [(52.37, 4.90), (51.92, 4.48)],
    "SE": [(59.33, 18.07), (57.71, 11.97), (55.60, 13.00)],
    "DK": [(55.68, 12.57), (56.16, 10.20)],
    "PL": [(52.23, 21.01), (50.06, 19.94), (52.41, 16.93)],
    "CZ": [(50.08, 14.44), (49.20, 16.61)],
    "AT": [(48.21, 16.37), (47.07, 15.44), (48.31, 14.29)],
    "HU": [(47.50, 19.04), (47.69, 17.63)],
    "RO": [(44.43, 26.10), (46.77, 23.59)],
}


@cache
def truth_config(path: str | None = None) -> dict[str, Any]:
    p = path or os.environ.get("CPA_FLEET") or str(CONFIG / "fleet.yaml")  # held-out fleets set CPA_FLEET
    return yaml.safe_load(open(p).read())


@dataclass(eq=False)
class Episode:
    """One fault on one charger. Start and natural end are drawn up front; remedies may end it sooner."""

    id: int
    charger: int
    kind: str
    start: int
    evse: int  # 0 = whole station
    natural_end: float  # epoch seconds, inf when it never ends on its own
    onset_end: int = 0  # power module: degradation until this time, then the module trips
    remedy_u: np.ndarray = field(default_factory=lambda: np.zeros(REMEDY_DRAWS))
    attempts: int = 0
    end: float = math.inf
    end_reason: str = ""
    relief_until: int = 0
    injection: bool = False


@dataclass
class World:
    cfg: dict[str, Any]
    t0: int
    hours: int
    sites: pl.DataFrame
    chargers: pl.DataFrame
    technicians: pl.DataFrame
    lam: np.ndarray  # expected arrivals per charger and hour of the horizon, shape (hours, chargers)
    episodes: list[Episode]
    outages: list[tuple[int, int, int]]  # (site index, start, end)
    rollout_day: dict[tuple[str, str], np.ndarray]  # (model, version) -> day per charger (inf = never)
    seed: int

    @property
    def history_end(self) -> int:
        return self.t0 + self.cfg["history_days"] * DAY

    @property
    def end(self) -> int:
        return self.t0 + self.hours * HOUR

    def session_draws(self, hour: int) -> tuple[np.ndarray, np.ndarray]:
        """Arrival counts and per-arrival uniforms for one hour of the horizon: identical for every policy."""
        g = np.random.Generator(np.random.Philox(key=[self.seed, 1_000_003 + hour]))
        counts = np.minimum(g.poisson(self.lam[hour]), MAX_PER_HOUR)
        u = g.random((len(self.lam[hour]), MAX_PER_HOUR, SESSION_FIELDS))
        return counts, u

    def hour_uniforms(self, hour: int, n: int) -> np.ndarray:
        g = np.random.Generator(np.random.Philox(key=[self.seed, 2_000_003 + hour]))
        return g.random((n, 8))


def _haversine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.radians, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def build(cfg: dict[str, Any] | None = None) -> World:
    cfg = cfg or truth_config()
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    t0 = int(datetime.fromisoformat(cfg["start"]).replace(tzinfo=UTC).timestamp())
    days = cfg["history_days"] + cfg["test_days"]
    hours = days * 24
    fleet = cfg["fleet"]
    countries = list(fleet["countries"])
    shares = np.array([fleet["countries"][c]["share"] for c in countries])
    n = int(fleet["chargers"])

    # sites: clusters around cities, a third of them on motorways further out
    site_rows: list[dict[str, Any]] = []
    charger_rows: list[dict[str, Any]] = []
    model_names = list(cfg["models"])
    model_share = np.array([cfg["models"][m]["share"] for m in model_names])
    lo, hi = fleet["chargers_per_site"]
    while len(charger_rows) < n:
        c = countries[rng.choice(len(countries), p=shares / shares.sum())]
        city = CITIES[c][rng.integers(len(CITIES[c]))]
        far = rng.random() < 0.33
        dist = rng.uniform(40, fleet["site_radius_km"]) if far else abs(rng.normal(0, 18))
        bearing = rng.uniform(0, 2 * np.pi)
        lat = city[0] + dist / 111 * np.cos(bearing)
        lon = city[1] + dist / (111 * np.cos(np.radians(city[0]))) * np.sin(bearing)
        sid = len(site_rows)
        k = int(rng.integers(lo, hi + 1))
        hpc_site = far and rng.random() < 0.6
        site_rows.append(
            {
                "site_id": f"S{sid:04d}",
                "site": sid,
                "country": c,
                "lat": round(float(lat), 5),
                "lon": round(float(lon), 5),
                "utc_offset": fleet["countries"][c]["utc"],
                "demand": float(np.exp(rng.normal(0, cfg["drivers"]["site_demand_sd"]))),
                "motorway": far,
            }
        )
        for _ in range(k):
            if len(charger_rows) >= n:
                break
            p = model_share.copy()
            if hpc_site:
                p[model_names.index("HPC-350")] *= 3
            m = model_names[rng.choice(len(model_names), p=p / p.sum())]
            spec = cfg["models"][m]
            idx = len(charger_rows)
            charger_rows.append(
                {
                    "charger_id": f"{c}-{idx:04d}",
                    "charger": idx,
                    "site_id": f"S{sid:04d}",
                    "site": sid,
                    "country": c,
                    "model": m,
                    "vendor": spec["vendor"],
                    "kind": spec["kind"],
                    "kw": spec["kw"],
                    "connectors": spec["connectors"],
                    "firmware": spec["firmware"][0],
                }
            )
    sites = pl.DataFrame(site_rows)
    chargers = pl.DataFrame(charger_rows)

    # technicians live in the cities of their country
    tech_rows = []
    for c in countries:
        for j in range(fleet["countries"][c]["technicians"]):
            base = CITIES[c][j % len(CITIES[c])]
            tech_rows.append(
                {
                    "tech_id": f"T-{c}-{j + 1}",
                    "country": c,
                    "lat": base[0],
                    "lon": base[1],
                    "utc_offset": fleet["countries"][c]["utc"],
                    "lang": fleet["countries"][c]["lang"],
                }
            )
    technicians = pl.DataFrame(tech_rows)

    # expected arrivals per charger-hour: model rate x site demand x hour-of-day (local) x weekend
    prof = np.array(cfg["drivers"]["hourly_profile"], dtype=float)
    prof = prof / prof.mean()
    site_demand = sites["demand"].to_numpy()[chargers["site"].to_numpy()]
    per_day = np.array([cfg["models"][m]["sessions_per_day"] for m in chargers["model"]])
    offset = (
        chargers["country"].replace_strict({c: fleet["countries"][c]["utc"] for c in countries}).to_numpy()
    )
    h = np.arange(hours)
    local = (h[:, None] + offset[None, :]) % 24
    dow = ((h[:, None] + offset[None, :]) // 24) % 7  # t0 is a Monday
    weekend = np.where(dow >= 5, cfg["drivers"]["weekend_factor"], 1.0)
    lam = (per_day * site_demand)[None, :] / 24 * prof[local] * weekend
    lam = lam.astype(np.float32)

    # fault episodes over the whole horizon
    episodes: list[Episode] = []
    models_arr = chargers["model"].to_list()
    for kind, f in cfg["faults"].items():
        if kind == "grid_outage":
            continue
        for ci, m in enumerate(models_arr):
            rate = f["hazard"].get(m, 0.0) * cfg.get("hazard_scale", 1.0) / (30 * DAY)
            if rate <= 0:
                continue
            t = t0 + rng.exponential(1 / rate)
            while t < t0 + hours * HOUR:
                nat = f.get("natural_end_hours")
                evse = int(rng.integers(1, 3)) if kind in ("connector_lock", "cable_wear") else 0
                ep = Episode(
                    id=len(episodes),
                    charger=ci,
                    kind=kind,
                    start=int(t),
                    evse=evse,
                    natural_end=t + rng.exponential(nat * HOUR) if nat else math.inf,
                    remedy_u=rng.random(REMEDY_DRAWS),
                    injection=bool(rng.random() < cfg["untrusted_text"]["injection_share"]),
                )
                if kind == "power_module":
                    a, b = f["onset_days"]
                    ep.onset_end = int(t + rng.uniform(a, b) * DAY)
                episodes.append(ep)
                t += rng.exponential(1 / rate)
    episodes.sort(key=lambda e: e.start)
    for i, e in enumerate(episodes):
        e.id = i

    go = cfg["faults"]["grid_outage"]
    outages = []
    srate = go["hazard_per_site"] * cfg.get("hazard_scale", 1.0) / (30 * DAY)
    for s in range(len(sites)):
        t = t0 + rng.exponential(1 / srate)
        while t < t0 + hours * HOUR:
            d = rng.uniform(*go["duration_hours"]) * HOUR
            outages.append((s, int(t), int(t + d)))
            t += d + rng.exponential(1 / srate)

    rollout_day: dict[tuple[str, str], np.ndarray] = {}
    for r in cfg["firmware_rollouts"]:
        days_arr = np.full(n, np.inf)
        sel = np.array([m == r["model"] for m in models_arr])
        days_arr[sel] = r["day"] + rng.geometric(r["share_per_day"], sel.sum()) - 1
        rollout_day[(r["model"], r["version"])] = days_arr

    return World(
        cfg=cfg,
        t0=t0,
        hours=hours,
        sites=sites,
        chargers=chargers,
        technicians=technicians,
        lam=lam,
        episodes=episodes,
        outages=outages,
        rollout_day=rollout_day,
        seed=seed,
    )


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(_haversine(np.array(a), np.array(b)))
