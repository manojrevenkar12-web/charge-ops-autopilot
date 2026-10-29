"""Training tables from what operations observed: labelled incidents with their features, and daily
failure-risk rows.

Raw OCPP messages are reduced once to three compact tables (hourly signals, status changes, commands
sent); training works from those, so retraining inside a run (history plus the autopilot's own weeks)
stays small."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .. import settings
from . import features, incidents, signals


@dataclass
class Prepared:
    hourly: pl.DataFrame
    status: pl.DataFrame
    actions: pl.DataFrame
    visits: pl.DataFrame

    @staticmethod
    def concat(parts: list[Prepared]) -> Prepared:
        return Prepared(
            *(
                pl.concat([getattr(p, k) for p in parts], how="diagonal_relaxed")
                for k in ("hourly", "status", "actions", "visits")
            )
        )


def actions_of(msgs: pl.DataFrame) -> pl.DataFrame:
    return msgs.filter(
        (pl.col("direction") == "cs") & pl.col("action").is_in(["Reset", "UnlockConnector", "UpdateFirmware"])
    ).select("charger_id", "ts", "action")


def prepare(observed: Path) -> Prepared:
    cache = observed / "prepared"
    if (cache / "hourly.parquet").exists():
        return Prepared(*(pl.read_parquet(cache / f"{k}.parquet") for k in ("hourly", "status", "actions", "visits")))
    m = pl.read_parquet(observed / "messages.parquet")
    p = Prepared(
        signals.hourly(m),
        signals.status_events(m),
        actions_of(m),
        pl.read_parquet(observed / "visits.parquet"),
    )
    cache.mkdir(exist_ok=True)
    for k in ("hourly", "status", "actions", "visits"):
        getattr(p, k).write_parquet(cache / f"{k}.parquet")
    return p


def build(
    p: Prepared, chargers: pl.DataFrame, read_note: Callable[[str, str], dict], until: int | None = None
) -> dict[str, pl.DataFrame | int]:
    hr, st, acts, visits = p.hourly, p.status, p.actions, p.visits
    if until is not None:
        hr = hr.filter(pl.col("hour") < until // signals.HOUR)
        st = st.filter(pl.col("received") < until)
        acts = acts.filter(pl.col("ts") < until)
        visits = visits.filter(pl.col("ts") < until)
    h0 = int(hr["hour"].min())
    h1 = int(hr["hour"].max()) + 1
    det = settings.ops()["detection"]
    horizon = settings.models()["risk"]["horizon_hours"]

    def finish(f: pl.DataFrame) -> pl.DataFrame:
        f = features.symptoms(f, det)
        f = incidents.next_good(f)
        f = f.with_columns(
            ahead=(pl.col("faulted_now") > 0)
            .cast(pl.Int8)
            .reverse()
            .rolling_max(window_size=horizon, min_samples=1)
            .reverse()
            .shift(-1)
            .over("charger_id")
        )
        keep = f.filter(pl.col("symptom") | (pl.col("hour") % 24 == 6))
        return keep.select(
            "charger_id",
            "hour",
            "symptom",
            *incidents.SYMPTOMS,
            "ahead",
            "next_good",
            "info",
            *features.FEATURES,
        ).with_columns(pl.col(pl.Float64).cast(pl.Float32))

    f = features.build_chunked(hr, st, chargers, h0, h1, finish)
    inc = incidents.detect(f, h1)
    lab = incidents.label(inc, acts, visits, read_note).filter(pl.col("label").is_not_null())
    offsets = settings.models()["diagnosis"]["samples_per_incident"]
    ex = []
    for k in offsets:
        e = lab.select("charger_id", "onset", "label", "last").with_columns(hour=pl.col("onset") + k)
        e = e.filter(pl.col("hour") <= pl.col("last"))
        ex.append(e.join(f, on=["charger_id", "hour"], how="inner").with_columns(age=pl.lit(float(k))))
    diag = pl.concat(ex, how="diagonal_relaxed").sort("onset")
    risk = (
        f.filter(
            (pl.col("hour") % 24 == 6)
            & (pl.col("faulted_now") == 0)
            & (pl.col("hours_silent") < 2)
            & (pl.col("hour") + horizon < h1)
        )
        .with_columns(y=pl.col("ahead").fill_null(0))
        .sort("hour")
    )
    return {"incidents": lab, "diagnosis": diag, "risk": risk, "h0": h0, "h1": h1}
