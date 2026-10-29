"""Incidents and their labels, from observable data only.

An incident opens at the first symptomatic hour and closes after a quiet day with at least one
good session in it (a broken charger is often quiet at night only because nobody comes). Its label is
what was seen to fix it: a technician's part (from the structured field or, when that is empty,
read from the note), a technician's restart, a remote action shortly before the symptoms stopped,
or nothing (it went away on its own). Labels are noisy on purpose: notes are sometimes wrong and
the baseline tried a station reset first almost everywhere, so "reset" is observed only where the
baseline tried one."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .signals import HOUR

QUIET_HOURS = 24
CLASSES = [
    "reset",
    "unlock",
    "firmware",
    "onsite_restart",
    "self_resolved",
    "part:lock_actuator",
    "part:power_module",
    "part:rfid_reader",
    "part:payment_terminal",
    "part:cable",
    "part:modem",
]
SYMPTOMS = ["sym_faulted", "sym_offline", "sym_starts", "sym_evcomm", "sym_rfid", "sym_central"]


def next_good(f: pl.DataFrame) -> pl.DataFrame:
    """For every charger-hour, the next later hour with a good session (null if none yet). Needs the full grid."""
    return f.with_columns(
        next_good=pl.when(pl.col("tx_good") > 0).then(pl.col("hour")).shift(-1).backward_fill().over("charger_id")
    )


def detect(f: pl.DataFrame, last_hour: int) -> pl.DataFrame:
    """Incidents from symptomatic charger-hours (with `next_good`): onset, last symptomatic hour, closed or not.
    A new incident starts only after the previous one closed: a quiet day containing a good session."""
    s = f.filter(pl.col("symptom")).select("charger_id", "hour", "next_good", *SYMPTOMS).sort("charger_id", "hour")
    s = s.with_columns(
        prev=pl.col("hour").shift(1).over("charger_id"),
        prev_good=pl.col("next_good").shift(1).over("charger_id"),
    )
    s = s.with_columns(
        new=(
            pl.col("prev").is_null()
            | (
                (pl.col("hour") - pl.col("prev") > QUIET_HOURS)
                & pl.col("prev_good").is_not_null()
                & (pl.col("prev_good") < pl.col("hour"))
            )
        )
    )
    s = s.with_columns(inc=pl.col("new").cast(pl.Int64).cum_sum().over("charger_id"))
    inc = s.group_by("charger_id", "inc").agg(
        onset=pl.col("hour").min(),
        last=pl.col("hour").max(),
        good_after=pl.col("next_good").last(),
        **{k: pl.col(k).first() for k in SYMPTOMS},
        **{f"{k}_any": pl.col(k).any() for k in SYMPTOMS},
    )
    return (
        inc.with_columns(
            closed=(pl.col("last") + QUIET_HOURS < last_hour)
            & pl.col("good_after").is_not_null()
            & (pl.col("good_after") < last_hour)
        )
        .drop("inc")
        .sort("onset")
    )


def label(
    incidents: pl.DataFrame,
    actions: pl.DataFrame,
    visits: pl.DataFrame,
    read_note: Callable[[str, str], dict[str, str | None]],
) -> pl.DataFrame:
    """Attach the observed remedy to each closed incident. `actions`: commands sent (charger_id, ts, action)."""
    actions = actions.select("charger_id", pl.col("ts").alias("at"), "action")
    v = visits.filter(pl.col("outcome") != "not_reached").select(
        "charger_id", pl.col("ts").alias("at"), "outcome", "part_used", "note", "lang"
    )
    notes = {}
    for r in v.filter(pl.col("part_used").is_null() & (pl.col("outcome") == "fixed")).iter_rows(named=True):
        notes[(r["charger_id"], r["at"])] = read_note(r["note"], r["lang"]).get("part")
    by_charger_a = actions.partition_by("charger_id", as_dict=True)
    by_charger_v = v.partition_by("charger_id", as_dict=True)
    labels = []
    for r in incidents.iter_rows(named=True):
        if not r["closed"]:
            labels.append(None)
            continue
        lo, hi = r["onset"] * HOUR, (r["good_after"] + 1) * HOUR  # up to the first good session after it
        lab = "self_resolved"
        va = by_charger_v.get((r["charger_id"],))
        fixed = None
        if va is not None:
            w = va.filter((pl.col("at") >= lo) & (pl.col("at") <= hi)).sort("at")
            for x in w.iter_rows(named=True):
                if x["outcome"] == "fixed":
                    part = x["part_used"] or notes.get((r["charger_id"], x["at"]))
                    fixed = f"part:{part}" if part else "onsite_restart"
        if fixed:
            lab = fixed
        else:
            aa = by_charger_a.get((r["charger_id"],))
            if aa is not None:
                w = aa.filter((pl.col("at") >= lo) & (pl.col("at") <= hi)).sort("at")
                if w.height:
                    lab = {"Reset": "reset", "UnlockConnector": "unlock", "UpdateFirmware": "firmware"}[w["action"][-1]]
        labels.append(lab)
    return incidents.with_columns(label=pl.Series(labels, dtype=pl.Utf8))


def risk_labels(f: pl.DataFrame, horizon_h: int = 48, at_hour_utc: int = 6) -> pl.DataFrame:
    """Daily rows (06:00 UTC) per charger that is not faulted now: does a connector go Faulted within 48 h?"""
    fut = f.select("charger_id", "hour", (pl.col("faulted_now") > 0).cast(pl.Int8).alias("f"))
    fut = fut.with_columns(
        ahead=pl.col("f")
        .reverse()
        .rolling_max(window_size=horizon_h, min_samples=1)
        .reverse()
        .shift(-1)
        .over("charger_id")
    )
    rows = f.filter((pl.col("hour") % 24 == at_hour_utc) & (pl.col("faulted_now") == 0) & (pl.col("hours_silent") < 2))
    return (
        rows.join(fut.select("charger_id", "hour", "ahead"), on=["charger_id", "hour"], how="left")
        .with_columns(y=pl.col("ahead").fill_null(0).cast(pl.Int8))
        .drop("ahead")
    )
