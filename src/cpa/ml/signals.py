"""Hourly signals per charger from OCPP messages, bucketed by when the CSMS *received* them.

The same function runs offline over months of logs (training) and online over the last hour
(the autopilot), so training and serving compute features the same way; tests/test_features.py
checks that the online records equal the offline ones for a whole run."""

from __future__ import annotations

import polars as pl

HOUR = 3600

SIGNALS = [
    "msgs",
    "heartbeats",
    "auth_rfid",
    "auth_central",
    "remote_ok",
    "tx_started",
    "tx_good",
    "evcomm_fail",
    "interrupted",
    "watchdog",
    "problem",
    "boots",
    "temp_max",
    "power_sum",
    "power_n",
    "energy_kwh",
    "faulted_events",
    "available_events",
    "resets",
    "unlocks",
]


def hourly(msgs: pl.DataFrame) -> pl.DataFrame:
    """One row per charger and hour (epoch hour of `received`) with counts of what happened."""
    a, d = pl.col("action"), pl.col("direction")
    ended = (a == "TransactionEvent") & (pl.col("event_type") == "Ended")
    evfail = ended & (
        (pl.col("trigger_reason") == "EVCommunicationLost")
        | ((pl.col("stopped_reason") == "Other") & (pl.col("energy_wh").fill_null(0) < 100))
    )
    good = ended & pl.col("stopped_reason").is_in(["EVDisconnected", "Local", "Remote"])
    return (
        msgs.with_columns(hour=pl.col("received") // HOUR)
        .group_by("charger_id", "hour")
        .agg(
            msgs=pl.len(),
            heartbeats=(a == "Heartbeat").sum(),
            auth_rfid=((a == "Authorize") & (pl.col("id_token_type") == "ISO14443")).sum(),
            auth_central=((a == "Authorize") & (pl.col("id_token_type") == "Central")).sum(),
            remote_ok=(
                (a == "RequestStartTransaction") & (d == "res") & (pl.col("status") == "Accepted")
            ).sum(),
            tx_started=((a == "TransactionEvent") & (pl.col("event_type") == "Started")).sum(),
            tx_good=good.sum(),
            evcomm_fail=evfail.sum(),
            interrupted=(ended & pl.col("stopped_reason").is_in(["PowerLoss", "ImmediateReset"])).sum(),
            watchdog=((a == "NotifyEvent") & (pl.col("variable") == "Watchdog")).sum(),
            problem=((a == "NotifyEvent") & (pl.col("variable") == "Problem")).sum(),
            boots=(a == "BootNotification").sum(),
            temp_max=pl.when((a == "NotifyEvent") & (pl.col("variable") == "Temperature"))
            .then(pl.col("value"))
            .max(),
            power_sum=pl.when(good & pl.col("power_w").is_not_null()).then(pl.col("power_w")).sum(),
            power_n=(good & pl.col("power_w").is_not_null()).sum(),
            energy_kwh=pl.when(good).then(pl.col("energy_wh") / 1000).sum(),
            faulted_events=((a == "StatusNotification") & (pl.col("status") == "Faulted")).sum(),
            available_events=((a == "StatusNotification") & (pl.col("status") == "Available")).sum(),
            resets=((a == "Reset") & (d == "cs")).sum(),
            unlocks=((a == "UnlockConnector") & (d == "cs")).sum(),
            code=pl.col("tech_code").drop_nulls().last(),
            info=pl.col("tech_info").drop_nulls().last(),
            firmware=pl.col("firmware").drop_nulls().last(),
        )
        .with_columns(pl.col(c).cast(pl.Float64) for c in SIGNALS)
    )


def status_events(msgs: pl.DataFrame) -> pl.DataFrame:
    """Connector status changes as received, for status as of any time."""
    return (
        msgs.filter(pl.col("action") == "StatusNotification")
        .select("charger_id", "evse", "status", "received")
        .sort("received")
    )


def grid(hourly_df: pl.DataFrame, chargers: list[str], h0: int, h1: int) -> pl.DataFrame:
    """A complete charger x hour grid (zeros where nothing was received), sorted for rolling windows."""
    hours = pl.DataFrame({"hour": list(range(h0, h1))}, schema={"hour": pl.Int64})
    base = pl.DataFrame({"charger_id": chargers}).join(hours, how="cross")
    out = base.join(hourly_df, on=["charger_id", "hour"], how="left")
    return out.with_columns(pl.col(c).fill_null(0.0) for c in SIGNALS if c != "temp_max").sort(
        "charger_id", "hour"
    )
