"""Point-in-time features per charger and hour, from hourly signals and connector status.

A feature at hour h uses only messages received up to the end of hour h. The same code computes
the training table over months of history and the autopilot's features over its last week of
signals each hour."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .signals import HOUR, grid

WINDOWS = (6, 24, 72, 168)
CATEGORICAL = ["model", "kind", "country", "firmware", "code"]


def _roll(col: str, w: int) -> pl.Expr:
    return pl.col(col).rolling_sum(window_size=w, min_samples=1).over("charger_id")


def window_features(g: pl.DataFrame) -> pl.DataFrame:
    """Rolling counts and trends over the hourly grid (sorted by charger, hour)."""
    g = g.with_columns(
        failed_starts=(pl.col("auth_rfid") + pl.col("auth_central") + pl.col("remote_ok") - pl.col("tx_started")).clip(
            lower_bound=0
        ),
        silent_hour=(pl.col("msgs") == 0).cast(pl.Float64),
    )
    exprs = []
    for w in WINDOWS:
        for c in (
            "failed_starts",
            "evcomm_fail",
            "tx_good",
            "tx_started",
            "auth_rfid",
            "auth_central",
            "watchdog",
            "problem",
            "boots",
            "interrupted",
            "resets",
            "silent_hour",
            "faulted_events",
        ):
            if w == 168 and c not in ("tx_good", "tx_started", "auth_rfid", "auth_central", "evcomm_fail"):
                continue
            exprs.append(_roll(c, w).alias(f"{c}_{w}"))
    g = g.with_columns(exprs)
    # how long since the CSMS last heard from the charger, in hours (capped at a week)
    g = g.with_columns(_last=pl.when(pl.col("msgs") > 0).then(pl.col("hour")))
    g = g.with_columns(_last=pl.col("_last").forward_fill().over("charger_id"))
    g = g.with_columns(
        hours_silent=(pl.col("hour") - pl.col("_last")).fill_null(168).clip(upper_bound=168).cast(pl.Float64)
    )
    # temperature: the last day against the two days before
    g = g.with_columns(_t=pl.col("temp_max").forward_fill(limit=8).over("charger_id"))
    g = g.with_columns(
        temp_24=pl.col("_t").rolling_mean(window_size=24, min_samples=1).over("charger_id"),
        temp_prev=pl.col("_t").shift(24).rolling_mean(window_size=48, min_samples=1).over("charger_id"),
        power_72=(_roll("power_sum", 72) / _roll("power_n", 72).clip(lower_bound=1)),
        code=pl.col("code").forward_fill(limit=72).over("charger_id"),
        info=pl.col("info").forward_fill(limit=72).over("charger_id"),
        firmware_seen=pl.col("firmware").forward_fill().over("charger_id"),
    )
    g = g.with_columns(temp_trend=(pl.col("temp_24") - pl.col("temp_prev")).fill_null(0.0))
    return g.drop("_last", "_t")


def status_features(g: pl.DataFrame, status: pl.DataFrame) -> pl.DataFrame:
    """Faulted and Unavailable connectors as of the end of each hour."""
    s = status.with_columns(hour=pl.col("received") // HOUR).sort("hour")
    out = g
    for e in (1, 2):
        se = (
            s.filter(pl.col("evse") == e)
            .group_by("charger_id", "hour")
            .agg(pl.col("status").last())
            .sort("hour")
            .rename({"status": f"st{e}"})
        )
        out = out.sort("hour").join_asof(se, on="hour", by="charger_id", strategy="backward", check_sortedness=False)
    out = out.with_columns(
        faulted_now=((pl.col("st1") == "Faulted").cast(pl.Float64) + (pl.col("st2") == "Faulted").cast(pl.Float64)),
        unavailable_now=(
            (pl.col("st1") == "Unavailable").cast(pl.Float64) + (pl.col("st2") == "Unavailable").cast(pl.Float64)
        ),
    ).with_columns(pl.col("faulted_now", "unavailable_now").fill_null(0.0))
    return out.sort("charger_id", "hour")


def site_features(g: pl.DataFrame, chargers: pl.DataFrame) -> pl.DataFrame:
    g = g.join(
        chargers.select(
            "charger_id",
            "site_id",
            "model",
            "kind",
            "country",
            "kw",
            pl.col("firmware").alias("firmware_ref"),
        ),
        on="charger_id",
        how="left",
    )
    site = g.group_by("site_id", "hour").agg(site_silent=(pl.col("hours_silent") >= 2).mean(), site_n=pl.len())
    g = g.join(site, on=["site_id", "hour"], how="left")
    return g.with_columns(
        firmware=pl.coalesce("firmware_seen", "firmware_ref"),
        power_ratio=pl.col("power_72") / (pl.col("kw") * 1000),
    ).drop("firmware_ref")


def build(hourly_df: pl.DataFrame, status: pl.DataFrame, chargers: pl.DataFrame, h0: int, h1: int) -> pl.DataFrame:
    g = grid(hourly_df, chargers["charger_id"].to_list(), h0, h1)
    g = window_features(g)
    g = status_features(g, status)
    g = site_features(g, chargers)
    return g.with_columns(local_hour=(pl.col("hour") % 24).cast(pl.Float64))


def build_chunked(
    hourly_df: pl.DataFrame,
    status: pl.DataFrame,
    chargers: pl.DataFrame,
    h0: int,
    h1: int,
    finish: Callable[[pl.DataFrame], pl.DataFrame],
    chargers_per_chunk: int = 110,
) -> pl.DataFrame:
    """The same features, a group of whole sites at a time (windows are per charger, site features per
    site), passing each chunk through `finish` to keep only the rows and columns needed."""
    sites = chargers.group_by("site_id", maintain_order=True).len()
    chunks, cur, n = [], [], 0
    for sid, k in sites.iter_rows():
        cur.append(sid)
        n += k
        if n >= chargers_per_chunk:
            chunks.append(cur)
            cur, n = [], 0
    if cur:
        chunks.append(cur)
    out = []
    for ids in chunks:
        ch = chargers.filter(pl.col("site_id").is_in(ids))
        cids = ch["charger_id"]
        f = build(
            hourly_df.filter(pl.col("charger_id").is_in(cids)),
            status.filter(pl.col("charger_id").is_in(cids)),
            ch,
            h0,
            h1,
        )
        out.append(finish(f))
    return pl.concat(out, how="diagonal_relaxed")


NUMERIC = [
    f"{c}_{w}"
    for w in WINDOWS
    for c in (
        "failed_starts",
        "evcomm_fail",
        "tx_good",
        "tx_started",
        "auth_rfid",
        "auth_central",
        "watchdog",
        "problem",
        "boots",
        "interrupted",
        "resets",
        "silent_hour",
        "faulted_events",
    )
    if w != 168 or c in ("tx_good", "tx_started", "auth_rfid", "auth_central", "evcomm_fail")
] + [
    "hours_silent",
    "temp_24",
    "temp_trend",
    "power_ratio",
    "faulted_now",
    "unavailable_now",
    "site_silent",
    "kw",
    "local_hour",
]
FEATURES = NUMERIC + CATEGORICAL


def symptoms(f: pl.DataFrame, d: dict) -> pl.DataFrame:
    """Which symptoms each charger-hour shows (config/ops.yaml detection). Same rule offline and online."""
    return f.with_columns(
        sym_faulted=pl.col("faulted_now") > 0,
        sym_offline=pl.col("hours_silent") >= 2,
        sym_starts=pl.col("failed_starts_24") >= d["silent_start_failures"],
        sym_evcomm=pl.col("evcomm_fail_72") >= d["evcomm_failures"],
        sym_rfid=(pl.col("auth_rfid_168") >= 6) & (pl.col("auth_rfid_72") == 0) & (pl.col("tx_good_72") >= 2),
        sym_central=(pl.col("auth_central_168") >= 5) & (pl.col("auth_central_72") == 0) & (pl.col("tx_good_72") >= 2),
    ).with_columns(
        symptom=pl.any_horizontal("sym_faulted", "sym_offline", "sym_starts", "sym_evcomm", "sym_rfid", "sym_central")
    )


def online(
    signal_window: pl.DataFrame, status_now: pl.DataFrame, chargers: pl.DataFrame, hour: int, span: int = 170
) -> pl.DataFrame:
    """Features for every charger at `hour` from the last `span` hours of signals and the CSMS's current
    connector status: the same computation as `build`, restricted to one hour (tests compare the two)."""
    window = signal_window.filter(pl.col("hour") > hour - span)
    first = int(window["hour"].min()) if window.height else hour  # no invented silence before the first data
    g = grid(window, chargers["charger_id"].to_list(), max(hour - span + 1, first), hour + 1)
    g = window_features(g).filter(pl.col("hour") == hour)
    g = status_features(g, status_now)
    g = site_features(g, chargers)
    return g.with_columns(local_hour=(pl.col("hour") % 24).cast(pl.Float64))
