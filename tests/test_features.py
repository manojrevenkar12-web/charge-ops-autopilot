"""Training/serving skew: the features the autopilot computes each hour from its last week of signals
equal the features the training pipeline computes from the full message log, hour for hour."""

import numpy as np
import polars as pl
import pytest

from cpa.ml import features, signals
from cpa.ops.autopilot import Autopilot
from cpa.sim import world as W
from cpa.sim.engine import Engine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run():
    w = W.build()
    pilot = Autopilot("parity", reader=False, retrain=False)
    pilot.keep_features = True
    start = w.t0 + 20 * 86400
    e = Engine(w, pilot, start, start + 2 * 86400, "parity")
    e.run()
    msgs = pl.concat(e.frames)
    return w, pilot, msgs


def test_online_features_equal_offline(run):
    w, pilot, msgs = run
    online = pl.concat(pilot.feature_log)
    ch = w.chargers.select(
        "charger_id", "site_id", "country", "model", "vendor", "kind", "kw", "connectors", "firmware"
    )
    h0 = int(msgs["received"].min() // 3600)
    h1 = int(msgs["received"].max() // 3600) + 1
    offline = features.build(signals.hourly(msgs), signals.status_events(msgs), ch, h0, h1)
    hours = sorted(online["hour"].unique().to_list())[-12:]  # once a full window exists for the rolling sums
    on = online.filter(pl.col("hour").is_in(hours)).sort("charger_id", "hour")
    off = offline.filter(pl.col("hour").is_in(hours)).select(on.columns).sort("charger_id", "hour")
    assert on.height == off.height > 0
    for c in features.NUMERIC:
        a, b = on[c].cast(pl.Float64).to_numpy(), off[c].cast(pl.Float64).to_numpy()
        ok = np.isclose(a, b, equal_nan=True, atol=1e-6)
        assert ok.all(), (c, int((~ok).sum()), a[~ok][:3], b[~ok][:3])
    for c in ("model", "kind", "country", "code"):
        assert on[c].to_list() == off[c].to_list(), c
