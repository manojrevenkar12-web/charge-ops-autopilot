"""The CSMS's view of the fleet, from OCPP rows."""

import polars as pl

from cpa import ocpp_flat
from cpa.ops.state import FleetView

REF = {"chargers": pl.DataFrame([{"charger_id": "C1", "site_id": "S1", "country": "DE", "model": "DC-150",
                                  "vendor": "N", "kind": "DC", "kw": 150, "connectors": 2, "firmware": "1"}]),
       "sites": pl.DataFrame([{"site_id": "S1", "country": "DE", "lat": 52.5, "lon": 13.4, "utc_offset": 1,
                               "motorway": False}])}


def rows(*items):
    out = {c: [] for c in ocpp_flat.COLUMNS}
    for it in items:
        base = {c: None for c in ocpp_flat.COLUMNS} | {"charger_id": "C1", "direction": "cp", "evse": 1} | it
        base["received"] = base["ts"]
        for c in ocpp_flat.COLUMNS:
            out[c].append(base[c])
    return ocpp_flat.frame(out)


def test_a_start_after_failed_ones_is_not_a_failure():
    v = FleetView(REF)
    # two authorisations that never start (a stalled controller), a reset, then one that does start
    v.update(rows({"ts": 1000, "action": "Authorize", "id_token_type": "ISO14443"},
                  {"ts": 1500, "action": "Authorize", "id_token_type": "ISO14443"}), now=1600)
    v.update(rows({"ts": 5000, "action": "Authorize", "id_token_type": "ISO14443"},
                  {"ts": 5020, "action": "TransactionEvent", "event_type": "Started", "tx_id": "t1"}), now=5100)
    v.update(rows(), now=9000)
    assert sorted(v.chargers["C1"].failed_starts) == [1000, 1500]


def test_open_transaction_blocks_only_its_evse():
    v = FleetView(REF)
    v.update(rows({"ts": 10, "action": "TransactionEvent", "event_type": "Started", "tx_id": "t", "evse": 2}), now=20)
    assert v.busy("C1", 2) and v.busy("C1", 0) and not v.busy("C1", 1)
    v.update(rows({"ts": 30, "action": "TransactionEvent", "event_type": "Ended", "tx_id": "t", "evse": 2,
                   "stopped_reason": "EVDisconnected", "energy_wh": 9000.0}), now=40)
    assert not v.busy("C1", 0)
