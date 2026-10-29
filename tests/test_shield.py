"""The shield's rules, against the CSMS's view."""

import polars as pl

from cpa.ops.interface import Command
from cpa.ops.shield import Shield
from cpa.ops.state import FleetView

REF = {
    "chargers": pl.DataFrame(
        [
            {
                "charger_id": c,
                "site_id": s,
                "country": "DE",
                "model": "DC-150",
                "vendor": "N",
                "kind": "DC",
                "kw": 150,
                "connectors": 2,
                "firmware": "1",
            }
            for c, s in (("A", "S1"), ("B", "S1"), ("C", "S2"))
        ]
    ),
    "sites": pl.DataFrame(
        [
            {"site_id": s, "country": "DE", "lat": 52.5, "lon": 13.4, "utc_offset": 1, "motorway": False}
            for s in ("S1", "S2")
        ]
    ),
}


def shield():
    v = FleetView(REF)
    return Shield(v), v


def test_insulation_code_blocks_any_reset():
    s, v = shield()
    v.chargers["A"].codes.append((0, "0x1F01", "isolation monitor fault", 0))
    for a in ("evse_reset", "station_reset"):
        assert "no-reset-on-insulation-fault" in s.check(Command("A", a, 1, {"station_level": True}), 100).blocked_by


def test_insulation_wording_blocks_even_without_a_known_code():
    s, v = shield()
    v.chargers["A"].codes.append((0, "X-NEW-99", "Riso below threshold after rain", 0))
    assert not s.check(Command("A", "evse_reset", 1), 100).allowed


def test_immediate_reset_waits_for_running_session_onidle_does_not():
    s, v = shield()
    v.chargers["A"].open_tx["t"] = (1, 50)
    assert (
        "no-reset-during-transaction" in s.check(Command("A", "evse_reset", 1, {"type": "Immediate"}), 100).blocked_by
    )
    assert s.check(Command("A", "evse_reset", 2, {"type": "Immediate"}), 100).allowed  # the other EVSE is free
    assert s.check(Command("A", "evse_reset", 1, {"type": "OnIdle"}), 100).allowed


def test_cooldown_and_daily_cap():
    s, v = shield()
    c = Command("C", "evse_reset", 1, {"type": "OnIdle"})
    assert s.check(c, 0).allowed
    s.record(c, 0)
    assert "cooldown" in s.check(c, 3600).blocked_by
    assert s.check(c, 5 * 3600).allowed


def test_station_reset_needs_an_evse_reset_first_unless_station_level():
    s, v = shield()
    assert "escalate-reset" in s.check(Command("C", "station_reset", 0, {"type": "OnIdle"}), 0).blocked_by
    assert s.check(Command("C", "station_reset", 0, {"type": "OnIdle", "station_level": True}), 0).allowed


def test_one_reset_per_site_at_a_time():
    s, v = shield()
    a = Command("A", "evse_reset", 1, {"type": "OnIdle"})
    s.record(a, 0)
    assert "site-blast-radius" in s.check(Command("B", "evse_reset", 1, {"type": "OnIdle"}), 60).blocked_by
    assert s.check(Command("C", "evse_reset", 1, {"type": "OnIdle"}), 60).allowed


def test_actions_outside_the_allowlist_do_not_exist():
    s, v = shield()
    assert s.check(Command("A", "set_all_operative"), 0).blocked_by == ["action-allowlist"]
    assert s.check(Command("NOPE", "evse_reset"), 0).blocked_by == ["action-allowlist"]


def test_firmware_only_through_an_approved_stage():
    s, v = shield()
    assert "firmware-staged" in s.check(Command("C", "firmware", 0, {"version": "1"}), 0).blocked_by
    assert s.check(Command("C", "firmware", 0, {"version": "1", "stage_approved": True}), 0).allowed


def test_disabled_shield_allows_everything():
    v = FleetView(REF)
    assert Shield(v, enabled=False).check(Command("A", "set_all_operative"), 0).allowed


def test_a_code_from_before_the_last_boot_does_not_block():
    s, v = shield()
    v.chargers["A"].codes.append((0, "0x1F01", "isolation monitor fault", 0))  # repaired since
    v.chargers["A"].last_boot = 500  # the technician's power-up after the repair
    assert s.check(Command("A", "evse_reset", 1, {"type": "OnIdle"}), 1000).allowed
    v.chargers["A"].codes.append((600, "0x1F01", "isolation monitor fault", 0))  # re-reported after boot
    assert not s.check(Command("A", "evse_reset", 1, {"type": "OnIdle"}), 1000).allowed
