"""The CSMS's PostgreSQL record: rows come back as flat OCPP rows, and the record is append-only."""

import asyncio
import os

import pytest

from cpa import ocpp_flat
from cpa.csms.store import PostgresStore

URL = os.environ.get("CPA_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not URL, reason="CPA_TEST_DATABASE_URL not set")]


async def scenario():
    s = await PostgresStore(URL).open()
    await s.conn.execute("alter table ocpp_message disable trigger ocpp_message_append_only")
    await s.conn.execute("truncate ocpp_message, command")
    await s.conn.execute("alter table ocpp_message enable trigger ocpp_message_append_only")
    rows = ocpp_flat.from_payload(
        "StatusNotification",
        {"timestamp": "2026-06-01T10:00:00Z", "connectorStatus": "Faulted", "evseId": 1, "connectorId": 1},
        charger_id="DE-1",
        received=1_780_308_000,
    )
    await s.append(rows, {"connectorStatus": "Faulted"})
    got = await s.between(1_780_307_000, 1_780_309_000)
    blocked = None
    try:
        await s.conn.execute("update ocpp_message set action = 'Heartbeat'")
    except Exception as e:  # the trigger refuses
        blocked = str(e)
    await s.close()
    return got, blocked


def test_record_round_trip_and_append_only():
    got, blocked = asyncio.run(scenario())
    assert got.height == 1 and got["status"][0] == "Faulted" and got["evse"][0] == 1
    assert blocked and "append-only" in blocked
