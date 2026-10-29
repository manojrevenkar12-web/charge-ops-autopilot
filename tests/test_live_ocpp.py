"""The autopilot running a live CSMS over real OCPP 2.0.1 WebSockets (payloads schema-validated by the
ocpp library in both directions), on an accelerated clock."""

import asyncio

from cpa.csms.live import LiveLoop, reference
from cpa.csms.server import CSMS, Clock
from cpa.csms.station import connect
from cpa.ops.autopilot import Autopilot

T0 = 1_780_000_000 - 1_780_000_000 % 3600 + 9 * 3600  # 09:00 UTC, NOC hours


async def scenario(store=None):
    clock = Clock(T0, speed=20_000)  # one simulated hour in 0.18 s
    csms = CSMS(clock, store)
    port = await csms.serve()
    url = f"ws://127.0.0.1:{port}"
    ids = [{"id": "DE-HANG", "site": "S1"}, {"id": "DE-ISO", "site": "S2"}, {"id": "DE-BUSY", "site": "S3"}]
    pilot = Autopilot("live", retrain=False)
    pilot.start(reference(ids), T0)
    for model in ("DC-150",):  # as if these rungs had been earned already
        for a in ("evse_reset", "station_reset", "set_inoperative", "unlock"):
            pilot.ladder.rungs[(a, model)].level = "act"
    loop = LiveLoop(csms, pilot, clock)
    st = {}
    for s in ids:
        st[s["id"]], _, _ = await connect(url, s["id"], clock)
    await st["DE-HANG"].fault("controller_hang")
    await st["DE-ISO"].fault("isolation_fault")
    await st["DE-BUSY"].drive(1, leave_running=True)  # a session that runs throughout
    for h in range(10):
        await clock.sleep_sim(900)  # a quarter of an hour passes (stations finish rebooting)
        for s in st.values():
            await s.heartbeat()
        if h < 6:
            await st["DE-HANG"].drive(2, minutes=1)
        await loop.hour(h)
    await clock.sleep_sim(900)
    ok_after = await st["DE-HANG"].drive(2, minutes=1)
    await csms.close()
    return st, loop, pilot, ok_after, csms


def test_autopilot_over_real_ocpp():
    st, loop, pilot, ok_after, csms = asyncio.run(scenario())
    hang, iso, busy = st["DE-HANG"], st["DE-ISO"], st["DE-BUSY"]
    # the silent controller stall was found from failed starts and fixed with a real Reset
    assert sum(r.startswith("Reset") for r in hang.received) == 1, (hang.received, loop.sent)
    inc = next(i for i in [*pilot.closed, *pilot.incidents.values()] if i.charger_id == "DE-HANG")
    assert inc.resolution.endswith("reset") and inc.state in ("watch", "resolved")
    assert ok_after, "after the reset a driver can charge again"
    # the insulation fault was never reset; the station was taken out of service instead
    assert not any(r.startswith("Reset") for r in iso.received), iso.received
    assert any("Inoperative" in r for r in iso.received), (iso.received, pilot.shield.log)
    # nobody's session was cut and nothing unsafe happened anywhere
    assert hang.harms == iso.harms == busy.harms == []
    # every command went through the shield and was recorded with its verdict and the station's answer
    assert csms.store.commands and all("shield" in c and c["result"] for c in csms.store.commands)


def test_csms_records_wire_payloads():
    st, loop, pilot, ok_after, csms = asyncio.run(scenario())
    actions = {r["action"] for r in csms.store.rows}
    assert {"BootNotification", "Heartbeat", "StatusNotification", "Authorize", "TransactionEvent",
            "NotifyEvent"} <= actions
    assert any(p.get("eventData") for p in csms.store.payloads)  # payloads kept exactly as received
