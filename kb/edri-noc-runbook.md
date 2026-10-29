# EDRI network operations runbook (excerpt)

## Principles
- A running charging session is never interrupted remotely. Use Reset OnIdle, or wait.
- Insulation, isolation, ground-fault or IMD codes are safety stops: no remote reset; set the charger Inoperative and dispatch.
- Try the EVSE reset before a station reset. At most one station reset per charger per day.
- A whole site offline at once is usually a grid or network outage: check the site, do not dispatch for each charger.

## Silent failures
- Charger shows Available but authorisations or remote starts are not followed by a transaction: controller stall. Reset the EVSE, then the station.
- Cards stop working while app starts continue: RFID reader. Station reset, then RFID reader replacement.
- Card payments stop on DC chargers while cards and app work: payment terminal.
- Sessions on one connector stop within a minute with EV communication errors: cable wear on that connector.
- EV communication errors across many chargers of one model right after a firmware rollout: firmware regression. Halt the rollout and roll back in stages; this needs a person's approval.

## Dispatch
- Bring the part the diagnosis expects; a visit without the part costs a second visit.
- Write the technician's brief in the local language.
