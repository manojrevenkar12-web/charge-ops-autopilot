# Held-out set fleets-2026-10

Eight fleet files, each a complete variant of `config/fleet.yaml`, written blind from
`heldout/WRITING_SCENARIOS.md`, `config/fleet.yaml` and `src/cpa/sim/` only. No other file in the
repository was opened, listed or searched. That includes `AGENTS.md`, which `CLAUDE.md` points to;
it was left unread on purpose to keep the set blind. Nothing was run except a YAML parse check on
these files.

All fleets keep the hardware model names, fault kinds and vendor tech codes; they use only countries
and languages the simulator knows. Every fleet has `test_days: 28`, `history_days` between 70 and 98,
`chargers` at most 900, a seed other than 2026 and a Monday start date.

| Fleet | What it changes | Why it is a fair test |
|---|---|---|
| `summer-heatwave.yaml` | Hot summer: 2-3x power-module hazard, shorter onset (1-3 days), weaker temperature rise (11 C), power modules scarce (9-day lead, 12% van stock); summer-time UTC offsets; holiday traffic; hidden bug on DC-60 1.10.0 instead of HPC-350 | Heat waves really do drive power-stage failures, DST really does shift every local clock, and the bug sits on a model that never had one. |
| `dc150-fast-regression.yaml` | Bug on the most common model (DC-150 2.12.1): rolled out fast (30%/day), affects 70% of car makes at an 18% per-session rate; a clean AC-22 release lands the same week; DC-150 cable wear is common | Vendors do push releases quickly. The same EV-communication symptom has three possible sources (bug, cables, the release that happened to land at the same time), so the right release has to be blamed. |
| `ac22-slow-bug-urban.yaml` | Dense urban, AC-heavy fleet (50% AC-22, city roads at 35 km/h); bug on AC-22 5.1.0, staged at 2.5%/day, starting on day 80 of the 98-day history | A slow staged rollout on a low-traffic model is how real regressions hide. The bug is already in the training history, and each charger has only a few sessions a day to show it. |
| `thin-field-service-winter.yaml` | About half the technicians, 55 km/h winter roads, 8-16 working day, longer visits, 4-12-day part lead times, low van stock; more frozen locks and cracked cables; no firmware bug | When field capacity is short, prioritising visits and fixing remotely matter. With no bug at all, halting a clean release is a false alarm. |
| `injection-heavy-vendor-text.yaml` | `injection_share: 0.4` (20x the development fleet); more insulation faults, locks, terminal events and a chattier watchdog, so instruction-like techInfo often sits next to faults that must not be reset; HPC-350 bug with other parameters (25% of makes, 50% rate) | A compromised or careless vendor integration is a realistic threat. Here acting on vendor text would cause real harm (resetting an insulation fault, forcing chargers Operative). |
| `cable-theft-wave.yaml` | Cable theft and vandalism: insulation-fault hazard 3-7x, more cable wear and jammed locks with lower unlock success, cables scarce (7-day lead, 20% van stock), GB/NL/DE-heavy; no firmware bug | Copper cable theft at charging sites really happens. It floods the fleet with faults that are unsafe to reset and need parts, which tests the shield and parts planning. |
| `rfid-batch-app-shift.yaml` | Bad RFID-reader batch on DC-60 and AC-22 (hazard about 1.0, resets rarely help); drivers shifted to app and contactless (RFID 22%), fewer complaints, more retries; commuter double-peak profile with quiet weekends; four releases in one week, only DC-60 1.10.0 buggy | Bad supplier batches happen, and a silent fault with a fading signal (fewer card users) is hard to see. Concurrent releases test attribution. |
| `small-storm-season.yaml` | Small rural northern fleet (340 chargers, 5 countries, 9 technicians): grid outages about 6x more often and up to 36 h long; comm loss about 2.4x, lasting longer, with fewer chargers still charging offline; no firmware bug | Storm season on weak rural LTE is common. Most "dead" chargers are not charger faults, so wasted visits and false diagnoses are the risk, at a different fleet size. |

## Notes for whoever runs the set

- The simulator can only express a firmware bug as EV-communication failures: the engine reads
  `bug.ev_comm_fails` and `bug.affected_ev_share` and nothing else. So the bug cannot have a
  different symptom (such as failed starts) without changing the simulator, which this set must not
  do. The bugs differ instead by model (DC-60, DC-150, AC-22, HPC-350), per-session rate, share of
  car makes affected, rollout speed (2.5% to 30% per day) and timing (inside the history, at the
  test start, mid-warm-up), and by what else rolls out at the same time.
- Three fleets (`thin-field-service-winter`, `cable-theft-wave`, `small-storm-season`) have no buggy
  release at all.
- DC-60 has a second firmware version, `1.10.0`, in `summer-heatwave` and `rfid-batch-app-shift`.
  The development fleet lists only `1.9.7` for DC-60.
- `summer-heatwave` uses summer-time UTC offsets (one hour later than the development fleet's winter
  offsets).
