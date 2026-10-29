# 1. A simulated fleet that speaks OCPP, with the truth kept apart

Status: accepted

## Context
An AI that acts on chargers has to be judged on what happened to drivers, not on what its own logs
say. Real fleet data has no ground truth: the cause recorded on a ticket is wrong some of the time,
status pages show a charger as Available while it fails every start, and nobody knows what a
different decision would have done.

## Decision
The fleet is simulated: 900 chargers in 11 countries, four fictional hardware models, nine kinds of
fault, grid outages, firmware rollouts, drivers, hotline complaints and technicians. Chargers emit
OCPP 2.0.1 messages (validated against the official JSON schemas); the operations code sees only
those messages, complaints, visit reports and logs. The truth (fault episodes, true usability of
every connector-hour, harms) is written separately and read only by the evaluation.

Every policy runs the same test weeks with common random numbers: the same faults at the same
minutes, the same drivers, the same uniform number deciding whether the k-th remote action on an
episode works. Differences between policies are what they did.

## Consequences
Results show that the methods work on a fleet built to resemble a real one, not what EDRI's fleet
would show. Blind held-out fleets, written by someone who never read the operations code, test
whether the autopilot was tuned to the development fleet.
