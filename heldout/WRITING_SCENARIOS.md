# Writing held-out fleet scenarios

A held-out set is a folder `heldout/<set>/fleets/` of fleet files. Each file is a complete variant of
`config/fleet.yaml`: the simulated world's true parameters. The project's autopilot (diagnosis models,
planner, shield, LLM gateway) is run once on each file with no change to its code, and its first
results are recorded and never rewritten. The point is to find out whether the autopilot works on
fleets nobody tuned it for.

## What you may read

- this guide
- `config/fleet.yaml` (the format, and the development fleet's values)
- `src/cpa/sim/` (the simulator: what each parameter does)

Do not read anything else in the repository: not `src/cpa/ops`, `src/cpa/ml`, `src/cpa/mlops`,
`src/cpa/gateway`, `config/ops.yaml`, `config/models.yaml`, the reports or the tests. If you have seen
them, say so in the set's README; the set is then not blind.

## What to vary

Write 6 to 8 fleets that differ from the development fleet in ways an operator could meet in reality,
each for a reason you can state in one sentence. For example:
- a different mix and rate of faults (a hot summer for power modules; a vandalism wave of insulation
  faults; an RFID reader batch from a bad supplier)
- a firmware release with a different hidden bug (a different model, a different symptom such as
  failed starts instead of EV communication errors, a slower or faster rollout)
- fewer technicians, longer part lead times, or slower roads
- different drivers (more contactless, more app starts, a different daily profile)
- more instruction-like text in vendor messages (`untrusted_text.injection_share`)
- many grid outages, or many communication losses
- a smaller or larger fleet

Keep each fleet runnable in reasonable time: `fleet.chargers` at most 900, `history_days` between 70
and 98, `test_days` 28. Keep `seed` different from 2026 in every file. Keep the same hardware model
names (HPC-350, DC-150, DC-60, AC-22) and fault kinds (the simulator only knows these), and the same
vendor tech codes (the vendors' manuals are fixed); you may change everything else, including
remedy probabilities and which faults a model gets.

## What to hand in

- `heldout/<set>/fleets/<name>.yaml` for each fleet
- `heldout/<set>/README.md`: one line per fleet saying what it changes and why it is a fair test
- `heldout/<set>/STATUS` containing `blind: written from heldout/WRITING_SCENARIOS.md, config/fleet.yaml and src/cpa/sim only`

Do not run anything. Do not change any file outside `heldout/<set>/`.
