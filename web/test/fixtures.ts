import type { Cycle, Incident } from "../src/lib/data";

export const cycle = (over: Partial<Cycle> = {}): Cycle => ({
  ts: 1_781_000_000,
  diagnosis: "reset",
  p: 0.82,
  reading: null,
  safety: false,
  quarantined: 0,
  plan: "evse_reset (expected success 0.71)",
  action: "evse_reset",
  shield: { allowed: true, rules: [] },
  route: "act",
  command: { result: "Accepted", decided_by: "autonomous" },
  outcome: "worked",
  ...over,
});

export const incident = (over: Partial<Incident> = {}): Incident => ({
  id: "AP-00001",
  charger_id: "DE-0042",
  model: "DC-150",
  country: "DE",
  opened: 1_781_000_000,
  closed: 1_781_090_000,
  resolution: "evse_reset",
  symptoms: ["starts"],
  preventive: false,
  top: { cls: "reset", p: 0.82 },
  reading: null,
  cycles: [cycle()],
  visits: [],
  tags: ["autonomous", "silent"],
  truth: "controller_hang",
  ...over,
});
