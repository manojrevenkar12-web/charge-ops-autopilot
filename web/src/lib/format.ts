export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const pts = (v: number, digits = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)} pts`;
export const eur = (v: number) =>
  new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: v < 10 ? 2 : 0 }).format(v);
export const int = (v: number) => new Intl.NumberFormat("en-GB").format(Math.round(v));
export const when = (ts: number) =>
  new Date(ts * 1000).toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";
export const hours = (a: number, b: number) => `${((b - a) / 3600).toFixed(1)} h`;

const ACTIONS: Record<string, string> = {
  evse_reset: "EVSE reset",
  station_reset: "station reset",
  unlock: "unlock connector",
  set_inoperative: "take out of service",
  set_operative: "back in service",
  get_log: "fetch diagnostic log",
  firmware: "firmware rollback",
  halt_rollout: "halt rollout",
};
export const action = (a: string) => ACTIONS[a] ?? a.replaceAll("_", " ");

const CLASSES: Record<string, string> = {
  reset: "a reset fixes it",
  unlock: "unlocking fixes it",
  onsite_restart: "a technician's restart",
  self_resolved: "clears on its own",
  firmware: "firmware",
  other: "unclear",
};
export const remedy = (c: string) => CLASSES[c] ?? (c.startsWith("part:") ? `needs a ${c.slice(5).replaceAll("_", " ")}` : c);
