import { useQuery } from "@tanstack/react-query";

/** The files `cpa export-web` writes to public/data. Shapes are checked where they are read. */
export interface PolicyResult {
  policy: string;
  label: string;
  description: string;
  days: number;
  availability: { true: number; reported: number; gap: number; rapid_share_97: number; rapid_share_99: number };
  drivers: { charging_success_rate: number; failed_attempts: number; turned_away_unavailable: number; lost_revenue_eur: number };
  field: { visits: number; cost_eur: number; no_fault_found: number; part_missing: number };
  harms: { interrupted_sessions: number; unsafe_resets: number; injected_actions: number; total: number };
  firmware: { buggy_installs: number; bug_failures: number };
  incidents?: {
    opened: number;
    false: number;
    diagnosis_top1_matches_truth: number | null;
    silent_detected: number;
    silent_detection_median_hours: number | null;
    preventive_visits: number;
    preventive_with_failing_module: number;
  };
  llm?: { calls: number; cost_eur_per_day: number; quarantined_fields: number; masked_values: number };
}

export interface GateCheck {
  id: string;
  passed: boolean;
  detail: string;
}

export interface Overview {
  generated: string;
  days: number;
  order: string[];
  policies: Record<string, PolicyResult>;
  gate: { passed: boolean; checks: GateCheck[] };
}

/** One pass through the decision line: what the model thought, what the reading added, what the planner chose,
 * what the shield said, who decided, what went over OCPP, and whether it worked. */
export interface Cycle {
  ts: number;
  diagnosis: string | null;
  p: number | null;
  reading: string | null;
  safety: boolean;
  quarantined: number;
  plan: string;
  action: string | null;
  shield: { allowed: boolean; rules: string[] } | null;
  route: "act" | "propose" | null;
  command: { result: string; decided_by: string } | null;
  outcome: "worked" | "did not work" | null;
}

export interface Visit {
  ts: number;
  outcome: string;
  part_used: string | null;
  part_needed: string | null;
}

export interface Incident {
  id: string;
  charger_id: string;
  model: string;
  country: string;
  opened: number;
  closed: number | null;
  resolution: string;
  symptoms: string[];
  preventive: boolean;
  top: { cls: string; p: number } | null;
  reading: { first_action?: string; safety_concern?: boolean; confidence?: number; quarantined?: number } | null;
  cycles: Cycle[];
  visits: Visit[];
  tags: string[];
  truth?: string | null;
}

export interface Rung {
  action: string;
  model: string;
  level: "act" | "propose";
  outcomes: number;
  successes: number;
  wilson_lower: number;
  wilson_upper: number;
  changes: { ts: number; to: string }[];
}

export interface Autonomy {
  ladder: Rung[];
  thresholds: { min_outcomes: number; wilson_lower: number };
  never_autonomous: string[];
}

export interface ModelEvent {
  ts: number;
  event: string;
  [k: string]: unknown;
}

export interface Models {
  bootstrap: Record<string, { version: string; metrics: Record<string, unknown>; checks: Record<string, number>; card: Record<string, unknown> }>;
  events: ModelEvent[];
  drift: { ts: number; rows: number; top: { feature: string; psi: number }[]; alert: boolean }[];
  champion: Record<string, string | null>;
  psi_alert: number;
}

export interface UseCase {
  name: string;
  owner: string;
  purpose: string;
  route: string[];
  daily_budget_eur: number;
  tokens_per_minute: number;
  calls: number;
  cache_hits: number;
  cost_eur_per_day: number;
  tokens_in: number;
  tokens_out: number;
  quarantined: number;
  masked: number;
  refused: number;
}

export interface Gateway {
  provider: string;
  use_cases: UseCase[];
  note: string;
}

const FILES = { overview: "overview", incidents: "incidents", autonomy: "autonomy", models: "models", gateway: "gateway" } as const;
export type DataName = keyof typeof FILES;

export async function fetchData<T>(name: DataName): Promise<T> {
  const res = await fetch(`./data/${FILES[name]}.json`);
  if (!res.ok) throw new Error(`Could not load ${name} (HTTP ${res.status}). Run cpa export-web and rebuild.`);
  return (await res.json()) as T;
}

export const useData = <T,>(name: DataName) =>
  useQuery<T, Error>({ queryKey: [name], queryFn: () => fetchData<T>(name), staleTime: Infinity, retry: 1 });
