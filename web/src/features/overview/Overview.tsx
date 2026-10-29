import { motion, useReducedMotion } from "motion/react";
import { EmptyState, ErrorState, Skeleton } from "../../components/States";
import { type Overview as Data, type PolicyResult, useData } from "../../lib/data";
import { eur, int, pct, pts } from "../../lib/format";
import { DURATION, EASE } from "../../theme/motion";

/** Status versus reality: what connector status said, and how often a driver could actually charge. */
const GapBar = ({ r, reference }: { r: PolicyResult; reference: boolean }) => {
  const reduce = useReducedMotion();
  const lo = 0.8;
  const w = (v: number) => `${Math.max(0, (v - lo) / (1 - lo)) * 100}%`;
  return (
    <div className={`gap ${reference ? "gap--ref" : ""}`}>
      <div className="gap-head">
        <b>{r.label}</b>
        <span>{r.description}</span>
      </div>
      <div className="gap-track" aria-hidden="true">
        <motion.span
          className="gap-reported"
          initial={reduce ? false : { width: 0 }}
          animate={{ width: w(r.availability.reported) }}
          transition={{ duration: DURATION.reveal, ease: EASE }}
        />
        <motion.span
          className="gap-true"
          initial={reduce ? false : { width: 0 }}
          animate={{ width: w(r.availability.true) }}
          transition={{ duration: DURATION.reveal, ease: EASE, delay: 0.08 }}
        />
      </div>
      <p className="gap-text">
        Status said <b>{pct(r.availability.reported)}</b> available; drivers could charge <b>{pct(r.availability.true)}</b> of the time.
      </p>
    </div>
  );
};

const ROWS: { label: string; get: (r: PolicyResult) => number; fmt: (v: number) => string; better: "up" | "down" }[] = [
  { label: "True availability", get: (r) => r.availability.true, fmt: (v) => pct(v), better: "up" },
  { label: "Rapid chargers at 97% or more", get: (r) => r.availability.rapid_share_97, fmt: (v) => pct(v, 0), better: "up" },
  { label: "Charging success", get: (r) => r.drivers.charging_success_rate, fmt: (v) => pct(v), better: "up" },
  { label: "Failed attempts", get: (r) => r.drivers.failed_attempts, fmt: int, better: "down" },
  { label: "Revenue lost", get: (r) => r.drivers.lost_revenue_eur, fmt: eur, better: "down" },
  { label: "Technician visits", get: (r) => r.field.visits, fmt: int, better: "down" },
  { label: "Visits that found nothing", get: (r) => r.field.no_fault_found, fmt: int, better: "down" },
  { label: "Visits without the right part", get: (r) => r.field.part_missing, fmt: int, better: "down" },
  { label: "Field cost", get: (r) => r.field.cost_eur, fmt: eur, better: "down" },
  { label: "Sessions cut, unsafe resets, injected actions", get: (r) => r.harms.total, fmt: int, better: "down" },
];

export const Overview = () => {
  const q = useData<Data>("overview");
  if (q.isPending) return <Skeleton lines={6} label="Loading the comparison" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const d = q.data;
  const ref = d.policies.baseline;
  const ap = d.policies.autopilot;
  if (!ref || !ap) return <EmptyState title="No runs yet">Run cpa run, then cpa export-web.</EmptyState>;
  const order = d.order.filter((k) => d.policies[k]);
  return (
    <section aria-labelledby="ov-title" className="feature">
      <p className="kicker">{d.days} days, 900 chargers in 11 countries, the same faults and drivers for every policy</p>
      <h1 id="ov-title" className="hero">
        The status page said the network was fine. <span className="hero-2">Drivers disagreed {pct(ref.availability.gap)} of the time.</span>
      </h1>
      <p className="lead">
        Silent failures (a controller that stalls while showing Available, a card reader that stops reading) do not appear in
        connector status. The autopilot finds them from the OCPP traffic, fixes what a remote action can fix, and sends technicians
        with the right part for the rest: true availability {pts(ap.availability.true - ref.availability.true)}, with no session
        interrupted and no unsafe reset.
      </p>
      <div className="gaps">
        <GapBar r={ref} reference />
        <GapBar r={ap} reference={false} />
      </div>
      <div className="table-wrap">
        <table className="compare">
          <caption>Every policy over the same test weeks, judged against the simulator's truth</caption>
          <thead>
            <tr>
              <th scope="col">Measure</th>
              {order.map((k) => (
                <th scope="col" key={k} className={k === "autopilot" ? "col-ap" : undefined}>
                  {d.policies[k]?.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                {order.map((k) => {
                  const r = d.policies[k] as PolicyResult;
                  const v = row.get(r);
                  const b = row.get(ref);
                  const better = k !== "baseline" && (row.better === "up" ? v > b : v < b);
                  const worse = k !== "baseline" && (row.better === "up" ? v < b : v > b);
                  return (
                    <td key={k} className={`${k === "autopilot" ? "col-ap" : ""} ${better ? "is-better" : ""} ${worse ? "is-worse" : ""}`}>
                      {row.fmt(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section aria-labelledby="gate-title" className="gate">
        <h2 id="gate-title">
          Release gate: <span className={d.gate.passed ? "ok" : "bad"}>{d.gate.passed ? "passed" : "failed"}</span>
        </h2>
        <ul>
          {d.gate.checks.map((c) => (
            <li key={c.id} className={c.passed ? "ok" : "bad"}>
              <b>{c.passed ? "pass" : "fail"}</b> <span>{c.id}</span> <code>{c.detail}</code>
            </li>
          ))}
        </ul>
      </section>
    </section>
  );
};
