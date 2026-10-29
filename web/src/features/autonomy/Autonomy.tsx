import { EmptyState, ErrorState, Skeleton } from "../../components/States";
import { type Autonomy as Data, type Rung, useData } from "../../lib/data";
import { action, pct, when } from "../../lib/format";

const Bound = ({ r, threshold }: { r: Rung; threshold: number }) => (
  <svg viewBox="0 0 100 14" className="bound" role="img" aria-label={`success between ${pct(r.wilson_lower, 0)} and ${pct(r.wilson_upper, 0)}`}>
    <line x1={0} x2={100} y1={7} y2={7} className="bound-axis" />
    <line x1={r.wilson_lower * 100} x2={r.wilson_upper * 100} y1={7} y2={7} className={`bound-range bound-range--${r.level}`} />
    {r.outcomes > 0 && <circle cx={(r.successes / r.outcomes) * 100} cy={7} r={2.6} className={`bound-dot bound-dot--${r.level}`} />}
    <line x1={threshold * 100} x2={threshold * 100} y1={1} y2={13} className="bound-threshold" />
  </svg>
);

export const Autonomy = () => {
  const q = useData<Data>("autonomy");
  if (q.isPending) return <Skeleton lines={6} label="Loading the autonomy ladder" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const d = q.data;
  const rungs = d.ladder.filter((r) => r.outcomes > 0 || r.level === "act");
  if (!rungs.length) return <EmptyState title="No outcomes yet">The autopilot has not seen any of its actions resolve.</EmptyState>;
  const models = [...new Set(rungs.map((r) => r.model))].sort();
  const actions = [...new Set(rungs.map((r) => r.action))].sort();
  return (
    <section aria-labelledby="au-title" className="feature">
      <h1 id="au-title" className="title">Autonomy is earned per action and hardware model</h1>
      <p className="lead">
        Every action starts as a proposal an operator approves. Once at least {d.thresholds.min_outcomes} outcomes the autopilot
        observed itself put the lower 95% bound of its success rate above {pct(d.thresholds.wilson_lower, 0)} (the tick), the relay
        closes and it acts alone; if the rate falls, it goes back to asking. Some actions always need a person:{" "}
        {d.never_autonomous.map(action).join(" and ")}.
      </p>
      <div className="table-wrap">
        <table className="ladder">
          <caption>Observed success of each remote action (dot), its 95% interval (bar) and who decides now</caption>
          <thead>
            <tr>
              <th scope="col">Action</th>
              {models.map((m) => (
                <th scope="col" key={m}>
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {actions.map((a) => (
              <tr key={a}>
                <th scope="row">{action(a)}</th>
                {models.map((m) => {
                  const r = rungs.find((x) => x.action === a && x.model === m);
                  if (!r) return <td key={m} className="ladder-empty">no outcomes</td>;
                  const change = r.changes.at(-1);
                  return (
                    <td key={m} className={`rung rung--${r.level}`}>
                      <span className="rung-level">{r.level === "act" ? "acts alone" : "operator approves"}</span>
                      <Bound r={r} threshold={d.thresholds.wilson_lower} />
                      <span className="rung-count">
                        {r.successes} of {r.outcomes} worked{change ? `; ${change.to === "act" ? "earned" : "lost"} ${when(change.ts)}` : ""}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};
