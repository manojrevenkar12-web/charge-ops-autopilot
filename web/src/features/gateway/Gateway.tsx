import { EmptyState, ErrorState, Skeleton } from "../../components/States";
import { type Gateway as Data, useData } from "../../lib/data";
import { eur, int } from "../../lib/format";

export const Gateway = () => {
  const q = useData<Data>("gateway");
  if (q.isPending) return <Skeleton lines={5} label="Loading the gateway" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const d = q.data;
  if (!d.use_cases.length) return <EmptyState title="No model calls">The autopilot made no model calls in this run.</EmptyState>;
  return (
    <section aria-labelledby="gw-title" className="feature">
      <h1 id="gw-title" className="title">The LLM gateway, by use case</h1>
      <p className="lead">
        Every model call goes through one gateway: each use case has an owner, a route with a fallback, a token limit and a daily
        budget; personal data is masked before a prompt leaves, and text from chargers or drivers that reads like an instruction is
        quarantined. {d.note}
      </p>
      <div className="table-wrap">
        <table className="compare">
          <caption>Calls during the test weeks (provider: {d.provider})</caption>
          <thead>
            <tr>
              <th scope="col">Use case</th>
              <th scope="col">Route</th>
              <th scope="col">Calls (cache)</th>
              <th scope="col">Cost a day</th>
              <th scope="col">Budget a day</th>
              <th scope="col">Quarantined</th>
              <th scope="col">Masked</th>
              <th scope="col">Refused</th>
            </tr>
          </thead>
          <tbody>
            {d.use_cases.map((u) => (
              <tr key={u.name}>
                <th scope="row">
                  {u.name}
                  <small>{u.purpose}</small>
                </th>
                <td>{u.route.join(" → ")}</td>
                <td>
                  {int(u.calls)} ({int(u.cache_hits)})
                </td>
                <td>{eur(u.cost_eur_per_day)}</td>
                <td>{eur(u.daily_budget_eur)}</td>
                <td>{int(u.quarantined)}</td>
                <td>{int(u.masked)}</td>
                <td>{int(u.refused)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};
