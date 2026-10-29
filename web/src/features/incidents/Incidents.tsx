import { useMemo, useState } from "react";
import { DecisionLine } from "../../components/DecisionLine";
import { EmptyState, ErrorState, Skeleton } from "../../components/States";
import { type Incident, useData } from "../../lib/data";
import { hours, remedy, when } from "../../lib/format";

const FILTERS: { id: string; label: string; test: (i: Incident) => boolean }[] = [
  { id: "all", label: "All", test: () => true },
  { id: "blocked", label: "Shield blocked", test: (i) => i.tags.includes("blocked") },
  { id: "autonomous", label: "Acted alone", test: (i) => i.tags.includes("autonomous") },
  { id: "silent", label: "Silent failures", test: (i) => i.tags.includes("silent") },
  { id: "safety", label: "Safety", test: (i) => i.tags.includes("safety") },
  { id: "preventive", label: "Preventive", test: (i) => i.preventive },
  { id: "quarantined", label: "Text quarantined", test: (i) => i.tags.includes("quarantined") },
];

export const Incidents = () => {
  const q = useData<Incident[]>("incidents");
  const [filter, setFilter] = useState("blocked");
  const [picked, setPicked] = useState<string | null>(null);
  const list = useMemo(() => (q.data ?? []).filter(FILTERS.find((f) => f.id === filter)?.test ?? (() => true)), [q.data, filter]);
  if (q.isPending) return <Skeleton lines={8} label="Loading incidents" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const current = list.find((i) => i.id === picked) ?? list[0];
  return (
    <section aria-labelledby="inc-title" className="feature">
      <h1 id="inc-title" className="title">Incidents, decision by decision</h1>
      <p className="lead">
        Each line is one pass through the system. Follow the wire: where the shield opens the breaker, nothing reaches the charger;
        where the autonomy relay routes to an operator, the action waited for a person.
      </p>
      <div className="chips" role="group" aria-label="Show incidents">
        {FILTERS.map((f) => (
          <button key={f.id} type="button" className="chip" aria-pressed={filter === f.id} onClick={() => { setFilter(f.id); setPicked(null); }}>
            {f.label}
          </button>
        ))}
      </div>
      {list.length === 0 || !current ? (
        <EmptyState title="Nothing in this view">No incident in the exported weeks matches this filter. Try another one.</EmptyState>
      ) : (
        <div className="inc-grid">
          <ul className="inc-list" aria-label="Incidents">
            {list.slice(0, 60).map((i) => (
              <li key={i.id}>
                <button type="button" aria-current={i.id === current.id ? "true" : undefined} onClick={() => setPicked(i.id)}>
                  <span className="inc-id">{i.charger_id}</span>
                  <span className="inc-what">{i.top ? remedy(i.top.cls) : i.symptoms.join(", ")}</span>
                  <span className="inc-meta">
                    {i.model} · {i.resolution || "open"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <article className="inc-detail" aria-labelledby="inc-detail-title">
            <h2 id="inc-detail-title">
              {current.charger_id} <small>{current.model}, {current.country}</small>
            </h2>
            <p className="inc-sub">
              Opened {when(current.opened)} on {current.symptoms.join(", ") || "a risk score"}
              {current.closed ? `, closed after ${hours(current.opened, current.closed)} (${current.resolution})` : ", still open"}.
              {current.truth ? (
                <>
                  {" "}
                  <span className="truth">Simulator's truth: {current.truth.replaceAll("_", " ")}.</span>
                </>
              ) : null}
            </p>
            {current.cycles.length ? (
              <DecisionLine key={current.id} cycles={current.cycles} title={`Decisions for ${current.charger_id}`} />
            ) : (
              <EmptyState title="No remote decision">This incident went straight to a technician or was watched until it cleared.</EmptyState>
            )}
            {current.visits.length > 0 && (
              <ul className="visits" aria-label="Technician visits">
                {current.visits.map((v) => (
                  <li key={v.ts}>
                    <b>{when(v.ts)}</b> {v.outcome.replaceAll("_", " ")}
                    {v.part_used ? `, used a ${v.part_used.replaceAll("_", " ")}` : ""}
                    {v.part_needed ? `, needs a ${v.part_needed.replaceAll("_", " ")}` : ""}
                  </li>
                ))}
              </ul>
            )}
          </article>
        </div>
      )}
    </section>
  );
};
