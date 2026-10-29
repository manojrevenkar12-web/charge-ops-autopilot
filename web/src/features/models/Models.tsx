import { EmptyState, ErrorState, Skeleton } from "../../components/States";
import { type Models as Data, useData } from "../../lib/data";
import { when } from "../../lib/format";

const num = (v: unknown, d = 3) => (typeof v === "number" ? v.toFixed(d) : "—");

export const Models = () => {
  const q = useData<Data>("models");
  if (q.isPending) return <Skeleton lines={6} label="Loading the models" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const d = q.data;
  const diag = d.bootstrap.diagnosis;
  const risk = d.bootstrap.risk;
  const drift = d.drift.at(-1);
  return (
    <section aria-labelledby="mo-title" className="feature">
      <h1 id="mo-title" className="title">Models in production</h1>
      <p className="lead">
        Two models decide nothing on their own: diagnosis says what will probably fix an incident, failure risk says which charger
        is about to fail. A new version is trained every Monday, must beat the champion on the most recent weeks, give the same
        answers in ONNX and fit the latency budget, then is scored in the shadow of the champion before it may be used.
      </p>
      <div className="cards">
        {diag && (
          <article className="card">
            <h2>Diagnosis, champion v{d.champion.diagnosis ?? diag.version}</h2>
            <dl>
              <dt>top-1 / top-2 on the last 4 weeks</dt>
              <dd>
                {num(diag.metrics.accuracy, 2)} / {num(diag.metrics.top2, 2)}
              </dd>
              <dt>macro-F1</dt>
              <dd>{num(diag.metrics.macro_f1, 2)}</dd>
              <dt>ONNX agreement, p99 difference</dt>
              <dd>
                {num(diag.checks.argmax_agreement, 4)}, {Number(diag.checks.p99_abs_diff).toExponential(1)}
              </dd>
              <dt>p95 latency, 64 incidents</dt>
              <dd>
                {num(diag.checks.p95_ms_onnx, 1)} ms ONNX, {num(diag.checks.p95_ms_native, 1)} ms native
              </dd>
            </dl>
          </article>
        )}
        {risk && (
          <article className="card">
            <h2>Failure risk, champion v{d.champion.risk ?? risk.version}</h2>
            <dl>
              <dt>PR-AUC (base rate)</dt>
              <dd>
                {num(risk.metrics.pr_auc, 2)} ({num(risk.metrics.base_rate, 3)})
              </dd>
              <dt>precision in each day's top 2%</dt>
              <dd>{num(risk.metrics.precision_top2pct, 2)}</dd>
              <dt>p95 latency, 64 chargers</dt>
              <dd>{num(risk.checks.p95_ms_onnx, 2)} ms ONNX</dd>
            </dl>
          </article>
        )}
      </div>
      <h2 className="sub">What happened to the models during the test weeks</h2>
      {d.events.length === 0 ? (
        <EmptyState title="No retraining yet">The first retraining runs on the first Monday of a run.</EmptyState>
      ) : (
        <ol className="timeline">
          {d.events.map((e, i) => (
            <li key={`${e.ts}-${i}`}>
              <b>{when(e.ts)}</b> <span>{describe(e)}</span>
            </li>
          ))}
        </ol>
      )}
      <h2 className="sub">Drift of the features the diagnosis model sees</h2>
      {drift ? (
        <p className="lead">
          Last check: {d.drift.filter((x) => x.alert).length} of {d.drift.length} daily checks raised an alert (PSI above {d.psi_alert}).
          Largest shifts: {drift.top.map((t) => `${t.feature} ${t.psi.toFixed(2)}`).join(", ")}.
        </p>
      ) : (
        <EmptyState title="No drift checks yet">Drift is checked every day once the model has scored incidents.</EmptyState>
      )}
    </section>
  );
};

function describe(e: Record<string, unknown>): string {
  if (e.event === "start") return `run starts with diagnosis v${e.diagnosis} and risk v${e.risk}`;
  if (e.event === "retrain") {
    const g = e.diagnosis_gate as { passed: boolean; reasons: string[]; candidate_macro_f1?: number; champion_macro_f1?: number };
    const r = e.risk_gate as { passed: boolean; reasons: string[] };
    return `retrained on ${e.examples} examples: diagnosis ${g.passed ? `passed the gate (macro-F1 ${num(g.candidate_macro_f1, 2)} vs ${num(g.champion_macro_f1, 2)}), shadow starts` : `held back (${g.reasons.join("; ")})`}; risk ${r.passed ? `promoted to v${e.risk_promoted}` : `held back (${r.reasons.join("; ")})`}`;
  }
  if (e.event === "shadow_verdict")
    return `shadow verdict on ${e.n} resolved incidents: challenger right ${e.challenger_correct}, champion ${e.champion_correct}; ${e.promoted ? `v${e.version} promoted` : "champion kept"}`;
  return String(e.event);
}
