import { useEffect, useState } from "react";
import { Autonomy } from "./features/autonomy/Autonomy";
import { Gateway } from "./features/gateway/Gateway";
import { Incidents } from "./features/incidents/Incidents";
import { Models } from "./features/models/Models";
import { Overview } from "./features/overview/Overview";

const SECTIONS = [
  { id: "overview", label: "Status and reality", note: "every policy, same weeks", view: Overview },
  { id: "incidents", label: "Incidents", note: "decision by decision", view: Incidents },
  { id: "autonomy", label: "Autonomy", note: "earned per action and model", view: Autonomy },
  { id: "models", label: "Models", note: "registry, gates, drift", view: Models },
  { id: "gateway", label: "LLM gateway", note: "use cases, cost, guardrails", view: Gateway },
] as const;

const fromHash = () => {
  const h = window.location.hash.replace("#", "");
  return SECTIONS.some((s) => s.id === h) ? h : "overview";
};

export const App = () => {
  const [id, setId] = useState(fromHash);
  useEffect(() => {
    const on = () => setId(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const current = SECTIONS.find((s) => s.id === id) ?? SECTIONS[0];
  const View = current.view;
  return (
    <div className="room">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <nav className="rail" aria-label="Sections">
        <div>
          <p className="brand">charge-ops autopilot</p>
          <p className="brand-sub">Operations room for a simulated fleet of 900 chargers. The autopilot reads OCPP; code decides.</p>
        </div>
        <ul>
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a href={`#${s.id}`} aria-current={s.id === id ? "page" : undefined}>
                <span className="rail-label">{s.label}</span>
                <span className="rail-note">{s.note}</span>
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <main id="main" className="main" tabIndex={-1}>
        <View />
      </main>
    </div>
  );
};
