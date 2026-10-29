import { motion, useReducedMotion } from "motion/react";
import type { Cycle } from "../lib/data";
import { action as actionName, pct, remedy } from "../lib/format";
import { DURATION, EASE } from "../theme/motion";

/** An incident's decisions drawn as a single-line diagram: the wire runs from the symptom through the
 * model, the reading and the planner to the shield (a breaker, drawn open when a rule blocks) and the
 * autonomy relay (straight through when the autopilot may act, to the operator when it must ask), then
 * to the OCPP command and its outcome. */

const NODES = ["model", "reading", "planner", "shield", "who", "OCPP", "outcome"] as const;
const W = 860;
const PAD = 70;
const STEP = (W - 2 * PAD) / (NODES.length - 1);
const ROW = 92;
const BUS = 34;
const BREAKER_GAP = 18;

const x = (i: number) => PAD + i * STEP;

type Tone = "act" | "propose" | "open" | "held" | "idle";

function tones(c: Cycle): Tone[] {
  const blocked = c.shield && !c.shield.allowed;
  const reached = !!c.action;
  const route: Tone = c.route === "act" ? "act" : "propose";
  return [
    "act",
    "act",
    reached ? "act" : "idle",
    blocked ? "open" : reached ? "act" : "idle",
    blocked || !reached ? "idle" : route,
    blocked || !reached || !c.command ? "idle" : route,
    c.outcome === "worked" ? "held" : c.outcome === "did not work" ? "open" : "idle",
  ];
}

const label = (c: Cycle, i: number): [string, string] => {
  switch (NODES[i]) {
    case "model":
      return [c.diagnosis ? remedy(c.diagnosis) : "no model", c.p != null ? pct(c.p, 0) : ""];
    case "reading":
      return c.reading
        ? [c.safety ? "safety concern" : `suggests ${actionName(c.reading)}`, c.quarantined ? `${c.quarantined} field quarantined` : "text evidence"]
        : ["no text", "nothing to read"];
    case "planner":
      return [c.action ? actionName(c.action) : c.plan, c.action ? "chosen" : ""];
    case "shield":
      return c.shield ? [c.shield.allowed ? "allowed" : "blocked", c.shield.allowed ? "every rule passed" : c.shield.rules.join(", ")] : ["—", ""];
    case "who":
      return c.action && c.shield?.allowed ? [c.route === "act" ? "autopilot" : "operator", c.route === "act" ? "acts alone" : "approves first"] : ["—", ""];
    case "OCPP":
      return c.command ? [c.command.result, c.command.decided_by] : ["—", ""];
    case "outcome":
      return [c.outcome ?? "pending", ""];
  }
  return ["", ""];
};

export const DecisionLine = ({ cycles, title }: { cycles: Cycle[]; title: string }) => {
  const reduce = useReducedMotion();
  const h = BUS + cycles.length * ROW + 10;
  return (
    <figure className="sld">
      <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-labelledby="sld-title sld-desc">
        <title id="sld-title">{title}</title>
        <desc id="sld-desc">
          {cycles
            .map((c, k) => `Decision ${k + 1}: ${label(c, 2)[0]}; shield ${label(c, 3)[0]}; ${label(c, 4)[0]}; result ${label(c, 6)[0]}.`)
            .join(" ")}
        </desc>
        {NODES.map((n, i) => (
          <text key={n} x={x(i)} y={16} className="sld-head" textAnchor="middle">
            {n}
          </text>
        ))}
        {cycles.map((c, k) => {
          const y = BUS + k * ROW + 24;
          const t = tones(c);
          return (
            <g key={`${c.ts}-${k}`}>
              {NODES.slice(0, -1).map((_, i) => {
                const tone = t[i + 1] === "idle" ? "idle" : t[i] === "open" ? "idle" : t[i + 1];
                const x0 = x(i) + (NODES[i] === "shield" ? BREAKER_GAP : 0);
                const x1 = x(i + 1) - (NODES[i + 1] === "shield" ? BREAKER_GAP : 0);
                return (
                  <motion.path
                    key={i}
                    d={`M${x0},${y} L${x1},${y}`}
                    className={`wire wire--${tone}`}
                    initial={reduce ? false : { pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: DURATION.flow / NODES.length, delay: (k * NODES.length + i) * DURATION.step, ease: EASE }}
                  />
                );
              })}
              {NODES.map((n, i) => {
                const [main, sub] = label(c, i);
                const tone = t[i];
                return (
                  <g key={n} transform={`translate(${x(i)},${y})`}>
                    {n === "shield" ? (
                      <g className={`breaker breaker--${tone}`}>
                        <circle cx={-BREAKER_GAP} cy={0} r={3} />
                        <circle cx={BREAKER_GAP} cy={0} r={3} />
                        <line x1={-BREAKER_GAP} y1={0} x2={BREAKER_GAP} y2={tone === "open" ? -BREAKER_GAP : 0} />
                      </g>
                    ) : (
                      <circle r={n === "outcome" ? 7 : 5} className={`node node--${tone}`} />
                    )}
                    <text y={26} textAnchor="middle" className="sld-main">
                      {main}
                    </text>
                    <text y={41} textAnchor="middle" className="sld-sub">
                      {sub}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <figcaption className="sld-key">
        <span>
          <i className="key key--act" /> the autopilot acted
        </span>
        <span>
          <i className="key key--propose" /> an operator approved
        </span>
        <span>
          <i className="key key--open" /> a breaker the shield opened, or a fix that did not hold
        </span>
        <span>
          <i className="key key--held" /> fixed and holding
        </span>
      </figcaption>
    </figure>
  );
};
