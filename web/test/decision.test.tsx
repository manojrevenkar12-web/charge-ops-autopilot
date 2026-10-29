import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionLine } from "../src/components/DecisionLine";
import { cycle } from "./fixtures";

describe("DecisionLine", () => {
  it("draws a closed breaker and an autonomous route when the shield allows and the autopilot may act", () => {
    const { container } = render(<DecisionLine cycles={[cycle()]} title="Decisions" />);
    expect(container.querySelector(".breaker--act")).not.toBeNull();
    expect(container.querySelector(".breaker--open")).toBeNull();
    expect(screen.getByText("autopilot")).toBeInTheDocument();
    expect(screen.getByText("worked")).toBeInTheDocument();
  });

  it("opens the breaker and names the rule when the shield blocks; nothing reaches the charger", () => {
    const c = cycle({ shield: { allowed: false, rules: ["no-reset-on-insulation-fault"] }, command: null, outcome: null });
    const { container } = render(<DecisionLine cycles={[c]} title="Decisions" />);
    expect(container.querySelector(".breaker--open")).not.toBeNull();
    expect(screen.getByText("no-reset-on-insulation-fault")).toBeInTheDocument();
    expect(container.querySelectorAll(".wire--act").length).toBeLessThan(4);
  });

  it("routes to an operator when autonomy has not been earned", () => {
    render(<DecisionLine cycles={[cycle({ route: "propose", command: { result: "Accepted", decided_by: "operator" } })]} title="d" />);
    expect(screen.getAllByText("operator").length).toBeGreaterThan(0);
  });

  it("gives screen readers a sentence per decision", () => {
    const { container } = render(<DecisionLine cycles={[cycle(), cycle({ outcome: "did not work" })]} title="Two decisions" />);
    expect(container.querySelector("desc")?.textContent).toMatch(/Decision 1: .*Decision 2: /);
  });
});
