import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Incidents } from "../src/features/incidents/Incidents";
import { incident } from "./fixtures";

const wrap = (ui: React.ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{ui}</QueryClientProvider>
);

afterEach(() => vi.restoreAllMocks());

const respond = (body: unknown, ok = true) =>
  vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok, status: ok ? 200 : 500, json: async () => body } as Response);

describe("Incidents", () => {
  it("shows a skeleton while loading", () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise(() => {}));
    render(wrap(<Incidents />));
    expect(screen.getByRole("status", { name: "Loading incidents" })).toBeInTheDocument();
  });

  it("shows an error with a retry that refetches", async () => {
    const f = respond({}, false);
    render(wrap(<Incidents />));
    expect(await screen.findByRole("alert", {}, { timeout: 5000 })).toHaveTextContent("HTTP 500");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(f.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("says when a filter matches nothing", async () => {
    respond([incident({ tags: ["autonomous"] })]);
    render(wrap(<Incidents />));
    expect(await screen.findByText("Nothing in this view")).toBeInTheDocument(); // default filter: shield blocked
    await userEvent.click(screen.getByRole("button", { name: "Acted alone" }));
    expect(screen.getByRole("heading", { level: 2, name: /DE-0042/ })).toBeInTheDocument();
  });

  it("shows the simulator's truth next to the incident", async () => {
    respond([incident({ tags: ["blocked"] })]);
    render(wrap(<Incidents />));
    expect(await screen.findByText(/Simulator's truth: controller hang/)).toBeInTheDocument();
  });
});
