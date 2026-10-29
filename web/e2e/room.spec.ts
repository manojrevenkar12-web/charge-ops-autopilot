import { expect, test } from "@playwright/test";

const SHOTS = !!process.env.SCREENSHOTS;
const shot = async (page: import("@playwright/test").Page, name: string) => {
  if (SHOTS) await page.screenshot({ path: `../docs/screenshots/${name}.png`, fullPage: true });
};

test("the overview compares status with reality and shows the gate", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("./");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("The status page said the network was fine");
  await expect(page.getByRole("table", { name: /Every policy/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /Release gate/ })).toBeVisible();
  await shot(page, "01-overview");
  expect(errors, "no console errors under the production CSP").toEqual([]);
});

test("an incident's decision line shows where the shield opened the breaker", async ({ page }) => {
  await page.goto("./#incidents");
  await expect(page.getByRole("heading", { level: 1, name: /Incidents/ })).toBeVisible();
  const line = page.getByRole("img", { name: /Decisions for/ });
  await expect(line).toBeVisible();
  await expect(page.locator(".breaker--open").first()).toBeVisible();
  await page.waitForTimeout(1500);
  await shot(page, "02-incident-blocked");
  await page.getByRole("button", { name: "Acted alone" }).click();
  await expect(page.locator(".wire--act").first()).toBeVisible();
  await page.waitForTimeout(1500);
  await shot(page, "03-incident-autonomous");
});

test("autonomy, models and gateway load", async ({ page }) => {
  await page.goto("./#autonomy");
  await expect(page.getByRole("table", { name: /Observed success/ })).toBeVisible();
  await shot(page, "04-autonomy");
  await page.getByRole("link", { name: /Models/ }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Models in production" })).toBeVisible();
  await shot(page, "05-models");
  await page.getByRole("link", { name: /LLM gateway/ }).click();
  await expect(page.getByRole("table", { name: /Calls during/ })).toBeVisible();
  await shot(page, "06-gateway");
});

test("a missing data file shows an error with a retry, not a blank page", async ({ page }) => {
  await page.route("**/data/autonomy.json", (r) => r.fulfill({ status: 503, body: "" }));
  await page.goto("./#autonomy");
  await expect(page.getByRole("alert")).toContainText("HTTP 503", { timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});
