import { expect, test } from "@playwright/test";

test("reads on a phone without sideways scrolling of the page", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await page.goto("./#incidents");
  await expect(page.getByRole("img", { name: /Decisions for/ })).toBeVisible();
  const overflow2 = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow2).toBeLessThanOrEqual(1);
  if (process.env.SCREENSHOTS) await page.screenshot({ path: "../docs/screenshots/07-phone.png", fullPage: true });
});
