import { test, expect } from "@playwright/test";

const replaySnapshot = (symbol: "BTCUSDT" | "ETHUSDT") => ({
  symbol,
  timestamp_utc: "2026-10-23T00:00:00Z",
  bids: [["82188.14", "9.75345"], ["82188.13", "0.00035"]],
  asks: [["82188.15", "0.63238"], ["82188.16", "0.00035"]],
  features: { mid_price: 82188.145, relative_spread_bps: 0.0012 },
  quality: { book_valid: true, seconds_since_gap: null, sequence_gap: false },
});

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/data-status", (route) => route.fulfill({ json: {
    status: "ready",
    source: "controlled e2e fixture",
    symbols: Object.fromEntries(["BTCUSDT", "ETHUSDT"].map((symbol) => [symbol, {
      status: "ready", partition_count: 1, row_count: 600, first_date_utc: "2026-10-23",
      last_date_utc: "2026-10-23", reason: null,
    }])),
  } }));

  await page.route("**/api/v1/replay/**", (route) => {
    const url = new URL(route.request().url());
    const symbol = url.pathname.includes("ETHUSDT") ? "ETHUSDT" : "BTCUSDT";
    const snapshot = replaySnapshot(symbol);
    return route.fulfill({ json: url.pathname.endsWith("/at") ? { snapshot } : { snapshots: [snapshot] } });
  });

  await page.route("**/api/v1/meter/**", (route) => {
    const url = new URL(route.request().url());
    const symbol = url.pathname.includes("ETHUSDT") ? "ETHUSDT" : "BTCUSDT";
    return route.fulfill({ json: {
      status: "ok", symbol, timestamp_utc: "2026-10-23T00:00:00Z",
      horizon_seconds: Number(url.searchParams.get("horizon_seconds") ?? 5),
      spread: "0.01", depth_imbalance: { L1: 0.87, L5: 0.88, L10: 0.87, L20: 0.86 },
      ofi_5s: 2.15, p_up: 0.5, p_flat: 0.3, p_down: 0.2, p_stress: 0.65,
      stress_probability_threshold: 0.5, model_version: "fixture-model-v1",
      quality: { book_valid: true },
    } });
  });
});

test("replays a valid book and shows model probabilities and threshold", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "replay" }).click();

  await expect(page.getByText("liquidity stress meter", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "bid · mua" })).toBeVisible();
  await expect(page.getByText("fixture-model-v1").first()).toBeVisible();
  await expect(page.getByText("65.0%").first()).toBeVisible();
  await expect(page.getByText("vượt ngưỡng 50.0%")).toBeVisible();
  await expect(page.getByText("✓ hợp lệ")).toBeVisible();

  await page.getByRole("button", { name: "ETH/USDT" }).click();
  await expect(page.getByRole("button", { name: "ETH/USDT" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("fixture-model-v1").first()).toBeVisible();
});

test("mobile dashboard fits a 390 pixel viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "replay" }).click();
  await expect(page.getByRole("region", { name: "Điều khiển replay" })).toBeVisible();
  const pageWidth = await page.locator("body").evaluate((element) => element.scrollWidth);
  expect(pageWidth).toBeLessThanOrEqual(391);
});
