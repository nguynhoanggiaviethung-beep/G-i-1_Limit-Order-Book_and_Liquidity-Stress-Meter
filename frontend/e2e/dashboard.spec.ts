import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/meta*", (route) => route.fulfill({ json: {
    mode: "research", as_of_utc: "2026-10-01T00:00:00Z", data_version: "test-data-version",
    config_hash: "test-config-hash", symbols: ["BTCUSDT", "ETHUSDT"], horizons_days: [1, 7, 30],
    models: ["har"], confidence_levels: [0.95, 0.975, 0.99], source: "test fixture",
  } }));
  await page.route("**/api/v1/live-market-data/*", (route) => route.fulfill({ json: {
    symbol: "BTCUSDT", connection_status: "connected", candle: null,
  } }));
  await page.route("**/api/v1/regimes/*", (route) => route.fulfill({ json: {
    symbol: "BTCUSDT", mode: "research", as_of_utc: "2026-10-01T00:00:00Z",
    latest_available_date_utc: "2026-09-30T00:00:00Z", regime: "neutral", data_version: "test",
    source: "test fixture", counts: { neutral: 5 },
  } }));
  await page.route("**/api/v1/forecasts/history*", (route) => route.fulfill({ json: {
    symbol: "BTCUSDT", model: "har", horizon_days: 1, mode: "research", risk_method: "test",
    confidence: 0.95, data_version: "test", config_hash: "test", points: [],
  } }));
  await page.route("**/api/v1/forecasts", (route) => route.fulfill({ json: {
    forecast_id: "test", mode: "research", symbol: "BTCUSDT", model: "har", horizon_days: 1,
    origin_time: "2026-10-01T00:00:00Z", target_start: "2026-10-01T00:00:00Z",
    target_end: "2026-10-02T00:00:00Z", data_version: "test", config_hash: "test",
    actual_rv_mean: null, rv_forecast: 1, sigma_h_approx: 1, status: "ok", var_es: [],
  } }));
});

test("loads the six research views without horizontal page overflow", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Tổng quan thị trường" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Điều hướng chính" })).toBeVisible();
  for (const label of ["Dự báo", "Đánh giá mô hình", "Rủi ro & backtest", "Regime & chiến lược", "Phương pháp & dữ liệu"]) {
    await expect(page.getByRole("link", { name: new RegExp(label) })).toBeVisible();
  }
  const violations = await new AxeBuilder({ page }).analyze();
  expect(violations.violations).toEqual([]);
  const pageWidth = await page.locator("body").evaluate((element) => element.scrollWidth);
  const viewportWidth = page.viewportSize()?.width ?? 0;
  expect(pageWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("mobile layout fits a 390 pixel viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Điều hướng di động" })).toBeVisible();
  const pageWidth = await page.locator("body").evaluate((element) => element.scrollWidth);
  expect(pageWidth).toBeLessThanOrEqual(391);
});
