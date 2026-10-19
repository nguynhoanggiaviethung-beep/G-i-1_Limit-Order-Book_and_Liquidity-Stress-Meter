import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";

const installedChrome = "C:/Program Files/Google/Chrome/Application/chrome.exe";

export default defineConfig({
  testDir: "./e2e",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:5173",
    ...(existsSync(installedChrome) ? { launchOptions: { executablePath: installedChrome } } : {}),
  },
  webServer: { command: "npm run dev", url: "http://127.0.0.1:5173", reuseExistingServer: !process.env.CI },
});
