const fs = require("fs");
const { defineConfig } = require("@playwright/test");

const externalBaseUrl = process.env.BASE_URL;
const runtimeDir = process.env.XDG_RUNTIME_DIR || "/tmp";
const snapChromium = "/snap/chromium/current/usr/lib/chromium-browser/chrome";
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  || (fs.existsSync(snapChromium) ? snapChromium : undefined);

module.exports = defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: {
    timeout: 15_000,
  },
  reporter: "list",
  outputDir: `${runtimeDir}/pulse-playwright-results`,
  use: {
    baseURL: externalBaseUrl || "http://127.0.0.1:8891",
    browserName: "chromium",
    launchOptions: executablePath ? { executablePath } : {},
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: externalBaseUrl
    ? undefined
    : {
        command: "python3 -u app.py --host 127.0.0.1 --port 8891",
        url: "http://127.0.0.1:8891",
        reuseExistingServer: true,
        timeout: 120_000,
      },
});
