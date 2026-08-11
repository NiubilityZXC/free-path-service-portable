const { test, expect } = require("@playwright/test");

const failuresByPage = new WeakMap();

test.beforeEach(async ({ page }) => {
  const failures = [];
  failuresByPage.set(page, failures);
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    failures.push(`requestfailed: ${request.method()} ${request.url()}`);
  });
  page.on("response", (response) => {
    if (response.status() >= 400) {
      failures.push(`http ${response.status()}: ${response.url()}`);
    }
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("#deepModelSelect option")).toHaveCount(11);
  await expect(page.locator("#deepFeatureSelect option")).toHaveCount(5);
  await expect(page.locator("#deepBestTestModel")).toHaveText("量化感知混合模型");
  await expect(page.locator("#engineeringStatus")).not.toHaveText("待评估");
  await expect(page.locator("#run")).toBeEnabled();
});

test.afterEach(async ({ page }) => {
  expect(failuresByPage.get(page)).toEqual([]);
});

async function chartColorCounts(canvas) {
  return canvas.evaluate((element) => {
    const pixels = element
      .getContext("2d")
      .getImageData(0, 0, element.width, element.height).data;
    const targets = {
      actual: [14, 118, 109],
      predicted: [39, 100, 168],
    };
    const counts = { actual: 0, predicted: 0 };
    for (let index = 0; index < pixels.length; index += 4) {
      for (const [name, color] of Object.entries(targets)) {
        if (
          Math.abs(pixels[index] - color[0]) <= 3 &&
          Math.abs(pixels[index + 1] - color[1]) <= 3 &&
          Math.abs(pixels[index + 2] - color[2]) <= 3 &&
          pixels[index + 3] > 200
        ) {
          counts[name] += 1;
        }
      }
    }
    return counts;
  });
}

async function coloredPixelCount(canvas) {
  return canvas.evaluate((element) => {
    const pixels = element
      .getContext("2d")
      .getImageData(0, 0, element.width, element.height).data;
    let colored = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (
        pixels[index + 3] > 0 &&
        (pixels[index] < 235 || pixels[index + 1] < 235 || pixels[index + 2] < 235)
      ) {
        colored += 1;
      }
    }
    return colored;
  });
}

test("all 11 models and 5 features draw actual and predicted curves", async ({ page }) => {
  const modelSelect = page.locator("#deepModelSelect");
  const featureSelect = page.locator("#deepFeatureSelect");
  const canvas = page.locator("#deepPredictionChart");
  const modelOptions = await modelSelect.locator("option").evaluateAll((options) =>
    options.map((option) => ({ value: option.value, label: option.textContent })),
  );
  const featureOptions = await featureSelect.locator("option").evaluateAll((options) =>
    options.map((option) => ({ value: option.value, label: option.textContent })),
  );

  const reportShape = await page.evaluate(async () => {
    const response = await fetch("/api/deep-learning-report");
    const report = await response.json();
    return {
      models: report.models.length,
      groups: report.modelTestSeries.length,
      series: report.modelTestSeries.reduce((sum, group) => sum + group.series.length, 0),
      pointCounts: report.modelTestSeries.flatMap((group) =>
        group.series.map((series) => series.points.length),
      ),
    };
  });
  expect(reportShape.models).toBe(11);
  expect(reportShape.groups).toBe(11);
  expect(reportShape.series).toBe(55);
  expect(new Set(reportShape.pointCounts)).toEqual(new Set([91]));

  for (const model of modelOptions) {
    await modelSelect.selectOption(model.value);
    for (const feature of featureOptions) {
      await featureSelect.selectOption(feature.value);
      const title = page.locator("#deepPredictionTitle");
      await expect(title).toContainText(model.label.split("（")[0]);
      await expect(title).toContainText(feature.label);
      const colors = await chartColorCounts(canvas);
      expect(colors.actual, `${model.label} / ${feature.label}: missing actual curve`).toBeGreaterThan(50);
      expect(colors.predicted, `${model.label} / ${feature.label}: missing predicted curve`).toBeGreaterThan(50);
    }
  }

  await modelSelect.selectOption("gru");
  await featureSelect.selectOption("VoltageMaxRaw");
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  const chartHost = canvas.locator("xpath=..");
  const tooltip = chartHost.locator(".chart-hover-tooltip");
  await expect(tooltip).toBeVisible();
  await expect(tooltip).toContainText("独立测试目标行");
  await expect(tooltip).toContainText("真实值");
  await expect(tooltip).toContainText("GRU预测值");
  await expect(chartHost.locator(".chart-crosshair-vertical")).toBeVisible();
  await expect(chartHost.locator(".chart-crosshair-horizontal")).toBeVisible();
});

test("unified comparison is left of the patent page and both tabs render", async ({ page }) => {
  const tabs = page.locator('[role="tab"]');
  await expect(tabs).toHaveText(["全部模型对比预测", "专利 GM(1,1)"]);
  await expect(page.locator("#deepLearningTab")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#deepLearningPage")).toBeVisible();

  await page.locator("#patentTab").click();
  await expect(page.locator("#patentTab")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#patentPage")).toBeVisible();
  await expect(page.locator("#deepLearningPage")).toBeHidden();
  await expect.poll(() => coloredPixelCount(page.locator("#chart"))).toBeGreaterThan(1000);

  await page.locator("#deepLearningTab").click();
  await expect(page.locator("#deepLearningPage")).toBeVisible();
});

test("life result separates validated range from exploratory extrapolation", async ({ page }) => {
  await expect(page.locator("#deepLifeStatus")).toHaveText("有限寿命暂不可辨识");
  await expect(page.locator("#deepLifeRemaining")).toContainText(">120 步");
  await expect(page.locator("#deepLifeExploratory")).toContainText("未验证");
  expect(await page.locator("#deepLifeRows tr").count()).toBe(5);
  const audit = await page.evaluate(async () =>
    (await (await fetch("/api/deep-learning-report")).json()).lifeForecast.horizonValidation,
  );
  expect(audit.validatedHorizonSteps).toBe(120);
  expect(audit.results.find((item) => item.horizon === 120).accuracyPercent).toBeGreaterThan(85);
});

test("mobile layout has no page overflow and chart tooltip stays inside", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page.locator("#deepModelSelect option")).toHaveCount(11);
  await expect(page.locator("#engineeringStatus")).not.toHaveText("待评估");

  const widths = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  expect(widths.document).toBe(widths.viewport);
  expect(widths.body).toBe(widths.viewport);

  const canvas = page.locator("#deepPredictionChart");
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);

  const chartHost = canvas.locator("xpath=..");
  const tooltip = chartHost.locator(".chart-hover-tooltip");
  await expect(tooltip).toBeVisible();
  const tooltipBox = await tooltip.boundingBox();
  const hostBox = await chartHost.boundingBox();
  expect(tooltipBox).not.toBeNull();
  expect(hostBox).not.toBeNull();
  expect(tooltipBox.x).toBeGreaterThanOrEqual(hostBox.x);
  expect(tooltipBox.x + tooltipBox.width).toBeLessThanOrEqual(hostBox.x + hostBox.width + 1);
});
