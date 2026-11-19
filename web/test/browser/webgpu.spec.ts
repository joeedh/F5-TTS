import { expect, test } from "@playwright/test";

test("WebGPU adapter and device are available", async ({ page }, testInfo) => {
  await page.goto("/");
  const info = await page.evaluate(async () => {
    if (!navigator.gpu) return { hasGpu: false };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { hasGpu: true, adapter: false };
    const device = await adapter.requestDevice();
    return {
      hasGpu: true,
      adapter: true,
      device: !!device,
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      shaderF16: adapter.features.has("shader-f16"),
    };
  });
  testInfo.annotations.push({ type: "adapter", description: JSON.stringify(info) });
  expect(info).toMatchObject({ hasGpu: true, adapter: true, device: true });
});
