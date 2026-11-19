import { defineConfig } from "@playwright/test";

// Browser tests that need WebGPU. On Windows the installed Chrome is used; elsewhere (CI, the
// cloud container) Playwright's Chromium on SwiftShader, which has WebGPU but no GPU.
// PW_CHANNEL overrides the channel, PW_GPU=nvidia picks the discrete GPU on the laptop.
// See docs/research/webgpu-laptop.md and docs/research/webgpu-environment.md.
const channel = process.env.PW_CHANNEL ?? (process.platform === "win32" ? "chrome" : "chromium");
const args = ["--enable-unsafe-webgpu"];
if (process.env.PW_GPU === "nvidia") args.push("--force_high_performance_gpu");
if (process.platform !== "win32") {
  args.push(
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  );
}
const port = 5174;

export default defineConfig({
  testDir: "test/browser",
  reporter: "list",
  use: {
    baseURL: `http://localhost:${port}`,
    channel,
    launchOptions: { args },
  },
  // WebGPU needs a secure context, which http://localhost is
  webServer: {
    command: `npx vite --port ${port} --strictPort`,
    url: `http://localhost:${port}`,
    reuseExistingServer: !process.env.CI,
  },
});
