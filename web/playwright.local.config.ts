import { defineConfig } from "@playwright/test";
import base from "./playwright.config";

// Local-only browser tests that need the real model exports (test/browser-local/). One worker: each test
// holds about 1.3 GB of weights on the GPU, and the two GPUs are picked per run with PW_GPU.
export default defineConfig({
  ...base,
  testDir: "test/browser-local",
  workers: 1,
  timeout: 60 * 60_000,
});
