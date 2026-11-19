import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Node tests only; test/browser/ is Playwright's (npm run test:browser)
    include: ["test/**/*.test.ts"],
    exclude: ["test/browser/**", "node_modules/**"],
  },
});
