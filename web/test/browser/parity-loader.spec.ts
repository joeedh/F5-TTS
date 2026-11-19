// The parity runner's browser loader reads the committed tiny case over HTTP (plan 00, step 6).
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

const TESTDATA = join(import.meta.dirname, "../../../testdata/webgpu");

test("browser loader reads the tiny case and it passes against itself", async ({ page }) => {
  // testdata/ is outside Vite's root, so the test serves it directly
  await page.route("**/testdata/webgpu/**", async (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^\/testdata\/webgpu\//, "");
    try {
      await route.fulfill({ body: await readFile(join(TESTDATA, decodeURIComponent(path))) });
    } catch {
      await route.fulfill({ status: 404 });
    }
  });
  await page.goto("/");
  const summary = await page.evaluate(async () => {
    const url = "/test/parity/index.ts";
    const p = await import(/* @vite-ignore */ url);
    const read = p.httpReader("/testdata/webgpu");
    const golden = await p.loadCase(read, "tiny");
    const tolerances = await p.loadTolerances(read);
    const results = [];
    for (const e of golden.manifest.tensors) {
      if (p.checksFor(e.name).length === 0) continue;
      results.push(...(await p.compare(golden, tolerances, e.name, await golden.tensor(e.name))));
    }
    const r = p.report(results);
    return { comparisons: results.length, failed: r.failed, last: r.text.split("\n").at(-1) };
  });
  expect(summary.failed, summary.last).toBe(0);
  expect(summary.comparisons).toBeGreaterThan(30);
});
