// Plan 03 step 6: builds bench/text/ with Vite's production build, serves it as a CDN would (brotli when the
// browser accepts it, application/wasm), and runs runTextBench() in a fresh browser context per run, so each
// starts with a cold HTTP cache. Usage (from web/): node bench/text/run.mjs [--channel chrome|msedge] [--runs 3]
// Writes results/<channel>.json: the built assets' sizes, and each run's result.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";
import { chromium } from "playwright";
import { build } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "../..");
const argv = process.argv.slice(2);
const arg = (name, dflt) => (argv.includes(`--${name}`) ? argv[argv.indexOf(`--${name}`) + 1] : dflt);
const channel = arg("channel", "chrome");
const runs = Number(arg("runs", "3"));

const outDir = mkdtempSync(join(tmpdir(), "text-bench-"));
await build({
  root: web,
  configFile: false,
  logLevel: "warn",
  base: "/",
  build: { outDir, emptyOutDir: true, rollupOptions: { input: join(here, "index.html") } },
});

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8",
};
const files = new Map();
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else {
      const raw = readFileSync(p);
      const br = brotliCompressSync(raw, {
        params: { [constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_SIZE_HINT]: raw.length },
      });
      files.set("/" + p.slice(outDir.length + 1).replaceAll("\\", "/"), {
        raw,
        br,
        gzip: gzipSync(raw, { level: 9 }).length,
      });
    }
  }
};
walk(outDir);
rmSync(outDir, { recursive: true, force: true });
const built = [...files].map(([path, f]) => ({ path, bytes: f.raw.length, gzip9: f.gzip, brotli11: f.br.length }));

const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replaceAll("\\", "/");
  const f = files.get(path === "/" ? "/bench/text/index.html" : path);
  if (!f) return res.writeHead(404).end();
  const br = /\bbr\b/.test(req.headers["accept-encoding"] ?? "");
  res.writeHead(200, {
    "content-type": TYPES[extname(path)] ?? "application/octet-stream",
    vary: "accept-encoding",
    ...(br ? { "content-encoding": "br" } : {}),
  });
  res.end(br ? f.br : f.raw);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({ channel });
const results = [];
for (let i = 0; i < runs; i++) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  await page.goto(`${url}/bench/text/index.html`);
  await page.waitForSelector("body[data-ready='1']", { timeout: 60_000 });
  const r = await page.evaluate(() => window.runTextBench());
  results.push({ ...r, pageErrors });
  console.log(
    JSON.stringify({
      timing: r.timing,
      longTasksMs: r.longTasksMs,
      perSentenceMs: r.perSentenceMs,
      afterEnglish: r.afterEnglish,
    }),
  );
  await context.close();
}
await browser.close();
server.close();

console.log("assets:", JSON.stringify(results[0]?.afterChinese.assets));
for (const b of built.filter((b) => /jieba|pinyin_dict|\.js$/.test(b.path))) console.log("built:", JSON.stringify(b));
mkdirSync(join(here, "results"), { recursive: true });
writeFileSync(join(here, `results/${channel}.json`), JSON.stringify({ channel, built, results }, null, 1) + "\n");
