// Vite dev-server plugin that serves folders outside web/ (golden data, exported models) by URL prefix.
// Files are streamed with a length, so the 1.3 GB DiT weights never sit in Node's memory.
import { createReadStream, statSync } from "node:fs";
import { normalize, resolve } from "node:path";
import type { Plugin } from "vite";

/** `roots` maps a URL prefix ending in "/" to a folder. Missing files are a 404, not a fallthrough. */
export function serveData(roots: Record<string, string>): Plugin {
  const dirs = Object.entries(roots).map(([prefix, dir]) => [prefix, resolve(dir)] as const);
  return {
    name: "serve-data",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url ?? "").split("?")[0] ?? "");
        const hit = dirs.find(([prefix]) => url.startsWith(prefix));
        if (!hit) return next();
        const [prefix, dir] = hit;
        const file = normalize(resolve(dir, url.slice(prefix.length)));
        let size: number;
        try {
          if (!file.startsWith(dir)) throw new Error("outside the root");
          size = statSync(file).size;
        } catch {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Content-Length", size);
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Cache-Control", "max-age=86400");
        createReadStream(file).pipe(res);
      });
    },
  };
}
