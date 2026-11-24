// Samples a Chrome instance's memory from outside the browser (dit-load/sample-memory.ps1, Windows only): the
// private bytes of its process tree and its GPU memory, about every 1.5 s. Used by dit-load and variant-load.
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const GB = 1024 ** 3;

/** Starts sampling the Chrome instance whose command line contains `profile`. */
export function startSampler(profile) {
  const samples = [];
  // -ExecutionPolicy Bypass for this process only: spawned from Node, the default policy refuses the script
  const proc = spawn("powershell.exe", [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    join(here, "dit-load/sample-memory.ps1"),
    "-Profile",
    profile,
  ]);
  proc.stderr.on("data", (d) => process.stderr.write(`sample-memory.ps1: ${d}`));
  createInterface({ input: proc.stdout }).on("line", (line) => {
    try {
      samples.push(JSON.parse(line));
    } catch {
      // not a sample
    }
  });
  return {
    samples,
    /** Peaks between two Date.now() times, in GB, or null if no sample fell between them. */
    peak(from, to) {
      const inside = samples.filter((s) => s.t >= from && s.t <= to);
      const before = samples.filter((s) => s.t < from).at(-1);
      const max = (k) => Math.max(...inside.map((s) => s[k]));
      const gb = (x) => Math.round((x / GB) * 100) / 100;
      if (!inside.length) return null;
      return {
        samples: inside.length,
        baseline: before ? { privateGB: gb(before.private), gpuDedicatedGB: gb(before.gpuDedicated) } : null,
        privateGB: gb(max("private")),
        maxProcessPrivateGB: gb(max("maxPrivate")),
        rendererGB: gb(max("renderer")),
        gpuProcessGB: gb(max("gpuProcess")),
        gpuDedicatedGB: gb(max("gpuDedicated")),
        gpuSharedGB: gb(max("gpuShared")),
      };
    },
    /** Windows' per-process peaks so far, in GB, from a sample whose process list was read after this call. */
    async peaksSoFar() {
      const t = Date.now() + 3000; // a sample reads the process list up to about 1.5 s before its time
      let s;
      for (let i = 0; i < 100 && !s; i++) {
        await new Promise((r) => setTimeout(r, 200));
        s = samples.find((x) => x.t >= t);
      }
      const gb = (x) => Math.round((x / GB) * 100) / 100;
      return s ? { rendererGB: gb(s.peakRenderer), gpuProcessGB: gb(s.peakGpuProcess) } : null;
    },
    stop() {
      proc.kill();
    },
  };
}
