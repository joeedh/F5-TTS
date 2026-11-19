// Browser-side DiT parity run (plan 01, step 3), teacher-forced like tools/webgpu/check_dit_ort.py: every
// step of every chunk gets the golden x, t and step_cond. Checks dit.out, dit.v and dit.x_next per step,
// text_embed for chunk 0, and with `debug` the internals at the golden dit_internal steps.
// Loaded into the page by the Playwright tests in test/browser/ and test/browser-local/.
import * as ort from "onnxruntime-web/webgpu";
import { DiT } from "../../src/dit";
import {
  compare,
  httpReader,
  judge,
  loadCase,
  loadTolerances,
  metrics,
  report,
  xNextGolden,
  type Result,
  type Tensor,
} from "../parity";

export interface DitParityOptions {
  caseName: string;
  /** Folder URL with text.onnx and step*.onnx, such as "/testdata/webgpu/tiny/onnx/" or "/model/dit/base/". */
  modelBase: string;
  goldenBase?: string;
  attention?: "unfused" | "mha" | "onnx23";
  /** Run step_debug*.onnx instead of step*.onnx and check the internal outputs too. */
  debug?: boolean;
  /** Only the first steps of each chunk. */
  maxSteps?: number;
}

/** Node placement per session, read from ORT's verbose log while the sessions are created. */
export interface Placement {
  /** Per session in creation order (text, then step): execution provider to node count. */
  sessions: Record<string, number>[];
  /** "OpType (name)" of every node placed on the CPU, per session. */
  cpuNodes: string[][];
  /** CPU nodes whose op type is not shape arithmetic; the criterion allows none. */
  offenders: string[];
  /** MemcpyTransformer's "modified" flags; 1 means copy nodes were inserted. */
  memcpyModified: number[];
}

// Ops that may sit on the CPU when they compute shapes (placement criterion, plan 01). The log gives op
// types and names, not whether the inputs are shape tensors, so the names of any CPU nodes are reported too.
const SHAPE_OPS = new Set(["Shape", "Gather", "Concat", "Range", "Slice", "Unsqueeze", "Squeeze", "Cast", "Reshape"]);

export function parsePlacement(lines: string[]): Placement {
  const sessions: Record<string, number>[] = [];
  const cpuNodes: string[][] = [];
  const memcpyModified: number[] = [];
  let ep = "";
  for (const line of lines) {
    const memcpy = /GraphTransformer MemcpyTransformer modified: (\d)/.exec(line);
    if (memcpy) memcpyModified.push(Number(memcpy[1]));
    if (line.includes("VerifyEachNodeIsAssignedToAnEp] Node placements")) {
      sessions.push({});
      cpuNodes.push([]);
      continue;
    }
    const placed = /(?:All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/.exec(line);
    const current = sessions.at(-1);
    if (placed && current) {
      ep = placed[1] ?? "";
      current[ep] = Number(placed[2]);
      continue;
    }
    const node = /VerifyEachNodeIsAssignedToAnEp\]\s{2,}(\S+ \(.*\))\s*$/.exec(line);
    if (node && ep === "CPUExecutionProvider") cpuNodes.at(-1)?.push(node[1] ?? "");
  }
  const offenders = cpuNodes.flat().filter((n) => !SHAPE_OPS.has(n.split(" ")[0] ?? ""));
  return { sessions, cpuNodes, offenders, memcpyModified };
}

function cpuTensor(t: Tensor): ort.Tensor {
  return new ort.Tensor(t.dtype, t.data, t.shape);
}

async function readBack(t: ort.Tensor): Promise<Tensor> {
  return { dtype: "float32", shape: [...t.dims], data: (await t.getData()) as Float32Array };
}

/** Frames [start, n) of a (1, n, d) tensor. */
function fromFrame(t: Tensor, start: number): Float32Array {
  const d = t.shape[2] ?? 0;
  return (t.data as Float32Array).subarray(start * d);
}

const pad2 = (s: number) => String(s).padStart(2, "0");

export async function runDitParity(options: DitParityOptions) {
  const t0 = performance.now();
  const read = httpReader(options.goldenBase ?? "/testdata/webgpu");
  const golden = await loadCase(read, options.caseName);
  const tolerances = await loadTolerances(read);
  const settings = golden.manifest.settings as { cfg_strength: number; nfe_step: number; dit_internal_steps: number[] };
  const suffix = (options.attention ?? "unfused") === "unfused" ? "" : `_${options.attention}`;
  const stepFile = `${options.debug ? "step_debug" : "step"}${suffix}.onnx`;

  // ORT prints node placement to the console at verbose level while it creates a session
  const logLines: string[] = [];
  const saved = {
    log: console.log,
    info: console.info,
    warn: console.warn,
    error: console.error,
    debug: console.debug,
  };
  for (const level of Object.keys(saved) as (keyof typeof saved)[]) {
    console[level] = (...args: unknown[]) => void logLines.push(args.map(String).join(" "));
  }
  ort.env.logLevel = "verbose";
  let dit: DiT;
  try {
    dit = await DiT.create({
      textUrl: `${options.modelBase}text.onnx`,
      stepUrl: `${options.modelBase}${stepFile}`,
      logSeverityLevel: 0,
    });
  } finally {
    ort.env.logLevel = "warning";
    Object.assign(console, saved);
  }
  const placement = parsePlacement(logLines);
  const loadMs = performance.now() - t0;

  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));

  const results: Result[] = [];
  const derived = (check: string, name: string, got: ArrayLike<number>, ref: ArrayLike<number>) =>
    results.push(judge(golden, tolerances, "dit", name, check, metrics(got, ref)));
  const internalSteps = new Set(settings.dit_internal_steps);
  const nfe = options.maxSteps === undefined ? settings.nfe_step : Math.min(settings.nfe_step, options.maxSteps);
  try {
    for (const [c, chunk] of golden.manifest.chunks.entries()) {
      const n = (chunk as { duration: number }).duration;
      const ids = (await golden.tensor(`c${c}.dit.text_ids`)).data as Int32Array;
      const text = await dit.embedText(ids, n);
      if (c === 0) {
        results.push(...(await compare(golden, tolerances, "c0.dit_internal.text_embed", await readBack(text.cond))));
        results.push(
          ...(await compare(golden, tolerances, "c0.dit_internal.text_embed_uncond", await readBack(text.uncond))),
        );
      }
      const grid = (await golden.tensor(`c${c}.plan.t`)).data;
      const stepCond = cpuTensor(await golden.tensor(`c${c}.dit.step_cond`));
      for (let s = 0; s < nfe; s++) {
        const p = `c${c}.dit.s${pad2(s)}`;
        const out = await dit.step({
          x: cpuTensor(await golden.tensor(`${p}.x`)),
          stepCond,
          text,
          time: (await golden.tensor(`${p}.t`)).data[0] ?? NaN,
          cfgStrength: settings.cfg_strength,
          dt: (grid[s + 1] ?? NaN) - (grid[s] ?? NaN),
        });
        try {
          const pred = await readBack(out.pred);
          const half = pred.data.length / 2;
          const shape = [1, n, 100];
          const cond = pred.data.subarray(0, half);
          const uncond = pred.data.subarray(half);
          results.push(
            ...(await compare(golden, tolerances, `${p}.out_cond`, { dtype: "float32", shape, data: cond })),
          );
          results.push(
            ...(await compare(golden, tolerances, `${p}.out_uncond`, { dtype: "float32", shape, data: uncond })),
          );
          const gc = (await golden.tensor(`${p}.out_cond`)).data;
          const gu = (await golden.tensor(`${p}.out_uncond`)).data;
          const cfg = settings.cfg_strength;
          const vRef = Float32Array.from(gc, (a, i) => a + (a - (gu[i] ?? NaN)) * cfg);
          derived("dit.v", `${p}.v`, (await readBack(out.v)).data, vRef);
          const { start, ref } = await xNextGolden(golden, c, s);
          derived("dit.x_next", `${p}.x_next`, fromFrame(await readBack(out.x_next), start), ref.data);
          if (options.debug && c === 0 && internalSteps.has(s)) {
            for (const key of dit.stepOutputNames.slice(3)) {
              const name = `c0.dit_internal.s${pad2(s)}.${key}`;
              const t = out[key];
              if (t && golden.entries.has(name))
                results.push(...(await compare(golden, tolerances, name, await readBack(t))));
            }
          }
        } finally {
          for (const t of Object.values(out)) t.dispose();
        }
      }
      text.cond.dispose();
      text.uncond.dispose();
    }
  } finally {
    await dit.release();
  }
  const r = report(results);
  const lines = r.text.split("\n");
  // per check: the error as a fraction of its tolerance, so near misses show as well as failures
  const ratios: Record<string, { name: string; relL2: number; maxAbs: number }[]> = {};
  for (const x of results) {
    if (!x.metrics || !x.tolerance) continue;
    const rel = x.tolerance.relL2 && x.metrics.relL2 !== null ? x.metrics.relL2 / x.tolerance.relL2 : 0;
    const abs = x.tolerance.maxAbs ? x.metrics.maxAbs / x.tolerance.maxAbs : 0;
    (ratios[x.check] ??= []).push({ name: x.name, relL2: rel, maxAbs: abs });
  }
  const errorSpread = Object.fromEntries(
    Object.entries(ratios).map(([check, xs]) => {
      const rel = xs.map((x) => x.relL2).sort((a, b) => a - b);
      const q = (p: number) => Number((rel[Math.min(rel.length - 1, Math.floor(p * rel.length))] ?? 0).toFixed(3));
      const worst = [...xs].sort((a, b) => b.relL2 - a.relL2).slice(0, 3);
      return [check, { count: xs.length, relL2Median: q(0.5), relL2P90: q(0.9), relL2Max: q(1), worst }];
    }),
  );
  return {
    caseName: options.caseName,
    attention: options.attention ?? "unfused",
    adapter: { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture },
    comparisons: results.length,
    failed: r.failed,
    summary: lines.at(-1) ?? "",
    failures: lines.filter((l) => l.startsWith("FAIL")),
    /** Every internal comparison, in order, when `debug` is set. */
    internals: options.debug ? lines.filter((l) => l.includes(" dit_internal ")) : [],
    errorSpread,
    placement,
    gpuErrors,
    loadMs: Math.round(loadMs),
    totalMs: Math.round(performance.now() - t0),
  };
}

export type DitParityResult = Awaited<ReturnType<typeof runDitParity>>;
