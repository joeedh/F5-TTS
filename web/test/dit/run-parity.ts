// Browser-side DiT parity run (plan 01, step 3), teacher-forced like tools/webgpu/check_dit_ort.py: every
// step of every chunk gets the golden x, t and step_cond. Checks dit.out, dit.v and dit.x_next per step,
// text_embed for chunk 0, and with `debug` the internals at the golden dit_internal steps.
// Loaded into the page by the Playwright tests in test/browser/ and test/browser-local/.
import * as ort from "onnxruntime-web/webgpu";
import { DiT } from "../../src/dit";
import { errorSpread, withPlacement } from "../ort/diagnostics";
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

  const { value: dit, placement } = await withPlacement(() =>
    DiT.create({
      textUrl: `${options.modelBase}text.onnx`,
      stepUrl: `${options.modelBase}${stepFile}`,
      logSeverityLevel: 0,
    }),
  );
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
    errorSpread: errorSpread(results),
    placement,
    gpuErrors,
    loadMs: Math.round(loadMs),
    totalMs: Math.round(performance.now() - t0),
  };
}

export type DitParityResult = Awaited<ReturnType<typeof runDitParity>>;
