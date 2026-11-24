// Plan 05 step 3: loads one weight variant (tools/webgpu/convert_precision.py) through DiT.create and
// Vocoder.create, as TTS.create does, then runs one teacher-forced step of chunk 0 and the vocoder on
// chunk 0's golden mel. Loading and running are separate calls, so run.mjs can read peak memory between. The comparisons with the fp32 golden data are
// informational here; step 4 checks each variant against golden data made with its own weights.
import * as ort from "onnxruntime-web/webgpu";
import { compare, httpReader, loadCase, loadTolerances, type Result, type Tensor } from "../../test/parity";
import { DiT } from "../../src/dit";
import { Vocoder } from "../../src/vocoder";
import { checkFeatures, variantSources } from "../../src/weights/variant";

const ms = (t0: number) => Math.round(performance.now() - t0);

function cpu(t: Tensor): ort.Tensor {
  return new ort.Tensor(t.dtype, t.data, t.shape);
}

let models: { dit: DiT; vocoder: Vocoder } | null = null;

async function loadVariant(variant: string): Promise<Record<string, unknown>> {
  const t0 = performance.now();
  const { info: variantInfo, sources } = await variantSources(`/variants/${variant}/`);
  await checkFeatures(variantInfo);
  const dit = await DiT.create({ text: sources.text, step: sources.step });
  const ditMs = ms(t0);
  const vocoder = await Vocoder.create({ source: sources.vocos });
  const loadMs = ms(t0);
  models = { dit, vocoder };
  const device = await ort.env.webgpu.device;
  const { vendor, architecture } = device.adapterInfo;
  return { ditMs, loadMs, adapter: { vendor, architecture } };
}

async function runStep(caseName: string): Promise<Record<string, unknown>> {
  if (!models) throw new Error("call loadVariant first");
  const { dit, vocoder } = models;
  const read = httpReader("/golden");
  const golden = await loadCase(read, caseName);
  const tolerances = await loadTolerances(read);
  const n = (golden.manifest.chunks[0] as { duration: number }).duration;
  const cfg = golden.manifest.settings.cfg_strength as number;
  const grid = (await golden.tensor("c0.plan.t")).data;

  const results: Result[] = [];
  const ids = (await golden.tensor("c0.dit.text_ids")).data as Int32Array;
  const text = await dit.embedText(ids, n);
  const feeds = {
    x: cpu(await golden.tensor("c0.dit.s00.x")),
    stepCond: cpu(await golden.tensor("c0.dit.step_cond")),
    text,
    time: (await golden.tensor("c0.dit.s00.t")).data[0] as number,
    cfgStrength: cfg,
    dt: (grid[1] as number) - (grid[0] as number),
  };
  const runMs: number[] = [];
  for (let i = 0; i < 3; i++) {
    const r0 = performance.now();
    const out = await dit.step(feeds);
    const pred = (await out.pred.getData()) as Float32Array;
    runMs.push(ms(r0));
    if (i === 0) {
      const half = pred.length / 2;
      for (const [name, data] of [
        ["c0.dit.s00.out_cond", pred.subarray(0, half)],
        ["c0.dit.s00.out_uncond", pred.subarray(half)],
      ] as const)
        results.push(...(await compare(golden, tolerances, name, { dtype: "float32", shape: [1, n, 100], data })));
    }
    for (const t of Object.values(out)) t.dispose();
  }
  text.cond.dispose();
  text.uncond.dispose();
  const mel = cpu(await golden.tensor("c0.mel_out"));
  const { wave } = await vocoder.decode(mel);
  results.push(
    ...(await compare(golden, tolerances, "c0.vocoder.wav", { dtype: "float32", shape: [wave.length], data: wave })),
  );
  await dit.release();
  await vocoder.release();
  models = null;
  const comparisons = results.map((r) => ({ name: r.name, passed: r.pass, relL2: r.metrics?.relL2 ?? null }));
  return { caseName, n, stepMs: runMs, comparisons };
}

declare global {
  interface Window {
    loadVariant: typeof loadVariant;
    runStep: typeof runStep;
  }
}
window.loadVariant = loadVariant;
window.runStep = runStep;
document.body.dataset.ready = "1";
