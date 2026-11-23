// Browser-side end-to-end run of plan 04 step 5: generateWith from a golden case's input.wav and texts, with
// seed 0 (whose noise is bit-identical to the golden noise, step 2), against final.wav (final.free_running)
// and each chunk's vocoder.wav. Loaded into the page by test/browser-local/final.spec.ts.
import * as ort from "onnxruntime-web/webgpu";
import { parseWav } from "../../src/audio/wav";
import { DiT } from "../../src/dit";
import { DEFAULTS, generateWith, type Settings } from "../../src/pipeline/generate";
import { TensorLedger } from "../../src/pipeline/sample";
import { parseVocab, TextFrontend } from "../../src/text/tokens";
import { Vocoder } from "../../src/vocoder";
import { compare, httpReader, loadCase, loadTolerances, report, type Result } from "../parity";

export interface FinalRunOptions {
  caseName: string;
  ditBase: string;
  vocoderUrl: string;
}

export async function runFinal(options: FinalRunOptions) {
  const read = httpReader("/testdata/webgpu");
  const golden = await loadCase(read, options.caseName);
  const tolerances = await loadTolerances(read);
  const man = golden.manifest as unknown as {
    input: { ref_text: string; gen_text: string };
    settings: Record<string, number | null>;
  };
  const s = man.settings;
  const settings: Settings = {
    ...DEFAULTS,
    nfeStep: s.nfe_step as number,
    cfgStrength: s.cfg_strength as number,
    swaySamplingCoef: s.sway_sampling_coef ?? null,
    speed: s.speed as number,
    crossFadeDuration: s.cross_fade_duration as number,
    fixDuration: s.fix_duration ?? null,
    targetRms: s.target_rms as number,
    seed: 0,
  };
  // ORT Web refuses to create two WebGPU sessions at once ("another WebGPU EP inference session is being created")
  const dit = await DiT.create({ textUrl: `${options.ditBase}text.onnx`, stepUrl: `${options.ditBase}step_mha.onnx` });
  const vocoder = await Vocoder.create({ url: options.vocoderUrl });
  const vocabText = await fetch("/vocab/vocab.txt").then((r) => r.text());
  const wavBuf = await fetch(`/testdata/webgpu/${options.caseName}/input.wav`).then((r) => r.arrayBuffer());
  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));
  const ledger = new TensorLedger();
  const models = { dit, vocoder, vocab: parseVocab(vocabText), text: new TextFrontend(), device, ledger };
  const t0 = performance.now();
  const chunkDone: number[] = [];
  let out;
  try {
    out = await generateWith(
      models,
      { refAudio: parseWav(wavBuf), refText: man.input.ref_text, genText: man.input.gen_text },
      settings,
      { onChunk: () => chunkDone.push(Math.round(performance.now() - t0)) },
    );
  } finally {
    await dit.release();
    await vocoder.release();
  }
  const totalMs = Math.round(performance.now() - t0);
  const results: Result[] = [];
  for (const [c, ch] of out.chunks.entries())
    results.push(
      ...(await compare(
        golden,
        tolerances,
        `c${c}.vocoder.wav`,
        { dtype: "float32", shape: [ch.wave.length], data: ch.wave },
        ["final.free_running"],
      )),
    );
  const wav = { dtype: "float32" as const, shape: [out.wave.length], data: out.wave };
  results.push(...(await compare(golden, tolerances, "final.wav", wav, ["final.free_running"])));
  const r = report(results);
  return {
    caseName: options.caseName,
    adapter: device.adapterInfo.vendor,
    lines: r.text.split("\n"),
    failed: r.failed,
    durations: out.chunks.map((c) => c.duration),
    chunkDoneMs: chunkDone,
    totalMs,
    audioSeconds: out.wave.length / out.sampleRate,
    live: ledger.live,
    gpuErrors,
    wave: Array.from(out.wave),
  };
}

export type FinalRunResult = Awaited<ReturnType<typeof runFinal>>;
