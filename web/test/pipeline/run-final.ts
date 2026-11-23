// Browser-side end-to-end runs of plan 04 steps 5 and 6, through the TTS API: a golden case's input.wav and
// texts with seed 0 (whose noise is bit-identical to the golden noise, step 2), against final.wav
// (final.free_running) and each chunk's vocoder.wav. With `apiChecks`, also the API's behaviour: missing
// seed, cfg below 1e-5, empty reference text, a second call while one runs, and abort.
// Loaded into the page by test/browser-local/final.spec.ts and test/browser/tts-tiny.spec.ts.
import * as ort from "onnxruntime-web/webgpu";
import { parseWav } from "../../src/audio/wav";
import { TTS, type GenerateRequest, type Settings } from "../../src/pipeline";
import { compare, httpReader, loadCase, loadTolerances, report, type Result } from "../parity";

export interface FinalRunOptions {
  caseName: string;
  ditBase: string;
  stepFile?: string;
  vocoderUrl: string;
  apiChecks?: boolean;
}

/** The outcome of each API behaviour check, by name: true when it behaved as it should. */
async function runApiChecks(
  tts: TTS,
  base: GenerateRequest,
  reference: Float32Array,
): Promise<Record<string, boolean>> {
  const checks: Record<string, boolean> = {};
  const same = (a: Float32Array, b: Float32Array) => a.length === b.length && a.every((v, i) => v === b[i]);
  const rejects = async (p: Promise<unknown>, pattern: RegExp) => {
    try {
      await p;
      return false;
    } catch (e) {
      return pattern.test(String(e));
    }
  };
  const settings = base.settings ?? {};

  checks["same seed, same wave"] = same((await tts.generate(base)).wave, reference);
  const random = await tts.generate({ ...base, settings: { ...settings, seed: undefined } });
  checks["a missing seed is drawn and returned"] =
    Number.isInteger(random.seed) && random.seed >= 0 && random.seed <= 0xffffffff;
  const cfg0 = await tts.generate({ ...base, settings: { ...settings, cfgStrength: 0 } });
  const cfgNeg = await tts.generate({ ...base, settings: { ...settings, cfgStrength: -0.5 } });
  const cfgTiny = await tts.generate({ ...base, settings: { ...settings, cfgStrength: 5e-6 } });
  checks["cfg -0.5 and 5e-6 equal cfg 0"] = same(cfgNeg.wave, cfg0.wave) && same(cfgTiny.wave, cfg0.wave);
  checks["cfg 0 differs from cfg 2"] = !same(cfg0.wave, reference);
  checks["empty reference text is refused"] = await rejects(
    tts.generate({ ...base, refText: " \t\n" }),
    /refText is empty/,
  );
  const first = tts.generate(base);
  checks["a second call while one runs is refused"] = await rejects(tts.generate(base), /already running/);
  await first;
  const ctrl = new AbortController();
  let stepsSeen = 0;
  const aborted = tts.generate(base, {
    signal: ctrl.signal,
    onProgress: (p) => {
      stepsSeen++;
      if (p.step === 2) ctrl.abort(new Error("test abort"));
    },
  });
  checks["abort stops the generation"] = await rejects(aborted, /test abort/);
  checks["abort stops after the current step"] = stepsSeen === 3;
  checks["abort frees every tensor"] = tts.liveTensors === 0;
  checks["it generates again after an abort"] = same((await tts.generate(base)).wave, reference);
  return checks;
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
  const settings: Partial<Settings> = {
    nfeStep: s.nfe_step as number,
    cfgStrength: s.cfg_strength as number,
    swaySamplingCoef: s.sway_sampling_coef ?? null,
    speed: s.speed as number,
    crossFadeDuration: s.cross_fade_duration as number,
    fixDuration: s.fix_duration ?? null,
    targetRms: s.target_rms as number,
    seed: 0,
  };
  const tts = await TTS.create({
    dit: {
      textUrl: `${options.ditBase}text.onnx`,
      stepUrl: `${options.ditBase}${options.stepFile ?? "step_mha.onnx"}`,
    },
    vocoder: { url: options.vocoderUrl },
    vocabUrl: "/vocab/vocab.txt",
  });
  const wavBuf = await fetch(`/testdata/webgpu/${options.caseName}/input.wav`).then((r) => r.arrayBuffer());
  const request: GenerateRequest = {
    refAudio: parseWav(wavBuf),
    refText: man.input.ref_text,
    genText: man.input.gen_text,
    settings,
  };
  const device = await ort.env.webgpu.device;
  const gpuErrors: string[] = [];
  device.addEventListener("uncapturederror", (e) => gpuErrors.push((e as GPUUncapturedErrorEvent).error.message));
  const t0 = performance.now();
  const chunkDone: number[] = [];
  let out;
  let api: Record<string, boolean> = {};
  let live: number;
  try {
    out = await tts.generate(request, { onChunk: () => chunkDone.push(Math.round(performance.now() - t0)) });
    if (options.apiChecks) api = await runApiChecks(tts, request, out.wave);
    live = tts.liveTensors;
  } finally {
    await tts.release();
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
    live,
    api,
    gpuErrors,
    wave: Array.from(out.wave),
  };
}

export type FinalRunResult = Awaited<ReturnType<typeof runFinal>>;
