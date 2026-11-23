// The whole generation, as infer_process and infer_batch_process do it (plan 04 steps 5 and 6): reference
// prep, text fix-up and chunking, then per chunk the plan, noise, ODE loop on the GPU, reference restore and
// crop, vocoder and output rescale; then the cross-fade. Chunks run in order (D3), so the noise stream and
// the result are reproducible, unlike upstream's thread pool. Touches no DOM, so it can run in a worker.
import type * as ort from "onnxruntime-web/webgpu";
import * as ortRuntime from "onnxruntime-web/webgpu";
import { logMel, SAMPLE_RATE } from "../audio/mel";
import { rescaleOutput } from "../audio/output";
import { downmixRmsResample, trimAndClip } from "../audio/prepare";
import type { Pcm } from "../audio/pydub";
import type { DiT } from "../dit";
import { addTrailingSpace, chunkText, fixRefText, maxChars, pyStrip } from "../text/strings";
import { textToIds, type TextFrontend } from "../text/tokens";
import type { Vocoder } from "../vocoder";
import { crossFade } from "./assemble";
import { finishChunk, stepCond } from "./cond";
import { TorchGenerator } from "./noise";
import { planChunks, timeGrid } from "./plan";
import { sampleChunk, type TensorLedger } from "./sample";

export interface Settings {
  nfeStep: number;
  cfgStrength: number;
  /** null for no sway. */
  swaySamplingCoef: number | null;
  speed: number;
  /** Seconds between chunks. */
  crossFadeDuration: number;
  /** Total seconds, reference included, or null to work it out from the text. */
  fixDuration: number | null;
  targetRms: number;
  /** A uint32; torch's CPU generator uses only the low 32 bits of a seed (plan 04, D1). */
  seed: number;
}

/** infer_process's defaults (utils_infer.py:48-60). */
export const DEFAULTS: Readonly<Omit<Settings, "seed">> = {
  nfeStep: 32,
  cfgStrength: 2,
  swaySamplingCoef: -1,
  speed: 1,
  crossFadeDuration: 0.15,
  fixDuration: null,
  targetRms: 0.1,
};

export interface Models {
  dit: DiT;
  vocoder: Vocoder;
  vocab: Map<string, number>;
  text: TextFrontend;
  device: GPUDevice;
  ledger: TensorLedger;
}

export interface GenerateInput {
  /** The reference file, decoded at its own rate (parseWav). */
  refAudio: Pcm;
  refText: string;
  genText: string;
}

export interface Progress {
  chunk: number;
  chunks: number;
  step: number;
  steps: number;
}

export interface GenerateHooks {
  /** After each step has finished on the GPU. */
  onProgress?: (p: Progress) => void;
  /** After each chunk's vocoder and rescale, in order: for playback before the last chunk is done. */
  onChunk?: (c: { index: number; wave: Float32Array }) => void;
  /** Checked between steps; on abort the GPU work finishes, every tensor is freed, and the reason is thrown. */
  signal?: AbortSignal;
}

export interface Generated {
  wave: Float32Array;
  sampleRate: number;
  seed: number;
  chunks: { text: string; duration: number; wave: Float32Array }[];
}

export async function generateWith(
  m: Models,
  input: GenerateInput,
  settings: Settings,
  hooks: GenerateHooks = {},
): Promise<Generated> {
  // upstream transcribes an empty reference text with Whisper (utils_infer.py:366); that is plan 06's
  if (!pyStrip(input.refText)) throw new Error("refText is empty; transcribing the reference is not supported");
  const prepared = trimAndClip(input.refAudio);
  const fixed = fixRefText(input.refText);
  const frames = prepared.samples.length / prepared.channels;
  const texts = chunkText(input.genText, maxChars(fixed, frames, prepared.sampleRate, settings.speed));
  const ref = downmixRmsResample(prepared, settings.targetRms);
  const refTextSpaced = addTrailingSpace(fixed);
  const ids: Int32Array[] = [];
  for (const t of texts) ids.push(textToIds(await m.text.tokens(refTextSpaced + t), m.vocab));
  const plans = planChunks({
    refSamples: ref.audio.length,
    refTextSpaced,
    chunks: texts,
    tokenCounts: ids.map((a) => a.length),
    speed: settings.speed,
    fixDuration: settings.fixDuration,
    crossFadeDuration: settings.crossFadeDuration,
  });
  const mel = logMel(ref.audio);
  const grid = timeGrid(settings.nfeStep, settings.swaySamplingCoef);
  const gen = new TorchGenerator(settings.seed);
  const chunks: Generated["chunks"] = [];
  for (const [c, plan] of plans.entries()) {
    const x = await sampleChunk(
      m.dit,
      m.device,
      m.ledger,
      {
        ids: ids[c] as Int32Array,
        noise: gen.randn(plan.duration * 100),
        stepCond: stepCond(mel, plan.duration),
        duration: plan.duration,
        grid,
        cfgStrength: settings.cfgStrength,
      },
      {
        signal: hooks.signal,
        onStep: (step, steps) => hooks.onProgress?.({ chunk: c, chunks: plans.length, step, steps }),
      },
    );
    const out = finishChunk(x, mel, plan.refAudioLen);
    const melT: ort.Tensor = new ortRuntime.Tensor("float32", out, [1, 100, plan.duration - plan.refAudioLen]);
    const { wave } = await m.vocoder.decode(melT);
    const scaled = rescaleOutput(wave, ref, settings.targetRms);
    chunks.push({ text: plan.text, duration: plan.duration, wave: scaled });
    hooks.onChunk?.({ index: c, wave: scaled });
  }
  const wave = crossFade(
    chunks.map((ch) => ch.wave),
    settings.crossFadeDuration,
  );
  return { wave, sampleRate: SAMPLE_RATE, seed: settings.seed, chunks };
}
