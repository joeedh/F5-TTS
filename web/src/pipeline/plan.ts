// The chunk plan (plan 04 step 1): each chunk's speed, duration and reference lengths, and the ODE time grid,
// as infer_batch_process (utils_infer.py:491-519) and CFM.sample (cfm.py:155-226) compute them.
import { HOP, linspaceF32, SAMPLE_RATE } from "../audio/mel";
import { utf8Length } from "../text/strings";

const f32 = Math.fround;

/** The step graph's rotary table covers this many frames (plan 01); upstream clamps at 65536 instead. */
export const MAX_FRAMES = 8192;

export interface ChunkPlanInput {
  /** Samples of the prepared 24 kHz reference (after downmix, RMS raise and resample). */
  refSamples: number;
  /** The reference text after the fix-up and the trailing-space rule (addTrailingSpace). */
  refTextSpaced: string;
  /** The target text's chunks (chunkText). */
  chunks: string[];
  /** Token count of each chunk's joined text, refTextSpaced + chunk: CFM.sample's text_len. */
  tokenCounts: number[];
  speed?: number;
  fixDuration?: number | null;
  crossFadeDuration?: number;
  /** Refuse a chunk longer than this many frames (D7). */
  maxFrames?: number;
}

export interface ChunkPlan {
  text: string;
  localSpeed: number;
  /** Frames for this chunk: reference plus generated, after CFM.sample's minimum-duration rule. */
  duration: number;
  /** samples // 256: where infer_batch_process crops the output. */
  refAudioLen: number;
  /** samples // 256 + 1: the reference mel's frames, which CFM.sample restores after the last step. */
  nRef: number;
}

/** infer_batch_process's fix_duration per chunk (utils_infer.py:491), in the same order of operations. */
function fixDurations(input: ChunkPlanInput, refAudioLen: number, crossFade: number): (number | null)[] {
  const fix = input.fixDuration ?? null;
  if (fix === null || input.chunks.length <= 1) return input.chunks.map(() => fix);
  const refSec = (refAudioLen * HOP) / SAMPLE_RATE;
  const targetTotal = fix - refSec;
  const weights = input.chunks.map(utf8Length);
  const totalW = weights.reduce((a, b) => a + b, 0);
  const allocatable = targetTotal + crossFade * (input.chunks.length - 1);
  return weights.map((w) => refSec + (allocatable * w) / totalW);
}

/** Each chunk's plan. Throws if a chunk would need more than maxFrames frames (D7). */
export function planChunks(input: ChunkPlanInput): ChunkPlan[] {
  const speed = input.speed ?? 1;
  const refAudioLen = Math.floor(input.refSamples / HOP);
  const nRef = refAudioLen + 1;
  const refTextLen = utf8Length(input.refTextSpaced);
  const maxFrames = input.maxFrames ?? MAX_FRAMES;
  if (input.tokenCounts.length !== input.chunks.length) throw new Error("planChunks: one token count per chunk");
  const fixes = fixDurations(input, refAudioLen, input.crossFadeDuration ?? 0.15);
  return input.chunks.map((text, i) => {
    const genTextLen = utf8Length(text);
    const localSpeed = genTextLen < 10 ? speed * 0.3 : speed;
    const fix = fixes[i] ?? null;
    let duration =
      fix !== null
        ? Math.trunc((fix * SAMPLE_RATE) / HOP)
        : refAudioLen + Math.trunc(((refAudioLen / refTextLen) * genTextLen) / localSpeed);
    // CFM.sample: at least the text or the reference plus one frame, and at most 65536
    duration = Math.min(Math.max(Math.max(input.tokenCounts[i] as number, nRef) + 1, duration), 65536);
    if (duration > maxFrames)
      throw new Error(
        `chunk ${i + 1} needs ${duration} frames (${((duration * HOP) / SAMPLE_RATE).toFixed(1)} s), more than ` +
          `${maxFrames}. Add sentence punctuation to split it, or shorten the text or the fixed duration.`,
      );
    return { text, localSpeed, duration, refAudioLen, nRef };
  });
}

/** get_epss_timesteps' table: Empirically Pruned Step Sampling, in 32nds. */
const EPSS: Record<number, number[]> = {
  5: [0, 2, 4, 8, 16, 32],
  6: [0, 2, 4, 6, 8, 16, 32],
  7: [0, 2, 4, 6, 8, 16, 24, 32],
  10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
  12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
  16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
};

/**
 * CFM.sample's time grid, nfeStep + 1 points from 0 to 1: EPSS where the table has nfeStep, else float32
 * linspace; then sway, t + coef * (cos(pi / 2 * t) - 1 + t), unless swayCoef is null. Each float32 op of torch
 * is rounded with fround; the scalars are float32 as torch casts them.
 */
export function timeGrid(nfeStep: number, swayCoef: number | null): Float32Array {
  const table = EPSS[nfeStep];
  const t = table ? table.map((k) => f32((1 / 32) * k)) : Array.from(linspaceF32(0, 1, nfeStep + 1));
  if (swayCoef === null) return Float32Array.from(t);
  const halfPi = f32(Math.PI / 2);
  const coef = f32(swayCoef);
  return Float32Array.from(t, (x) => {
    const c = f32(f32(f32(Math.cos(f32(halfPi * x))) - 1) + x);
    return f32(x + f32(coef * c));
  });
}
