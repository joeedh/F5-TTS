// Conditioning and output per chunk (plan 04 step 3), on the CPU: CFM.sample's step_cond and its final
// where(cond_mask, cond, x), then infer_batch_process's crop at ref_audio_len (cfm.py:155-236).
import type { LogMel } from "../audio/mel";

const N_MELS = 100;

/** step_cond, (1, duration, 100) frame-major: the reference mel transposed, then zeros to `duration` frames. */
export function stepCond(mel: LogMel, duration: number): Float32Array {
  if (duration <= mel.frames)
    throw new Error(`stepCond: duration ${duration} must exceed the ${mel.frames} reference frames`);
  const out = new Float32Array(duration * N_MELS);
  for (let k = 0; k < N_MELS; k++)
    for (let f = 0; f < mel.frames; f++) out[f * N_MELS + k] = mel.data[k * mel.frames + f] as number;
  return out;
}

/**
 * The vocoder's input, (1, 100, duration - refAudioLen) mel-major, from the last step's x (1, duration, 100):
 * the first n_ref = mel.frames frames are the reference mel again, and the crop starts at refAudioLen, one
 * frame before n_ref, so the output's frame 0 is the reference's last frame.
 */
export function finishChunk(x: Float32Array, mel: LogMel, refAudioLen: number): Float32Array {
  const duration = x.length / N_MELS;
  if (!Number.isInteger(duration) || duration <= mel.frames)
    throw new Error("finishChunk: x is not (1, duration, 100)");
  if (refAudioLen < 0 || refAudioLen >= mel.frames) throw new Error("finishChunk: refAudioLen must be under n_ref");
  const frames = duration - refAudioLen;
  const out = new Float32Array(N_MELS * frames);
  for (let f = 0; f < frames; f++) {
    const src = f + refAudioLen;
    for (let k = 0; k < N_MELS; k++)
      out[k * frames + f] = (src < mel.frames ? mel.data[k * mel.frames + src] : x[src * N_MELS + k]) as number;
  }
  return out;
}
