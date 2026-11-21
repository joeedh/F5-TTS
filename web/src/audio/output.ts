// The output rescale in infer_batch_process (plan 02 step 4): a reference that was raised to RMS 0.1
// gets its generated audio scaled back down, as `generated_wave * rms / target_rms` in float32.
import { TARGET_RMS, type Reference } from "./prepare";

export function rescaleOutput(wave: Float32Array, ref: Pick<Reference, "rms" | "raised">): Float32Array {
  if (!ref.raised) return wave;
  const rms = Math.fround(ref.rms);
  const target = Math.fround(TARGET_RMS);
  return wave.map((v) => Math.fround(Math.fround(v * rms) / target));
}
