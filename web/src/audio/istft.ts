// The Vocos ISTFT head after its linear layer (plan 02 step 4, D2 option A): log magnitude to magnitude
// clipped at 100, phase to cos and sin, then torch.istft(n_fft 1024, hop 256, periodic Hann, center=True).
// Runs in float64 on the CPU; the input is the vocoder graph's head_out, read back from the GPU.
import { FFT } from "./fft";
import { HOP, hannWindow, N_FFT } from "./mel";

const BINS = N_FFT / 2 + 1;

let cache: { fft: FFT; window: Float64Array } | undefined;

/** One frame's inverse real FFT, as torch.fft.irfft(n=1024): the imaginary parts of bins 0 and 512 are ignored. */
export function irfftFrame(re: ArrayLike<number>, im: ArrayLike<number>, out: Float64Array): void {
  cache ??= { fft: new FFT(N_FFT), window: hannWindow(N_FFT) };
  const xr = new Float64Array(N_FFT);
  const xi = new Float64Array(N_FFT);
  xr[0] = re[0] as number;
  xr[N_FFT / 2] = re[N_FFT / 2] as number;
  for (let k = 1; k < N_FFT / 2; k++) {
    xr[k] = re[k] as number;
    xi[k] = im[k] as number;
    xr[N_FFT - k] = re[k] as number;
    xi[N_FFT - k] = -(im[k] as number);
  }
  cache.fft.inverseUnscaled(xr, xi);
  for (let i = 0; i < N_FFT; i++) out[i] = (xr[i] as number) / N_FFT;
}

/**
 * Audio from the head's linear output, (frames, 1026) row-major: 513 log magnitudes, then 513 phases.
 * Output length (frames - 1) * 256.
 */
export function istft(headOut: Float32Array, frames: number): Float32Array {
  if (headOut.length !== frames * 2 * BINS) throw new Error(`istft: ${headOut.length} values for ${frames} frames`);
  cache ??= { fft: new FFT(N_FFT), window: hannWindow(N_FFT) };
  const { window } = cache;
  const total = N_FFT + HOP * (frames - 1);
  const signal = new Float64Array(total);
  const envelope = new Float64Array(total);
  const re = new Float64Array(BINS);
  const im = new Float64Array(BINS);
  const frame = new Float64Array(N_FFT);
  for (let t = 0; t < frames; t++) {
    const row = t * 2 * BINS;
    for (let k = 0; k < BINS; k++) {
      const mag = Math.min(Math.exp(headOut[row + k] as number), 100);
      const phase = headOut[row + BINS + k] as number;
      re[k] = mag * Math.cos(phase);
      im[k] = mag * Math.sin(phase);
    }
    irfftFrame(re, im, frame);
    const start = t * HOP;
    for (let i = 0; i < N_FFT; i++) {
      const w = window[i] as number;
      signal[start + i] = (signal[start + i] as number) + (frame[i] as number) * w;
      envelope[start + i] = (envelope[start + i] as number) + w * w;
    }
  }
  // center=True: drop n_fft / 2 at both ends, then divide by the summed squared window
  const pad = N_FFT / 2;
  const out = new Float32Array(total - 2 * pad);
  for (let i = 0; i < out.length; i++) out[i] = (signal[pad + i] as number) / (envelope[pad + i] as number);
  return out;
}
