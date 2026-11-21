// The Vocos log-mel spectrogram (plan 02 step 1), as upstream's get_vocos_mel_spectrogram computes it
// with torchaudio's MelSpectrogram: 24 kHz, n_fft and window 1024, hop 256, 100 mels, magnitude (power 1),
// center with reflect padding, periodic Hann window, HTK mel scale from 0 to 12 kHz, no filterbank
// normalisation, then log(max(mel, 1e-5)). Computed in float64 and returned as float32.
import { FFT } from "./fft";

export const SAMPLE_RATE = 24000;
export const N_FFT = 1024;
export const HOP = 256;
export const N_MELS = 100;
export const N_FREQS = N_FFT / 2 + 1;

/** Periodic Hann window of `n` points, as `torch.hann_window(n)`. */
export function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

const f32 = Math.fround;

/** `torch.linspace(start, end, steps)` in float32: the CPU kernel fills each half from its own end. */
function linspaceF32(start: number, end: number, steps: number): Float64Array {
  const s = f32(start);
  const e = f32(end);
  const step = f32((e - s) / (steps - 1));
  const half = Math.floor(steps / 2);
  return Float64Array.from({ length: steps }, (_, i) =>
    i < half ? f32(s + f32(step * i)) : f32(e - f32(step * (steps - i - 1))),
  );
}

/**
 * torchaudio's `melscale_fbanks(513, 0, 12000, 100, 24000, norm=None, mel_scale="htk")`, row-major
 * (frequency, mel). torchaudio computes it in float32, and each step here is rounded the same way: the
 * float64 result is up to 9e-6 away, which moves the mel 10 times further from the golden one.
 */
export function melFilterbank(): Float64Array {
  const allFreqs = linspaceF32(0, SAMPLE_RATE / 2, N_FREQS);
  const mMax = 2595 * Math.log10(1 + SAMPLE_RATE / 2 / 700); // a Python float, as in _hz_to_mel
  const fPts = linspaceF32(0, mMax, N_MELS + 2).map((m) => f32(700 * f32(f32(10 ** f32(m / 2595)) - 1)));
  const fDiff = Float64Array.from({ length: N_MELS + 1 }, (_, i) => f32((fPts[i + 1] as number) - (fPts[i] as number)));
  const fb = new Float64Array(N_FREQS * N_MELS);
  for (let f = 0; f < N_FREQS; f++) {
    const hz = allFreqs[f] as number;
    for (let m = 0; m < N_MELS; m++) {
      const down = f32(-f32((fPts[m] as number) - hz) / (fDiff[m] as number));
      const up = f32(f32((fPts[m + 2] as number) - hz) / (fDiff[m + 1] as number));
      fb[f * N_MELS + m] = Math.max(0, Math.min(down, up));
    }
  }
  return fb;
}

export interface LogMel {
  /** (N_MELS, frames), row-major: the golden `mel.ref` (1, 100, n) without the batch axis. */
  data: Float32Array;
  frames: number;
}

let cache: { fft: FFT; window: Float64Array; fb: Float64Array } | undefined;

/** Log-mel of 24 kHz mono audio. Frames: samples / 256 + 1, rounded down. */
export function logMel(audio: Float32Array | Float64Array): LogMel {
  const pad = N_FFT / 2;
  const len = audio.length;
  // torch's reflect padding needs more samples than the pad
  if (len <= pad) throw new Error(`logMel needs more than ${pad} samples, got ${len}`);
  cache ??= { fft: new FFT(N_FFT), window: hannWindow(N_FFT), fb: melFilterbank() };
  const { fft, window, fb } = cache;
  const at = (j: number) => {
    // index into the reflect-padded signal
    const k = j - pad;
    if (k < 0) return audio[-k] as number;
    if (k >= len) return audio[2 * (len - 1) - k] as number;
    return audio[k] as number;
  };
  const frames = Math.floor(len / HOP) + 1;
  const out = new Float32Array(N_MELS * frames);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const mag = new Float64Array(N_FREQS);
  for (let t = 0; t < frames; t++) {
    const start = t * HOP;
    for (let i = 0; i < N_FFT; i++) {
      re[i] = at(start + i) * (window[i] as number);
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let f = 0; f < N_FREQS; f++) mag[f] = Math.hypot(re[f] as number, im[f] as number);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      for (let f = 0; f < N_FREQS; f++) s += (fb[f * N_MELS + m] as number) * (mag[f] as number);
      out[m * frames + t] = Math.log(Math.max(s, 1e-5));
    }
  }
  return { data: out, frames };
}
