// torchaudio's default resampler (transforms.Resample: sinc_interp_hann, lowpass_filter_width 6,
// rolloff 0.99), as infer_batch_process uses it (plan 02, D4). The kernel follows
// _get_sinc_resample_kernel's dtypes: the phase offsets are float32, the rest float64, and the taps are
// rounded to float32. The convolution runs in float64.

const WIDTH = 6;
const ROLLOFF = 0.99;

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

interface Kernel {
  orig: number;
  next: number;
  width: number;
  /** (next, 2 * width + orig), row-major */
  taps: Float64Array;
}

const kernels = new Map<string, Kernel>();

function kernel(origFreq: number, newFreq: number): Kernel {
  const key = `${origFreq}:${newFreq}`;
  const cached = kernels.get(key);
  if (cached) return cached;
  const g = gcd(origFreq, newFreq);
  const orig = origFreq / g;
  const next = newFreq / g;
  const base = Math.min(orig, next) * ROLLOFF;
  const width = Math.ceil((WIDTH * orig) / base);
  const k = 2 * width + orig;
  const taps = new Float64Array(next * k);
  for (let j = 0; j < next; j++) {
    const phase = Math.fround(-j / next); // torch.arange(0, -new, -1) / new, in float32
    for (let i = 0; i < k; i++) {
      let t = (phase + (i - width) / orig) * base;
      t = Math.min(Math.max(t, -WIDTH), WIDTH);
      const w = Math.cos((t * Math.PI) / WIDTH / 2) ** 2;
      t *= Math.PI;
      const sinc = t === 0 ? 1 : Math.sin(t) / t;
      taps[j * k + i] = Math.fround(sinc * (w * (base / orig)));
    }
  }
  const out = { orig, next, width, taps };
  kernels.set(key, out);
  return out;
}

/** Resamples one channel from `origFreq` to `newFreq` Hz. Output length: ceil(new * length / orig). */
export function resample(x: Float32Array, origFreq: number, newFreq: number): Float32Array {
  if (origFreq === newFreq) return x;
  const { orig, next, width, taps } = kernel(origFreq, newFreq);
  const k = 2 * width + orig;
  const len = x.length;
  const outLen = Math.ceil((next * len) / orig);
  const out = new Float32Array(outLen);
  // the input is zero-padded by `width` on the left and `width + orig` on the right
  for (let n = 0; n < outLen; n++) {
    const frame = Math.floor(n / next);
    const j = n - frame * next;
    const start = frame * orig - width;
    let s = 0;
    const lo = Math.max(0, -start);
    const hi = Math.min(k, len - start);
    for (let i = lo; i < hi; i++) s += (taps[j * k + i] as number) * (x[start + i] as number);
    out[n] = s;
  }
  return out;
}
