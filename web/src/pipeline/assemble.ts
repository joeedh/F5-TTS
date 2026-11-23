// The cross-fade between chunk waves (plan 04 step 5), as infer_batch_process does it in numpy
// (utils_infer.py:575-611): float32 chunks, float64 fades, and so a float64 result once any fade has run.
import { SAMPLE_RATE } from "../audio/mel";

/** np.linspace(start, stop, num): i * step + start, with the last point set to stop; one point is [start]. */
function linspace(start: number, stop: number, num: number): Float64Array {
  if (num === 1) return Float64Array.of(start);
  const step = (stop - start) / (num - 1);
  const y = Float64Array.from({ length: num }, (_, i) => i * step + start);
  y[num - 1] = stop;
  return y;
}

/** The cross-faded wave in float64, value for value what numpy returns. */
export function crossFadeF64(waves: readonly Float32Array[], seconds: number): Float64Array {
  if (waves.length === 0) throw new Error("crossFade: no waves");
  const concat = (parts: readonly ArrayLike<number>[]) => {
    const out = new Float64Array(parts.reduce((a, p) => a + p.length, 0));
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    return out;
  };
  if (seconds <= 0) return concat(waves);
  let prev: Float64Array = Float64Array.from(waves[0] as Float32Array);
  for (const next of waves.slice(1)) {
    const n = Math.min(Math.trunc(seconds * SAMPLE_RATE), prev.length, next.length);
    if (n <= 0) {
      prev = concat([prev, next]);
      continue;
    }
    const out = linspace(1, 0, n);
    const inn = linspace(0, 1, n);
    const overlap = Float64Array.from(
      { length: n },
      (_, i) => (prev[prev.length - n + i] as number) * (out[i] as number) + (next[i] as number) * (inn[i] as number),
    );
    prev = concat([prev.subarray(0, prev.length - n), overlap, next.subarray(n)]);
  }
  return prev;
}

/** The final wave, cast to float32 once at the end, as the golden final.wav is. */
export function crossFade(waves: readonly Float32Array[], seconds: number): Float32Array {
  return Float32Array.from(crossFadeF64(waves, seconds));
}
