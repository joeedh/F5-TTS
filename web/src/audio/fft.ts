// In-place radix-2 complex FFT in float64, for the mel (plan 02 step 1) and the inverse STFT (step 4).
// Sizes are powers of two; twiddles and the bit-reversal table are built once per size.

export class FFT {
  readonly size: number;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly rev: Uint32Array;

  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) throw new Error(`FFT size ${size} is not a power of two`);
    this.size = size;
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((2 * Math.PI * i) / size);
    }
    const bits = Math.log2(size);
    this.rev = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** Forward transform, X[k] = sum_j x[j] e^(-2 pi i jk / N), in place. */
  forward(re: Float64Array, im: Float64Array): void {
    this.transform(re, im, -1);
  }

  /** Inverse transform without the 1/N factor, in place. */
  inverseUnscaled(re: Float64Array, im: Float64Array): void {
    this.transform(re, im, 1);
  }

  private transform(re: Float64Array, im: Float64Array, sign: 1 | -1): void {
    const n = this.size;
    if (re.length !== n || im.length !== n) throw new Error(`FFT of size ${n} given ${re.length} values`);
    for (let i = 0; i < n; i++) {
      const j = this.rev[i] as number;
      if (j > i) {
        const tr = re[i] as number;
        re[i] = re[j] as number;
        re[j] = tr;
        const ti = im[i] as number;
        im[i] = im[j] as number;
        im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len *= 2) {
      const half = len / 2;
      const step = n / len;
      for (let start = 0; start < n; start += len) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step] as number;
          const wi = sign * (this.sin[k * step] as number);
          const a = start + k;
          const b = a + half;
          const br = re[b] as number;
          const bi = im[b] as number;
          const tr = br * wr - bi * wi;
          const ti = br * wi + bi * wr;
          re[b] = (re[a] as number) - tr;
          im[b] = (im[a] as number) - ti;
          re[a] = (re[a] as number) + tr;
          im[a] = (im[a] as number) + ti;
        }
      }
    }
  }
}
