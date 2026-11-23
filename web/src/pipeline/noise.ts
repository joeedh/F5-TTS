// torch.manual_seed and torch.randn on the CPU (plan 04 step 2, D1), so a seed gives the same noise as Python.
// Ported from torch 2.15's headers: ATen/core/MT19937RNGEngine.h, TransformationHelper.h (uniform_real),
// ATen/native/cpu/DistributionTemplates.h (normal_fill) and avx_mathfun.h. Its AVX2 Box-Muller is float32
// throughout, so each operation is rounded with fround; the golden data was made on an AVX2 machine.

const f32 = Math.fround;
const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;

/** at::mt19937: the 32-bit Mersenne Twister, seeded as torch seeds it (the low 32 bits of the seed). */
export class TorchGenerator {
  private readonly state = new Uint32Array(N);
  private left = 1;
  private next = 0;

  constructor(seed: number) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("seed must be a uint32");
    this.state[0] = seed >>> 0;
    for (let j = 1; j < N; j++) {
      const prev = this.state[j - 1] as number;
      this.state[j] = (Math.imul(1812433253, (prev ^ (prev >>> 30)) >>> 0) + j) >>> 0;
    }
  }

  private nextState(): void {
    const s = this.state;
    const twist = (u: number, v: number) => (((u & 0x80000000) | (v & 0x7fffffff)) >>> 1) ^ (v & 1 ? MATRIX_A : 0);
    let p = 0;
    for (; p < N - M; p++) s[p] = ((s[p + M] as number) ^ twist(s[p] as number, s[p + 1] as number)) >>> 0;
    for (; p < N - 1; p++) s[p] = ((s[p + M - N] as number) ^ twist(s[p] as number, s[p + 1] as number)) >>> 0;
    s[p] = ((s[p + M - N] as number) ^ twist(s[p] as number, s[0] as number)) >>> 0;
    this.left = N;
    this.next = 0;
  }

  /** One 32-bit output, as CPUGeneratorImpl::random(). */
  random(): number {
    if (--this.left === 0) this.nextState();
    let y = this.state[this.next++] as number;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** uniform_real_distribution<float>(0, 1): 24 bits, so exact in float32. */
  uniform(): number {
    return (this.random() & 0xffffff) / 0x1000000;
  }

  /** torch.randn(size) on the CPU, size 16 or more (normal_fill); smaller sizes take another path in torch. */
  randn(size: number): Float32Array {
    if (size < 16) throw new Error("randn: sizes under 16 use torch's serial path, which is not ported");
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = this.uniform();
    for (let i = 0; i < size - 15; i += 16) normalFill16(data, i);
    if (size % 16 !== 0) {
      // torch redraws the last 16 values, overlapping the last full block
      for (let i = size - 16; i < size; i++) data[i] = this.uniform();
      normalFill16(data, size - 16);
    }
    return data;
  }
}

const bitsBuf = new DataView(new ArrayBuffer(4));
const bitsOf = (x: number) => (bitsBuf.setFloat32(0, x), bitsBuf.getUint32(0));
const fromBits = (b: number) => (bitsBuf.setUint32(0, b >>> 0), bitsBuf.getFloat32(0));

// avx_mathfun.h's constants, as the float32 values _PS256_CONST makes of them
const SQRTHF = f32(0.707106781186547524);
const LOG_P = [
  7.0376836292e-2, -1.151461031e-1, 1.167699874e-1, -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1,
  2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1,
].map(f32);
const LOG_Q1 = f32(-2.1219444e-4);
const LOG_Q2 = f32(0.693359375);
const MIN_NORM_POS = fromBits(0x00800000);
const DP1 = f32(-0.78515625);
const DP2 = f32(-2.4187564849853515625e-4);
const DP3 = f32(-3.77489497744594108e-8);
const SINCOF = [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1].map(f32);
const COSCOF = [2.443315711809948e-5, -1.388731625493765e-3, 4.166664568298827e-2].map(f32);
const FOPI = f32(1.27323954473516);
const TWO_PI = f32(2 * Math.PI);

/** avx_mathfun's log256_ps for one lane, x > 0. */
function logPs(x0: number): number {
  const x1 = Math.max(x0, MIN_NORM_POS);
  const bits = bitsOf(x1);
  let x = fromBits((bits & ~0x7f800000) | bitsOf(0.5));
  let e = f32((bits >>> 23) - 0x7f + 1);
  const mask = x < SQRTHF;
  const tmp0 = mask ? x : 0;
  x = f32(x - 1);
  e = f32(e - (mask ? 1 : 0));
  x = f32(x + tmp0);
  const z = f32(x * x);
  let y = LOG_P[0] as number;
  for (let k = 1; k < 9; k++) y = f32(f32(y * x) + (LOG_P[k] as number));
  y = f32(y * x);
  y = f32(y * z);
  y = f32(y + f32(e * LOG_Q1));
  y = f32(y - f32(z * 0.5));
  x = f32(x + y);
  return f32(x + f32(e * LOG_Q2));
}

/** avx_mathfun's sincos256_ps for one lane, x >= 0 (theta = 2 pi u). Returns [sin, cos]. */
function sinCosPs(xIn: number): [number, number] {
  let x = Math.abs(xIn);
  let j = Math.trunc(f32(x * FOPI)) | 0;
  j = (j + 1) & ~1;
  const y0 = j;
  const swapSin = (j & 4) !== 0;
  const polyMask = (j & 2) === 0;
  x = f32(x + f32(y0 * DP1));
  x = f32(x + f32(y0 * DP2));
  x = f32(x + f32(y0 * DP3));
  const negCos = (~(j - 2) & 4) !== 0;
  const negSin = xIn < 0 !== swapSin;
  const z = f32(x * x);
  let y = COSCOF[0] as number;
  y = f32(y * z);
  y = f32(y + (COSCOF[1] as number));
  y = f32(y * z);
  y = f32(y + (COSCOF[2] as number));
  y = f32(y * z);
  y = f32(y * z);
  y = f32(y - f32(z * 0.5));
  y = f32(y + 1);
  let y2 = SINCOF[0] as number;
  y2 = f32(y2 * z);
  y2 = f32(y2 + (SINCOF[1] as number));
  y2 = f32(y2 * z);
  y2 = f32(y2 + (SINCOF[2] as number));
  y2 = f32(y2 * z);
  y2 = f32(y2 * x);
  y2 = f32(y2 + x);
  const sin = polyMask ? y2 : y;
  const cos = polyMask ? y : y2;
  return [negSin ? -sin : sin, negCos ? -cos : cos];
}

/** NormalFill16<float, true>, the AVX2 Box-Muller, on data[i..i+16), mean 0 and std 1. */
function normalFill16(data: Float32Array, i: number): void {
  for (let j = 0; j < 8; j++) {
    const u1 = f32(1 - (data[i + j] as number));
    const u2 = data[i + j + 8] as number;
    const radius = f32(Math.sqrt(f32(-2 * logPs(u1))));
    const [sin, cos] = sinCosPs(f32(TWO_PI * u2));
    data[i + j] = f32(radius * cos);
    data[i + j + 8] = f32(radius * sin);
  }
}
