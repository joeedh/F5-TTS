// The parts of pydub 0.25.1 (and audioop) that upstream's reference preparation uses, reproduced exactly
// for 16-bit PCM (plan 02, D5). Each function names the pydub one it follows; the rules that exact parity
// depends on are listed in docs/plans/webgpu-02-mel-and-vocoder.md, "pydub details".

/** Interleaved 16-bit PCM. */
export interface Pcm {
  samples: Int16Array;
  channels: number;
  sampleRate: number;
}

const MAX_AMPLITUDE = 32768; // max_possible_amplitude for sample width 2

/** Python's round() on a float: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** audioop.ratecv(data, 2, channels, inRate, outRate, None) with the default weights (1, 0). */
export function ratecv(samples: Int16Array, channels: number, inRate: number, outRate: number): Int16Array {
  const g = gcd(inRate, outRate);
  const inR = inRate / g;
  const outR = outRate / g;
  let frames = samples.length / channels;
  const out: number[] = [];
  const prev = new Array<number>(channels).fill(0);
  const cur = new Array<number>(channels).fill(0);
  let d = -outR;
  let p = 0;
  for (;;) {
    while (d < 0) {
      if (frames === 0) return Int16Array.from(out);
      for (let c = 0; c < channels; c++) {
        prev[c] = cur[c] as number;
        cur[c] = (samples[p++] as number) * 65536; // GETSAMPLE32: the sample shifted to 32 bits
      }
      frames--;
      d += outR;
    }
    while (d >= 0) {
      for (let c = 0; c < channels; c++) {
        const v = Math.trunc(((prev[c] as number) * d + (cur[c] as number) * (outR - d)) / outR);
        out.push(v >> 16); // SETSAMPLE32: arithmetic shift back to 16 bits
      }
      d -= inR;
    }
  }
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** An AudioSegment of 16-bit samples. Immutable, like pydub's. */
export class Segment {
  constructor(
    readonly samples: Int16Array,
    readonly channels: number,
    readonly rate: number,
  ) {}

  static fromPcm(pcm: Pcm): Segment {
    return new Segment(pcm.samples, pcm.channels, pcm.sampleRate);
  }

  /** AudioSegment.silent(duration, frame_rate=11025): mono zeros. */
  static silent(durationMs: number, rate = 11025): Segment {
    return new Segment(new Int16Array(Math.trunc(rate * (durationMs / 1000))), 1, rate);
  }

  get frames(): number {
    return this.samples.length / this.channels;
  }

  /** len(segment): milliseconds, rounded half to even. */
  get lengthMs(): number {
    return roundHalfEven(1000 * (this.frames / this.rate));
  }

  /** _parse_position: milliseconds to a frame index, through a truncated double product. */
  private position(ms: number): number {
    const v = ms < 0 ? this.lengthMs - Math.abs(ms) : ms;
    return Math.trunc(v * (this.rate / 1000));
  }

  /**
   * segment[start:end] in milliseconds (__getitem__): both ends are capped at len(segment), and a slice
   * that the rounded length makes overshoot the data is padded with zero frames, at most 2 ms of them.
   * Returns the frame range and the padding, so callers can measure a slice without copying it.
   */
  sliceFrames(startMs = 0, endMs?: number): { start: number; end: number; pad: number } {
    const len = this.lengthMs;
    const start = this.position(Math.min(startMs, len));
    const end = this.position(Math.min(endMs ?? len, len));
    const lo = Math.min(start, this.frames);
    const hi = Math.max(lo, Math.min(end, this.frames));
    const missing = end - start - (hi - lo);
    // pydub builds the padding from the slice's first frame, so an empty slice gets none
    if (missing <= 0 || hi === lo) return { start: lo, end: hi, pad: 0 };
    if (missing > 2 * (this.rate / 1000)) throw new Error(`slice overshoots by ${missing} frames (pydub raises)`);
    return { start: lo, end: hi, pad: missing };
  }

  slice(startMs = 0, endMs?: number): Segment {
    const { start, end, pad } = this.sliceFrames(startMs, endMs);
    const ch = this.channels;
    const out = new Int16Array((end - start + pad) * ch);
    out.set(this.samples.subarray(start * ch, end * ch));
    return new Segment(out, ch, this.rate);
  }

  /** audioop.rms: the square root of the mean square over all channels' samples, truncated. */
  get rms(): number {
    let sum = 0;
    for (const v of this.samples) sum += v * v;
    return rmsOf(sum, this.samples.length);
  }

  /** segment.dBFS: 20 * log(rms / 32768, 10), or -Infinity for silence. */
  get dBFS(): number {
    return ratioToDb(this.rms / MAX_AMPLITUDE);
  }

  reverse(): Segment {
    return new Segment(this.samples.slice().reverse(), this.channels, this.rate);
  }

  /** segment + other (append with no crossfade): both are converted to the larger channel count and rate. */
  append(other: Segment): Segment {
    const channels = Math.max(this.channels, other.channels);
    const rate = Math.max(this.rate, other.rate);
    const a = this.convert(channels, rate);
    const b = other.convert(channels, rate);
    const out = new Int16Array(a.samples.length + b.samples.length);
    out.set(a.samples);
    out.set(b.samples, a.samples.length);
    return new Segment(out, channels, rate);
  }

  /** set_channels then set_frame_rate, as _sync does. Only mono to more channels is supported. */
  private convert(channels: number, rate: number): Segment {
    let s: Segment = this;
    if (channels !== s.channels) {
      if (s.channels !== 1) throw new Error(`cannot convert ${s.channels} channels to ${channels}`);
      const out = new Int16Array(s.frames * channels);
      for (let i = 0; i < s.frames; i++) out.fill(s.samples[i] as number, i * channels, (i + 1) * channels);
      s = new Segment(out, channels, s.rate);
    }
    if (rate !== s.rate && s.samples.length > 0)
      s = new Segment(ratecv(s.samples, s.channels, s.rate, rate), s.channels, rate);
    else if (rate !== s.rate) s = new Segment(s.samples, s.channels, rate);
    return s;
  }

  toPcm(): Pcm {
    return { samples: this.samples, channels: this.channels, sampleRate: this.rate };
  }
}

function rmsOf(sumSquares: number, count: number): number {
  return count === 0 ? 0 : Math.trunc(Math.sqrt(sumSquares / count));
}

/** pydub's ratio_to_db: 20 * log(ratio, 10), which Python computes as log(ratio) / log(10). */
function ratioToDb(ratio: number): number {
  return ratio === 0 ? -Infinity : 20 * (Math.log(ratio) / Math.log(10));
}

/**
 * Root-mean-square of many slices of one segment. Sums of squared int16 samples are exact in float64
 * up to 2^53, so a prefix sum gives the same integer rms as audioop in linear time. Past that it
 * falls back to summing each slice, as audioop does.
 */
class RmsIndex {
  private readonly prefix: Float64Array | null;

  constructor(private readonly seg: Segment) {
    const prefix = new Float64Array(seg.samples.length + 1);
    let sum = 0;
    for (let i = 0; i < seg.samples.length; i++) {
      const v = seg.samples[i] as number;
      sum += v * v;
      prefix[i + 1] = sum;
    }
    this.prefix = sum < 2 ** 53 ? prefix : null;
  }

  rms(startMs: number, endMs: number): number {
    const { start, end, pad } = this.seg.sliceFrames(startMs, endMs);
    const ch = this.seg.channels;
    let sum: number;
    if (this.prefix) sum = (this.prefix[end * ch] as number) - (this.prefix[start * ch] as number);
    else {
      sum = 0;
      for (let i = start * ch; i < end * ch; i++) sum += (this.seg.samples[i] as number) ** 2;
    }
    return rmsOf(sum, (end - start + pad) * ch);
  }
}

/** silence.detect_silence: silent [start, end] ranges in ms. */
export function detectSilence(
  seg: Segment,
  minSilenceLen: number,
  silenceThresh: number,
  seekStep: number,
): number[][] {
  const segLen = seg.lengthMs;
  if (segLen < minSilenceLen) return [];
  const thresh = 10 ** (silenceThresh / 20) * MAX_AMPLITUDE; // db_to_float, compared linearly
  const lastStart = segLen - minSilenceLen;
  const starts: number[] = [];
  for (let i = 0; i <= lastStart; i += seekStep) starts.push(i);
  if (lastStart % seekStep) starts.push(lastStart);
  const index = new RmsIndex(seg);
  const silent = starts.filter((i) => index.rms(i, i + minSilenceLen) <= thresh);
  if (silent.length === 0) return [];
  const ranges: number[][] = [];
  let prev = silent[0] as number;
  let rangeStart = prev;
  for (const s of silent.slice(1)) {
    const continuous = s === prev + seekStep;
    const hasGap = s > prev + minSilenceLen;
    if (!continuous && hasGap) {
      ranges.push([rangeStart, prev + minSilenceLen]);
      rangeStart = s;
    }
    prev = s;
  }
  ranges.push([rangeStart, prev + minSilenceLen]);
  return ranges;
}

/** silence.detect_nonsilent. */
export function detectNonsilent(
  seg: Segment,
  minSilenceLen: number,
  silenceThresh: number,
  seekStep: number,
): number[][] {
  const silent = detectSilence(seg, minSilenceLen, silenceThresh, seekStep);
  const len = seg.lengthMs;
  if (silent.length === 0) return [[0, len]];
  if (silent[0]?.[0] === 0 && silent[0][1] === len) return [];
  let prevEnd = 0;
  const out: number[][] = [];
  for (const [s, e] of silent) {
    out.push([prevEnd, s as number]);
    prevEnd = e as number;
  }
  if ((silent[silent.length - 1]?.[1] as number) !== len) out.push([prevEnd, len]);
  if (out[0]?.[0] === 0 && out[0][1] === 0) out.shift();
  return out;
}

/** silence.split_on_silence with a numeric keep_silence. */
export function splitOnSilence(
  seg: Segment,
  minSilenceLen: number,
  silenceThresh: number,
  keepSilence: number,
  seekStep: number,
): Segment[] {
  const ranges = detectNonsilent(seg, minSilenceLen, silenceThresh, seekStep).map(([s, e]) => [
    (s as number) - keepSilence,
    (e as number) + keepSilence,
  ]);
  for (let i = 0; i + 1 < ranges.length; i++) {
    const a = ranges[i] as number[];
    const b = ranges[i + 1] as number[];
    if ((b[0] as number) < (a[1] as number)) {
      a[1] = Math.floor(((a[1] as number) + (b[0] as number)) / 2);
      b[0] = a[1];
    }
  }
  const len = seg.lengthMs;
  return ranges.map(([s, e]) => seg.slice(Math.max(s as number, 0), Math.min(e as number, len)));
}

/** silence.detect_leading_silence: the ms where leading silence ends. */
export function detectLeadingSilence(seg: Segment, silenceThreshold: number, chunkSize = 10): number {
  const len = seg.lengthMs;
  let trim = 0;
  while (trim < len && seg.slice(trim, trim + chunkSize).dBFS < silenceThreshold) trim += chunkSize;
  return Math.min(trim, len);
}
