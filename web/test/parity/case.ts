// Loads a golden case (manifest plus lazily loaded tensors) and the tolerance table.
// Reads from the filesystem in Node and over HTTP in the browser; see the `Reader` argument.
import { parseNpy, toArrayBuffer, type Tensor } from "./npy";

export interface TensorEntry {
  name: string;
  file: string;
  stage: string;
  shape: number[];
  axes: string;
  dtype: string;
  sha256: string;
}

export interface Manifest {
  case: string;
  dtype: string;
  settings: Record<string, unknown>;
  audio: { rms: number; raised: boolean };
  chunks: Record<string, unknown>[];
  tensors: TensorEntry[];
  [key: string]: unknown;
}

export interface CheckTolerance {
  note: string;
  rel_l2?: number;
  max_abs?: number;
  tiny?: { rel_l2?: number; max_abs?: number }; // used only for the tiny case
  [key: string]: unknown;
}

export interface Tolerances {
  checks: Record<string, CheckTolerance>;
  [key: string]: unknown;
}

/** Reads a file relative to some base (a directory in Node, a URL in the browser). */
export type Reader = (path: string) => Promise<ArrayBuffer>;

export function httpReader(baseUrl: string): Reader {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return async (path) => {
    const res = await fetch(base + path);
    if (!res.ok) throw new Error(`${base + path}: HTTP ${res.status}`);
    return res.arrayBuffer();
  };
}

export async function fsReader(baseDir: string): Promise<Reader> {
  const { readFile } = await import("node:fs/promises");
  const { join } = await import("node:path");
  return async (path) => toArrayBuffer(await readFile(join(baseDir, path)));
}

async function readJson<T>(read: Reader, path: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await read(path))) as T;
}

export class GoldenCase {
  private readonly cache = new Map<string, Promise<Tensor>>();
  readonly entries: Map<string, TensorEntry>;

  constructor(
    readonly manifest: Manifest,
    private readonly read: Reader,
  ) {
    this.entries = new Map(manifest.tensors.map((t) => [t.name, t]));
  }

  get name(): string {
    return this.manifest.case;
  }

  entry(name: string): TensorEntry {
    const e = this.entries.get(name);
    if (!e) throw new Error(`${this.name}: no golden tensor ${name}`);
    return e;
  }

  tensor(name: string): Promise<Tensor> {
    let t = this.cache.get(name);
    if (!t) {
      t = this.read(this.entry(name).file).then(parseNpy);
      this.cache.set(name, t);
    }
    return t;
  }
}

/**
 * The golden value of the Euler update after step `s` of chunk `c`, and the first frame to compare.
 * Before the last step it is the next step's x. After it, it is mel_out on the generated frames:
 * mel_out starts at ref_audio_len = samples / 256, but cond covers n_ref = that + 1 frames, so its
 * first frame is the reference's last (cfm.py:235). Python twin: parity.x_next_golden.
 */
export async function xNextGolden(golden: GoldenCase, c: number, s: number): Promise<{ start: number; ref: Tensor }> {
  const nfe = golden.manifest.settings.nfe_step as number;
  if (s + 1 < nfe) return { start: 0, ref: await golden.tensor(`c${c}.dit.s${String(s + 1).padStart(2, "0")}.x`) };
  const refLen = (golden.manifest.chunks[c] as { ref_audio_len: number }).ref_audio_len;
  const nRef = (golden.manifest.mel as { n_ref: number }).n_ref;
  const mel = await golden.tensor(`c${c}.mel_out`); // (1, 100, frames)
  const [, d = 0, frames = 0] = mel.shape;
  const off = nRef - refLen;
  const data = new Float32Array((frames - off) * d);
  for (let f = off; f < frames; f++)
    for (let k = 0; k < d; k++) data[(f - off) * d + k] = mel.data[k * frames + f] as number;
  return { start: nRef, ref: { dtype: "float32", shape: [1, frames - off, d], data } };
}

/** `read` is rooted at testdata/webgpu/; the case lives in its subfolder. */
export async function loadCase(read: Reader, caseName: string): Promise<GoldenCase> {
  const inCase: Reader = (path) => read(`${caseName}/${path}`);
  return new GoldenCase(await readJson<Manifest>(inCase, "manifest.json"), inCase);
}

export function loadTolerances(read: Reader): Promise<Tolerances> {
  return readJson<Tolerances>(read, "tolerances.json");
}
