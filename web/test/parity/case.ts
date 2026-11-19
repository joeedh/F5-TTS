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

/** `read` is rooted at testdata/webgpu/; the case lives in its subfolder. */
export async function loadCase(read: Reader, caseName: string): Promise<GoldenCase> {
  const inCase: Reader = (path) => read(`${caseName}/${path}`);
  return new GoldenCase(await readJson<Manifest>(inCase, "manifest.json"), inCase);
}

export function loadTolerances(read: Reader): Promise<Tolerances> {
  return readJson<Tolerances>(read, "tolerances.json");
}
