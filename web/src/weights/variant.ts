// A weight variant from tools/webgpu/convert_precision.py (plan 05, D1 and D4): three graphs whose external
// data is split into shards on tensor boundaries. A shard is stored compactly (fp16, or int8 with block
// scales) and widened here into the fp32 layout the graph's offsets describe. ORT takes a graph's data all at
// once, so loading holds every widened shard of that graph in memory.

export type StoredDtype = "f32" | "raw" | "f16" | "i8";

export interface TensorEntry {
  name: string;
  dtype: StoredDtype;
  count: number;
  shape: number[];
  layoutOffset: number;
  layoutLength: number;
  storedOffset: number;
  storedLength: number;
  /** i8 only: where the fp32 scales start, from storedOffset, and the values per scale. */
  scalesOffset?: number;
  block?: number;
}

export interface Shard {
  file: string;
  layoutSize: number;
  storedSize: number;
  tensors: TensorEntry[];
}

export interface GraphFiles {
  graph: string;
  graphSize: number;
  shards: Shard[];
}

export type GraphName = "text" | "step" | "vocos";

export interface VariantInfo {
  variant: string;
  /** WebGPU features the graphs need, such as "shader-f16". */
  features?: string[];
  graphs: Record<GraphName, GraphFiles>;
}

/** A graph and its external data, as ORT's InferenceSession.create takes them. */
export interface GraphBytes {
  graph: Uint8Array;
  externalData: { path: string; data: Uint8Array | string }[];
}

/** Fetched only when the session is created, so one graph's data is in memory at a time. */
export type GraphSource = () => Promise<GraphBytes>;

let f16Bits: Uint32Array | null = null;

/** fp32 bit patterns of all 65536 fp16 values, subnormals, infinities and NaN payloads included. */
export function f16ToF32Bits(): Uint32Array {
  if (f16Bits) return f16Bits;
  const t = new Uint32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const man = h & 0x3ff;
    if (exp === 0 && man === 0) t[h] = sign >>> 0;
    else if (exp === 0) {
      let shifts = 0; // a subnormal: shift the mantissa up to its leading 1
      let m = man;
      while ((m & 0x400) === 0) {
        m <<= 1;
        shifts++;
      }
      t[h] = (sign | ((113 - shifts) << 23) | ((m & 0x3ff) << 13)) >>> 0;
    } else if (exp === 31) t[h] = (sign | 0x7f800000 | (man << 13)) >>> 0;
    else t[h] = (sign | ((exp + 112) << 23) | (man << 13)) >>> 0;
  }
  f16Bits = t;
  return t;
}

/** Whether the stored bytes differ from the layout: by dtype, or only by where the tensors are packed. */
export function needsWidening(shard: Shard): boolean {
  return (
    shard.storedSize !== shard.layoutSize ||
    shard.tensors.some((t) => t.dtype === "f16" || t.dtype === "i8" || t.storedOffset !== t.layoutOffset)
  );
}

/** One shard's stored bytes to the fp32 layout ORT reads. */
export function widenShard(stored: Uint8Array, shard: Shard): Uint8Array {
  if (stored.byteLength !== shard.storedSize)
    throw new Error(`${shard.file}: ${stored.byteLength} bytes, expected ${shard.storedSize}`);
  const out = new Uint8Array(shard.layoutSize);
  for (const t of shard.tensors) {
    const src = stored.subarray(t.storedOffset, t.storedOffset + t.storedLength);
    if (t.dtype === "f32" || t.dtype === "raw") {
      out.set(src, t.layoutOffset);
    } else if (t.dtype === "f16") {
      const table = f16ToF32Bits();
      const h = aligned(src, Uint16Array, t.count);
      const o = new Uint32Array(out.buffer, t.layoutOffset, t.count);
      for (let i = 0; i < t.count; i++) o[i] = table[h[i] as number] as number;
    } else {
      const block = t.block ?? 0;
      const q = new Int8Array(src.buffer, src.byteOffset, t.count);
      const scales = aligned(src.subarray(t.scalesOffset ?? 0), Float32Array, Math.ceil(t.count / block));
      const o = new Float32Array(out.buffer, t.layoutOffset, t.count);
      for (let i = 0; i < t.count; i++) o[i] = Math.fround((q[i] as number) * (scales[(i / block) | 0] as number));
    }
  }
  return out;
}

/** A typed view of `bytes`, copied first if its offset does not suit the element size. */
function aligned<T extends Uint16Array | Float32Array>(
  bytes: Uint8Array,
  Kind: { new (buffer: ArrayBuffer, offset: number, length: number): T; BYTES_PER_ELEMENT: number },
  length: number,
): T {
  const b = bytes.byteOffset % Kind.BYTES_PER_ELEMENT === 0 ? bytes : bytes.slice();
  return new Kind(b.buffer as ArrayBuffer, b.byteOffset, length);
}

async function fetchBytes(url: string, signal?: AbortSignal): Promise<Uint8Array> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * The variant at `baseUrl` (a folder with variant.json), as one source per graph. Shards that need no
 * widening go to ORT as URLs, which it fetches itself; the others are fetched and widened here.
 */
export async function variantSources(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<{ info: VariantInfo; sources: Record<GraphName, GraphSource> }> {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const res = await fetch(`${base}variant.json`, { signal });
  if (!res.ok) throw new Error(`${base}variant.json: HTTP ${res.status}`);
  const info = (await res.json()) as VariantInfo;
  const source =
    (files: GraphFiles): GraphSource =>
    async () => {
      const graph = await fetchBytes(base + files.graph, signal);
      const externalData: GraphBytes["externalData"] = [];
      for (const shard of files.shards) {
        const url = base + shard.file;
        externalData.push({
          path: shard.file,
          data: needsWidening(shard) ? widenShard(await fetchBytes(url, signal), shard) : url,
        });
      }
      return { graph, externalData };
    };
  return {
    info,
    sources: { text: source(info.graphs.text), step: source(info.graphs.step), vocos: source(info.graphs.vocos) },
  };
}

/** Throws if the adapter lacks a feature the variant needs; `adapter` defaults to the browser's own. */
export async function checkFeatures(info: VariantInfo, adapter?: GPUAdapter | null): Promise<void> {
  const needed = info.features ?? [];
  if (!needed.length) return;
  const a = adapter ?? (await navigator.gpu?.requestAdapter());
  const missing = needed.filter((f) => !a?.features.has(f));
  if (missing.length)
    throw new Error(`weight variant ${info.variant} needs ${missing.join(", ")}, which this GPU does not offer`);
}
