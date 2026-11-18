// Minimal .npy reader for the spike: little-endian float32, C order only.

export interface NpyArray {
  shape: number[];
  data: Float32Array;
}

export function parseNpy(buf: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (bytes[0] !== 0x93 || magic !== "NUMPY") throw new Error("not an .npy file");
  const major = bytes[6];
  const view = new DataView(buf);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen));
  if (!header.includes("'descr': '<f4'")) throw new Error(`unsupported dtype in ${header}`);
  if (header.includes("'fortran_order': True")) throw new Error("fortran order not supported");
  const shapeText = /'shape': \(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
  const offset = headerStart + headerLen;
  return { shape, data: new Float32Array(buf.slice(offset)) };
}

export async function fetchNpy(url: string): Promise<NpyArray> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return parseNpy(await res.arrayBuffer());
}
