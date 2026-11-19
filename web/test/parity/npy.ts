// .npy reader for golden data: little-endian float32 or int32, C order, format versions 1 to 3.

export type DType = "float32" | "int32";

export interface Tensor {
  dtype: DType;
  shape: number[];
  data: Float32Array | Int32Array;
}

export function parseNpy(buf: ArrayBuffer): Tensor {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (bytes[0] !== 0x93 || magic !== "NUMPY") throw new Error("not an .npy file");
  const major = bytes[6];
  const view = new DataView(buf);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.subarray(start, start + headerLen));
  const descr = /'descr': '([^']+)'/.exec(header)?.[1];
  if (header.includes("'fortran_order': True")) throw new Error("fortran-order .npy is not supported");
  const shape = (/'shape': \(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const offset = start + headerLen;
  const body = buf.slice(offset, offset + count * 4);
  if (descr === "<f4") return { dtype: "float32", shape, data: new Float32Array(body) };
  if (descr === "<i4") return { dtype: "int32", shape, data: new Int32Array(body) };
  throw new Error(`unsupported .npy dtype ${descr}`);
}

/** Copies a Node Buffer (which may sit at an unaligned offset in a shared pool) into its own ArrayBuffer. */
export function toArrayBuffer(view: Uint8Array): ArrayBuffer {
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(view);
  return out;
}
