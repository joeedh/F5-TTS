// Word segmentation (plan 03 step 3, D1): rjieba.cut, from our wasm build of jieba-rs 0.11.0
// (tools/webgpu/jieba_wasm). The wasm has a plain C ABI; see its src/lib.rs.

interface JiebaExports {
  memory: WebAssembly.Memory;
  init(): void;
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  cut(ptr: number, len: number, hmm: number): number;
  result_ptr(): number;
}

/** The built wasm, for a browser fetch; Vite rewrites this to the hashed asset. */
export const JIEBA_WASM_URL = new URL("./jieba.wasm", import.meta.url);

const encoder = new TextEncoder();
// ignoreBOM keeps a U+FEFF at the start of a token, which jieba returns like any other character
const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

export class Segmenter {
  private constructor(private readonly wasm: JiebaExports) {}

  /** Instantiates the wasm and builds the dictionary. `source` is a URL to fetch, or the wasm's bytes. */
  static async load(source: URL | string | BufferSource = JIEBA_WASM_URL): Promise<Segmenter> {
    const { instance } =
      source instanceof URL || typeof source === "string"
        ? await WebAssembly.instantiateStreaming(fetch(source))
        : await WebAssembly.instantiate(source);
    const wasm = instance.exports as unknown as JiebaExports;
    wasm.init();
    return new Segmenter(wasm);
  }

  /** rjieba.cut(text, hmm). Throws on a lone surrogate, which Python cannot pass to rjieba either. */
  cut(text: string, hmm = true): string[] {
    if (!text.isWellFormed()) throw new Error("text has a lone surrogate");
    const bytes = encoder.encode(text);
    if (bytes.length === 0) return [];
    const ptr = this.wasm.alloc(bytes.length);
    try {
      new Uint8Array(this.wasm.memory.buffer, ptr, bytes.length).set(bytes);
      const n = this.wasm.cut(ptr, bytes.length, hmm ? 1 : 0);
      if (n === 0xffffffff) throw new Error("jieba: input is not UTF-8");
      // the memory may have grown during cut, so views are made after it
      const ends = new Uint32Array(this.wasm.memory.buffer, this.wasm.result_ptr(), n);
      const out: string[] = [];
      let start = 0;
      for (const end of ends) {
        out.push(decoder.decode(bytes.subarray(start, end)));
        start = end;
      }
      return out;
    } finally {
      this.wasm.dealloc(ptr, bytes.length);
    }
  }
}
