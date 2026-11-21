// Runs the exported Vocos graph (plan 02) with ORT Web on WebGPU, then the inverse STFT in TypeScript
// (plan 02 D2, option A). The graph takes a mel (1, 100, T) and gives head_out (1, T, 1026); vocos_debug.onnx
// also gives the backbone output.
import * as ort from "onnxruntime-web/webgpu";
import { istft } from "../audio/istft";

export interface VocoderOptions {
  /** URL of vocos.onnx or vocos_debug.onnx; one file, no external data. */
  url: string;
  /** ORT log severity while creating the session: 0 is verbose, which prints node placement. */
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
}

/** Where a decode spent its time, in ms. */
export interface DecodeTiming {
  /** session.run until the GPU has finished. */
  graphMs: number;
  /** Copying head_out (T x 1026 floats) from the GPU. */
  readbackMs: number;
  istftMs: number;
}

export class Vocoder {
  private constructor(readonly session: ort.InferenceSession) {}

  static async create(options: VocoderOptions): Promise<Vocoder> {
    const res = await fetch(options.url);
    if (!res.ok) throw new Error(`${options.url}: HTTP ${res.status}`);
    const graph = new Uint8Array(await res.arrayBuffer());
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ["webgpu"],
      preferredOutputLocation: "gpu-buffer",
      logSeverityLevel: options.logSeverityLevel ?? 2,
      logVerbosityLevel: 0,
    });
    return new Vocoder(session);
  }

  get outputNames(): readonly string[] {
    return this.session.outputNames;
  }

  /** The graph's outputs, on the GPU; the caller disposes them. `mel` is (1, 100, T). */
  async run(mel: ort.Tensor): Promise<Record<string, ort.Tensor>> {
    const out = await this.session.run({ mel });
    if (!out.head_out) throw new Error("vocoder graph: missing head_out");
    return out;
  }

  /** Mel (1, 100, T) to audio ((T - 1) * 256 samples at 24 kHz), before the output RMS rescale. */
  async decode(mel: ort.Tensor): Promise<{ wave: Float32Array; timing: DecodeTiming }> {
    const device = await ort.env.webgpu.device;
    const t0 = performance.now();
    const out = await this.run(mel);
    try {
      await device.queue.onSubmittedWorkDone();
      const t1 = performance.now();
      const head = out.head_out as ort.Tensor;
      const data = (await head.getData()) as Float32Array;
      const t2 = performance.now();
      const wave = istft(data, head.dims[1] as number);
      const t3 = performance.now();
      return { wave, timing: { graphMs: t1 - t0, readbackMs: t2 - t1, istftMs: t3 - t2 } };
    } finally {
      for (const t of Object.values(out)) t.dispose();
    }
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}
