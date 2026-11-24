// Runs the exported DiT graphs (plan 01) with ORT Web on WebGPU: the text graph once per chunk, the step
// graph once per ODE step. Outputs stay on the GPU; callers read them back only when they need to.
// Graph inputs and outputs are described in docs/plans/webgpu-01-dit-export.md, D1.
import * as ort from "onnxruntime-web/webgpu";
import type { GraphSource } from "../weights/variant";

/** Each graph comes from a URL (its weights at the same URL plus ".data") or from a source (plan 05). */
export interface DiTOptions {
  /** URL of text.onnx. */
  textUrl?: string;
  /** URL of step.onnx (or a variant such as step_mha.onnx, or step_debug.onnx for internal outputs). */
  stepUrl?: string;
  /** The text graph and its data, from a weight variant (weights/variant.ts); used instead of textUrl. */
  text?: GraphSource;
  step?: GraphSource;
  /** ORT log severity while creating the sessions: 0 is verbose, which prints node placement. */
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
}

/** The text embeddings for one chunk, (1, n, 512) each, on the GPU. Reuse them for every step of the chunk. */
export interface TextEmbedding {
  n: number;
  cond: ort.Tensor;
  uncond: ort.Tensor;
}

export interface StepInputs {
  x: ort.Tensor; // (1, n, 100)
  stepCond: ort.Tensor; // (1, n, 100)
  text: TextEmbedding;
  time: number;
  cfgStrength: number;
  dt: number; // t_{k+1} - t_k
}

/** pred (2, n, 100), v and x_next (1, n, 100); the debug graph adds its internal outputs. */
export type StepOutputs = Record<string, ort.Tensor> & { pred: ort.Tensor; v: ort.Tensor; x_next: ort.Tensor };

/** A URL's graph, with its data at the URL plus ".data". */
export function urlSource(url: string): GraphSource {
  return async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const file = url.split("/").pop() ?? url;
    return {
      graph: new Uint8Array(await res.arrayBuffer()),
      externalData: [{ path: `${file}.data`, data: `${url}.data` }],
    };
  };
}

export async function loadSession(
  source: GraphSource,
  logSeverityLevel: 0 | 1 | 2 | 3 | 4,
): Promise<ort.InferenceSession> {
  const { graph, externalData } = await source(); // dropped once ORT has its copy
  return ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    preferredOutputLocation: "gpu-buffer",
    externalData,
    logSeverityLevel,
    logVerbosityLevel: 0,
  });
}

function pick(source: GraphSource | undefined, url: string | undefined, name: string): GraphSource {
  if (source) return source;
  if (url) return urlSource(url);
  throw new Error(`DiT.create: no ${name} graph`);
}

function scalar(v: number): ort.Tensor {
  return new ort.Tensor("float32", new Float32Array([v]), [1]);
}

export class DiT {
  private constructor(
    readonly textSession: ort.InferenceSession,
    readonly stepSession: ort.InferenceSession,
  ) {}

  static async create(options: DiTOptions): Promise<DiT> {
    const level = options.logSeverityLevel ?? 2;
    const text = await loadSession(pick(options.text, options.textUrl, "text"), level);
    try {
      return new DiT(text, await loadSession(pick(options.step, options.stepUrl, "step"), level));
    } catch (e) {
      await text.release();
      throw e;
    }
  }

  get stepOutputNames(): readonly string[] {
    return this.stepSession.outputNames;
  }

  /** `ids` are the chunk's token ids; they are cut or padded to n with -1, as TextEmbedding does. */
  async embedText(ids: Int32Array, n: number): Promise<TextEmbedding> {
    const padded = new Int32Array(n).fill(-1);
    padded.set(ids.subarray(0, Math.min(n, ids.length)));
    const out = await this.textSession.run({ text: new ort.Tensor("int32", padded, [1, n]) });
    const cond = out.text_cond;
    const uncond = out.text_uncond;
    if (!cond || !uncond) throw new Error("text graph: missing text_cond or text_uncond");
    return { n, cond, uncond };
  }

  /** One ODE step: the DiT on both CFG halves, v and the Euler update. The caller disposes the outputs. */
  async step(inputs: StepInputs): Promise<StepOutputs> {
    const out = await this.stepSession.run({
      x: inputs.x,
      step_cond: inputs.stepCond,
      text_cond: inputs.text.cond,
      text_uncond: inputs.text.uncond,
      time: scalar(inputs.time),
      cfg_strength: scalar(inputs.cfgStrength),
      dt: scalar(inputs.dt),
    });
    if (!out.pred || !out.v || !out.x_next) throw new Error("step graph: missing pred, v or x_next");
    return out as StepOutputs;
  }

  async release(): Promise<void> {
    await this.stepSession.release();
    await this.textSession.release();
  }
}
