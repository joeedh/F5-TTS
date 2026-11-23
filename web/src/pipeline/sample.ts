// The ODE loop for one chunk on WebGPU (plan 04 step 4, D2): the text graph once, then nfe Euler steps with
// classifier-free guidance, each feeding the step graph's x_next back as the next x on the GPU. Only the
// final x is read back. Every GPU tensor goes through a TensorLedger, which is the leak test (WebGPU has no
// memory query).
import * as ort from "onnxruntime-web/webgpu";
import type { DiT } from "../dit";

/** Counts the GPU tensors the pipeline holds. `live` returns to its start when nothing leaks. */
export class TensorLedger {
  private readonly held = new Set<ort.Tensor>();

  get live(): number {
    return this.held.size;
  }

  track<T extends ort.Tensor>(t: T): T {
    this.held.add(t);
    return t;
  }

  dispose(t: ort.Tensor): void {
    if (!this.held.delete(t)) throw new Error("TensorLedger: disposing a tensor it does not hold");
    t.dispose();
  }
}

async function download(device: GPUDevice, buf: GPUBuffer, bytes: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: buf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, buf.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange().slice(0, bytes));
  } finally {
    staging.destroy();
  }
}

/** Uploads float32 data once; the tensor's dispose frees the buffer. */
export function uploadF32(device: GPUDevice, data: Float32Array, dims: number[]): ort.Tensor {
  const buf = device.createBuffer({
    size: Math.max(16, Math.ceil(data.byteLength / 16) * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return ort.Tensor.fromGpuBuffer(buf, {
    dataType: "float32",
    dims,
    download: () => download(device, buf, data.byteLength),
    dispose: () => buf.destroy(),
  });
}

export interface ChunkInputs {
  /** The joined text's token ids (convertCharToPinyin then textToIds). */
  ids: Int32Array;
  /** The starting noise, (1, duration, 100). */
  noise: Float32Array;
  /** stepCond's output, (1, duration, 100). */
  stepCond: Float32Array;
  duration: number;
  /** timeGrid's output: nfe + 1 points. */
  grid: Float32Array;
  cfgStrength: number;
}

export interface SampleHooks {
  /** After each step has finished on the GPU. `x` is the new state, owned by the loop: read it, do not keep it. */
  onStep?: (step: number, steps: number, x: ort.Tensor) => void | Promise<void>;
  /** Checked between steps; on abort the loop waits for the GPU, frees its tensors and throws the reason. */
  signal?: AbortSignal;
}

/** Runs one chunk and returns the last step's x, (1, duration, 100), before the reference restore. */
export async function sampleChunk(
  dit: DiT,
  device: GPUDevice,
  ledger: TensorLedger,
  input: ChunkInputs,
  hooks: SampleHooks = {},
): Promise<Float32Array> {
  const n = input.duration;
  const dims = [1, n, 100];
  if (input.noise.length !== n * 100 || input.stepCond.length !== n * 100)
    throw new Error("sampleChunk: noise and stepCond must be (1, duration, 100)");
  const steps = input.grid.length - 1;
  // below 1e-5 upstream runs the conditional pass alone (cfm.py:179); the batch-2 graph equals it only at 0
  const cfg = input.cfgStrength < 1e-5 ? 0 : input.cfgStrength;
  const stepCond = ledger.track(uploadF32(device, input.stepCond, dims));
  let x = ledger.track(uploadF32(device, input.noise, dims));
  let text: { cond: ort.Tensor; uncond: ort.Tensor } | undefined;
  try {
    const emb = await dit.embedText(input.ids, n);
    text = { cond: ledger.track(emb.cond), uncond: ledger.track(emb.uncond) };
    for (let s = 0; s < steps; s++) {
      hooks.signal?.throwIfAborted();
      const t0 = input.grid[s] as number;
      const out = await dit.step({
        x,
        stepCond,
        text: { n, ...text },
        time: t0,
        cfgStrength: cfg,
        dt: (input.grid[s + 1] as number) - t0,
      });
      for (const t of Object.values(out)) ledger.track(t);
      for (const [k, t] of Object.entries(out)) if (k !== "x_next") ledger.dispose(t);
      ledger.dispose(x);
      x = out.x_next;
      // session.run resolves before the GPU is done; wait, so progress and abort are honest
      await device.queue.onSubmittedWorkDone();
      await hooks.onStep?.(s, steps, x);
    }
    return (await x.getData()) as Float32Array;
  } finally {
    await device.queue.onSubmittedWorkDone();
    ledger.dispose(x);
    ledger.dispose(stepCond);
    if (text) {
      ledger.dispose(text.cond);
      ledger.dispose(text.uncond);
    }
  }
}
