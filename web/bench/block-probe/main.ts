// Runs tools/webgpu/probe_block.py's one-block graph on WebGPU and compares every intermediate with the
// float64 reference. Loaded by run.mjs into the main dev server's page.
import * as ort from "onnxruntime-web/webgpu";
import { metrics, parseNpy, type Tensor } from "../../test/parity";

const BASE = "/model/dit/probe/";

async function npy(file: string): Promise<Tensor> {
  const res = await fetch(BASE + file);
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  return parseNpy(await res.arrayBuffer());
}

export async function runProbe(): Promise<Record<string, unknown>> {
  const info = (await (await fetch(`${BASE}probe.json`)).json()) as {
    outputs: string[];
    cpu_fp32_rel_l2: Record<string, number>;
  };
  const graph = new Uint8Array(await (await fetch(`${BASE}block.onnx`)).arrayBuffer());
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"],
    externalData: [{ path: "block.onnx.data", data: `${BASE}block.onnx.data` }],
  });
  const x = await npy("in.x.npy");
  const t = await npy("in.t.npy");
  const out = await session.run({
    x: new ort.Tensor("float32", x.data, x.shape),
    t: new ort.Tensor("float32", t.data, t.shape),
  });
  const rows: Record<string, unknown>[] = [];
  for (const name of info.outputs) {
    const ref = await npy(`ref.${name}.npy`);
    const got = (await out[name]!.getData()) as Float32Array;
    const m = metrics(got, ref.data);
    rows.push({ name, relL2: m.relL2, maxAbs: m.maxAbs, cpuFp32RelL2: info.cpu_fp32_rel_l2[name] });
  }
  await session.release();
  const device = await ort.env.webgpu.device;
  return { adapter: device.adapterInfo.vendor, rows };
}
