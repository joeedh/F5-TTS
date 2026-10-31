# WebGPU in this cloud environment

Measured 2026-10-31 with `tools/webgpu-probe/probe.cjs`.

## Hardware

- No GPU. No `/dev/dri`, no `/dev/nvidia*`, no `nvidia-smi`, no Vulkan ICD files.
- 4 vCPU (Intel Xeon @ 2.10 GHz), 15 GB RAM.

## What works

Chromium 153 (Playwright's bundled build, new headless via `channel: "chromium"`) exposes `navigator.gpu` and returns an adapter. It is software only: vendor `google`, architecture `swiftshader`, which is Vulkan emulated on the CPU. This is automatic; no flags are needed, and the `default`, `swiftshader` and `vulkan` flag variants in the probe all end up on the same adapter.

| Check | Result |
|---|---|
| Vector add, 65536 floats | correct |
| Matmul 256x256, vs CPU on sampled cells | correct, max abs error 2e-5 |
| Compute shader compile and dispatch | works, no uncaptured errors |
| `maxBufferSize`, `maxStorageBufferBindingSize` | 1 GiB each |
| Allocating a storage buffer | 128, 256, 512 MB ok; 1024 MB failed with out-of-memory even though it equals the limit |
| `maxStorageBuffersPerShaderStage` | 10 |
| `maxComputeInvocationsPerWorkgroup` | 256 |
| `maxComputeWorkgroupStorageSize` | 32 KiB |
| `maxBindGroups` | 4 |
| Features on the device | `subgroups` (min and max subgroup size 4), `timestamp-query` available |
| `shader-f16` | not available |
| Naive 1024^3 f32 matmul | about 3.6 s, roughly 0.6 GFLOPS |

Full limits and feature list: run the probe and read the JSON.

## What this means for the port

- Good for correctness testing: shaders compile, run, and results can be compared against golden data (plan 00). Behaviour of WebGPU validation is real, since this is Chromium's real implementation (Dawn).
- Not usable for performance work. A rough estimate: one DiT forward pass on a 1500-token sequence is on the order of 1 TFLOP (about 330M parameters x 2 x tokens, attention extra). Sampling needs about 64 passes per chunk. At 0.6 GFLOPS (naive kernel, so a pessimistic number) that is days. A tuned kernel on 4 cores is perhaps 10 to 50 times faster, still far too slow. End-to-end generation of the full model is not practical here. Use small test shapes or truncated models.
- Not usable for fp16 testing: no `shader-f16`. fp16 paths need a machine with a real GPU.
- The 1 GiB buffer limit is lower than many desktop GPUs offer (they commonly allow 2 GiB or more). Weights of 0.7 to 1.3 GB will need sharding across buffers to be safe anywhere. Plan 05 already has this as a task.
- Subgroups are available (size 4 here, 32 or 64 on real GPUs), so subgroup-based kernels can at least be compiled for testing.
- Real GPU timing needs another machine: a desktop browser, or a CI runner with a GPU. Plan 07 should assume that.

## Running the probe

```
cd tools/webgpu-probe
node probe.cjs            # default flags
node probe.cjs swiftshader
node probe.cjs vulkan
```

Needs the `playwright` package and the Chromium build at `/opt/pw-browsers`, both pre-installed here. It starts a local HTTP server because WebGPU needs a secure context (`about:blank` is not one, and the default headless shell has no WebGPU). Output is JSON on stdout. Set `PROBE_CHANNEL` to choose a different Playwright channel.

## Caveats

- One run per variant on one machine. Timings varied between runs (matmul 256 took 70 to 100 ms).
- The 1024 MB allocation failure was seen once and not investigated; it may depend on free memory.
