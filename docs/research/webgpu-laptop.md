# WebGPU on the Windows laptop

Measured 2026-11-02 with `tools/webgpu-probe/probe.cjs` (plan 00, step 2). Compare with the cloud container: [webgpu-environment.md](webgpu-environment.md).

## Hardware and browsers

- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, driver 591.49. WebGPU reports vendor `nvidia`, architecture `blackwell`.
- Intel integrated graphics, driver 32.0.101.8509. WebGPU reports vendor `intel`, architecture `xe-3lpg`.
- Chrome 156.0.8078.76 and Edge 156.0.4314.58, driven by Playwright 1.64 (`channel: "chrome"` / `"msedge"`, so the installed browsers are used and no Playwright browser download is needed).

## Choosing the GPU

The browser defaults to the **Intel** GPU, and the page cannot change that.

| How | Result |
|---|---|
| default flags | Intel |
| `requestAdapter({ powerPreference: "high-performance" })` | Intel. Chrome logs: "The powerPreference option is currently ignored when calling requestAdapter() on Windows. See https://crbug.com/369219127" |
| `--use-webgpu-power-preference=force-high-performance` | no adapter at all ("No available adapters"), headless and headed |
| `--use-webgpu-power-preference=force-low-power` | Intel |
| `--force_high_performance_gpu` | **NVIDIA**, in Chrome and Edge, headless and headed |

- `--force_high_performance_gpu` moves Chrome's whole GPU process to the discrete GPU, and WebGPU then offers that adapter. The probe's `dgpu` variant uses it. My reading of the failed `force-high-performance` row is that WebGPU only offers adapters that match the GPU process's adapter, but this was not checked in the Chrome source.
- For tests and benchmarks, launch with `--force_high_performance_gpu` to get the NVIDIA GPU, and with no flag to get Intel. Record the adapter that came back; never assume.
- For real users this matters a lot: on a Windows laptop like this one, a web page gets the integrated GPU unless the user changes the Windows graphics setting for their browser (Settings, System, Display, Graphics). Plan 06 should detect the adapter (`adapter.info.vendor`) and tell the user if they are on an integrated GPU. Plan 07 should treat the integrated GPU as the default case on Windows laptops, not the edge case.

## Adapter capabilities

Identical on both GPUs unless noted.

| Check | NVIDIA | Intel |
|---|---|---|
| `maxBufferSize` | 2 GiB | 2 GiB |
| `maxStorageBufferBindingSize` | 2 GiB - 4 | 2 GiB - 4 |
| Allocating a storage buffer, 128 MB to 2048 MB | all succeed | all succeed |
| `maxStorageBuffersPerShaderStage` | 16 | 16 |
| `maxComputeInvocationsPerWorkgroup` | 1024 | 1024 |
| `maxComputeWorkgroupStorageSize` | 32 KiB | 32 KiB |
| `maxBindGroups` | 4 | 4 |
| `shader-f16` | yes, and an f16 shader compiles | yes, and an f16 shader compiles |
| `subgroups` | yes, size 32 | yes, size 16 to 32 |
| `subgroup-size-control`, `timestamp-query` | yes | yes |
| Vector add, matmul 256 checked against the CPU | correct (max abs error 2e-5) | correct (4e-5) |
| Naive 1024^3 f32 matmul | 24 ms, about 85 to 90 GFLOPS | 51 to 68 ms, about 30 to 42 GFLOPS |
| Uncaptured WebGPU errors | none | none |

- The naive matmul is a lower bound, not a capability figure: one thread per output, no tiling, and the time includes submit and readback. For scale, CUDA cuBLAS on the same NVIDIA GPU does 7.6 TFLOPS fp32 and 25.4 TFLOPS fp16 (`docs/debugging.md`, 2026-11-02). Real WebGPU numbers come from the plan 00 spike.
- Both GPUs allow 2 GiB buffers, against 1 GiB on SwiftShader in the cloud container. Weight sharding (plan 05) is still needed for other devices.
- Full feature lists: run the probe and read the JSON.

## Running the probe here

```
cd tools/webgpu-probe
npm install
$env:PROBE_CHANNEL = "chrome"     # or msedge
node probe.cjs default            # Intel
node probe.cjs dgpu               # NVIDIA
```

`PROBE_HEADED=1` shows the window; headless gave the same adapters and features here. `PROBE_POWER` passes a `powerPreference` to `requestAdapter`, which Chrome on Windows ignores.

## Caveats

- One or two runs per configuration. Naive matmul times varied by about 20 percent between runs.
- Windows GPU selection may change with Chrome versions (the crbug above is open), with driver updates, or with the Windows graphics setting for the browser. That setting was left at its default for these runs.
