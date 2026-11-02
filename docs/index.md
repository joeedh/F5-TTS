# Docs index

| File | Contents |
|---|---|
| [architecture.md](architecture.md) | Package layout, model, training and inference data flow |
| [shape-ledger.md](shape-ledger.md) | Record of problems caused by unchecked tensor shapes; decides if pyright gets extended |
| [debugging.md](debugging.md) | Running log of bugs and diagnosis tips; first entries are gotchas found while checking the architecture doc |
| [research/webgpu-environment.md](research/webgpu-environment.md) | WebGPU in this cloud container: software only (SwiftShader); probe results and what they mean for the port |
| [research/webgpu-laptop.md](research/webgpu-laptop.md) | WebGPU on the Windows laptop (RTX 5050 + Intel iGPU): adapter limits, and how to get the discrete GPU (`--force_high_performance_gpu`) |
| [plans/webgpu-port-tasks.md](plans/webgpu-port-tasks.md) | Task list: Python typing/tooling plan (pyright) and the WebGPU port plans |
| [plans/webgpu-00-scope-and-parity.md](plans/webgpu-00-scope-and-parity.md) | WebGPU plan 00: scope, ORT Web vs WGSL spike, golden-data generator, tolerances, parity runner |

Add a row here whenever a doc is added.
