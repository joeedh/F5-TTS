# Docs index

| File | Contents |
|---|---|
| [architecture.md](architecture.md) | Package layout, model, training and inference data flow |
| [shape-ledger.md](shape-ledger.md) | Record of problems caused by unchecked tensor shapes; decides if pyright gets extended |
| [debugging.md](debugging.md) | Running log of bugs and diagnosis tips; first entries are gotchas found while checking the architecture doc |
| [research/webgpu-environment.md](research/webgpu-environment.md) | WebGPU in this cloud container: software only (SwiftShader); probe results and what they mean for the port |
| [research/webgpu-laptop.md](research/webgpu-laptop.md) | WebGPU on the Windows laptop (RTX 5050 + Intel iGPU): adapter limits, and how to get the discrete GPU (`--force_high_performance_gpu`) |
| [research/webgpu-runtime-spike.md](research/webgpu-runtime-spike.md) | Plan 00 step 3: ORT Web vs hand-written WGSL on one DiT block; D3 decided for ORT Web; fp16 accuracy and attention-cost findings |
| [research/webgpu-audio.md](research/webgpu-audio.md) | Plan 02: reference prep exact with pydub, the Vocos mel and float32 filterbank, the resampler, the vocoder graph on WebGPU, istft timing and D2 |
| [research/webgpu-dit-export.md](research/webgpu-dit-export.md) | Plan 01 step 4: full DiT step graph on WebGPU; `mha` and dynamic n chosen, graph capture fails in ORT Web 1.31, per-step time and RTF on both GPUs |
| [research/webgpu-text-frontend.md](research/webgpu-text-frontend.md) | Plan 03: jieba-rs 0.11.0 segmentation and the no-CJK rule, pypinyin's lazy_pinyin path written out exactly (mmseg, TONE3, tone sandhi quirks), and how each part was checked |
| [plans/webgpu-port-tasks.md](plans/webgpu-port-tasks.md) | Task list: Python typing/tooling plan (pyright) and the WebGPU port plans |
| [plans/webgpu-00-scope-and-parity.md](plans/webgpu-00-scope-and-parity.md) | WebGPU plan 00: scope, ORT Web vs WGSL spike, golden-data generator, tolerances, parity runner |
| [plans/webgpu-01-dit-export.md](plans/webgpu-01-dit-export.md) | WebGPU plan 01: DiT export to ONNX (text and step graphs), attention and sequence-length choices, browser parity |
| [plans/webgpu-02-mel-and-vocoder.md](plans/webgpu-02-mel-and-vocoder.md) | WebGPU plan 02: reference audio prep, mel, Vocos graph and inverse STFT in TypeScript |
| [plans/webgpu-03-text-frontend.md](plans/webgpu-03-text-frontend.md) | WebGPU plan 03: chunking, Chinese segmentation and pinyin, token ids, matched against Python |
| [plans/webgpu-04-sampler-pipeline.md](plans/webgpu-04-sampler-pipeline.md) | WebGPU plan 04: chunk plan, torch-compatible noise, the ODE loop on the GPU, vocoder and cross-fade, the generation API, and a WER/similarity comparison with PyTorch |

Add a row here whenever a doc is added.
