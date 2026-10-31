# WebGPU port: task list

Goal: run F5-TTS inference (voice cloning from a reference clip) in a browser on WebGPU. Inference only, no training.

Background and op inventory: see the WebGPU discussion summarized in section "Starting assumptions" below; model details in [../architecture.md](../architecture.md).

Status values: `plan not written`, `plan drafted`, `plan pressure-tested`, `in progress`, `done`.

Every plan, once written, is saved in this folder and pressure-tested by a fresh agent (given only the plan file) before work starts. Findings get folded into the plan.

## Starting assumptions

These are decisions to confirm in plan 00 before anything else builds on them.

- Model scope: `F5TTS_v1_Base` only (DiT, Vocos, 24 kHz, 100 mels, hop 256). Other configs later.
- Runtime: ONNX Runtime Web with the WebGPU execution provider. Hand-written WGSL only for ops it cannot do.
- ODE loop, chunking, cross-fade and text front end live in JS/TS.
- Language scope: English and Chinese, as the pretrained model supports.
- No in-browser ASR in the first version; the user supplies the reference text.
- We publish the converted ONNX weights in our own Hugging Face repo (decided); the browser downloads from there. Not LFS in a GitHub repo: bandwidth quota, and Pages does not serve LFS content.

## Plans

| # | Plan (file to be written) | Depends on | Status |
|---|---|---|---|
| 00 | [Scope, runtime decision and parity harness](webgpu-00-scope-and-parity.md) | none | plan not written |
| 01 | [DiT export to ONNX](webgpu-01-dit-export.md) | 00 | plan not written |
| 02 | [Mel spectrogram and vocoder](webgpu-02-mel-and-vocoder.md) | 00 | plan not written |
| 03 | [Text front end](webgpu-03-text-frontend.md) | 00 | plan not written |
| 04 | [Sampler and generation pipeline](webgpu-04-sampler-pipeline.md) | 01, 02, 03 | plan not written |
| 05 | [Weight delivery, precision and quantization](webgpu-05-weights-precision.md) | 01, 02 | plan not written |
| 06 | [Web app and public API](webgpu-06-web-app.md) | 04, 05 | plan not written |
| 07 | [Performance and browser validation](webgpu-07-perf-validation.md) | 04, 05 | plan not written |

```
00 -> 01 -> 05 -> 06
 |    |      \-> 07
 |    \-> 04 -/
 +-> 02 -> 04
 +-> 03 -> 04
```

01, 02 and 03 can run in parallel once 00 is done.

## Tasks per plan

### 00 Scope, runtime decision and parity harness
- [ ] Confirm the starting assumptions above, or change them.
- [ ] Choose ORT Web versus custom WGSL, with a small spike: run one DiT block and a depthwise conv on WebGPU.
- [ ] Pick the repo layout for web code (where it lives, build tool, language).
- [ ] Build the golden-data generator: fixed seed, fixed noise tensor, fixed inputs; dump text ids, reference mel, per-step DiT output, final mel, final wav from PyTorch.
- [ ] Define tolerances per stage (mel, DiT output, wav).
- [ ] Write the test runner that compares any port stage to the golden data.

### 01 DiT export to ONNX
- [ ] Wrap `DiT.forward` for export (fixed input signature, CFG batch handled outside or inside).
- [ ] Replace or precompute rotary embeddings so they export cleanly.
- [ ] Handle the text embedding cache (cond and uncond) in the exported graph.
- [ ] Settle dynamic versus bucketed sequence lengths; check the attention mask path.
- [ ] Export with external weight data; verify parity against golden data in onnxruntime (CPU).
- [ ] Verify the graph loads and matches on ORT Web WebGPU; list any unsupported or slow ops.

### 02 Mel spectrogram and vocoder
- [ ] Specify the exact Vocos mel (STFT params, filterbank, log/clamp) from `model/modules.py`.
- [ ] Implement mel extraction in JS or WGSL; match golden mel.
- [ ] Export the Vocos backbone to ONNX (start from `runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py`).
- [ ] Implement inverse STFT outside ONNX; match golden wav.
- [ ] Reference-audio prep: mono, resample to 24 kHz, RMS normalise, clip to about 12 s.

### 03 Text front end
- [ ] Specify `convert_char_to_pinyin` behaviour precisely, including polyphones and non-Chinese text.
- [ ] Choose a JS pinyin and word-segmentation library; measure mismatches against Python on a test corpus.
- [ ] Port vocab loading (`infer/examples/vocab.txt`; index 0 is space and unknown).
- [ ] Port `chunk_text` (UTF-8 byte budget, ASCII and CJK sentence punctuation).
- [ ] Port reference-text punctuation fix-up.
- [ ] Build a corpus test: token ids from JS equal token ids from Python.

### 04 Sampler and generation pipeline
- [ ] ODE Euler loop with CFG (`pred + (pred - null) * strength`).
- [ ] Sway sampling and EPSS timestep table.
- [ ] Duration estimate from reference text and audio, `fix_duration` split, short-chunk speed rule.
- [ ] Noise generation with seedable RNG; decide whether to match PyTorch noise or accept non-identical output.
- [ ] Conditioning mel assembly, output crop of reference frames, mel to vocoder.
- [ ] Per-chunk loop and cross-fade assembly; optional streaming of chunks.
- [ ] End-to-end test against golden wav with injected noise.

### 05 Weight delivery, precision and quantization
- [ ] Hugging Face hosting (decided). The browser needs converted ONNX files, so we publish them in our own repo (decided). Derivative of CC-BY-NC weights: repo owner, CC-BY-NC licence tag, model card crediting the F5-TTS authors and the Emilia dataset, link to the original `SWivid/F5-TTS` repo, and a statement that use is non-commercial.
- [ ] Script that converts the original safetensors to the published ONNX files and uploads them, so the repo can be rebuilt.
- [ ] Download plan for roughly 0.7 to 1.3 GB: CORS check on the chosen URLs, progress UI, resume, file sharding.
- [ ] Cache in OPFS or IndexedDB; versioning and eviction.
- [ ] fp16 path with `shader-f16` detection; fp32 fallback.
- [ ] Buffer and binding size limits across target devices.
- [ ] Try int8 and 4-bit quantization; judge quality with listening tests and the eval metrics already in `src/f5_tts/eval/`.

### 06 Web app and public API
- [ ] Minimal demo page: reference upload or record, reference text, target text, play and download.
- [ ] Reusable module API (load, generate, cancel, progress events).
- [ ] Run inference in a Worker so the UI stays responsive.
- [ ] Decide on optional reference transcription (transformers.js Whisper) and its cost.
- [ ] Error handling: no WebGPU, out of memory, download failure.

### 07 Performance and browser validation
- [ ] Benchmark script: real-time factor, time to first audio, peak GPU memory.
- [ ] Profile per-step cost; find the slowest ops (attention, convs).
- [ ] Optimisations: step count (EPSS 16 steps), batching the CFG pair, sequence bucketing.
- [ ] Browser and device matrix: Chrome, Edge, Safari, Firefox; desktop GPU, integrated GPU, mobile.
- [ ] Record limits and known failures in [../debugging.md](../debugging.md).

## Open questions

- Is exact numerical match with PyTorch required, or only equal audio quality?
- Exact attribution wording for the model card, and confirming the app stays non-commercial. Gate for plan 05: nothing is published until this is settled.
- Which browsers are in scope.
- Whether the Python package should gain a `webgpu/` export script entry point or this stays a separate project.
