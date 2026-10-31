# Task list: Python tooling, then the WebGPU port

Two bodies of work. Plan P (Python typing and tooling) comes first and is independent of the port. Plans 00 to 07 are the WebGPU port: run F5-TTS inference (voice cloning from a reference clip) in a browser on WebGPU, inference only, no training.

Model details: [../architecture.md](../architecture.md). WebGPU in the cloud container: [../research/webgpu-environment.md](../research/webgpu-environment.md).

Status values: `plan not written`, `plan drafted`, `plan pressure-tested`, `in progress`, `done`.

Every plan, once written, is saved in this folder and pressure-tested by a fresh agent (given only the plan file) before work starts. Findings get folded into the plan.

## Plan P: Python typing, formatter and linter

Plan file (not written): [python-pyright-tooling.md](python-pyright-tooling.md). Status: `plan not written`. Depends on: nothing. Blocks: nothing, but the port's Python export scripts (plans 01, 02, 05) should be written under it.

Where things stand (checked 2026-10-31): no type checker is configured. Ruff does lint, format and import sorting through pre-commit and the `pre-commit` GitHub workflow. Pyright 1.1.414 and mypy 2.3.1 are installed in the cloud container, but not torch. First pyright run without torch: 672 errors, 23 warnings over 47 files. Largest groups: missing imports 239, possibly-unbound variables 130, attribute access 109, invalid type arguments 69 (the ein-notation strings like `float["b n d"]`), argument types 60.

Approach decided: pyright, and port many files at once, then drive the error count to zero afterwards. Not file by file, and not gated in CI until the count is zero.

### Decisions to make first
- [x] Formatter: keep `ruff format` (decided).
- [x] Linter: keep `ruff` lint (decided). Still open inside it: whether to turn on extra rule sets (for example bugbear, pyupgrade, unused arguments) or keep the current set.
- [ ] How to annotate tensor shapes now that the ein strings are invalid types. Options: `jaxtyping` (new dependency), plain `Tensor` plus comments, or type aliases (`Float = Annotated[Tensor, ...]`). Pick one before the bulk pass, since it touches most of `model/`. Tested 2026-10-31 with jaxtyping 0.3.11 on numpy arrays (torch could not be installed in the container, so torch itself is untested): pyright reads `Float[np.ndarray, "b n d"]` as a plain `ndarray`. It checks the array class but ignores dtype and shape, so wrong shapes are not caught statically. Ruff reports F722 on shape strings and F821 on single-name dims like `"b"`, so the `F722`/`F821` ignore the repo uses today stays needed (set it per file in `ruff.toml`). Runtime shape checking is optional (beartype or typeguard import hook) and off by default.
- [ ] Pyright mode to end at: `basic`, `standard` or `strict`, and for which paths. Strict only for new code (web-port tooling) is one option.
- [ ] Which paths are out of scope: `src/f5_tts/runtime/triton_trtllm/` (TensorRT-LLM APIs), `src/third_party/`, notebooks or data prep scripts.
- [ ] Policy on suppressions: only `# pyright: ignore[ruleName]` with the rule name, never bare `# type: ignore`. Comments that are temporary start with `CLAUDENOTE:` per `CLAUDE.md`.

### Setup
- [ ] Make a reproducible environment with torch, torchaudio and the other runtime dependencies installed, so errors are real and not missing-import noise. Record how in `docs/debugging.md`.
- [ ] Pin the pyright version (pre-commit, dev dependency, or both). Pyright updates often add errors.
- [ ] Add `[tool.pyright]` to `pyproject.toml`: python version 3.10, include/exclude, start in `basic` mode, report missing imports as errors only after the environment is fixed.
- [ ] Record the real baseline count with torch installed, by rule. This replaces the 672 figure.

### Bulk conversion (many files at once)
- [ ] Group files into a few batches by area: `model/` (cfm, backbones, modules, utils, dataset, trainer), `infer/` plus `api.py` and the socket files, `train/`, `eval/`, `scripts/`.
- [ ] Apply the shape-annotation decision across `model/` in one pass.
- [ ] Add parameter and return annotations across each batch without stopping to fix every error on the way. Behaviour must not change; run the existing inference path before and after (golden output from plan 00's harness, or a fixed-seed run).
- [ ] Commit per batch so a bad batch can be reverted alone.

### Drive errors to zero
- [ ] Burn down by rule category, biggest first (possibly-unbound, attribute access, argument types, then the rest), fixing real bugs separately from annotation fixes.
- [ ] Any real bug found goes into `docs/debugging.md` with its fix.
- [ ] Review every suppression added; each needs a rule name and a reason.
- [ ] Reach zero errors in the chosen mode and paths.

### Lock it in
- [ ] Add pyright to pre-commit and to the GitHub workflow, failing on errors.
- [ ] Add the chosen formatter and linter settings to `pyproject.toml` or `ruff.toml` and run them across the repo once, as a single formatting commit.
- [ ] Update `CLAUDE.md` with the commands to run (type check, lint, format) and the suppression policy.
- [ ] Decide when to ratchet strictness (for example `standard` then `strict` per directory).

# WebGPU port

Goal as above. Plans 00 to 07 below.

## Starting assumptions

These are decisions to confirm in plan 00 before anything else builds on them.

- Model scope: `F5TTS_v1_Base` only (DiT, Vocos, 24 kHz, 100 mels, hop 256). Other configs later.
- Runtime: ONNX Runtime Web with the WebGPU execution provider. Hand-written WGSL only for ops it cannot do.
- ODE loop, chunking, cross-fade and text front end live in JS/TS.
- Language scope: English and Chinese, as the pretrained model supports.
- No in-browser ASR in the first version; the user supplies the reference text.
- App code stays MIT for now. Revisit the app licence when the weights are published (decided).
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
- [ ] Keep weights out of the app repo and its licence: the app downloads them from the Hugging Face repo, and the app README states the weights carry their own licence.
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
- Exact attribution wording for the model card. Gate for plan 05: nothing is published until this is settled.
- App licence at publish time. MIT code with separately licensed CC-BY-NC weights is how upstream does it, so the code may not need to change as long as the weights are downloaded separately and not bundled in the app repo. If the app is hosted as a product and non-commercial use is intended, decide whether the app should say so. Revisit when plan 05 publishes.
- Which browsers are in scope.
- Whether the Python package should gain a `webgpu/` export script entry point or this stays a separate project.
