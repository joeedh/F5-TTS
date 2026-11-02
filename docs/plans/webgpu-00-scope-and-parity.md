# Plan 00: scope, runtime decision and parity harness

Status: `plan pressure-tested` (drafted and reviewed 2026-11-02; findings folded in, see the end of this file). D1 and D2 accepted by the user on 2026-11-02; step 1 is done. Depends on: nothing. Blocks: plans 01, 02, 03 (and through them everything else).

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). Model details: [../architecture.md](../architecture.md).

## What this plan delivers

- Confirmed scope: the starting assumptions in the task list, accepted or changed.
- An answer to the parity question: what "the port is correct" means, stage by stage.
- A runtime decision, ONNX Runtime Web (ORT Web) or hand-written WGSL, backed by a measured spike on this laptop.
- A repo layout for the web code and the Python export tools, with working build, test and CI scaffolding.
- A golden-data generator: PyTorch runs with fixed inputs and noise that dump every intermediate a port stage needs to check itself.
- A tolerance table, derived from measured noise floors rather than guessed.
- A parity runner (TypeScript, with a Python twin) that compares any port stage to the golden data and reports the first stage that fails.

Out of scope: exporting the full DiT (plan 01), mel and vocoder ports (02), text front end (03), sampler (04), quantization (05). The spike exports one block and a few ops only, and is frozen after the decision.

## Machines

- **Laptop (primary, from 2026-11-02).** Windows 11. NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, driver 591.49 (CUDA 13.1), compute capability 12 (Blackwell), plus Intel integrated graphics. Chrome 156 and Edge 156 installed. Node 24.22.1, npm 11.19.1. Python 3.15.1 only; `.venv` has torch 2.15.0+cpu.
- **Cloud container.** No GPU; Chromium WebGPU runs on SwiftShader (CPU). Fine for correctness tests at small shapes, useless for timing. See [../research/webgpu-environment.md](../research/webgpu-environment.md).
- Package and tool facts checked 2026-11-02:
  - `onnx` has an abi3 wheel that installs on Python 3.15. `onnxscript` and `onnxconverter-common` are pure Python.
  - `onnxruntime` has **no** Python 3.15 wheel. CUDA torch for Python 3.15 exists only as 2.14.1+cu130 (no 2.15). `torchaudio.load` does not work on Python 3.15 either: torchcodec has no cp315 wheel, so pip installs a placeholder.
  - So there are two venvs (details in `docs/debugging.md`, 2026-11-02):
    - `.venv` (Python 3.15, torch 2.15.0+cpu): type checking, lint, behaviour guard, golden-data generation on CPU. Its torch build is what `tools/typing_guard/golden.json` and the CI pin expect, so it is not changed.
    - `.venv313` (Python 3.13.17, made and checked 2026-11-02): torch 2.15.0+cu132, working on the RTX 5050; `onnxruntime` 1.31.1 (CPU); torchcodec 0.17.0; `onnx`, `onnxscript`, `onnxconverter-common`, `soundfile`; `f5-tts` installed editable. Used for the CUDA runs in step 5 and for ONNX checks in Python.
  - The generator loads audio with `soundfile` anyway, so it runs in either venv.
  - pydub reads and writes `.wav` without FFmpeg, so `preprocess_ref_audio_text` should work on the wav references. FFmpeg 8.1 is available from `C:\dev\blender\main\lib\windows_x64\ffmpeg\lib`, with `aom.dll` from `...\windows_x64\aom\bin`. Python code that loads it through torchcodec must register both folders with `os.add_dll_directory`.

## Decisions

Each has a proposal. Items marked **(user)** need the user's answer before step 2 starts; the rest are settled by the steps below and recorded here.

### D1. Starting assumptions (decided 2026-11-02, as proposed)

Proposal: keep all of them as written in the task list. Two additions:
- Target browsers for v1: Chrome and Edge on desktop (both are on the laptop). Safari and Firefox are tested in plan 07 but do not block v1. This answers the task list's "which browsers" question.
- Target GPUs for v1: one discrete (the RTX 5050) and one integrated (the laptop's Intel GPU). The integrated GPU stands in for ordinary hardware.

### D2. What parity means (decided 2026-11-02, as proposed)

This is the task list's open question "exact numerical match, or only equal audio quality?". Proposal, a split answer:
- **Per stage, numerical.** Each stage of the port is fed the golden inputs for that stage (teacher forcing), and its output must be within the stage tolerance. This catches bugs where they happen, and does not let error pile up across 32 ODE steps.
- **End to end, quality.** With the golden noise injected, the free-running port's final mel must stay within a loose numeric bound (it will drift, especially in fp16), and the audio must sound as good as PyTorch's. In this plan that is judged by listening and by the mel bound. A metric-based comparison (WER, speaker similarity) over a set large enough to mean something belongs to plans 04 and 07; five short utterances are too few for a WER threshold (one word in 20 is 5 points).
- Bit-exact or sample-exact audio is not a goal.
- The port does not have to reproduce PyTorch's random numbers. Noise is an input: tests inject the golden noise, real use draws its own.

### D3. Runtime: ORT Web or hand-written WGSL

Settled by the spike in step 3. Default is ORT Web. Criteria, measured on the RTX 5050 in Chrome:
1. **Placement.** No compute op of the block runs on the CPU, and no activation tensor is copied from GPU to CPU inside the block. Shape-only nodes (Shape, Gather or Reshape on shape tensors) placed on the CPU by ORT on purpose do not count.
2. **Correctness.** fp32 output matches PyTorch fp32 within relative L2 1e-4 (replaced by the step 5 DiT tolerance once it exists).
3. **Speed.** Projected real-time factor (generation time divided by seconds of new audio) of at most 5 in fp32 for a 1024-frame chunk (about 10.9 s in total, about 5.7 s of it new audio after a 5.2 s reference). Projection: run the block session 22 times in a row with inputs and outputs kept on the GPU (`preferredOutputLocation: "gpu-buffer"`), so per-run overhead is included, then multiply by 32 steps. RTF 5 is the "usable but slow" bar; the fp16 RTF is recorded as the number plan 05 tries to improve.
4. **fp16.** An fp16 version of the block runs, if the adapter has `shader-f16`.
5. **Integrated GPU.** On the Intel GPU the block runs and gives correct output. Time is recorded, not judged.

How the result is read:
- All of 1 to 3 pass: ORT Web.
- ORT Web passes 1 and 2, but its MatMul is more than 2 times slower than the hand-written WGSL matmul at the same shape: still ORT Web, and plan 07 gets a task to look at custom kernels for the hot path.
- 1 fails on one or two ops: split the graph around them and run those ops as hand-written WGSL between ORT sessions (ORT Web's WebGPU provider cannot load custom ops).
- 2 fails, or 3 fails by more than 3 times while the hand-written matmul shows the hardware could do it: a hand-written WGSL DiT, which would need its own plan.

### D4. Repo layout

Proposal: web code lives in this repo, not a separate one. The golden generator and export scripts are Python in this repo, and parity tests cross both sides; one commit can change an export and its test. Revisit if the web side grows its own release cycle.

```
web/                     npm package (package.json, tsconfig.json, vite.config.ts)
  src/                   library code; subfolders added by plans 02 to 06
  test/                  Vitest tests that run in Node
  test/browser/          Playwright tests that need WebGPU
  test/parity/           the parity runner (step 6)
  spikes/ort-block/      the step 3 spike; frozen after the decision
tools/webgpu/            Python: golden generator, export scripts, parity twin
testdata/webgpu/<case>/  golden data; only the tiny case is committed
```

- Language and tools: TypeScript (strict), Vite for dev server and bundling, Vitest for Node tests, Playwright for browser tests. Formatting with Prettier. ESLint is not added now; `tsc --noEmit` is the type gate.
- Not `tests/`: that name is in `.gitignore`.
- Python under `tools/webgpu/` follows Plan P: add `tools/webgpu` to `[tool.pyright] include` in `pyproject.toml`; ruff already covers `tools/`. Add a `webgpu` extra (`onnx`, `onnxscript`, `onnxconverter-common`, `soundfile`) and install it in `.github/workflows/typecheck.yaml`.
- Python ONNX Runtime: available in `.venv313` (CPU provider). Plan 01 can use it for export checks.
- Answers the task list's question "should the Python package gain a `webgpu/` export entry point": no, for now. Export tools stay in `tools/webgpu/` and are not shipped in the `f5_tts` package.

### D5. Golden data format

- One folder per case: `testdata/webgpu/<case>/manifest.json` plus one `.npy` file per tensor.
- `.npy` because Python writes it with numpy and the format is simple to parse in TypeScript (a header plus raw little-endian data). Integer tensors are saved as int32, so JavaScript does not need `BigInt64Array`.
- Every tensor is stored in the layout PyTorch has at that point; the manifest gives the shape and the axis names (for example `b d n` for a mel, which is (1, 100, frames)), so nobody has to guess.
- `manifest.json` holds: case name, git commit, Python version, torch version, OS, device, dtype, `torch.get_num_threads()`, all inference settings, the input strings, the scalar values listed below, and one entry per tensor (name, file, stage, shape, axis names, dtype, sha256). Strings and string lists (chunks, pinyin tokens) go in the manifest directly.
- Inputs that do not change between DiT calls (text ids, `step_cond`) are stored once per chunk, not once per call.
- The tiny case is committed. Real-checkpoint cases are about 40 MB per 1000-frame chunk (about 80 MB at 2048 frames) plus the DiT internals, and are gitignored; they are rebuilt with one command. `--dump-steps` limits which steps are stored if size becomes a problem.

## Golden data

### Cases

| Case | Model | Input | Settings | Committed |
|---|---|---|---|---|
| `tiny` | Small DiT and small Vocos with seeded random weights (see below) | first 1 s of `basic_ref_en.wav`, a short English sentence | nfe 8 | yes, with its weights as safetensors (a few MB) |
| `en-short` | `F5TTS_v1_Base`, Vocos | `basic_ref_en.wav` and its text, one English sentence | defaults: nfe 32, cfg 2.0, sway -1, speed 1 | no |
| `zh-short` | same | `basic_ref_zh.wav`, Chinese sentence with polyphones and some Latin letters | defaults | no |
| `en-multi` | same | long English text that splits into 3 chunks, one of them under 10 bytes (triggers the speed x0.3 rule) | defaults | no |
| `en-multi-fixdur` | same | as `en-multi` | `fix_duration` set (exercises the per-chunk split) | no |
| `en-prep` | same | `basic_ref_en.wav` made into a 44.1 kHz stereo file, looped past 12 s and scaled to RMS 0.03 (exercises downmix, resample, RMS raise and output rescale, 12 s clip) | defaults | no (the derived wav is made by the generator) |
| `en-epss16` | same | as `en-short` | nfe 16 (the only step count in the EPSS table; nfe 8 and 32 use linspace) | no |

- `tiny` exists so the harness runs in CI and on SwiftShader in seconds. It checks shapes, plumbing and op coverage, not real numeric ranges.
  - DiT: dim 128, depth 2, heads 2, dim_head 64, ff_mult 2, text_dim 64, conv_layers 1, the real vocab. Small Vocos: backbone dim 64, 2 layers, the real ISTFT head settings (n_fft 1024, hop 256).
  - `DiT.initialize_weights` sets the AdaLN linears, `norm_out` and `proj_out` to zero (`dit.py:280`), which would make every tiny DiT output exactly zero. After construction, every parameter is re-initialised from a fixed seed (normal, std 0.02).
  - Weights are committed because random init can change between torch versions.
- Real weights: `hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors` (about 1.35 GB) and `charactr/vocos-mel-24khz`. Neither is downloaded on the laptop yet.

### What is dumped, per case

Stage names are the keys the runner and tolerance table use. Shapes are for batch 1.

| Stage | Tensors and values | Used by plan |
|---|---|---|
| `input` | raw reference wav (copied), reference text, target text | all |
| `prep` | reference audio after `preprocess_ref_audio_text` (trim, clip), fixed-up reference text | 02, 03 |
| `text` | `max_chars`; chunk strings; reference text after the trailing-space rule in `infer_batch_process`; per chunk the joined `ref_text + gen_text`, its pinyin token list, token ids (int32, shape (1, nt)) | 03 |
| `audio` | reference rms before scaling and whether the raise to 0.1 fired; reference audio as passed to `CFM.sample` (mono, raised to RMS 0.1 only if quieter, 24 kHz), shape (1, samples) | 02 |
| `mel` | reference mel (1, 100, n_ref), where n_ref = samples // 256 + 1 because the STFT uses `center=True` | 02 |
| `plan` | per chunk: `local_speed`, duration in frames, `ref_audio_len` = samples // 256 (the crop index, one less than n_ref, so one reference frame survives the crop), time grid `t` (after sway) | 04 |
| `noise` | per chunk: `y0` (1, frames, 100) | 04 |
| `dit` | per chunk: `step_cond` (1, frames, 100) and text ids once; per step k: `x_k`, `t_k`, conditional and unconditional output (the packed CFG output split in two) | 01, 04 |
| `dit_internal` | per chunk: text embedding for cond and uncond (computed once per chunk at step 0, then cached); at steps 0 and 16 only: time embedding, `input_embed` output, output of each of the 22 blocks, `norm_out` output | 01 |
| `mel_out` | per chunk: the cropped mel passed to the vocoder, (1, 100, frames - ref_audio_len) | 04 |
| `vocoder` | per chunk: Vocos backbone output; `head.out` output (the linear layer before `exp`, the clip at 100 and cos/sin); wav after the output RMS rescale | 02 |
| `final` | cross-faded wav (`.npy` and a `.wav` for listening) | 04 |

### How the generator gets these without changing model code

- **Model loading.** Builds the `CFM` the same way `load_model` does, then calls `load_checkpoint(..., dtype=torch.float32)` directly and asserts the dtype. `load_model` has no dtype parameter and `load_checkpoint` picks fp16 on CUDA with compute capability 7 or above, which includes the RTX 5050.
- **Pipeline.** Calls `preprocess_ref_audio_text` with the reference text given (no ASR), loads its output with `soundfile`, repeats the two lines of `infer_process` that compute `max_chars` and call `chunk_text`, then calls the real `infer_batch_process((audio, sr), ...)`. Everything after the audio load is upstream code. Step 2 checks whether `preprocess_ref_audio_text` (pydub) works on the laptop without FFmpeg; if not, FFmpeg is installed and that is recorded.
- **Recording wrapper** around the DiT:
  - It is an `nn.Module` subclass (`CFM.transformer` is a registered submodule; assigning anything else raises `TypeError`).
  - It registers the real DiT as a child, so `next(model.parameters())` in `CFM.sample` still finds the weights.
  - It exposes `dim` and forwards `clear_cache()`, which `CFM.sample` calls after each chunk. Without that, the text-embedding cache would leak into the next chunk.
  - It records each call's `x` (this is `x_k`; the first call's `x` is `y0`) and `time`, and the packed CFG output. `cond` and `text` are recorded on the first call of each chunk only.
  - Forward hooks for `dit_internal` are registered on the real DiT's submodules (`text_embed`, `time_embed`, `input_embed`, each block, `norm_out`) before it is wrapped. After wrapping, `model.transformer.text_embed` would point at the wrapper.
  - The text-embedding cache is thread-local (`dit.py:255`); with the synchronous executor below, everything runs on one thread.
- **Other hooks.** A forward hook on `model.mel_spec` gives the `audio` stage (its input) and the `mel` stage (its output). Hooks on `vocoder.backbone` and `vocoder.head.out` give the `vocoder` stage, and the backbone hook's input is `mel_out`.
- **Ordering.** `utils_infer.ThreadPoolExecutor` is patched to a synchronous executor during the run, so chunks run in order and each recorded call belongs to a known chunk. See risk R1.
- **Determinism.** Runs once with `seed_everything(seed)` and a fixed `torch.set_num_threads`. A second run must give identical hashes; that is the generator's own test.
- **Command.** `python tools/webgpu/make_golden.py --case en-short` (or `--all`).
- **Run time.** At 1000 frames one step is about 1.4 TFLOP with the CFG pair, about 45 TFLOP per chunk: roughly 1.5 to 4 minutes per chunk on the laptop CPU in fp32, longer for 2048-frame chunks, doubled by the determinism rerun. The full set is roughly an hour.

## Tolerances

Method, applied per stage:
1. Measure the noise floor: the difference between two legitimate fp32 implementations of that stage. Here: PyTorch CPU fp32 against PyTorch CUDA fp32 with TF32 off (separate CUDA venv), both with teacher-forced inputs. The floor is the maximum over all cases, chunks and steps, not one sample. Integer and string stages are exact.
2. Tolerance = 10 times the measured floor, rounded up to a power of ten.
3. fp16 tolerances are measured the same way against PyTorch CUDA fp16, but are set in plan 05, not here.

Metrics, all reported for every float comparison: max absolute error, relative L2 error `||a - b|| / ||b||`, cosine similarity. The pass check uses relative L2 plus a max-abs ceiling. Where `||b||` is near zero (below 1e-6 times the tensor size), relative L2 is not used and max-abs decides.

Special cases:
- **Log mel.** Bins just above the 1e-5 clamp turn tiny absolute differences into large log differences. Compare only bins where the golden linear magnitude is above 1e-4, plus a separate check on the linear magnitude.
- **Vocoder head output.** The second half of `head.out` is phase. Phase error where the magnitude is tiny does not matter, so phase is compared as the complex STFT value (magnitude times cos/sin) by relative L2, not as raw phase.

Starting table. Every float row is a placeholder until step 5 replaces it with measured values.

| Stage | Check | Starting tolerance |
|---|---|---|
| `text` | `max_chars`, chunk strings, tokens, ids | exact |
| `plan` | `local_speed`, duration, `ref_audio_len` | exact |
| `plan` | time grid | max abs 1e-6 |
| `audio` | reference audio after downmix, RMS and resample | plan 02 decides; a different resampler may be accepted and judged at `mel` |
| `mel` | log mel from golden audio, masked as above | max abs 1e-3 |
| `dit` | one DiT call, golden inputs | relative L2 1e-4 |
| `vocoder` | complex STFT from golden mel | relative L2 1e-4 |
| `vocoder` | wav from golden mel | max abs 1e-3 |
| `mel_out` | free-running ODE with golden noise | relative L2 1e-2 |
| `final` | end-to-end | listening check against the golden wav; metric comparison deferred to plans 04 and 07 (see D2) |

- The free-running `mel_out` bound is the one most likely to move. If the measured floor is large, D2 already says end-to-end is judged by quality.

## Steps

### Step 1. Decisions D1 and D2 (user)
- Ask the user to confirm or change D1 and D2. Record the answers in this file and the task list.
- Done when: both are marked decided. (Done 2026-11-02: both accepted as proposed.)

### Step 2. Laptop environment and minimal scaffolding
- Port `tools/webgpu-probe/probe.cjs` to run on Windows: install `playwright` locally in `tools/webgpu-probe` (not globally) and launch with `channel: "chrome"` and `channel: "msedge"`.
- Reaching both GPUs: Chrome on Windows may ignore `powerPreference`. Try, in order: `powerPreference` in `requestAdapter`; the Chrome switch `--use-webgpu-power-preference=force-high-performance` or `force-low-power`; the Windows per-app graphics setting. Always record which adapter actually came back (vendor and description), not which one was asked for.
- Record adapters, limits, features (`shader-f16`, `subgroups`, `timestamp-query`) and the naive matmul time in a new `docs/research/webgpu-laptop.md`; add it to `docs/index.md`.
- Download the real checkpoint and Vocos into the Hugging Face cache.
- Check that `preprocess_ref_audio_text` runs on the laptop. pydub's wav reading is already confirmed to work without FFmpeg. If anything else needs FFmpeg, use the Blender copy (see Machines).
- Done 2026-11-02, before step 2 started: the `.venv313` venv with CUDA torch, onnxruntime and torchcodec. `torchaudio.load` works there with the Blender FFmpeg (see Machines).
- `pyproject.toml`: the `webgpu` extra; install it in `.venv`.
- Minimal `web/`: `package.json` with Vite, TypeScript, `onnxruntime-web` and Playwright; strict `tsconfig.json`; `.gitignore` entries for `web/node_modules/` and `web/dist/`. Enough for the spike; tests come in step 4.
- Install the `webgpu` extra into `.venv313` as well, so both venvs match.
- Done when: the research doc exists, both GPUs show up with WebGPU in Chrome, weights are cached, and `npm install` in `web/` works.

### Step 3. Runtime spike (decision D3)
- Before starting, read existing community ONNX exports of F5-TTS (for example `DakeQQ/F5-TTS-ONNX` on GitHub) for known export problems. Read only; do not depend on them.
- `tools/webgpu/spike_export.py`: build the DiT from `F5TTS_v1_Base.yaml`, load the real weights in fp32, and export to ONNX with `torch.onnx.export(dynamo=True)`:
  - DiT block 0: inputs `x` (2, n, 1024), time embedding (2, 1024), rotary frequencies precomputed in Python and passed as an input. Batch 2 is the CFG pair.
  - One `MatMul` alone at the shape of the feed-forward up-projection, (2n, 1024) x (1024, 2048). This is the op that dominates per-step cost (about 8.4M multiply-adds per token per block, 22 blocks), and the one the WGSL baseline is compared on.
  - `ConvPositionEmbedding` (grouped conv, kernel 31, groups 16, 1024 channels; runs every step) and one `ConvNeXtV2Block` (depthwise, kernel 7). The ConvNeXt block matters little for the DiT, where its output is cached per chunk, but it is the building block of the Vocos backbone, so it stands in for the vocoder.
  - One static export per n = 256, 1024, 2048 (2048 is about the longest chunk: 22 s at hop 256). Dynamic sequence length is plan 01's question.
  - fp16 copies made with `onnxconverter-common` (`convert_float_to_float16`, `keep_io_types=True`).
  - Record the opset and what attention became (a fused op, or matmul plus softmax).
  - Save inputs and PyTorch fp32 outputs as `.npy`.
- `web/spikes/ort-block/`: a Vite page that loads each model with `onnxruntime-web` on the WebGPU provider, feeds the inputs, compares with the saved outputs, and times 20 runs after 3 warm-up runs. Placement comes from ORT's verbose log of which provider each node landed on. The block is also run 22 times chained on GPU buffers for criterion 3.
- Hand-written WGSL baseline: a tiled matmul at the same shape as the exported `MatMul`, compared against ORT's. This shows how much speed ORT leaves on the table and what a hand-written kernel costs to write.
- Driven by a Playwright script so results come back as JSON; run on both GPUs, in Chrome and Edge.
- Write the results and the decision into `docs/research/webgpu-runtime-spike.md` (add to the index) and into D3 above.
- Done when: D3 is decided with numbers against each criterion.

### Step 4. Test scaffolding and golden generator
- `web/`: scripts `typecheck`, `test`, `test:browser`; one passing Vitest test and one Playwright test that only checks `navigator.gpu`.
- `.gitignore`: `testdata/webgpu/*` except `testdata/webgpu/tiny/`.
- `pyproject.toml`: `tools/webgpu` in the pyright include.
- `tools/webgpu/make_golden.py` as described under Golden data, plus `tools/webgpu/tiny_model.py` for the tiny case's models.
- Generate all cases. Check determinism (two runs, same hashes). Listen to each `final` wav.
- Done when: pyright and ruff pass on the new Python, all cases are generated, and the tiny case is committed.

### Step 5. Measure noise floors and fix tolerances
- `tools/webgpu/measure_floor.py`, run in `.venv313`: rerun each stage on CUDA fp32 (TF32 off) with golden inputs, compare with the CPU golden data, take the maximum over all cases, chunks and steps.
- Replace the placeholder rows in the tolerance table with the measured values. Store the table as `testdata/webgpu/tolerances.json`, which both runners read; this file is the single source.
- If CUDA stops working in `.venv313`, use two CPU runs with different thread counts as a weaker floor and say so.
- Done when: `tolerances.json` is committed and the table above matches it.

### Step 6. Parity runner
- `web/test/parity/`: `.npy` reader, `loadCase(dir)` (manifest plus lazily loaded tensors; reads from the filesystem in Node and over HTTP in the browser), `compare(stage, name, actual, golden)` against `tolerances.json` with the special cases above, and a report: one line per comparison, pass or fail, and the first failing stage printed last so it is easy to find.
- `tools/webgpu/parity.py`: the same metrics and tolerance lookup in Python, for plans 01 and 02 to check exports before anything runs in a browser.
- Self-tests:
  - Golden data compared with itself passes.
  - A deliberately perturbed tensor fails, with the right stage named.
  - A tensor in the wrong layout (for example a mel transposed to (1, n, 100)) fails with a shape error, not a numeric one.
  - Smoke test, not a proof (it passes by construction, since the tolerances come from it): the CUDA fp32 rerun from step 5 goes through the runner as if it were a port.
- Done when: the self-tests pass in Node, and the browser loader reads the tiny case in Chrome on the laptop and on SwiftShader in the cloud container.

### Step 7. CI
- New workflow `.github/workflows/web.yaml`: `npm ci`, `typecheck`, Vitest, and the Playwright test on the tiny case. GitHub's Ubuntu runners have no GPU, so this is expected to run on SwiftShader as in the cloud container; confirm on the first run.
- Python job (in `typecheck.yaml` or a new one): regenerate the tiny case on CPU and compare it with the committed one through `parity.py`. Compare with tolerances, not hashes: the committed case comes from Python 3.15 on Windows and CI runs Python 3.12 on Linux, so the last bits differ. This also flags upstream behaviour changes in the model code.
- Done when: both pass on GitHub.

### Step 8. Close out
- Update the task list: plan 00 status, the ticked tasks, the answered open questions (parity, browsers, export entry point).
- `docs/debugging.md` entries for anything found on the way (R1 at least).
- If D1 to D3 change the plan materially, pressure-test this file again.

## Risks and known issues

- **R1. Multi-chunk output is not reproducible upstream, even with a seed.** `infer_batch_process` runs chunks in a `ThreadPoolExecutor`, and each chunk draws its noise from the global torch RNG inside `CFM.sample`, so which chunk gets which noise depends on thread timing. The generator avoids this by running chunks in order and recording `y0`; the port gets the noise from the golden data. Worth a `docs/debugging.md` entry. Not fixed upstream by this plan.
- **R2. fp16 is chosen silently on CUDA.** See "Model loading" above. Golden data must be fp32 and say so in the manifest.
- **R3. Reference preprocessing uses pydub** (silence trimming, clipping). The browser cannot match that exactly. The golden data records `prep` separately from `audio`, so plan 02 can test from either point.
- **R4. CPU golden data depends on torch version, OS and thread count.** Hashes are only for the determinism check on one machine; all cross-machine comparisons use tolerances.
- **R5. The tiny model's numbers are not representative.** Random small weights do not show fp16 overflow or real attention patterns. It is for plumbing; numeric conclusions come from the real cases.
- **R6. Spike numbers are from one laptop.** Plan 07 owns the device matrix. If the RTX 5050 result is borderline, the integrated GPU will not be usable for full-length chunks, and plan 07 needs to know early.
- **R7. Dynamo ONNX export of `scaled_dot_product_attention`** may produce an op ORT Web does not support, depending on the opset. The spike finds out; plan 01 owns the fix.
- **R8. Python ONNX Runtime is not available on Python 3.15.** It is installed in `.venv313` instead.
- **R9. The generator repeats two lines of `infer_process`** (`max_chars`, `chunk_text`) to avoid `torchaudio.load`. If upstream changes them, the golden data drifts from upstream silently. The generator asserts that the two lines it copies still appear verbatim in `infer_process`'s source.

## What the next plans get

- Plan 01: block-level export experience from the spike, the `dit` and `dit_internal` golden data (per-block outputs to bisect with), the DiT tolerance, `parity.py`.
- Plan 02: `prep`, `audio`, `mel`, `vocoder` golden data and tolerances, including the `en-prep` case for resampling and RMS handling, and the browser loader.
- Plan 03: `text` golden data for the cases, including Chinese with polyphones. Plan 03 will need a much larger text corpus; this plan does not build it.
- Plan 04: `plan`, `noise`, `dit`, `mel_out`, `final` golden data, the off-by-one crop (`ref_audio_len` versus n_ref), the free-running tolerance, and the open task of a metric-based quality check.
- Plans 05 and 07: the measurement method for fp16 tolerances, the laptop GPU facts, and the spike timings.

## Pressure test, 2026-11-02

A fresh agent given only this file reviewed the first draft against the code. Findings and what changed:
- `infer_process` calls `torchaudio.load`, which does not work on the laptop: the generator now loads with `soundfile` and calls `infer_batch_process` directly (R9 guards the copied lines).
- The tiny DiT would output all zeros because of zero-initialised AdaLN and `proj_out`: parameters are now re-initialised; the runner handles near-zero references.
- The recording wrapper was under-specified (must be an `nn.Module`, register the DiT, expose `dim`, forward `clear_cache`, hooks registered before wrapping): spelled out.
- `load_model` cannot force fp32: the generator calls `load_checkpoint` with an explicit dtype.
- Values later plans need were missing (rms, crop index, `max_chars`, `local_speed`, joined text, `step_cond`, DiT internals): added; per-call duplicates removed.
- Mel layout was wrong ((1, 100, n), not (1, n, 100)); "RMS-normalised" was wrong; the vocoder tap and log-mel comparison needed care: fixed.
- No case exercised resampling, downmix, clipping, the short-chunk rule or `fix_duration`: `en-prep`, `en-multi` changes and `en-multi-fixdur` added.
- The spike needed packages and weights that came later: moved into step 2.
- D3 criteria rewritten: placement ignores shape-only ops, speed is measured on chained GPU-resident runs with a stated RTF bar, the WGSL baseline is a matmul (the cost that dominates), and there is a rule for reading the result.
- Tolerance floor is the maximum over all samples; the WER bar on five utterances was dropped; the CUDA-rerun self-test is now labelled a smoke test.
- `powerPreference` may be ignored on Windows: fallbacks listed, and the adapter actually returned is recorded.
- Run-time and size estimates raised.
