# Debugging guide

Running log. Add an entry for each bug or useful technique: symptom, cause, fix or diagnosis steps.

## 2026-11-24: loading weight variants in Chrome (plan 05 step 3)

- **A shard can need rewriting even when no tensor changes dtype.** `q4` failed with `Deserialize tensor val_2372_scales failed. Failed to load external data file ""step.data.3"", error: Out of bounds.`
  - Cause: in a widened graph, stored shards are packed at 64 bytes and the layout at 4096. `needsWidening` looked only at dtypes, so a shard of raw tensors was passed to ORT as stored.
  - It was caught only because the sizes differed. With equal sizes and different offsets, ORT would have read wrong weights silently.
  - The unit test followed the same wrong rule, so it passed.
  - Fix: compare sizes and every tensor's offsets too.
- **The memory sampler missed the peak of a short load.** It gave `fp32` a 1.11 GB peak while loading 1.4 GB of weights: 1 to 3 samples per load, about 1.5 s apart.
  - Fix: read `PeakPageFileUsage` from `Win32_Process`, which Windows tracks per process (`sample-memory.ps1`). To check it, compare with the largest live sample, which must not be higher.
- **Pages in one browser inflate each other's memory.** In a shared browser, `w8`'s renderer sampled 3.4 GB and the next variant's baseline was 1.8 GB. With a fresh browser per variant, `w8`'s peak is 1.8 GB. Measure each variant in its own browser.
- **`ort.env.webgpu.device` is not always a promise.** After a session exists, `ort.env.webgpu.device.catch` is not a function. Use `await`, which handles both.
- **`npx` outside `web/` downloads packages.** The repo root has no `package.json`, so `npx tsc` there fetched and ran an unrelated package of that name from the registry. Use `web/node_modules/.bin/`, or `npx --no-install`.

## 2026-11-23: onnxconverter-common's fp16 block list does less than it says (plan 05 step 2)

`convert_float_to_float16(..., node_block_list=[...], keep_io_types=True)` (onnxconverter-common 1.16.0) keeps the listed nodes' arithmetic in fp32, but not the values between them. Three effects, each found in the converted DiT step graph:

- **It rounds every float initializer**, even one that only blocked nodes read, and inserts a Cast back to fp32 in front of the reader. The time-embedding frequencies, the rope table and `freqs_cis` came out fp16-rounded behind fp32 nodes.
  - Found by checking the tables bit for bit against the fp32 export.
  - Fix: put the original initializer back and drop those Casts (`restore_tables` in `tools/webgpu/convert_precision.py`).
- **It casts every blocked node's output to fp16,** even when the only readers are other blocked nodes (which it then casts back) or graph outputs.
  - So `time * freq` was rounded to fp16 between two fp32 nodes, before `Sin` and `Cos`.
  - ORT CPU refused the graph: `Type parameter (T) of Optype (Mul) bound to different types (tensor(float) and tensor(float16)`.
  - Fix: blocked readers and casts back to fp32 read the fp32 value, and the fp16 Cast stays only for fp16 readers (`keep_fp32_between_blocked`).
- **It declared the outputs `v` and `x_next` fp16** despite `keep_io_types`, because a blocked node produced them: `Type (tensor(float16)) of output arg (v) of node (node_add_179) does not match expected type (tensor(float))`. Fix: restore the output types from the source graph.
- **To diagnose:** walk the producers of a node that must be fp32, and print each Cast's `to`. A Cast with `to` 10 (fp16) between two blocked nodes is this bug.
- **A related check on the same tooling.** `onnx.checker.check_model(path)` refuses external data files with more than one hard link ("indicating a potential hardlink attack"), so variant folders hold copies, not links.

## 2026-11-23: four traps while probing weights and storage (plan 05 step 1)

- **The memory sampler failed silently.** `web/bench/dit-load/sample-memory.ps1`, started from Node with `spawn("powershell.exe", ["-File", ...])`, printed nothing, so every memory figure came out `null`. On stderr: "running scripts is disabled on this system". The default execution policy blocks the script when it is not started from an interactive PowerShell. Fix: `-ExecutionPolicy Bypass` on that one process, which changes no system setting. `run.mjs` now forwards the sampler's stderr, so a failure shows.
- **ORT CPU runs "fp16" graphs in fp32.** A graph that casts to fp16, runs `MatMul`, `Gemm`, `Conv` or `MultiHeadAttention`, and casts back gave outputs that were not even fp16 values. ORT CPU has no fp16 kernel for those ops, so it inserts casts to fp32 and then removes the Cast pairs, even with `graph_optimization_level` set to `ORT_DISABLE_ALL`. To diagnose: check whether the outputs lie on the fp16 grid. Do not use ORT CPU as an fp16 reference; compute the reference in float64 (`tools/webgpu/probe_kernels.py`).
- **Playwright's `chromium.launch` gives an in-memory OPFS.** It reported a 10 GB quota, and a worker's `createSyncAccessHandle().write` of 64 MiB failed with `QuotaExceededError: No space available for this operation`. Use `launchPersistentContext` with a profile directory for any storage measurement. That reported a 10 GB quota too, with 137 GB free, so check a normal Chrome profile before relying on the figure.
- **"CPUExecutionProvider" in ORT's verbose log is not a placement.** With optimizations on, each session logs `GraphTransformer TransposeOptimizer_CPUExecutionProvider modified: 0`, which is the name of an optimizer pass. Only the lines after `Node placements` say where nodes run: `All nodes placed on [WebGpuExecutionProvider]` or a list per provider.

## 2026-11-23: running upstream's Seed-TTS eval on Windows with Python 3.13 (plan 04)

Five obstacles, each with its fix. `tools/webgpu/quality_score.py` handles them, apart from the environment variables.

- **faster-whisper against PyAV 19:** `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. faster-whisper 1.2.1 passes an argument PyAV 19 removed. Fix: `pip install av==16.1.0`, which accepts it. (Upstream pins faster-whisper 0.10.1, which needs `av==10.*`, with no wheel for Python 3.12 and newer.)
- **CTranslate2 needs CUDA 12:** `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`. faster-whisper's backend wants CUDA 12's cuBLAS and cuDNN 9, but `.venv313`'s torch ships CUDA 13. Fix: `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`, and register their `bin` folders with `os.add_dll_directory` and on `PATH` (`dll_dirs.register_cuda12`). torch keeps its own CUDA 13 libraries.
- **`get_seed_tts_test` reads `meta.lst` as cp1252:** `UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d`. Upstream's `open(metalst)` has no encoding, so on Windows the Chinese list fails. Fix without touching upstream: run with `PYTHONUTF8=1`.
- **torch.hub asks before trusting s3prl:** `EOFError: EOF when reading a line` from `_check_repo_is_trusted`, when the run has no console. Upstream's ECAPA-TDNN loads WavLM through `torch.hub.load("s3prl/s3prl", ...)`, which runs code from s3prl's main branch. Once trusted (by the user, 2026-11-23), add `s3prl_s3prl` to `~/.cache/torch/hub/trusted_list`.
- **s3prl's hubconf against torchaudio 2.9 and newer:** `AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'`, then `ModuleNotFoundError: No module named 'torchaudio.sox_effects'`. The hubconf imports every model s3prl has, and some call removed APIs at import time. Fix: `stub_torchaudio_for_s3prl()` makes `set_audio_backend` a no-op (backend selection no longer exists) and stubs `sox_effects` with a function that raises if called. WavLM uses neither. Other removed APIs (`info`, `load_wav`) are used only inside other models' functions, so they need nothing.
- **Same family, from plan 04's PyTorch generation:** upstream prints the texts, and a redirected stdout on Windows is cp1252. `quality_torch.py` reconfigures stdout to UTF-8, as `text_reference.py` does.

## 2026-11-22: ORT Web cannot create two WebGPU sessions at once (plan 04)

- **Symptom:** `Error: another WebGPU EP inference session is being created.` from `InferenceSession.create`, thrown at once, when the DiT and the vocoder were loaded together with `Promise.all`.
- **Cause:** the error is thrown from `webgpuRegisterDevice` in ORT Web's bundle, inside `InferenceSession.create`, so it is a guard on overlapping session creation. (Read from the stack trace, not from ORT's source.) Sessions created one after another run together without trouble, as plan 04's runs show.
- **Fix:** create sessions one after another (`web/test/pipeline/run-final.ts`). Anything that loads several models, such as plan 04's `TTS.create` and plan 06's loader, must do the same. Fetching the weights can still overlap; only the `create` calls must not.

## 2026-11-22: measuring GPU memory of a WebGPU page on Windows, and the DiT at n = 8192 (plan 04)

- **Problem:** WebGPU has no memory query, so a test cannot see how much GPU memory the page holds, and reading Chrome's task manager by hand does not fit a scripted run.
- **Technique:** Windows' `GPU Process Memory` performance counters give per-process `Dedicated Usage` (a discrete GPU) and `Shared Usage` (an integrated GPU, which uses system memory). Poll `Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage'` every 2 s, keep the instances whose `pid_<n>_` is a `chrome.exe` process, and take the maximum. The GPU process is the one that holds the memory.
- **Finding:** the DiT with `step_mha.onnx` peaks at about 3.2 GB at n = 4096 and the same at 8192, on both laptop GPUs (dedicated on the RTX 5050, shared on the Intel GPU).
- **Finding:** at n = 8192 a step takes 30 s on the RTX 5050 but 22 s on the Intel GPU. NVIDIA's step time grows 10 times from n = 4096 (2.9 s) to 8192, Intel's 3 times. Not diagnosed; handed to plan 07.
- **Finding:** the unfused-attention graph (`step.onnx`) fails at n = 8192 with ORT's `SafeIntOnOverflow ... Integer overflow`: its attention matrix, 2 x 16 heads x 8192^2 floats, is over 2^32 bytes. It is also 4 times slower than `step_mha.onnx` at 4096. Use `step_mha.onnx`.
- **How to reproduce:** `PW_GPU=nvidia npx playwright test -c playwright.local.config.ts large-n` from `web/`, with `LARGE_N`, `LARGE_N_STEPS` and `LARGE_N_STEP_FILE` to choose sizes, steps and graph.

## 2026-11-22: `vocab.txt` has CRLF line endings in a Windows checkout (plan 03)

- **Symptom (avoided):** a JS vocab loader that splits on `\n` keeps a `\r` on every entry. Every lookup misses, so every token id is 0, the unknown id. The tokens still look right.
- **Cause:** the file is stored with LF, but `core.autocrlf=true` checks it out with CRLF (`git ls-files --eol src/f5_tts/infer/examples/vocab.txt` gives `i/lf w/crlf`). Python never notices: text mode turns `\r\n` into `\n`, and `get_tokenizer` keys each line on `line[:-1]`.
- **Fix:** `parseVocab` in `web/src/text/tokens.ts` follows Python's text mode (`\r\n` and `\r` become `\n`, then `line[:-1]`), and a test checks that CRLF, LF and CR files give the same map. For generated data that must stay byte-identical, such as `web/src/text/pinyin_dict.txt`, `.gitattributes` marks the file `-text`.

## 2026-11-22: `\uXXXX` escapes written by an agent become the literal characters (plan 03)

- **Symptom:** source written as `"　"` or `/[-]/` was saved holding the characters themselves. Tests still passed, but the file held invisible private-use, combining and space characters that no reader could see. It happened three times in plan 03 (`strings.test.ts`, `pinyin.ts`, `pinyin_reference.py`). A `\\` typed into a shell command also lost a backslash once.
- **Cause:** Claude Code's file-writing tools, and sometimes its shell input, decode `\uXXXX` before writing. The 8-digit Python form `\U0002a6d6` survived.
- **Fix:** after writing code with escapes, scan for invisible characters (Unicode categories Cf, Co, Cn, Mn, Cc, Zs, Zl, Zp, other than space and tab), and re-escape what it finds with a script that builds `\` from `String.fromCharCode(92)`. In tests, `String.fromCodePoint(0x3000)` avoids the problem entirely.

## 2026-11-22: a Rust wasm build embeds the build machine's paths (plan 03, step 3)

- **Symptom:** `jieba.wasm` held `C:\Users\<name>\.cargo\...` 4 times, so committing it would publish the user name and make the build depend on where it ran.
- **Cause:** panic locations (`file:line`) of the dependencies stay in the binary even with `panic = "abort"` and `strip = true`.
- **Fix:** build with `RUSTFLAGS="--remap-path-prefix=<cargo home>=/cargo --remap-path-prefix=<crate dir>=/build"`. `tools/webgpu/jieba_wasm/build.py` does this, then fails if a drive path or the home folder is still in the wasm.
- **Diagnose:** `grep -a -o -E "[A-Za-z]:[\\/][^\"']{0,80}" file.wasm`.

## 2026-11-22: `TextDecoder` drops a U+FEFF at the start of what it decodes (plan 03, step 3)

- **Symptom:** the jieba wasm wrapper returned `['', 'leading', ...]` where rjieba returns `['\ufeff', 'leading', ...]`. The wasm itself was right.
- **Cause:** `TextDecoder` treats a leading U+FEFF as a byte order mark and removes it, unless it is built with `ignoreBOM: true`. Decoding each token separately hits this for any token that is a lone U+FEFF.
- **Fix:** `new TextDecoder("utf-8", { fatal: true, ignoreBOM: true })` in `web/src/text/segment.ts`. Decode any text a port must keep byte-exact that way.

## 2026-11-22: `jieba-wasm` 2.4.0 does not segment like rjieba (plan 03, step 3)

- **Symptom:** `jieba-wasm` splits `x-ray`, `v1.2.3` and `Wi-Fi` where rjieba keeps them whole. 14 of 1,711 Seed-TTS English texts differ, which changes F5-TTS tokens through the space rule (`x-ray` becomes `x- ray`).
- **Cause:** `jieba-wasm` embeds a fork of jieba-rs 0.7.0. rjieba 0.2.2 embeds jieba-rs 0.11.0 (see its SBOM in `rjieba-*.dist-info/sboms`), whose `skip_match_len` joins alphanumeric runs across `.`, `_` and `-`.
- **Fix:** our own wasm build of jieba-rs 0.11.0, `tools/webgpu/jieba_wasm/`. When upstream moves rjieba, rebuild at the version its SBOM names.

## 2026-11-21: `torch.onnx.export` fails with `UnicodeEncodeError` on Windows (plan 02, step 3)

- **Symptom:** the export fails at the end with `UnicodeEncodeError: 'charmap' codec can't encode character '✅'`, raised from a `print("[torch.onnx]", ...)`.
- **Cause:** the dynamo exporter prints a check-mark emoji. When stdout is a pipe on Windows (as in the Bash tool, or `| tee`), Python encodes it as cp1252, which has no such character.
- **Fix:** set `PYTHONIOENCODING=utf-8`, or reconfigure stdout and stderr to UTF-8 at start-up, as `tools/webgpu/export_vocos.py` does.

## 2026-11-21: a silent reference gives upstream an all-NaN reference (plan 02, step 2)

- **Symptom:** `export_audio_fixtures.py`'s `silence-only` fixture (1.5 s of zeros) prepares fine, to pydub's 50 ms of silence at 11025 Hz. But its reference after `infer_batch_process` is all NaN.
- **Cause:** the rms is 0, which is below 0.1, so upstream scales by `target_rms / rms` (`utils_infer.py:479`), a division by zero.
- **Not fixed here:** the port reproduces upstream, NaN included. Plan 06 should refuse a reference with no sound before it gets this far.

## 2026-11-21: a float64 mel filterbank is 10 times further from the golden mel than torchaudio's (plan 02, step 1)

- **Symptom:** the TypeScript mel passed, but `mel.linear` sat at rel L2 1.9e-6, 10 times the CPU-against-CUDA floor of 1.9e-7. The FFT and window are float64, so they were expected to land at the floor.
- **Cause:** torchaudio builds the mel filterbank in float32 (`torch.linspace`, then `10 ** x`, then the triangle slopes). The same formula in float64 is up to 9.0e-6 away from torchaudio's matrix, and the mel follows the matrix, not the exact maths.
- **Fix:** `melFilterbank` in `web/src/audio/mel.ts` rounds every step with `Math.fround`, and fills `linspace` from both ends as torch's CPU kernel does. That brings `mel.linear` to 4.2e-7. The remaining gap is torch's vectorised float32 `pow`, which differs from a correctly rounded one at 5 of 102 points; torch's own scalar `pow` gives the same 5 differences.
- **Diagnose:** run the port with the reference's own constant (here torchaudio's matrix, exported as `testdata/webgpu/audio-fixtures/mel_fbank.npy`). If the error drops to the floor, the constant is the problem, not the algorithm. Then compare each intermediate in numpy float32 to find the step that rounds differently.

## 2026-11-19: ORT Web graph capture fails with a `createBindGroup` error (plan 01, step 4)

- **Symptom:** with `enableGraphCapture: true`, the first `session.run()` throws `Failed to execute 'createBindGroup' on 'GPUDevice': ... Failed to read the 'buffer' property from 'GPUBufferBinding': Required member is undefined.` Seen in ORT Web 1.31.1, Chrome and Edge, NVIDIA and Intel.
- **Where:** every `mha` step graph (tiny and base), and the base unfused step graph with static n, even at n = 64. Only the tiny unfused graph works.
- **Not:** the rotary-table `Gather`. Changing its layout did not help, and the change was reverted.
- **Related:** passing preallocated `fetches` with capture on fails first, with `BufferManager::MemCpy ... copy_size <= src_size && copy_size <= dst_size was false`. Leave the outputs to ORT when capture is on.
- **Status:** unresolved; capture is not used (`docs/research/webgpu-dit-export.md`). To retry with a newer ORT: `node bench/dit-step/run.mjs --gpu nvidia --only unfused-static-1024-capture` from `web/`, or `--graph <file> --n 64 --capture` for one graph.

## 2026-11-19: rotary cos and sin lose precision on the NVIDIA GPU (plan 01, step 3)

- **Symptom:** the DiT step graph on WebGPU failed `dit.v` and `dit.out` narrowly (relative L2 1.1e-4 to 1.4e-4 against 1e-4) on the RTX 5050, at steps 0 to 4 of a chunk only. The Intel GPU passed. The median error was 2 to 10 percent of the tolerance.
- **Not:**
  - A uniformly noisier backend: at the failing steps the error was 30 to 40 times the CUDA floor at the same steps, elsewhere about 2 times.
  - The attention kernel: fused (`mha`) and unfused attention gave identical errors to three digits.
  - LayerNorm's one-pass variance: every row's mean was under 0.15 of its spread.
- **Cause:** rotary embedding computed `cos` and `sin` of angles up to about 1,900 radians (position times frequency) in the graph. The NVIDIA GPU's shader `cos`/`sin` are inaccurate for arguments that large: Q and K after rotary were 5.8e-5 off where the projections were 4.9e-7. Where attention is sharp (logits up to 294, top two scores 0.22 apart; block 4 at step 0), that error grows to 4.4e-4 at the attention output, and later blocks carry it.
- **Fix:** `export_dit.py` precomputes the cos and sin tables in float64 (`rope_tables`) and swaps upstream's `_apply_rope` for `rope_from_tables` during export only (`precomputed_rope`). The graph keeps one `Cos` and one `Sin`, for the time embedding, whose arguments stay below 1,000. Block 4's output error on NVIDIA went from 1.3e-4 to 1.8e-6 (Intel 1.75e-6, CPU 1.2e-6).
- **Diagnose:**
  - Run the browser check with the debug graph (`DIT_DEBUG=1` in `test/browser-local/dit.spec.ts`) to find the first failing block.
  - Then `tools/webgpu/probe_block.py --block N` exports that block with every intermediate as an output. `web/bench/block-probe/run.mjs --gpu nvidia|intel` compares them with a float64 reference.
  - Any in-graph `Sin`, `Cos` or `Exp` of a large argument is a suspect on this GPU.

## 2026-11-19: pyright passes locally but fails in CI on onnxruntime code

- **Symptom:** `pyright` reports 0 errors locally, but the typecheck job fails on `check_dit_ort.py` (`SparseTensor` has no `__getitem__`, and similar).
- **Cause:** the local `.venv` (Python 3.15) has no onnxruntime wheel, so the import is unresolved and its types are `Unknown`. CI installs it, and `InferenceSession.run` is typed to return dense arrays, sparse tensors, lists or maps.
- **Fix:** cast the result in one helper (`check_dit_ort.run`).
- **Diagnose:** also type check code that uses onnxruntime against `.venv313`: `.venv\Scripts\pyright.exe --pythonpath .venv313\Scripts\python.exe <files>`.

## 2026-11-19: exporting the DiT with a dynamic sequence length (plan 01, step 1)

Three problems with `torch.onnx.export(dynamo=True)` (torch 2.15, onnxscript 0.7.2), all handled in `tools/webgpu/export_dit.py`:

- **`TextEmbedding` fixes n.**
  - **Symptom:** the exported text graph's `n` input dimension is a fixed number, not the `Dim("n")` that was declared.
  - **Cause:** `TextEmbedding.forward` calls `int(seq_len)` (`dit.py:108`), which turns the symbolic size into a constant.
  - **Fix:** `TextGraph` calls the submodules (`text_embed`, the `freqs_cis` slice, `text_blocks`) directly with the same masking. The export asserts that n is symbolic, so a regression fails loudly.
- **`com.microsoft::MultiHeadAttention` has no schema in the `onnx` package.**
  - **Symptom:** onnxscript refuses to build the op from a `custom_translation_table` entry, because `onnx.defs` does not know it.
  - **Fix:** build an `onnx.defs.OpSchema` by hand for the inputs and attributes used, and wrap it in `onnxscript.values.Op` (`msft_mha_op`).
- **Opset 23 `Attention` returns a tuple.**
  - **Symptom:** the translation function fails when its result is used as a tensor.
  - **Fix:** take `[0]`; the other outputs are the optional present key and value.
- **Diagnose:** `export_dit.py` prints each graph's op counts. Check that the fused variants have one attention op per block (22) and no `Softmax`.

## 2026-11-19: onnxruntime CPU check of the real DiT stopped for low memory (plan 01, step 1)

- **Symptom:** `check_dit_ort.py --debug` on the base model was killed for low memory, with about 15 GB free on a 32 GB machine.
- **Cause:** it held the step graph and its debug twin in two sessions. Each carries 1.27 GB of weights, and ORT prepacks weights into a further copy.
- **Fix:** with `--debug` only the debug graph is loaded and used for every step; its first three outputs are the normal graph's. Prepacking is disabled (`session.disable_prepacking`). The full 7-case run then completed.
- **Also:** run the floor measurement, the exports and the checks one at a time, not concurrently.

## 2026-11-18: each chunk's mel starts with the reference's last frame (plan 01, step 1)

- **Symptom:** the Euler update after the last ODE step matched every generated frame of the golden `mel_out` exactly, except frame 0, which was off by up to 14 (tiny case).
- **Cause:** an off-by-one between two upstream lengths.
  - The reference mel has `samples // 256 + 1` frames (`center=True`). `CFM.sample` restores that many frames from `cond` after the ODE (`cfm.py:235`).
  - `infer_batch_process` crops `ref_audio_len = samples // 256` frames (`utils_infer.py:512, 534`).
  - So frame 0 of every chunk's `mel_out` is the reference's last mel frame, not generated audio. The vocoder renders it as about 10 ms of reference audio at the start of each chunk.
- **Not a bug to fix here.** The port must reproduce it to match upstream; plan 04 owns this.
- **Diagnose:** compare the generated part per frame. `parity.x_next_golden` encodes the right offset (`n_ref - ref_audio_len`).

## 2026-11-18: multi-chunk inference is not reproducible with a fixed seed (plan 00, risk R1)

- **Symptom:** `infer_batch_process` with several text chunks gives different audio from run to run, even after `seed_everything(0)`. Measured on CUDA fp32 with the `en-multi` golden inputs (3 chunks): 6 runs gave 4 different output hashes. The length was the same every time, because durations do not depend on the noise.
- **Control:** the single-chunk `en-short` gave the same hash in all 4 runs on the same GPU, so the variation does not come from the CUDA kernels.
- **Cause:** `infer_batch_process` submits every chunk to a `ThreadPoolExecutor` at once. Each chunk draws its noise `y0` with `torch.randn` from the global generator inside `CFM.sample`, so which chunk gets which noise depends on thread timing. `CFM.sample`'s `seed` argument would not help: it calls `torch.manual_seed`, which seeds the same global generator, so parallel chunks would still race. A per-chunk `torch.Generator` would fix it.
- **Workaround used:** `tools/webgpu/make_golden.py` swaps in a synchronous executor during the run, so chunks run in order on one thread. It also records each chunk's `y0`, which the port takes as an input. Upstream is not changed.
- **Diagnose:** hash the output of a few repeated runs. If only multi-chunk inputs vary, this is the cause.

## 2026-11-18: .npy files saved from permuted tensors (plan 00, step 6)

- **Symptom:** the TypeScript parity reader throws `fortran-order .npy is not supported` on some files written by `measure_floor.py --dump`.
- **Cause:** `np.save` of a tensor that was permuted on the torch side (for example `mel_out`, a `permute(0, 2, 1)` view) writes the array in Fortran order, with `'fortran_order': True` in the header, rather than copying it to C order. Reshaping does not always force a copy.
- **Fix:** wrap anything saved for the web side in `np.ascontiguousarray`. `make_golden.py` already did this; the dump path now does too. The readers reject Fortran order on purpose, rather than silently reading the data transposed.

## 2026-11-18: recording intermediates without changing model code (plan 00, step 4)

- `tools/webgpu/make_golden.py` shows how to take every intermediate out of the real inference path. The same approach works for a one-off debugging dump:
  - Forward hooks on submodules; `register_forward_pre_hook(..., with_kwargs=True)` for the DiT, which `CFM.sample` calls with keyword arguments only.
  - Temporary patches of module globals: `f5_tts.model.cfm.odeint`, `utils_infer.convert_char_to_pinyin`, `utils_infer.ThreadPoolExecutor`. These work because the callers look the names up in their own module's globals at call time.
  - A synchronous stand-in for `ThreadPoolExecutor`, so that multi-chunk runs are reproducible (risk R1 in plan 00).
- With that stand-in, a fixed seed and 8 threads, a CPU fp32 run is bit-identical between two runs in one process, for all 7 golden cases.
- **PowerShell 5.1 `Get-Content` / `Set-Content` corrupt UTF-8 source files.** `Get-Content` without `-Encoding UTF8` reads UTF-8 as the ANSI code page. `Set-Content -Encoding ascii` then writes every non-ASCII character as `?`. This wiped the Chinese test strings in `make_golden.py` once; they were caught and restored before use. Edit such files with an editor tool, not a PowerShell round trip. `Set-Content -Encoding utf8` adds a BOM, which also shows up as a diff on line 1.

## 2026-11-18: onnxruntime-web on WebGPU (plan 00, step 3)

- **ORT ignores a `GPUDevice` you give it.** In onnxruntime-web 1.31.1, setting `ort.env.webgpu.device = myDevice` before the first session has no effect. ORT creates its own device and writes it back to `env.webgpu.device`.
  - Symptom: inputs made with `ort.Tensor.fromGpuBuffer` on your own device fail in `session.run` with `WebGPU error scope failed ... [Buffer (unlabeled)] is associated with [Device], and cannot be used with [Device]`, inside `CreateBindGroup` of the first kernel.
  - Fix: create a session first, then `const device = await ort.env.webgpu.device`, and make every shared buffer on that device. `web/spikes/ort-block/main.ts` (`ortDevice`) does this.
- **ORT destroys its device when the last session is released.**
  - Symptom: after `session.release()` on the only open session, the next run fails with `[Invalid Buffer (unlabeled)] ...`, and your own WebGPU work fails with `The Device was lost before mapping was resolved`.
  - Fix: keep one session open for as long as the device is needed.
- **Checking node placement.** Create the session with `logSeverityLevel: 0` and set `ort.env.logLevel = "verbose"`. ORT then logs `Node placements` followed by `All nodes placed on [WebGpuExecutionProvider]. Number of nodes: N`, or a list per provider. `GraphTransformer MemcpyTransformer modified: 0` means no GPU/CPU copy nodes were inserted.
  - The verbose log is about 7 MB for 24 sessions.
  - Vite 8 forwards the page console to the terminal by default. Pass `server: { forwardConsole: false }` when driving the page from a script.
- **fp16 models are much less accurate on the WebGPU EP than on the CPU EP.** The same `convert_float_to_float16` model of one (2048, 1024) x (1024, 2048) MatMul gives relative L2 4.3e-3 on WebGPU and 1.5e-4 on ORT CPU. This looks like fp16 accumulation (not checked in ORT's source).
  - When comparing an fp16 port against PyTorch, check the plain MatMul first, before suspecting your own graph.
  - Numbers are in `docs/research/webgpu-runtime-spike.md`.

## 2026-11-02: audio loading and FFmpeg on the Windows laptop

- **`torchaudio.load` fails because torchcodec has no Python 3.15 wheel, not because FFmpeg is missing.** torchaudio 2.11.1 loads through torchcodec. `pip` found no torchcodec wheel for cp315 and installed `torchcodec 0.0.0.dev0`, a placeholder whose `__init__.py` only prints `Stay tuned...`, so `torchaudio.load` raises `ImportError: TorchCodec is required for load_with_torchcodec`. That is the same error with or without FFmpeg on PATH. A real wheel exists for Python 3.13 on Windows (`torchcodec-0.17.0-cp313-cp313-win_amd64.whl`). This corrects the earlier note above, which blamed "torchcodec with FFmpeg".
  - Diagnose: `pip show -f torchcodec`. A real install has native libraries; the placeholder has only `__init__.py`.
  - Workarounds: read audio with `soundfile`, or use the Python 3.13 venv below, where `torchaudio.load` works.
- **pydub reads and writes `.wav` without FFmpeg.** `AudioSegment.from_file` on a wav uses its own reader and only warns `Couldn't find ffmpeg`. Checked on `basic_ref_en.wav` (5333 ms, 24 kHz). Other formats (mp3, flac, m4a) need FFmpeg.
- **FFmpeg from a Blender checkout.** `C:\dev\blender\main\lib\windows_x64\ffmpeg\lib` has `ffmpeg.exe`, `ffprobe.exe` (FFmpeg 8.1) and the `av*-62` DLLs. On its own, `ffmpeg.exe` exits with `0xC0000135` (DLL not found). The one missing dependency is `aom.dll` (imported by `avcodec-62.dll`), which is in `C:\dev\blender\main\lib\windows_x64\aom\bin`. With both folders on PATH, `ffmpeg.exe` runs. The `blender\blender` and `blender\fork` checkouts have only `ffprobe.exe` and FFmpeg 6 (`av*-60`) DLLs.
  - Diagnose a `0xC0000135` exit by listing the PE imports of the exe and its DLLs, and checking which are missing from the folder and System32. There is no `dumpbin` here; reading the import table with `struct` takes about 40 lines of Python.
- **Python 3.13 venv `.venv313` (made 2026-11-02)** for what Python 3.15 lacks: CUDA torch, `onnxruntime`, a real torchcodec.
  - Created with `py install 3.13` (Python install manager; gave 3.13.17, and 3.15 stays the default `py`), then `py -V:3.13 -m venv .venv313`. The venv writes its own `.gitignore`, so git does not see it.
  - `pip install torch==2.15.0 torchaudio --index-url https://download.pytorch.org/whl/cu132`. torch 2.15.0 is only on the cu132 index (cu130 stops at 2.14.1, cu128 at 2.11.0). It runs on driver 591.49, which reports CUDA 13.1, and includes sm_120 code for the RTX 5050. A 4096x4096 matmul runs at 7.6 TFLOPS in fp32 with TF32 off (relative error 1e-6 against fp64) and 25.4 TFLOPS in fp16 (relative error 4e-4).
  - Then `pip install -e . onnxruntime torchcodec onnx onnxscript onnxconverter-common soundfile`. pip keeps the CUDA torch. Versions: onnxruntime 1.31.1 (CPU provider only), torchcodec 0.17.0, onnx 1.23.2, onnxscript 0.7.2.
  - `torchaudio.load` works there only after both Blender folders are registered with `os.add_dll_directory`. PATH alone is not enough: since Python 3.8, Windows DLL lookup for extension modules ignores PATH. Without them, torchcodec fails with `OSError: Could not load this library: ...libtorchcodec_core8.dll`.
  - `.venv313` is not the type-check venv. Pyright, ruff and `tools/typing_guard` stay on `.venv` (torch 2.15.0+cpu), which `golden.json` and CI match.

## 2026-11-02: CI type check and how to bump its pins (Plan P, step 6)

`.github/workflows/typecheck.yaml` runs `pyright --pythonpath "$(which python)"` on Ubuntu with Python 3.12, CPU torch 2.15.0 (`torchaudio` unpinned) from the PyTorch CPU index, and `pip install -e ".[typecheck]"` followed by the eval packages (`jiwer modelscope zhconv zhon`, then `faster_whisper==0.10.1 funasr` with `--no-deps`). `pythonVersion = "3.10"` in `[tool.pyright]` stays, so stdlib typing is checked at the project's minimum level. Python 3.12 rather than 3.10 because 3.10 only has torch up to 2.14.1 and the local check uses 2.15.0.

- CI history (2026-11-02): run 1 failed in the install step: `pip install -e ".[eval,typecheck]"` tries to build `av==10.*` (pulled in by `faster_whisper==0.10.1`), which has no wheel for Python 3.12 and needs FFmpeg headers. The same failure had shown up locally on 3.15; I wrongly assumed 3.12 would have a wheel. The workflow now installs the eval packages separately. Run 2: install passed and pyright reported one error, `"bigvgan" is unknown import symbol` at `from third_party.BigVGAN import bigvgan` in `utils_infer.py`, because CI does not check out the BigVGAN submodule (an empty directory is an empty namespace package). Reproduced locally by moving `src/third_party/BigVGAN` aside; fixed with a narrow `# pyright: ignore[reportAttributeAccessIssue]`, verified with the submodule both present and empty. Run 3 (commit f6c231c): **pyright passed, 0 errors, 3m27s**. The pre-commit job failed on that same commit: the ruff sorter (`ruff check --select I --fix`, run only by pre-commit) wanted the over-long import line wrapped. Lesson: before pushing, run all three ruff hooks over the tree (`ruff check --fix`, `ruff format`, `ruff check --select I --fix`), not just `ruff check`.
- Third-party packages other than torch and pyright are not pinned. If CI starts failing with no code change, a dependency (gradio, transformers, x_transformers, ...) changed its types: reproduce locally with `pip install -U` of that package, then fix the code or add a narrow ignore with a reason.
- To bump pyright: change `pyright==` in the `typecheck` extra of `pyproject.toml`, run it locally, fix or suppress what it reports, and commit all of it together. To bump torch: change `torch==` in the workflow and rerun locally with the same version. Ruff is bumped in `.pre-commit-config.yaml` only, and needs a full `ruff check`/`ruff format --check` run first.
- The guard (`golden.py`) is not in CI: its numeric hashes are per torch build and OS, and `golden.json` was made on Windows.

## 2026-11-02: open bugs fixed after Plan P step 5

Found while annotating (pyright did not flag them). All in `train/finetune_gradio.py` unless stated; one commit per file or area.

### Data-prep workers failed with `NameError` under spawn or forkserver
- Symptom: `prepare_emilia.py` and `prepare_wenetspeech4tts.py` raised `NameError: name 'tokenizer' is not defined` (and `polyphone`) inside every worker process on Windows and macOS, and on Linux from Python 3.14 where the default start method became `forkserver`.
- Cause: the worker functions (`deal_with_audio_dir`, `deal_with_sub_path_files`) read module globals that are only assigned inside `if __name__ == "__main__":`. A forked child inherits them; a spawned or forkserver child re-imports the module and never runs that block. The other two `ProcessPoolExecutor` scripts (`prepare_emilia_v2.py`, `prepare_libritts.py`) do not read such globals.
- Fix: `tokenizer` and `polyphone` are now parameters of the worker functions and are passed in `executor.submit(...)`.
- Also removed the unused `polyphone` parameter of `process_audio_file` in `prepare_csv_wavs.py` (it only matters later, in `batch_convert_texts`, which still receives it).
- Tested: `prepare_emilia.deal_with_audio_dir` imported as a module (so no `__main__` globals exist, as in a spawned worker) and called on a one-line jsonl returns the row, duration and vocab. The scripts themselves need the Emilia and WenetSpeech datasets and were not run; `prepare_wenetspeech4tts.py` got the same change but was not exercised.

### Eval scripts: unknown settings left variables unbound
- `eval/eval_infer_batch.py`: an unknown `--testset` left `metainfo` unbound (`UnboundLocalError` later), and an unknown `mel_spec_type` from the model yaml left `vocoder_local_path` unbound. Both now raise `ValueError` naming the accepted values.
- `eval/eval_seedtts_testset.py` and `eval/eval_librispeech_test_clean.py`: `main` reused the name `args` for both the parsed `Namespace` and the per-GPU argument lists. It worked (the `Namespace` values were copied out first) but was easy to break; the lists are now `asr_args` and `sim_args`. No behaviour change.
- Not tested by running (the scripts need models and test sets).

### Speaker-similarity model swallowed every error and patched `torch.hub` for the whole process
- Symptom: `ECAPA_TDNN(...)` in `eval/ecapa_tdnn.py` hid why loading s3prl from the local hub cache failed (a bare `except:` also caught `KeyboardInterrupt`), and after the first construction `torch.hub._validate_not_a_forked_repo` stayed replaced for every later `torch.hub.load` in the process (for example UTMOS in `eval_utmos.py`).
- Fix: the patch is restored in a `finally`, and the fallback catches `Exception` and prints the reason before downloading `s3prl/s3prl`. The fallback itself is unchanged.
- Tested with a stub in place of `torch.hub.load`: the original function is restored after construction, both when the local load succeeds and when it fails and falls back. Not tested against the real s3prl (needs a download).

### Multi-style TTS tab fails when a segment has no usable reference audio
- Symptom: a segment whose type has no reference audio (or whose text is blank) raised `ValueError: not enough values to unpack` at `sr, audio_data = audio_out` in `generate_multistyle_speech` (`infer/infer_gradio.py`), after `infer()` had already shown a warning.
- Cause: `infer()` returns `gr.update()` placeholders instead of audio on those early exits (see the entry for the 3-versus-4 value fix), and the caller treated the first value as `(sample_rate, audio)`.
- Fix: such a segment is skipped (the warning is already on screen). If every segment is skipped the existing "No audio generated." path runs. `sr` was never unbound in practice: it is only read when at least one segment produced audio.
- Not tested by running the app.

### Fine-tuning app: several flow bugs
- "Train run already!" never reached the UI: `start_training` is a generator, so `return value` is discarded by gradio. It now `yield`s the message and then returns (return annotation is now `Generator[..., None, None]`).
- `Slicer.slice` returned a bare `[waveform]` for audio no longer than `min_length`, so `transcribe_all`'s `for chunk, start, end in ...` raised `ValueError: too many values to unpack`. It now returns `(waveform, 0, n_samples)` like the long-audio path; every path returns `(chunk, start, end)` tuples. Reproduced before the fix with a 500-sample signal. Side observation, not changed: `slice` compares `samples.shape[0]` (samples) with `self.min_length` (already converted to frames), so the short path only triggers for very short clips. It is inherited from the upstream slicer.
- `transcribe_all`: `if error_num != []` compared an int with a list, so "error files" was appended to the result every time; it is now `error_num > 0`. A silent chunk divided by `tmp_max == 0` and produced NaNs, so the loudness blend is skipped when `tmp_max` is 0. The double division by `tmp_max` for clipped chunks is unchanged: it matches the upstream GPT-SoVITS slicer.
- `create_metadata` passed a list as `total=` to the progress bar; it passes `len(...)`.
- `infer` returned 2 values on the "checkpoint not found" path but its click handler has 3 outputs (audio, device, seed), so gradio raised instead of showing the message. It returns `None, message, ""`.
- The "System Info" tab ended with `gr.update(fn=auto_update, ...)`, which only builds a dict and does nothing. It is replaced by `app.load(fn=update_stats, ...)` so the box is filled when the page opens (the evident intent of `auto_update`).
- Correction to the earlier worklist: `stop_signal` was **not** a crash. It is only read inside `start_training` after being assigned there. It is now also defined at module level for clarity.
- Left alone on purpose: `infer` creates its output wav with `delete=False`; gradio needs the file to exist after the function returns, so it cannot be removed there.
- Not tested by running the app (needs a GPU environment): the slicer was run directly; the rest was checked by reading the code and the click-handler wiring.

## 2026-11-01: bugs found by pyright and fixed (Plan P, step 5)

Each entry is one commit. All were found by reading pyright errors, then confirmed by reading the code (not by running the Gradio apps, which need a GPU and models).

### Gradio TTS tab crashes on blank text or missing reference audio
- Symptom: clicking Synthesize with no reference audio or empty text raised `ValueError: not enough values to unpack` instead of showing the warning.
- Cause: `infer()` in `infer/infer_gradio.py` returned 3 values on its two early-exit paths (after `gr.Warning`), but `basic_tts`, `generate_multistyle_speech` and `generate_audio_response` unpack 4 (audio, spectrogram, ref text, seed).
- Fix: both early returns now give a fourth `gr.update()` (leaves the seed box unchanged), and the return annotation is a single 4-tuple.
- Still open: `generate_multistyle_speech` unpacks `sr, audio_data = audio_out`, so a segment whose reference audio is blank still fails there (the early return carries a `gr.update()` dict, not audio). Needs a decision on whether to skip such a segment.

### `infer()` runs with no model for an unknown model name or an empty custom path
- Symptom: `UnboundLocalError: ema_model` for a model choice that is neither the default, "E2-TTS" nor a `("Custom", ...)` tuple; or a crash inside `infer_process` for a custom model whose checkpoint path is empty (`pre_custom_path` starts as `""`, so the load is skipped and `custom_ema_model` stays `None`).
- Fix: an `else` branch raises `ValueError` naming the choice, and a `None` model raises `ValueError` saying the custom path is empty.

### `load_dataset` fails for two of its three dataset types
- Symptom: `dataset_type="CustomDatasetPath"` raised `UnboundLocalError: preprocessed_mel`; `dataset_type="HFDataset"` raised `TypeError: load_dataset() got an unexpected keyword argument 'split'`; an unknown `dataset_type` or `audio_type` raised `UnboundLocalError` on `train_dataset` / `preprocessed_mel`.
- Cause: `model/dataset.py` defines its own `load_dataset`, which shadows the Hugging Face function of the same name, so the HFDataset branch called itself with arguments it does not take. The CustomDatasetPath branch passed `preprocessed_mel` without ever setting it.
- Fix: the Hugging Face function is imported as `hf_load_dataset` and used in that branch; `CustomDatasetPath` sets `preprocessed_mel = False` (it only reads the raw format); unknown types raise `ValueError`.
- Tested: a tiny on-disk dataset loads through `CustomDatasetPath` and the unknown-type errors fire. Not tested: the HFDataset branch needs a download and says in its own comment that its path must be edited by hand; only the call is corrected.

### Eval metadata reader reuses the previous line on a malformed line
- Symptom: `get_seedtts_testset_metainfo` and `get_seed_tts_test` (`eval/utils_eval.py`) either raised `UnboundLocalError: utt` (malformed first line) or silently copied the previous line's fields into a duplicate entry (malformed line later in the file, including a blank line in the middle of the file).
- Cause: the `if len(fields) == 5 / elif len(fields) == 4` chain had no `else`, so other lengths fell through with stale variables.
- Fix: blank lines are skipped and any other line without 4 or 5 fields raises `ValueError` naming the file and the line. Behaviour on well-formed files is unchanged.

### `f5-tts_finetune-gradio` fails at startup with gradio 6
- Symptom: `TypeError: Blocks.launch() got an unexpected keyword argument 'show_api'` when starting the fine-tuning app.
- Cause: `pyproject.toml` requires `gradio>=6.15.0`, and gradio 6 removed `show_api` from `launch()` (checked with `inspect.signature` on 6.31.0: no `show_api`, no `**kwargs`). It was replaced by `footer_links`. `infer_gradio.py` never used it.
- Fix: `launch(..., footer_links=None if api else ["gradio", "settings"])`. With `--api` (the default) the footer is gradio's default; without it the "api" link is left out. `queue(api_open=api)` still controls whether the API is open.
- Not tested by launching the app (needs the full UI and a GPU environment); the signature was checked instead.

### Metadata creation crashes on an absolute audio path with an unsupported extension
- Symptom: `TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType` in `create_metadata` (`train/finetune_gradio.py`).
- Cause: `get_correct_audio_path` returns `None` for an absolute path whose extension is not in the supported list, and `create_metadata` passed that straight to `os.path.isfile`.
- Fix: a `None` result is recorded as an "error path" entry (using the name from the transcript) like any other missing file, and the loop continues.

### Auto-setting training parameters fails on a machine without a GPU
- Symptom: `UnboundLocalError: gpu_count` (or `total_memory`) when pressing the auto-setting button on a machine with no CUDA, XPU or MPS device.
- Cause: the `if cuda / elif xpu / elif mps` chain in `train/finetune_gradio.py` had no `else`.
- Fix: an `else` returns the unchanged values with the message "no GPU found, cannot estimate the batch size", the same shape as the existing "project not found !" return.

### Audio duration fallback in `prepare_csv_wavs.py` could never run
- Symptom: if `soundfile` could not read a file, `get_audio_duration` raised `FileNotFoundError` (no `ffprobe` installed) or `RuntimeError: ... module 'torchaudio' has no attribute 'info'` instead of using its last fallback.
- Cause: two separate faults. `ffprobe` missing raises `FileNotFoundError`, an `OSError`, which the `except (TimeoutExpired, SubprocessError, ValueError)` did not list. And `torchaudio.info` no longer exists in torchaudio 2.9+ (installed here: 2.11.1; `pyproject.toml` allows `>=2.0`).
- Fix: `OSError` is caught, and the last fallback decodes with `torchaudio.load` and divides frames by sample rate. It decodes the whole file, which is acceptable because `soundfile` and `ffprobe` have already failed.
- Tested: with `soundfile.info` forced to fail and no `ffprobe` on the machine, the old code raised `FileNotFoundError` and the new code reaches the `torchaudio.load` fallback. A stubbed `torchaudio.load` (24000 frames at 24 kHz) gives 1.0. Not tested end to end: `torchaudio.load` needs `torchcodec` with FFmpeg, which is unusable on this Windows/Python 3.15 machine (corrected 2026-11-02: the cause is that torchcodec has no Python 3.15 wheel, not FFmpeg; see the entry of that date).

## 2026-11-01: type-checking environment (Plan P, step 1)

How the venv used for pyright and ruff was built, on Windows with Python 3.15.1. Commands run from the repo root; `.venv` is already in `.gitignore`.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e . ruff==0.11.2 pyright
.venv\Scripts\python.exe -m pip install jiwer modelscope zhconv zhon "jaxtyping>=0.3.0"
.venv\Scripts\python.exe -m pip install --no-deps faster_whisper==0.10.1 funasr
```

Versions that resulted: torch 2.15.0+cpu, torchaudio 2.11.1, numpy 2.5.4, accelerate 1.15.0, x-transformers 2.31.11, vocos 0.1.0, ema-pytorch 0.8.3, gradio 6.31.0, ruff 0.11.2, pyright 1.1.414.

### `pip install -e ".[eval]"` fails building `av`
- Symptom: `Failed to build 'av' when getting requirements to build wheel` (a Cython `CompileError` on `av\logging.pyx`).
- Cause: the `eval` extra pins `faster_whisper==0.10.1`, which needs `av`. PyAV has no wheel for Python 3.15 and its source build fails.
- Workaround: install the rest of the extra normally and install `faster_whisper` with `--no-deps`. Pyright only needs its source to read, so type checking is unaffected. Do not run the eval scripts from this venv.
- `funasr` fails the same way (`sentencepiece` has no 3.15 wheel), so it is also installed with `--no-deps`; importing it prints "Please install torch_complex firstly". An earlier version of this entry said `funasr` installed normally. It did not: the pip output was cut to its last lines and only the exit code was checked, so pyright reported its imports as unresolved. After any pip install, check `pip list`, not the exit code.

### Ruff must be scoped to `src/f5_tts tools`
- Symptom: `ruff check src` reports 75 errors, all in `src/third_party/BigVGAN`.
- Cause: the BigVGAN submodule is checked out on this machine (the plan assumed an empty directory). Pre-commit only sees tracked files, so it never lints it.
- Check: `ruff check src/f5_tts tools` and `ruff format --check src/f5_tts` both pass with 0.11.2 (47 files formatted).

### Torch wheels by Python version
- Checked with `pip index versions torch --python-version <v> --only-binary=:all:`: Python 3.10 tops out at torch 2.14.1; Python 3.12 gets 2.15.0. Torch typing differs between versions, so the CI job in step 6 needs one Python and one torch pinned together. Python 3.12 with torch 2.15.0 matches the local venv's torch.

## 2026-11-01: behaviour guard for the typing work (Plan P, step 2)

`tools/typing_guard/golden.py` compares the code against `tools/typing_guard/golden.json`, which was written on untouched `main` with torch 2.15.0+cpu on Windows.

```
.venv\Scripts\python.exe tools/typing_guard/golden.py          # check; exit 0 ok, 1 mismatch, 2 numeric skipped
.venv\Scripts\python.exe tools/typing_guard/golden.py --write  # only on a known-good baseline
```

- Checks: syntax of every file under `src/f5_tts` (also with the 3.10 grammar), imports of `f5_tts.model.*`, `api`, `device_utils`, `infer.utils_infer`, a hash of `CFM.forward` and three `CFM.sample` runs on a tiny random DiT, `convert_char_to_pinyin` / `get_epss_timesteps` / `chunk_text` outputs, and parameter names, kinds and defaults of 117 public functions and methods.
- The numeric part only runs when torch version, OS and CPU architecture match `golden.json`. Otherwise it prints SKIPPED and exits 2; regenerate the file on the baseline commit in the new environment.
- Verified it works as a tripwire: an annotation-only edit passes, a changed default (`steps=32` to `31`) fails in signatures, and a 0.1% change to the CFG formula fails in numeric.
- Harmless warnings on stderr: pydub cannot find ffmpeg, and `torch.jit.script` on Python 3.14+. Set `PYTHONWARNINGS=ignore` to hide them.

### Checking that an annotation batch changed nothing else
- `python tools/typing_guard/annotation_only.py <files>` compares each file with `HEAD` after removing annotations, imports and the named type aliases from both syntax trees. It prints `annotation-only` or `DIFFERS` per file and exits 1 on any difference. Run it before committing, since it compares against `HEAD`.
- Unlike `golden.py`, it covers every file, including the Gradio apps, trainer and eval code. If a batch adds a new module-level alias, add its name to `TYPE_ALIASES` in the script.
- Verified: a change of `self.sigma = sigma` to `sigma + 0` is reported as `DIFFERS`.

### Missing jaxtyping import is invisible to ruff
- Symptom: none from ruff. The per-file ignores in `ruff.toml` turn off F821 (undefined name) in the six model files, so a missing `Bool`, `Float`, `Int` or `Tensor` import is not reported there.
- Cause: the ignore is needed for the shape strings (`"b n d"`), and it covers genuine undefined names too.
- Diagnose: run pyright and look for "is not defined". Caught once so far: `Bool` was used but not imported in `model/cfm.py` before the batch A commit.

### A freshly built DiT outputs all zeros
- Symptom: any test of a new `DiT` returns zeros however the input changes.
- Cause: `DiT.initialize_weights` zero-fills the AdaLN layers and `proj_out`, as in the paper, so only trained weights produce output.
- Fix for tests: overwrite every parameter with seeded random values after construction, as `build_cfm` in the guard does.

## 2026-10-31: findings from checking docs/architecture.md against the code

Each entry is a behaviour that can look like a bug. Source is `src/f5_tts/` unless noted. Read from code, not reproduced by running.

### BigVGAN vocoder fails to import
- Symptom: printed "You need to follow the README to init submodule", then a NameError on `bigvgan`.
- Cause: `src/third_party/BigVGAN` is a git submodule (`.gitmodules`) and the directory is empty in a plain clone. `load_vocoder` in `infer/utils_infer.py` imports `third_party.BigVGAN`.
- Fix: `git submodule update --init`. Vocos does not need it.

### Assertion about space at index 0 when loading a vocab
- Symptom: `AssertionError: make sure space is of idx 0` from `get_tokenizer` (`model/utils.py`).
- Cause: for `pinyin` and `char`, the first line of `vocab.txt` must be a space. Index 0 is also what any character missing from the vocab maps to, so unknown characters are silently read as space.
- Not a bug: `data/Emilia_ZH_EN_pinyin/vocab.txt` is committed with CRLF line endings. Text-mode reads normalize them. Content matches `infer/examples/vocab.txt`.

### Very short gen_text sounds slow
- Cause: `infer_batch_process` multiplies speed by 0.3 for any chunk under 10 UTF-8 bytes, which lengthens the output.
- Check: print the chunks from `chunk_text`. A short last sentence after splitting hits this too.

### Output length does not match fix_duration with several chunks
- Cause: with more than one chunk, `fix_duration` is split across chunks by byte weight of each chunk, with cross-fade time added back. It is not applied per chunk.

### Reference text or audio seems ignored or stale
- `preprocess_ref_audio_text` clips reference audio to about 12 s, cutting at silences if it can. A long reference loses its tail, so `ref_text` may no longer match the audio. Shorten both.
- Processed audio and auto-transcripts are cached in memory by MD5 of the input file. Edited `ref_text` is not cached, but an old auto-transcript is reused until the process restarts or the audio changes.
- `ref_text` is forced to end with ". " or a full stop, so trailing punctuation you did not write is expected.

### fp16 versus fp32 differences
- `load_checkpoint` uses fp16 on CUDA with compute capability 7 or higher, fp32 otherwise. BigVGAN models are forced to fp32. `socket_server.py` defaults to fp32.

### Training resumes from an unexpected checkpoint
- `Trainer.load_checkpoint` order: `model_last.pt`, else the highest-numbered `model_<update>.pt`, else a `pretrained_*` file. Files named `pretrained_*` are excluded from `keep_last_n_checkpoints` rotation.
- Fine-tuning (`finetune_cli.py --finetune`) copies the base checkpoint to `ckpts/<dataset>/pretrained_<name>` only if that file does not already exist. A stale copy there is used over a new `--pretrain` path.

### Eval scripts cannot find files
- Seed-TTS test set is not in the repo. `eval_infer_batch.py` expects `data/seedtts_testset/{zh,en}/meta.lst`.
- ASR and WavLM checkpoint paths are hard-coded to `../checkpoints/` (`Systran/faster-whisper-large-v3`, `funasr`, `UniSpeech/wavlm_large_finetune.pth`) in `eval/eval_*.py`.
- The speaker model loads WavLM through `torch.hub` from `s3prl/s3prl`, which needs network access or `~/.cache/torch/hub/s3prl_s3prl_main`. `s3prl` is not in the `eval` extra.
- UTMOS also uses `torch.hub` (`tarepan/SpeechMOS`).

### Socket server handles one client at a time
- `start_server` in `socket_server.py` calls `handle_client` inline in the accept loop. A second client waits until the first disconnects. The first request after connecting is split into smaller chunks on purpose, for lower latency.

### Triton setup
- `runtime/triton_trtllm/run.sh` stage 1 copies `patch/*` into `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/models`. A different Python version or install path needs that line changed.
- Supported models are F5TTS_v1_Base, F5TTS_Base, F5TTS_v1_Small, F5TTS_Small, with the Vocos vocoder. E2-TTS and BigVGAN are not handled by `convert_checkpoint.py` or the vocoder export.
- Fine-tuned fp32 checkpoints need `--dtype float32` when converting (runtime README).
- `patch/` is TensorRT-LLM code, not a copy of `model/`. Changing `model/` does not need a matching change there, but changing the architecture does need a new engine.
