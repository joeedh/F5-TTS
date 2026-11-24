# Plan 05: Weight delivery, precision and quantization

Status: `plan pressure-tested` (written 2026-11-23; review findings folded in, see the end of this file). Depends on: plans 01 and 02 (done); plan 04's quality harness (done). Blocks: plans 06 and 07.

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). This plan starts from the hand-offs in [plan 01](webgpu-01-dit-export.md) (step 5), [plan 02](webgpu-02-mel-and-vocoder.md) (step 6), [plan 03](webgpu-03-text-frontend.md) (step 7) and [plan 04](webgpu-04-sampler-pipeline.md) (step 8).

## What this plan delivers

- Smaller weights that keep quality, chosen by measurement. The candidates are fp16 or int8 storage with fp32 compute, fp16 compute, and 8-bit and 4-bit quantized matmuls. Each gets a numeric check and a quality check, on both GPUs.
- A rule for which variant a browser gets, depending on what the adapter supports.
- A delivery format: a manifest, external-data shard files with hashes, and a pinned revision.
- A downloader with progress, resume, abort and hash checks, and a browser cache with versioning and eviction. `TTS.create` loads from it.
- A script that rebuilds every published file from upstream's checkpoints, and writes the model card.
- The Hugging Face repo, published only after the attribution wording is settled (the gate below).
- The buffer sizes each variant needs at each n, so plan 07's device matrix knows what a device must support.

Not in this plan:
- The download UI, error screens and the worker (plan 06). This plan gives plan 06 the events and errors to show.
- Speed work beyond what a precision change brings (plan 07). Speed is measured here because fp16's main case on the Intel GPU is speed.
- Other checkpoints than F5TTS v1 Base.

## The gate

Nothing is uploaded until the user has settled the model card's attribution wording. That is an open question in the task list. Steps 1 to 8 need no upload. Step 9 drafts the card and stops for the user. The upload itself is outward-facing, so it happens only with the user's approval at that point, with their account and token.

## What exists

- **The graphs** (plans 01 and 02), all fp32 and ONNX IR version 10:

  | File | Graph | Weights |
  |---|---|---|
  | `step_mha.onnx` | 2.1 MB | 1,330 MB external data |
  | `text.onnx` | 0.1 MB | 38.9 MB external data |
  | `vocos.onnx` | 54.3 MB, weights inside | none |

  Built by `tools/webgpu/export_dit.py --model base --attention mha` and `export_vocos.py --model base` into `build/webgpu/`.
- **Where the step graph's stored values go** (counted 2026-11-23):
  - `MatMul` nodes: 185.4M.
  - `Gemm` nodes: 142.0M. That includes the 22 `attn_norm.linear` weights (6144 x 1024), which are the AdaLN modulation weights.
  - `Conv`: 4.1M.
  - The rest is smaller.
- **Stored values that are not checkpoint weights.** The export precomputed these, so rounding them is not "rounding the weights":
  - `rope_tables` (1.05M values) and `unsqueeze_1`, the 128 time-embedding frequencies, in the step graph.
  - `freqs_cis` (4.19M values, 43% of `text.onnx.data`) in the text graph.
  - Rounding the time frequencies to fp16 alone gives a sinusoid error of 5e-3 to 5e-2 (pressure test). The `time_embed` tolerance is 1e-6.
- **The checkpoint**: `SWivid/F5-TTS`, `F5TTS_v1_Base/model_1250000.safetensors`, fp32. Measured 2026-11-23:
  - 337.1M floats under `ema_model`. The largest absolute value is 8.84, so nothing overflows fp16.
  - 761,607 values (0.23%) become fp16 subnormals, and 387 flush to zero.
  - Rounding every weight to fp16 gives a relative L2 error of 2.1e-4 over all weights.
  - The largest tensor has 6.3M values (25 MB fp32).
- **The Vocos checkpoint**: `charactr/vocos-mel-24khz`, `pytorch_model.bin`, a pickle, loaded by upstream's `load_vocoder`.
- **Loading today**: `DiT.create` passes `externalData: [{ path, data: url }]`, so ORT fetches the data file itself. `Vocoder.create` fetches one file. Neither caches.
  - ORT Web mounts each `externalData` entry as its own file, and reads each initializer at its offset (`ort.webgpu.mjs:3496-3508`). An entry can be a URL, a `Uint8Array` or a `Blob`.
  - So a graph whose data is split over several files needs no single large buffer.
- **Plan 01's loading measurements** (fp32, local server):
  - Session creation took 8 to 9 s on NVIDIA and 10 to 16 s on Intel.
  - Browser RAM peaked at 2 to 3.5 GB while loading.
  - Chrome's HTTP cache does not keep the 1.33 GB file.
  - Cache Storage does keep it: filling it took 25 to 51 s, and loading from it 2.8 to 3.8 s.
  - OPFS and IndexedDB were not tried.
- **The plan 00 spike's fp16 findings**, on one DiT block with `onnxconverter-common`'s `convert_float_to_float16(keep_io_types=True)`:
  - Relative L2 against fp32 on the WebGPU EP: 4.3e-3 for a lone K = 1024 MatMul, 8.2e-3 for the block at n = 1024, 1.0e-2 for convpos, and 3.7e-3 for the ConvNeXt.
  - That is 20 to 30 times ORT CPU fp16. It fits fp16 accumulation, but that was never checked.
  - Speed gain: 1.35 to 1.6 times on NVIDIA, 1.7 to 1.8 times on Intel.
- **Both GPUs report `shader-f16`** and a 2 GiB `maxBufferSize` (`../research/webgpu-laptop.md`).
- **ORT Web 1.31.1's WebGPU build** is the native C++ WebGPU EP (the bundle loads the `asyncify` wasm, not JSEP). Its wasm contains:
  - `contrib_ops/webgpu/quantization/matmul_nbits.cc`, `dp4a_matmul_nbits.cc` and `subgroup_matrix_matmul_nbits.cc`;
  - the message "Only 4-bit and 8-bit quantization are supported for MatMulNBits".
- **ORT's source is not reachable.** `github.com/microsoft/onnxruntime` returns 404, also through the GitHub API (checked 2026-11-23). So the kernels' behaviour has to be probed, not read.
- **Tools in `.venv313`:**
  - onnx 1.23.2, which writes IR 14 by default. ORT 1.31.1 accepts at most IR 13.
  - onnxruntime 1.31.1 (CPU), whose `MatMulNBits` quantizer does 2, 4 and 8 bits. Its default `op_types_to_quantize` is `MatMul` only, so `Gemm` is left alone.
  - onnxconverter-common 1.16.0.
  - huggingface_hub 1.33.0.
- **Plan 04's quality harness**: `web/bench/quality/run.mjs` and `tools/webgpu/quality_score.py`.
  - It has the PyTorch fp32 baseline stored for 100 English and 100 Chinese utterances. The fp32 port equals it: identical WER, and SIM within 1e-5.
  - One side, both languages, took about 30 to 45 minutes on the RTX 5050.
- **The golden data**: about 1.5 GB per case under `testdata/webgpu/`, git-ignored. `tolerances.json` holds the fp32 tolerances. Plan 00 left fp16 tolerances to this plan.

## Decisions

### D1. Variants to build (proposal: storage variants first, compute variants after)

"Checkpoint tensors" below means the tensors that come from the checkpoint. The precomputed tables (`rope_tables`, the time frequencies, `freqs_cis`) stay fp32 in every variant.

| Variant | Checkpoint tensors stored as | Computed in | Needs | DiT download (estimate) |
|---|---|---|---|---|
| `fp32` | fp32 | fp32 | nothing | 1.37 GB (today) |
| `w16` | fp16 | fp32; the loader widens to fp32 before ORT sees them | nothing | about 0.70 GB |
| `w8` | int8 with per-block scales | fp32; the loader dequantizes | nothing | about 0.38 GB |
| `fp16` | fp16 | fp16, with fp32 graph inputs and outputs | `shader-f16` | about 0.70 GB |
| `mixed` | fp16, and fp32 where kept | fp16, except a block list (D2) | `shader-f16` | 0.70 GB or more |
| `q8` | `MatMulNBits` 8-bit, other tensors as in `w16` | fp32 activations, weights dequantized in the kernel | step 1's answer | about 0.38 GB |
| `q4` | the same, 4-bit | the same | step 1's answer | about 0.22 GB |

- **The storage variants (`w16`, `w8`)** use the fp32 graph unchanged. They need no new kernel and run anywhere `fp32` does. `w16` halves the download for the cost of rounding the weights; `w8` nearly quarters it. Their only cost is the weight error, which the quality harness judges.
- **`fp16` and `mixed`** matter for speed on the Intel GPU, the default GPU on Windows laptops. Step 1 found ORT's fp16 `MatMul`, `Gemm` and `Conv` accumulate in fp16 (12 to 32 times the rounding floor), so `mixed` would have to keep them in fp32 to help, which leaves it little speed to gain.
- **`q8` and `q4`** need the kernel to work as step 1 finds. Their quantized activations stay fp32, with `accuracy_level` 0 or unset: level 4 selects the DP4A path, which quantizes the activations to int8. They are not put through the fp16 converter.
  - **Their sizes depend on `Gemm`.** The estimates above assume step 2 rewrites the `Gemm` nodes as `MatMul` plus `Add` so they are quantized too. Without that, the 142M `Gemm` weights stay 16-bit, and `q8` and `q4` come to about 0.52 and 0.42 GB.
  - **The AdaLN modulation weights** (`attn_norm.linear`, 141M of them) may be the most sensitive to quantization. Step 5 tries `q4` with those kept at 8 bits if plain `q4` fails.
  - **The quantization method**: round-to-nearest first. The installed quantizer also has HQQ and GPTQ, tried only if round-to-nearest fails at 4 bits; on random weights it costs 8% weight error at 4 bits.
- **The vocoder** gets `w16` only if that is free. It is 54 MB, 4% of the total, and fp16 compute saves it little time (plan 02). Storing it as `w16` means re-saving `vocos.onnx` with external data.
- **Sizes are estimates** until step 2 builds the files.

### D2. What a variant must meet (proposal: separate weight error from kernel error)

A variant's error has two sources: the weights it approximates, and the kernels that compute with them. The checks keep them apart.

- **Kernel error, checked numerically.** For `w16`, `w8`, `q8` and `q4`, make golden data from PyTorch on the CPU with exactly the variant's weights.
  - For `w16`, the checkpoint tensors are rounded to fp16.
  - For the others, the values come from the built files: the stored `q`, zero point and scale are read back and dequantized as `(q - zp) * scale` in fp32. The quantizer is not reimplemented. The pressure test found ORT CPU's `MatMulNBits` matches `A @ deq` within 3e-7.
  - The port, given those weights, must then match that golden data within the existing fp32 tolerances. The stage checks stay as strict as plan 00's, and a kernel problem shows up as a failure.
  - Two golden cases per variant: en-short and zh-short, about 3 GB.
- **fp16 compute** (`fp16` and `mixed`) has no exact reference. CUDA's fp16 kernels are not WebGPU's, and upstream's PyTorch fp16 keeps the time embedding in fp32.
  - Its stage errors are measured against PyTorch CUDA fp16's own floors (`measure_floor.py` gains an fp16 mode), recorded, not gated.
  - The converter changes more than precision: it clamps any value in (0, 1e-7) to 1e-7, and it would put the time embedding's `time * 1000` and the ODE update `x + dt * v` in fp16. So `mixed`'s first block list keeps these in fp32:
    - the time embedding (`Mul`, `Sin`, `Cos`);
    - the CFG combination and the ODE update at the end of the step;
    - then MatMul and Conv, if step 4's numbers point there.
  - `op_block_list` replaces the converter's default list rather than adding to it, so the default list is passed along with ours.
- **Weight error, and every variant's quality, judged by plan 04's harness.** The gate is D5's bounds: mean WER within 0.5 points of PyTorch fp32 and mean SIM within 0.01, per language, on the same 200 utterances and seeds. Variants are screened on the RTX 5050. Those that pass and might ship are confirmed on the Intel GPU.
- **A third harness side**: PyTorch CUDA fp16 with CPU noise, which is what upstream's default users hear. It shows how much fp16 costs upstream itself, as context. It is not a gate.
- **Listening**: the five utterances per language with the largest SIM differences, per variant, for the user.

### D3. Which variant a browser gets (proposal: decided in step 6 with the numbers)

- The likely shape: the smallest variant that passes D2 and that the adapter can run. `fp16` and `mixed` need `shader-f16`. The storage variants run everywhere. The quantized variants need what step 1 finds.
- Integrated and discrete GPUs could get different variants, if `fp16` is only worth its error on the slower one.
- The user decides, from step 6's table of size, speed and quality per GPU.

### D4. Delivery format (proposal: a manifest and tensor-aligned shard files)

- **Shards are real external-data files.** Each graph's data is written as several files of at most 64 MiB, split on tensor boundaries (the largest tensor is 25 MB). ORT mounts each as its own `externalData` entry, so nothing is reassembled and no single large buffer is needed.
- **Storage variants**: a `w16` or `w8` shard holds its tensors in the stored dtype. The manifest gives each shard's tensor table: name, offset in the stored file, stored dtype, and offset in the fp32 file the graph expects. The loader widens or dequantizes one shard at a time into the fp32 layout, and hands that to ORT.
- **The manifest** (`manifest.json`) records:
  - the format version;
  - per variant, every file with its size and SHA-256 (sizes come from here, not from response headers);
  - the vocab;
  - the hashes of both source checkpoints;
  - the ORT Web version each variant was validated with;
  - the commit of this repo that built it.
- **A pinned revision**: the app loads the manifest from a fixed Hugging Face commit, and the app build embeds the manifest's SHA-256. A published file can never change under a cached copy. A new export means a new commit and a new app build.
- **Resume by shard.** A shard cut off partway is fetched again whole. Step 1 found `Range` working through Hugging Face's redirect today, but a forum report (2026-01) says it fails CORS on the Xet bridge (`cas-bridge.xethub.hf.co`). So `Range` resume inside a shard is an optional extra, never needed.

### D5. Cache (proposal: OPFS, verified on write)

- **The store**: each file is stored in the origin private file system (OPFS), named by its hash. A file is complete only once its hash checks out and it has been moved into place, so a partial download never looks complete.
- **Browser support**:
  - `createWritable` exists in Chrome 86, Firefox 111 and Safari 26, on the main thread too.
  - `createSyncAccessHandle` (workers only) and `move()` exist in Chrome 102, Firefox 111 and Safari 15.2.
  - Step 1 picked them: write with `createWritable`, piping the fetch body into it, which took about 3 s per 1.31 GiB. Read with a sync access handle in a worker, which took 0.4 s against 4 to 7 s for `File.arrayBuffer()`, and hand ORT the bytes.
  - OPFS and `crypto.subtle` both need a secure context, so testing from another device on the LAN needs HTTPS.
- **Verification**:
  - Every file is hashed once, when it is written.
  - On later loads only sizes are checked, because rehashing 0.4 to 1.4 GB on every load would cost more than it saves.
  - If session creation fails, the loader rehashes, deletes and refetches the bad files, then tries once more.
- **Fetching**: `fetch(url, { cache: "no-store" })`, so Chrome's HTTP cache doesn't keep a second copy. A 429 waits and retries, honouring `Retry-After`. Anonymous downloads are limited to 3,000 `resolve` requests per 5 minutes per IP, far above the roughly 25 shards per variant. A failed redirect retries from the `resolve` URL.
- **Eviction**: after a successful load, delete stored files that the current manifest does not list, under any variant. A device whose GPU changes keeps both variants.
- **Persistence and quota**:
  - Ask for `navigator.storage.persist()`. Firefox shows a prompt for it, and Safari may still evict after 7 days without a visit.
  - Before downloading, check `navigator.storage.estimate()`, and fail with a clear error if there isn't room.
- **The API**: `loadWeights({ manifestUrl, variant, onProgress, signal })` returns the `externalData` entries and graph bytes for `DiT.create` and `Vocoder.create`. `TTS.create` gains an option that takes them. Progress reports bytes done and total, and which files came from the cache. Plan 06 builds the UI on it.

### D6. Hosting and licence (decided earlier; wording open)

- **The repo**: a Hugging Face model repo under the user's account, named by the user. The browser downloads from it (decided before plan 00). It holds the manifest, the shards, the graphs, `vocab.txt` and the model card.
- **Licences**:
  - F5-TTS weights: CC-BY-NC 4.0, because of the Emilia training data (upstream's README, line 278).
  - Vocos weights: their own licence, which step 9 reads from the source rather than assuming.
  - The card tags the repo CC-BY-NC 4.0, credits the F5-TTS authors and Emilia, links `SWivid/F5-TTS`, and says use is non-commercial.
  - It also states the modifications (conversion to ONNX, fp16 rounding, quantization), as the licence's attribution terms require for adapted material (section 3(a)(1)(B)).
- **Open, the gate:** the exact attribution wording. Step 9 drafts it and the user decides.
- The app's README says the weights carry their own licence (task list). That is a one-line change, made in step 9.

## Steps

### Step 1. Facts the variants and the delivery depend on

ORT's source is not reachable, so this step probes. Write the results to `docs/research/webgpu-weights.md`, and add it to `docs/index.md`.

- **ORT Web 1.31.1's WebGPU kernels, by tiny graphs in Chrome on both GPUs:**
  - **fp16 accumulation:** an fp16 `MatMul` with K = 4096 on inputs of all ones. It returns 2048 if it accumulates in fp16 (the sum stops growing there) and 4096 if in fp32. The same for `Conv`, and for `Gemm`.
  - **fp16 `MultiHeadAttention`, `LayerNormalization`, `Softmax` and `Gelu`:** each against fp32 on inputs with a large range, to see whether they compute internally in fp32.
  - **`MatMulNBits`:** at 4 and 8 bits, block sizes 32 and 128, fp32 and fp16 activations, and `accuracy_level` unset. Does it run on the GPU (verbose placement log), and how far is it from `A @ deq`?
  - **Device limits:** which limits ORT requests when it creates its device. Wrap `GPUAdapter.requestDevice` before the first session.
- **The tools:**
  - the `MatMulNBits` quantizer's options and stored layout (zero point defaults, nibble order, scale shape), and how to read them back;
  - `convert_float_to_float16`'s block lists and default list;
  - how to write external data split on tensor boundaries with an IR version ORT accepts.
- **Hugging Face from a browser page**, against a small public file:
  - Does a plain GET of a `resolve` URL work, through the redirect, with CORS?
  - What does a `Range` GET do?
  - Which headers can the page read?
  - If a probe itself fails CORS, that is the answer; nothing large is fetched.
- **OPFS in Chrome on this laptop:**
  - `createWritable` against a sync access handle in a worker;
  - the quota `estimate()` reports;
  - writing 1.4 GB in 64 MiB files and reading it back, timed;
  - passing OPFS `File` objects (Blobs) straight to `externalData`, against `Uint8Array`s.
- Done when: the research doc answers each question with the probe result. Any answer that rules out a variant updates D1 before step 2.
- Result, 2026-11-23: done; details in [research/webgpu-weights.md](../research/webgpu-weights.md). Nothing rules out a variant.
  - **fp16 compute** is the same on both GPUs:
    - `MatMul`, `Gemm` and the convpos `Conv` accumulate in fp16, at 12 to 32 times the fp16 rounding floor, and more as K grows.
    - `MultiHeadAttention` is at 5 to 7 times. LayerNorm (statistics in fp32), Gelu, Mish, Sigmoid, Sin and the depthwise conv are at 1 to 2 times.
    - So `mixed` with the matmuls in fp32 would keep most of the compute in fp32; step 4 decides whether it is worth building.
  - **`MatMulNBits` with fp32 activations is exact on WebGPU:** within 2e-7 of `A @ deq`, at 4 and 8 bits, blocks 32 and 128, symmetric or not, on both GPUs.
    - `q8` and `q4` stay, and need no `shader-f16`.
    - With fp16 activations it accumulates in fp16 (8.5 times the floor).
    - The read-back dequantization D2 needs matches both ORT CPU and WebGPU.
  - **Device limits:** ORT asks for the adapter's maximum on ten limits, 2 GiB buffers included, and for `shader-f16` and subgroups when present.
  - **Hugging Face:** from a page, a GET through the redirect to `us.aws.cdn.hf.co` works with CORS. `Range` requests work too (206), every header is exposed, and the ETag is the SHA-256. The `Range` problem the pressure test found reported does not occur today.
  - **OPFS:**
    - Writing with `createWritable` took about 3 s per 1.31 GiB; a worker's sync handle took 6 to 18 s to write.
    - Reading through the sync handle took 0.4 s, against 4 to 7 s for `File.arrayBuffer()`.
    - Through ORT, the step graph loaded from OPFS bytes in 1.7 to 1.9 s, against 2.3 to 2.4 s from Cache Storage and 6.4 to 6.7 s from a URL. Filling OPFS took 17 s, against 25 to 48 s for Cache Storage.
    - Peak browser memory was 1 to 3 GB above the URL path's 2.5 GB, which is R4. Steps 3 and 7 measure it with tensor-aligned shards.
    - The quota was 10 GB in Playwright's Chrome profile, with 137 GB free.
  - **Tools:** `convert_float_to_float16` clamps constants to 1e4 and 1e-7, which step 2 checks the graphs for. Tensor-aligned shards need our own external-data writer.
  - **New:** `tools/webgpu/probe_kernels.py`, `web/bench/weights-probe/`, and OPFS scenarios in `web/bench/dit-load/`. The memory sampler there had been failing silently and now works (debugging.md, 2026-11-23).

### Step 2. Build the variants

- `tools/webgpu/convert_precision.py`: from the fp32 graphs in `build/webgpu/`, writes each variant step 1 kept to `build/webgpu/variants/<variant>/`, with tensor-aligned data shards (D4).
  - `w16` and `w8`: the fp32 graph, with its checkpoint tensors stored as fp16, or as int8 with block scales, and their tensor tables.
  - `fp16` and `mixed`: the converter with D2's block lists, on the fp32 graph.
  - `q8` and `q4`: if step 1 found the kernel usable, a pass that rewrites each `Gemm` with a constant weight as `MatMul` plus `Add` (checked on ORT CPU against the original), then the quantizer.
- A JSON summary per variant: file sizes, op counts by type and dtype, and the converter versions.
- The checks:
  - every graph passes `onnx.checker`, at IR version 13 or below;
  - ORT CPU runs it on one golden step;
  - `w16`'s and `w8`'s widened data equals the approximated weights bit for bit;
  - the precomputed tables are bit-identical to `fp32`'s in every variant.
- Done when: every variant step 1 kept is built and passes these checks, and ruff and pyright pass.
- Result, 2026-11-23: five variants built besides `fp32`, all passing the checks. `mixed` is not built: step 1 found it would have to keep the matmuls in fp32 to help, so it waits for step 4's `fp16` numbers.
  - `tools/webgpu/convert_precision.py` writes `build/webgpu/variants/<variant>/`: the three graphs, the stored shards, `variant.json` (every shard's tensor table and the check results), `approx_weights.safetensors` for step 4's golden data, and `layout/` (the widened shards, for the ORT CPU checks only).
  - **Sizes, all three graphs** (text, step and vocoder):

    | Variant | Stored | Step graph | Step prediction against `fp32`, ORT CPU |
    |---|---|---|---|
    | `fp32` | 1.43 GB | 1,330 MB in 23 shards | — |
    | `w16` | 0.73 GB | 667 MB in 23 shards | 1.3e-4 |
    | `fp16` | 0.73 GB | 668 MB in 11 shards | 3.4e-4 (ORT CPU computes fp16 graphs in fp32; not the WebGPU error) |
    | `w8` | 0.43 GB | 377 MB in 23 shards | 4.4e-3 |
    | `q8` | 0.41 GB | 353 MB in 6 shards | 4.0e-3 |
    | `q4` | 0.27 GB | 223 MB in 4 shards | 4.4e-2 |

    - `q4` is above D1's 0.22 GB estimate: block-32 scales and zero points add 28% to the 4-bit weights, and the vocoder (27 MB at f16) and the fp32 tables (21 MB) come on top.
    - Shards are cut at 64 MiB of fp32 layout, so a `w16` or `w8` step graph has 23 small shards (29 and 16 MB). Step 7 can cut by stored size instead if that matters.
  - **Checkpoint tensors are found by value,** directly or transposed: 41 in the text graph, 321 in the step graph and 80 in the vocoder, which is every float tensor of 1 KB or more except the three tables and one all-ones LayerNorm scale. The exporter's `val_*` initializers are transposed checkpoint weights.
  - **The checks, per variant:**
    - `onnx.checker` passes, at IR 10;
    - the loader's widening, written in Python, reproduces the layout bit for bit;
    - the tables are bit-identical to `fp32`'s;
    - chunk 0, step 0 of en-short runs through all three graphs on ORT CPU with finite outputs.
    - The `Gemm` rewrite alone changes the fp32 step's outputs by 4e-7.
  - **Quantized graphs:** all 25 `Gemm`s rewritten; 159 `MatMulNBits` in the step graph and 8 in the text graph; no `MatMul` with constant weights left.
  - **`fp16`:** 14 step nodes stay fp32 (the time embedding through `Sin`, `Cos` and `Concat`, the rope table's `Slice`, and the CFG and Euler tail), and the text graph's `freqs_cis` reader.
  - **The converter needed three fixes** (debugging.md, 2026-11-23). It rounds every float initializer even when only blocked nodes read it. It casts blocked nodes' outputs to fp16 even when only blocked nodes read them. And it declared `v` and `x_next` fp16 despite `keep_io_types`. The table check caught the first; an fp32 `Sin` input in the converted graph confirms the time embedding is fp32 now.

### Step 3. Load a variant in the browser

- `DiT.create` and `Vocoder.create` take `externalData` entries, from URLs or bytes.
- For `w16` and `w8`, `widenShard` turns one shard's stored tensors into the fp32 layout, from the manifest's tensor table.
- `TTS.create` takes a variant name. A variant that needs `shader-f16` is refused with a clear error on an adapter without it.
- Node tests: `widenShard` on every fp16 bit pattern, and on int8 with scales, against a numpy reference.
- Record per variant and GPU: load time from a local server, and peak browser RAM while loading.
- Done when: every variant loads on both GPUs, or its failure is recorded with the reason.
- Result, 2026-11-24: all six variants load and run on both GPUs.
  - `web/src/weights/variant.ts` reads `variant.json`, widens shards and checks GPU features. `DiT.create` and `Vocoder.create` take a source (graph and data) or a URL.
  - `TTS.create` takes `variantUrl`, the variant's folder, not a name. Names come with step 7's manifest.
  - **Load time from a local server, and peak memory while loading.** Chrome 158, headless, a fresh browser per variant, one run per GPU (the last of two on the RTX 5050, whose load times differed by up to 11%). Memory is Windows' own peak of committed memory for each process, so the sum is an upper bound. The GPU process figure includes memory for GPU buffers.

    | Variant | RTX 5050: load | renderer + GPU process | Intel: load | renderer + GPU process |
    |---|---|---|---|---|
    | `fp32` | 3.6 s | 1.65 + 1.72 GB | 3.8 s | 1.65 + 1.62 GB |
    | `w16` | 4.9 s | 1.60 + 1.66 GB | 4.7 s | 1.60 + 1.62 GB |
    | `w8` | 4.4 s | 1.79 + 1.69 GB | 4.4 s | 1.75 + 1.62 GB |
    | `fp16` | 3.1 s | 0.95 + 1.02 GB | 3.2 s | 0.95 + 0.99 GB |
    | `q8` | 2.1 s | 0.78 + 0.70 GB | 2.1 s | 0.77 + 0.75 GB |
    | `q4` | 2.2 s | 0.62 + 0.56 GB | 2.1 s | 0.62 + 0.61 GB |

  - **The storage variants cost no extra peak memory over `fp32`,** although the loader holds every widened shard of a graph at once (ORT takes a graph's data all together, so R4's "one at a time" is not possible). They cost about 1 s more to load.
  - **One step at n = 1913 (en-short, the later of three runs):**

    | Variant | RTX 5050 | Intel |
    |---|---|---|
    | `fp32` | 0.88 s | 2.5 s |
    | `w16`, `w8` | 0.89 to 0.90 s | 2.5 s |
    | `fp16` | 0.76 s | 2.0 s |
    | `q8` | **3.3 s** | **9.2 s** |
    | `q4` | 0.89 s | 2.6 s |

    - `q8` is 3.7 times slower per step than `fp32` on both GPUs, and `q4` is no faster than `fp32`. The cause is not known: `q8` differs from `q4` in bits and in block size (128 against 32). Step 4 should find out which before it spends quality runs on `q8`.
    - `fp16` is 14% faster on the RTX 5050 and 19% faster on the Intel GPU.
  - **Against the fp32 golden data, informational** (step 4 compares the approximated variants with their own). Step 0's conditional output, relative L2, RTX 5050: `fp32` 1.2e-6, `w16` 1.5e-4, `q8` 4.7e-3, `w8` 4.8e-3, `fp16` 6.1e-3, `q4` 4.9e-2. The approximated weights match their ORT CPU figures from step 2; `fp16` is 18 times its ORT CPU figure, as step 1's kernel probe predicted. The vocoder gives 5.0e-3 for every variant with an fp16-stored vocoder, against 4.3e-4 for `fp32`.
  - **Through `TTS.create({ variantUrl })`, RTX 5050, en-short:** the `fp32` variant passes `final.free_running` at 7.8e-4 (tolerance 0.1). `w8` generates audio, and differs from the `fp32` golden audio by 0.38; step 4 compares it with its own golden data. `WEIGHT_VARIANT=<v>` on `final.spec.ts` runs this.
  - **Tests:** 11 Node tests on fixtures from `convert_precision.save()` (`tools/webgpu/weights_fixtures.py`): all 65,536 fp16 values, every stored dtype byte for byte, misaligned input, wrong sizes, which shards need widening, and the `shader-f16` refusal with a fake adapter. Both real GPUs have `shader-f16`, so the refusal has not run in a browser.
  - **Step 1's OPFS memory figures are in doubt.** `dit-load` takes them from the same 1.5 s sampler, with every scenario in one browser. Step 7 should measure again, with the peak counters and a fresh browser, before choosing bytes or `Blob`s. `dit-load` now uses the shared `bench/memory-sampler.mjs`, and has not been rerun since.
  - **A loader bug found by `q4`** (debugging.md, 2026-11-24): the loader decided by dtype alone whether a shard needed widening.
  - The rebuild for the new `features` field was stopped partway for low memory. `fp16` was rebuilt on its own; `q8` and `q4` are step 2's builds and have no `features` field, which the loader reads as none needed.

### Step 4. Numeric checks (D2)

- `make_golden.py --weights <variant>`: the same pipeline, with the variant's weights as D2 says. Cases en-short and zh-short. Output goes to `testdata/webgpu/variants/<variant>/<case>/` (git-ignored, about 3 GB per variant).
- `measure_floor.py --fp16`: PyTorch CUDA fp16 against the fp32 golden data, per check, recorded in its own floors file.
- In the browser, on both GPUs:
  - `w16`, `w8`, `q8` and `q4` against their own golden data, through the existing parity runner and fp32 tolerances: `dit.out` teacher-forced, `mel_out.free_running` and `final.free_running`.
  - `fp16` and `mixed` against the fp32 golden data. Their relative L2 per check is recorded next to CUDA fp16's floor.
- Speed: seconds per step at n = 1024, 2048 and 4096 per variant, on both GPUs.
- Done when:
  - the table of errors and speeds is in the research doc;
  - a variant that fails its kernel check has been fixed, or dropped with the reason;
  - `mixed`'s block list is settled.

### Step 5. Quality (D2)

- `quality_torch.py --fp16`: the PyTorch CUDA fp16 side, with CPU noise, for both languages, scored like the others.
- `run.mjs --variant <v>`: the port side per variant, in `gen/webgpu-<v>/`. `quality_score.py` takes any list of sides, and `--report` compares each with PyTorch fp32.
- Order: the storage variants first, then `q8` and `q4` (and `q4` with 8-bit AdaLN weights if plain `q4` fails), then `fp16` and `mixed`. All on the RTX 5050. The variants that pass and might ship are also run on the Intel GPU.
- **Time budget:** about 45 minutes per variant on the RTX 5050 and about 2 hours on the Intel GPU, both languages, plus scoring. Seven variants and the PyTorch fp16 side come to roughly 8 hours of NVIDIA time, and 2 to 3 Intel runs another 4 to 6. One job at a time.
- Listening: for each variant, the five utterances per language with the largest SIM differences, both sides, in `build/webgpu/listen/variants/`. They go to the user.
- Done when: the quality table is in `docs/research/webgpu-quality.md` and the listening files are with the user.

### Step 6. Choose (D3)

- Put the table to the user: per variant and GPU, the download size, load time, seconds per step, WER and SIM differences, and the listening.
- The user picks the published variants and the rule for which one a browser gets.
- Done when: D3 is decided and written into this plan.

### Step 7. Delivery (D4, D5)

- `tools/webgpu/publish_weights.py`, which writes `build/webgpu/publish/`:
  - It runs the exporters and the converter for the chosen variants, from the checkpoints `hf_hub_download` fetches, so the whole repo can be rebuilt from scratch.
  - It hashes everything and writes the manifest.
  - Upload is a separate subcommand (step 9).
- `web/src/weights/`:
  - the manifest parser;
  - the shard downloader (parallel, checked against hashes, abortable, retrying on 429 and failed redirects);
  - the OPFS store, written from a worker;
  - verification on write, and the repair path on a failed session;
  - eviction, `persist()`, and the quota check;
  - `loadWeights`, wired into `TTS.create`.
- Tests:
  - Node: the manifest parser, the tensor tables, a hash mismatch rejected, and a cut-off shard fetched again.
  - Browser, local only, from a local server that serves `build/webgpu/publish/` and can drop connections and return 429:
    - a first load, then a second load from the cache with the server stopped;
    - a load cut off partway that resumes with only the unfinished shards fetched;
    - files from an old manifest evicted after a new one loads;
    - a corrupted cached shard repaired after the session fails;
    - a 429 retried.
- Measure the first load and the cached load for the chosen variants on both GPUs, against plan 01's Cache Storage numbers, and record peak browser RAM.
- Done when: all pass; the cached load is no slower than Cache Storage's; and ruff, pyright, tsc, prettier and vitest pass.

### Step 8. Buffer limits

- With step 1's wrapper, also wrap `GPUDevice.createBuffer`, and record the largest buffer ORT creates. Do this per chosen variant at n = 1024, 2048, 4096 and 8192.
- From that, a table: the largest n that fits each common limit (`maxBufferSize` and `maxStorageBufferBindingSize` of 128 MiB, 256 MiB, 1 GiB, 2 GiB). Plan 07's device matrix and plan 06's long-chunk warning use it.
- Done when: the table is in the research doc.

### Step 9. Publish (the gate)

- Read the Vocos weights' licence and the F5-TTS licence terms from their sources.
- Draft the model card in `build/webgpu/publish/README.md`. It covers:
  - the licence tags;
  - credits to the F5-TTS authors, Emilia and Vocos, and links;
  - the modifications made;
  - the non-commercial statement;
  - what each variant is and how it was checked, with the quality numbers;
  - how to rebuild the files.
- Stop, and put the draft card and the repo name to the user.
- Once the user approves both: `publish_weights.py upload`, with the user's token. The upload runs only with that approval, given at that point.
- Then the real thing: load from the pinned Hugging Face revision in Chrome on both GPUs, and check that CORS, the redirects and resume work against the real host.
- Add the README line from D6 to the app.
- Done when: the user has approved the card, the repo is up, and a load from it works on both GPUs. Or, if the wording is still open, everything else is done and the step waits.

### Step 10. Close out

- Hand-off to plan 06: `loadWeights`, its progress events and errors, the quota and persistence behaviour, the secure-context requirement, and the variant rule.
- Hand-off to plan 07: speed per variant and GPU, the buffer-limit table, and what step 1 found about ORT's kernels, such as fp16 accumulation, which might be worth a custom kernel.
- `docs/debugging.md` entries for anything surprising.
- Update the task list.

## Risks

- **R1. fp16 on the WebGPU EP is too inaccurate, and no block list fixes it cheaply.** Then `fp16` and `mixed` are dropped, a storage variant ships, and the Intel GPU stays at fp32 speed. Plan 07 could still try an fp32-accumulating matmul kernel.
- **R2. `MatMulNBits` on WebGPU is CPU-only, fp16-only or inexact.** Then `q8` and `q4` are dropped, and `w8` is the small variant. Step 1 finds out before anything is built.
- **R3. Weight error that the metrics miss.** WER and SIM may not catch prosody or noise-floor changes from 4-bit or 8-bit weights. Listening is the backstop, which is why step 5 gives the user the worst cases.
- **R4. Memory while loading.** Tensor-aligned shards avoid one big buffer, but ORT may still hold the data in wasm memory while it uploads. For the storage variants, the loader also holds every widened shard of a graph at once. Step 3 measures peak browser RAM on the Intel GPU, which shares system memory. If `Blob`s from OPFS beat `Uint8Array`s there, the cache hands over `Blob`s.
- **R5. Hugging Face's CDN behaviour changes**, for example around Xet storage, redirects or CORS. The shards don't depend on `Range`, the sizes come from the manifest, and the pinned revision keeps contents fixed. Step 9 tests against the real host.
- **R6. Golden data disk use.** About 3 GB per approximated variant, on top of about 9 GB today. Only two cases per variant, and variants dropped in step 5 can have theirs deleted.
- **R7. The gate does not close.** Delivery is built and tested against a local server, so only the final upload and the real-host test wait on it.
- **R8. ORT's source stays unreachable.** Kernel behaviour is known only from probes. A newer ORT Web may change it silently, which is why the manifest records the ORT version each variant was validated with.

## Pressure test, 2026-11-23

A fresh agent read only this file and checked it against the repo, the installed packages and public documentation. Findings, and what changed:

1. **`w16` as first written rounded tensors that are not weights** (wrong). The data files also hold the rotary tables, `freqs_cis` and the time-embedding frequencies. Rounding the frequencies alone breaks `time_embed`'s tolerance many times over. Now only checkpoint tensors are approximated, and D2 says which.
2. **The quantized sizes were wrong** (wrong). 142M of the step graph's values are in `Gemm`, which the quantizer skips by default. D1 now has a `Gemm` rewrite, both sets of sizes, and a note on the AdaLN weights.
3. **ORT's source is gone from GitHub** (gap), so step 1 could not read it. Step 1 now probes, starting with the K = 4096 all-ones test for fp16 accumulation.
4. **The quantized variants were described as both fp16 and fp32** (wrong). They are now fp32 activations, `accuracy_level` unset, and not put through the fp16 converter.
5. **Golden data for quantized weights** (risk). Read back from the built files, not reimplemented. The agent checked the dequantization formula on ORT CPU.
6. **The fp16 converter changes more than precision** (gap): the 1e-7 clamp, the time embedding, and the ODE update. `op_block_list` replaces the default. `mixed`'s first block list now covers these.
7. **Shards reassembled into one buffer** (risk). ORT mounts each `externalData` entry separately, so shards are now real tensor-aligned data files.
8. **OPFS support was out of date** (minor). Corrected, with the secure-context requirement, Firefox's prompt and Safari's eviction.
9. **Rehashing on every load, and eviction by variant** (gap). Now hashed once on write, repaired on failure, evicted by manifest, and fetched with `no-store`.
10. **Hugging Face** (risk): `Range` across the Xet redirect, hidden headers, and rate limits. Shards avoid `Range`, sizes come from the manifest, and 429 and failed redirects are handled.
11. **IR version** (minor). onnx 1.23.2 writes IR 14, which ORT 1.31.1 refuses. Step 2 checks for 13 or below.
12. **Integrity and versioning** (gap). The manifest hash is in the app build, and the manifest records the ORT version and both checkpoints' hashes. The card states the modifications.
13. **Smaller gaps**, each added:
    - a storage-only `w8` variant;
    - lint and type checks in the "done when";
    - `vocos.onnx` needing external data for `w16`;
    - a time budget for the quality runs.

Checked and found right:
- the graph and checkpoint sizes;
- the largest tensor;
- the quantizer's bit widths and options;
- the converter's options;
- `externalData` taking URLs, bytes or Blobs;
- the device-wrapping approach;
- per-shard hashing;
- both GPUs' limits and `shader-f16`;
- the upstream licence statement;
- D5's bounds;
- the fp32 tolerances;
- the golden data sizes.
