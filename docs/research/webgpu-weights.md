# Weights: ORT Web's kernels in fp16, MatMulNBits, Hugging Face downloads and OPFS

Plan 05 step 1, 2026-11-23. Laptop: RTX 5050 and Intel Xe-3 LPG, Chrome 158, ORT Web 1.31.1.

ORT's source could not be read: `github.com/microsoft/onnxruntime` returns 404, also through the GitHub API. Everything about the kernels below is measured, not read.

## How the kernels compute in fp16

**Method.** `tools/webgpu/probe_kernels.py` builds one-op graphs in the shapes and attributes the DiT uses. `web/bench/weights-probe/` runs them on both GPUs; results are in `web/bench/weights-probe/results/`.
- Each graph casts fp32 inputs to fp16, runs the op on fp16 initializers, and casts back. The inputs are chosen to be exactly representable in fp16, so the input cast changes nothing.
- The reference is float64 on exactly those values.
- "Floor" is the error of rounding that reference once to fp16. A kernel that accumulates in fp32 and rounds once reaches it.

**Results.** Relative L2 against the float64 reference, and the same error divided by the floor:

| Probe | NVIDIA | Intel | times the floor |
|---|---|---|---|
| fp32 `MatMul`, K = 1024 (control) | 5.8e-7 | 5.8e-7 | |
| fp16 `MatMul`, K = 1024 | 4.7e-3 | 4.7e-3 | 23 |
| fp16 `MatMul`, K = 2048 | 6.6e-3 | 6.6e-3 | 32 |
| fp16 `Gemm` (transB), K = 1024, N = 6144 | 2.4e-3 | 2.4e-3 | 12 |
| fp16 `Conv`, convpos (1024 channels, kernel 31, 16 groups; K = 1984) | 6.4e-3 | 6.4e-3 | 31 |
| fp16 `Conv`, depthwise (512 channels, kernel 7) | 4.6e-4 | 4.6e-4 | 2.2 |
| fp16 `MultiHeadAttention`, 16 heads, s = 512 | 1.0e-3 | 1.4e-3 | 5 to 7 |
| fp16 `LayerNormalization`, input mean 0 / mean 100 | 3.4e-4 / 4.2e-4 | the same | 1.6 / 2.0 |
| fp16 `Gelu`, tanh / exact | 1.9e-4 / 1.5e-4 | the same | 2.1 / 1.5 |
| fp16 `Mish` | 4.0e-4 | 3.9e-4 | 2.2 |
| fp16 `Sigmoid` | 3.2e-4 | 3.1e-4 | 2.0 |
| fp16 `Sin` | 1.7e-4 | 1.7e-4 | 1.0 |

- **`MatMul`, `Gemm` and the grouped `Conv` accumulate in fp16** (or in fp16 partial sums), on both GPUs. Their error grows with K, to 12 to 32 times the floor.
  - On NVIDIA, the all-ones check (K = 4096, B = 1 + 2^-10) returns 4096 where fp32 accumulation gives 4100.
  - On Intel the same check comes out exact, but its random-input errors equal NVIDIA's to two digits. So the two take the same path, and the all-ones values simply happen to sum exactly there.
- **Attention is 5 to 7 times the floor.** The element-wise ops, LayerNorm and the depthwise conv are at 1 to 2 times, which is fine.
- **LayerNorm keeps its statistics in fp32** (stash_type 1). An input with mean 100 would overflow `sum(x^2)` in fp16; it gives the same error as mean 0.
- **Every probe ran entirely on `WebGpuExecutionProvider`,** with ORT's optimizations on or off, and gave the same numbers either way.
- **What it means for D1:**
  - `fp16` will carry the error of an fp16-accumulating matmul in every block. This is the spike's 8e-3 for a block.
  - A `mixed` variant that keeps `MatMul`, `Gemm` and the convpos `Conv` in fp32 keeps most of the compute in fp32, so it would gain little speed.
  - Step 4 measures what both do to a whole step.
- **ORT CPU is no fp16 baseline.** It has no fp16 `MatMul`, `Gemm`, `Conv` or `MultiHeadAttention` kernel. It inserts casts to fp32 and then removes the Cast pairs, even with graph optimizations off, so its "fp16" output is fp32 and not even on the fp16 grid.
  - The plan 00 spike's "ORT CPU fp16" figures (1.5e-4 for a MatMul) were therefore fp32 compute on fp16-rounded inputs and weights, which is what the spike guessed.

## MatMulNBits

**Method.** Each probe is made by onnxruntime 1.31.1's own quantizer (`MatMulNBitsQuantizer` with `DefaultWeightOnlyQuantConfig`, round to nearest) from a fp32 `MatMul` (K = N = 1024, weights ~ N(0, 0.05)). The reference dequantizes the stored values as `(q - zp) * scale`:
- `B` is [N, blocks, block * bits / 8], low nibble first.
- The scales are [N, blocks], fp32.
- The zero points are packed the same way. When there are none, the zero point is 2^(bits - 1).

**Results:**

| Probe | NVIDIA | Intel | Weight error from quantizing |
|---|---|---|---|
| 4-bit, block 32, fp32 activations | 2.1e-7 | 2.1e-7 | 8.1e-2 |
| 4-bit, block 128, fp32 activations | 1.3e-7 | 1.1e-7 | 1.0e-1 |
| 4-bit, block 32, symmetric, fp32 activations | 2.1e-7 | 2.2e-7 | 8.6e-2 |
| 8-bit, block 32, fp32 activations | 2.1e-7 | 2.2e-7 | 4.8e-3 |
| 8-bit, block 128, fp32 activations | 1.3e-7 | 1.1e-7 | 5.9e-3 |
| 4-bit, block 32, fp16 activations | 1.8e-3 (8.5 x floor) | 1.8e-3 | 8.1e-2 |
| 8-bit, block 32, fp16 activations | 1.8e-3 (8.4 x floor) | 1.8e-3 | 4.8e-3 |

- **With fp32 activations, `MatMulNBits` is exact on WebGPU:** within 2e-7 of `A @ deq`, as good as the fp32 `MatMul`, at 4 and 8 bits, blocks 32 and 128, symmetric or not. It runs on the GPU on both adapters, with `accuracy_level` unset.
  - So `q8` and `q4` stay in D1.
  - Their kernel error will pass D2's fp32 tolerances; their quality is all weight error.
  - They need no `shader-f16`.
- **With fp16 activations, it accumulates in fp16 like `MatMul` does.**
- **The read-back dequantization matches.** It agrees with ORT CPU to 5e-7 and with WebGPU to 2e-7, so D2's golden data can be made from the built files.
- **Weight error:** on these random weights, round-to-nearest costs 8 to 10% at 4 bits and 0.5 to 0.6% at 8 bits. Real weights will differ.

## Device limits ORT asks for

- ORT requests these features: `timestamp-query`, `shader-f16`, `subgroups` and `subgroup-size-control`.
- It requests the adapter's maximum for ten limits: `maxBufferSize` (2 GiB on both GPUs), `maxStorageBufferBindingSize` (2 GiB minus 4), `maxStorageBuffersPerShaderStage` (16), `maxBindGroups`, `maxComputeWorkgroupStorageSize`, and the workgroup sizes and counts.
- So a device that offers less does not fail at creation. It fails when a buffer is too big, which is what step 8 measures.
- **ORT creates a new device whenever it has none.** It destroys its device when the last session is released, so sequential sessions requested a device 22 times in one page (plan 00 saw the same).

## Hugging Face from a page

Probed from `http://127.0.0.1` against `charactr/vocos-mel-24khz` (public), reading only headers and the first chunk:

| Request | Result |
|---|---|
| GET `resolve/main/config.yaml` (a plain file) | 200; redirected within `huggingface.co` |
| GET `resolve/main/pytorch_model.bin` (54 MB, LFS) | 200; redirected to `us.aws.cdn.hf.co` |
| The same with `Range: bytes=0-1023` | 206, `Content-Range: bytes 0-1023/54365991`, through the same redirect |
| HEAD | 200, `Content-Length` 54365991 |

- **CORS:** the CDN sends `Access-Control-Allow-Origin: *` and `Access-Control-Expose-Headers: *`, so the page can read every header.
- **The ETag** is the file's SHA-256 in quotes (`"f2b45ec3…c95ba"`).
- **Range requests work** after the redirect. The forum report from 2026-01 that `Range` fails CORS on `cas-bridge.xethub.hf.co` does not apply to this file today; it was served from `us.aws.cdn.hf.co`. A repo stored in Xet may still be served differently, so step 9 checks the real repo.
- **For D4:** shards stay, for hashing and parallel downloads. But resuming inside a shard with `Range` is possible against today's host.

## OPFS

**`web/bench/weights-probe/` (`probeOPFS`):** 21 files of 64 MiB (1.31 GiB), written and read back, in a persistent Chrome profile launched by Playwright:

| | First run (NVIDIA flag) | Second run (Intel flag) |
|---|---|---|
| Write, `createWritable` on the main thread, then `move()` into place | 3.5 s | 2.9 s |
| Read, `getFile()` then `arrayBuffer()` | 4.4 s | 7.3 s |
| Write, `createSyncAccessHandle` in a worker | 18.2 s | 6.3 s |
| Read, `createSyncAccessHandle` in a worker | 0.41 s | 0.36 s |

- **Quota:** `navigator.storage.estimate()` reported a 10 GB quota with 137 GB free on the disk. A non-persistent Playwright context gives an in-memory OPFS, where the worker's write failed with `QuotaExceededError`. Step 7 checks the quota in a normal Chrome profile.
- `move()` exists.
- **For D5:** write with `createWritable` (a fetch body can be piped into it), and read with a sync access handle in a worker. That read is 10 to 20 times faster than `File.arrayBuffer()`.

**The real step graph through ORT** (`web/bench/dit-load/`, scenarios `opfs-blob` and `opfs-bytes`; results `chrome-*-mha-opfs.json`). en-short's first chunk, the text graph plus `step_mha`. Load is from the start until the step session exists:

| Source | NVIDIA load | Intel load | Peak private memory, load to first steps |
|---|---|---|---|
| URL, ORT fetches it (today) | 6.7 s | 6.4 s | 2.5 GB |
| OPFS, first time: streamed from the server into OPFS, then loaded | 17.8 s | 16.5 s | 3.5 to 3.8 GB |
| OPFS `File` (a Blob) handed to ORT | 2.9 s | 3.7 s | 5.2 to 5.4 GB |
| OPFS read by a worker's sync handle, bytes handed to ORT | 1.8 to 1.9 s | 1.7 s | 3.5 to 5.5 GB |
| Cache Storage, first time (`cache.put`) | 25.1 s | 48.0 s | 3.9 / 3.5 GB |
| Cache Storage, cached | 2.3 s | 2.4 s | 3.9 / 3.5 GB |

- **Correctness:** every scenario passed the step's parity checks (4 of 4).
- **The fastest cached load is OPFS read by a worker,** 1.7 to 1.9 s. Filling OPFS by streaming the response is about 1.5 to 3 times faster than `cache.put`.
- **Memory is the cost.** Handing ORT bytes or a Blob peaks 1 to 3 GB above letting ORT fetch the URL, probably because our buffer and ORT's copy are alive together until garbage collection.
  - Memory was sampled about every 1.5 s from outside the browser, so short peaks may be missed and these are rough.
  - This is plan 05's R4. Steps 3 and 7 measure it again with tensor-aligned shards, handing over one shard at a time and dropping each reference once ORT has it.

## The conversion tools

- **`convert_float_to_float16`** (onnxconverter-common 1.16.0):
  - Its default `op_block_list` is the classic ONNX-ML ops, none of which the DiT uses.
  - It also clamps every constant to `max_finite_val` (1e4) and `min_positive_val` (1e-7). Step 2 checks the graphs for constants beyond either.
- **`MatMulNBitsQuantizer`** quantizes `MatMul` only unless told otherwise (`op_types_to_quantize`), which is why step 2 rewrites `Gemm`.
  - Defaults: block 128, asymmetric, 4 bits, `accuracy_level` unset.
  - Other algorithms: RTN, HQQ, GPTQ and k-quant.
- **IR versions:** onnx 1.23.2 writes IR 14 unless told otherwise, and ORT 1.31.1 refuses it. The probes set IR 10, as the exports do.
- **Tensor-aligned data shards:** onnx's helper writes one data file, or one file per tensor. Shards of up to 64 MiB on tensor boundaries need our own writer, which sets each tensor's `location`, `offset` and `length`. That is step 2's.
