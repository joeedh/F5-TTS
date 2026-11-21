# Plan 02: Mel spectrogram, reference audio and vocoder

Status: `done` 2026-11-21 (written 2026-11-18; review findings folded in, see the end of this file). Depends on: plan 00 (done). Blocks: plans 04 and 05. Hand-offs to plans 04, 05 and 06 are under step 6.

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). Golden data and tolerances: [webgpu-00-scope-and-parity.md](webgpu-00-scope-and-parity.md).

## What this plan delivers

- TypeScript for the audio ends of the pipeline:
  - Reference preparation: silence trim and 12 s clip, downmix, RMS raise, resample to 24 kHz.
  - Mel extraction.
  - The inverse STFT and the output RMS rescale.
- A Vocos ONNX graph (backbone plus the head's linear layer) that runs on ORT Web's WebGPU EP.
- Parity against the golden `prep`, `audio`, `mel` and `vocoder` stages, in Node and in Chrome.
- Tolerances for the stages plan 00 left to this plan (`prep`, `audio.resample`), plus a tight `vocoder.istft` check.

Out of scope:
- Decoding compressed audio files (mp3, m4a): the browser's decoder is used, and its output is the input to this plan. Plan 06 owns file handling.
- Cross-fading chunks (plan 04).
- The vocoder in fp16 (plan 05).

## What upstream does

### Reference preparation (`infer/utils_infer.py`)

1. `preprocess_ref_audio_text`, which uses pydub on the file:
   - `split_on_silence` (min 1000 ms of silence at -50 dBFS, keep 1000 ms, seek step 10 ms), with segments concatenated until the result would pass 12 s once it is over 6 s.
   - If still over 12 s, the same with 100 ms at -40 dBFS.
   - If still over 12 s, a hard cut at 12 s (`utils_infer.py:355`). No golden case reaches it.
   - Then `remove_silence_edges` (leading and trailing silence below -42 dBFS) and `AudioSegment.silent(duration=50)` appended.
     - That silence is made at pydub's default 11025 Hz (551 frames). `append` converts it to the input's rate with `audioop.ratecv`, so it is 1198 samples at 24 kHz and 2201 at 44.1 kHz, not 50 ms exactly. The golden `prep.audio` files confirm both numbers.
     - `append` syncs to the higher of the two rates. An input below 11025 Hz is therefore upsampled as a whole by `ratecv`'s linear interpolation.
   - Writes a wav and fixes up the reference text's final punctuation. The text fix-up belongs to plan 03.
   - pydub keeps the channels. `en-prep`'s `prep.audio` is stereo, shape (2, 304947).
2. `infer_batch_process`, after that:
   - Averages channels to mono.
   - Computes `rms = sqrt(mean(x^2))` and, if it is below 0.1, scales to 0.1.
   - Resamples to 24 kHz with `torchaudio.transforms.Resample` if needed (`sinc_interp_hann`, `lowpass_filter_width` 6, `rolloff` 0.99). torchaudio builds the kernel in float64 and caches it as float32 (`_transforms.py:931-936`).
   - The same `rms` later rescales the generated wav: `wave * rms / 0.1` when the raise fired.
- pydub works on integer samples at the file's own sample width. For a 16-bit wav, the golden `prep.audio` is int16 PCM read back as float: every value is a multiple of 1/32768.

### pydub details that exact parity depends on

- `audioop.rms` returns an integer: the square root of the mean square, truncated.
- `detect_silence` compares that integer with `db_to_float(thresh) * max_possible_amplitude` (`silence.py:29,44`), a linear comparison. Only `detect_leading_silence` compares in dBFS, where an rms of 0 gives minus infinity.
- Milliseconds become frames as `int(ms * (frame_rate / 1000.0))` (`audio_segment.py:445-450`, `:991-997`). That is a truncated double-precision expression, and 44.1 is not exact in binary. The port must evaluate the same expression, not exact integer arithmetic.
- `len(segment)` in ms uses Python's `round`, which rounds half to even (`:283`). JS `Math.round` rounds half up. At 24 kHz they disagree whenever the frame count is 12 mod 24.
- `segment[a:b]` pads with up to 2 ms of zero frames when the rounded length overshoots the data (`:319-328`).
- RMS runs over the interleaved samples of all channels; slicing is per frame.
- Sample width: 24-bit wavs become 32-bit, with `max_possible_amplitude` 2^31 (`:235-246`). Float wavs are refused by the fast path (`:118`) and go through ffmpeg as `pcm_s32le` (`:720-735`).

### Mel (`model/modules.py`, `get_vocos_mel_spectrogram`)

- `torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100, power=1, center=True, normalized=False, norm=None)`. The rest are torchaudio defaults:
  - Periodic Hann window (`torch.hann_window`).
  - Reflect padding of n_fft/2 on both sides.
  - HTK mel scale, f_min 0, f_max 12000.
  - The filterbank is not normalised.
- Output: `log(clamp(mel, 1e-5))`, shape (1, 100, samples // 256 + 1).

### Vocoder (`vocos`, `charactr/vocos-mel-24khz`)

- `VocosBackbone`:
  - Conv1d 100 to 512 (kernel 7), then LayerNorm.
  - 8 ConvNeXt blocks of 512: depthwise kernel 7, LayerNorm, 512 to 1536, exact (erf) GELU, 1536 to 512, layer scale. The DiT uses the tanh GELU; this one does not.
  - A final LayerNorm.
  - Output (1, T, 512).
- `ISTFTHead`, with `padding: center` from the model's HF `config.yaml`. The library default is `same` (`heads.py:38`), which would give T * 256 samples.
  - Linear 512 to 1026.
  - The first 513 values are a log magnitude: `exp`, then clipped at 100. The rest are a phase.
  - `S = mag * (cos p + i sin p)`. The phase of bins 0 and 512 is generally nonzero, and `irfft` ignores the imaginary part of those two bins.
  - `torch.istft(S, n_fft=1024, hop=256, win=1024, hann, center=True)`.
  - Output length (T - 1) * 256.
- Golden taps: `c*.mel_out` (the input), `c*.vocoder.backbone`, `c*.vocoder.head_out` (the linear output) and `c*.vocoder.wav` (after the RMS rescale).
- The tiny Vocos is dim 64, intermediate 192, 2 layers (`tools/webgpu/tiny_model.py:41-49`).

## Decisions

### D1. Where the mel runs (proposal: TypeScript on the CPU)

- The reference mel is computed once per generation, for at most about 12 s of audio. That is about 1,130 frames of a 1024-point real FFT, a few milliseconds in JS.
- On the GPU it would need an STFT op (ORT Web's WebGPU EP support is unknown) or WGSL. Neither is worth it for a one-off.
- The filterbank is computed in TS with torchaudio's formula and checked against torchaudio's matrix, which is exported once as a test fixture.
- The FFT runs in float64. The golden mel is float32 PyTorch, and float64 is at least as close to the exact result.

### D2. Where the inverse STFT runs (proposal: TypeScript on the CPU first, GPU only if profiling says so)

- **Option A.** ONNX outputs `head_out`. JS reads it back and does `exp`, clip, `cos`/`sin`, an inverse real FFT per frame, the window, overlap-add and the window-envelope division.
  - For a 2000-frame chunk the readback is 2000 x 1026 floats (8 MB), and the CPU work is about 2000 FFTs of 1024 points.
  - Simple, and exact in float64.
- **Option B.** The iSTFT inside the ONNX graph as a convolution with a fixed inverse-DFT kernel, as `runtime/triton_trtllm/scripts/conv_stft.py` does for TensorRT. That keeps everything on the GPU, but the edge handling has to be checked against `torch.istft(center=True)`.
- Proposal: A first, timed. B only if A costs more than 5 percent of a chunk's time on the integrated GPU.
- Decided 2026-11-21: option A. Readback plus istft is about 75 ms per 1,448-frame chunk on both GPUs, 0.1 to 0.25 percent of the chunk's time (step 5).

### D3. Vocoder graph boundary (proposal: backbone plus `head.out`, outputs `head_out`)

- The graph's output is exactly the golden `c*.vocoder.head_out` tap, so the parity runner checks it directly (`vocoder.stft_complex`).
- The `exp`/clip and the trigonometry happen in JS with D2's option A. If D2 moves to option B they join the graph.

### D4. Resampler (proposal: port torchaudio's `sinc_interp_hann` resampler to TS)

- Web Audio can resample (`OfflineAudioContext`), but its algorithm is browser-specific and would not match the golden `audio.ref`.
- torchaudio's resampler is a fixed windowed-sinc kernel: about 60 lines with the kernel construction.
  - The TS kernel is built in float64 and each tap rounded with `Math.fround`, to match torchaudio's float32 cache.
- Tolerance: plan 00's rule on the measured CPU-against-CUDA floor (1.8e-7 max abs) gives 1e-5, the value already in `tolerances.json`. (This line said 1e-6 until 2026-11-21; ten times the floor rounds up to 1e-5.) The port must meet that; the tolerance is not re-derived from the port.
- The input is decoded PCM at its own sample rate. Plan 06 must decode without the browser resampling it. For example, use `decodeAudioData` on an `AudioContext` created at the file's rate, or parse wav directly.

### D5. Silence trim and clip (proposal: port pydub's algorithm, exact for 16-bit PCM wav)

- R3 in plan 00 said the browser "cannot match exactly". On reading pydub, it can, for 16-bit PCM wav at 11025 Hz or above, provided the port reproduces every detail in "pydub details that exact parity depends on" above. That list is the exactness contract.
- Scope of exact parity:
  - 16-bit PCM wav, any channel count, 11025 Hz or above.
  - Everything else (24-bit and float wav, compressed formats, rates below 11025 Hz) is approximate. (Step 2 made rates below 11025 Hz exact too, by porting `ratecv` in full.) The port converts it to int16 and proceeds, and the UI does not promise identical output. With 32-bit samples the sum of squares is also no longer exact in float64.
- Input: interleaved int16 samples plus `channels` and `sampleRate`.
- The appended silence reproduces `ratecv`'s output length for 551 zero frames at 11025 Hz converted to the input rate. The values are all zeros, so only the length matters.
- Performance: `detect_silence` rescans a 1000 ms window every 10 ms, which is billions of operations for a multi-minute upload. The port uses a prefix sum of squared samples. For int16 the sums stay below 2^53, so float64 is exact and the result is identical, in linear time.
- Check: the port's output against `prep.audio`, exact length and values. New check `prep.audio`, exact, in `tolerances.json` and both runners' `checksFor`.

## Steps

### Step 1. Mel

- `web/src/audio/mel.ts`:
  - Reflect pad, periodic Hann, real FFT (a small radix-2 implementation; no dependency needed).
  - Magnitude, the HTK mel filterbank, `log(max(x, 1e-5))`.
  - Output layout (100, frames), the same as the golden (1, 100, n) without the batch axis.
- Test fixture: torchaudio's filterbank (513 x 100) exported by a small `tools/webgpu/export_audio_fixtures.py` and committed (200 KB).
- Tests: Vitest against `mel.ref` of the tiny case (committed) and, locally, of every real case, with `mel.log_masked` and `mel.linear`.
- Done when: all cases pass both mel checks.
- Result, 2026-11-21:
  - Code: `web/src/audio/fft.ts` (radix-2 complex FFT in float64, reused by step 4) and `web/src/audio/mel.ts` (`logMel`, `melFilterbank`, `hannWindow`). Tests in `web/test/audio/mel.test.ts`.
  - Fixture: `testdata/webgpu/audio-fixtures/mel_fbank.npy`, the `mel_scale.fb` of the same `MelSpectrogram` upstream builds, written by `tools/webgpu/export_audio_fixtures.py` (205 KB).
  - The filterbank reproduces torchaudio's float32 arithmetic (`Math.fround` at each step, and `torch.linspace`'s two-ended fill), not D1's plain float64. With float64, `mel.linear` was 1.9e-6, 10 times the CPU-against-CUDA floor (1.9e-7); with torchaudio's own matrix it was 1.4e-7. The float64 matrix is 9.0e-6 away from torchaudio's, and that is the whole gap (`docs/debugging.md`).
  - The emulation matches the fixture to 3.8e-6 at worst, with the same zero pattern. The rest is torch's vectorised float32 `pow`, which rounds 5 of the 102 centre frequencies differently from a correctly rounded one. Pinning those 5 values would close it; not done, since the mel is already far inside its tolerance.
  - All 7 cases pass, the tiny one in CI. Worst over the real cases: `mel.linear` rel L2 4.2e-7 (tolerance 1e-5), `mel.log_masked` max abs 1.6e-3 (tolerance 0.1; zh-short).
  - Speed in Node: about 5 ms per second of audio, so about 70 ms for a 12 s reference. Most of it is the dense filterbank product; skipping its zeros would cut it if it ever matters.

### Step 2. Reference preparation

- `web/src/audio/prepare.ts`:
  - `trimAndClip(samples: Int16Array, channels, sampleRate)`: D5.
  - `downmixRmsResample(channels, sampleRate)`: mean, RMS raise, D4.
  - Returns the 24 kHz mono reference and the `rms`/`raised` values plan 04 needs for the output rescale.
- Golden inputs: there are 4 distinct `input.wav` files, not 7. `en-short`, `en-multi`, `en-multi-fixdur` and `en-epss16` share one. None of them reaches the 12 s hard cut, and the committed `tiny` case (24 kHz mono, rms 0.147) triggers neither the resample, the downmix nor the RMS raise.
- Synthetic fixtures, generated with pydub and torchaudio by `export_audio_fixtures.py` and committed (each under 1 s unless the case needs more):
  - A quiet 44.1 kHz stereo clip: resample, downmix and RMS raise.
  - Silence only.
  - No silence and longer than 12 s: the hard cut.
  - A long silence in the middle.
  - One fixture for each item of D5's exactness contract that the above miss. These are mainly frame counts at 12 mod 24, where Python's `round` and JS's `Math.round` disagree, and a slice that overshoots and gets padded.
- Tests:
  - `prep.audio` exactly, from `input.wav` for every golden case and every fixture.
  - `audio.ref` from `prep.audio`, with `audio.resample` at 1e-5 (D4).
- Done when: every golden case and fixture passes both, and the fixtures run in CI.
- Result, 2026-11-21:
  - Code, all in `web/src/audio/`:
    - `pydub.ts`: `Segment` (pydub's slicing, length, rms, dBFS, append) plus `audioop.ratecv`, `split_on_silence`, `detect_nonsilent`, `detect_silence` and `detect_leading_silence`.
    - `prepare.ts`: `trimAndClip` and `downmixRmsResample`, in upstream's order.
    - `resample.ts`: torchaudio's sinc resampler.
    - `wav.ts`: a 16-bit PCM wav reader, used by the tests; plan 06 can reuse it.
  - Golden cases: all 7 give `prep.audio` exactly. `audio.ref` passes on all of them. On en-prep, the only one that resamples, the worst error is 1.8e-7, equal to the CPU-against-CUDA floor. The other cases match exactly, since they need no resample. `rms` is within one float32 ulp, and `raised` matches.
  - `audio.resample` is no longer informational, in both runners. `prep.audio` is an exact check in `tolerances.json`, `measure_floor.py` and both runners' `checksFor`.
  - Fixtures: `tools/webgpu/export_audio_fixtures.py` writes 9 synthetic 16-bit wavs with upstream's output for each (`testdata/webgpu/audio-fixtures/prep/`, 2.8 MB, deterministic). All 9 match exactly in CI:
    - `quiet-44k-stereo`: downmix, RMS raise, resample, and overlapping `keep_silence`.
    - `silence-only`: nothing survives the split, so the output is pydub's 50 ms at 11025 Hz (551 frames).
    - `hard-cut`: the 12 s cut.
    - `join-6-12`: the join loop stops after 9.25 s.
    - `mid-silence`: a long silence in the middle.
    - `thresholds`: rms 103 against 104 at -50 dBFS, and 260 against 261 at -42 dBFS.
    - `round-half-down` and `round-half-up`: 750.5 and 751.5 ms, where Python's `round` and JS's `Math.round` disagree. The second also overshoots and is padded.
    - `low-rate-8k`: pydub upsamples it to 11025 Hz with `ratecv`.
  - Two of them also check the resampled reference against torchaudio: `quiet-44k-stereo` and `low-rate-8k`. `silence-only` cannot, see below.
  - Each contract rule was checked by breaking it in the port and seeing a fixture fail: `Math.round` for `round`, no slice padding, an integer threshold, and a dB offset.
  - The fixtures are longer than the 1 s the plan aimed for: the 12 s rules need more than 12 s. They use 11025 Hz, the lowest exact rate, to stay small.
  - Exact parity now also covers rates below 11025 Hz, since `ratecv` is ported in full.
  - Found: a silent reference makes upstream divide by an rms of 0, so its `audio.ref` is all NaN. Plan 06 should refuse such input; the port reproduces the NaN.
  - Speed in Node, en-prep: `trimAndClip` takes 7 ms on the 16 s 44.1 kHz stereo input, and `downmixRmsResample` 28 ms on the 7 s result.

### Step 3. Vocoder export and CPU parity

- `tools/webgpu/export_vocos.py`:
  - Wraps `vocos.backbone` and `vocos.head.out`.
  - Dynamic T, dynamo, a single file (about 54 MB, no external data needed).
  - Also exports the tiny Vocos for CI (committed). It is built from `build_tiny_vocos()` and the committed `vocos.safetensors`, not from `from_pretrained`.
  - Logs which op the exact GELU became (opset-20 `Gelu`, or an `Erf` decomposition), for step 5's placement check.
- Python check in onnxruntime CPU: feed `c*.mel_out`, compare with `c*.vocoder.backbone` (a debug output) and `c*.vocoder.head_out` through `parity.py`.
- Done when: every chunk of every case passes `vocoder.backbone` and `vocoder.stft_complex`.
- Result, 2026-11-21:
  - `tools/webgpu/export_vocos.py` writes `vocos.onnx` (`mel` (1, 100, T) to `head_out` (1, T, 1026)) and `vocos_debug.onnx` (also `backbone` (1, T, 512)). T is dynamic, from 8 to 16384.
    - Tiny: `testdata/webgpu/tiny/onnx/`, 0.7 MB each, committed.
    - Real: `build/webgpu/vocos/`, 54.3 MB each, one file with no external data.
  - The exact GELU exports as opset 20 `Gelu` with `approximate="none"`, one per ConvNeXt block (8), not an `Erf` decomposition. Step 5 checks it is placed on the GPU (R6).
  - The real graph is 96 nodes: `Conv` 9, `Transpose` 19, `LayerNormalization` 10, `MatMul` 17, `Add` 25, `Gelu` 8, `Mul` 8.
  - `tools/webgpu/check_vocos_ort.py` (onnxruntime CPU, golden `c*.mel_out` in): every chunk of all 7 cases passes. Worst over the real cases, all in en-multi-fixdur chunk 0:
    - `vocoder.backbone`: rel L2 1.3e-5 (tolerance 1e-3).
    - `vocoder.stft_complex`: rel L2 2.6e-4 (tolerance 1e-2).
    - Both are below the CPU-against-CUDA floors (4.3e-5 and 7.9e-4).
  - The export is deterministic. Node metadata records source line numbers, though, so editing the exporter changes the files' bytes.

### Step 4. Inverse STFT and output rescale

- `web/src/audio/istft.ts`: D2 option A, with the `torch.istft(center=True)` semantics:
  - Inverse real FFT per frame, ignoring the imaginary part of bins 0 and 512 as `irfft` does.
  - Periodic Hann window.
  - Overlap-add at hop 256.
  - Divide by the sum of squared windows.
  - Trim n_fft/2 at both ends.
- `web/src/audio/output.ts`: the RMS rescale (`wave * rms / 0.1` when `raised`).
- New check `vocoder.istft`, rel_l2 1e-5: from the golden `head_out` to `vocoder.wav`. The existing `vocoder.wav` tolerance (rel_l2 1e-2, max_abs 0.1) was sized for the whole vocoder on CPU against CUDA, and is too loose to catch a wrong window or envelope here. `vocoder.istft` is used only by this test.
- Tests:
  - Every chunk of every case, from `c*.vocoder.head_out`.
  - A synthetic frame with nonzero phase in bins 0 and 512, compared with `torch.fft.irfft`.
- Done when: every chunk passes `vocoder.istft`, and the synthetic frame matches.
- Result, 2026-11-21:
  - Code: `web/src/audio/istft.ts` (`istft`, `irfftFrame`) and `web/src/audio/output.ts` (`rescaleOutput`, float32 like upstream). Tests in `web/test/audio/istft.test.ts`.
  - `vocoder.istft` (rel L2 1e-5) is a fixed entry in `measure_floor.py` (`FIXED`), so regenerating `tolerances.json` keeps it. `--from-floors` reproduces the committed file exactly, step 2's changes included.
  - Every chunk of all 7 cases passes, at rel L2 1.1e-7 to 1.2e-7 and max abs 3e-7 at worst: the float32 rounding of the output.
  - The synthetic fixture (`testdata/webgpu/audio-fixtures/istft/`, from `export_audio_fixtures.py`) has 6 frames, nonzero DC and Nyquist phases, and one bin clipped at 100. It matches torch's float64 `irfft` and `istft` to rel L2 below 1e-6.
  - Mutations caught: a symmetric Hann window fails 8 of the 9 tests, and a constant envelope 7. A 0.2 percent ripple gives rel L2 1.4e-3, which `vocoder.wav`'s tolerance (1e-2) would pass and `vocoder.istft` fails.
  - Keeping the imaginary part of bin 0 changes nothing here: the port builds the full Hermitian spectrum and keeps only the real part of the inverse FFT, so that term cannot leak. The fixture guards a future packed real FFT, where it would.
  - Speed in Node: about 75 microseconds per frame, so 110 ms for a 1,450-frame chunk and about 150 ms for 2,000 frames. Against a chunk's 30 to 90 s of DiT steps (plan 01), that is well under D2's 5 percent. Step 5 adds the readback in the browser and decides D2.

### Step 5. Browser

- Run the vocoder graph on WebGPU (both GPUs, Chrome) from the golden `mel_out`, then the TS iSTFT, then compare with `vocoder.wav`.
- The tiny case runs in CI (SwiftShader); the real cases run locally.
- Time the graph and the iSTFT per chunk on both GPUs, for D2.
- Placement: no float compute node on the CPU and no inserted copy of an activation tensor. The GELU's op from step 3 is checked by name.
- Done when: the real cases pass on both GPUs, and D2 is decided with numbers.
- Result, 2026-11-21 (Chrome, ORT Web 1.31.1):
  - Code:
    - `web/src/vocoder/` (`Vocoder`: `create`, `run`, `decode`, `release`). `decode` returns the audio before the RMS rescale, and its timing.
    - `web/test/vocoder/run-parity.ts`, the browser twin of `check_vocos_ort.py` plus the istft and rescale.
    - `test/browser/vocoder-tiny.spec.ts` (CI) and `test/browser-local/vocoder.spec.ts` (`VOCODER_CASES`).
    - `vite.config.ts` serves `build/webgpu/vocos/` at `/model/vocos/`.
    - The DiT runner's placement parsing and error spread moved to `web/test/ort/diagnostics.ts` (`withPlacement`, `parsePlacement`, `errorSpread`), shared by both runners.
  - Parity: every chunk of all 6 real cases passes `vocoder.backbone`, `vocoder.stft_complex` and `vocoder.wav` on both GPUs. The tiny case passes in the CI test.
    - Worst, as a fraction of the tolerance: `vocoder.stft_complex` 5.0 percent on NVIDIA and 4.6 percent on Intel, and `vocoder.wav` 4.3 and 3.9 percent, all at en-short chunk 0 (the same mel as en-multi's).
  - Placement: all 69 nodes on the WebGPU EP, none on the CPU, on both GPUs. The 8 exact `Gelu` nodes run on the GPU, so R6's fallback is not needed.
  - Time per chunk, median of 5 (ms):

    | GPU | frames | graph | readback | istft |
    |---|---|---|---|---|
    | NVIDIA | 1448 | 15 | 6 | 70 |
    | NVIDIA | 470 | 7.5 | 4.7 | 21 |
    | Intel | 1448 | 47 | 5 | 67 to 73 |
    | Intel | 470 | 18.5 | 4.4 | 20 |

    The graph time is `session.run` until the GPU finishes. The readback copies 1448 x 1026 floats (5.9 MB).
  - **D2 decided: option A.** The readback and the istft together cost about 75 ms for a 1,448-frame chunk on either GPU. A chunk's DiT steps take about 30 s on NVIDIA and 85 s on Intel (plan 01), so that is 0.1 to 0.25 percent, far below the 5 percent that would call for option B.
  - The istft is still the largest part of the vocoder's time, 4 to 5 times the graph on NVIDIA. If plan 07 wants it gone from the critical path, there are two ways. It can run in a worker while the next chunk's DiT steps run. Or a packed real FFT would halve it.

### Step 6. Close out

- `docs/research/webgpu-audio.md` (add it to `docs/index.md`): the mel and filterbank spec as implemented, the pydub exactness contract, the iSTFT timing and D2's outcome, and the vocoder graph's per-chunk time.
- Hand off to plan 04: the function signatures and the `rms`/`raised` contract.
- Hand off to plan 05: the vocoder's size and fp32 timing.
- Hand off to plan 06: the input formats that get exact parity, and the decoding requirement in D4.
- `docs/debugging.md` entries; `docs/shape-ledger.md` entries for layout slips. The mel is (b, d, n) and the backbone output is (b, n, d), so this boundary is where they happen.
- Result, 2026-11-21: plan 02 is done. Findings are in `docs/research/webgpu-audio.md`; the hand-offs are below.
  - No `docs/shape-ledger.md` entries. No layout slip happened at the mel/backbone boundary: the golden layouts are fixed in the manifest, and `compare` refuses a wrong shape before it compares values. The one layout hazard left is plan 04's, below.
  - `docs/debugging.md` has three entries from this plan:
    - The float32 filterbank.
    - The all-NaN reference for silent input.
    - The exporter's emoji on a Windows pipe.

### Hand-off to plan 04 (sampler and pipeline)

- Functions, all synchronous except the vocoder:
  - `trimAndClip(pcm: Pcm): Pcm` and `downmixRmsResample(pcm: Pcm): Reference` in `web/src/audio/prepare.ts`. `Reference` is `{ audio: Float32Array (24 kHz mono), rms, raised }`.
  - `logMel(audio): { data, frames }` in `mel.ts`. `data` is (100, frames) row-major, mel-major.
  - `Vocoder.create({ url })`, then `decode(mel: ort.Tensor)` giving `{ wave, timing }` (`web/src/vocoder/`). The mel is (1, 100, T); `wave` is (T - 1) * 256 samples, before the rescale.
  - `rescaleOutput(wave, reference)` in `output.ts`.
- The `rms`/`raised` contract: `rms` is measured on the downmixed reference before any raise, in float32. When `raised`, the reference is scaled up to RMS 0.1, and every generated chunk must be scaled by `rms / 0.1` after the vocoder (`rescaleOutput`). Keep the `Reference` for the whole generation.
- Layouts: `logMel` gives (100, frames), but the DiT's `step_cond` and `x` are (1, n, 100) (plan 01). The vocoder wants (1, 100, T) again, cropped to the generated frames. So the pipeline transposes twice. It also crops `ref_audio_len = samples / 256` frames, rounded down, which is one fewer than the reference mel has (plan 01's frame-0 note).
- Cost per chunk: under 100 ms for the vocoder, readback and istft together on either GPU. `decode` reports the split.

### Hand-off to plan 05 (weights and precision)

- `vocos.onnx`: 54.3 MB fp32, one file, no external data. Rebuilt by `tools/webgpu/export_vocos.py --model base`.
- fp32 time per 1,448-frame chunk: graph 15 ms on NVIDIA and 47 ms on Intel; readback 5 to 6 ms. fp16 would save little time here; its case is download size.
- The exact GELU runs on WebGPU as `Gelu`; an fp16 export must keep it exact (`approximate="none"`).

### Hand-off to plan 06 (web app)

- Exact parity with upstream holds for 16-bit PCM wav at any rate and channel count, through `parseWav`.
- Anything else (24-bit, 32-bit or float wav, mp3, m4a) must be decoded and converted to int16 by the app. The output may then differ from upstream's.
- Decode at the file's own rate. The browser must not resample: either an `AudioContext` created at the file's rate, or `parseWav` for wav. The port's API takes the rate explicitly, so a mismatch shows (D4, R5).
- Refuse a reference with no sound: upstream divides by an rms of 0 and produces NaN, and the port reproduces it.

## Risks

- **R1. The torchaudio filterbank details** (HTK scale, f_max, normalisation, the triangle edges at bin boundaries) are easy to get subtly wrong. Mitigation: the exported matrix is the reference, and it is compared before the mel is.
- **R2. pydub edge cases** beyond the exactness contract. One example: `split_on_silence` with `keep_silence` overlaps neighbouring chunks, and its handling is order-dependent. The exact `prep.audio` check catches any difference, but only on the inputs it is given, which is why step 2 adds the synthetic fixtures.
- **R3. Float64 in TS against float32 in PyTorch.** Expected to help, not hurt. If a check fails by a hair, compare with a float32 path before suspecting a bug.
- **R4. The iSTFT readback** (8 MB per long chunk) may be slow on some browsers. D2 option B exists for that.
- **R5. Browser decoding resamples silently** if the `AudioContext` rate differs from the file's. That is plan 06's to handle, but this plan's API takes the sample rate explicitly, so a mismatch is visible.
- **R6. The exact GELU's ONNX form** may not run on the WebGPU EP. If `Erf` lands on the CPU, the fallback is an export-time substitution with the same numerics. A tanh approximation would not match.

## Pressure test, 2026-11-18

A fresh agent given only this file reviewed the first draft against the code, pydub, torchaudio, vocos and the golden data. Findings and what changed:
- **The appended "50 ms" is 1198 samples at 24 kHz and 2201 at 44.1 kHz.** pydub makes it at 11025 Hz and converts it with `ratecv`. Confirmed against every golden `prep.audio`. Now stated, with the low-rate upsampling it implies.
- **"pydub only slices, so nothing can drift" was wrong.** These details all matter for exactness:
  - Integer RMS and the linear threshold comparison.
  - Truncated double-precision ms-to-frame conversion.
  - Python's round-half-even.
  - Zero padding on overshooting slices.
  
  They are now listed as the exactness contract, each with a test.
- **Format handling was wrong.** pydub keeps 24-bit audio as 32-bit and sends float wav through ffmpeg, so it does not convert to int16. Exact parity is now limited to 16-bit PCM wav at 11025 Hz or above. The port's API takes interleaved samples and a channel count, because `en-prep` is stereo.
- **The iSTFT check was too loose to catch a wrong window or envelope.** A new `vocoder.istft` check at rel_l2 1e-5 is used for the head-to-wav test.
- **The resample tolerance was circular.** It is now the floor-derived 1e-5 (first written here as 1e-6, corrected 2026-11-21), and the port must meet it. The kernel is rounded to float32 as torchaudio's is.
- **CI exercised none of resample, downmix, RMS raise or clipping,** and the golden set has only 4 distinct inputs. Synthetic fixtures are added, including one that reaches the 12 s hard cut.
- **`detect_silence` is quadratic in practice.** The port uses a prefix sum instead, which gives identical results for int16.
- **Smaller fixes:**
  - DC and Nyquist bins in the iSTFT.
  - The tiny Vocos is built from the committed weights.
  - Exact GELU and its ONNX placement (new risk R6).
  - The `center` padding is cited from the model's config.
  - The downmix happens after pydub.
