# Plan 04: Sampler and generation pipeline

Status: `plan pressure-tested` (written 2026-11-22; review findings folded in, see the end of this file). D5's thresholds and scope accepted by the user on 2026-11-22. Depends on: plans 01, 02 and 03 (all done). Blocks: plans 06 and 07.

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). Golden data and tolerances: [webgpu-00-scope-and-parity.md](webgpu-00-scope-and-parity.md). The pieces this plan joins up come from the hand-offs in [plan 01](webgpu-01-dit-export.md) (step 5), [plan 02](webgpu-02-mel-and-vocoder.md) (step 6) and [plan 03](webgpu-03-text-frontend.md) (step 7).

## What this plan delivers

- TypeScript that turns a reference wav, its text and a target text into a wav, as `infer_process` does, on WebGPU:
  - The chunk plan: each chunk's duration, the time grid, and the short-chunk speed rule.
  - The noise, from a seed.
  - The conditioning, the ODE loop with classifier-free guidance on the GPU, and the reference-frame restore and crop.
  - The vocoder per chunk, the output rescale, and the cross-fade.
- A module API that plan 06 can put behind a UI and a worker: load, generate, progress, cancel, release.
- Proof against the golden data at each new stage, end to end with the golden noise, and a metric-based quality comparison with PyTorch on part of the Seed-TTS test set.

Not in this plan:
- The UI, the worker, and the asset loading strategy (plan 06).
- Transcribing the reference when its text is empty. Upstream does that with Whisper (`utils_infer.py:366`). The API here refuses an empty reference text, and the task list already has transcription as a plan 06 decision.
- fp16, quantization and weight hosting (plan 05).
- Speed work: batch-1 graphs for cfg 0, overlapping the istft with the next chunk, graph capture (plan 07).
- `remove_silence`, an option of upstream's CLI and API, not of `infer_process`. Plan 02 already ported pydub's `split_on_silence`, so plan 06 can add it cheaply.

## What upstream does

Per generation, `infer_process` and `infer_batch_process` in `infer/utils_infer.py`, then `CFM.sample` in `model/cfm.py`. The settings and their defaults: `nfe_step` 32, `cfg_strength` 2, `sway_sampling_coef` -1 (or `None` for no sway), `speed` 1, `cross_fade_duration` 0.15 s, `fix_duration` none, `target_rms` 0.1. Euler is the only ODE method used (`ode_method = "euler"`).

1. Reference prep, text fix-up, `max_chars` and `chunk_text`: plans 02 and 03.
   - `chunk_text` never splits inside a sentence. A long text with no `;:,.!?` or CJK sentence punctuation is one chunk, however long.
   - An empty reference text is transcribed with Whisper (`utils_infer.py:366`).
2. `fix_duration`, when set and there is more than one chunk, is split across chunks (`utils_infer.py:491`):
   - `ref_sec = (samples // 256) * 256 / 24000`.
   - Each chunk gets `ref_sec + allocatable * w / total_w`, where `w` is the chunk's UTF-8 byte length and `allocatable = fix_duration - ref_sec + cross_fade_duration * (chunks - 1)`.
   - With one chunk, the chunk gets `fix_duration` itself.
3. Per chunk (`_infer_basic`):
   - `local_speed` is `speed * 0.3` if the chunk is under 10 UTF-8 bytes, otherwise `speed`.
   - The text is `ref_text_spaced + chunk`, through `convert_char_to_pinyin` (plan 03).
   - `ref_audio_len = samples // 256`.
   - Duration: `int(fix_dur * 24000 / 256)` if there is a fixed duration. Otherwise `ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)`, with both text lengths in UTF-8 bytes and `ref_text_len` that of the spaced reference text.
4. `CFM.sample`, with `cond` the prepared reference wave:
   - `cond = mel_spec(wave)`, (1, n_ref, 100), with `n_ref = samples // 256 + 1`. `lens = n_ref`.
   - `duration = max(max(text_len, n_ref) + 1, duration)`, clamped to 65536. `text_len` counts the ids, not the -1 padding.
   - `step_cond` is the mel padded with zeros to `duration` frames. `cond_mask` is true for the first `n_ref` frames.
   - Noise: `torch.randn(duration, 100, device=self.device)` (`cfm.py:212`) from torch's global generator.
     - `infer_batch_process` passes no seed, so the generator continues across chunks. `api.py` seeds it once per generation (`seed_everything`, a random seed up to 2^63 - 1 if none is given).
     - The device matters. On the CPU the generator is mt19937, seeded from the low 32 bits of the seed (`manual_seed(2**32 + 5)` equals `manual_seed(5)`). On CUDA it is Philox, and the same seed gives entirely different noise.
   - Precision: `load_checkpoint` picks fp16 on any CUDA GPU of compute capability 7 or more (`utils_infer.py:199`), so upstream on a modern NVIDIA GPU runs in fp16 by default. The golden data is CPU fp32.
   - The time grid: EPSS (`get_epss_timesteps`) for `nfe_step` in {5, 6, 7, 10, 12, 16}, otherwise `linspace(0, 1, nfe_step + 1)`, all float32. Then, unless the coefficient is `None`, sway: `t + coef * (cos(pi / 2 * t) - 1 + t)`.
   - `odeint` with Euler: `x_{k+1} = x_k + (t_{k+1} - t_k) * v(t_k, x_k)`, with `v = pred + (pred - null_pred) * cfg_strength`.
     - For any `cfg_strength < 1e-5`, negatives included, upstream runs only the conditional pass (`cfm.py:179`). The batch-2 graph gives the same result only at exactly 0, so the port maps every value below 1e-5 to 0.
   - After the last step: `out = where(cond_mask, cond, x)`.
5. Back in `_infer_basic`:
   - Crop `out[:, ref_audio_len:]`. That keeps the reference's last mel frame as frame 0 (plan 01's off-by-one note).
   - Vocos decode, then `wave * rms / target_rms` if the reference was raised (plan 02).
6. Upstream runs the chunks in a `ThreadPoolExecutor`. With more than one chunk, the order in which they draw noise is not fixed, so a multi-chunk generation is not reproducible even in Python. `make_golden.py` runs them in order (its `SyncExecutor`).
7. Cross-fade (`utils_infer.py:575`), in numpy:
   - `n = min(int(cross_fade_duration * 24000), len(prev), len(next))`, truncated. If n <= 0, concatenate.
   - Otherwise `prev[:-n] + (prev[-n:] * linspace(1, 0, n) + next[:n] * linspace(0, 1, n)) + next[n:]`.
   - The fade is float64, so the running result is float64 from the first fade on, and `final.wav` is cast to float32 once at the end.

## What exists

- Plan 01: `DiT.create`, `embedText(ids, n)` and `step({ x, stepCond, text, time, cfgStrength, dt })`.
  - `step` returns `pred` (2, n, 100), `v` and `x_next` (1, n, 100), all on the GPU. The caller disposes all three.
  - n runs from 16 to 8192 frames, the rotary table's size. Only n up to 2048 has been run; plan 01 measured GPU memory once, 1.7 GB at n = 1913.
  - Per step at n = 1024 and 2048: about 0.4 and 0.96 s on the RTX 5050, and 1.16 and 2.78 s on the Intel GPU. The first run of a session costs 1.4 to 3.3 s more.
  - `session.run` resolves before the GPU has finished. `Vocoder.decode` and the bench wait on `device.queue.onSubmittedWorkDone()` to time anything.
- Plan 02: `trimAndClip`, `downmixRmsResample`, `logMel` ((100, frames), mel-major), `Vocoder.decode(mel (1, 100, T))` and `rescaleOutput`. The vocoder, readback and istft take under 100 ms per chunk.
- Plan 03: `fixRefText`, `maxChars`, `chunkText`, `addTrailingSpace`, `utf8Length`, `TextFrontend.tokens`, `parseVocab` and `textToIds`.
- Golden data per chunk: `plan.t`, `noise.y0`, `dit.step_cond`, every step's `dit.sNN.x` and `t`, `dit.text_ids`, `mel_out`, `vocoder.wav`. Per case: `final.wav`, and in the manifest `chunks[i].duration`, `ref_audio_len` and `local_speed`.
  - Verified in the pressure test: every chunk's `noise.y0` in every case equals `manual_seed(0)` followed by successive `randn(duration, 100)` calls, bit for bit.
  - The golden data does not reach every rule. The minimum-duration rule never fires (the smallest duration is 515, against a floor of about 467), the 65536 clamp is never reached, and no chunk wave is shorter than the 3600-sample fade.
- Checks in `tolerances.json`:
  - `plan.duration`, `plan.ref_audio_len` and `plan.local_speed` must be exact.
  - `plan.t`: max abs 1e-6.
  - `mel_out.free_running`: rel L2 1e-3. That is the CPU against CUDA floor of 2.8e-5, times 10, rounded up to a power of ten.
  - `noise.y0`, `dit.step_cond` and `final.wav` have no check: both runners treat them as inputs (`checksFor` returns nothing). `measure_floor.py`'s free-running run stops at the cropped mel, so there is no floor for `final.wav` either.

## Decisions

### D1. Noise (proposal: port torch's CPU `randn`, and prove it against the golden noise)

- The golden noise is torch's CPU stream from seed 0, chunk after chunk (verified, see above). So a port of torch's CPU generator can be checked against every chunk of every golden case.
- What torch does, from the pressure test's numpy model of it: mt19937 with its standard seeding from the low 32 bits; each float32 uniform from 24 bits of one 32-bit output; for tensors of 16 or more, all uniforms first, then Box-Muller 16 at a time, with the last 16 redrawn and recomputed when the size is not a multiple of 16. Most durations are not (1913 is not), so the redraw is always exercised.
- How close a port can get: the model reproduced the golden noise to max abs 4.8e-7 (at most 3 ulp, 86% bit-exact) with float32 maths, and 1.4e-6 with float64 maths rounded to float32. Bit-exactness is not the goal.
  - torch's own scalar path (`ATEN_CPU_CAPABILITY=default`) is 2e-6 away from the golden data, which was made on an AVX2 machine. So the golden noise depends on the CPU that made it. `make_golden.py` should record `torch.backends.cpu.get_cpu_capability()` in the manifest.
- Seeds are 32-bit unsigned integers in the port, since that is all the CPU generator uses. A random seed is drawn from [0, 2^32). That also avoids JavaScript's 2^53 limit, which `api.py`'s seeds up to 2^63 - 1 would break.
- If it matches, the same seed gives the same noise as Python on the CPU. Then the port and PyTorch can be compared utterance by utterance (D5).
- Fallback, if the port cannot meet step 2's bound in about a day: any seeded normal generator. Nothing numeric depends on it, because every parity test injects the golden noise. D5 then loses its pairing (see there).

### D2. Where the loop's state lives (proposal: on the GPU, read back once per chunk)

- Per chunk, upload the noise and `step_cond` once, and run the text graph once.
- Each step feeds the previous step's `x_next` GPU tensor as `x`, then disposes the old `x` and the step's `pred` and `v`.
  - Plan 01's bench chains steps on the GPU by copying `x_next` into a fixed `x` buffer instead. Step 4 checks that feeding the output tensor straight back works in ORT Web, and falls back to the copy if not.
- After the last step, read `x` back. Restore the reference frames, crop, and transpose to the vocoder's (1, 100, T) on the CPU. That is 100 floats per frame, under 1 MB per chunk.
- Each step waits for `device.queue.onSubmittedWorkDone()` before reporting progress. That keeps progress honest and lets abort stop cleanly; at 0.4 s or more per step, the wait costs nothing measurable.

### D3. Chunk order (proposal: one chunk at a time, in order, with a per-chunk callback)

- There is one GPU, so running chunks in parallel gains nothing.
- In order, the noise stream matches the golden data's order, and generation is reproducible, unlike upstream's thread pool.
- A callback per finished chunk, with its wave, lets plan 06 start playback before the last chunk is done.

### D4. New checks (proposal: measure, then set as plan 00 did)

- `noise.torch_randn`: the port's noise against `c*.noise.y0`, max abs 1e-5. That is about 20 times the pressure test's float32 model, and leaves room for float64 maths.
- `final.crossfade`: the cross-fade from the golden chunk waves against the golden `final.wav`. It is float64 arithmetic in both, so it should be exact or nearly so. Measure, then set.
- `final.free_running`: end to end from `input.wav` and the texts, with the golden noise injected, against `final.wav`.
  - Its floor needs new code. `measure_floor.py`'s CUDA free-running run stops at the cropped mel. It must also run the vocoder, the rescale and the cross-fade, then compare with `final.wav`.
  - Write the new floors into `tolerances.json` with `--from-floors` or a merge, so that existing tolerances do not shift.
- The same golden `final.wav` is compared in two modes, so its checks are chosen by mode, not by tensor name. The runners take an explicit check name for it.
- `mel_out.free_running` keeps its rel L2 of 1e-3. WebGPU float32 may drift further than CUDA over 32 steps. If it does, the numbers go to the user before any tolerance is loosened, as plan 00 D2 says the end-to-end judge is quality.
- Both runners get the new checks: `tools/webgpu/parity.py` and `web/test/parity/compare.ts`, with their self-tests.

### D5. Quality against PyTorch (thresholds and scope accepted by the user, 2026-11-22)

- Plan 00 D2 left a metric-based end-to-end check to this plan:
  - Take 100 English and 100 Chinese utterances from the Seed-TTS test set (prompt wav, prompt text, target text), picked with a fixed seed.
  - Generate each with PyTorch and with the port (Chrome, RTX 5050), with the same seed per utterance.
  - Score both with upstream's `eval/`: WER (faster-whisper large-v3 for English, paraformer-zh for Chinese) and speaker similarity (WavLM large). UTMOS optionally.
  - Pass if the port's mean WER is within 0.5 points of PyTorch's and its mean similarity within 0.01. Also look at, and listen to, the 5 utterances with the largest difference.
- Accepted 2026-11-22: 100 utterances per language, WER within 0.5 points, similarity within 0.01.
- The PyTorch side must not use `F5TTS` as it is:
  - `F5TTS` loads fp16 on the RTX 5050. Build the model as `make_golden.load_models` does, with `dtype=torch.float32` on CUDA.
  - Draw the noise on the CPU and move it to the GPU, by wrapping `odeint` as `measure_floor.py` does (around line 244). CUDA's own `randn` would give unrelated noise for the same seed.
  - Run the chunks in order, as the port does.
- Pairing is what makes the thresholds meaningful. With the same noise, the two outputs differ only by GPU arithmetic, and a 0.5-point WER difference is a real signal. Without pairing (D1's fallback), 100 utterances against 0.5 points is dominated by sampling noise: one word in a 20-word sentence is 5 points. In that case report the numbers, and ask the user whether to enlarge the set or loosen the bound.
- Still needs the user: the WavLM checkpoint (`wavlm_large_finetune.pth`), which `eval_seedtts_testset.py` expects in `../checkpoints/UniSpeech/` and which is a manual download. Whisper and paraformer download themselves, a few GB together.
- Cost: about 15 s per utterance on the RTX 5050, so roughly an hour per language for the port, plus the scoring.

### D6. API (proposal)

- `TTS.create({ dit, vocoder, vocab, text? })` loads the sessions and the vocab. `text` takes `TextFrontend` loaders (plan 03).
- `generate({ refAudio: Pcm, refText, genText, settings }, { onProgress, onChunk, signal })` returns `{ wave: Float32Array, sampleRate: 24000, seed, chunks }`.
  - `settings` has upstream's names and defaults: `nfeStep`, `cfgStrength`, `swaySamplingCoef` (a number, or `null` for no sway), `speed`, `crossFadeDuration`, `fixDuration`, `targetRms`, `seed`.
  - `seed` is a 32-bit unsigned integer (D1). A missing seed is drawn at random and returned, as `api.py` does.
  - `cfgStrength` below 1e-5 is treated as 0, as upstream's cond-only path behaves.
  - An empty or all-whitespace `refText` is refused with a clear error, since upstream would transcribe it (plan 06).
  - `onProgress({ chunk, chunks, step, steps })` after each step has finished on the GPU (D2). `onChunk({ index, wave })` after each chunk's vocoder.
  - `signal` is an `AbortSignal`, checked between steps. On abort, the run waits for the queued GPU work, then disposes every GPU tensor it made.
- `release()`.
- The module touches no DOM, so plan 06 can run it in a worker.

### D7. Long chunks (proposal: refuse any chunk past 8192 frames, by computed duration)

- The step graph's rotary table covers 8192 frames, about 87 s. Upstream clamps at 65536.
- Two ways past it: a large `fix_duration`, or a long text with no sentence punctuation, which `chunk_text` leaves as one chunk. So the check is on each chunk's computed duration, whatever produced it. The error names the chunk and suggests punctuation or a shorter text.
- n between 2048 and 8192 has never been run. Step 4 runs n = 4096 and 8192 once on each GPU, to see whether they fit and how long they take (R3). If 8192 does not fit on the Intel GPU, the refusal limit becomes what does fit, per GPU, and plan 06 shows it.
- If longer chunks matter, plan 05 can export a larger table.

## Steps

### Step 1. The chunk plan

- `web/src/pipeline/plan.ts`:
  - `planChunks({ refSamples, refTextSpaced, chunks, chunkIdCounts, speed, fixDuration, crossFadeDuration })`: per chunk, `local_speed`, the duration (the `fix_duration` split, the duration formula, and the minimum-duration rule, which needs each chunk's id count from `TextFrontend`), `ref_audio_len` and `n_ref`.
  - `timeGrid(nfeStep, swayCoef)`: EPSS or linspace, then sway unless the coefficient is null, emulating torch's float32 steps where they matter. Plan 02 found that torch's `linspace` fills from both ends.
- Tests (Node, Vitest):
  - On every golden case: `plan.duration`, `plan.ref_audio_len` and `plan.local_speed` exact, and `plan.t` within its tolerance. en-multi-fixdur covers the split, and en-epss16 covers EPSS.
  - Synthetic cases for what the golden data never reaches, with values computed by upstream's Python in the test's fixture script:
    - the minimum-duration rule, using a long text with a short duration;
    - a chunk under 10 bytes;
    - no sway;
    - an `nfe_step` outside the EPSS table;
    - a chunk past 8192 frames, which must be refused.
- Done when: all pass.
- Result, 2026-11-22: all pass.
  - `web/src/pipeline/plan.ts`: `planChunks` and `timeGrid`, and `MAX_FRAMES` (8192). `planChunks` takes `tokenCounts` (each joined text's token count) rather than ids. `linspaceF32` is now exported from `audio/mel.ts`.
  - Golden cases (`web/test/pipeline/plan.test.ts`): all 7 give the exact `duration`, `ref_audio_len` and `local_speed` of every chunk, and pass `plan.t` through the parity runner.
  - The time grid is exact without sway. With sway, one point of each grid is one float32 ulp off (6e-8), always the same point, which is the CPU against CUDA floor and well inside 1e-6. torch's vectorized `cos` rounds that one value differently from `Math.cos` after `fround`.
  - Synthetic cases: `tools/webgpu/plan_fixtures.py` runs upstream's real `infer_batch_process` and `CFM.sample`, with a stub transformer and vocoder, and records each chunk's duration as asked, its token count, its final duration and its time grid. That gives 13 cases in `testdata/webgpu/pipeline/plan_cases.json`, which is committed (a `.gitignore` exception):
    - the minimum-duration rule, which fires (185 frames asked, 407 tokens, 408 frames);
    - chunks of 8 and exactly 10 bytes, and a 4-character Chinese chunk of 12 bytes;
    - speed 0.7;
    - `fix_duration` with one chunk, and with three chunks and a 0.25 s cross-fade;
    - EPSS at 7 and 16 steps, linspace at 20, no sway, positive sway;
    - Chinese, and an odd sample count.
    - Every case's token counts (from `convertCharToPinyin`) and durations match. Every time grid is within 6e-8, one float32 ulp.
  - D7: a chunk is refused past 8192 frames, whether from `fix_duration` (90 s gives 8437 frames) or from an unpunctuated long text. The error names the chunk and the fix.
  - Mutation checks: dropping the minimum-duration rule, `<= 10` for the short-chunk rule, and counting UTF-16 units instead of bytes each fail tests. The last two needed the 10-byte and 12-byte cases, which were added for that. Two mutants survive, both equivalent on any realistic input: the other order of operations in the `fix_duration` split, and float64 `pi / 2`.

### Step 2. Noise (D1)

- `web/src/pipeline/noise.ts`: a port of torch's CPU `manual_seed` and `randn`.
  - First read torch 2.15's source for the generator, the uniform conversion and the normal fill, both scalar and AVX2, and write it down in `docs/research/`.
  - `make_golden.py` records the CPU capability in the manifest. Regenerating is not needed for the existing cases; record what this machine reports, since it made them.
- Tests: with seed 0, every chunk's draw against `c*.noise.y0`, chunk after chunk, in every golden case, through the new `noise.torch_randn` check. Also seeds 1, 2^31 and 2^32 - 1 against values from Python, for the seeding.
- Done when: every chunk is within max abs 1e-5, or the fallback is taken after about a day, with the reason recorded.

### Step 3. Conditioning and output (CPU)

- `web/src/pipeline/cond.ts`:
  - `stepCond(mel, duration)`: the (100, n_ref) mel from `logMel`, transposed to (1, duration, 100) and padded with zeros.
  - `finishChunk(x, mel, refAudioLen)`: the reference restore for the first `n_ref` frames, the crop from `ref_audio_len`, and the transpose to (1, 100, T) for the vocoder.
- Tests (Node):
  - `stepCond` from the golden `mel.ref` against `c*.dit.step_cond`: exact.
  - `finishChunk`: build a final `x` whose first `n_ref` frames are arbitrary and whose remaining frames are the golden `mel_out` (from frame `n_ref - ref_audio_len` on). `finishChunk` must give back the golden `mel_out` exactly, with the reference's last frame as frame 0.
- Done when: exact.

### Step 4. The ODE loop on WebGPU (D2)

- `web/src/pipeline/sample.ts`: per chunk, the text graph once, then `nfeStep` steps with `x` on the GPU, then one readback.
- A small tensor ledger: every GPU tensor the pipeline creates or receives goes through one wrapper that counts live tensors. WebGPU has no memory query, so this count is the leak test. Chrome's task manager (GPU memory column), read by hand before and after a multi-chunk run, is the cross-check.
- Browser test, local only (needs `build/webgpu/dit/base/` and the golden cases): for each chunk of each golden case, inject the golden `noise.y0` and `dit.step_cond`, and compare the result with `mel_out` (`mel_out.free_running`). Run on both GPUs in Chrome.
  - Also record the error after each step against the golden `x`, so drift can be seen growing.
  - Also run once with this plan's own noise (step 2) and step_cond (step 3), so the whole chain from the reference wave is covered.
  - The live-tensor count is the same after every chunk, and back to its start at the end.
- Once per GPU, run single steps at n = 4096 and 8192 with random inputs: does it fit, how long does it take, and what does the task manager show (D7).
- Record time per chunk, and the first-run cost.
- Done when: every chunk passes on both GPUs, or the numbers are with the user (D4); the tensor count is flat; and D7's limit is set.

### Step 5. Vocoder, rescale and cross-fade

- `web/src/pipeline/assemble.ts`: `crossFade(waves, seconds)`, with `n = int(seconds * 24000)` truncated, in float64 as numpy does, cast to float32 once at the end.
- Node tests:
  - The golden `c*.vocoder.wav` through `crossFade` against `final.wav` (`final.crossfade`).
  - Synthetic: a wave shorter than the fade, a fade of 0, and a single chunk, against numpy's result.
- Browser test, local only: end to end, from `input.wav` and the texts, with the golden noise injected (`final.free_running`). Write each result as a wav for listening.
- `tools/webgpu/measure_floor.py` gains the vocoder, rescale and cross-fade in its free-running run. Then `tolerances.json` (merged, D4) and both runners get `noise.torch_randn`, `final.crossfade` and `final.free_running`.
- Done when: both final checks pass on both GPUs, and the en-multi and zh-short results have been listened to.

### Step 6. The API (D6)

- `web/src/pipeline/index.ts`: `TTS`, `generate`, progress, per-chunk callback, abort, release.
- Tests:
  - The defaults equal upstream's.
  - cfg below 1e-5, negatives included, gives the same result as 0. An empty reference text is refused.
  - Abort mid-chunk stops after the current step, and the live-tensor count returns to its start.
  - In CI, on the tiny model (SwiftShader): with seed 0 and the tiny case's settings, `generate` reproduces the tiny case's `final.wav` within `final.free_running` (or the tiny variant that plan 00 used for other checks). That is an end-to-end regression check in CI, not just a smoke test. It holds only if D1 holds; otherwise inject the golden noise.
  - Locally, the same for every real golden case on the RTX 5050.
- Done when: all pass.

### Step 7. Quality against PyTorch (D5)

- `tools/webgpu/quality_set.py`:
  - Extracts the Seed-TTS meta lists and the needed prompt wavs from the local tar into `data/seedtts_testset/{en,zh}/`, the layout `eval_seedtts_testset.py` expects. `data/` is already ignored by git.
  - Picks the subset with a fixed seed, and writes a manifest of prompt wav, texts and seed per utterance.
- `tools/webgpu/quality_torch.py`: generates the subset with upstream's pipeline on CUDA in fp32, with CPU-drawn noise per D5, one seed per utterance, chunks in order.
- `web/bench/quality/`: generates the same subset with the port in Chrome on the RTX 5050, and writes the wavs.
- Score both with upstream's `eval/` scripts, one job at a time. Compare WER, similarity and optionally UTMOS, per language.
- Done when: the comparison is recorded in `docs/research/` and judged against D5's thresholds.

### Step 8. Close out

- Hand-off to plan 06: the API, the per-chunk callback for streaming playback, the worker, abort, the empty-reference refusal, and the long-chunk limit per GPU.
- Hand-off to plan 07: time per chunk on both GPUs, the first-run cost, the istft overlap, cfg 0, and the large-n timings.
- `docs/debugging.md` entries for anything surprising.
- Update the task list.

## Risks

- **R1. WebGPU drift over 32 free-running steps.** Per-step error on WebGPU is larger than CPU against CUDA, and Euler integrates it. Step 4 records it per step. If `mel_out.free_running` fails while the audio sounds right, D4 sends it to the user.
- **R2. torch's `randn` is harder to reproduce than it looks.** The pressure test's model came within 4.8e-7, so the structure is known. What remains is detail: the tail redraw, the seeding, and the golden data's dependence on the CPU capability of the machine that made it. Step 2 is bounded, and the fallback costs only D5's pairing.
- **R3. GPU memory.** The DiT (1.33 GB fp32), the vocoder (54 MB) and the activations must fit together, on the Intel GPU too, which shares system memory. Plan 01 measured 1.7 GB only at n = 1913; nothing past 2048 has run. Undisposed ORT tensors (`pred` and `v` every step, as well as `x`) would grow it. Step 4 counts live tensors and runs n = 4096 and 8192.
- **R4. Test time.** A three-chunk case is about 100 steps: about a minute on the RTX 5050, and three to four on the Intel GPU. The local browser tests need long timeouts and one worker, as plan 01's do.
- **R5. Quality-check dependencies.** About 4 GB of ASR models, a manual WavLM download, and Python eval dependencies that do not install on Python 3.12 and newer (`av==10.*`; see CLAUDE.md). The check may need its own environment.
- **R6. Upstream defaults differ from what we compare against.** Upstream on a modern NVIDIA GPU runs fp16 with CUDA noise. The port is compared with CPU fp32 (golden data) and CUDA fp32 with CPU noise (D5). That is deliberate: it isolates the port's own error. But it means "matches PyTorch" here does not mean "matches what a user of upstream's default CLI hears" sample for sample.

## Pressure test, 2026-11-22

A fresh agent read only this file and checked it against the code. It reproduced the golden noise with a numpy model of torch's generator, and compared CUDA with CPU `randn`. Findings, and what changed:

1. **D5's pairing could not work as written** (wrong). `F5TTS` loads fp16 on the RTX 5050, and `cfm.py:212` draws the noise on the model's device, where CUDA's Philox gives unrelated noise for the same seed. D5 now builds the model in fp32 and draws the noise on the CPU, and says why pairing matters for the thresholds.
2. **D7 missed long unpunctuated text** (wrong). `chunk_text` never splits inside a sentence, so any text can make an oversized chunk. D7 now checks every chunk's computed duration, and step 4 runs n = 4096 and 8192, which had never been run.
3. **Negative `cfg_strength`** (wrong). Upstream's cond-only path covers every value below 1e-5. The port maps those to 0.
4. **The harness had no checks for the new comparisons** (gap). `noise.y0`, `dit.step_cond` and `final.wav` are inputs to both runners, and `measure_floor.py` has no `final.wav` floor. D4 now adds `noise.torch_randn`, chooses `final.wav`'s check by mode, extends `measure_floor.py`, and merges new floors without shifting old ones.
5. **D1's exactness** (risk). The golden noise was verified as torch's CPU stream from seed 0. A float32 model reaches 4.8e-7, not bit-exact, and the golden data depends on the CPU capability (AVX2). Step 2's bound is now a number, the manifest records the capability, and seeds are 32-bit, which is all torch's CPU generator uses.
6. **Rules the golden data never reaches** (gap): the minimum-duration rule, the 65536 clamp, and the short-wave cross-fade. Steps 1 and 5 add synthetic tests. `planChunks` takes the id counts it needs.
7. **"GPU memory stays flat" could not be checked** (gap). WebGPU has no memory query. Step 4 adds a live-tensor ledger, with Chrome's task manager as the cross-check. D2 disposes `pred` and `v` as well as `x`.
8. **Progress and abort timing** (risk). `session.run` resolves before the GPU finishes. D2 waits for `onSubmittedWorkDone` each step.
9. **Upstream behaviour not covered** (gap): an empty reference text (Whisper), `target_rms`, no sway, and a cross-fade length taken from the setting. All four are now in D6 or the steps.
10. **The Seed-TTS data was not in the eval's layout** (gap). Step 7 extracts it. The tiny model in CI now reproduces its golden `final.wav` end to end (step 6).
11. **D4's wording** (minor). The free-running tolerance is the floor times 10 rounded up to a power of ten, about 36 times the floor.

Checked and found right: the `fix_duration` split (recomputed for en-multi-fixdur's 1820, 1903 and 515), the short-chunk rule, the duration formula, n_ref, the minimum-duration rule and clamp, the EPSS table, float32 linspace, sway, the Euler step, the final restore, the crop, the rescale, the float64 cross-fade, the defaults, and the TypeScript APIs as described.
