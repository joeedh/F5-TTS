# Audio ends of the WebGPU port: reference prep, mel, vocoder (plan 02)

Measured 2026-11-21 on the laptop ([webgpu-laptop.md](webgpu-laptop.md)). Plan: [../plans/webgpu-02-mel-and-vocoder.md](../plans/webgpu-02-mel-and-vocoder.md).

**Outcome:**

- Reference preparation is bit-exact with upstream for 16-bit PCM wav at any rate.
- The mel, resampler and inverse STFT match the golden data at or near the CPU-against-CUDA floor.
- The vocoder graph runs entirely on WebGPU.
- D2: the inverse STFT stays in TypeScript on the CPU.

## Pipeline as implemented

All in `web/src/audio/` unless named otherwise.

1. `parseWav` (`wav.ts`): 16-bit PCM wav to interleaved int16 `Pcm`, at the file's own rate.
2. `trimAndClip` (`prepare.ts`, on `pydub.ts`): upstream's `preprocess_ref_audio_text` audio steps:
   - Split on 1 s silences at -50 dBFS, keeping 1 s around each part. Parts are joined until the result is over 6 s and the next part would take it past 12 s.
   - If still over 12 s, the same with 100 ms silences at -40 dBFS.
   - If still over 12 s, a cut at 12 s.
   - Trim leading and trailing audio below -42 dBFS, then append pydub's "50 ms" of silence.
   - The output keeps the input's channels, and its rate unless that is below 11025 Hz.
3. `downmixRmsResample` (`prepare.ts`, `resample.ts`): mean of channels; raise to RMS 0.1 if quieter; torchaudio's sinc resampler to 24 kHz. Returns the audio with `rms` and `raised`.
4. `logMel` (`mel.ts`, `fft.ts`): the Vocos mel, (100, frames) with frames = samples / 256 + 1.
5. `Vocoder` (`web/src/vocoder/`): `vocos.onnx` on WebGPU, mel (1, 100, T) to head_out (1, T, 1026), read back.
6. `istft` (`istft.ts`): exp and clip at 100, cos/sin, inverse real FFT, periodic Hann, overlap-add at hop 256, divide by the summed squared window, trim 512 at both ends. Output (T - 1) * 256 samples.
7. `rescaleOutput` (`output.ts`): `wave * rms / 0.1` in float32 when `raised`.

## Mel and filterbank

- torchaudio `MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100, power=1, center=True, normalized=False, norm=None)`:
  - Reflect padding of 512 on both sides.
  - Periodic Hann window.
  - Magnitude, not power.
  - HTK mel scale from 0 to 12 kHz, filterbank not normalised.
  - `log(max(mel, 1e-5))`.
- The FFT runs in float64. The filterbank does not: torchaudio builds it in float32, and the port reproduces that rounding step by step (`Math.fround`, and `torch.linspace`'s fill from both ends).
  - A float64 filterbank is up to 9.0e-6 away from torchaudio's. It put the mel 10 times further from the golden one (rel L2 1.9e-6 against 1.9e-7).
  - The emulation is within 3.8e-6 of torchaudio's matrix. The rest is torch's vectorised float32 `pow`, which rounds 5 of the 102 centre frequencies differently.
  - Result: `mel.linear` rel L2 4.2e-7 at worst (tolerance 1e-5), and `mel.log_masked` max abs 1.6e-3 (tolerance 0.1).
- Reference: `testdata/webgpu/audio-fixtures/mel_fbank.npy`, torchaudio's own matrix, checked in `web/test/audio/mel.test.ts`.

## pydub exactness contract

Exact for 16-bit PCM wav, any channel count and any rate. Each rule below has a fixture in `testdata/webgpu/audio-fixtures/prep/`, from `tools/webgpu/export_audio_fixtures.py`.

- `len(segment)` is milliseconds rounded half to even, as Python's `round` does. JS's `Math.round` would differ at frame counts of 12 mod 24 at 24 kHz.
- Milliseconds become frames as `int(ms * (rate / 1000.0))`, a truncated double product.
- Slices are capped at `len(segment)`. A slice that the rounded length makes overshoot the data is padded with zero frames, at most 2 ms of them. An empty slice gets no padding.
- RMS is `audioop.rms`: over the interleaved samples of all channels, truncated to an integer.
- `detect_silence` compares that integer linearly with `10 ** (dB / 20) * 32768`. `detect_leading_silence` compares `20 * log(rms / 32768, 10)` in dB, with an rms of 0 giving minus infinity.
- Joining converts both parts to the larger channel count and rate. The appended silence is made at 11025 Hz and converted with `audioop.ratecv`: 1198 samples at 24 kHz, 2201 at 44.1 kHz.
- Inputs below 11025 Hz are upsampled to 11025 Hz by `ratecv` when joined. The port has `ratecv` in full, so these stay exact.
- A wholly silent input of 1 s or more leaves nothing after the split; the output is pydub's 551 frames of silence at 11025 Hz.
- `detect_silence` uses a prefix sum of squared samples. It is exact while the total stays below 2^53, and falls back to per-window sums past that.

What is not exact:

- 24-bit, 32-bit and float wav, and compressed formats. pydub keeps 24-bit as 32-bit and sends float wav through ffmpeg. The port takes int16, so plan 06 must convert these, and the output may differ.
- A silent reference: upstream divides by an rms of 0 and its reference is all NaN. The port reproduces that.

## Resampler

- torchaudio's `Resample` defaults: `sinc_interp_hann`, `lowpass_filter_width` 6, `rolloff` 0.99.
- The kernel follows `_get_sinc_resample_kernel`'s dtypes: the phase offsets are float32, the rest float64, and the taps are rounded to float32. The convolution runs in float64.
- 44.1 to 24 kHz: worst error 1.8e-7 on en-prep, equal to the CPU-against-CUDA floor (tolerance 1e-5).

## Vocoder graph

- `tools/webgpu/export_vocos.py`: backbone plus `head.out`, dynamic T from 8 to 16384, opset 20. One file of 54.3 MB, no external data.
- 96 nodes as exported, 69 after ORT's optimisations. All 69 run on the WebGPU EP on both GPUs, including the 8 exact `Gelu` (opset 20, `approximate="none"`).
- Parity on WebGPU, every chunk of the 6 real cases: worst `vocoder.stft_complex` at 5.0 percent of its tolerance and `vocoder.wav` at 4.3 percent (NVIDIA; Intel 4.6 and 3.9).
- On onnxruntime CPU: `vocoder.backbone` rel L2 1.3e-5 and `vocoder.stft_complex` 2.6e-4 at worst, both below their CPU-against-CUDA floors.

## Inverse STFT and D2

- `istft` matches the golden `vocoder.wav` from the golden head_out at rel L2 1.2e-7, the float32 rounding of the output. A new check, `vocoder.istft` (rel L2 1e-5), guards it. The older `vocoder.wav` tolerance (1e-2) would pass a 0.2 percent ripple from a wrong window.
- Time per chunk in Chrome, median of 5 (ms):

  | GPU | frames | graph | readback | istft |
  |---|---|---|---|---|
  | NVIDIA | 1448 | 15 | 6 | 70 |
  | NVIDIA | 470 | 7.5 | 4.7 | 21 |
  | Intel | 1448 | 47 | 5 | 67 to 73 |
  | Intel | 470 | 18.5 | 4.4 | 20 |

- D2 option A (TypeScript istft) is kept. Readback plus istft is about 75 ms per long chunk on either GPU. That is 0.1 to 0.25 percent of the chunk's DiT time (about 30 s on NVIDIA, 85 s on Intel), far under the 5 percent that would call for an in-graph istft.
- The istft is 4 to 5 times the graph's time on NVIDIA. Running it in a worker during the next chunk's DiT steps, or a packed real FFT, would remove it from the critical path or halve it. That is plan 07's call.

## Other timings (Node)

- `trimAndClip`: 7 ms for en-prep's 16 s 44.1 kHz stereo input.
- `downmixRmsResample`: 28 ms for its 7 s result.
- `logMel`: about 5 ms per second of audio, about 70 ms for a 12 s reference.

## Caveats

- One laptop, Chrome only for the vocoder; Edge was not timed.
- Timings are single sessions; power and thermal state were not recorded.
- The tiny case covers none of the resample, downmix or RMS raise; the fixtures do, in CI.
