# Quality of the WebGPU port against PyTorch

Plan 04 step 7 (D5). Run 2026-11-22 and 2026-11-23 on the laptop: RTX 5050, Chrome, Windows 11.

## What was compared

- **The set:** 100 English and 100 Chinese utterances from the Seed-TTS test set (`meta.lst`). They were picked with a fixed seed by `tools/webgpu/quality_set.py`, which also gives each utterance a uint32 generation seed. The data lives in `data/seedtts_testset/`, which git ignores.
- **PyTorch** (`tools/webgpu/quality_torch.py`): upstream's own `preprocess_ref_audio_text` and `infer_process`, on CUDA in fp32 with TF32 off, using upstream's default settings. Two changes pair it with the port:
  - chunks run in order;
  - the starting noise is drawn on the CPU from `torch.manual_seed(seed)` (D1).
- **The port** (`web/bench/quality/run.mjs`): `TTS.generate` in Chrome on the RTX 5050, with the same seed per utterance.
- **Scoring** (`tools/webgpu/quality_score.py`): upstream's own `get_seed_tts_test`, `run_asr_wer` and `run_sim`, on one GPU.
  - WER uses faster-whisper large-v3 (English) and paraformer-zh (Chinese).
  - Similarity uses WavLM-Large with an ECAPA-TDNN head (`wavlm_large_finetune.pth`, SHA-256 `51f07e3b…f94b`).

## Results

| | PyTorch | Port | Port minus PyTorch | D5 bound | |
|---|---|---|---|---|---|
| English WER | 3.289% | 3.289% | 0 | 0.5 points | pass |
| Chinese WER | 1.430% | 1.430% | 0 | 0.5 points | pass |
| English SIM | 0.68915 | 0.68914 | -0.00001 | 0.01 | pass |
| Chinese SIM | 0.75352 | 0.75352 | +0.00001 | 0.01 | pass |

- **WER is identical for every utterance**, in both languages: the ASR gives the same transcript for each pair.
- **SIM differs only in the fourth decimal.** The largest single difference is 8e-4.
- **The pairing held throughout.** Every port wave has exactly the length of its PyTorch partner. Their relative L2 difference has a median of 5.1e-4 (English) and 2.7e-4 (Chinese), a 90th percentile of 3.1e-3 and 1.1e-3, and a maximum of 0.062 and 0.0099.
- **What it means:** on this set, the port is as good as upstream's fp32 PyTorch, as far as these metrics can tell.
  - It says nothing about upstream's default fp16 on CUDA, which also draws different noise (plan 04, R6).
  - The absolute numbers are not comparable with published Seed-TTS results. This used `infer_process`, not the batch eval script, a newer faster-whisper, and a 100-utterance subset. Only the difference between the two sides matters here.

## Time

| | PyTorch, CUDA fp32 | Port, Chrome on the RTX 5050 |
|---|---|---|
| English, 387 s of audio | 800 s (median 7.1 s per utterance) | 1046 s (median 10.5 s) |
| Chinese, 547 s of audio | 847 s (median 7.5 s) | 1454 s (median 13.3 s) |

- PyTorch's time includes its reference preprocessing; the port's includes fetching the prompt wav.
- On this GPU, the port runs 1.3 to 1.7 times slower than PyTorch in fp32.

## Versions and setup

- **Scoring packages:** faster-whisper 1.2.1, PyAV 16.1.0, funasr 1.4.28 (paraformer-zh from ModelScope), jiwer 4.0.0, and s3prl from its main branch (commit `ec8064b`, dated 2025-06-13, fetched 2026-11-23).
- **Install:** the packages went into `.venv313`, with torch, torchaudio and numpy pinned so they stayed as they were. faster-whisper's GPU backend also needs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`, since torch ships CUDA 13.
- **Scoring needs:** `FFMPEG_DLL_DIRS` (see `tools/webgpu/dll_dirs.py`), `PYTHONUTF8=1`, and `s3prl_s3prl` in torch.hub's trusted list.
- **Problems on the way:** recorded in `docs/debugging.md`, 2026-11-23.
- **To rerun**, from the repo root:
  1. `quality_set.py <tar>`
  2. `quality_torch.py --lang en|zh`
  3. `node web/bench/quality/run.mjs --lang en|zh --gpu nvidia`
  4. `quality_score.py --lang en|zh --task wer|sim`
  5. `quality_score.py --lang en|zh --report`
