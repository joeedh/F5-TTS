# Debugging guide

Running log. Add an entry for each bug or useful technique: symptom, cause, fix or diagnosis steps.

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
- Tested: with `soundfile.info` forced to fail and no `ffprobe` on the machine, the old code raised `FileNotFoundError` and the new code reaches the `torchaudio.load` fallback. A stubbed `torchaudio.load` (24000 frames at 24 kHz) gives 1.0. Not tested end to end: `torchaudio.load` needs `torchcodec` with FFmpeg, which is unusable on this Windows/Python 3.15 machine.

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
