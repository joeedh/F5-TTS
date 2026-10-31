# Debugging guide

Running log. Add an entry for each bug or useful technique: symptom, cause, fix or diagnosis steps.

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
