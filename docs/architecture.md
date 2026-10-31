# Architecture

Python package `f5_tts` under `src/f5_tts/`. Installed with console scripts `f5-tts_infer-cli`, `f5-tts_infer-gradio`, `f5-tts_finetune-cli`, `f5-tts_finetune-gradio` (see `pyproject.toml`).

## What the model does

Text-to-speech by voice cloning. Input: a reference clip, its transcript, and the text to speak. Output: a waveform in the reference voice.

1. Reference audio becomes a mel spectrogram (100 channels, 24 kHz, hop 256).
2. Text is tokenized to character/pinyin ids.
3. A transformer predicts a flow-matching velocity field over mel frames. The reference mel is the prompt; the span after it is masked and generated.
4. An ODE solver (`torchdiffeq`, default Euler) integrates that field from noise to mel.
5. A vocoder (Vocos or BigVGAN) turns the generated mel into audio.

No separate duration model. Total length is estimated from the ratio of reference text length to reference audio length, or forced with `fix_duration`.

## Layers

```
 entry points         api.py  infer/*  train/*  socket_server.py  runtime/triton_trtllm
      |
 inference glue       infer/utils_infer.py   (chunking, ref prep, vocoder, batch loop)
      |
 model                model/cfm.py  ->  model/backbones/{dit,mmdit,unett}.py  ->  model/modules.py
      |
 support              model/utils.py  model/dataset.py  model/trainer.py  device_utils.py
      |
 config               configs/*.yaml  (Hydra/OmegaConf)
```

## Model

- `model/cfm.py` `CFM`: wraps a backbone. `forward` is the training loss: a random span covering 70-100% of each sequence (`frac_lengths_mask`) is blanked in the conditioning mel, the model predicts `x1 - x0` along a linear noise-to-mel path, and the MSE is taken only inside the blanked span. The audio condition is dropped with probability 0.3, and audio plus text together with probability 0.2. `sample` is inference (ODE integration with classifier-free guidance, sway sampling, EPSS timesteps on by default when starting from t=0). Raw-waveform conditioning is converted to mel inside `sample`. Duration is clamped to at least prompt length, text length, plus one.
- Backbones, picked by `model.backbone` in the YAML and resolved with `hydra.utils.get_class("f5_tts.model.<name>")`:
  - `DiT` (`backbones/dit.py`): used by F5-TTS. Text is embedded with a ConvNeXt stack and concatenated with noisy mel plus masked reference mel; AdaLN timestep conditioning.
  - `MMDiT` (`backbones/mmdit.py`): separate text and audio embeddings fed through `MMDiTBlock`s with joint attention (`JointAttnProcessor` in `modules.py`).
  - `UNetT` (`backbones/unett.py`): transformer with U-Net style skip connections (`skip_connect_type` concat/add/none); used by the `E2TTS_*` configs.
- `model/modules.py`: shared blocks. Mel spectrogram, rotary embeddings, attention, feed-forward, ConvNeXt V2, AdaLN, sinusoidal time embedding.
- `model/__init__.py` exports `CFM`, `DiT`, `MMDiT`, `UNetT`, `Trainer`.

## Configs

`configs/<name>.yaml`, one per model: `F5TTS_v1_Base`, `F5TTS_v1_Small`, `F5TTS_Base`, `F5TTS_Small`, `E2TTS_Base`, `E2TTS_Small`. Sections: `datasets`, `optim`, `model` (backbone, arch, mel_spec, vocoder, tokenizer), `ckpts`, `hydra`. `arch` also carries `attn_backend` (`torch` or `flash_attn`) and `checkpoint_activations`. Training output goes to `ckpts/<model>_<mel>_<tokenizer>_<dataset>/`. The same file drives training (`train/train.py`, via `@hydra.main`) and inference (`api.py` loads it with OmegaConf).

## Inference path

1. `api.F5TTS` loads the YAML, vocoder (`load_vocoder`) and checkpoint (`load_model`, downloaded from Hugging Face via `cached_path` unless `ckpt_file` is given).
2. `preprocess_ref_audio_text`: trims/clips reference audio, transcribes with `openai/whisper-large-v3-turbo` (transformers pipeline) if no `ref_text` is given. Processed audio and transcripts are cached by MD5 of the input file. Reference audio is clipped to about 12 s, and `ref_text` is forced to end in sentence punctuation.
3. `infer_process`: derives a per-chunk budget in UTF-8 bytes from the reference speaking rate, calls `chunk_text` to split `gen_text` at sentence punctuation (ASCII and CJK).
4. `infer_batch_process`: reference audio is made mono, raised to RMS 0.1 if quieter, and resampled to 24 kHz. Per chunk it builds `ref_text + gen_text`, converts to pinyin, runs `CFM.sample`, drops the reference frames from the output, and decodes with the vocoder. Chunks under 10 bytes get speed x0.3. Non-streaming runs chunks in a `ThreadPoolExecutor` and cross-fades them (0.15 s default); streaming mode runs them in order and yields slices. With `fix_duration` and several chunks, the total is split across chunks by byte weight.

Inference precision: fp16 on CUDA devices with compute capability 7+, else fp32; BigVGAN always fp32. Vocos is fetched from `charactr/vocos-mel-24khz`. BigVGAN (`nvidia/bigvgan_v2_24khz_100band_256x`) needs the `src/third_party/BigVGAN` git submodule, which is empty in this checkout.
5. Optional: `remove_silence_for_generated_wav`, `save_spectrogram`.

Other inference front ends reuse steps 2-5: `infer/infer_cli.py` (TOML config, CLI flags override it; named `[voices.<name>]` tables, selected in the text with `[name]` tags, `gen_file` for text from a file), `infer/infer_gradio.py` (tabs Basic-TTS, Multi-Speech, Voice-Chat, Credits; chat replies come from a Qwen2.5 model through `transformers`; custom checkpoint loading), `infer/speech_edit.py` (a script, not a CLI: settings and the edit regions are constants edited in the file, passed to `CFM.sample` as `edit_mask`; checkpoint comes from Hugging Face), `socket_server.py` / `socket_client.py`.

`socket_server.py`: plain TCP, default port 9998, one client at a time. The client sends UTF-8 text; the server replies with raw float32 samples at 24 kHz and a final `END` marker. The first request is split into smaller chunks to cut time to first audio. Output is also written to `output.wav` by a background thread. `socket_client.py` plays the stream with PyAudio.

`api.F5TTS.infer` wraps steps 2-3 with a seed (random if unset), then optional wav and spectrogram export.

## Training path

1. Data prep scripts in `train/datasets/prepare_*.py` write an Arrow dataset plus `duration.json` and `vocab.txt` under `data/<dataset>_<tokenizer>/`.
2. `train/train.py` builds `CFM(transformer=backbone(...))`, a `Trainer`, and the dataset from `model/dataset.py`.
3. `model/dataset.py`: `load_dataset` returns `CustomDataset` (reads `data/<name>_<tokenizer>/raw.arrow` or `raw/`, or `mel.arrow` for precomputed mels, plus `duration.json`), or `HFDataset` for Hugging Face sources. `DynamicBatchSampler` sorts by frame length and packs batches under a per-GPU frame threshold, capped by `max_samples`. `collate_fn` zero-pads mels and keeps text as a list.
4. `model/trainer.py` `Trainer`: Accelerate for multi-GPU, EMA copy, linear warmup then linear decay LR (`SequentialLR`), optional bitsandbytes 8-bit AdamW, wandb/tensorboard logging, checkpoints `model_<update>.pt` and `model_last.pt` (model, optimizer, EMA, scheduler, update count). Resume prefers `model_last.pt`, then the newest `model_*`, then a `pretrained_*` file.
5. Fine-tuning: `train/finetune_cli.py` with `--finetune` copies a pretrained checkpoint (`--pretrain`, default downloaded from Hugging Face per model) into `ckpts/<dataset>/pretrained_<name>` and trains from it. `train/finetune_gradio.py` is a UI over the whole flow: Whisper transcription of uploaded audio, metadata/Arrow/vocab creation, vocab extension (`expand_model_embeddings` adds randomly initialised rows), launching `accelerate launch finetune_cli.py` as a subprocess, and checkpoint pruning.

## Tokenizer

`model/utils.py` `get_tokenizer(dataset_name, tokenizer)`: `pinyin` (Chinese to pinyin, `convert_char_to_pinyin`), `char`, `byte`, or `custom` (path to a `vocab.txt`). Vocab file lives next to the dataset. Inference does not use these paths: `load_model` always uses tokenizer `custom` with a vocab file (default `infer/examples/vocab.txt`, same content as `data/Emilia_ZH_EN_pinyin/vocab.txt` apart from line endings) and always runs text through `convert_char_to_pinyin`. `vocab.txt` must have space at index 0 (asserted); index 0 also stands in for any unknown character. Text batches are padded with `-1`. `byte` tokenizer has no vocab file and uses 256 ids.

## Evaluation

`eval/`, needs the `eval` extra. Two stages.

1. Generate: `eval_infer_batch.py` (`-n` expname, `-c` ckpt step, `-nfe`, `-t` testset: LibriSpeech-PC test-clean or Seed-TTS zh/en). Builds prompts with `utils_eval.get_inference_prompt`, samples with `CFM`, writes wavs. Uses Accelerate for multi-GPU. Only the LibriSpeech-PC list is in `data/`; the Seed-TTS test set is expected at `data/seedtts_testset/{zh,en}/meta.lst` and must be downloaded separately.
2. Score, one script per test set (`eval_librispeech_test_clean.py`, `eval_seedtts_testset.py`), `-e wer|sim`, fanned out across GPUs with multiprocessing:
   - WER (`utils_eval.run_asr_wer`): transcribe with FunASR paraformer-zh for `zh` or faster-whisper large-v3 for `en`; `zhconv` converts to simplified Chinese, `zhon` punctuation is stripped, `jiwer.process_words` computes the error rate.
   - Speaker similarity (`run_sim`): ECAPA-TDNN (`ecapa_tdnn.py`, adapted from Microsoft UniSpeech) on WavLM-large features loaded through `torch.hub` from `s3prl/s3prl`; score is cosine similarity of the two embeddings. Needs `wavlm_large_finetune.pth`.
   - UTMOS (`eval_utmos.py`): `utmos22_strong` from `tarepan/SpeechMOS` via `torch.hub`; averages over a directory of wavs.
   - Checkpoint paths are hard-coded relative to `../checkpoints/` in the eval scripts.

## Deployment

`runtime/triton_trtllm/`. NVIDIA GPU, TensorRT-LLM 0.16.0, Triton 24.12 (`Dockerfile.server`). F5-TTS (DiT) models only: `run.sh` accepts `F5TTS_v1_Base`, `F5TTS_Base`, `F5TTS_v1_Small`, `F5TTS_Small`, and the vocoder export is Vocos only. `scripts/convert_checkpoint.py` takes DiT hyperparameters as flags (defaults match the Base config).

`run.sh <start_stage> <stop_stage> [model]`:

| Stage | Does |
|---|---|
| 0 | Download checkpoint from Hugging Face, export Vocos to ONNX, build TensorRT engine |
| 1 | Convert checkpoint to TRT-LLM format; copy `patch/*` into the installed `tensorrt_llm/models` (path hard-coded to python3.12 dist-packages); `trtllm-build` |
| 2 | Export Vocos vocoder |
| 3 | Build Triton model repo from `model_repo_f5_tts/`, fill `config.pbtxt` placeholders |
| 4 | Run `tritonserver` |
| 5, 6 | Test with `client_grpc.py`, `client_http.py` |
| 7, 8 | Benchmark offline with TRT-LLM and native PyTorch (`benchmark.py --backend-type trt|pytorch`) |

Triton model repo has two models:
- `f5_tts`: Python backend, `max_batch_size` 4, GPU instance. `1/model.py` does text tokenization (rjieba, pypinyin) and mel extraction, then calls `F5TTS` in `1/f5_tts_trtllm.py`, which runs the TRT-LLM engine and the ODE loop. Inputs: reference wav, its length, reference text, target text. Output: waveform.
- `vocoder`: TensorRT backend, loads `vocoder.plan`, mel in and waveform out.

`patch/f5tts/` (`model.py`, `modules.py`) is a TensorRT-LLM `PretrainedModel` definition of the F5-TTS DiT, written with `tensorrt_llm` layers. It is not a copy of `src/f5_tts/model/`. `benchmark.py` does import from `f5_tts` (mel, tokenizer, `padded_mel_batch`, and `DiT` for the PyTorch backend). The Triton server's own `f5_tts_trtllm.py` does not import `f5_tts`; it carries its own `TextEmbedding` and `ConvNeXtV2Block`.

## Misc

- `device_utils.py` `get_default_device`: `torch.accelerator.current_accelerator()` if the torch build has it, otherwise probes cuda, npu, xpu, mps in that order, else cpu.
- `scripts/`: parameter/FLOP counting and epoch math helpers.
- `data/`: tokenizer vocab and eval lists. `ckpts/`: checkpoint output dir.
