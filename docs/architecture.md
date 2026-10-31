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

- `model/cfm.py` `CFM`: wraps a backbone. `forward` is the training loss (random mask span, conditional flow matching, audio/text condition dropout). `sample` is inference (ODE integration with classifier-free guidance, sway sampling, optional EPSS timesteps).
- Backbones, picked by `model.backbone` in the YAML and resolved with `hydra.utils.get_class("f5_tts.model.<name>")`:
  - `DiT` (`backbones/dit.py`): used by F5-TTS. Text is embedded with a ConvNeXt stack and concatenated with noisy mel plus masked reference mel; AdaLN timestep conditioning.
  - `MMDiT` (`backbones/mmdit.py`): two-stream joint attention, text and audio streams.
  - `UNetT` (`backbones/unett.py`): transformer with U-Net style skip connections (`skip_connect_type` concat/add/none); used by the `E2TTS_*` configs.
- `model/modules.py`: shared blocks. Mel spectrogram, rotary embeddings, attention, feed-forward, ConvNeXt V2, AdaLN, sinusoidal time embedding.
- `model/__init__.py` exports `CFM`, `DiT`, `MMDiT`, `UNetT`, `Trainer`.

## Configs

`configs/<name>.yaml`, one per model: `F5TTS_v1_Base`, `F5TTS_v1_Small`, `F5TTS_Base`, `F5TTS_Small`, `E2TTS_Base`, `E2TTS_Small`. Sections: `datasets`, `optim`, `model` (backbone, arch, mel_spec, vocoder, tokenizer), `ckpts`, `hydra`. The same file drives training (`train/train.py`, via `@hydra.main`) and inference (`api.py` loads it with OmegaConf).

## Inference path

1. `api.F5TTS` loads the YAML, vocoder (`load_vocoder`) and checkpoint (`load_model`, downloaded from Hugging Face via `cached_path` unless `ckpt_file` is given).
2. `preprocess_ref_audio_text`: trims/clips reference audio, transcribes with `openai/whisper-large-v3-turbo` (transformers pipeline) if no `ref_text` is given. Processed audio and transcripts are cached by MD5 of the input file. Reference audio is clipped to about 12 s, and `ref_text` is forced to end in sentence punctuation.
3. `infer_process`: derives a per-chunk character budget from reference speaking rate, calls `chunk_text` to split `gen_text`.
4. `infer_batch_process`: per chunk, runs `CFM.sample`, decodes with the vocoder, then cross-fades chunks together.
5. Optional: `remove_silence_for_generated_wav`, `save_spectrogram`.

Other inference front ends reuse steps 2-5: `infer/infer_cli.py` (TOML config + flags), `infer/infer_gradio.py` (web UI, multi-speaker and chat tabs), `infer/speech_edit.py` (replace a span of existing audio), `socket_server.py` / `socket_client.py` (streaming over TCP).

## Training path

1. Data prep scripts in `train/datasets/prepare_*.py` write an Arrow dataset plus `duration.json` and `vocab.txt` under `data/<dataset>_<tokenizer>/`.
2. `train/train.py` builds `CFM(transformer=backbone(...))`, a `Trainer`, and the dataset from `model/dataset.py`.
3. `model/dataset.py`: `HFDataset` / `CustomDataset` read samples; `DynamicBatchSampler` packs batches by total frames; `collate_fn` pads.
4. `model/trainer.py` `Trainer`: Accelerate for multi-GPU, EMA copy, linear warmup then linear decay LR (`SequentialLR`), optional bitsandbytes 8-bit AdamW, wandb/tensorboard logging, periodic checkpoints to `ckpts/`.
5. Fine-tuning: `train/finetune_cli.py` and `train/finetune_gradio.py` start from a pretrained checkpoint (`--pretrain`, default downloaded from Hugging Face per model) on a custom dataset.

## Tokenizer

`model/utils.py` `get_tokenizer(dataset_name, tokenizer)`: `pinyin` (Chinese to pinyin, `convert_char_to_pinyin`), `char`, `byte`, or `custom` (path to a `vocab.txt`). Vocab file lives next to the dataset. `vocab.txt` must have space at index 0 (asserted); index 0 also stands in for any unknown character. Text batches are padded with `-1`. `byte` tokenizer has no vocab file and uses 256 ids.

## Evaluation

`eval/`: batch inference over LibriSpeech-PC test-clean and Seed-TTS test sets, then WER (faster-whisper large-v3 for English, FunASR paraformer-zh for Chinese), speaker similarity (ECAPA-TDNN on WavLM-large features, `ecapa_tdnn.py`), and UTMOS. Checkpoints are expected under `../checkpoints/`. Needs the `eval` extra.

## Deployment

`runtime/triton_trtllm/`: TensorRT-LLM engine conversion (`scripts/convert_checkpoint.py`), vocoder ONNX export, Triton model repo, gRPC/HTTP clients, benchmarks, Docker files. `patch/` is not a copy of `model/`. It holds F5-TTS model definitions written against the TensorRT-LLM API, and `run.sh` copies it into the installed `tensorrt_llm/models` package before `trtllm-build`.

## Misc

- `device_utils.py` `get_default_device`: `torch.accelerator.current_accelerator()` if the torch build has it, otherwise probes cuda, npu, xpu, mps in that order, else cpu.
- `scripts/`: parameter/FLOP counting and epoch math helpers.
- `data/`: tokenizer vocab and eval lists. `ckpts/`: checkpoint output dir.
