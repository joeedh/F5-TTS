# CLAUDE.md

## Comments

- Temporary code comments (debug notes, scaffolding, "remove later") start with `CLAUDENOTE:` so they can be stripped with a grep.
- Permanent comments in code are at most 5 lines. Doc comments (docstrings, API docs) are not limited.
- Exception: one comment longer than 5 lines is allowed per 500 lines of code.
- Exception: comments written before October 2026 keep their length. Do not trim them.

## Key source files

Paths under `src/f5_tts/`. Overview in `docs/architecture.md`; doc list in `docs/index.md`.

| File | Role |
|---|---|
| `api.py` | `F5TTS` class, the Python entry point for inference |
| `model/cfm.py` | `CFM`: flow-matching training loss and ODE sampling |
| `model/backbones/dit.py` | `DiT` backbone (F5-TTS) |
| `model/backbones/mmdit.py`, `unett.py` | `MMDiT`, `UNetT` (E2-TTS) backbones |
| `model/modules.py` | Mel spec, attention, rotary, ConvNeXt, AdaLN blocks |
| `model/trainer.py` | `Trainer` (Accelerate, EMA, checkpoints) |
| `model/dataset.py` | Datasets, `DynamicBatchSampler`, `collate_fn` |
| `model/utils.py` | Tokenizers, masks, seeding, EPSS timesteps |
| `infer/utils_infer.py` | Ref audio prep, text chunking, vocoder/model loading, batch inference |
| `infer/infer_cli.py`, `infer_gradio.py`, `speech_edit.py` | Inference front ends |
| `train/train.py` | Hydra training entry point |
| `train/finetune_cli.py`, `finetune_gradio.py` | Fine-tuning front ends |
| `train/datasets/prepare_*.py` | Dataset preparation |
| `configs/*.yaml` | Per-model architecture, optimizer, dataset, checkpoint config |
| `socket_server.py` | Streaming TTS server |
| `eval/` | WER, speaker similarity, UTMOS evaluation |
| `runtime/triton_trtllm/` | TensorRT-LLM / Triton deployment (has patched copies of model code) |

## Docs layout

| Kind | Location |
|---|---|
| Design docs | `docs/` |
| Research | `docs/research/` |
| Plans | `docs/plans/` |
| Debugging guide | `docs/debugging.md` |
| Index of all docs | `docs/index.md` (update when adding a doc) |
| Tensor shape ledger | `docs/shape-ledger.md` |
| Architecture | `docs/architecture.md` |

## Writing plans and docs

- Do not imitate the prose style of existing documents in the repo. Write in your own plain style.
- Plans are saved to the repo in `docs/plans/`, not left in chat.
- Pressure test every plan with a fresh agent (no shared context, give it only the plan file). Fold its findings back into the plan file before treating the plan as final.

## Debugging guide

`docs/debugging.md` is a running log. When a bug is found and fixed, or a debugging technique proves useful, add an entry: symptom, cause, fix or how to diagnose. Append new entries; edit old ones only to correct them.

## Tensor shapes

Shape annotations use `jaxtyping`, e.g. `Float[Tensor, "b n d"]`. Pyright does not check the shapes, only that the value is a `Tensor`. Whenever a bug, a long debugging session or a near miss comes from a wrong tensor shape or dtype that a shape-aware checker would have caught, add an entry to `docs/shape-ledger.md` in the same commit as the fix. Do this even if the cost was small; the ledger is how we decide whether to extend pyright.
