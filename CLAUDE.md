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

## Type checking, linting, formatting

Run from the repo root, inside a venv with the project installed (`pip install -e ".[typecheck]"`, plus `pip install jiwer modelscope zhconv zhon` and `pip install --no-deps faster_whisper==0.10.1 funasr` so `eval/` resolves; the `eval` extra itself fails where `av==10.*` has no wheel, i.e. Python 3.12 and newer. See `docs/debugging.md`). Versions are pinned: ruff 0.11.2 (`.pre-commit-config.yaml`) and pyright 1.1.414 (`typecheck` extra in `pyproject.toml`). Do not use a newer ruff; its default rules report errors that are not real debt.

| Task | Command |
|---|---|
| Type check | `pyright --pythonpath "$(which python)"` (Windows: `.venv\Scripts\pyright.exe --pythonpath .venv\Scripts\python.exe`). Must report 0 errors; CI runs it (`.github/workflows/typecheck.yaml`). |
| Lint | `ruff check src/f5_tts tools` (not `src`: the BigVGAN submodule under `src/third_party` is not linted). Also run `ruff check --select I --fix src/f5_tts tools`: pre-commit runs this import sorter as a separate hook and fails on it. |
| Format | `ruff format src/f5_tts tools` |
| Behaviour guard | `python tools/typing_guard/golden.py` before and after a change that should not alter behaviour; `--write` only on a known-good baseline |
| Annotation-only check | `python tools/typing_guard/annotation_only.py <files>` before committing a change that should only add annotations |

Suppressions: only `# pyright: ignore[ruleName]  # reason`, never a bare `# type: ignore`. Every `Any` gets a trailing `# Any: reason` comment. Pyright runs in `basic` mode; `runtime/triton_trtllm` and `src/third_party` are excluded. Ruff's `F722`/`F821` are ignored per file in `ruff.toml` for files that carry jaxtyping shape strings, so a missing import there only shows up in pyright ("is not defined").

## Tensor shapes

Shape annotations use `jaxtyping`, e.g. `Float[Tensor, "b n d"]`. Pyright does not check the shapes, only that the value is a `Tensor`. Whenever a bug, a long debugging session or a near miss comes from a wrong tensor shape or dtype that a shape-aware checker would have caught, add an entry to `docs/shape-ledger.md` in the same commit as the fix. Do this even if the cost was small; the ledger is how we decide whether to extend pyright.
