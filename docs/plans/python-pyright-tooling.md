# Plan P: pyright typing, with ruff for format and lint

Status: drafted, not yet pressure-tested. Task list entry: [webgpu-port-tasks.md](webgpu-port-tasks.md).

## Goal

`pyright` passes with zero errors on `src/f5_tts` (minus the exclusions below), runs in CI, and tensor shapes are written with `jaxtyping`. Ruff stays the only formatter and linter. Behaviour of the code does not change.

## Decisions already made

| Topic | Decision |
|---|---|
| Type checker | pyright |
| Formatter | `ruff format` |
| Linter | `ruff check` (includes import sorting) |
| Shape annotations | `jaxtyping`, e.g. `Float[Tensor, "b n d"]`. Pyright sees plain `Tensor`. Cost of missing shape checks is tracked in [../shape-ledger.md](../shape-ledger.md). |
| Conversion strategy | Convert many files at once in batches, then drive errors to zero afterwards. No file-by-file perfection, and no CI gate until the count is zero. |
| Suppressions | Only `# pyright: ignore[ruleName]` with a reason. No bare `# type: ignore`. |
| Branch | Work directly on `main`, one commit per step below. |

## Scope

In: every `.py` file under `src/f5_tts/` except the exclusions, and `tools/`.

Out (excluded from pyright, still linted and formatted by ruff):
- `src/f5_tts/runtime/triton_trtllm/`: written against the TensorRT-LLM API, which is only importable inside NVIDIA's container. Pyright cannot resolve it.
- `src/third_party/`: git submodule (BigVGAN), empty in a plain clone.

Non-goals: changing runtime behaviour, adding unit tests for model quality, runtime shape checking, strict mode.

## Measured starting point (2026-10-31)

- Python 3.13 in the cloud container; the project targets 3.10 (`ruff.toml`).
- pyright 1.1.414 on `src/f5_tts` without torch installed: 672 errors, 23 warnings, 47 files. Main groups: missing imports 239, possibly unbound 130, attribute access 109, invalid type arguments 69 (the ein strings), argument type 60, call issue 21, missing module source 16, assignment type 8. The missing-imports group is mostly an artefact of torch not being installed, so the real baseline must be remeasured in step 1.
- Ruff installed in the container is 0.16.9. `.pre-commit-config.yaml` pins ruff v0.11.2. `ruff check src tools` with 0.16.9 reports 168 errors. Most-common rules: C408 (26), RUF059 (15), UP037 (13), BLE001 (13), RUF100 (12). The pinned 0.11.2 may report a different set. The mismatch has to be resolved first or lint results will not be reproducible.
- `ruff format --check src`: 53 files already formatted.
- Existing suppressions: 4 `# type: ignore` (2 in `model/modules.py`, 2 in the Triton copy), plus `# ruff: noqa: F722 F821` at the top of 6 model files for the ein strings.
- Largest files: `train/finetune_gradio.py` 1876 lines, `infer/infer_gradio.py` 1137, `model/modules.py` 862, `infer/utils_infer.py` 625.

## Step 0: guard against behaviour change

Do this before touching any code, because every later step is checked against it.

1. Write `tools/typing_guard/golden.py`. It builds a small `DiT` (for example dim 64, depth 2, heads 2, text_dim 32, random init with `torch.manual_seed`), wraps it in `CFM` with a tiny made-up vocab, and runs: one `CFM.forward` call on a fixed random mel and text, and one `CFM.sample` call (steps 4, fixed `seed`). It writes sha256 of each output tensor (as float32 bytes) and a few summary stats to a JSON file.
2. Do the same for `chunk_text`, `convert_char_to_pinyin` and `get_epss_timesteps` on a fixed set of strings and step counts.
3. Also an import check: `python -c "import importlib, pkgutil; ..."` importing every module in `f5_tts` except the excluded and GPU-only ones (list the ones that fail to import for non-typing reasons before starting, so they are not mistaken for regressions).
4. Run it on the untouched `main`, commit the JSON as the golden file.
5. After each later batch, rerun and diff against the golden file. A difference means the batch changed behaviour and must be fixed or explained.

The hash comparison needs a deterministic CPU run on the same torch version. Record the torch version in the JSON and compare only within one environment.

Limits of this guard: it covers the model core and a few text helpers, not the Gradio apps, trainer, dataset code, eval code or the socket server. Those get a syntax/import check only, plus review.

## Step 1: environment and baseline

1. Create a virtualenv with the project's dependencies and CPU torch. The container's pip resolves torch (2.15.0 listed) from the default index; the PyTorch CPU index returned no match, so use the default index and expect a large download. If the download is not possible, say so in the plan and fall back to a stub-based check (see risks).
2. Install `pyright` (pip package `pyright`, pinned) into the same environment. Record versions in [../debugging.md](../debugging.md) along with the exact commands.
3. Run pyright with the draft config from step 2 and record the new baseline per rule in this plan (replace the 672 figure). Also record per-file counts so batches can be sized.

## Step 2: configuration

1. `pyproject.toml`: add `[tool.pyright]` with `pythonVersion = "3.10"`, `include = ["src/f5_tts", "tools"]`, `exclude` for the two out-of-scope paths, `typeCheckingMode = "basic"`, and `venvPath`/`venv` so CI and local runs resolve the same packages.
2. Add `jaxtyping` to `dependencies` (it is imported by model code at import time, so it is a runtime dependency).
3. `ruff.toml`: add `[lint.per-file-ignores]` entries for `F722` and `F821` on the files that use shape strings, then delete the 6 inline `# ruff: noqa: F722 F821` lines. Verify with `ruff check` that no new errors appear.
4. Resolve the ruff version question: pick one version (default proposal: the newest that passes, and update the pre-commit `rev` to it), apply `ruff check --fix` for safe fixes and `ruff format` once in a separate commit labelled formatting-only, and fix or deliberately ignore the rest. Keep the existing rule selection; do not add rule sets in this plan.
5. Commit config separately from any code change.

## Step 3: bulk conversion

Batches, in this order. Each batch is one commit and is followed by the guard from step 0.

| Batch | Files | Notes |
|---|---|---|
| A | `model/utils.py`, `model/modules.py`, `model/cfm.py`, `model/backbones/*.py` | Shape-annotation conversion happens here. |
| B | `model/dataset.py`, `model/trainer.py` | Trainer has many Accelerate/EMA/wandb calls with weak typing. |
| C | `infer/utils_infer.py`, `api.py`, `device_utils.py`, `socket_server.py`, `socket_client.py`, `infer/infer_cli.py`, `infer/speech_edit.py` | Module-level script code in `infer_cli.py` and `speech_edit.py` produces many possibly-unbound errors. |
| D | `infer/infer_gradio.py`, `train/*.py`, `train/datasets/*.py`, `scripts/*.py` | Gradio files are large; expect many `Optional` issues. |
| E | `eval/*.py`, `tools/` | `ecapa_tdnn.py` is adapted third-party code; keep edits minimal. |

Within a batch:
1. Convert ein-notation annotations to jaxtyping using this map: `float[...]` to `Float[Tensor, ...]`, `int[...]` to `Int[Tensor, ...]`, `bool[...]` to `Bool[Tensor, ...]`. Unions such as `float["b n d"] | float["b nw"]` keep the `|`. Names appearing only in the shape strings stay strings.
2. Add parameter and return annotations to every function in the batch. Use `Any` only where a real type is unknowable without large effort, and mark each such place with a `CLAUDENOTE:` comment so they can be found later.
3. Do not fix the errors this surfaces yet, other than trivial ones that arise from the annotation itself. Do not reformat or reorder code.
4. Run the guard, then `ruff check` and `ruff format --check`. Commit.

Out-of-band rule: a bug found while annotating is not fixed in the batch commit. Write it down (step 4).

## Step 4: drive errors to zero

1. Re-run pyright, group by rule, and work biggest group first. Likely order: possibly unbound, attribute access, argument type, optional member access, call issue, assignment type, the rest.
2. Known patterns to expect, so they are fixed once and consistently:
   - Hydra/OmegaConf config objects (`model_cfg.model.backbone`) are `DictConfig`; attribute access on them reports errors. Use one helper or a typed alias, not scattered ignores.
   - Module-level globals set later (ASR pipeline, chat model state) need explicit `Optional` handling.
   - Conditional imports and `if/elif` chains that define a variable in only some branches (vocoder selection, ckpt type).
   - Untyped third-party packages (`x_transformers`, `vocos`, `ema_pytorch`, `accelerate`, `faster_whisper`, `funasr`): decide per package between a local stub under `typings/`, `reportMissingTypeStubs` off, or a narrow ignore.
3. Separate commits for fixes that change runtime behaviour (anything beyond annotations and casts). Each such commit gets an entry in [../debugging.md](../debugging.md); each fix of a wrong tensor shape or dtype also gets an entry in [../shape-ledger.md](../shape-ledger.md), as `CLAUDE.md` requires.
4. Every suppression carries the rule name and a one-line reason. At the end, list all suppressions (`grep "pyright: ignore"`) and review them once.
5. Done when `pyright` reports 0 errors. Warnings are listed and either fixed or accepted in writing.

## Step 5: lock it in

1. CI: add a `typecheck` job to `.github/workflows` that installs the project with CPU torch and runs pyright. Pre-commit stays ruff-only, because pyright needs the full dependency set and is slow for a commit hook.
2. Pin pyright and the torch version used for checking, and document how to bump them. A pyright upgrade can add errors; bump deliberately, not automatically.
3. Update `CLAUDE.md` with the commands: type check, lint, format.
4. Update [architecture.md](../architecture.md) if any file was renamed or moved (none are planned).
5. Decide whether to ratchet to `standard` mode, per directory, and write the outcome in this plan. Not part of this plan's completion.

## Acceptance criteria

- `pyright` exits 0 locally and in CI.
- `ruff check` and `ruff format --check` exit 0 with the pinned ruff version, locally, in pre-commit and in CI.
- The step 0 golden comparison matches after every batch.
- No bare `# type: ignore`. Every `# pyright: ignore` names a rule and gives a reason.
- `CLAUDENOTE:` `Any` markers are either resolved or listed with a reason.
- `jaxtyping` is in `pyproject.toml` dependencies and no ein-notation string annotations remain.

## Risks

- Torch typing varies by version; the pinned torch decides what is an error. Mitigation: pin it in the CI job.
- If torch cannot be installed in the working environment, pyright will see most tensor code as `Unknown` and the error count will be misleading. Mitigation: do not proceed past step 1 without it; if impossible, work on a machine that has it.
- Large diffs are hard to review. Mitigation: batches, commit per batch, formatting-only commit kept separate.
- The step 0 guard covers only part of the code. Behaviour changes in Gradio, trainer or eval code could slip through. Mitigation: keep batch edits to annotations; review diffs of those files line by line.
- jaxtyping with `from __future__ import annotations`: annotation strings are never evaluated at runtime unless a runtime checker is enabled, so this is safe. Enabling a checker later needs a separate evaluation.

## Rollback

Each step is a commit on `main`. A bad batch is reverted alone with `git revert`. Config-only commits (steps 2 and 5) can be reverted without touching code.

## Questions to settle during the work

- Which ruff version to pin (step 2.4).
- Whether any untyped third-party package deserves a local stub (step 4.2).
- Whether to add `typings/` to the repo.
