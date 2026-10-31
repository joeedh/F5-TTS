# Plan P: pyright typing, with ruff for format and lint

Status: pressure-tested 2026-10-31 by a fresh agent; its findings are folded in below (section "Review log"). Task list entry: [webgpu-port-tasks.md](webgpu-port-tasks.md).

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
| Branch | Work directly on `main`. One commit per batch or per fix group, never mixing formatting, typing and behaviour fixes. |

## Scope

In: every `.py` file under `src/f5_tts/` except the exclusions, and `tools/typing_guard/` (the only Python under `tools/`, created in step 2; today `tools/` holds only a `.cjs` file).

Out (excluded from pyright, still linted and formatted by ruff):
- `src/f5_tts/runtime/triton_trtllm/`: written against the TensorRT-LLM API, which is only importable inside NVIDIA's container. Pyright cannot resolve it.
- `src/third_party/`: git submodule (BigVGAN), empty in a plain clone.

Non-goals: changing runtime behaviour, adding unit tests for model quality, runtime shape checking, strict mode.

## Measured starting point (2026-10-31)

- Python 3.13 in the cloud container; the project targets 3.10 (`ruff.toml`).
- pyright 1.1.414 on `src/f5_tts` without torch installed: 672 errors, 23 warnings, 47 files. Main groups: missing imports 239, possibly unbound 130, attribute access 109, invalid type arguments 69 (the ein strings), argument type 60, call issue 21, missing module source 16, assignment type 8. The missing-imports group is mostly an artefact of torch not being installed, so the real baseline must be remeasured in step 1.
- Ruff: `.pre-commit-config.yaml` pins v0.11.2, and `ruff.toml` has no `select` or `ignore`, so the rule set is ruff's default for the pinned version. With 0.11.2, `ruff check src tools` passes and `ruff format --check src` reports all 47 checked files already formatted (run in a scratch virtualenv, 2026-10-31). The ruff preinstalled in the cloud container is 0.16.9, whose newer defaults report 168 errors (C408, RUF059, UP037, BLE001, RUF100 lead). Those 168 are not real debt under the pinned version. Decision: stay on v0.11.2, change no ruff rules, and run ruff from a virtualenv with exactly that version.
- Existing suppressions: 4 bare `# type: ignore` (`model/modules.py:214-215`, and 2 in the excluded Triton copy `f5_tts_trtllm.py:118-119`), plus `# ruff: noqa: F722 F821` at the top of 6 model files for the ein strings.
- There is no `src/f5_tts/__init__.py` (namespace package); keep that in mind for module walking and pyright `include`.
- Largest files: `train/finetune_gradio.py` 1876 lines, `infer/infer_gradio.py` 1137, `model/modules.py` 862, `infer/utils_infer.py` 625.

## Step 1: environment

1. Create a virtualenv with the project's dependencies, the `eval` extra, and CPU torch. pip resolves torch (2.15.0 listed) from the default index; the PyTorch CPU index returned no match, so use the default index and expect a large download. Without torch the baseline is meaningless, so do not continue until it works. If it truly cannot be installed here, move this plan to a machine that has it.
2. Install ruff 0.11.2 and a pinned `pyright` pip package into the same environment. Record versions and the exact commands in [../debugging.md](../debugging.md).
3. Check that the torch version still supports the Python version CI will use (see step 6.1).

## Step 2: guard against behaviour change

Do this on untouched `main` before any code change. Every later step is checked against it. It is a tripwire, not a proof: annotation-only edits cannot change numerical output, so a golden mismatch mostly means an accidental edit, a changed default or a changed signature. Batches that only add annotations are expected to leave it unchanged.

Files: `tools/typing_guard/golden.py` and the stored result `tools/typing_guard/golden.json`.

1. Numerical part. Seed both `random` and `torch` (the model uses Python's `random.random` and `torch.rand/randn`). Build a small `DiT` with small dims and random init, put it in `CFM` with a tiny made-up `vocab_char_map`, and call `.eval()` where dropout would matter.
   - `CFM.forward`: pass a fixed random 3D mel `(b, n, 100)` (3D skips `MelSpec`) and a list of strings in the tiny vocab.
   - `CFM.sample`: pass a fixed 3D `cond`, text as a list of strings, an integer `duration`, `steps=4`, a fixed `seed`.
   - Store sha256 of the float32 bytes of each output plus a few summary stats.
2. Helpers part: call `convert_char_to_pinyin` and `get_epss_timesteps` (both in `model/utils.py`) and `chunk_text` (in `infer/utils_infer.py`, heavy imports) on a fixed set of inputs; store outputs.
3. Signature snapshot: for the public functions and classes of an allow-list of modules, record `inspect.signature` parameter names, kinds and defaults. Annotations are ignored in the comparison, so this catches changed defaults or parameters but not annotation edits.
4. Syntax check for everything: `python -m compileall -q src/f5_tts` (no imports executed).
5. Import check for an allow-list only: `f5_tts.model` and its submodules, `f5_tts.infer.utils_infer`, `f5_tts.api`, `f5_tts.device_utils`. Do not import `infer_cli`, `speech_edit`, the Gradio apps, `socket_*`, `train/*`, `eval/*` or the data prep scripts: they run module-level code, load checkpoints, download files or need extras.
6. Record the torch version in the JSON. Compare only within the same environment.
7. Run once, commit script and JSON.

What the guard does not cover: Gradio apps, trainer, dataset code, eval code, socket server, CLI scripts. Those get the syntax check and line-by-line diff review only.

## Step 3: configuration and baseline

1. `pyproject.toml`: add `[tool.pyright]` with `pythonVersion = "3.10"`, `include = ["src/f5_tts", "tools/typing_guard"]`, `exclude` for `src/f5_tts/runtime/triton_trtllm` and `src/third_party`, and `typeCheckingMode = "basic"`. Do not put `venvPath` or `venv` in the committed file: those are machine-specific and break in CI. Run pyright with `--pythonpath "$(which python)"` locally and in CI.
2. Add `jaxtyping` to `dependencies`. Decision: import it normally at the top of each model file (not under `TYPE_CHECKING`), so it is a runtime dependency. This keeps the option of runtime shape checking open. The alternative, a `TYPE_CHECKING`-only import, would avoid the dependency but loses that option.
3. `ruff.toml`: add `[lint.per-file-ignores]` for `F722` and `F821` on the 6 files that carry `# ruff: noqa: F722 F821` today (`model/utils.py`, `model/modules.py`, `model/cfm.py`, `model/backbones/{dit,mmdit,unett}.py`), then delete those inline lines. Confirm `ruff check` still passes with 0.11.2 and rerun the guard.
4. Do not change ruff rules, and do not run `ruff check --fix` or a reformat: the tree already passes under the pinned version.
5. Run pyright with this config and record the real baseline here: total, per rule, per file. Replace the 672 figure in this plan.
6. Commit config separately from any code change.

## Step 4: bulk conversion

Batches, in this order. Each batch is one commit and is followed by the guard from step 2.

| Batch | Files | Notes |
|---|---|---|
| A | `model/__init__.py`, `model/utils.py`, `model/modules.py`, `model/cfm.py`, `model/backbones/*.py` | Shape-annotation conversion happens here. Also replaces the two bare `# type: ignore` at `modules.py:214-215` with rule-named `# pyright: ignore[...]` or a real fix. |
| B | `model/dataset.py`, `model/trainer.py` | Trainer has many Accelerate/EMA/wandb calls with weak typing. |
| C | `infer/utils_infer.py`, `api.py`, `device_utils.py`, `socket_server.py`, `socket_client.py`, `infer/infer_cli.py`, `infer/speech_edit.py` | Module-level script code in `infer_cli.py` and `speech_edit.py` produces many possibly-unbound errors. |
| D | `infer/infer_gradio.py`, `train/*.py`, `train/datasets/*.py`, `scripts/*.py` | Gradio files are large; expect many `Optional` issues. |
| E | `eval/*.py`, `tools/typing_guard/golden.py` | `ecapa_tdnn.py` is adapted third-party code; keep edits minimal. |

Within a batch:
1. Convert ein-notation annotations to jaxtyping using this map: `float[...]` to `Float[Tensor, ...]`, `int[...]` to `Int[Tensor, ...]`, `bool[...]` to `Bool[Tensor, ...]`. Unions such as `float["b n d"] | float["b nw"]` keep the `|`. A parameter annotated with a tensor type but defaulting to `None` (for example `mask: bool["b n"] = None`) gets `| None`. Names appearing only in the shape strings stay strings.
2. Add parameter and return annotations to every function in the batch. Use `Any` only where a real type is unknowable without large effort, and give each such place a short trailing comment saying why. Do not use `CLAUDENOTE:` for this: `CLAUDE.md` reserves it for temporary code comments that get stripped. The final review (step 5.4) greps for `Any`.
3. Edits are limited to annotations, `| None` on `None` defaults, and invalid annotation syntax such as `dict[str:int]` at `model/cfm.py:50` (becomes `dict[str, int]`; no runtime effect under `from __future__ import annotations`). Anything that touches an executable line belongs to step 5. Do not reformat or reorder code.
4. Run the guard (step 2), then `ruff check` and `ruff format --check` with the pinned version. Commit.

Out-of-band rule: a bug found while annotating is not fixed in the batch commit. Write it down for step 5.

## Step 5: drive errors to zero

1. Re-run pyright, group by rule, and work biggest group first. Likely order: possibly unbound, attribute access, argument type, optional member access, call issue, assignment type, the rest.
2. Known patterns to expect, so they are fixed once and consistently:
   - Hydra/OmegaConf config objects (`model_cfg.model.backbone`) are `DictConfig`; attribute access on them reports errors. Use one helper or a typed alias, not scattered ignores.
   - Module-level globals set later (ASR pipeline, chat model state) need explicit `Optional` handling.
   - Conditional imports and `if/elif` chains that define a variable in only some branches (vocoder selection, ckpt type).
   - Untyped third-party packages (`x_transformers`, `vocos`, `ema_pytorch`, `accelerate`, `faster_whisper`, `funasr`): decide per package between a local stub under `typings/`, `reportMissingTypeStubs` off, or a narrow ignore.
3. Separate commits for fixes that change runtime behaviour (anything beyond annotations and casts). Each such commit gets an entry in [../debugging.md](../debugging.md); each fix of a wrong tensor shape or dtype also gets an entry in [../shape-ledger.md](../shape-ledger.md), as `CLAUDE.md` requires.
4. Every suppression carries the rule name and a one-line reason. At the end, list all suppressions (`grep "pyright: ignore"`) and review them once.
5. Done when `pyright` reports 0 errors. Warnings are listed and either fixed or accepted in writing.

## Step 6: lock it in

1. CI: add a `typecheck` job (new workflow file or a job next to `pre-commit.yaml`, which today only runs pre-commit). It must: choose a Python version (default: 3.10, matching `pythonVersion`; if the pinned torch has no wheel for it, use the oldest version that does and keep `pythonVersion = "3.10"`), install with `pip install -e ".[eval]"` plus the pinned CPU torch, install the pinned pyright pip package (it downloads Node on first run, so allow network), and run `pyright --pythonpath "$(which python)"`. Pre-commit stays ruff-only, because pyright needs the full dependency set and is slow for a commit hook.
2. Pin pyright, ruff and the torch version used for checking, and document how to bump them. A pyright upgrade can add errors; bump deliberately, not automatically.
3. Update `CLAUDE.md` with the commands: type check, lint, format.
4. Update [architecture.md](../architecture.md) if any file was renamed or moved (none are planned).
5. Decide whether to ratchet to `standard` mode, per directory, and write the outcome in this plan. Not part of this plan's completion.

## Acceptance criteria

- `pyright` exits 0 locally and in CI.
- `ruff check` and `ruff format --check` exit 0 with the pinned ruff version, locally, in pre-commit and in CI.
- The step 2 golden comparison matches after every batch.
- No bare `# type: ignore` in non-excluded files: `grep -rn "type: ignore" src --include=*.py --exclude-dir=triton_trtllm` is empty. Every `# pyright: ignore` names a rule and gives a reason.
- Every `Any` has a reason comment, or has been replaced by a real type.
- `jaxtyping` is in `pyproject.toml` dependencies and no ein-notation annotations remain: `grep -rnE '\b(float|int|bool)\["' src --include=*.py --exclude-dir=triton_trtllm` is empty.
- The per-file ruff ignores for F722/F821 replace all inline `# ruff: noqa: F722 F821` lines.

## Risks

- Torch typing varies by version; the pinned torch decides what is an error. Mitigation: pin it in the CI job.
- If torch cannot be installed in the working environment, pyright will see most tensor code as `Unknown` and the error count will be misleading. Mitigation: do not proceed past step 1 without it; if impossible, work on a machine that has it.
- Large diffs are hard to review. Mitigation: batches, commit per batch, formatting-only commit kept separate.
- The step 2 guard covers only part of the code and is a tripwire, not proof. Behaviour changes in Gradio, trainer or eval code could slip through. Mitigation: keep batch edits to annotations; review diffs of those files line by line.
- jaxtyping with `from __future__ import annotations`: annotation strings are never evaluated at runtime unless a runtime checker is enabled, so this is safe. Enabling a checker later needs a separate evaluation.

## Rollback

Each step is a commit on `main`. A bad batch is reverted alone with `git revert`. Config-only commits (steps 2 and 5) can be reverted without touching code.

## Questions to settle during the work

- Whether any untyped third-party package deserves a local stub (step 5.2).
- Whether to add `typings/` to the repo.

## Review log

Pressure test by a fresh agent, 2026-10-31. What changed because of it:

- `tools/` has no Python today: scope, pyright `include` and batch E now name only `tools/typing_guard`.
- Steps reordered: environment, guard, config and baseline. The guard needs torch, and the baseline needs the config.
- Guard rewritten: concrete calls and inputs, signature snapshot, compileall for syntax, import check limited to an allow-list, explicit statement that it is a tripwire.
- Ruff: checked with v0.11.2 in a scratch virtualenv; passes check and format today. The 168 errors came from the preinstalled 0.16.9. Pin v0.11.2, no `--fix` or reformat in this plan.
- `venvPath`/`venv` removed from committed config; CI job specified (Python version, `.[eval]`, pyright download).
- `model/__init__.py` added to batch A; bare `type: ignore` in `modules.py` scheduled; acceptance criteria scoped to non-excluded files and made greppable.
- `dict[str:int]` and `| None` defaults allowed in annotation batches; the "trivial fix" wording replaced with a precise rule.
- One commit per batch or fix group instead of one per step.
- `CLAUDENOTE:` no longer used for `Any`; jaxtyping runtime-dependency choice recorded with its alternative.
- Not adopted: none.
