# CLAUDE.md

## Comments

- Temporary code comments (debug notes, scaffolding, "remove later") start with `CLAUDENOTE:` so they can be stripped with a grep.
- Permanent comments in code are at most 5 lines. Doc comments (docstrings, API docs) are not limited.
- Exception: one comment longer than 5 lines is allowed per 500 lines of code.
- Exception: comments written before October 2026 keep their length. Do not trim them.

## Docs layout

| Kind | Location |
|---|---|
| Design docs | `docs/` |
| Research | `docs/research/` |
| Plans | `docs/plans/` |
| Debugging guide | `docs/debugging.md` |

## Writing plans and docs

- Do not imitate the prose style of existing documents in the repo. Write in your own plain style.
- Plans are saved to the repo in `docs/plans/`, not left in chat.
- Pressure test every plan with a fresh agent (no shared context, give it only the plan file). Fold its findings back into the plan file before treating the plan as final.

## Debugging guide

`docs/debugging.md` is a running log. When a bug is found and fixed, or a debugging technique proves useful, add an entry: symptom, cause, fix or how to diagnose. Append new entries; edit old ones only to correct them.
