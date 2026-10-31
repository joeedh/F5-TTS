# Tensor shape ledger

Decision (2026-10-31): shape annotations use `jaxtyping` (`Float[Tensor, "b n d"]`). Pyright sees these as plain `Tensor` and does not check shapes or dtypes. If that costs us enough, we consider extending pyright (a fork or plugin-like patch) so it checks them.

This file records each time a missing shape check caused a problem, so the decision can be made from evidence. Add an entry whenever a bug, wasted debugging time or near miss traces to a wrong shape or dtype that a shape-aware checker would have caught.

Entry format, newest at the bottom:

```
### YYYY-MM-DD short title
- Where: file and function
- What happened: wrong shape or dtype, and the visible symptom
- How found: runtime error, wrong audio, test, review
- Time lost: rough estimate
- Would a shape-aware static checker have caught it: yes / no / unsure, and why
- Fix:
```

Review the ledger when it has about five entries, or before starting plan 01 (ONNX export), whichever comes first.

## Entries

None yet.
