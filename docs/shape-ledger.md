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

### 2026-11-01 MMDiT text embedding annotated as integer
- Where: `model/backbones/mmdit.py`, `TextEmbedding.forward`
- What happened: the return was annotated `int["b nt d"]`, but the function returns the output of `nn.Embedding` plus positional embeddings, which is a float tensor. No runtime effect, because the annotation was never evaluated or checked.
- How found: review while converting ein notation to jaxtyping in Plan P, batch A.
- Time lost: none, but the wrong dtype sat in the code and would have misled anyone reading it.
- Would a shape-aware static checker have caught it: unsure. A checker that infers dtypes through `nn.Embedding` and `+` would. Pyright with jaxtyping does not, since it reads both as `Tensor`.
- Fix: annotation changed to `Float[Tensor, "b nt d"]`.
