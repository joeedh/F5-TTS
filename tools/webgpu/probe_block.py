"""Exports one DiT block with every intermediate as an output, for bisecting a WebGPU precision gap.

Inputs are the golden block input at one step (the previous block's output) and the time embedding, so
the graph sees exactly what the full step graph saw. References are the same module run in float64 on
the CPU. web/bench/block-probe/ runs the graph in the browser and compares.

  .venv313\\Scripts\\python.exe tools/webgpu/probe_block.py --case en-short --step 0 --block 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dit import MAX_N, export, load_base, rope_from_tables, rope_tables  # noqa: E402
from parity import GoldenCase  # noqa: E402

from f5_tts.model.modules import DiTBlock, _apply_rope  # noqa: E402


OUT = Path(__file__).resolve().parents[2] / "build/webgpu/dit/probe"
NAMES = [
    "emb",
    "norm1",
    "mod1",
    "q",
    "k",
    "v",
    "q_rot",
    "k_rot",
    "sdpa",
    "attn",
    "x1",
    "norm2",
    "mod2",
    "h1",
    "gelu",
    "ff",
    "x2",
]


class BlockProbe(nn.Module):
    """DiTBlock.forward (modules.py:774) written out step by step; x2 is the block's output."""

    rope: Tensor

    def __init__(self, block: DiTBlock, rope: Tensor, tables: bool) -> None:
        """`rope` is the rotary angle table, or with `tables` export_dit.rope_tables (cos and sin)."""
        super().__init__()
        self.block = block
        self.tables = tables
        self.register_buffer("rope", rope, persistent=False)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        b = self.block
        emb = b.attn_norm.linear(b.attn_norm.silu(t))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        norm1 = b.attn_norm.norm(x)
        mod1 = norm1 * (1 + scale_msa[:, None]) + shift_msa[:, None]
        a = b.attn
        q, k, v = a.to_q(mod1), a.to_k(mod1), a.to_v(mod1)

        def heads(z: Tensor) -> Tensor:
            return z.view(z.shape[0], -1, a.heads, z.shape[-1] // a.heads).transpose(1, 2)

        rope = self.rope[:, : x.shape[1]]
        apply = rope_from_tables if self.tables else _apply_rope
        q_rot, k_rot = apply(heads(q), rope, 1.0), apply(heads(k), rope, 1.0)
        sdpa = F.scaled_dot_product_attention(q_rot, k_rot, heads(v)).transpose(1, 2).reshape(q.shape)
        attn = a.to_out[1](a.to_out[0](sdpa))  # Linear, Dropout (modules.py, Attention.to_out)
        x1 = x + gate_msa.unsqueeze(1) * attn
        norm2 = b.ff_norm(x1)
        mod2 = norm2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        project_in, _dropout, project_out = b.ff.ff  # Sequential(Sequential(Linear, GELU), Dropout, Linear)
        linear_in, _gelu = cast(nn.Sequential, project_in)
        h1 = linear_in(mod2)
        gelu = F.gelu(h1, approximate="tanh")
        ff = project_out(gelu)
        x2 = x1 + gate_mlp.unsqueeze(1) * ff
        return emb, norm1, mod1, q, k, v, q_rot, k_rot, sdpa, attn, x1, norm2, mod2, h1, gelu, ff, x2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="en-short")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--upstream-rope", action="store_true", help="export cos/sin computed in the graph, as before")
    args = p.parse_args()

    g = GoldenCase(args.case)
    s = f"s{args.step:02d}"
    prev = f"c0.dit_internal.{s}.block{args.block - 1:02d}" if args.block else f"c0.dit_internal.{s}.input_embed"
    if args.block == 0:
        x = np.concatenate([g.tensor(prev), g.tensor(prev + "_uncond")])
    else:
        x = g.tensor(prev)
    t1 = g.tensor(f"c0.dit_internal.{s}.time_embed")
    t = np.concatenate([t1, t1])
    dit = load_base()
    block = dit.transformer_blocks[args.block]
    assert isinstance(block, DiTBlock)
    freqs, _scale = dit.rotary_embed.forward_from_seq_len(MAX_N)
    upstream = BlockProbe(block, freqs.detach().clone(), tables=False).eval()
    probe = upstream if args.upstream_rope else BlockProbe(block, rope_tables(dit), tables=True).eval()

    xt, tt = torch.from_numpy(x), torch.from_numpy(t)
    with torch.no_grad():
        ref64 = copy.deepcopy(upstream).double()(xt.double(), tt.double())
        block_out = block(xt, tt, mask=None, rope=(upstream.rope[:, : x.shape[1]], 1.0))
        assert torch.equal(upstream(xt, tt)[-1], block_out), "BlockProbe does not reproduce DiTBlock.forward"
        got = probe(xt, tt)  # what the exported graph computes, in float32 on the CPU
    qr, kr = ref64[NAMES.index("q_rot")], ref64[NAMES.index("k_rot")]
    logits = (qr @ kr.transpose(-1, -2)) / qr.shape[-1] ** 0.5
    top2 = logits.topk(2).values
    print(
        f"attention logits: |max| {logits.abs().max():.1f}, median gap top 1 to 2 {(top2[..., 0] - top2[..., 1]).median():.2f}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    n = torch.export.Dim("n", min=16, max=MAX_N)
    export(probe, (xt, tt), ["x", "t"], NAMES, {"x": {1: n}, "t": {}}, OUT / "block.onnx", 20, None)
    np.save(OUT / "in.x.npy", x)
    np.save(OUT / "in.t.npy", t)
    summary = {"case": args.case, "step": args.step, "block": args.block, "outputs": NAMES, "cpu_fp32_rel_l2": {}}
    for name, r64, r32 in zip(NAMES, ref64, got):
        ref = r64.numpy()
        np.save(OUT / f"ref.{name}.npy", np.ascontiguousarray(ref.astype(np.float32)))
        err = float(np.linalg.norm(r32.numpy().astype(np.float64) - ref) / np.linalg.norm(ref))
        summary["cpu_fp32_rel_l2"][name] = err
        print(f"{name:6s} shape {list(ref.shape)}  |ref|max {np.abs(ref).max():.3g}  torch CPU fp32 relL2 {err:.2e}")
    (OUT / "probe.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
