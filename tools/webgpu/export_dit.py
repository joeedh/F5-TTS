"""Plan 01: export the F5-TTS DiT as two ONNX graphs for ORT Web.

  text.onnx  text (1, n) int32, padded with -1  ->  text_cond, text_uncond (1, n, text_dim)
  step.onnx  x, step_cond (1, n, 100), text_cond, text_uncond, time, cfg_strength, dt (1,)
             ->  pred (2, n, 100), v (1, n, 100), x_next (1, n, 100)

n is dynamic (16 to 8192; freqs_cis and the rotary table are precomputed to 8192). The wrappers
call the upstream submodules in DiT.forward's order and change none of them.

--attention picks how SDPA is exported (plan 01 D3):
  unfused  the exporter's default (MatMul, Softmax, MatMul)
  mha      com.microsoft.MultiHeadAttention
  onnx23   ONNX Attention (opset 23)
--debug also outputs time_embed, input_embed(_uncond), every block and norm_out, for bisecting
against the golden dit_internal tensors.

Run in .venv313:
  .venv313\\Scripts\\python.exe tools/webgpu/export_dit.py --model tiny
  .venv313\\Scripts\\python.exe tools/webgpu/export_dit.py --model base --attention mha
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import onnx
import onnxscript.onnx_opset
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from onnxscript.values import Op, Opset
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.export import Dim
from x_transformers.x_transformers import rotate_half

import f5_tts.model.modules as f5_modules
from f5_tts.model.backbones.dit import DiT, TextEmbedding
from f5_tts.model.modules import DiTBlock
from f5_tts.model.utils import get_tokenizer


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "src/f5_tts/configs/F5TTS_v1_Base.yaml"
VOCAB = REPO / "src/f5_tts/infer/examples/vocab.txt"
TINY_WEIGHTS = REPO / "testdata/webgpu/tiny/weights"
MIN_N, MAX_N = 16, 8192
EXPORT_N = 256  # example length for tracing; any n in range works afterwards

# onnxscript's opset signatures take its own tensor types; the exporter traces with symbolic tensors
op20: Any = onnxscript.onnx_opset.opset20  # Any: see above
op23: Any = onnxscript.onnx_opset.opset23  # Any: see above
MSFT = Opset("com.microsoft", 1)


def msft_mha_op() -> Op:
    """onnx has no schema for ORT's contrib ops, so onnxscript cannot build the node without one. This
    declares the three-input, one-output form of com.microsoft.MultiHeadAttention that the export uses."""
    S = onnx.defs.OpSchema
    schema = S(
        "MultiHeadAttention",
        "com.microsoft",
        1,
        inputs=[S.FormalParameter(n, "T") for n in ("query", "key", "value")],
        outputs=[S.FormalParameter("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
        attributes=[S.Attribute("num_heads", S.AttrType.INT, "", required=True)],
    )
    return Op(MSFT, "MultiHeadAttention", schema)


class TextGraph(nn.Module):
    """TextEmbedding.forward for both CFG halves at once. It calls the submodules directly because
    forward's int(seq_len) would fix n at export (plan 01 step 1)."""

    def __init__(self, te: TextEmbedding) -> None:
        super().__init__()
        assert te.extra_modeling and te.mask_padding and not te.average_upsampling
        self.te = te

    def forward(self, text: Tensor) -> tuple[Tensor, Tensor]:
        ids = text + 1  # -1 padding becomes the 0 filler, as in TextEmbedding.forward
        filler = (ids == 0).unsqueeze(-1)  # from the real ids, also for the uncond half (dit.py:118-122)
        freqs = self.te.freqs_cis[: ids.shape[1], :]
        cond = self.te.text_embed(ids) + freqs
        uncond = self.te.text_embed(torch.zeros_like(ids)) + freqs  # drop_text
        h = torch.cat((cond, uncond), dim=0).masked_fill(filler, 0.0)
        for block in self.te.text_blocks:
            h = block(h).masked_fill(filler, 0.0)
        return h[:1], h[1:]


def rope_tables(dit: DiT) -> Tensor:
    """(1, MAX_N, rot_dim, 2): cos and sin of the rotary angles, computed in float64 here. The angles reach
    thousands of radians, and the NVIDIA GPU's shader cos/sin lose precision there (plan 01 D4, step 3)."""
    freqs, scale = dit.rotary_embed.forward_from_seq_len(MAX_N)
    assert scale == 1.0
    f = freqs.detach().double()
    return torch.stack((f.cos(), f.sin()), dim=-1).float()


def rope_from_tables(t: Tensor, tables: Tensor, scale: float | Tensor = 1.0) -> Tensor:
    """x_transformers.apply_rotary_pos_emb for full-width rotary at scale 1, with cos and sin read from
    rope_tables (sliced to n) instead of computed from the angles."""
    assert isinstance(scale, float) and scale == 1.0 and t.shape[-1] == tables.shape[-2]
    cos, sin = tables[..., 0].unsqueeze(1), tables[..., 1].unsqueeze(1)  # (1, 1, n, rot_dim) for the heads
    return t * cos + rotate_half(t) * sin


@contextmanager
def precomputed_rope() -> Iterator[None]:
    """While exporting, upstream attention calls rope_from_tables instead of apply_rotary_pos_emb; the
    `rope` it is handed must then be rope_tables sliced to n. Upstream code is not changed on disk."""
    saved = f5_modules._apply_rope
    f5_modules._apply_rope = rope_from_tables
    try:
        yield
    finally:
        f5_modules._apply_rope = saved


class StepGraph(nn.Module):
    """One CFG DiT call plus the Euler update CFM.sample and torchdiffeq apply around it. Export and run
    it inside precomputed_rope()."""

    rope_tables: Tensor

    def __init__(self, dit: DiT, debug: bool) -> None:
        super().__init__()
        assert dit.long_skip_connection is None and not dit.checkpoint_activations
        self.dit = dit
        self.debug = debug
        # rotary angles depend only on the position, so a sliced table equals forward_from_seq_len(n) (D4)
        self.register_buffer("rope_tables", rope_tables(dit), persistent=False)

    def forward(
        self,
        x: Tensor,
        step_cond: Tensor,
        text_cond: Tensor,
        text_uncond: Tensor,
        time: Tensor,
        cfg_strength: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, ...]:
        d = self.dit
        t = d.time_embed(time)
        # both halves through InputEmbedding at once; drop_audio_cond zeroes cond for the uncond half
        h = d.input_embed(
            torch.cat((x, x), dim=0),
            torch.cat((step_cond, torch.zeros_like(step_cond)), dim=0),
            torch.cat((text_cond, text_uncond), dim=0),
        )
        debug = [t, h[:1], h[1:]]
        t2 = torch.cat((t, t), dim=0)
        rope = (self.rope_tables[:, : x.shape[1]], 1.0)
        for block in d.transformer_blocks:
            h = block(h, t2, mask=None, rope=rope)
            debug.append(h)
        h = d.norm_out(h, t2)
        debug.append(h)
        pred = d.proj_out(h)
        cond, uncond = pred[:1], pred[1:]
        v = cond + (cond - uncond) * cfg_strength  # cfm.py:203
        x_next = x + dt * v  # torchdiffeq Euler: y0 + dt * f0
        outs = (pred, v, x_next)
        return outs + tuple(debug) if self.debug else outs


def step_output_names(depth: int, debug: bool) -> list[str]:
    names = ["pred", "v", "x_next"]
    if debug:
        names += ["time_embed", "input_embed", "input_embed_uncond"]
        names += [f"block{i:02d}" for i in range(depth)] + ["norm_out"]
    return names


# --- attention translations (D3) ---


def sdpa_mha(heads: int) -> Callable[..., Any]:  # Any: onnxscript symbolic tensors
    def sdpa(
        query: Any,  # Any: onnxscript symbolic tensor, (b, h, n, dh)
        key: Any,  # Any: as above
        value: Any,  # Any: as above
        attn_mask: Any = None,  # Any: as above
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:  # Any: as above
        assert attn_mask is None and not is_causal and scale is None and not enable_gqa
        flat = op20.Constant(value_ints=[0, 0, -1])

        def to_bsd(t: Any) -> Any:  # Any: as above
            return op20.Reshape(op20.Transpose(t, perm=[0, 2, 1, 3]), flat)

        out = msft_mha_op()(to_bsd(query), to_bsd(key), to_bsd(value), num_heads=heads)
        heads_shape = op20.Constant(value_ints=[0, 0, heads, -1])
        return op20.Transpose(op20.Reshape(out, heads_shape), perm=[0, 2, 1, 3])

    return sdpa


def sdpa_onnx23(
    query: Any,  # Any: onnxscript symbolic tensor, (b, h, n, dh)
    key: Any,  # Any: as above
    value: Any,  # Any: as above
    attn_mask: Any = None,  # Any: as above
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:  # Any: as above
    assert attn_mask is None and not is_causal and scale is None and not enable_gqa
    out = op23.Attention(query, key, value)
    return out[0] if isinstance(out, tuple) else out  # Y; the KV-cache and qk outputs are unused


# --- models ---


def load_base() -> DiT:
    cfg = OmegaConf.load(CONFIG)
    arch: dict[str, Any] = OmegaConf.to_container(cfg.model.arch)  # pyright: ignore[reportAssignmentType]  # Any: yaml values
    ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    prefix = "ema_model.transformer."
    sd = {k[len(prefix) :]: v for k, v in load_file(ckpt).items() if k.startswith(prefix)}
    _vocab, vocab_size = get_tokenizer(str(VOCAB), "custom")
    dit = DiT(**arch, text_num_embeds=vocab_size, mel_dim=100)
    dit.load_state_dict(sd, strict=True)
    return dit.float().eval()


def load_tiny_dit() -> DiT:
    from tiny_model import build_tiny_dit  # pyright: ignore[reportImplicitRelativeImport]  # sibling script

    _vocab, vocab_size = get_tokenizer(str(VOCAB), "custom")
    dit = build_tiny_dit(vocab_size)
    dit.load_state_dict(load_file(str(TINY_WEIGHTS / "dit.safetensors")), strict=True)
    return dit.float().eval()


# --- export ---


def export(
    module: nn.Module,
    args: tuple[Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic: dict[str, dict[int, Any]],  # Any: torch.export Dim
    path: Path,
    opset: int,
    table: dict[Any, Callable[..., Any]] | None,  # Any: aten overloads and onnxscript functions
) -> onnx.ModelProto:
    prog = torch.onnx.export(
        module,
        args,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic,
        opset_version=opset,
        dynamo=True,
        optimize=True,
        custom_translation_table=table,
    )
    assert prog is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in (path, path.with_suffix(".onnx.data")):
        old.unlink(missing_ok=True)
    prog.save(str(path), external_data=True)
    model = onnx.load(str(path), load_external_data=False)
    for inp in model.graph.input:  # n must stay symbolic (plan 01 step 1)
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 3 or inp.name == "text":
            assert dims[1].dim_param, f"{path.name}: input {inp.name} has a fixed n ({dims[1].dim_value})"
    return model


def describe(model: onnx.ModelProto) -> dict[str, Any]:  # Any: json summary
    ops: dict[str, int] = {}
    for node in model.graph.node:
        key = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        ops[key] = ops.get(key, 0) + 1
    return {
        "opset": {imp.domain or "ai.onnx": imp.version for imp in model.opset_import},
        "nodes": len(model.graph.node),
        "ops": dict(sorted(ops.items())),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["tiny", "base"], required=True)
    p.add_argument("--attention", choices=["unfused", "mha", "onnx23"], default="unfused")
    p.add_argument("--debug", action="store_true", help="also export step_debug.onnx with internal outputs")
    p.add_argument("--out", type=Path, help="default: testdata/webgpu/tiny/onnx or build/webgpu/dit/base")
    args = p.parse_args()
    out: Path = args.out or (
        REPO / "testdata/webgpu/tiny/onnx" if args.model == "tiny" else REPO / "build/webgpu/dit/base"
    )

    dit = load_tiny_dit() if args.model == "tiny" else load_base()
    block0 = dit.transformer_blocks[0]
    assert isinstance(block0, DiTBlock)
    heads = block0.attn.heads
    text_dim = dit.text_embed.text_embed.embedding_dim
    opset = 23 if args.attention == "onnx23" else 20
    sdpa = torch.ops.aten.scaled_dot_product_attention.default
    table = {"unfused": None, "mha": {sdpa: sdpa_mha(heads)}, "onnx23": {sdpa: sdpa_onnx23}}[args.attention]
    suffix = "" if args.attention == "unfused" else f"_{args.attention}"
    summary: dict[str, Any] = {"model": args.model, "torch": torch.__version__, "onnx": onnx.__version__}  # Any: json

    n_text, n_step = Dim("n", min=MIN_N, max=MAX_N), Dim("n", min=MIN_N, max=MAX_N)
    gen = torch.Generator().manual_seed(0)
    text = torch.randint(0, 100, (1, EXPORT_N), generator=gen, dtype=torch.int32)
    text[:, EXPORT_N // 2 :] = -1
    with torch.inference_mode():
        tc, tu = TextGraph(dit.text_embed)(text)

    if args.attention == "unfused":  # the text graph has no attention; one export serves every variant
        t0 = time.perf_counter()
        m = export(
            TextGraph(dit.text_embed),
            (text,),
            ["text"],
            ["text_cond", "text_uncond"],
            {"text": {1: n_text}},
            out / "text.onnx",
            20,
            None,
        )
        summary["text"] = {**describe(m), "export_s": round(time.perf_counter() - t0, 1)}
        print("text.onnx", json.dumps(summary["text"]["ops"]))

    step_args = (
        torch.randn(1, EXPORT_N, 100, generator=gen),
        torch.randn(1, EXPORT_N, 100, generator=gen),
        tc.clone(),
        tu.clone(),
        torch.tensor([0.5]),
        torch.tensor([2.0]),
        torch.tensor([1.0 / 32]),
    )
    names = ["x", "step_cond", "text_cond", "text_uncond", "time", "cfg_strength", "dt"]
    dynamic: dict[str, dict[int, Any]] = {k: {1: n_step} for k in names[:4]}  # Any: Dim
    dynamic.update({k: {} for k in names[4:]})
    variants = [("step", False)] + ([("step_debug", True)] if args.debug else [])
    for stem, debug in variants:
        t0 = time.perf_counter()
        with precomputed_rope():
            m = export(
                StepGraph(dit, debug),
                step_args,
                names,
                step_output_names(dit.depth, debug),
                dynamic,
                out / f"{stem}{suffix}.onnx",
                opset,
                table,
            )
        trig = [n.op_type for n in m.graph.node if n.op_type in ("Cos", "Sin")]
        assert sorted(trig) == ["Cos", "Sin"], f"only the time embedding should compute cos and sin, got {trig}"
        summary[f"{stem}{suffix}"] = {**describe(m), "export_s": round(time.perf_counter() - t0, 1)}
        print(f"{stem}{suffix}.onnx", json.dumps(summary[f"{stem}{suffix}"]["ops"]))
    summary["heads"], summary["text_dim"], summary["depth"] = heads, text_dim, dit.depth
    path = out / f"export{suffix}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
