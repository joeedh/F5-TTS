"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, cast

import torch
import torch.nn.functional as F
import torchaudio
from jaxtyping import Bool, Float, Int
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor, nn
from x_transformers.x_transformers import apply_rotary_pos_emb

from f5_tts.model.utils import is_package_available


# (freqs, xpos_scale) as returned by x_transformers' RotaryEmbedding.forward_from_seq_len
Rope = tuple[Tensor, Tensor | float | None]

# x_transformers types the xpos scale as int; a float or a tensor is accepted at runtime
_apply_rope = cast(Callable[[Tensor, Tensor, float | Tensor], Tensor], apply_rotary_pos_emb)


# raw wav to mel spec


mel_basis_cache: dict[str, Tensor] = {}
hann_window_cache: dict[str, Tensor] = {}
vocos_mel_stft_cache: dict[str, torchaudio.transforms.MelSpectrogram] = {}


def get_bigvgan_mel_spectrogram(
    waveform: Float[Tensor, "b nw"],
    n_fft: int = 1024,
    n_mel_channels: int = 100,
    target_sample_rate: int = 24000,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: int = 0,
    fmax: int | None = None,
    center: bool = False,
) -> Float[Tensor, "b d n"]:  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform: Float[Tensor, "b nw"] | Float[Tensor, "b 1 nw"],
    n_fft: int = 1024,
    n_mel_channels: int = 100,
    target_sample_rate: int = 24000,
    hop_length: int = 256,
    win_length: int = 1024,
) -> Float[Tensor, "b d n"]:
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{device}"
    if key not in vocos_mel_stft_cache:
        vocos_mel_stft_cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).to(device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = vocos_mel_stft_cache[key](waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
        target_sample_rate: int = 24_000,
        mel_spec_type: str = "vocos",
    ) -> None:
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav: Float[Tensor, "b nw"]) -> Float[Tensor, "b d n"]:
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[Tensor, "b"], scale: float = 1000) -> Float[Tensor, "b d"]:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: Float[Tensor, "b n d"], mask: Bool[Tensor, "b n"] | None = None) -> Float[Tensor, "b n d"]:
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# rotary positional embedding related


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0
) -> Float[Tensor, "n d"]:
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(
    start: Int[Tensor, "b"], length: int, max_pos: int, scale: float | Tensor = 1.0
) -> Int[Tensor, "b n"]:
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x: Tensor) -> Tensor:
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: Float[Tensor, "b n d"], emb: Float[Tensor, "b d"] | None = None
    ) -> tuple[
        Float[Tensor, "b n d"],
        Float[Tensor, "b d"],
        Float[Tensor, "b d"],
        Float[Tensor, "b d"],
        Float[Tensor, "b d"],
    ]:
        assert emb is not None
        emb_out = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb_out, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Float[Tensor, "b n d"], emb: Float[Tensor, "b d"]) -> Float[Tensor, "b n d"]:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, dim_out: int | None = None, mult: float = 4, dropout: float = 0.0, approximate: str = "none"
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: Float[Tensor, "b n d"],  # noised input x
        c: Float[Tensor, "b n d"] | None = None,  # context c
        mask: Bool[Tensor, "b n"] | None = None,
        rope: Rope | None = None,  # rotary position embedding for x
        c_rope: Rope | None = None,  # rotary position embedding for c
        c_mask: Bool[Tensor, "b nt"] | None = None,  # text mask
    ) -> Tensor | tuple[Tensor, Tensor]:  # a tuple when a context c is given (joint attention)
        if c is not None:
            return cast(JointAttnProcessor, self.processor)(
                self, x, c=c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask
            )
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor

if is_package_available("flash_attn"):
    from flash_attn import (  # pyright: ignore[reportMissingImports]  # optional dependency
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_attn.bert_padding import (  # pyright: ignore[reportMissingImports]  # optional dependency
        pad_input,
        unpad_input,
    )


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ) -> None:
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."
        if attn_backend == "torch" and attn_mask_enabled:
            warnings.warn(
                "attn_mask_enabled=True with attn_backend='torch' can consume large GPU memory. "
                "Please switch attn_backend to 'flash_attn'.",
                UserWarning,
            )

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: Float[Tensor, "b n d"],  # noised input x
        mask: Bool[Tensor, "b n"] | None = None,
        rope: Rope | None = None,  # rotary position embedding
    ) -> Tensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = _apply_rope(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = _apply_rope(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = _apply_rope(query, freqs, q_xpos_scale)
                key = _apply_rope(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(
        self,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ) -> None:
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."
        if attn_backend == "torch" and attn_mask_enabled:
            warnings.warn(
                "attn_mask_enabled=True with attn_backend='torch' can consume large GPU memory. "
                "Please switch attn_backend to 'flash_attn'.",
                UserWarning,
            )

        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: Float[Tensor, "b n d"],  # noised input x
        c: Float[Tensor, "b nt d"] | None = None,  # context c, here text
        mask: Bool[Tensor, "b n"] | None = None,
        rope: Rope | None = None,  # rotary position embedding for x
        c_rope: Rope | None = None,  # rotary position embedding for c
        c_mask: Bool[Tensor, "b nt"] | None = None,  # text mask
    ) -> tuple[Tensor, Tensor]:
        residual = x
        audio_mask = mask

        assert c is not None
        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = _apply_rope(query, freqs, q_xpos_scale)
            key = _apply_rope(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = _apply_rope(c_query, freqs, q_xpos_scale)
            c_key = _apply_rope(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # build combined mask for joint attention: audio mask + text mask
        if self.attn_mask_enabled and mask is not None:
            if c_mask is not None:
                mask = torch.cat([mask, c_mask], dim=1)
            else:
                mask = F.pad(mask, (0, c.shape[1]), value=True)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                total_seq_len = query.shape[1]
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, total_seq_len)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c_out = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c_out = attn.to_out_c(c_out)

        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c_out = c_out.masked_fill(~c_mask.unsqueeze(-1), 0.0)

        return x, c_out


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        t: Float[Tensor, "b d"],
        mask: Bool[Tensor, "b n"] | None = None,
        rope: Rope | None = None,
    ) -> Float[Tensor, "b n d"]:  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
        context_dim: int | None = None,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = False,
    ) -> None:
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        c: Float[Tensor, "b nt d"],
        t: Float[Tensor, "b d"],
        mask: Bool[Tensor, "b n"] | None = None,
        rope: Rope | None = None,
        c_rope: Rope | None = None,
        c_mask: Bool[Tensor, "b nt"] | None = None,
    ) -> tuple[
        Float[Tensor, "b nt d"] | None, Float[Tensor, "b n d"]
    ]:  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)

        # process attention output for context c
        if self.context_pre_only:
            c_out = None
        else:  # if not last layer
            assert self.ff_norm_c is not None and self.ff_c is not None
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c_out = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c_out, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: Float[Tensor, "b"]) -> Float[Tensor, "b d"]:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time
