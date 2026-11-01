"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    MMDiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


# text embedding


class TextEmbedding(nn.Module):
    def __init__(self, out_dim: int, text_num_embeds: int, mask_padding: bool = True) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: Int[Tensor, "b nt"], drop_text: bool = False) -> Float[Tensor, "b nt d"]:
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b nt -> b nt d

        # sinus pos emb
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)
        batch_text_len = text.shape[1]
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]

        text = text + text_pos_embed

        if self.mask_padding:
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return text


# noised input & masked cond audio embedding


class AudioEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(
        self, x: Float[Tensor, "b n d"], cond: Float[Tensor, "b n d"], drop_audio_cond: bool = False
    ) -> Float[Tensor, "b n d"]:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using MM-DiT blocks


class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: float = 4,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_mask_padding: bool = True,
        qk_norm: str | None = None,
        checkpoint_activations: bool = False,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(dim, text_num_embeds, mask_padding=text_mask_padding)
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                    qk_norm=qk_norm,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    # `_cache_local` is lazily initialized on first inference-time cache write so that
    # training models (which never touch the cache) stay deepcopy-friendly for EMA.
    def _get_cache_local(self) -> threading.local:
        cache = self.__dict__.get("_cache_local")
        if cache is None:
            cache = threading.local()
            self.__dict__["_cache_local"] = cache
        return cache

    @property
    def text_cond(self) -> Float[Tensor, "b n d"] | None:
        cache = self.__dict__.get("_cache_local")
        return getattr(cache, "text_cond", None) if cache is not None else None

    @text_cond.setter
    def text_cond(self, value: Float[Tensor, "b n d"] | None) -> None:
        self._get_cache_local().text_cond = value

    @property
    def text_uncond(self) -> Float[Tensor, "b n d"] | None:
        cache = self.__dict__.get("_cache_local")
        return getattr(cache, "text_uncond", None) if cache is not None else None

    @text_uncond.setter
    def text_uncond(self, value: Float[Tensor, "b n d"] | None) -> None:
        self._get_cache_local().text_uncond = value

    def initialize_weights(self) -> None:
        # Zero-out AdaLN layers in MMDiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module: nn.Module) -> Callable[..., Any]:  # Any: forwards a block's own inputs and outputs
        def ckpt_forward(*inputs: Any) -> Any:
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x: Float[Tensor, "b n d"],
        cond: Float[Tensor, "b n d"],
        text: Int[Tensor, "b nt"],
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b nt d"]]:
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        return x, c

    def clear_cache(self) -> None:
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: Float[Tensor, "b n d"],  # nosied input audio
        cond: Float[Tensor, "b n d"],  # masked cond audio
        text: Int[Tensor, "b nt"],  # text
        time: Float[Tensor, "b"] | Float[Tensor, ""],  # time step
        mask: Bool[Tensor, "b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ) -> Float[Tensor, "b n d"]:
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        c_mask = (text + 1) != 0  # True = valid, False = padding (-1 tokens)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, c_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            x, c = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache
            )

        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                c, x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, t, mask, rope_audio, rope_text, c_mask, use_reentrant=False
                )
            else:
                c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
