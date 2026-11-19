"""Small DiT and Vocos with seeded random weights, for the `tiny` golden case (plan 00).

The weights are committed (testdata/webgpu/tiny/weights/) because random init can change
between torch versions. `make_golden.py --case tiny` loads them, or creates them with
--init-tiny when they are missing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone

from f5_tts.model.backbones.dit import DiT


DIT_ARCH = dict(
    dim=128,
    depth=2,
    heads=2,
    dim_head=64,
    ff_mult=2,
    text_dim=64,
    conv_layers=1,
    text_mask_padding=True,
    attn_mask_enabled=False,
)
INIT_SEED = 1234
INIT_STD = 0.02


def build_tiny_dit(vocab_size: int) -> DiT:
    return DiT(**DIT_ARCH, text_num_embeds=vocab_size, mel_dim=100)  # pyright: ignore[reportArgumentType]  # dict of mixed values


def build_tiny_vocos() -> Vocos:
    # the real ISTFT head and mel settings (n_fft 1024, hop 256), with a small backbone
    return Vocos(
        feature_extractor=MelSpectrogramFeatures(
            sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"
        ),
        backbone=VocosBackbone(input_channels=100, dim=64, intermediate_dim=192, num_layers=2),
        head=ISTFTHead(dim=64, n_fft=1024, hop_length=256, padding="center"),
    )


def reinit(module: nn.Module, seed: int) -> None:
    """Every parameter from N(0, 0.02). DiT.initialize_weights zeroes AdaLN and proj_out, which would
    make every tiny DiT output exactly zero."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * INIT_STD)


def save_weights(dit: DiT, vocos: Vocos, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(folder / "dit.safetensors"))
    save_file({k: v.contiguous() for k, v in vocos.state_dict().items()}, str(folder / "vocos.safetensors"))


def load_tiny(vocab_size: int, folder: Path, init: bool) -> tuple[DiT, Vocos]:
    """Builds the tiny models; loads committed weights, or makes and saves them when `init` is set."""
    dit, vocos = build_tiny_dit(vocab_size), build_tiny_vocos()
    if init:
        reinit(dit, INIT_SEED)
        reinit(vocos, INIT_SEED + 1)
        save_weights(dit, vocos, folder)
    else:
        if not (folder / "dit.safetensors").exists():
            raise FileNotFoundError(f"{folder}: no tiny weights; run with --init-tiny to create them")
        dit.load_state_dict(load_file(str(folder / "dit.safetensors")), strict=True)
        vocos.load_state_dict(load_file(str(folder / "vocos.safetensors")), strict=True)
    return dit.float().eval(), vocos.float().eval()
