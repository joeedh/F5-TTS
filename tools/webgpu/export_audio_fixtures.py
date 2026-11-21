"""Fixtures for the TypeScript audio code (plan 02).

  .venv313\\Scripts\\python.exe tools/webgpu/export_audio_fixtures.py

Writes testdata/webgpu/audio-fixtures/ (committed):
  mel_fbank.npy  the mel filterbank torchaudio uses for the Vocos mel, (513, 100) float32, taken from
                 the same MelSpectrogram that model/modules.py get_vocos_mel_spectrogram builds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "testdata" / "webgpu" / "audio-fixtures"


def mel_fbank() -> np.ndarray:
    # the arguments of get_vocos_mel_spectrogram (model/modules.py); the filterbank is its mel_scale.fb
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=100,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    )
    mel_scale = mel.mel_scale
    assert isinstance(mel_scale, torchaudio.transforms.MelScale)
    fb_tensor = mel_scale.fb
    assert isinstance(fb_tensor, torch.Tensor)
    fb = fb_tensor.numpy()
    assert fb.shape == (513, 100) and fb.dtype == np.float32, (fb.shape, fb.dtype)
    return fb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mel_fbank.npy", mel_fbank())
    print(f"wrote {out / 'mel_fbank.npy'}")


if __name__ == "__main__":
    main()
