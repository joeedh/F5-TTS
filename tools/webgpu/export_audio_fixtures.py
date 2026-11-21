"""Fixtures for the TypeScript audio code (plan 02).

  .venv313\\Scripts\\python.exe tools/webgpu/export_audio_fixtures.py

Writes testdata/webgpu/audio-fixtures/ (committed):
  mel_fbank.npy        the mel filterbank torchaudio uses for the Vocos mel, (513, 100) float32, taken from
                       the same MelSpectrogram that model/modules.py get_vocos_mel_spectrogram builds
  prep/<name>.in.wav   a synthetic 16-bit reference, for a path of the reference preparation that the
                       golden cases miss (plan 02, step 2)
  prep/<name>.out.wav  what upstream's preprocess_ref_audio_text wrote for it
  prep/<name>.ref.npy  the 24 kHz reference after infer_batch_process's downmix, RMS raise and resample,
                       (1, samples) float32; only for the fixtures that resample
  prep/manifest.json   per fixture: rates, lengths, rms and raised
  istft/head_out.npy   (6, 1026) float32 head output: log magnitudes (some clipped at 100) and phases,
                       nonzero at bins 0 and 512 too (plan 02, step 4)
  istft/irfft.npy      (6, 1024) torch.fft.irfft of each frame's complex spectrum, in float64
  istft/wav.npy        (1280,) torch.istft of the whole spectrum as the Vocos head runs it, in float64
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import f5_tts.infer.utils_infer as ui


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "testdata" / "webgpu" / "audio-fixtures"
SR = 24000

# infer_batch_process's reference handling, run here on each fixture; asserted against the source
COPIED_LINES = [
    "audio = torch.mean(audio, dim=0, keepdim=True)",
    "rms = torch.sqrt(torch.mean(torch.square(audio)))",
    "audio = audio * target_rms / rms",
    "resampler = torchaudio.transforms.Resample(sr, target_sample_rate)",
]


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


@dataclass
class Fixture:
    name: str
    rate: int
    note: str
    make: Callable[[np.random.Generator, int], np.ndarray]  # (frames, channels) float, scaled to int16 range
    resamples: bool = False


def burst(rng: np.random.Generator, rate: int, seconds: float, amp: float = 6000.0) -> np.ndarray:
    """Speech-like: a few tones plus noise, with a slow envelope. Loud enough to be non-silent throughout."""
    t = np.arange(int(rate * seconds)) / rate
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 420 * t + 1.0) + 0.3 * rng.standard_normal(t.size)
    return amp * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * tone / 1.6


def zeros(rate: int, seconds: float) -> np.ndarray:
    return np.zeros(int(rate * seconds))


def square(rate: int, seconds: float, amp: int) -> np.ndarray:
    """+-amp alternating: its integer rms is exactly amp."""
    return amp * (1 - 2 * (np.arange(int(rate * seconds)) % 2)).astype(np.float64)


def mono(x: np.ndarray) -> np.ndarray:
    return x[:, None]


FIXTURES = [
    Fixture(
        "quiet-44k-stereo",
        44100,
        "44.1 kHz stereo at rms about 0.02: downmix, RMS raise, resample, and keep_silence overlap",
        lambda rng, r: (lambda x: np.stack([x, 0.8 * x], axis=1))(
            0.15 * np.concatenate([burst(rng, r, 0.3), zeros(r, 1.05), burst(rng, r, 0.3)])
        ),
        resamples=True,
    ),
    Fixture(
        "silence-only",
        SR,
        "1.5 s of silence: split_on_silence returns nothing, so the output is pydub's 50 ms at 11025 Hz. Upstream"
        " then divides by an rms of 0, so there is no reference to compare",
        lambda rng, r: mono(zeros(r, 1.5)),
    ),
    Fixture(
        "hard-cut",
        11025,
        "12.4 s with no silence at all: both splits fail and the audio is cut at 12 s",
        lambda rng, r: mono(burst(rng, r, 12.4, 3000.0)),
    ),
    Fixture(
        "join-6-12",
        11025,
        "three 4 s bursts: the join stops once the result is over 6 s and the next part takes it over 12 s",
        lambda rng, r: mono(
            np.concatenate([burst(rng, r, 4.0), zeros(r, 1.2), burst(rng, r, 4.0), zeros(r, 1.2), burst(rng, r, 4.0)])
        ),
    ),
    Fixture(
        "mid-silence",
        SR,
        "a 2.5 s silence in the middle, cut down to the kept 1 s on each side",
        lambda rng, r: mono(np.concatenate([burst(rng, r, 0.4), zeros(r, 2.5), burst(rng, r, 0.4)])),
    ),
    Fixture(
        "thresholds",
        SR,
        "rms exactly at the thresholds: 103 is silent at -50 dBFS and 104 is not; 260 is trimmed at -42 dBFS"
        " and 261 is not",
        lambda rng, r: mono(
            np.concatenate(
                [
                    square(r, 0.2, 260),
                    burst(rng, r, 0.3),
                    square(r, 1.5, 103),
                    burst(rng, r, 0.3),
                    square(r, 1.5, 104),
                    burst(rng, r, 0.3),
                    square(r, 0.2, 261),
                ]
            )
        ),
    ),
    Fixture(
        "round-half-down",
        SR,
        "18012 frames is 750.5 ms, which Python rounds to 750: the last 12 frames are dropped",
        lambda rng, r: mono(burst(rng, r, 18012 / r)),
    ),
    Fixture(
        "round-half-up",
        SR,
        "18036 frames is 751.5 ms, rounded to 752: the slice overshoots and pydub pads 12 zero frames",
        lambda rng, r: mono(burst(rng, r, 18036 / r)),
    ),
    Fixture(
        "low-rate-8k",
        8000,
        "8 kHz: pydub's join upsamples it to 11025 Hz with audioop.ratecv",
        lambda rng, r: mono(burst(rng, r, 0.6)),
        resamples=True,
    ),
]


def reference(path: Path) -> tuple[np.ndarray, float, bool]:
    """infer_batch_process's handling of the prepared reference (COPIED_LINES)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    raised = bool(rms < ui.target_rms)
    if raised:
        audio = audio * ui.target_rms / rms
    if sr != SR:
        audio = torchaudio.transforms.Resample(sr, SR)(audio)
    return audio.numpy(), float(rms), raised


def prep_fixtures(out: Path) -> None:
    src = inspect.getsource(ui.infer_batch_process)
    for line in COPIED_LINES:
        assert line in src, f"infer_batch_process changed; update export_audio_fixtures.py: {line}"
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for i, fx in enumerate(FIXTURES):
        x = fx.make(np.random.default_rng(1000 + i), fx.rate)
        pcm = np.clip(np.round(x), -32768, 32767).astype(np.int16)
        in_path = out / f"{fx.name}.in.wav"
        sf.write(str(in_path), pcm, fx.rate, subtype="PCM_16")
        prep_path, _ = ui.preprocess_ref_audio_text(str(in_path), "x.", show_info=lambda _s: None)
        shutil.copyfile(prep_path, out / f"{fx.name}.out.wav")
        info = sf.info(prep_path)
        audio, rms, raised = reference(Path(prep_path))
        if fx.resamples:
            np.save(out / f"{fx.name}.ref.npy", audio)
        manifest[fx.name] = {
            "note": fx.note,
            "input": {"sample_rate": fx.rate, "channels": pcm.shape[1], "frames": pcm.shape[0]},
            "prep": {"sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames},
            "audio": {"rms": rms, "raised": raised, "ref": fx.resamples},
        }
        print(f"{fx.name}: {pcm.shape[0]} frames at {fx.rate} -> {info.frames} at {info.samplerate}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def istft_fixture(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frames = 6
    log_mag = rng.normal(0.0, 1.5, (frames, 513))
    log_mag[:, 100] = 5.0  # exp(5) = 148, clipped to 100 as ISTFTHead does
    phase = rng.uniform(-np.pi, np.pi, (frames, 513))  # bins 0 and 512 included: irfft ignores their imaginary part
    head_out = np.concatenate([log_mag, phase], axis=1).astype(np.float32)
    h = torch.from_numpy(head_out).double()
    mag = torch.clip(torch.exp(h[:, :513]), max=1e2)
    spec = mag * (torch.cos(h[:, 513:]) + 1j * torch.sin(h[:, 513:]))  # (frames, 513), ISTFTHead's formula
    irfft = torch.fft.irfft(spec, n=1024, dim=1)
    window = torch.hann_window(1024, dtype=torch.float64)
    wav = torch.istft(spec.T, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True)
    assert wav.shape == ((frames - 1) * 256,), wav.shape
    np.save(out / "head_out.npy", head_out)
    np.save(out / "irfft.npy", irfft.numpy().astype(np.float32))
    np.save(out / "wav.npy", wav.numpy().astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mel_fbank.npy", mel_fbank())
    print(f"wrote {out / 'mel_fbank.npy'}")
    prep_fixtures(out / "prep")
    istft_fixture(out / "istft")


if __name__ == "__main__":
    main()
