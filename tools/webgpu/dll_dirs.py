"""Registers FFmpeg's DLL folders for torchcodec, which torchaudio.load goes through, on Windows.

Set FFMPEG_DLL_DIRS to the folders, separated by os.pathsep (;). PATH is not enough: since Python 3.8, Windows
looks up an extension module's DLLs only in registered folders (docs/debugging.md, 2026-11-02).
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def register_ffmpeg() -> None:
    if sys.platform != "win32":
        return
    for folder in filter(None, os.environ.get("FFMPEG_DLL_DIRS", "").split(os.pathsep)):
        os.add_dll_directory(folder)


def register_cuda12() -> None:
    """faster-whisper's CTranslate2 needs CUDA 12's cuBLAS and cuDNN 9, from the nvidia-cublas-cu12 and
    nvidia-cudnn-cu12 wheels; torch's own CUDA 13 libraries do not serve. Registers their bin folders and puts
    them on PATH, since CTranslate2 loads cuBLAS lazily."""
    if sys.platform != "win32":
        return
    purelib = Path(sysconfig.get_paths()["purelib"])
    for folder in sorted((purelib / "nvidia").glob("*/bin")):
        os.add_dll_directory(str(folder))
        os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
