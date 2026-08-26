"""Locate modern ffmpeg/ffprobe binaries.

Prefers the binaries bundled by the `static-ffmpeg` package (ffmpeg 8.x, includes ffprobe), then
falls back to whatever is on PATH. Override with env vars FFMPEG_BIN / FFPROBE_BIN.
"""
import os
import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def _resolve() -> tuple[str, str]:
    ffmpeg, ffprobe = os.environ.get("FFMPEG_BIN"), os.environ.get("FFPROBE_BIN")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    try:
        from static_ffmpeg import run
        ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
        return ffmpeg, ffprobe
    except Exception:  # noqa: BLE001
        pass
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    return ffmpeg, ffprobe


def ffmpeg() -> str:
    return _resolve()[0]


def ffprobe() -> str:
    return _resolve()[1]
