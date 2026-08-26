"""Whisper-based lyric alignment: which words are actually sung during each clip window."""
import math

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_windows(audio_path, n: int, clip_seconds: float, on_progress=None) -> list:
    """Return n strings: the words sung during each consecutive clip window (may be empty)."""
    if on_progress:
        on_progress("transcribing the song for lyric timing")
    segs, _info = _get_model().transcribe(str(audio_path), word_timestamps=True)
    words = [(w.start, w.end, w.word) for s in segs for w in (s.words or [])]
    windows = []
    for i in range(n):
        lo, hi = i * clip_seconds, (i + 1) * clip_seconds
        text = "".join(w for a, b, w in words if lo <= (a + b) / 2 < hi).strip()
        windows.append(text)
    return windows
