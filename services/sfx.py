"""Sound effects: the short whoosh that marks a card appearing.

The repository ships no audio assets and adding a binary raises a licensing question nobody wants,
so the hit is SYNTHESIZED once with ffmpeg and cached - pink noise through a band-pass with a fast
attack and an exponential decay, which is what a whoosh is. Set config.SFX_FILE to use your own.

It is written in the exact format tts.synthesize produces, so services.tts.concat_timed can lay the
hits out on one full-length track: that way however many cards a video has, the mix gains ONE extra
input rather than one per hit.
"""
from pathlib import Path

import config
from services import tts
from services.ffbin import ffmpeg
from services.video import _run

MIN_GAP = 0.5        # two hits closer than this are one hit
MAX_HITS = 24


def asset_path() -> Path:
    return config.OUTPUT_DIR / "_assets" / "whoosh.wav"


def ensure_whoosh(path=None) -> Path | None:
    """The whoosh, synthesized once and cached. None when it cannot be made."""
    if config.SFX_FILE and Path(config.SFX_FILE).exists():
        return Path(config.SFX_FILE)
    out = Path(path or asset_path())
    if out.exists() and out.stat().st_size > 1000:
        return out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
              "-i", "anoisesrc=c=pink:r=44100:d=0.55",
              "-af", ("bandpass=f=1100:width_type=o:w=2.2,"
                      "volume='min(1,t/0.05)*exp(-5.5*t)':eval=frame,"
                      "afade=t=out:st=0.45:d=0.1"),
              *tts.AUDIO_ARGS, str(out)])
        return out
    except Exception as e:  # noqa: BLE001 - a missing whoosh is never worth a failed mix
        print(f"whoosh synthesis failed: {e}")
        return None


def dedupe(hits, min_gap: float = MIN_GAP, max_hits: int = MAX_HITS) -> list:
    """Ordered hit times at least min_gap apart - concat_timed needs non-overlapping parts."""
    out = []
    for t in sorted(float(h) for h in hits or []):
        if t < 0:
            continue
        if not out or t - out[-1] >= min_gap:
            out.append(round(t, 3))
        if len(out) >= max_hits:
            break
    return out


def build_track(hits, out_wav, total: float, whoosh=None) -> Path | None:
    """One full-length wav with a whoosh at each hit time, or None when there is nothing to mark."""
    total = float(total or 0.0)
    times = [t for t in dedupe(hits) if t < total]
    if not times or total <= 0:
        return None
    w = whoosh or ensure_whoosh()
    if not w or not Path(w).exists():
        return None
    out_wav = Path(out_wav)
    tts.concat_timed([(Path(w), t) for t in times], out_wav, total=total)
    return out_wav
