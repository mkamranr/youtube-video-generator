"""Fetch a YouTube transcript: youtube-transcript-api first, yt-dlp auto-subs as fallback."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def video_id(url: str) -> str:
    u = urlparse(url.strip())
    if u.hostname == "youtu.be":
        return u.path.lstrip("/").split("/")[0]
    if u.hostname and "youtube" in u.hostname:
        if u.path == "/watch":
            return parse_qs(u.query)["v"][0]
        m = re.match(r"^/(shorts|embed|live|v)/([A-Za-z0-9_-]{6,})", u.path)
        if m:
            return m.group(2)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError(f"Cannot extract a YouTube video id from: {url}")


def _clean(text: str) -> str:
    """Drop non-speech cue tags like [Music], [Applause] and collapse whitespace."""
    text = re.sub(r"\[[^\]]{0,40}\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _via_api(vid: str) -> tuple:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    listing = api.list(vid)
    try:
        t = listing.find_transcript(["en", "en-US", "en-GB"])
    except Exception:
        t = next(iter(listing))
        if not t.language_code.startswith("en") and t.is_translatable:
            t = t.translate("en")
    snippets = [s for s in t.fetch() if s.text.strip()]
    text = _clean(" ".join(s.text.strip() for s in snippets))
    seconds = max((s.start + (s.duration or 0)) for s in snippets) if snippets else 0.0
    return text, seconds


def _via_ytdlp(vid: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--skip-download", "--write-auto-sub", "--write-sub",
             "--sub-lang", "en.*,en", "--sub-format", "json3", "-o", f"{d}/%(id)s",
             f"https://www.youtube.com/watch?v={vid}"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            errs = [l for l in (r.stderr or "").splitlines() if l.startswith("ERROR")]
            raise RuntimeError(errs[-1] if errs else (r.stderr or "yt-dlp failed").strip()[-300:])
        files = list(Path(d).glob("*.json3"))
        if not files:
            raise RuntimeError("yt-dlp found no subtitles")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        words = []
        for ev in data.get("events", []):
            for seg in ev.get("segs", []) or []:
                words.append(seg.get("utf8", ""))
        text = _clean("".join(words))
        if not text:
            raise RuntimeError("subtitle file was empty")
        ends = [ev.get("tStartMs", 0) + ev.get("dDurationMs", 0) for ev in data.get("events", [])]
        return text, (max(ends) / 1000.0 if ends else 0.0)


def get_transcript_info(url: str) -> dict:
    """Transcript text plus the seconds of video the captions cover."""
    vid = video_id(url)
    errors = []
    for fn in (_via_api, _via_ytdlp):
        try:
            text, seconds = fn(vid)
            if len(text) > 20:
                return {"text": text, "seconds": seconds}
            errors.append(f"{fn.__name__}: transcript too short")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("This video cannot be processed: no transcript is available. " + " | ".join(errors))


def get_transcript(url: str) -> str:
    return get_transcript_info(url)["text"]


if __name__ == "__main__":
    info = get_transcript_info(sys.argv[1])
    print(f"coverage: {info['seconds']:.0f}s, words: {len(info['text'].split())}")
    print(info["text"][:2000])
