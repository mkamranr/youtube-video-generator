"""Whisper-based lyric timing: word timestamps, alignment to the lyrics we wrote ourselves,
and the per-clip timeline (which words are sung when, relative to each clip's start)."""
import difflib
import re

import config

_model = None
_model_name = None


def _get_model():
    global _model, _model_name
    if _model is None or _model_name != config.WHISPER_MODEL:   # Settings may switch the size
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        _model_name = config.WHISPER_MODEL
    return _model


def words_from_segments(segs) -> list:
    """Flatten faster-whisper segments to [{start, end, word, p}]; whisper words carry a
    leading space, which is stripped here."""
    out = []
    for s in segs:
        for w in (s.words or []):
            t = str(w.word).strip()
            if t:
                out.append({"start": float(w.start), "end": float(w.end), "word": t,
                            "p": float(getattr(w, "probability", 1.0) or 0.0)})
    return out


def transcribe_words(audio_path, initial_prompt: str = "", on_progress=None) -> list:
    """Word-level timestamps of everything sung in the track. `initial_prompt` (the lyrics we
    wrote) primes whisper's vocabulary so mis-hearings are rarer."""
    if on_progress:
        on_progress("transcribing the song for lyric timing")
    segs, _info = _get_model().transcribe(str(audio_path), word_timestamps=True, language="en",
                                          initial_prompt=(initial_prompt or None),
                                          condition_on_previous_text=False)
    return words_from_segments(segs)


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", "", tok.lower())


def lyric_tokens(lyrics: str) -> list:
    """[(normalised, original, line_index)] of the sung words in the lyrics: section tags like
    [Chorus] and parentheticals are dropped; line_index groups the words into lyric lines."""
    out, line = [], 0
    for raw in (lyrics or "").splitlines():
        text = re.sub(r"\[[^\]]*\]", " ", raw)
        text = re.sub(r"\([^)]*\)", " ", text)
        toks = [(_norm(t), t) for t in text.split()]
        toks = [(n, t) for n, t in toks if n]
        if not toks:
            continue
        out.extend((n, t, line) for n, t in toks)
        line += 1
    return out


def same_words(a: str, b: str) -> bool:
    """True when two lyric sheets sing the same word sequence (section tags, parentheticals, line
    breaks, case and punctuation ignored) - the guard behind "the user's words, verbatim"."""
    return [n for n, _, _ in lyric_tokens(a)] == [n for n, _, _ in lyric_tokens(b)]


def ensure_sections(text: str) -> str:
    """Lyrics the singer model can structure: a [Verse] tag is prepended when the user wrote none."""
    t = (text or "").strip()
    return t if re.search(r"^\s*\[[^\]]+\]", t, re.M) else "[Verse]\n" + t


def _text(w: dict) -> str:
    return w.get("text") or w.get("word") or ""


def _spread(out: list, toks: list, t0: float, t1: float, src: str):
    """Give the lyric tokens `toks` evenly spaced times over (t0, t1)."""
    k = len(toks)
    if not k:
        return
    t1 = max(t1, t0 + 0.2 * k)
    step = (t1 - t0) / k
    for m, (_, orig, line) in enumerate(toks):
        out.append({"start": round(t0 + m * step, 3), "end": round(t0 + (m + 1) * step, 3),
                    "text": orig, "line": line, "src": src})


def align_to_lyrics(words: list, lyrics: str) -> tuple:
    """Snap the whisper words onto the lyric text.

    Matched words keep their whisper timing but take the lyric spelling; mis-heard runs are
    re-spread over their time span; lyric words whisper missed get interpolated times; extra
    whisper words (ad-libs, a repeated chorus not written out) are dropped unless they form a
    long run, which is kept verbatim so the timeline never goes silent while singing is heard.
    Returns (aligned_words, match_ratio). Below LYRIC_ALIGN_MIN_RATIO (or without lyrics) the
    raw whisper words are returned unchanged, so a failed alignment never invents text."""
    raw = [{"start": w["start"], "end": w["end"], "text": _text(w), "src": "whisper"} for w in words]
    toks = lyric_tokens(lyrics)
    if not words or not toks:
        return raw, 0.0
    sm = difflib.SequenceMatcher(None, [_norm(_text(w)) for w in words], [n for n, _, _ in toks],
                                 autojunk=False)
    ratio = sm.ratio()
    if ratio < config.LYRIC_ALIGN_MIN_RATIO:
        return raw, ratio
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                w, (_, orig, line) = words[i1 + k], toks[j1 + k]
                out.append({"start": w["start"], "end": w["end"], "text": orig, "line": line, "src": "match"})
        elif tag == "replace":
            _spread(out, toks[j1:j2], words[i1]["start"], words[i2 - 1]["end"], "replace")
        elif tag == "insert":
            pad = 0.35 * (j2 - j1)
            t0 = words[i1 - 1]["end"] if i1 > 0 else max(0.0, words[0]["start"] - pad)
            t1 = words[i1]["start"] if i1 < len(words) else words[-1]["end"] + pad
            _spread(out, toks[j1:j2], t0, t1, "inferred")
        elif i2 - i1 >= 4:      # delete: a long run whisper heard but the lyrics lack (extra chorus)
            out.extend(raw[i1:i2])
    prev = 0.0
    for w in out:               # keep time monotonic and non-negative
        w["start"] = round(max(prev, w["start"]), 3)
        w["end"] = round(max(w["start"], w["end"]), 3)
        prev = w["end"]
    return out, ratio


def _rel(t: float, lo: float) -> float:
    return round(max(0.0, t - lo), 2)


def _same_line(a: dict, b: dict, gap: float) -> bool:
    if a.get("line") is not None and b.get("line") is not None:
        return a["line"] == b["line"] and b["start"] - a["end"] < 2.0
    return b["start"] - a["end"] < gap


def timeline(words: list, bounds: list, gap: float = 0.8, rest: float = 1.0) -> list:
    """One entry per clip: {"clip", "start", "end", "text", "lines"}.

    Words are bucketed by their midpoint against the real clip boundaries `bounds` (cumulative
    end times). Consecutive words of the same lyric line (or closer than `gap` seconds for raw
    whisper words) form a line; pauses longer than `rest` seconds become instrumental entries.
    Line times are relative to the clip's start, so they can be quoted in a scene prompt."""
    out, lo = [], 0.0
    for i, hi in enumerate(bounds):
        ws = [w for w in words if lo <= (w["start"] + w["end"]) / 2 < hi]
        groups = []
        for w in ws:
            if groups and _same_line(groups[-1][-1], w, gap):
                groups[-1].append(w)
            else:
                groups.append([w])
        lines, cursor = [], lo
        for g in groups:
            s, e = max(g[0]["start"], lo), min(g[-1]["end"], hi)
            if s - cursor > rest:
                lines.append({"start": _rel(cursor, lo), "end": _rel(s, lo), "text": "", "instrumental": True})
            lines.append({"start": _rel(s, lo), "end": _rel(max(s, e), lo), "text": " ".join(_text(w) for w in g)})
            cursor = max(cursor, e)
        if groups and hi - cursor > rest:
            lines.append({"start": _rel(cursor, lo), "end": _rel(hi, lo), "text": "", "instrumental": True})
        out.append({"clip": i + 1, "start": round(lo, 3), "end": round(hi, 3),
                    "text": " ".join(_text(w) for w in ws), "lines": lines})
        lo = hi
    return out


def format_timeline(seg: dict) -> str:
    """The clip's sung lines as prompt text: [0.0s-1.8s] "Good morning sun"  [1.8s-3.1s] (instrumental) ..."""
    lines = seg.get("lines") or []
    if not any(l.get("text") for l in lines):
        return "(instrumental for the whole clip - no words are heard)"
    parts = []
    for l in lines:
        span = f"[{l['start']:.1f}s-{l['end']:.1f}s]"
        parts.append(f"{span} (instrumental)" if l.get("instrumental") else f'{span} "{l["text"]}"')
    return "  ".join(parts)


def words_per_second(words: list, max_gap: float = 1.5) -> float:
    """Singing pace: words over the sung span (pauses count at most `max_gap` seconds each).
    0.0 when there are too few words to judge."""
    if len(words) < 8:
        return 0.0
    span = 0.0
    for a, b in zip(words, words[1:]):
        span += max(0.0, a["end"] - a["start"]) + min(max_gap, max(0.0, b["start"] - a["end"]))
    span += max(0.0, words[-1]["end"] - words[-1]["start"])
    return round(len(words) / span, 2) if span > 0 else 0.0


def transcribe_windows(audio_path, n: int, clip_seconds: float, on_progress=None) -> list:
    """Legacy helper: n strings, the words sung during consecutive fixed-length windows."""
    words, _ratio = align_to_lyrics(transcribe_words(audio_path, on_progress=on_progress), "")
    return [t["text"] for t in timeline(words, [clip_seconds * (i + 1) for i in range(n)])]
