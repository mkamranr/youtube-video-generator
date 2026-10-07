"""Burned-in captions: word timings in, an .ass subtitle track out.

Captions are an .ass file burned by libass as the last filter of the overlay pass that already
runs, NOT one PNG per phrase through video.composite. The arithmetic decides it: a three-minute
video is 60-120 phrases and a full-frame RGBA PNG at 4K is about 33 MB decoded, so the PNG route
would hand ffmpeg several gigabytes of input buffers and that many extra overlay steps. An .ass
track is one filter, one small file, no extra inputs - and it carries PlayResX/PlayResY, so it
scales itself to whatever size the video ended up after the upscale.

Where the word timings come from depends on the mode, and only one of the three costs anything:
  story  - exact by construction (every line is synthesized alone and laid on the clip grid)
  music  - already transcribed and aligned for the lyric-synced scenes
  voiceover - needs one whisper pass over narration.wav, cached in plan["caption_words"]
"""
import re
from pathlib import Path

import config

MAX_WORDS = 3
MAX_CHARS = 26
MAX_GAP = 0.6        # a silence longer than this is a breath: start a new phrase
MAX_DUR = 2.6
MIN_DUR = 0.45
HOLD = 0.25          # a gap smaller than this is absorbed, so captions do not flicker off and on
SHORTS_SAFE = 0.14   # portrait: the fraction of the height the Shorts UI owns (storyboard's rule)


def _text(w) -> str:
    return str((w.get("word") if "word" in w else w.get("text")) or "").strip()


def group(words, max_words: int = 0, max_chars: int = 0, max_gap: float = MAX_GAP,
          max_dur: float = MAX_DUR, min_dur: float = MIN_DUR, hold: float = HOLD) -> list:
    """Turn word timings into on-screen phrases: [{start, end, words: [{start, end, word}]}].

    A phrase breaks on whichever comes first: the word budget, the character budget, a real
    silence, a sentence ending, or the duration cap. Breaking on punctuation matters more than it
    looks - a caption that straddles a full stop reads as one thought and is actively confusing."""
    max_words = max_words or MAX_WORDS
    max_chars = max_chars or MAX_CHARS
    out, cur, text = [], [], ""
    for w in words or []:
        t = _text(w)
        if not t:
            continue
        item = {"start": float(w.get("start") or 0.0), "end": float(w.get("end") or 0.0),
                "word": t}
        if cur:
            prev = cur[-1]
            nxt = (text + " " + t).strip()
            if (len(cur) >= max_words or len(nxt) > max_chars
                    or item["start"] - prev["end"] > max_gap
                    or item["end"] - cur[0]["start"] > max_dur
                    or prev["word"][-1:] in ".!?"
                    or (prev["word"][-1:] in ",;:—" and len(cur) >= 2)):
                out.append(cur)
                cur, text = [], ""
        cur.append(item)
        text = (text + " " + t).strip()
    if cur:
        out.append(cur)

    phrases = [{"start": g[0]["start"], "end": max(g[-1]["end"], g[0]["start"] + 0.05),
                "words": g, "text": " ".join(x["word"] for x in g)} for g in out]
    for i, p in enumerate(phrases):           # stretch the very short ones, never past the next
        nxt = phrases[i + 1]["start"] if i + 1 < len(phrases) else None
        if p["end"] - p["start"] < min_dur:
            p["end"] = min(p["start"] + min_dur, nxt) if nxt else p["start"] + min_dur
        if nxt is not None and 0 < nxt - p["end"] <= hold:
            p["end"] = nxt
    return phrases


def clamp(phrases, total: float) -> list:
    """Bound everything into [0, total] and drop what is left over.

    The last clip is quantised shorter than the audio it carries (config.clip_durations), so a
    whisper timestamp can legitimately run past the end of the video."""
    total = float(total or 0.0)
    out = []
    for p in phrases or []:
        start, end = max(0.0, float(p["start"])), min(total, float(p["end"]))
        if end - start < 0.08 or not str(p.get("text") or "").strip():
            continue
        q = dict(p, start=round(start, 3), end=round(end, 3))
        q["words"] = [dict(w, start=max(start, min(float(w["start"]), end)),
                           end=max(start, min(float(w["end"]), end))) for w in p.get("words") or []]
        out.append(q)
    return out


def spread_words(text: str, start: float, end: float, min_word: float = 0.12) -> list:
    """Word timings for a line whose span is known but whose words were never timed.

    Weighted by word length rather than split evenly: "a" and "extraordinary" do not take the same
    time to say, and an even split visibly drifts by the end of a long line."""
    toks = [t for t in str(text or "").split() if t]
    if not toks:
        return []
    start, end = float(start), max(float(end), float(start) + 0.05)
    weights = [len(t) + 1 for t in toks]
    span = end - start
    total_w = sum(weights)
    out, at = [], start
    for tok, wt in zip(toks, weights):
        dur = max(min_word, span * wt / total_w)
        out.append({"start": round(at, 3), "end": round(min(end, at + dur), 3), "word": tok})
        at = min(end, at + dur)
    out[-1]["end"] = round(end, 3)
    return out


def words_from_timeline(timed) -> list:
    """Word timings out of a per-clip timeline (music mode's plan["timed_segments"])."""
    out = []
    for seg in timed or []:
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words") or []:
            t = _text(w)
            if t:
                out.append({"start": float(w.get("start") or 0.0),
                            "end": float(w.get("end") or 0.0), "word": t})
    return out


def words_from_lines(lines, clip_seconds, lead: float = 0.0) -> list:
    """Word timings for a story video, derived rather than transcribed.

    Every line is synthesized on its own and laid on the clip grid by tts.concat_timed, so each
    line's start is known exactly; only the words inside it are spread."""
    out, at = [], 0.0
    for ln, dur in zip(lines or [], clip_seconds or []):
        text = (ln.get("text") or "").strip() if isinstance(ln, dict) else str(ln or "").strip()
        spoken = float((ln.get("seconds") if isinstance(ln, dict) else 0) or 0.0)
        if text and spoken > 0:
            out += spread_words(text, at + lead, at + lead + spoken)
        at += float(dur or 0.0)
    return out


# ---- .ass writing ----------------------------------------------------------
def ass_time(t: float) -> str:
    """ASS wants h:mm:ss.cc with centisecond precision."""
    t = max(0.0, float(t or 0.0))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}.{int(round((t - int(t)) * 100)) % 100:02d}"


def ass_colour(hex_colour: str) -> str:
    """#rrggbb -> &HBBGGRR& - ASS stores colours byte-reversed."""
    c = (hex_colour or "#ffffff").lstrip("#")
    try:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        r, g, b = 255, 255, 255
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _escape(text: str) -> str:
    """ASS treats { } as override blocks and \\N as a line break."""
    return (str(text or "").replace("\\", "∖").replace("{", "(").replace("}", ")")
            .replace("\n", " ").strip())


def lift_spans(rects, size: tuple, gap: int = 0) -> list:
    """[(start, end, margin_v)] windows where a caption must sit higher than usual.

    A story board card owns the lower third for whole clips at a time, so suppressing captions
    while one is up would gut the feature on exactly the videos it matters most for. Instead the
    caption moves above the card for that window."""
    H = int(size[1])
    gap = gap or int(0.02 * H)
    out = []
    for start, end, rect in rects or []:
        if not rect:
            continue
        _x0, y0, _x1, _y1 = rect
        margin = max(0, H - int(y0) + gap)
        if margin > 0:
            out.append((float(start), float(end), margin))
    return out


def _margin_at(t: float, lifts, default: int) -> int:
    for start, end, margin in lifts or []:
        if start <= t < end:
            return margin
    return default


def _split_on_lifts(phrases, lifts) -> list:
    """Split any phrase that straddles a lift boundary, so each piece has one placement."""
    edges = sorted({e for span in (lifts or []) for e in span[:2]})
    out = []
    for p in phrases:
        cuts = [c for c in edges if p["start"] < c < p["end"]]
        if not cuts:
            out.append(p)
            continue
        at = p["start"]
        for c in cuts + [p["end"]]:
            words = [w for w in p["words"] if w["start"] < c and w["end"] > at] or p["words"]
            out.append(dict(p, start=at, end=c, words=words))
            at = c
    return out


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00141008,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{mx},{mx},{mv},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def write_ass(phrases, size: tuple, out_ass, accent: str = "", font: str = "",
              font_px: int = 0, bottom_margin: float = 0.0, highlight: str = "word",
              lifts=()) -> Path:
    """Write the caption track. Returns the path.

    Word highlight is done by emitting one event per word-state, each carrying the WHOLE phrase
    with a single word recoloured. Colour only, never size or weight: changing a word's metrics
    reflows the centred line and makes it jump on every single word."""
    W, H = int(size[0]), int(size[1])
    portrait = H > W * 1.05
    margin = bottom_margin or config.CAPTION_BOTTOM_MARGIN
    if portrait:                      # never below the Shorts UI, whatever the setting says
        margin = max(SHORTS_SAFE, margin)
    size_px = int(font_px or round(H * 0.052 * config.CAPTION_FONT_SCALE))
    hi = ass_colour(accent or config.CAPTION_ACCENT)
    white = "&H00FFFFFF"
    head = ASS_HEADER.format(W=W, H=H, font=font or "Arial", size=size_px,
                             outline=max(2, round(size_px * 0.08)),
                             shadow=max(1, round(size_px * 0.04)),
                             mx=int(0.06 * W), mv=int(margin * H))
    default_mv = int(margin * H)
    lines = []
    for p in _split_on_lifts(clamp(phrases, float("inf")), lifts):
        mv = _margin_at(p["start"], lifts, default_mv)
        over = f",,0,0,{mv if mv != default_mv else 0},,"
        text = _escape(p.get("text") or " ".join(w["word"] for w in p["words"]))
        if highlight != "word" or len(p["words"]) < 2:
            lines.append(f"Dialogue: 0,{ass_time(p['start'])},{ass_time(p['end'])},Cap{over}"
                         f"{{\\fad(70,70)}}{text}")
            continue
        words = p["words"]
        for i, w in enumerate(words):
            start = p["start"] if i == 0 else max(p["start"], float(w["start"]))
            end = p["end"] if i == len(words) - 1 else min(p["end"], float(words[i + 1]["start"]))
            if end - start < 0.02:
                continue
            parts = []
            for k, x in enumerate(words):
                tok = _escape(x["word"])
                parts.append(f"{{\\c{hi}&}}{tok}{{\\c{white}&}}" if k == i else tok)
            # fades only on the phrase's own edges, or it strobes once per word
            fade = "{\\fad(70,0)}" if i == 0 else ("{\\fad(0,70)}" if i == len(words) - 1 else "")
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap{over}"
                         f"{fade}{' '.join(parts)}")
    out_ass = Path(out_ass)
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    out_ass.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return out_ass


def font_family(bold: bool = True) -> tuple:
    """(family name, directory) of the face the story board cards use, for libass.

    libass resolves by family name, not by path, so the name is read back out of the very file
    the repo's own font chain picked - which keeps captions and cards in the same typeface on
    Windows and in the container alike."""
    from PIL import ImageFont

    from services import storyboard
    for cand in storyboard._font_candidates(bold):
        try:
            if cand and Path(cand).exists():
                return ImageFont.truetype(cand, 12).getname()[0], str(Path(cand).parent)
        except Exception:  # noqa: BLE001
            continue
    return "Arial", ""
