"""YouTube chapters: the timestamp list that splits the player into labelled sections.

Pure arithmetic over the measured clip cut points - no ffmpeg, no LLM, no disk. The titles come
from explicit marks when the caller already knows them (a compilation knows each part's own
title), then from the story board cards when the video has them (they are already short, sequenced
headlines), and fall back to the story beats or the per-clip audio segments.

YouTube's rules are strict and silent: the first chapter MUST be at 0:00, there must be at least
three, and every one of them - including the last - must run at least ten seconds. A list that
breaks any of those is ignored in full, so a half-valid list buys nothing and just leaves the
description looking broken. build() therefore returns an empty list rather than something partial.
"""
import re

MIN_GAP = 20.0       # chapters are deliberately coarser than cards: a clip is ~5 s and a card
                     # spans one or two of them, so one chapter per card would be mostly illegal
MIN_LAST = 10.0      # YouTube's own floor, which the LAST chapter has to clear too
MIN_COUNT = 3
MAX_COUNT = 12
MAX_TITLE = 55
INTRO = "Intro"


def fmt_time(seconds: float) -> str:
    """m:ss under an hour, h:mm:ss over it. The first chapter must read exactly '0:00'."""
    s = max(0, int(float(seconds or 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def clean_title(text: str, max_chars: int = MAX_TITLE) -> str:
    """A chapter label: no '#', no angle brackets, no trailing punctuation, word-boundary trim.

    The '#' matters more than it looks. YouTube reads the first three hashtags in a description
    and ignores EVERY hashtag once there are more than 15, so a '#1' surviving into a chapter line
    would either outrank the channel's real tags or void the lot - the same trap documented in
    services/resource.py.
    """
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.replace("<", "").replace(">", "")
    t = re.sub(r"(?<!\w)#(?=\w)", "", t)            # '#1' -> '1', '#tag' -> 'tag'
    t = t.lstrip("#*-•–— .").strip()
    t = re.sub(r"[\s,;:.!?\-–—]+$", "", t)
    if len(t) > max_chars:
        cut = t[:max_chars].rsplit(" ", 1)[0]
        t = (cut or t[:max_chars]).rstrip(" ,;:.-")
    return t


def _from_marks(marks) -> list:
    """Cut points the caller already knows, as [{"t", "title"}] or (t, title) pairs.

    A compilation is the case this exists for: each part contributes one chapter at its own start,
    titled with the video it came from, and nothing has to be inferred from the picture."""
    out = []
    for m in marks or []:
        if isinstance(m, dict):
            t, title = m.get("t"), m.get("title")
        elif isinstance(m, (tuple, list)) and len(m) >= 2:
            t, title = m[0], m[1]
        else:
            continue
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        title = clean_title(title)
        if title:
            out.append((max(0.0, t), title))
    return out


def _from_cards(cards, bounds) -> list:
    out = []
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        try:
            a = int(c.get("start_clip") or 1)
        except (TypeError, ValueError):
            continue
        # the cut the card's first clip starts on - NOT the card's fade-in (storyboard's
        # CARD_INSET), because a chapter marks the cut
        t = bounds[a - 2] if a > 1 and a - 2 < len(bounds) else 0.0
        title = clean_title(c.get("headline") or c.get("kicker") or "")
        if title:
            out.append((float(t), title))
    return out


def _from_beats(beats, bounds) -> list:
    out = []
    for i, b in enumerate(beats or []):
        if not isinstance(b, dict) or i >= len(bounds):
            continue
        title = clean_title(b.get("action") or b.get("setting") or "")
        if title:
            out.append((bounds[i - 1] if i else 0.0, title))
    return out


def _from_segments(segments, bounds) -> list:
    out = []
    for i, seg in enumerate(segments or []):
        if i >= len(bounds):
            break
        words = str(seg or "").split()
        title = clean_title(" ".join(words[:7]))
        if title:
            out.append((bounds[i - 1] if i else 0.0, title))
    return out


def _thin(cands: list, total: float, min_gap: float) -> list:
    """Keep 0:00, then only candidates at least min_gap past the last kept one, and never one so
    close to the end that its own chapter would be under ten seconds."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c[0])
    first = cands[0]
    kept = [(0.0, first[1] if first[0] < min_gap else INTRO)]
    for t, title in cands:
        if t <= 0.0:
            continue
        if t - kept[-1][0] < min_gap:
            continue
        if total - t < MIN_LAST:
            break
        kept.append((t, title))
    return kept


def build(bounds, cards=None, segments=None, beats=None, total: float = 0.0,
          min_gap: float = MIN_GAP, min_count: int = MIN_COUNT,
          max_count: int = MAX_COUNT, marks=None) -> list:
    """[{"t": seconds, "title": str}] for the description, or [] when the video cannot have them.

    Sources are tried in order of how good a chapter title they make - explicit marks the caller
    passed in, then story board headlines, then story beats, then the per-clip audio segments -
    and the first that survives the rules wins.
    """
    bounds = [float(b) for b in (bounds or [])]
    total = float(total or (bounds[-1] if bounds else 0.0))
    if total < min_count * min_gap:
        return []
    for cands in (_from_marks(marks), _from_cards(cards, bounds), _from_beats(beats, bounds),
                  _from_segments(segments, bounds)):
        kept = _thin(cands, total, min_gap)
        if len(kept) >= min_count:
            if len(kept) > max_count:          # keep 0:00 and sample the rest evenly
                rest = kept[1:]
                step = len(rest) / (max_count - 1)
                kept = [kept[0]] + [rest[int(i * step)] for i in range(max_count - 1)]
            return [{"t": round(t, 3), "title": title} for t, title in kept]
    return []


def format_block(chapters, header: str = "Chapters") -> str:
    """The description block. Empty string for an empty list, so callers can append blindly."""
    if not chapters:
        return ""
    lines = [f"{fmt_time(c['t'])} {c['title']}" for c in chapters]
    return "\n".join(([header] if header else []) + lines)
