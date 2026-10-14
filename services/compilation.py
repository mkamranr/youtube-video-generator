"""Compilations: one long video stitched from several finished ones.

A channel's standalone short videos each have to find their own audience, and most never do. A
compilation of the same material is one upload that holds a viewer for ten or twenty minutes
instead of forty seconds, which is the watch time the recommendation surfaces actually run on -
on this channel a single ten-minute compilation produced a quarter of all watch time.

This module is the pure part: which finished jobs may go in, what shape they have to be brought
to, and where the chapters fall. Arithmetic and validation only - no ffmpeg, no LLM, no disk and
no import of pipeline (which imports this) - so it is unit-testable without a GPU or a server.
The stitching itself is pipeline._run_compilation over services.video helpers.

Records are read through _get so a stored dict and a live pipeline.Job both work.
"""
import math

SOURCE = "compilation"          # Job.source of a compilation, and what marks a part ineligible
MIN_PARTS = 2
MAX_PARTS = 12                  # chapters.MAX_COUNT is 12, and past that a part gets no chapter
MAX_TITLE_CHARS = 120


def _get(rec, key: str, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def is_compilation(rec) -> bool:
    return str(_get(rec, "source", "") or "") == SOURCE


def eligible(rec) -> bool:
    """Can this job be a part of a compilation?

    A finished video with a file on disk. A compilation is deliberately not eligible: nesting one
    would duplicate whole videos inside a second encode for no gain, and its chapter titles would
    be the parts of the part."""
    return bool(_get(rec, "status", "") == "done"
                and _get(rec, "final", "")
                and not _get(rec, "archived", False)
                and not is_compilation(rec))


def validate_parts(parts: list) -> list:
    """The parts in the order given, or ValueError naming what is wrong.

    One channel and one aspect ratio. Mixing 9:16 and 16:9 has no good answer - padding a portrait
    video into a landscape frame wastes two thirds of it, cropping it throws the subject away - so
    it is refused here rather than decided badly in the filtergraph."""
    parts = list(parts or [])
    if len(parts) < MIN_PARTS:
        raise ValueError(f"a compilation needs at least {MIN_PARTS} videos")
    if len(parts) > MAX_PARTS:
        raise ValueError(f"a compilation takes at most {MAX_PARTS} videos")
    ids = [str(_get(p, "id", "")) for p in parts]
    if len(set(ids)) != len(ids):
        raise ValueError("the same video was picked twice")
    for p in parts:
        if is_compilation(p):
            raise ValueError(f"{_get(p, 'title', '') or _get(p, 'id', '')} is itself a compilation")
        if not eligible(p):
            raise ValueError(f"{_get(p, 'title', '') or _get(p, 'id', '')} is not a finished video")
    channels = {str(_get(p, "channel_id", "") or "") for p in parts}
    if len(channels) > 1:
        raise ValueError("every video in a compilation must come from the same channel")
    aspects = {str(_get(p, "aspect_ratio", "") or "") for p in parts}
    if len(aspects) > 1:
        raise ValueError("every video in a compilation must have the same aspect ratio "
                         f"(picked: {', '.join(sorted(a or 'unknown' for a in aspects))})")
    return parts


def target_size(sizes: list) -> tuple:
    """The frame every part is brought to: the largest of them.

    Scaling up and never down follows video.target_size - a part that was upscaled to 4K keeps its
    pixels, and the smaller ones are stretched rather than the big ones thrown away."""
    best = (0, 0)
    for s in sizes or []:
        try:
            w, h = int(s[0]), int(s[1])
        except (TypeError, ValueError, IndexError):
            continue
        if w > 0 and h > 0 and w * h > best[0] * best[1]:
            best = (w, h)
    if best == (0, 0):
        raise ValueError("could not read the size of any of the picked videos")
    return best


def marks(titles: list, bounds: list) -> list:
    """[{"t", "title"}] - one chapter per part, at the cut it starts on.

    `bounds` are cumulative END times (services.video.clip_boundaries), so part i starts where
    part i-1 ended and part 0 starts at 0:00, which is the timestamp YouTube insists on."""
    out = []
    for i, title in enumerate(titles or []):
        title = " ".join(str(title or "").split())
        if not title:
            continue
        start = 0.0
        if i:
            if i - 1 >= len(bounds):
                break
            start = float(bounds[i - 1])
        out.append({"t": round(max(0.0, start), 3), "title": title})
    return out


def part_file(k: int) -> str:
    """Name of a part's normalized copy inside the compilation's own job directory."""
    return f"part_{int(k):02d}.mp4"


def fmt_length(seconds: float) -> str:
    """"18 Minutes" / "45 Seconds" - the length a compilation leads its title with.

    Rounded down to the minute on purpose: a title promising more than the video delivers is the
    one kind of inaccuracy a viewer punishes immediately."""
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(math.floor(seconds / 60.0))
    if minutes < 1:
        return f"{int(seconds)} Seconds"
    return f"{minutes} Minute{'s' if minutes != 1 else ''}"


def default_title(titles: list, seconds: float, channel_name: str = "") -> str:
    """The title a compilation gets when the LLM call fails - never a blank job card.

    Not a fallback nobody sees: the SEO step is one vLLM call and the stitch must not be lost to
    it being down, so the whole compilation stays non-fatal around it."""
    names = [" ".join(str(t or "").split()) for t in (titles or [])]
    names = [n for n in names if n]
    lead = names[0] if names else (channel_name or "Compilation")
    text = f"{fmt_length(seconds)} of {lead}"
    if len(names) > 1:
        text += " & More"
    return text[:MAX_TITLE_CHARS].rstrip()
