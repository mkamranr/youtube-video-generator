"""Story videos: the clip grid, the spoken-line timing and the "don't describe the picture" check.

A story video inverts the song pipeline. A song is rendered first and the clip grid is derived from
its measured length; a story is planned first as N beats, one spoken line is written for each beat,
and the clips are then sized around the lines. Because every line is synthesized on its own we know
each timestamp exactly - so a story needs no whisper pass and no alignment at all. The per-clip
entries built here deliberately copy services/lyrics.timeline's shape, so llm.write_scenes and the
job page consume them unchanged.

Pure functions only: no job, no disk, no network.
"""
import re

import config

_REST = 0.5         # silence shorter than this is not worth its own timeline entry


def beat_count(target_seconds: float, clip_seconds: float, model: str = "") -> int:
    """How many scenes a story of this target length is told in, clamped to the STORY_*_BEATS range.

    A pure function of its arguments, so a resume recomputes the same number - but the pipeline
    freezes len(beats) anyway, so a Settings change can never re-cut a job already in flight.

    model: the job's video model, which decides what a requested clip length really renders as."""
    full = config.actual_clip_seconds(clip_seconds, model)
    n = round(float(target_seconds or 0.0) / full) if full else 0
    return max(int(config.STORY_MIN_BEATS), min(int(config.STORY_MAX_BEATS), int(n)))


def line_words(clip_seconds: float, model: str = "") -> int:
    """Word budget for one spoken line: what fits in a clip at the story speaking pace, with the
    lead-in and the tail silence taken out first."""
    speak = (config.actual_clip_seconds(clip_seconds, model)
             - config.STORY_LINE_LEAD - config.STORY_LINE_GAP)
    return max(4, int(speak * config.STORY_WPS))


def max_line_words() -> int:
    """Longest line that still fits the longest clip a story line may buy."""
    speak = config.STORY_MAX_CLIP_SECONDS - config.STORY_LINE_LEAD - config.STORY_LINE_GAP
    return max(6, int(speak * config.STORY_WPS))


def clip_durations(line_seconds: list, clip_seconds: float, model: str = "") -> list:
    """Real length of each clip: the preset, or longer when its line needs the room.

    The floor is what keeps the pacing - a run of terse lines still renders full-length clips, so a
    story can only ever come out longer than its target, never shorter. There is no ceiling: a line
    that still runs long after its one rewrite buys the clip it needs, because cutting a child off
    mid-sentence is worse than one slow shot.

    model: the job's video model, which decides what each length really renders as."""
    full = config.actual_clip_seconds(clip_seconds, model)
    out = []
    for s in line_seconds:
        want = config.STORY_LINE_LEAD + max(0.0, float(s)) + config.STORY_LINE_GAP
        out.append(config.actual_clip_seconds(max(full, want), model))
    return out


def timeline(lines: list, line_seconds: list, durs: list) -> list:
    """One entry per clip in services/lyrics.timeline's shape:
    {"clip", "start", "end", "text", "lines": [{start, end, text} | {..., instrumental: True}]}.

    Times inside "lines" are relative to the clip's start, so they can be quoted in a scene prompt.
    The silence around the line is marked instrumental exactly as an instrumental passage of a song
    is, so write_scenes already knows to carry motion there instead of inventing content."""
    out, lo = [], 0.0
    lead = float(config.STORY_LINE_LEAD)
    for i, dur in enumerate(durs):
        text = str((lines[i] or {}).get("text", "") if i < len(lines) else "").strip()
        spoken = max(0.0, float(line_seconds[i])) if i < len(line_seconds) else 0.0
        dur = float(dur)
        start = min(lead, max(0.0, dur - spoken)) if text else 0.0
        end = min(dur, start + spoken) if text else 0.0
        entries = []
        if text and start >= _REST:
            entries.append({"start": 0.0, "end": round(start, 3), "text": "", "instrumental": True})
        if text:
            entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        if dur - end >= _REST:
            entries.append({"start": round(end, 3), "end": round(dur, 3), "text": "",
                            "instrumental": True})
        out.append({"clip": i + 1, "start": round(lo, 3), "end": round(lo + dur, 3),
                    "text": text, "lines": entries})
        lo += dur
    return out


def format_timeline(seg: dict) -> str:
    """The clip's spoken line as prompt text, mirroring lyrics.format_timeline but saying "no
    speech" where a song says "instrumental" - the right word when nobody is singing."""
    entries = (seg or {}).get("lines") or []
    if not any(e.get("text") for e in entries):
        return "(no speech during this clip - nothing is said)"
    parts = []
    for e in entries:
        span = f"[{e['start']:.1f}s-{e['end']:.1f}s]"
        parts.append(f"{span} (no speech)" if e.get("instrumental") else f'{span} "{e["text"]}"')
    return "  ".join(parts)


# ---- "the voiceover must not describe the picture" --------------------------
# The spoken lines are written by a model that never sees the character looks (llm._system with
# spoken=True strips them), but a channel's own rules text, or the model's own habits, can still
# produce "Ellie the elephant in her yellow dress". This is the deterministic catch: pure string
# work, so it costs nothing, is unit-testable, and only ever asks for ONE line to be rewritten.

APPEARANCE_WORDS = {
    # colours
    "red", "orange", "yellow", "green", "blue", "teal", "purple", "violet", "lavender", "pink",
    "brown", "black", "white", "grey", "gray", "golden", "silver", "cream", "mustard", "emerald",
    "crimson", "scarlet", "turquoise", "beige", "tan", "amber",
    # clothing and worn props
    "hoodie", "scarf", "sneakers", "shoes", "boots", "shirt", "tshirt", "shorts", "trousers",
    "pants", "dress", "skirt", "overalls", "hat", "cap", "backpack", "jacket", "coat", "sweater",
    "gloves", "ribbon", "patch", "emblem", "badge", "apron", "vest",
    # body and texture
    "fur", "feathers", "feather", "plumage", "skin", "snout", "muzzle", "trunk", "beak", "paws",
    "paw", "claws", "whiskers", "mane", "tail", "ears", "wings", "tuft", "markings", "stripes",
    "spots", "fluffy", "furry", "plush", "velvety", "glossy",
}

# phrases that describe the shot rather than the story
GIVEAWAY_PHRASES = (
    "wearing", "wears", "dressed in", "dressed as", "we see", "you can see", "on screen",
    "on-screen", "in this scene", "in the scene", "the camera", "the shot", "the picture",
    "close-up", "zooms in", "pans across", "cuts to", "fades to",
)

_NEAR = 4       # an appearance word this many words after a name is describing that character
# ... but only while the words still belong to that character. An article opens a NEW noun phrase,
# so "Ellie boarded the blue train" is about a train, not about Ellie, and must not be flagged -
# whereas "Ellie's yellow hat" and "Ellie waved her blue scarf" both are.
_PHRASE_BREAK = {"the", "a", "an"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_KIND_SKIP = {"year", "old", "years", "baby", "young", "small", "tiny", "little", "giant", "adult",
              "the", "and", "with", "about"}


def _kind_words(kind: str) -> set:
    """The nouns of a character's kind worth catching ("5-year-old anthropomorphic red fox" ->
    {anthropomorphic, fox}); the generic age and size filler is dropped."""
    return {w.lower() for w in _WORD.findall(kind or "")
            if len(w) > 2 and w.lower() not in _KIND_SKIP and w.lower() not in APPEARANCE_WORDS}


def describes_look(text: str, chars: list) -> list:
    """The bits of a spoken line that describe the picture instead of telling the story.

    Three catches, deliberately narrow so a clean line is never rewritten:
      1. a character's KIND is said out loud ("Ellie the elephant") - the viewer can see that;
      2. an appearance word within a few words of a character's name ("Ellie's yellow hat");
      3. a phrase that describes the shot itself ("wearing", "we see", "the camera").
    Returns the offending snippets, empty when the line is clean."""
    low = (text or "").lower()
    if not low.strip():
        return []
    words = [w.lower() for w in _WORD.findall(low)]
    names = {str(c.get("name", "")).lower() for c in (chars or []) if c.get("name")}
    hits = []
    for c in chars or []:
        for k in sorted(_kind_words(c.get("kind", ""))):
            if k in names or k in hits:
                continue
            if re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", low):
                hits.append(k)
    for i, w in enumerate(words):
        if w.rstrip("s").rstrip("'") not in names and w not in names:
            continue
        for nxt in words[i + 1:i + 1 + _NEAR]:
            if nxt in _PHRASE_BREAK:
                break
            if nxt in APPEARANCE_WORDS and nxt not in hits:
                hits.append(nxt)
    for phrase in GIVEAWAY_PHRASES:
        if phrase in low and phrase not in hits:
            hits.append(phrase)
    return hits


def slug(name: str) -> str:
    """Filename-safe form of a speaker name, for the per-speaker voice seed in the job folder."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return s[:24] or "narrator"
