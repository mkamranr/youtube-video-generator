"""Channel cast: recurring branded characters (an elephant, a kid, ...) the channel's videos
are cast from. Not every character is in every video: the script writer picks the ones a
story needs (or exactly the ones the user named in the request); the picked names are stored
on the plan as `cast_used` and every later prompt only carries those.

Two mechanisms keep them consistent:
  * a character sheet injected into every LLM prompt (system prompt) - the full roster when
    the script is written, only the characters of this video afterwards - so scripts and
    scene prompts use the same names, looks and personalities;
  * a deterministic look prefix added to every ComfyUI prompt that names a character, so the
    canonical description reaches the video model even when the LLM paraphrased it;
plus an approved reference portrait per character: the first character of the story that has
one seeds the first frame of clip 1.
"""
import random
import re
import threading
import time

import config
from services import channels, schedule, video

ROLES = ("main", "supporting")


# ---- lookups ---------------------------------------------------------------
def all_characters(channel: dict | None) -> list:
    return [c for c in ((channel or {}).get("cast") or []) if isinstance(c, dict) and c.get("name")]


def enabled(channel: dict | None) -> bool:
    """Do this channel's videos have characters at all?

    Off = informative explainer: no protagonist, no recurring cast, only generic unnamed people on
    screen. A channel record or job snapshot written before the flag existed falls back to "the
    channel has a cast" - the same rule services.channels backfills the store with - so a job
    queued before the flag keeps behaving exactly as it did."""
    ch = channel or {}
    if "characters_enabled" in ch:
        return bool(ch["characters_enabled"])
    return bool(all_characters(ch))


def active(channel: dict | None) -> list:
    """The cast the PROMPTS see: empty while characters are off, so a channel keeps its roster and
    its portraits (the editor reads those through all_characters / channel["cast"]) while its
    videos have no actors. Everything that writes a prompt comes through here.

    A resting character is deliberately still in here. This function also gates render_prefix and
    voice_id_for, so filtering them out would strip the canonical look off a video that already
    used one the next time one of its clips was regenerated - the ranking is about what to cast
    NEXT, not a retraction of what is already made."""
    return all_characters(channel) if enabled(channel) else []


# A character's share of the channel's output (services.channels.CAST_PRIORITIES). Lower sorts
# first; the values are only ever compared, never shown.
_PRIORITY_ORDER = {"lead": 0, "normal": 1, "resting": 2}


def priority(c: dict) -> str:
    return str((c or {}).get("priority") or "normal").strip().lower()


def by_priority(chars: list) -> list:
    """Leads first, resting last, otherwise the roster's own order.

    A stable sort, so the editor's ordering still decides between two characters of equal standing
    and nothing reshuffles when a priority is left alone."""
    return sorted(chars or [], key=lambda c: _PRIORITY_ORDER.get(priority(c), 1))


def leads(channel: dict | None) -> list:
    return [c for c in active(channel) if priority(c) == "lead"]


def resting(channel: dict | None) -> list:
    return [c for c in active(channel) if priority(c) == "resting"]


def main_characters(channel: dict | None) -> list:
    """The usual protagonists, best performers first - so whatever takes "the first one" (the
    portrait that opens a video, the faces a thumbnail features) reaches for a lead."""
    return by_priority([c for c in active(channel) if c.get("role", "main") == "main"])


def names(chars: list) -> list:
    return [c["name"] for c in chars]


def used_characters(channel: dict | None, *texts: str) -> list:
    """Cast entries named in the texts (whole word, CASE-SENSITIVE), ordered by first mention.

    Case-sensitive on purpose: story text capitalises names, while "a penny", "a daisy" or
    "pip" are ordinary words that must not pull Penny, Daisy or Pip into a video."""
    blob = "\n".join(t or "" for t in texts)
    hits = []
    for c in active(channel):
        m = re.search(rf"(?<![\w]){re.escape(c['name'])}(?![\w])", blob)
        if m:
            hits.append((m.start(), c))
    return [c for _, c in sorted(hits, key=lambda x: x[0])]


def select_characters(channel: dict | None, used) -> list:
    """The cast entries whose names are in `used` (any case), in `used` order."""
    by_name = {c["name"].lower(): c for c in active(channel)}
    out, seen = [], set()
    for n in used or []:
        c = by_name.get(str(n).strip().lower())
        if c and c["name"] not in seen:
            seen.add(c["name"])
            out.append(c)
    return out


# ---- prompt text -----------------------------------------------------------
def _sheet_line(c: dict, spoken: bool = False, rank: bool = False) -> str:
    """One roster line; spoken=True leaves the look out (see sheet_text).

    rank=True adds the character's standing, which only the roster (script-writing) mode wants: by
    the time a later call gets the sheet the cast is already chosen and the ranking would just be
    an invitation to second-guess it."""
    role = "MAIN" if c.get("role", "main") == "main" else "supporting"
    kind = (c.get("kind") or "").strip()
    standing = f", {priority(c).upper()}" if rank and priority(c) != "normal" else ""
    line = f"- {c['name']} ({role}{standing}{', ' + kind if kind else ''})"
    line += "." if spoken else f": {c.get('look', '').strip()}"
    if (c.get("personality") or "").strip():
        line += f" Personality: {c['personality'].strip()}"
    return line


SPOKEN_RULE = ("This text is SPOKEN ALOUD: refer to the characters by name only. Never describe how a "
               "character looks - no fur, skin, colours, clothes, accessories or markings - the visuals "
               "carry the appearance. Never rename them and never invent other recurring named "
               "characters. ")
_LOOK_RULE = ("Never invent other recurring named characters, never rename these ones and never change "
              "their looks. Whenever a scene shows a cast character, name them and repeat their full look "
              "from this sheet. ")


def _RANK_RULE(cast: list) -> str:  # noqa: N802 - reads as a constant at the call site
    """What the roster's LEAD / RESTING marks mean to the writer.

    Only emitted when the channel actually ranked somebody, so a channel that never touched the
    field sends not one extra token and keeps the exact prompt it had before."""
    parts = []
    if any(priority(c) == "lead" for c in cast):
        parts.append("The characters marked LEAD are the ones this channel's audience comes back for: "
                     "cast one of them in the lead unless the request asks for someone else.")
    if any(priority(c) == "resting" for c in cast):
        parts.append("A character marked RESTING is not to be cast at all unless the request names "
                     "them explicitly.")
    return (" ".join(parts) + " ") if parts else ""


def sheet_text(channel: dict | None, used: list | None = None, spoken: bool = False) -> str:
    """The LLM-facing cast block; empty string when the channel has no cast.

    used=None  -> roster mode (script writing): every character, and the instruction to pick
                  only the ones this story needs.
    used=[...] -> video mode (every later call): full looks for the characters of this video
                  only, plus the names that must stay out.
    spoken     -> the call writes words a voice reads out (narration, script review, song lyrics):
                  names, kinds and personalities, no looks at all. A writer that never sees the
                  look cannot recite it into the audio; the visual calls keep the full sheet."""
    cast = active(channel)
    if not cast:
        return ""
    written = "names" if spoken else "names and looks"
    if used is None:
        return (f"RECURRING CAST of this channel (the roster - keep {written} EXACTLY as written):\n"
                + "\n".join(_sheet_line(c, spoken, rank=True) for c in by_priority(cast)) + "\n"
                "Pick only the characters this story needs (usually 1-3). MAIN characters are the usual "
                "protagonists; supporting characters appear when the story calls for them. Characters you "
                "do not pick stay out entirely - never name them. If the request names cast characters, use "
                "exactly those, in the roles the request gives them. "
                + _RANK_RULE(cast)
                + (SPOKEN_RULE if spoken else _LOOK_RULE))
    chosen = select_characters(channel, used)
    others = [c["name"] for c in cast if c not in chosen]
    if not chosen:
        return (f"This video features NONE of the channel's recurring cast ({', '.join(others)}): do not "
                "add any of them. ")
    return (f"CAST IN THIS VIDEO (keep {written} EXACTLY as written):\n"
            + "\n".join(_sheet_line(c, spoken) for c in chosen) + "\n"
            + (SPOKEN_RULE if spoken else
               "Whenever a scene shows one of them, name them and repeat their full look from this sheet; "
               "never rename them, never change their looks, never invent other recurring named "
               "characters. ")
            + (f"The channel's other cast members ({', '.join(others)}) do NOT appear in this video - do "
               "not add them. " if others else ""))


def mentioned(channel: dict | None, text: str) -> list:
    """Cast entries whose name occurs in text (whole word, case-insensitive)."""
    out = []
    low = text or ""
    for c in active(channel):
        if re.search(rf"(?<![\w]){re.escape(c['name'])}(?![\w])", low, flags=re.IGNORECASE):
            out.append(c)
    return out


def render_prefix(channel: dict | None, text: str, only: list | None = None) -> str:
    """Deterministic look lines for every character named in a ComfyUI prompt.
    only: restrict to these names (the video's cast_used), so a scene about "a daisy" does not
    drag in Daisy the fawn."""
    chars = mentioned(channel, text)
    if only is not None:
        allowed = {str(n).lower() for n in only}
        chars = [c for c in chars if c["name"].lower() in allowed]
    if not chars:
        return ""
    return "\n".join(f"{c['name']}: {c.get('look', '').strip()}" for c in chars if c.get("look", "").strip())


def looks_prefix(channel: dict | None, names: list | None) -> str:
    """`Name: look` lines for exactly the given cast names, whether or not the text names them
    (the thumbnail must show the video's characters even if the LLM wrote "the little fox")."""
    return "\n".join(f"{c['name']}: {c.get('look', '').strip()}"
                     for c in select_characters(channel, names) if c.get("look", "").strip())


def voice_id_for(channel: dict | None, name: str) -> str:
    """The reference voice configured for this character ("" = none, unknown, or characters off).

    Story videos speak every line in its speaker's own voice. The lookup goes through active(), so a
    channel with characters switched off resolves every speaker to the video's own voice, which is
    the same thing every other prompt-facing helper in this module does."""
    for c in select_characters(channel, [name]):
        return str(c.get("voice_id") or "").strip()
    return ""


def with_prefix(channel: dict | None, text: str, only: list | None = None) -> str:
    """text with the look prefix prepended when a cast character is named in it."""
    p = render_prefix(channel, text, only=only)
    return f"{p}\n\n{text}" if p else text


def portrait_prompt(channel: dict, c: dict) -> str:
    style = ""
    if channel.get("video_type") not in config.REALISTIC_VIDEO_TYPES:
        style = config.ART_STYLES.get((channel.get("defaults") or {}).get("art_style", ""), "")
    return (f"{style}\n\n{c['name']}: {c.get('look', '').strip()}\n\n"
            f"Character reference shot of {c['name']} alone: full body, centered, facing the camera with a "
            "friendly expression, simple soft uncluttered background, even lighting, nothing else in frame. "
            "No text, letters, logos or watermarks.").strip()


def opening_character(channel: dict | None, channel_id: str, used: list | None = None):
    """The character whose approved portrait opens this video, or None.

    used: the video's cast (plan["cast_used"], first-mention order) -> the first of them with a
    portrait; an empty list means no cast in this video -> None. used=None (plans written
    before cast selection existed) falls back to the first main character with a portrait."""
    if not channel or not channel.get("cast_opening", True) or not channel_id:
        return None
    pool = main_characters(channel) if used is None else select_characters(channel, used)
    for c in pool:
        if c.get("id") and channels.portrait_path(channel_id, c["id"]).exists():
            return c
    return None


# ---- portrait candidate generation ----------------------------------------
_state: dict = {}          # (cid, char_id) -> {"generating", "waiting", "done", "total", "errors"}
_lock = threading.Lock()


def candidates_state(cid: str, char_id: str) -> dict:
    st = dict(_state.get((cid, char_id)) or {"generating": False, "waiting": False,
                                              "done": 0, "total": 0, "errors": 0})
    st["candidates"] = [k for k in range(8) if channels.candidate_path(cid, char_id, k).exists()]
    return st


def start_candidates(cid: str, char_id: str, count: int = 3) -> bool:
    """Render `count` portrait candidates in a background thread. False if already running."""
    count = max(1, min(6, int(count)))
    with _lock:
        if (_state.get((cid, char_id)) or {}).get("generating"):
            return False
        _state[(cid, char_id)] = {"generating": True, "waiting": False, "done": 0,
                                  "total": count, "errors": 0}
    threading.Thread(target=_generate, args=(cid, char_id, count), daemon=True).start()
    return True


def _generate(cid: str, char_id: str, count: int):
    from services.comfy import ComfyClient
    st = _state[(cid, char_id)]
    try:
        ch = channels.get_channel(cid) or {}
        # the RAW roster on purpose: the editor renders portraits even while characters are off,
        # so the channel can be switched back on with its reference shots intact
        c = next((x for x in all_characters(ch) if x.get("id") == char_id), None)
        if c is None:
            return
        for k in range(8):   # start clean
            channels.candidate_path(cid, char_id, k).unlink(missing_ok=True)
        client = ComfyClient()
        template = client.load_workflow(config.T2V_WORKFLOW)
        aspect = (ch.get("defaults") or {}).get("aspect_ratio") or config.DEFAULT_ASPECT_RATIO
        w, h = config.THUMB_SIZES.get(aspect, (1280, 720))
        prompt = portrait_prompt(ch, c)
        for k in range(count):
            raw = config.CHANNELS_DIR / f"{cid}_cast_{char_id}_cand_{k}_raw.mp4"
            try:
                if not schedule.is_open():
                    st["waiting"] = True
                    schedule.wait_until_open()
                    st["waiting"] = False
                wf = client.build_thumbnail(template, prompt, f"videogen/_cast/{cid}_{char_id}_{k}",
                                            seed=random.randint(0, 2**53), aspect_ratio=aspect)
                client.run(wf, raw, cleanup=config.COMFY_CLEANUP)
                video.make_thumbnail(raw, channels.candidate_path(cid, char_id, k), w, h)
                st["done"] += 1
            except Exception:  # noqa: BLE001 - skip this candidate, keep going
                st["errors"] += 1
            finally:
                raw.unlink(missing_ok=True)
    finally:
        st["generating"] = False
        st["waiting"] = False
        st["finished"] = time.time()


def approve_candidate(cid: str, char_id: str, k: int):
    """Promote candidate k to the character's portrait (PNG) and drop the other candidates."""
    from PIL import Image
    src = channels.candidate_path(cid, char_id, k)
    if not src.exists():
        raise FileNotFoundError("candidate not found")
    img = Image.open(src).convert("RGB")
    img.save(channels.portrait_path(cid, char_id), "PNG")
    for j in range(8):
        channels.candidate_path(cid, char_id, j).unlink(missing_ok=True)
