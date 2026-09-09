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


def main_characters(channel: dict | None) -> list:
    return [c for c in all_characters(channel) if c.get("role", "main") == "main"]


def names(chars: list) -> list:
    return [c["name"] for c in chars]


def used_characters(channel: dict | None, *texts: str) -> list:
    """Cast entries named in the texts (whole word, CASE-SENSITIVE), ordered by first mention.

    Case-sensitive on purpose: story text capitalises names, while "a penny", "a daisy" or
    "pip" are ordinary words that must not pull Penny, Daisy or Pip into a video."""
    blob = "\n".join(t or "" for t in texts)
    hits = []
    for c in all_characters(channel):
        m = re.search(rf"(?<![\w]){re.escape(c['name'])}(?![\w])", blob)
        if m:
            hits.append((m.start(), c))
    return [c for _, c in sorted(hits, key=lambda x: x[0])]


def select_characters(channel: dict | None, used) -> list:
    """The cast entries whose names are in `used` (any case), in `used` order."""
    by_name = {c["name"].lower(): c for c in all_characters(channel)}
    out, seen = [], set()
    for n in used or []:
        c = by_name.get(str(n).strip().lower())
        if c and c["name"] not in seen:
            seen.add(c["name"])
            out.append(c)
    return out


# ---- prompt text -----------------------------------------------------------
def _sheet_line(c: dict) -> str:
    role = "MAIN" if c.get("role", "main") == "main" else "supporting"
    kind = (c.get("kind") or "").strip()
    line = f"- {c['name']} ({role}{', ' + kind if kind else ''}): {c.get('look', '').strip()}"
    if (c.get("personality") or "").strip():
        line += f" Personality: {c['personality'].strip()}"
    return line


def sheet_text(channel: dict | None, used: list | None = None) -> str:
    """The LLM-facing cast block; empty string when the channel has no cast.

    used=None  -> roster mode (script writing): every character, and the instruction to pick
                  only the ones this story needs.
    used=[...] -> video mode (every later call): full looks for the characters of this video
                  only, plus the names that must stay out."""
    cast = all_characters(channel)
    if not cast:
        return ""
    if used is None:
        return ("RECURRING CAST of this channel (the roster - keep names and looks EXACTLY as written):\n"
                + "\n".join(_sheet_line(c) for c in cast) + "\n"
                "Pick only the characters this story needs (usually 1-3). MAIN characters are the usual "
                "protagonists; supporting characters appear when the story calls for them. Characters you "
                "do not pick stay out entirely - never name them. If the request names cast characters, use "
                "exactly those, in the roles the request gives them. Never invent other recurring named "
                "characters, never rename these ones and never change their looks. Whenever a scene shows a "
                "cast character, name them and repeat their full look from this sheet. ")
    chosen = select_characters(channel, used)
    others = [c["name"] for c in cast if c not in chosen]
    if not chosen:
        return (f"This video features NONE of the channel's recurring cast ({', '.join(others)}): do not "
                "add any of them. ")
    return ("CAST IN THIS VIDEO (keep names and looks EXACTLY as written):\n"
            + "\n".join(_sheet_line(c) for c in chosen) + "\n"
            "Whenever a scene shows one of them, name them and repeat their full look from this sheet; "
            "never rename them, never change their looks, never invent other recurring named characters. "
            + (f"The channel's other cast members ({', '.join(others)}) do NOT appear in this video - do "
               "not add them. " if others else ""))


def mentioned(channel: dict | None, text: str) -> list:
    """Cast entries whose name occurs in text (whole word, case-insensitive)."""
    out = []
    low = text or ""
    for c in all_characters(channel):
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
