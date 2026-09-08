"""Channel cast: recurring branded characters (an elephant, a kid, ...) that appear across
all of a channel's videos.

Two mechanisms keep them consistent:
  * a character sheet injected into every LLM prompt (system prompt) so scripts and scene
    prompts use the same names, looks and personalities;
  * a deterministic look prefix added to every ComfyUI prompt that names a character, so the
    canonical description reaches the video model even when the LLM paraphrased it;
plus an approved reference portrait per character that seeds the first frame of clip 1.
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


def _label(c: dict) -> str:
    kind = (c.get("kind") or "").strip()
    return f"{c['name']} ({kind})" if kind else c["name"]


# ---- prompt text -----------------------------------------------------------
def sheet_text(channel: dict | None) -> str:
    """The LLM-facing cast block; empty string when the channel has no cast."""
    cast = all_characters(channel)
    if not cast:
        return ""
    lines = []
    for c in cast:
        role = "MAIN" if c.get("role", "main") == "main" else "supporting"
        line = f"- {c['name']} ({role}{', ' + c['kind'].strip() if (c.get('kind') or '').strip() else ''}): {c.get('look', '').strip()}"
        if (c.get("personality") or "").strip():
            line += f" Personality: {c['personality'].strip()}"
        lines.append(line)
    mains = [c["name"] for c in cast if c.get("role", "main") == "main"]
    rule_main = (f"The MAIN character(s) {', '.join(mains)} appear in EVERY video of this channel and are "
                 "named naturally in the narration; ") if mains else ""
    return ("RECURRING CAST of this channel - its branding, keep names and looks EXACTLY as written:\n"
            + "\n".join(lines) + "\n"
            + rule_main
            + "supporting characters appear only when the story calls for them. Never invent other "
            "recurring named characters, never rename these ones and never change their looks. Whenever "
            "a scene shows a cast character, name them and repeat their full look from this sheet. ")


def mentioned(channel: dict | None, text: str) -> list:
    """Cast entries whose name occurs in text (whole word, case-insensitive)."""
    out = []
    low = text or ""
    for c in all_characters(channel):
        if re.search(rf"(?<![\w]){re.escape(c['name'])}(?![\w])", low, flags=re.IGNORECASE):
            out.append(c)
    return out


def render_prefix(channel: dict | None, text: str) -> str:
    """Deterministic look lines for every character named in a ComfyUI prompt."""
    chars = mentioned(channel, text)
    if not chars:
        return ""
    return "\n".join(f"{c['name']}: {c.get('look', '').strip()}" for c in chars if c.get("look", "").strip())


def with_prefix(channel: dict | None, text: str) -> str:
    """text with the look prefix prepended when a cast character is named in it."""
    p = render_prefix(channel, text)
    return f"{p}\n\n{text}" if p else text


def portrait_prompt(channel: dict, c: dict) -> str:
    style = ""
    if channel.get("video_type") not in config.REALISTIC_VIDEO_TYPES:
        style = config.ART_STYLES.get((channel.get("defaults") or {}).get("art_style", ""), "")
    return (f"{style}\n\n{c['name']}: {c.get('look', '').strip()}\n\n"
            f"Character reference shot of {c['name']} alone: full body, centered, facing the camera with a "
            "friendly expression, simple soft uncluttered background, even lighting, nothing else in frame. "
            "No text, letters, logos or watermarks.").strip()


def opening_character(channel: dict | None, channel_id: str):
    """The main character whose approved portrait opens the videos, or None."""
    if not channel or not channel.get("cast_opening", True) or not channel_id:
        return None
    for c in main_characters(channel):
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
