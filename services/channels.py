"""Channel library: each channel has a name, purpose, audience, video type, per-video
defaults, an optional logo and its own YouTube OAuth token.

The channel decides how content is generated (tone/safety rules via the audience preset,
animated / realistic / tech look via video_type, prompt framing via the purpose text, the
story-board overlay default) and which
YouTube account the finished video is uploaded to. Index: channels/channels.json;
per-channel files: channels/<id>_logo.png, channels/<id>_yt_token.json and the cast
portraits channels/<id>_cast_<char>.png (see services/cast.py).
"""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import config

_lock = threading.Lock()

MAX_NAME_CHARS = 80
MAX_PURPOSE_CHARS = 2000
MAX_CAST = 8                  # recurring characters per channel
MAX_CAST_NAME_CHARS = 40
MAX_CAST_KIND_CHARS = 60
CAST_ROLES = ("main", "supporting")   # look / personality have no length cap: detail helps consistency

DEFAULT_KEYS = ("art_style", "audio_mode", "voice_id", "aspect_ratio", "video_length")
BOOL_DEFAULT_KEYS = ("storyboard",)   # story board overlay on by default for this channel's videos


def _read_index() -> dict:
    try:
        data = json.loads(config.CHANNELS_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("channels"), list):
            for c in data["channels"]:  # records created before the flags existed
                if isinstance(c, dict):
                    c.setdefault("made_for_kids", c.get("audience") == "kids")
                    c.setdefault("defaults", {}).setdefault("storyboard", c.get("video_type") == "tech")
                    c.setdefault("cast", [])
                    c.setdefault("cast_opening", True)
            return data
    except Exception:  # noqa: BLE001  (missing or corrupt index -> empty store)
        pass
    return {"channels": []}


def _write_index(data: dict):
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHANNELS_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_channels() -> list:
    return _read_index()["channels"]


def get_channel(cid: str):
    for c in list_channels():
        if c["id"] == cid:
            return c
    return None


def logo_path(cid: str) -> Path:
    return config.CHANNELS_DIR / f"{cid}_logo.png"


def token_path(cid: str) -> Path:
    return config.CHANNELS_DIR / f"{cid}_yt_token.json"


def secret_path(cid: str) -> Path:
    """The channel's own OAuth client file; missing = fall back to the shared one."""
    return config.CHANNELS_DIR / f"{cid}_client_secret.json"


def portrait_path(cid: str, char_id: str) -> Path:
    """Approved reference portrait of a cast character."""
    return config.CHANNELS_DIR / f"{cid}_cast_{char_id}.png"


def candidate_path(cid: str, char_id: str, k: int) -> Path:
    return config.CHANNELS_DIR / f"{cid}_cast_{char_id}_cand_{k}.jpg"


def _remove_character_files(cid: str, char_id: str):
    for p in config.CHANNELS_DIR.glob(f"{cid}_cast_{char_id}*"):
        try:
            p.unlink()
        except OSError:
            pass


def snapshot(rec: dict) -> dict:
    """The subset of a channel frozen onto each job, so prompts stay stable even if the
    channel is edited or deleted while the job is queued."""
    return {"name": rec["name"], "purpose": rec["purpose"],
            "audience": rec["audience"], "video_type": rec["video_type"],
            "made_for_kids": rec.get("made_for_kids", rec["audience"] == "kids"),
            "cast": [dict(c) for c in rec.get("cast") or []],
            "cast_opening": bool(rec.get("cast_opening", True))}


def _validate_cast(items, base_cast: list) -> list:
    """Normalize the cast list; ids of existing characters are kept, new ones get an id."""
    if not isinstance(items, list):
        raise ValueError("cast must be a list")
    if len(items) > MAX_CAST:
        raise ValueError(f"at most {MAX_CAST} cast characters per channel")
    known = {c.get("id") for c in base_cast if isinstance(c, dict)}
    out, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("each cast character must be an object")
        c = {"id": str(it.get("id") or "").strip(),
             "name": str(it.get("name") or "").strip(),
             "kind": str(it.get("kind") or "").strip(),
             "role": str(it.get("role") or "main").strip().lower(),
             "look": str(it.get("look") or "").strip(),
             "personality": str(it.get("personality") or "").strip()}
        if not c["id"] or c["id"] not in known or c["id"] in seen:
            c["id"] = uuid.uuid4().hex[:8]
        seen.add(c["id"])
        if not c["name"] or len(c["name"]) > MAX_CAST_NAME_CHARS:
            raise ValueError(f"cast: each character needs a name of 1-{MAX_CAST_NAME_CHARS} characters")
        if len(c["kind"]) > MAX_CAST_KIND_CHARS:
            raise ValueError(f"cast: kind must be at most {MAX_CAST_KIND_CHARS} characters")
        if c["role"] not in CAST_ROLES:
            raise ValueError(f"cast: role must be one of {list(CAST_ROLES)}")
        if not c["look"]:
            raise ValueError(f"cast: {c['name']} needs a look description")
        out.append(c)
    names = [c["name"].lower() for c in out]
    if len(set(names)) != len(names):
        raise ValueError("cast: character names must be unique")
    return out


def _validate(data: dict, base: dict | None = None) -> dict:
    """Merge data over base (or built-in defaults) and validate every field."""
    rec = {"name": "", "purpose": "", "audience": "general", "video_type": "animated",
           "made_for_kids": False, "cast": [], "cast_opening": True,
           "defaults": {"art_style": "", "audio_mode": config.AUDIO_MODE, "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": config.DEFAULT_VIDEO_LENGTH, "storyboard": None}}
    if base:
        for k in ("name", "purpose", "audience", "video_type", "made_for_kids", "cast", "cast_opening"):
            if k in base:
                rec[k] = base[k]
        rec["defaults"].update(base.get("defaults") or {})
    for k in ("name", "purpose", "audience", "video_type"):
        if k in data:
            rec[k] = str(data[k] or "").strip()
    if "made_for_kids" in data:
        rec["made_for_kids"] = bool(data["made_for_kids"])
    if "cast_opening" in data:
        rec["cast_opening"] = bool(data["cast_opening"])
    if "cast" in data:
        rec["cast"] = _validate_cast(data["cast"], (base or {}).get("cast") or [])
    d = data.get("defaults") or {}
    for k in DEFAULT_KEYS:
        if k in d:
            rec["defaults"][k] = str(d[k] or "").strip()
    for k in BOOL_DEFAULT_KEYS:
        if k in d and d[k] is not None:
            rec["defaults"][k] = bool(d[k])
    if not rec["name"] or len(rec["name"]) > MAX_NAME_CHARS:
        raise ValueError(f"name must be 1-{MAX_NAME_CHARS} characters")
    if len(rec["purpose"]) > MAX_PURPOSE_CHARS:
        raise ValueError(f"purpose must be at most {MAX_PURPOSE_CHARS} characters")
    if rec["audience"] not in config.AUDIENCES:
        raise ValueError(f"audience must be one of {config.AUDIENCES}")
    if rec["video_type"] not in config.VIDEO_TYPES:
        raise ValueError(f"video_type must be one of {config.VIDEO_TYPES}")
    df = rec["defaults"]
    if df.get("storyboard") is None:   # unset -> on for tech channels, off otherwise
        df["storyboard"] = rec["video_type"] == "tech"
    if rec["video_type"] in config.REALISTIC_VIDEO_TYPES:
        df["art_style"] = ""  # realistic channels use the built-in live-action direction
    elif df["art_style"] and df["art_style"] not in config.ART_STYLES:
        raise ValueError(f"art_style must be one of {list(config.ART_STYLES)}")
    if df["audio_mode"] and df["audio_mode"] not in config.AUDIO_MODES:
        raise ValueError(f"audio_mode must be one of {config.AUDIO_MODES}")
    if df["aspect_ratio"] and df["aspect_ratio"] not in config.ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of {config.ASPECT_RATIOS}")
    if df["video_length"] and df["video_length"] not in config.VIDEO_LENGTHS:
        raise ValueError(f"video_length must be one of {list(config.VIDEO_LENGTHS)}")
    if df["audio_mode"] == "music":
        df["voice_id"] = ""  # a reference voice is meaningless for songs
    return rec


def add_channel(data: dict) -> dict:
    rec = _validate(data)
    rec["id"] = uuid.uuid4().hex[:8]
    rec["created"] = time.time()
    with _lock:
        idx = _read_index()
        idx["channels"].append(rec)
        _write_index(idx)
    return rec


def update_channel(cid: str, data: dict) -> dict:
    with _lock:
        idx = _read_index()
        for i, c in enumerate(idx["channels"]):
            if c["id"] == cid:
                rec = _validate(data, base=c)
                rec["id"], rec["created"] = c["id"], c.get("created", time.time())
                idx["channels"][i] = rec
                _write_index(idx)
                kept = {x["id"] for x in rec["cast"]}
                dropped = [x["id"] for x in (c.get("cast") or []) if x.get("id") not in kept]
                break
        else:
            raise KeyError(cid)
    for char_id in dropped:   # portraits of removed characters
        _remove_character_files(cid, char_id)
    return rec


def delete_channel(cid: str):
    with _lock:
        idx = _read_index()
        keep = [c for c in idx["channels"] if c["id"] != cid]
        if len(keep) == len(idx["channels"]):
            raise KeyError(cid)
        if not keep:
            raise ValueError("cannot delete the last channel - add another one first")
        idx["channels"] = keep
        _write_index(idx)
    logo_path(cid).unlink(missing_ok=True)
    token_path(cid).unlink(missing_ok=True)
    secret_path(cid).unlink(missing_ok=True)
    _remove_character_files(cid, "")   # every <cid>_cast_* portrait and candidate


def ensure_default_channel():
    """One-time migration: seed the store from the legacy single-channel settings.

    Reads settings.json raw (the legacy CHANNEL_NAME / DEFAULT_* / AUDIO_MODE /
    YT_MADE_FOR_KIDS keys are no longer in EDITABLE_SETTINGS, so config skips them)
    and copies the legacy logo and YouTube token so nothing is lost.
    """
    if list_channels():
        return None
    raw = {}
    try:
        raw = json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        raw = {}

    def val(key, default, allowed=None):
        v = raw.get(key, default)
        return v if (allowed is None or v in allowed) else default

    rec = add_channel({
        "name": str(val("CHANNEL_NAME", "") or "").strip() or config.CHANNEL_NAME,
        "purpose": "",
        "audience": "kids" if val("YT_MADE_FOR_KIDS", config.YT_MADE_FOR_KIDS) else "general",
        "made_for_kids": bool(val("YT_MADE_FOR_KIDS", config.YT_MADE_FOR_KIDS)),
        "video_type": "animated",
        "defaults": {
            "art_style": val("DEFAULT_ART_STYLE", config.DEFAULT_ART_STYLE, config.ART_STYLES),
            "audio_mode": val("AUDIO_MODE", config.AUDIO_MODE, config.AUDIO_MODES),
            "voice_id": "",
            "aspect_ratio": val("DEFAULT_ASPECT_RATIO", config.DEFAULT_ASPECT_RATIO, config.ASPECT_RATIOS),
            "video_length": val("DEFAULT_VIDEO_LENGTH", config.DEFAULT_VIDEO_LENGTH, config.VIDEO_LENGTHS),
        },
    })
    # copy (not move): the legacy files stay as fallbacks for pre-existing jobs
    if config.LOGO_FILE.exists():
        shutil.copyfile(config.LOGO_FILE, logo_path(rec["id"]))
    if config.YT_TOKEN_FILE.exists():
        shutil.copyfile(config.YT_TOKEN_FILE, token_path(rec["id"]))
    return rec
