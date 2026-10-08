"""Channel library: each channel has a name, purpose, audience, video type, per-video
defaults, an optional logo and its own YouTube OAuth token.

The channel decides how content is generated (tone/safety rules via the audience preset,
animated / realistic / tech look via video_type, prompt framing via the purpose text and the
fixed style & rules text, whether its videos have characters at all via characters_enabled -
off means informative explainers with no cast - the recurring cast, the story-board overlay default) and which
YouTube account the finished video is uploaded to. Index: channels/channels.json;
per-channel files: channels/<id>_logo.png, channels/<id>_yt_token.json, the cast
portraits channels/<id>_cast_<char>.png (see services/cast.py) and the presenter
avatars channels/<id>_avatar_<aid>.png, which are lip-synced to the voiceover and
overlaid on the video rather than prompted into a scene.
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
MAX_RULES_CHARS = 6000        # channel style & rules text (e.g. the non-character sections of a character bible)
MAX_CAST = 24                 # recurring characters per channel (the LLM picks the ones each story needs)
MAX_CAST_NAME_CHARS = 40
MAX_CAST_KIND_CHARS = 60
CAST_ROLES = ("main", "supporting")   # look / personality have no length cap: detail helps consistency

MAX_CAST_VOICE_CHARS = 40     # a reference-voice id from services/voices.py
MAX_AVATARS = 12              # presenter portraits per channel; one is the default
MAX_AVATAR_NAME_CHARS = 40
DEFAULT_KEYS = ("art_style", "audio_mode", "voice_id", "aspect_ratio", "video_length", "song_voice",
                "video_mode", "character_voice", "avatar_id", "avatar_model", "avatar_coverage")
BOOL_DEFAULT_KEYS = ("storyboard", "captions", "polish")   # on by default for a tech channel


def _read_index() -> dict:
    try:
        data = json.loads(config.CHANNELS_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("channels"), list):
            for c in data["channels"]:  # records created before the flags existed
                if isinstance(c, dict):
                    c.setdefault("made_for_kids", c.get("audience") == "kids")
                    c.setdefault("defaults", {}).setdefault("storyboard", c.get("video_type") == "tech")
                    c["defaults"].setdefault("captions", c.get("video_type") == "tech")
                    c["defaults"].setdefault("polish", c.get("video_type") == "tech")
                    c["defaults"].setdefault("song_voice", default_song_voice(c.get("audience")))
                    c["defaults"].setdefault("video_mode", config.DEFAULT_VIDEO_MODE)
                    c["defaults"].setdefault("character_voice", "")
                    c.setdefault("cast", [])
                    for m in c["cast"]:         # characters created before they could have a voice
                        if isinstance(m, dict):
                            m.setdefault("voice_id", "")
                    c["defaults"].setdefault("avatar_id", "")
                    c["defaults"].setdefault("avatar_model", "")
                    c["defaults"].setdefault("avatar_coverage", config.DEFAULT_AVATAR_COVERAGE)
                    c.setdefault("avatars", [])
                    c.setdefault("cast_opening", True)
                    c.setdefault("characters_enabled", bool(c.get("cast")))
                    c.setdefault("rules", "")
                    c.setdefault("owner", "")   # user id; assigned by services.ownership when empty
            return data
    except Exception:  # noqa: BLE001  (missing or corrupt index -> empty store)
        pass
    return {"channels": []}


def default_song_voice(audience) -> str:
    """Kids channels sing with a child's voice unless told otherwise; others leave it to the model."""
    return "child" if audience == "kids" else "auto"


def _write_index(data: dict):
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHANNELS_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_channels(owner: str | None = None) -> list:
    """All channels, or only those owned by the given user id."""
    chans = _read_index()["channels"]
    if owner is None:
        return chans
    return [c for c in chans if c.get("owner") == owner]


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


def avatar_path(cid: str, avatar_id: str) -> Path:
    """A presenter portrait the videos lip-sync. Same shape as a cast portrait, but these
    are never prompted into a scene - they are animated and overlaid."""
    return config.CHANNELS_DIR / f"{cid}_avatar_{avatar_id}.png"


def _remove_character_files(cid: str, char_id: str):
    for p in config.CHANNELS_DIR.glob(f"{cid}_cast_{char_id}*"):
        try:
            p.unlink()
        except OSError:
            pass


def _remove_avatar_files(cid: str, avatar_id: str):
    for p in config.CHANNELS_DIR.glob(f"{cid}_avatar_{avatar_id}*"):
        try:
            p.unlink()
        except OSError:
            pass


def default_avatar(rec: dict) -> dict | None:
    """The avatar a new video starts with: the one flagged default, else the first."""
    items = [a for a in (rec.get("avatars") or []) if isinstance(a, dict)]
    return next((a for a in items if a.get("default")), items[0] if items else None)


def find_avatar(rec: dict, avatar_id: str) -> dict | None:
    return next((a for a in (rec.get("avatars") or [])
                 if isinstance(a, dict) and a.get("id") == avatar_id), None)


def snapshot(rec: dict) -> dict:
    """The subset of a channel frozen onto each job, so prompts stay stable even if the
    channel is edited or deleted while the job is queued."""
    return {"name": rec["name"], "purpose": rec["purpose"], "rules": str(rec.get("rules") or ""),
            "audience": rec["audience"], "video_type": rec["video_type"],
            "made_for_kids": rec.get("made_for_kids", rec["audience"] == "kids"),
            "cast": [{**c, "voice_id": str(c.get("voice_id") or "")}
                     for c in rec.get("cast") or []],
            # the voice every character without one of their own speaks with: a story video needs it
            # at synthesis time, long after the channel record is out of reach
            "character_voice": str((rec.get("defaults") or {}).get("character_voice") or ""),
            "cast_opening": bool(rec.get("cast_opening", True)),
            "characters_enabled": bool(rec.get("characters_enabled", bool(rec.get("cast")))),
            # the avatar roster travels with the job so a planned video can still resolve the
            # presenter it was set up with, even if the channel is edited before it runs
            "avatars": [dict(a) for a in rec.get("avatars") or []]}


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
             "personality": str(it.get("personality") or "").strip(),
             # reference voice this character speaks with in a story video ("" = one is cloned from
             # their first line and kept for that video). Deliberately NOT checked against the voice
             # library here: deleting a voice would otherwise make every channel using it unsavable,
             # and the job logs a fallback instead (pipeline resolves it at synthesis time).
             "voice_id": str(it.get("voice_id") or "").strip()}
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
        if len(c["voice_id"]) > MAX_CAST_VOICE_CHARS:
            raise ValueError(f"cast: {c['name']} has an invalid voice")
        if c["voice_id"] == "random":
            # "a new voice every time" is the one thing a recurring character must never be
            raise ValueError(f"cast: {c['name']} needs one steady voice, not the random voice")
        out.append(c)
    names = [c["name"].lower() for c in out]
    if len(set(names)) != len(names):
        raise ValueError("cast: character names must be unique")
    return out


def _validate_avatars(items, base_avatars: list) -> list:
    """Normalize the avatar list; ids of existing avatars are kept so their PNG still resolves.

    Exactly one avatar carries `default` whenever the list is non-empty - the editor can send
    none or several, and silently settling it here is better than rejecting a save over a radio
    button. Deleting the default promotes the next one the same way."""
    if not isinstance(items, list):
        raise ValueError("avatars must be a list")
    if len(items) > MAX_AVATARS:
        raise ValueError(f"at most {MAX_AVATARS} avatars per channel")
    known = {a.get("id") for a in base_avatars if isinstance(a, dict)}
    out, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("each avatar must be an object")
        a = {"id": str(it.get("id") or "").strip(),
             "name": str(it.get("name") or "").strip(),
             "default": bool(it.get("default"))}
        if not a["id"] or a["id"] not in known or a["id"] in seen:
            a["id"] = uuid.uuid4().hex[:8]
        seen.add(a["id"])
        if not a["name"] or len(a["name"]) > MAX_AVATAR_NAME_CHARS:
            raise ValueError(f"avatars: each one needs a name of 1-{MAX_AVATAR_NAME_CHARS} characters")
        out.append(a)
    names = [a["name"].lower() for a in out]
    if len(set(names)) != len(names):
        raise ValueError("avatars: names must be unique")
    if out and not any(a["default"] for a in out):
        out[0]["default"] = True
    seen_default = False
    for a in out:                      # keep only the first default
        if a["default"] and seen_default:
            a["default"] = False
        seen_default = seen_default or a["default"]
    return out


def _validate(data: dict, base: dict | None = None) -> dict:
    """Merge data over base (or built-in defaults) and validate every field."""
    rec = {"name": "", "purpose": "", "rules": "", "audience": "general", "video_type": "animated",
           "made_for_kids": False, "cast": [], "avatars": [], "cast_opening": True,
           "characters_enabled": False,
           "defaults": {"art_style": "", "audio_mode": config.AUDIO_MODE, "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": config.DEFAULT_VIDEO_LENGTH, "storyboard": None, "captions": None, "polish": None,
                        "song_voice": "", "video_mode": "", "character_voice": "",
                        "avatar_id": "", "avatar_model": "",
                        "avatar_coverage": config.DEFAULT_AVATAR_COVERAGE}}
    if base:
        for k in ("name", "purpose", "rules", "audience", "video_type", "made_for_kids", "cast",
                  "avatars", "cast_opening", "characters_enabled"):
            if k in base:
                rec[k] = base[k]
        rec["defaults"].update(base.get("defaults") or {})
    # the owner is server-assigned (add_channel / ownership migration) and never taken from a request
    rec["owner"] = str((base or {}).get("owner") or "")
    for k in ("name", "purpose", "rules", "audience", "video_type"):
        if k in data:
            rec[k] = str(data[k] or "").strip()
    if "made_for_kids" in data:
        rec["made_for_kids"] = bool(data["made_for_kids"])
    if "cast_opening" in data:
        rec["cast_opening"] = bool(data["cast_opening"])
    if "cast" in data:
        rec["cast"] = _validate_cast(data["cast"], (base or {}).get("cast") or [])
    if "avatars" in data:
        rec["avatars"] = _validate_avatars(data["avatars"], (base or {}).get("avatars") or [])
    if "characters_enabled" in data:
        rec["characters_enabled"] = bool(data["characters_enabled"])
    elif base is None:   # a new channel follows the migration rule: characters only if a cast was given
        rec["characters_enabled"] = bool(rec["cast"])
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
    if len(rec["rules"]) > MAX_RULES_CHARS:
        raise ValueError(f"rules must be at most {MAX_RULES_CHARS} characters")
    if rec["audience"] not in config.AUDIENCES:
        raise ValueError(f"audience must be one of {config.AUDIENCES}")
    if rec["video_type"] not in config.VIDEO_TYPES:
        raise ValueError(f"video_type must be one of {config.VIDEO_TYPES}")
    df = rec["defaults"]
    for key in BOOL_DEFAULT_KEYS:      # unset -> on for tech channels, off otherwise
        if df.get(key) is None:
            df[key] = rec["video_type"] == "tech"
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
    df["video_mode"] = df.get("video_mode") or config.DEFAULT_VIDEO_MODE
    if df["video_mode"] not in config.VIDEO_MODES:
        raise ValueError(f"video_mode must be one of {config.VIDEO_MODES}")
    # Demoted rather than rejected at the channel level: the audience is edited on the same form, so
    # rejecting would make a kids+story channel unsavable the moment it stops being a kids channel.
    # The per-video path (pipeline.resolve_video_mode) still raises, because there the user asked.
    if df["video_mode"] == "story" and rec["audience"] != "kids":
        df["video_mode"] = "song"
    if df["video_mode"] == "story":
        df["audio_mode"] = "narration+music"   # a story is told by its voiceover, never sung
    if df["character_voice"] == "random":
        # one voice per character is the whole point; a new one every line is the bug
        raise ValueError("the default character voice cannot be the random voice")
    if len(df["character_voice"]) > MAX_CAST_VOICE_CHARS:
        raise ValueError("invalid default character voice")
    if df["audio_mode"] == "music":
        df["voice_id"] = ""  # a reference voice is meaningless for songs
    # the song voice stays whatever the audio default is: the audio mode is chosen per video
    df["song_voice"] = df.get("song_voice") or default_song_voice(rec["audience"])
    if df["song_voice"] not in config.SONG_VOICES:
        raise ValueError(f"song_voice must be one of {config.SONG_VOICES}")
    # Avatar: "" engine means follow the global Settings default, so it is not an error.
    if df["avatar_model"] and df["avatar_model"] not in config.AVATAR_MODELS:
        raise ValueError(f"avatar_model must be one of {list(config.AVATAR_MODELS)}")
    df["avatar_coverage"] = df.get("avatar_coverage") or config.DEFAULT_AVATAR_COVERAGE
    if df["avatar_coverage"] not in config.AVATAR_COVERAGES:
        raise ValueError(f"avatar_coverage must be one of {list(config.AVATAR_COVERAGES)}")
    # A default pointing at a deleted avatar is cleared rather than rejected, then re-seeded
    # from the list, so the channel always names an avatar that actually has a PNG.
    if df["avatar_id"] and not any(a["id"] == df["avatar_id"] for a in rec["avatars"]):
        df["avatar_id"] = ""
    if not df["avatar_id"]:
        chosen = default_avatar(rec)
        df["avatar_id"] = chosen["id"] if chosen else ""
    return rec


def add_channel(data: dict, owner: str) -> dict:
    """Create a channel owned by the given user id ("" only for the legacy first-start seed,
    which the ownership migration assigns to the first admin)."""
    rec = _validate(data)
    rec["id"] = uuid.uuid4().hex[:8]
    rec["created"] = time.time()
    rec["owner"] = str(owner or "")
    with _lock:
        idx = _read_index()
        idx["channels"].append(rec)
        _write_index(idx)
    return rec


def assign_unowned(owner: str) -> int:
    """Give every channel without an owner to `owner`; returns how many changed."""
    n = 0
    with _lock:
        idx = _read_index()
        for c in idx["channels"]:
            if not c.get("owner"):
                c["owner"] = owner
                n += 1
        if n:
            _write_index(idx)
    return n


def reassign_owner(old: str, new: str) -> int:
    """Move every channel of user `old` to user `new` (used when an account is deleted)."""
    n = 0
    with _lock:
        idx = _read_index()
        for c in idx["channels"]:
            if c.get("owner") == old:
                c["owner"] = new
                n += 1
        if n:
            _write_index(idx)
    return n


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
                kept_av = {x["id"] for x in rec["avatars"]}
                dropped_av = [x["id"] for x in (c.get("avatars") or [])
                              if x.get("id") not in kept_av]
                break
        else:
            raise KeyError(cid)
    for char_id in dropped:   # portraits of removed characters
        _remove_character_files(cid, char_id)
    for avatar_id in dropped_av:   # images of removed avatars
        _remove_avatar_files(cid, avatar_id)
    return rec


def delete_channel(cid: str):
    with _lock:
        idx = _read_index()
        keep = [c for c in idx["channels"] if c["id"] != cid]
        if len(keep) == len(idx["channels"]):
            raise KeyError(cid)
        # zero channels is a valid state: a new user starts with none and creates their own
        idx["channels"] = keep
        _write_index(idx)
    logo_path(cid).unlink(missing_ok=True)
    token_path(cid).unlink(missing_ok=True)
    secret_path(cid).unlink(missing_ok=True)
    _remove_character_files(cid, "")   # every <cid>_cast_* portrait and candidate
    _remove_avatar_files(cid, "")      # every <cid>_avatar_* presenter portrait


def ensure_default_channel():
    """One-time migration: seed the store from the legacy single-channel settings.

    Reads settings.json raw (the legacy CHANNEL_NAME / DEFAULT_* / AUDIO_MODE /
    YT_MADE_FOR_KIDS keys are no longer in EDITABLE_SETTINGS, so config skips them)
    and copies the legacy logo and YouTube token so nothing is lost. Runs only while no
    channel index exists at all: once channels were ever saved (even if all of them were
    deleted later) nothing is seeded. The seeded channel has no owner yet - the ownership
    migration hands it to the first admin.
    """
    if config.CHANNELS_INDEX.exists():
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
    }, owner="")
    # copy (not move): the legacy files stay as fallbacks for pre-existing jobs
    if config.LOGO_FILE.exists():
        shutil.copyfile(config.LOGO_FILE, logo_path(rec["id"]))
    if config.YT_TOKEN_FILE.exists():
        shutil.copyfile(config.YT_TOKEN_FILE, token_path(rec["id"]))
    return rec
