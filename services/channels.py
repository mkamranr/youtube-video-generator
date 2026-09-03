"""Channel library: each channel has a name, purpose, audience, video type, per-video
defaults, an optional logo and its own YouTube OAuth token.

The channel decides how content is generated (tone/safety rules via the audience preset,
animated vs realistic look via video_type, prompt framing via the purpose text) and which
YouTube account the finished video is uploaded to. Index: channels/channels.json;
per-channel files: channels/<id>_logo.png and channels/<id>_yt_token.json.
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

DEFAULT_KEYS = ("art_style", "audio_mode", "voice_id", "aspect_ratio", "video_length")


def _read_index() -> dict:
    try:
        data = json.loads(config.CHANNELS_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("channels"), list):
            for c in data["channels"]:  # records created before the flag existed
                if isinstance(c, dict):
                    c.setdefault("made_for_kids", c.get("audience") == "kids")
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


def snapshot(rec: dict) -> dict:
    """The subset of a channel frozen onto each job, so prompts stay stable even if the
    channel is edited or deleted while the job is queued."""
    return {"name": rec["name"], "purpose": rec["purpose"],
            "audience": rec["audience"], "video_type": rec["video_type"],
            "made_for_kids": rec.get("made_for_kids", rec["audience"] == "kids")}


def _validate(data: dict, base: dict | None = None) -> dict:
    """Merge data over base (or built-in defaults) and validate every field."""
    rec = {"name": "", "purpose": "", "audience": "general", "video_type": "animated",
           "made_for_kids": False,
           "defaults": {"art_style": "", "audio_mode": config.AUDIO_MODE, "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": config.DEFAULT_VIDEO_LENGTH}}
    if base:
        for k in ("name", "purpose", "audience", "video_type", "made_for_kids"):
            if k in base:
                rec[k] = base[k]
        rec["defaults"].update(base.get("defaults") or {})
    for k in ("name", "purpose", "audience", "video_type"):
        if k in data:
            rec[k] = str(data[k] or "").strip()
    if "made_for_kids" in data:
        rec["made_for_kids"] = bool(data["made_for_kids"])
    d = data.get("defaults") or {}
    for k in DEFAULT_KEYS:
        if k in d:
            rec["defaults"][k] = str(d[k] or "").strip()
    if not rec["name"] or len(rec["name"]) > MAX_NAME_CHARS:
        raise ValueError(f"name must be 1-{MAX_NAME_CHARS} characters")
    if len(rec["purpose"]) > MAX_PURPOSE_CHARS:
        raise ValueError(f"purpose must be at most {MAX_PURPOSE_CHARS} characters")
    if rec["audience"] not in config.AUDIENCES:
        raise ValueError(f"audience must be one of {config.AUDIENCES}")
    if rec["video_type"] not in config.VIDEO_TYPES:
        raise ValueError(f"video_type must be one of {config.VIDEO_TYPES}")
    df = rec["defaults"]
    if rec["video_type"] == "real":
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
                return rec
    raise KeyError(cid)


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
