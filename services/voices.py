"""Reference-voice library for Fish Speech voice cloning.

Users upload short speech samples (plus the exact text spoken in them) on the Settings
page; one can then be picked per video to clone that voice for the narration. Uploads are
normalized to 44.1 kHz mono wav so browser playback and the Fish upload are always well
formed. Index: reference_audio/voices.json; audio files: reference_audio/<id>.wav.
"""
import json
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import config
from services.ffbin import ffmpeg
from services.tts import audio_duration

_lock = threading.Lock()

# Special voice_id values for the built-in (non-cloned) voice:
#   ""       -> default voice, kept the same for the whole video (the first narration chunk
#               becomes the reference for the following chunks)
#   "random" -> default voice, a new random speaker for every chunk (Fish Speech's raw behaviour)
RANDOM_VOICE = "random"

MIN_SECONDS = 1
MAX_SECONDS = 120
MAX_NAME_CHARS = 60
MAX_TEXT_CHARS = 2000


def _read_index() -> dict:
    try:
        data = json.loads(config.VOICES_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("voices"), list):
            return data
    except Exception:  # noqa: BLE001  (missing or corrupt index -> empty library)
        pass
    return {"voices": []}


def _write_index(data: dict):
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    config.VOICES_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_voices() -> list:
    return _read_index()["voices"]


def get_voice(voice_id: str):
    for v in list_voices():
        if v["id"] == voice_id:
            return v
    return None


def voice_audio_path(voice_id: str) -> Path:
    return config.VOICES_DIR / f"{voice_id}.wav"


def reference(voice_id: str):
    """(wav_path, transcript) for a usable library voice, or None.

    None for "" (the built-in voice), for RANDOM_VOICE, for an id whose record was deleted and for
    a record whose wav has gone missing - so every caller can treat "no clonable reference" as one
    case. Story videos resolve one of these per speaking character.
    """
    vid = (voice_id or "").strip()
    if not vid or vid == RANDOM_VOICE:
        return None
    v = get_voice(vid)
    if not v:
        return None
    path = voice_audio_path(vid)
    if not path.exists():
        return None
    return path, v.get("text", "")


def add_voice(name: str, text: str, data: bytes, orig_filename: str = "") -> dict:
    name = (name or "").strip()
    text = (text or "").strip()
    if not name or len(name) > MAX_NAME_CHARS:
        raise ValueError(f"name must be 1-{MAX_NAME_CHARS} characters")
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"reference text must be 1-{MAX_TEXT_CHARS} characters")
    voice_id = uuid.uuid4().hex[:8]
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    wav = voice_audio_path(voice_id)
    suffix = Path(orig_filename or "sample.bin").suffix or ".bin"
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"upload{suffix}"
        src.write_bytes(data)
        # decode + normalize; fails if the upload is not decodable audio
        r = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src),
                            "-vn", "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                           capture_output=True, text=True)
    if r.returncode != 0 or not wav.exists():
        wav.unlink(missing_ok=True)
        raise ValueError("not a valid audio file")
    try:
        duration = audio_duration(wav)
    except Exception:  # noqa: BLE001
        wav.unlink(missing_ok=True)
        raise ValueError("not a valid audio file")
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        wav.unlink(missing_ok=True)
        raise ValueError(f"sample must be {MIN_SECONDS}-{MAX_SECONDS} seconds long ({duration:.1f}s given)")
    rec = {"id": voice_id, "name": name, "text": text, "file": wav.name,
           "duration": round(duration, 2), "created": time.time()}
    with _lock:
        idx = _read_index()
        idx["voices"].append(rec)
        _write_index(idx)
    return rec


def delete_voice(voice_id: str):
    with _lock:
        idx = _read_index()
        keep = [v for v in idx["voices"] if v["id"] != voice_id]
        if len(keep) == len(idx["voices"]):
            raise KeyError(voice_id)
        idx["voices"] = keep
        _write_index(idx)
    voice_audio_path(voice_id).unlink(missing_ok=True)


if __name__ == "__main__":
    # smoke test: add a generated test tone, list, delete
    with tempfile.TemporaryDirectory() as d:
        tone = Path(d) / "tone.wav"
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=3", str(tone)], check=True)
        rec = add_voice("Smoke Test", "This is a test.", tone.read_bytes(), "tone.wav")
    print("added:", rec)
    print("listed:", [v["name"] for v in list_voices()])
    delete_voice(rec["id"])
    print("deleted:", rec["id"], "- remaining voices:", len(list_voices()))
