"""Staging area for a music track the user uploads instead of having one generated.

One-shot per video: POST /api/music-uploads validates and normalizes the file, POST /jobs
carries the returned id, and job creation copies the track to output/<job_id>/music.mp3 and
consumes the upload. Staged files live in output/_uploads/<id>.mp3 with an <id>.json sidecar
(owner, name, duration) - inside the already mapped output directory, so no new Docker volume.
Abandoned uploads are swept after TTL_SECONDS.

The track IS the song here: it is never trimmed, padded or re-timed, and its measured length
decides how many clips the video has (config.clip_durations at the music stage).
"""
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import config
from services.ffbin import ffmpeg
from services.tts import audio_duration

MAX_BYTES = 60 * 1024 * 1024    # ~6 min in any compressed format; a 6-min wav is meant to be rejected
MIN_SECONDS = 10.0              # below this the video is one or two clips and the timing is meaningless
MAX_SECONDS = config.MAX_SONG_SECONDS   # 360s: the same ceiling a generated song has
TTL_SECONDS = 24 * 3600         # abandoned uploads are swept after a day
MAX_PER_OWNER = 5               # pending uploads kept per user; the oldest are dropped first
ID_RE = re.compile(r"^[0-9a-f]{8}$")
MAX_NAME_CHARS = 120


def _dir() -> Path:
    """Resolved per call, never at import: the tests relocate config.OUTPUT_DIR at runtime
    (Job.dir is a property for the same reason)."""
    return config.OUTPUT_DIR / "_uploads"


def path(upload_id: str) -> Path:
    return _dir() / f"{upload_id}.mp3"


def _sidecar(upload_id: str) -> Path:
    return _dir() / f"{upload_id}.json"


def _clean_name(orig: str) -> str:
    """The client's filename, kept for display only - it is never a path component."""
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(orig or "").name).strip() or "track.mp3"
    return name[:MAX_NAME_CHARS]


def get(upload_id: str):
    """The staged record, or None when the id is unknown, expired or malformed."""
    uid = (upload_id or "").strip()
    if not ID_RE.match(uid):        # also the path-traversal guard
        return None
    try:
        rec = json.loads(_sidecar(uid).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing or corrupt sidecar = no such upload
        return None
    return rec if isinstance(rec, dict) and rec.get("id") == uid else None


def discard(upload_id: str):
    if ID_RE.match((upload_id or "").strip()):
        path(upload_id).unlink(missing_ok=True)
        _sidecar(upload_id).unlink(missing_ok=True)


def install(upload_id: str, dest: Path):
    """Copy the staged track to the job folder, then consume the upload. Copy-then-delete
    rather than move: a failed copy leaves the staged file in place for a retry."""
    src = path(upload_id)
    if not get(upload_id) or not src.exists():
        raise ValueError("that uploaded music track is no longer available - upload it again")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    discard(upload_id)


def sweep(ttl: float = TTL_SECONDS) -> int:
    """Delete staged tracks older than `ttl` plus any audio left without a sidecar."""
    d = _dir()
    if not d.exists():
        return 0
    now, gone = time.time(), 0
    for side in d.glob("*.json"):
        try:
            created = float(json.loads(side.read_text(encoding="utf-8")).get("created") or 0.0)
        except Exception:  # noqa: BLE001 - corrupt sidecar: treat as garbage
            created = 0.0
        if now - created > ttl:
            discard(side.stem)
            gone += 1
    for mp3 in d.glob("*.mp3"):
        if not _sidecar(mp3.stem).exists():
            mp3.unlink(missing_ok=True)
            gone += 1
    return gone


def _enforce_quota(owner: str):
    """At most MAX_PER_OWNER pending uploads per user; the oldest go first."""
    mine = []
    for side in _dir().glob("*.json"):
        rec = get(side.stem)
        if rec and rec.get("owner") == owner:
            mine.append(rec)
    mine.sort(key=lambda r: float(r.get("created") or 0.0))
    for rec in mine[:max(0, len(mine) - (MAX_PER_OWNER - 1))]:
        discard(rec["id"])


def stage(data: bytes, owner: str, orig_filename: str = "") -> dict:
    """Validate, normalize and stage an uploaded music track.

    Rejected rather than repaired: a track that is not the song the user picked is worse than
    no video, so a file outside the length bounds is refused instead of being trimmed to fit.
    """
    if len(data) > MAX_BYTES:
        raise ValueError(f"music file too large (max {MAX_BYTES // (1024 * 1024)} MB)")
    sweep()
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    _enforce_quota(owner)
    uid = uuid.uuid4().hex[:8]
    mp3 = path(uid)
    suffix = Path(orig_filename or "upload.bin").suffix[:10] or ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"upload{suffix}"
        src.write_bytes(data)
        # decode + normalize; fails if the upload is not decodable audio. -vn drops embedded
        # cover art (an mp3 with artwork carries a video stream) and -map_metadata -1 the tags.
        r = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vn",
                            "-map_metadata", "-1", "-ar", "44100", "-ac", "2",
                            "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                           capture_output=True, text=True)
    if r.returncode != 0 or not mp3.exists():
        mp3.unlink(missing_ok=True)
        raise ValueError("not a valid audio file")
    try:
        seconds = audio_duration(mp3)
    except Exception:  # noqa: BLE001
        mp3.unlink(missing_ok=True)
        raise ValueError("not a valid audio file")
    if seconds < MIN_SECONDS:
        mp3.unlink(missing_ok=True)
        raise ValueError(f"music track is too short ({seconds:.1f}s, at least {MIN_SECONDS:.0f}s)")
    if seconds > MAX_SECONDS:
        mp3.unlink(missing_ok=True)
        raise ValueError(f"music track is too long ({seconds / 60:.1f} minutes, "
                         f"at most {MAX_SECONDS / 60:.0f} minutes)")
    rec = {"id": uid, "owner": owner, "name": _clean_name(orig_filename),
           "seconds": round(seconds, 2), "bytes": mp3.stat().st_size, "created": time.time()}
    _sidecar(uid).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


if __name__ == "__main__":     # smoke test: python -m services.music_upload
    with tempfile.TemporaryDirectory() as tmp:
        tone = Path(tmp) / "tone.mp3"
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=12", str(tone)], check=True)
        rec = stage(tone.read_bytes(), "smoke", "My Song.mp3")
        print("staged:", rec)
        dest = Path(tmp) / "music.mp3"
        install(rec["id"], dest)
        print("installed:", dest.stat().st_size, "bytes; consumed:", get(rec["id"]) is None)
        try:
            stage(b"not audio at all", "smoke", "x.mp3")
        except ValueError as e:
            print("rejected non-audio:", e)
