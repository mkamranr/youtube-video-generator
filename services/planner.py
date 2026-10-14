"""Weekly content plans: seven video ideas for one channel, reviewed before anything is made.

A plan is written by the LLM (services.llm.write_week_plan) and then belongs to the user: rows
can be regenerated one by one, edited by hand, and confirmed into jobs. Confirming a row creates
a PAUSED job (pipeline status "planned") - the plan only decides *what* a video is about, the
generation parameters are still chosen on the job page.

Storage mirrors services/channels.py: one JSON index (plans/plans.json), a module lock, an
`owner` that is server-assigned and never read from a request. Plans are kept after their jobs
exist, so a channel has a history of the weeks it already covered.
"""
import json
import threading
import time
import uuid
from datetime import date, timedelta

import config
from services import cast

_lock = threading.Lock()

DAYS = 7                       # the week a plan covers, in days
MAX_VIDEOS = DAYS              # at most one video a day
MIN_VIDEOS = 2
DEFAULT_VIDEOS = DAYS          # what a plan with no stored count means: the daily week it was
MAX_TITLE_CHARS = 120
MAX_TEXT_CHARS = 1200          # description
MAX_LINE_CHARS = 400           # thumbnail idea / hook
MAX_NOTES_CHARS = 1000
MAX_ACTORS = 6
DAY_KEYS = ("title", "description", "thumbnail_idea", "hook")


def _read_index() -> dict:
    try:
        data = json.loads(config.PLANS_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("plans"), list):
            for p in data["plans"]:     # plans written before a week had a format
                if isinstance(p, dict):
                    p.setdefault("video_mode", config.DEFAULT_VIDEO_MODE)
                    # written before a week could hold fewer than seven: it held seven
                    p["videos"] = clamp_videos(p.get("videos", len(p.get("days") or []) or DAYS))
            return data
    except Exception:  # noqa: BLE001  (missing or corrupt index -> empty store)
        pass
    return {"plans": []}


def _write_index(data: dict):
    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLANS_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---- dates -----------------------------------------------------------------
def parse_week_start(value) -> str:
    """An ISO date for day 1; today when the value is missing or unreadable."""
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError:
        return date.today().isoformat()


def clamp_videos(value) -> int:
    """How many videos one week's plan holds (MIN_VIDEOS..MAX_VIDEOS).

    Anything unreadable - including a plan stored before the count existed - becomes the full
    daily week, so no existing plan changes shape when it is read back."""
    try:
        return max(MIN_VIDEOS, min(MAX_VIDEOS, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_VIDEOS


def day_date(week_start: str, day: int, videos: int = DAYS) -> str:
    """The date of row `day` when the week holds `videos` of them.

    Fewer than seven are SPREAD over the week rather than stacked on its first days: three videos
    go out Monday, Wednesday, Friday, not Monday to Wednesday followed by four silent days. The
    stride is integer division, so day 1 is always the week start and the last row always lands
    inside the week."""
    videos = clamp_videos(videos)
    stride = max(1, DAYS // videos)
    offset = (max(1, day) - 1) * stride
    return (date.fromisoformat(week_start) + timedelta(days=min(offset, DAYS - 1))).isoformat()


# ---- validation ------------------------------------------------------------
def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def validate_day(row, day: int, week_start: str, channel: dict | None = None,
                 base: dict | None = None, videos: int = DAYS) -> dict:
    """One plan row, clamped. `actors` keeps only names the channel's cast really has, so a
    hallucinated character never reaches the job - and stays empty altogether for a channel whose
    videos have no characters. Unknown keys are dropped."""
    row = row if isinstance(row, dict) else {}
    base = base or {}
    out = {"day": day, "date": day_date(week_start, day, videos)}
    limits = {"title": MAX_TITLE_CHARS, "description": MAX_TEXT_CHARS,
              "thumbnail_idea": MAX_LINE_CHARS, "hook": MAX_LINE_CHARS}
    for k in DAY_KEYS:
        out[k] = _clip(row.get(k, base.get(k, "")), limits[k])
    actors = row.get("actors", base.get("actors", []))
    if isinstance(actors, str):
        actors = [a for a in actors.replace(" and ", ",").split(",")]
    if not isinstance(actors, list):
        actors = []
    names = [str(a).strip() for a in actors if str(a).strip()]
    if channel and not cast.enabled(channel):
        out["actors"] = []      # this channel's videos have no characters: a row never carries any
    elif cast.active(channel):
        out["actors"] = cast.names(cast.select_characters(channel, names))[:MAX_ACTORS]
    else:
        out["actors"] = names[:MAX_ACTORS]   # no channel (orphaned plan): keep what was written
    out["job_id"] = str(row.get("job_id", base.get("job_id", "")) or "")
    return out


def normalize_days(rows, week_start: str, channel: dict | None = None, days: int = DAYS) -> list:
    """Exactly `days` rows, numbered and dated across the week from week_start. A short reply from
    the LLM is padded with empty rows (the user regenerates them); a long one is cut."""
    rows = rows if isinstance(rows, list) else []
    days = clamp_videos(days)
    out = []
    for i in range(days):
        out.append(validate_day(rows[i] if i < len(rows) else {}, i + 1, week_start, channel,
                                videos=days))
    return out


# ---- store -----------------------------------------------------------------
def list_plans(owner: str | None = None, channel_id: str | None = None) -> list:
    """Plans, newest first; filtered by owner and/or channel when given."""
    plans = _read_index()["plans"]
    if owner is not None:
        plans = [p for p in plans if p.get("owner") == owner]
    if channel_id:
        plans = [p for p in plans if p.get("channel_id") == channel_id]
    return sorted(plans, key=lambda p: p.get("created", 0), reverse=True)


def get_plan(pid: str):
    for p in _read_index()["plans"]:
        if p.get("id") == pid:
            return p
    return None


def create_plan(channel_id: str, owner: str, week_start: str, days: list, notes: str = "",
                video_mode: str = "", videos: int = DEFAULT_VIDEOS) -> dict:
    rec = {"id": uuid.uuid4().hex[:8], "channel_id": str(channel_id), "owner": str(owner or ""),
           "created": time.time(), "updated": time.time(),
           "week_start": week_start, "notes": _clip(notes, MAX_NOTES_CHARS),
           "video_mode": str(video_mode or config.DEFAULT_VIDEO_MODE),
           "videos": clamp_videos(videos), "days": days}
    with _lock:
        idx = _read_index()
        idx["plans"].append(rec)
        _write_index(idx)
    return rec


def _mutate(pid: str, fn) -> dict:
    """Apply fn to the stored plan under the lock and persist it."""
    with _lock:
        idx = _read_index()
        for i, p in enumerate(idx["plans"]):
            if p.get("id") == pid:
                fn(p)
                p["updated"] = time.time()
                idx["plans"][i] = p
                _write_index(idx)
                return p
    raise KeyError(pid)


def set_day(pid: str, day: int, row: dict) -> dict:
    """Replace one row; the day number, its date and an existing job_id are kept."""
    def apply(p):
        for i, d in enumerate(p.get("days") or []):
            if d.get("day") == day:
                p["days"][i] = {**row, "day": day, "date": d.get("date", row.get("date", "")),
                                "job_id": d.get("job_id", "")}
                return
        raise KeyError(f"day {day}")
    return _mutate(pid, apply)


def set_job(pid: str, day: int, job_id: str) -> dict:
    def apply(p):
        for d in p.get("days") or []:
            if d.get("day") == day:
                d["job_id"] = job_id
                return
        raise KeyError(f"day {day}")
    return _mutate(pid, apply)


def set_video_mode(pid: str, mode: str) -> dict:
    """The format every job confirmed from this plan is created in.

    A property of the whole plan, not of a row: rows are confirmed a few at a time, and a per-click
    choice would let one week come out as a mix of song and story videos."""
    def apply(p):
        p["video_mode"] = str(mode or config.DEFAULT_VIDEO_MODE)
    return _mutate(pid, apply)


def delete_plan(pid: str) -> bool:
    """Forget a plan. Jobs already created from it are untouched - they are real work."""
    with _lock:
        idx = _read_index()
        rest = [p for p in idx["plans"] if p.get("id") != pid]
        if len(rest) == len(idx["plans"]):
            return False
        idx["plans"] = rest
        _write_index(idx)
    return True


def reassign_owner(old: str, new: str) -> int:
    """Move every plan of user `old` to user `new` (used when an account is deleted)."""
    n = 0
    with _lock:
        idx = _read_index()
        for p in idx["plans"]:
            if p.get("owner") == old:
                p["owner"] = new
                n += 1
        if n:
            _write_index(idx)
    return n


def previous_titles(channel_id: str, limit: int = 40) -> list:
    """Titles this channel already planned, newest first - handed to the LLM so a new week
    does not repeat the last ones."""
    out = []
    for p in list_plans(channel_id=channel_id):
        for d in p.get("days") or []:
            if d.get("title"):
                out.append(d["title"])
            if len(out) >= limit:
                return out
    return out


def job_seed(row: dict) -> str:
    """The text a planned job is built from: the story itself. The hook travels separately, on
    Job.hook, and reaches the script writer as its own instruction (see pipeline._draft_script)."""
    return row.get("description", "").strip()
