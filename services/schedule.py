"""GPU schedule window: when enabled, ComfyUI renders may only be submitted between
GPU_WINDOW_START and GPU_WINDOW_END (server local time). LLM and TTS work is never gated.

This is *when* the GPU may be used. Which job is using it right now is the separate lease in
services/gpu.py, which also covers TTS - the two are checked one after the other.

All functions read `config` live, so a Settings save applies immediately. An overnight
window (start > end, e.g. 22:00-06:00) wraps past midnight; start == end counts as
"always open" (the per-key setting validators cannot compare the two values)."""
from datetime import datetime, timedelta
import time

import config


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_enabled() -> bool:
    return bool(config.GPU_SCHEDULE_ENABLED) and config.GPU_WINDOW_START != config.GPU_WINDOW_END


def is_open(now: datetime | None = None) -> bool:
    """True when ComfyUI work may be submitted right now."""
    if not is_enabled():
        return True
    now = now or datetime.now()
    t = now.hour * 60 + now.minute
    start, end = _minutes(config.GPU_WINDOW_START), _minutes(config.GPU_WINDOW_END)
    if start < end:
        return start <= t < end
    return t >= start or t < end        # overnight window


def _next_at(now: datetime, minutes: int) -> datetime:
    at = now.replace(hour=minutes // 60, minute=minutes % 60, second=0, microsecond=0)
    if at <= now:
        at += timedelta(days=1)
    return at


def next_open(now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    return now if is_open(now) else _next_at(now, _minutes(config.GPU_WINDOW_START))


def next_close(now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    return _next_at(now, _minutes(config.GPU_WINDOW_END))


def _fmt_delta(minutes: int) -> str:
    h, m = divmod(max(0, minutes), 60)
    if h and m:
        return f"{h} h {m} min"
    return f"{h} h" if h else f"{m} min"


def status(now: datetime | None = None) -> dict:
    """Snapshot for the API/UI."""
    now = now or datetime.now()
    enabled = is_enabled()
    open_ = is_open(now)
    d = {"enabled": enabled, "start": config.GPU_WINDOW_START, "end": config.GPU_WINDOW_END,
         "open": open_, "now": now.strftime("%H:%M"), "next_open": None, "next_close": None,
         "opens_in_min": 0, "closes_in_min": 0}
    if enabled:
        if open_:
            nc = next_close(now)
            d["next_close"] = nc.strftime("%H:%M")
            d["closes_in_min"] = int((nc - now).total_seconds() // 60)
        else:
            no = next_open(now)
            d["next_open"] = no.strftime("%H:%M")
            d["opens_in_min"] = int((no - now).total_seconds() // 60)
    return d


def describe_wait(now: datetime | None = None) -> str:
    """Detail text for work parked until the window opens."""
    s = status(now)
    if not s["enabled"] or s["open"]:
        return "waiting for the GPU window"
    return (f"waiting for the GPU window ({s['start']}-{s['end']}, opens at {s['next_open']} "
            f"in {_fmt_delta(s['opens_in_min'])})")


def wait_until_open(should_stop=None, on_wait=None, poll: float = 15.0) -> bool:
    """Block until the window is open (True) or should_stop() returns True (False).
    on_wait(status_dict) is called when the wait starts and whenever the minute changes."""
    last = None
    while not is_open():
        if should_stop is not None and should_stop():
            return False
        s = status()
        if on_wait is not None and s["now"] != last:
            last = s["now"]
            on_wait(s)
        time.sleep(poll)
    return True
