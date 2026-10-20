"""The GPU lease: which job currently owns the render hardware.

One job at a time may submit generation work - ComfyUI renders (clips, music, thumbnails, story
board images, presenter segments) and Fish TTS, which share the same GPU. Everything else a job
does - the vLLM calls, whisper timing, the transcript fetch, ffmpeg assembly - needs no lease and
runs while another job is rendering, which is the whole point of holding one.

The holder is a JOB id, not a thread: a job takes the lease at its first generation step and keeps
it until its run ends, across however many submissions and however many threads look at it. Taking
a lease the same job already holds is a no-op, so the gate can sit in front of every submission.

This is independent of services/schedule.py: the window says WHEN the GPU may be used at all, the
lease says WHO is using it right now. A job can hold the lease inside a closed window (TTS does)
and can be refused the lease inside an open one."""
import threading
import time

_lock = threading.Lock()
_holder: str | None = None


def acquire(job_id: str) -> bool:
    """Take the lease for `job_id`. True if it now holds it (already holding it counts)."""
    global _holder
    if not job_id:
        return False
    with _lock:
        if _holder is None or _holder == job_id:
            _holder = job_id
            return True
        return False


def release(job_id: str):
    """Give the lease back. A job that does not hold it releases nothing - a parked job calling
    this on its way out must not take the GPU away from the job that refused it."""
    global _holder
    with _lock:
        if _holder == job_id:
            _holder = None


def holder() -> str | None:
    with _lock:
        return _holder


def busy() -> bool:
    return holder() is not None


def held_by(job_id: str) -> bool:
    return bool(job_id) and holder() == job_id


def wait_until_free(job_id: str, should_stop=None, on_wait=None, poll: float = 5.0) -> bool:
    """Block until `job_id` holds the lease (True) or should_stop() returns True (False).

    For the work that bypasses the job queue - thumbnail regeneration, art-style previews, cast
    portraits - which has no queue slot to be parked in and so waits rather than raising. Mirrors
    schedule.wait_until_open, and is normally called right after it. on_wait(holder_id) is called
    when the wait starts and whenever the holder changes."""
    last = object()
    while not acquire(job_id):
        if should_stop is not None and should_stop():
            return False
        cur = holder()
        if on_wait is not None and cur != last:
            last = cur
            on_wait(cur)
        time.sleep(poll)
    return True


def reset():
    """Drop the lease whoever holds it. Tests only - a live release is per job id on purpose."""
    global _holder
    with _lock:
        _holder = None
