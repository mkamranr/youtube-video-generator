"""Job model + the end-to-end generation pipeline."""
import json
import os
import re
import shutil
import stat
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config
from services import (avatar, browser, cast, chapters, channels, comfy, compilation, gpu, llm,
                      lyrics, music_upload, repo, resource, schedule, sfx, story, storyboard,
                      thumbtext, transcript, tts, video, voices)
from services import cardtemplates


PROMPT_SOURCES = ("topic", "description", "transcript")   # the user's text IS the source text
REPO_SOURCE = "repo"                    # a GitHub / Hugging Face URL: README + page screenshot
DRAFT_SOURCES = PROMPT_SOURCES + (REPO_SOURCE,)   # every source that drafts a script for approval   # user text drafted by the LLM, then approved


# statuses that are real pipeline work; `planned`, `queued`, `awaiting_approval`, `done`, `failed`
# only close the open one
STAGE_STATUSES = frozenset({"drafting", "transcript", "planning", "tts", "music", "scenes",
                            "thumbnail", "storyboard", "avatar", "clips", "assembling"})


@dataclass
class Job:
    id: str
    url: str
    aspect_ratio: str = config.DEFAULT_ASPECT_RATIO
    megapixels: float | None = None  # clip resolution; None (legacy jobs) = the current Settings default
    # which ComfyUI workflow renders the clips (config.VIDEO_MODELS); "" (legacy jobs) = the
    # current Settings default. Frozen once the video is planned: plan["clip_seconds"] holds the
    # real clip lengths as THIS model quantises them, and the two models disagree, so changing it
    # afterwards would slide the audio out from under the clips.
    video_model: str = ""
    resolution: str = ""            # final-video upscale target (config.OUTPUT_RESOLUTIONS); "" (legacy) = native
    # on-demand upscales of the finished final.mp4 (job page): [{label, file, width, height, created}];
    # `upscaling` holds the label of the one being rendered right now ("" = idle)
    upscales: list = field(default_factory=list)
    upscaling: str = ""
    audio_mode: str = ""
    # The CONTENT FORMAT of this video: "" (legacy) and "song" are every video made before story
    # videos existed; "story" means the LLM invents the story beats and the voiceover is only the
    # story's spoken lines. Not the channel's `video_type` (animated / real / tech), which is its
    # look - this is what the video IS. Kids channels with characters only; always narration+music.
    video_mode: str = ""
    voice_id: str = ""              # reference voice for Fish Speech cloning ("" = default voice)
    song_voice: str = ""            # singer of a generated song (config.SONG_VOICES); "" = legacy job, auto
    art_style: str = ""
    video_length: str = "Auto"
    lyrics: str = ""                # kids + Generate Song: the user's own song lyrics, sung verbatim
                                    # (they set the length, so video_length is forced to Auto)
    music_upload: str = ""          # Generate Song: the user's own music track instead of a generated
                                    # song - the uploaded file's name (its audio was copied to music.mp3
                                    # when the job was created, and it sets the length, so video_length
                                    # is forced to Auto). "" = the song is written and generated for you.
    music_seconds: float = 0.0      # measured length of that track, so the clip-length rung and the
                                    # story's word target need no disk access
    source: str = "youtube"         # youtube | topic | description | transcript | compilation
    topic: str = ""                 # the topic, description or pasted transcript text
    # COMPILATION ONLY (source == "compilation"): the job ids of the finished videos this one is
    # stitched from, in the order they play. A compilation renders nothing - it is a local ffmpeg
    # join of their final.mp4 files - but it is a real Job so that the upload, the bundle, the
    # downloads, the thumbnail variants, the archive and the delete all work on it unchanged.
    parts: list = field(default_factory=list)
    channel_id: str = ""            # channel this video belongs to ("" = legacy pre-channel job)
    owner: str = ""                 # user id of the job's owner (the channel's owner); "" only before the ownership migration
    channel: dict = field(default_factory=dict)  # snapshot: {name, purpose, rules, audience, video_type, cast, characters_enabled, ...}
    # Presenter avatar: which of the channel's uploaded portraits speaks this video, which
    # engine animates it (config.AVATAR_MODELS; "" = the Settings default) and where it
    # appears (config.AVATAR_COVERAGES; "none" is the off switch). Frozen at submit time like
    # every other choice, so editing the channel later cannot change a queued video.
    avatar_id: str = ""
    avatar_model: str = ""
    avatar_coverage: str = "none"
    storyboard: bool = False        # story board overlay: styled fact cards + related images on the video
    page_scroll: bool = False       # repo jobs: scroll the page screenshot as a panel over the video
    captions: bool = False          # burned-in word-synced subtitles over the video
    polish: bool = False            # hook card, end card, progress bar and the card whoosh
    thumb_text: bool = True         # draw the headline over this video's thumbnail
    thumb_variant: int = -1         # active thumbnail text variant (0 = plain render,
                                    # -1 = nobody has chosen, so the model's best line wins)
    script_approved: bool = False
    # weekly planner (services/planner.py): the plan row this job was created from. A planned job
    # rests at status "planned" until the user sets its parameters on the job page.
    plan_id: str = ""
    setup_done: bool = False        # the setup form was submitted, so the job left "planned"
    planned_day: int = 0
    planned_for: str = ""           # ISO date of the plan row
    hook: str = ""                  # opening line the plan chose
    thumb_idea: str = ""            # thumbnail the plan asked for
    status: str = "queued"          # planned|queued|transcript|planning|tts|scenes|music|storyboard|clips|assembling|done|failed
    detail: str = ""
    title: str = ""
    clips_total: int = 0
    clips_done: int = 0
    clips: list = field(default_factory=list)   # relative file names of finished clips
    final: str = ""
    error: str = ""
    cancel_requested: bool = False
    pause_requested: bool = False
    gpu_wait: bool = False          # parked in the queue until the GPU schedule window opens
    archived: bool = False
    regen_queue: list = field(default_factory=list)   # clip numbers pending regeneration
    assemble_requested: bool = False
    needs_assemble: bool = False    # clips changed since the final video was assembled
    stale_clips: list = field(default_factory=list)   # clips whose scene prompt was edited
    yt_status: str = ""      # "", uploading, uploaded, failed
    yt_progress: int = 0
    yt_video_id: str = ""
    yt_error: str = ""
    log: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    # per-stage timing, stage -> {start, end, seconds, running, last_start} (epoch floats).
    # `start` is written once (first real work), `end` is the last time the stage stopped for any
    # reason (next stage, GPU-window park, pause, failure), `seconds` accumulates only active time.
    # Maintenance of a finished job is kept apart under `regen_clips` / `regen_assembling`.
    timings: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return config.OUTPUT_DIR / self.id

    def set(self, status=None, detail=None, **kw):
        if status and status != self.status:
            self._mark_stage(status)
        if status:
            self.status = status
        if detail is not None:
            self.detail = detail
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated = time.time()
        self.save()

    def close_running(self, at: float):
        """Stop the stage that is currently timed (at most one), crediting active time up to `at`."""
        for t in self.timings.values():
            if t.get("running"):
                t["running"] = False
                t["end"] = at
                t["seconds"] = round(t.get("seconds", 0.0) + max(0.0, at - t.get("last_start", at)), 1)

    def _mark_stage(self, new: str):
        """Status transition hook: close the running stage, open `new` if it is a work stage."""
        now = time.time()
        self.close_running(now)
        if new not in STAGE_STATUSES:
            return
        key = new
        if new in ("clips", "assembling") and (self.regen_queue or self.assemble_requested):
            key = "regen_" + new    # clip regeneration / reassembly of a finished job: keep the original render intact
        t = self.timings.setdefault(key, {"start": now, "seconds": 0.0})
        t["running"] = True
        t["last_start"] = now

    def add_log(self, msg: str):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.log = self.log[-200:]
        self.updated = time.time()
        self.save()

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "job.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
def _install_upload(job: Job, up):
    """Copy the user's staged track into the fresh job folder as music.mp3 and consume the
    upload. Called between job.save() and the queue, so a failure leaves nothing registered."""
    if not up:
        return
    try:
        music_upload.install(up["id"], job.dir / "music.mp3")
    except Exception as e:  # noqa: BLE001 - the job is not registered yet; surface this as a 400
        shutil.rmtree(job.dir, ignore_errors=True)
        raise ValueError(f"the uploaded music track could not be saved ({e})")
    job.add_log(f"your uploaded music track: {up['name']} ({up['seconds']:.0f}s)")


class JobCancelled(Exception):
    pass


class JobPaused(BaseException):
    """Raised at a checkpoint when a priority regeneration wants the worker."""
    pass


class JobWaitingWindow(BaseException):
    """Raised right before a ComfyUI submission when the GPU schedule window is closed.
    BaseException so the non-fatal `except Exception` wrappers cannot swallow it."""
    pass


class JobWaitingGpu(BaseException):
    """Raised right before a generation step when another job holds the GPU lease. The job is
    parked back in the queue with everything it has done so far kept, so the lane it was using
    goes to a job that still has LLM work to do. BaseException for the same reason as above."""
    pass


def _check_cancel(job: Job):
    if job.cancel_requested:
        raise JobCancelled()
    if job.pause_requested:
        raise JobPaused()


def _draft_dropped(job: Job) -> bool:
    """True when the user cancelled while this draft was waiting on the LLM. The draft runs in its
    own thread, blocked in an HTTP request that cannot be interrupted from outside, so cancel()
    fails the job there and then; whatever the writer returns afterwards is dropped rather than
    written over that."""
    if not job.cancel_requested:
        return False
    job.add_log("the draft came back after the cancel - discarded")
    return True


def _drafted(job: Job) -> bool:
    """True when a script the user could still review is on disk. A first draft writes plan.json
    only at its very end, so this holds for a rewrite and not for a first attempt - which is what
    lets a cancelled rewrite keep the script being reviewed."""
    try:
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - nothing drafted yet, or half-written
        return False
    return bool((plan.get("narration") or "").strip()
                or ((plan.get("story") or {}).get("beats")))


def _gpu_gate(job: Job, window: bool = True):
    """Checkpoint placed immediately before every generation step of the worker path.

    Takes the GPU lease, so only one job at a time renders or speaks while the other lane carries
    on with LLM work. The lease is kept from here until the job reaches `assembling` (local
    ffmpeg) or its run ends, whichever comes first - never handed back between clips, because a
    render is not worth interleaving.

    window=False is for Fish TTS: the schedule window gates ComfyUI submissions only, and giving
    narration an opinion about the window would park jobs the window was never meant to stop."""
    _check_cancel(job)
    if window and not schedule.is_open():
        raise JobWaitingWindow()
    if not gpu.acquire(job.id):
        raise JobWaitingGpu()


def _cleanup_warn(job: Job):
    def cb():
        if "cleanup not supported" not in " ".join(job.log[-50:]):
            job.add_log("note: ComfyUI file cleanup not supported by this server build")
    return cb


def _song_budget(job: Job) -> float:
    """Sung words per second the lyrics of this job are sized for (and paced against)."""
    return config.KIDS_SONG_WPS if (job.channel or {}).get("audience") == "kids" else 2.0


def _lyric_seconds(job: Job) -> float:
    """How long the user's own lyrics take to sing at this channel's pace."""
    words = len(llm.clean_lyrics(job.lyrics).split())
    return max(15.0, words / max(0.1, _song_budget(job)))


def _lyric_ceiling(seconds: float) -> float:
    """Hard cut handed to the music model for a lyric-driven song. Generous on purpose: the
    ceiling only ever truncates, so cutting the user's last verse is far worse than a little
    extra render time. 2x+20 matches what the presets effectively allow (mid target, high bound)."""
    return min(config.MAX_SONG_SECONDS, seconds * 2 + 20)


def _length_params(job: Job):
    """Per-job soft length targets: (words, target_seconds, max_seconds, seconds_range).
    All None for Auto - the global settings apply. A job with the user's own lyrics is sized
    by those lyrics and has NO range: a range is what triggers the "too short" rewrite, and
    the user's words must never be rewritten."""
    if getattr(job, "music_upload", "") and getattr(job, "music_seconds", 0.0):
        # the recording IS the length: nothing is written or rendered to fit a target, so only
        # the word count (the story the scenes are drawn from) is meaningful here
        secs = float(job.music_seconds)
        return int(secs * 2.5), secs, secs, None
    if getattr(job, "lyrics", ""):
        secs = _lyric_seconds(job)
        return int(secs * 2.5), secs, _lyric_ceiling(secs), None
    rng = config.video_length_range(job.video_length or "Auto")
    if not rng:
        return None, None, None, None
    lo, hi = rng
    mid = (lo + hi) / 2.0
    return int(mid * 2.5), mid, min(360.0, hi * 1.5 + 20), (lo, hi)


def _clip_seconds(job: Job, duration: float | None = None) -> float:
    """Clip length this job renders: longer targets use longer clips so a 5-minute video is
    ~27 clips instead of ~59 (Auto keeps the global CLIP_SECONDS). A lyric-driven song and an
    uploaded music track have no preset, so their measured length picks the rung instead.

    The Settings clip length is the FLOOR, the same way a story line's own length is
    (story.clip_durations): a target preset can only ever lengthen a clip, never shorten it, so
    asking for 10 s clips gets 10 s clips on a 30-45 second video instead of the rung's 5 s."""
    secs = duration or float(getattr(job, "music_seconds", 0.0) or 0.0)
    if (getattr(job, "lyrics", "") or getattr(job, "music_upload", "")) and secs:
        rung = config.clip_seconds_for_duration(secs)
    else:
        rung = config.video_length_clip_seconds(job.video_length or "Auto")
    return max(float(config.CLIP_SECONDS), rung)


def _truncated(t: dict, l_max: float) -> bool:
    """A song render that ran into the model's hard cut - its last lines are missing."""
    return float(t.get("duration") or 0.0) >= float(l_max) - 1.0


def _range_miss(t: dict, rng) -> float:
    """Seconds by which a rendered song falls outside the job's length range
    (0.0 inside it, and 0.0 for Auto jobs, which have no range)."""
    if not rng:
        return 0.0
    lo, hi = rng
    d = float(t.get("duration") or 0.0)
    return max(0.0, lo - d, d - hi)


def _resolve_lyrics(lyrics: str, channel: dict | None, audio_mode: str) -> str:
    """The user's own song lyrics, validated. Hiding the field in the UI is not enough - /jobs is
    a public API. Rejected rather than trimmed: a clamped sheet would lose a verse while the whole
    promise of the field is that these exact words are sung.

    Deliberately not a parameter of _resolve_choices - that signature is called positionally by
    tests/test_lyric_sync.py and tests/test_resolution.py."""
    text = (lyrics or "").strip()
    if not text:
        return ""
    if (channel or {}).get("audience") != "kids":
        raise ValueError("song lyrics are only available on kids channels")
    if audio_mode != "music":
        raise ValueError("song lyrics need the Generate Song audio mode")
    if len(text) > config.MAX_LYRICS_CHARS:
        raise ValueError(f"song lyrics are too long ({len(text)} characters, "
                         f"at most {config.MAX_LYRICS_CHARS})")
    words = len(llm.clean_lyrics(text).split())
    if words < config.MIN_LYRIC_WORDS:
        raise ValueError(f"song lyrics are too short ({words} sung words, "
                         f"at least {config.MIN_LYRIC_WORDS})")
    if words > config.MAX_LYRIC_WORDS:
        # beyond this the song runs into the model's hard cut and the last lines are never sung
        raise ValueError(f"song lyrics are too long ({words} sung words, "
                         f"at most {config.MAX_LYRIC_WORDS})")
    return text


def _is_story(job: Job) -> bool:
    """Is this a story video? Always go through here - "" is a legacy song job, not a falsy story."""
    return (getattr(job, "video_mode", "") or "song") == "story"


STORY_SOURCES = ("topic", "description", "youtube")


def resolve_video_mode(video_mode: str, channel: dict | None, source: str,
                        lyrics: str = "", upload_id: str = "") -> str:
    """"song" (the default, and every video made before this existed) or "story", validated.

    A story video tells a story the LLM invents, and its voiceover is only the spoken lines of that
    story. It needs a kids channel with characters (the format is built on named speakers) and a
    source the story can be written from - a pasted transcript is the one thing it cannot use, since
    those words ARE the script and a story video writes its own.

    Public (unlike _resolve_lyrics / _resolve_music_upload) because app.py resolves the format of
    a whole weekly plan up front, before it creates any job from it.

    Deliberately not a parameter of _resolve_choices - that signature is called positionally by
    tests/test_lyric_sync.py and tests/test_resolution.py, which also index its 11-tuple."""
    name = (video_mode or "").strip() or config.DEFAULT_VIDEO_MODE
    if name not in config.VIDEO_MODES:
        raise ValueError(f"unsupported video type: {video_mode}")
    if name != "story":
        return "song"
    # hiding the dropdown in the UI is not enough: /jobs is a public API
    if (channel or {}).get("audience") != "kids":
        raise ValueError("story videos are only available on kids channels")
    if not cast.enabled(channel) or not cast.active(channel):
        raise ValueError("a story video needs the channel's characters: switch characters on and "
                         "add a cast in Settings")
    if source not in STORY_SOURCES:
        raise ValueError("story videos are written from a topic, a description or a YouTube URL")
    if lyrics:
        raise ValueError("song lyrics belong to a song video - a story video is spoken, not sung")
    if upload_id:
        raise ValueError("an uploaded music track belongs to a song video - a story video is spoken")
    return "story"


def _resolve_music_upload(upload_id: str, audio_mode: str, lyrics: str):
    """The staged music track this job will use, validated; None when there is none. Hiding the
    control in the UI is not enough - /jobs is a public API. Rejected rather than repaired: a
    track that is not the song the user picked is worse than no video.

    Not gated to kids channels the way lyrics are - anyone may bring their own soundtrack.

    Deliberately not a parameter of _resolve_choices - that signature is called positionally by
    tests/test_lyric_sync.py and tests/test_resolution.py."""
    uid = (upload_id or "").strip()
    if not uid:
        return None
    if audio_mode != "music":
        raise ValueError("an uploaded music track needs the Generate Song audio mode")
    if lyrics:
        # lyrics are an instruction to the singer model; an uploaded recording has no singer
        raise ValueError("song lyrics cannot be combined with an uploaded music track - "
                         "the track you uploaded is the song")
    rec = music_upload.get(uid)
    if not rec or not music_upload.path(uid).exists():
        raise ValueError("that uploaded music track is no longer available - upload it again")
    return rec


def _fit_narration(job: Job, plan: dict, words: int | None):
    """Extend a drafted script that fell well under the job's word target (in place)."""
    if not words or job.source == "transcript":
        return
    have = len((plan.get("narration") or "").split())
    if not have or have >= words * llm.SHORT_SCRIPT_RATIO:
        return
    job.set(detail="script too short - asking vLLM for a longer one")   # keeps planning/drafting
    try:
        longer = llm.extend_narration(plan["narration"], words, channel=job.channel)
    except Exception as e:  # noqa: BLE001 - a short script is still a usable script
        job.add_log(f"script extension failed ({e}); keeping the {have}-word draft")
        return
    got = len(longer.split())
    if got <= have:     # the LLM did not manage a longer script: keep the draft we have
        job.add_log(f"script is {have} words for a {words}-word target - could not be extended")
        return
    plan["narration"] = longer
    job.add_log(f"script was {have} words for a {words}-word target - extended to {got}")


def _log_avatar_plan(job: Job):
    """What the presenter choice will actually buy, said at submit instead of an hour later at the
    avatar step. Both budgets decide this before anything renders, and "Whole video" is the one
    choice the default engine can almost never pay for, so the trade belongs where it is still a
    decision rather than in the log of a finished video."""
    if job.avatar_coverage in ("", "none") or not job.avatar_id:
        return
    engine = job.avatar_model or config.DEFAULT_AVATAR_MODEL
    label = config.AVATAR_COVERAGES.get(job.avatar_coverage, job.avatar_coverage)
    budget = config.avatar_budget_seconds(engine)
    if job.avatar_coverage == "full":
        line = (f"presenter: {label} on {engine} - the budgets pay for {budget:.0f}s of footage, so "
                f"a video longer than that keeps the presenter at the start and the end instead")
        best = max(config.AVATAR_MODELS, key=config.avatar_budget_seconds)
        if best != engine and config.avatar_budget_seconds(best) > budget + 1:
            line += f" ({best} would pay for {config.avatar_budget_seconds(best):.0f}s)"
    else:
        want = config.avatar_coverage_seconds(job.avatar_coverage, engine)
        mins = config.avatar_estimate_seconds(want, engine) / 60.0
        line = (f"presenter: {label} on {engine} - about {want:.0f}s of footage, "
                f"{'under a minute' if mins < 1 else f'{mins:.0f} min'} of GPU")
    job.add_log(line)


def _logo_for(job: Job):
    """The job's channel logo for the thumbnail overlay; legacy global logo as fallback."""
    if job.channel_id:
        p = channels.logo_path(job.channel_id)
        if p.exists():
            return p
    return config.LOGO_FILE if config.LOGO_FILE.exists() else None


def _repo_page_url(job_dir) -> str:
    """The project page the README was fetched from, off repo.json - "" if there is none.

    Only used to point the guide back at its source, so any failure to read it is silent."""
    try:
        meta = json.loads((Path(job_dir) / "repo.json").read_text(encoding="utf-8"))
        return str(meta.get("page_url") or "")
    except Exception:  # noqa: BLE001
        return ""


def _viewer_document(job: Job, plan: dict, source: str) -> tuple:
    """The document that goes out with this video: (res, origin, source_url).

    Two reasons a video owes the viewer something, in priority order:
      1. the final narration PROMISED an artefact - write exactly that (services.llm.write_resource);
      2. failing that, the video was made from a repository or a model page, and those always ship
         the setup guide for the project (services.llm.write_repo_guide).
    A promised resource wins, so a job never produces two documents."""
    res = llm.write_resource(
        plan.get("title", ""), plan.get("narration", ""), source_text=source,
        channel=job.channel, used=plan.get("cast_used"),
        max_chars=resource.markdown_budget(plan.get("seo_description", "")))
    if res or job.source != REPO_SOURCE:
        return res, resource.PROMISE, ""
    job.set("planning", "writing the setup guide from the README")
    url = _repo_page_url(job.dir)
    res = llm.write_repo_guide(
        plan.get("title", ""), plan.get("narration", ""), source, source_url=url,
        channel=job.channel, used=plan.get("cast_used"),
        max_chars=resource.GUIDE_MAX_CHARS)
    return res, resource.GUIDE, url


def _progress(job: Job, status: str):
    """Progress callback that also honors a pending cancel request."""
    def cb(msg):
        _check_cancel(job)
        job.set(status, msg)
    return cb


def run_job(job: Job):
    job.gpu_wait = False
    try:
        # A compilation has no transcript, plan, audio, scenes or clips - only other finished
        # videos - so it never enters _run. The branch is here rather than only at submit time
        # because _load_existing re-queues every non-terminal job after a restart, and one
        # interrupted mid-stitch has to resume down this path instead of failing in _run.
        if job.source == compilation.SOURCE:
            _run_compilation(job)
            return
        if job.regen_queue or job.assemble_requested:
            _run_maintenance(job)
            return
        _run(job)
    except JobWaitingGpu:
        job.gpu_wait = True
        job.add_log("another video is on the GPU - parked until it is free")
        job.set("queued", "waiting for the GPU - another video is rendering")
    except JobPaused:
        job.pause_requested = False
        job.add_log("paused - a priority clip regeneration runs first")
        job.set("queued", "paused for a priority clip regeneration")
    except JobWaitingWindow:
        job.gpu_wait = True
        job.add_log("GPU window closed - parked until it opens")
        job.set("queued", schedule.describe_wait())
    except JobCancelled:
        job.add_log("cancelled by user")
        job.set("failed", "cancelled by user", error="cancelled by user")
    except Exception as e:  # noqa: BLE001
        job.add_log(f"FAILED: {e}")
        job.add_log(traceback.format_exc()[-1500:])
        job.set("failed", str(e), error=str(e))
    finally:
        # however this run ended - finished, parked, cancelled or failed - the GPU goes back. A job
        # that never took the lease releases nothing, so a parked job cannot free the lease of the
        # job that refused it.
        gpu.release(job.id)


def _clip_on_disk(path: Path) -> bool:
    """A finished clip that the resume path reuses instead of rendering."""
    return path.exists() and path.stat().st_size > 10000


def _generate_clip(job: Job, client, t2v, i2v, plan: dict, i: int, dest: Path, label: str):
    """Render clip i from its scene prompt; i > 1 is seeded with clip i-1's last frame."""
    d = job.dir
    scene = plan['scenes'][i - 1]
    # cast characters named in the scene get their canonical look prepended (deterministic branding)
    prompt = f"{plan['style']}\n\n{cast.with_prefix(job.channel, scene, only=plan.get('cast_used'))}".strip()
    first_frame = None
    opening = plan.get("opening") if i == 1 else None
    if i > 1:
        frame = video.last_frame(d / f"clip_{i - 1:02d}.mp4", d / f"frame_{i - 1:02d}.jpg")
        first_frame = client.upload_image(frame, f"videogen_{job.id}_frame_{i - 1:02d}.jpg")
        # steer the I2V model to treat the seeded frame as an in-progress moment, not a fresh shot
        prompt += ("\n\nThe video starts exactly at the provided first frame, mid-action: continue that "
                   "ongoing motion smoothly in one unbroken shot - no cut, no scene change, no camera "
                   "jump at the start.")
    elif opening and (d / "cast_opening.jpg").exists():
        # clip 1 opens on the channel's main character: its approved portrait is the first frame
        first_frame = client.upload_image(d / "cast_opening.jpg", f"videogen_{job.id}_cast.jpg")
        prompt += (f"\n\nThe video starts exactly at the provided first frame, which shows {opening['name']} "
                   f"({opening.get('look', '')}). Bring that image to life from the very first moment and keep "
                   f"{opening['name']}'s exact appearance throughout - no cut or scene change at the start.")
    template = t2v if first_frame is None else i2v
    # the plan freezes each clip's real length (the last one is trimmed to the audio);
    # legacy plans without it render the Settings default
    clip_secs = plan.get("clip_seconds") or []
    seconds = clip_secs[i - 1] if i - 1 < len(clip_secs) else None
    # only LTX 2.5 has a negative prompt; build() drops it for a model whose graph has none
    negative = config.ltx_negative((job.channel or {}).get("video_type", ""))

    def progress(msg):
        job.set("clips", f"{label}: {msg}")

    for attempt in range(config.CLIP_RETRIES + 1):
        _gpu_gate(job)
        try:
            wf = client.build(template, prompt, f"videogen/{job.id}/clip_{i:02d}",
                              first_frame=first_frame, seconds=seconds, aspect_ratio=job.aspect_ratio,
                              megapixels=job.megapixels, model=job.video_model, negative=negative)
            client.run(wf, dest, on_progress=progress,
                       should_stop=lambda: job.cancel_requested,
                       cleanup=config.COMFY_CLEANUP, on_cleanup_failed=_cleanup_warn(job))
            break
        except comfy.ComfyError as e:
            _check_cancel(job)
            job.add_log(f"clip {i} attempt {attempt + 1} failed: {e}")
            if attempt >= config.CLIP_RETRIES:
                raise
    if first_frame is not None and config.COMFY_CLEANUP:
        client.scrub_file("input", "", first_frame)


def _load_plan(job: Job) -> dict:
    try:
        return json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_plan(job: Job, plan: dict):
    (job.dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _story_blob(plan: dict) -> str:
    """Everything a story's own text says, for cast detection: the premise, every beat and every
    spoken line. A character can be in the picture for the whole video without ever being named in
    the spoken lines, so the beats have to count too or their look would stop being sent."""
    s = plan.get("story") or {}
    parts = [s.get("premise", "")]
    for b in s.get("beats") or []:
        parts += [b.get("setting", ""), b.get("action", ""), " ".join(b.get("characters") or [])]
    for ln in s.get("lines") or []:
        parts += [ln.get("speaker", ""), ln.get("text", "")]
    return "\n".join(p for p in parts if p)


def _set_cast_used(job: Job, plan: dict, *extra: str) -> list:
    """plan["cast_used"]: the channel characters that are in this video, in first-mention order.

    Every roster name found in the final title/narration counts. The names the script writer
    reported choosing ("cast_chosen") are added when the text itself names nobody - a pasted
    transcript never mentions who plays in the visuals, and a script may say "the little fox"
    without naming Finn - otherwise the text (which the user may have edited) is the authority.

    extra: further text that also counts as "in this video" (a story video passes its beats)."""
    used = cast.names(cast.used_characters(job.channel, plan.get("title", ""),
                                           plan.get("narration", ""), *extra))
    if not used or job.source == "transcript":
        for n in plan.get("cast_chosen") or []:
            if n not in used:
                used.append(n)
    plan["cast_used"] = used
    return used


def _validate_story_lines(lines, current: list, used: list) -> list:
    """The spoken lines as the user edited them on the approval screen, validated.

    One line per scene, no more and no fewer: the scenes were drafted and the clips are built one
    per line, so adding or removing one here would silently re-cut the video. A speaker the roster
    does not know becomes the narrator, exactly as a drafted one does."""
    if not isinstance(lines, list):
        raise ValueError("the story must be a list of spoken lines")
    if len(lines) != len(current):
        raise ValueError(f"this story has {len(current)} scenes, but {len(lines)} lines were sent")
    out = []
    for i, (item, was) in enumerate(zip(lines, current), 1):
        if not isinstance(item, dict):
            raise ValueError(f"scene {i}: each line must be an object")
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if len(text) > config.MAX_STORY_LINE_CHARS:
            raise ValueError(f"scene {i}: the spoken line is too long ({len(text)} characters, "
                             f"at most {config.MAX_STORY_LINE_CHARS})")
        want = str(item.get("speaker") or "").strip()
        speaker = next((s for s in used if s.lower() == want.lower()), config.STORY_NARRATOR)
        out.append({**was, "speaker": speaker, "text": text})
    return out


def _drop_stale_lines(job: Job, lines: list):
    """Forget the audio of every line whose words changed, so it is spoken again."""
    vdir = job.dir / "voices"
    for i, ln in enumerate(lines, 1):
        wav = vdir / f"line_{i:02d}.wav"
        if ln.get("text", "") != ln.get("spoken_text") and wav.exists():
            wav.unlink(missing_ok=True)
            ln.pop("seconds", None)
        ln["spoken_text"] = ln.get("text", "")


def _clean_story_lines(job: Job, beats: list, lines: list, used: list, max_words: int) -> list:
    """Draft-time pass over the spoken lines: a line that says out loud what the viewer can already
    see, or that is too long to fit a clip, is written again - ONCE, and before the user ever reads
    it. Nothing the user approved is rewritten later; that is the same rule _fit_narration follows.

    Returns the issues to show on the approval screen (the lines are edited in place)."""
    chars = cast.select_characters(job.channel, used)
    hard, issues = story.max_line_words(), []
    for i, (ln, beat) in enumerate(zip(lines, beats), 1):
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        hits = story.describes_look(text, chars)
        reason = "long" if len(text.split()) > hard else ("describes" if hits else "")
        if not reason:
            continue
        job.set("drafting", f"rewriting line {i} of the story")
        try:
            new = llm.rewrite_story_line(ln, beat, reason, max_words, channel=job.channel,
                                         used=used, speakers=used)
        except Exception as e:  # noqa: BLE001 - one awkward line never costs a draft
            issues.append(f"line {i} could not be written again ({e})")
            continue
        fresh = (new.get("text") or "").strip()
        if not fresh or fresh == text:
            continue
        ln["text"], ln["rewritten"] = fresh, reason
        still = story.describes_look(fresh, chars)
        if reason == "describes":
            issues.append(f"line {i} described the picture ({', '.join(hits[:3])}) and was written "
                          "again" + (" - it still does, so edit it before approving" if still else ""))
        else:
            issues.append(f"line {i} took too long to say and was written again")
    return issues


def _youtube_transcript(job: Job, status: str = "transcript") -> str:
    """The captions of job.url, fetched once and kept in transcript.txt, quality-gated.

    status: the job status to show while fetching. A story video needs its source text in the DRAFT
    thread (it has to reach the approval step like every other story), the other sources fetch it in
    the worker, so the stage label differs but nothing else does."""
    tfile = job.dir / "transcript.txt"
    if tfile.exists() and tfile.stat().st_size > 50:
        job.add_log("resume: reusing transcript")
        return tfile.read_text(encoding="utf-8")
    job.set(status, "fetching YouTube transcript")
    info = transcript.get_transcript_info(job.url)
    text, coverage = info["text"], info["seconds"]
    words = len(text.split())
    if words < config.MIN_TRANSCRIPT_WORDS:
        raise RuntimeError(
            f"This video cannot be processed: no proper transcript available "
            f"(only {words} words of captions). Please choose a video with real speech.")
    if coverage > 60 and words < (coverage / 60.0) * config.MIN_TRANSCRIPT_WPM:
        raise RuntimeError(
            f"This video cannot be processed: the transcript is too sparse for its length "
            f"({words} words over {coverage / 60:.1f} minutes - likely music or little speech).")
    job.dir.mkdir(parents=True, exist_ok=True)
    tfile.write_text(text, encoding="utf-8")
    job.add_log(f"transcript: {words} words covering {coverage:.0f}s of video")
    return text


def _story_reference(job: Job, speaker: str, pins: dict, notes: list):
    """(ref_audio, ref_text, label) for one spoken line, in priority order:

      1. a voice already pinned for this speaker in this video (so a character never changes voice
         mid-story, and a resume reuses the same pin);
      2. the reference voice configured for that character on the channel;
      3. for the narrator, the video's own reference voice (the New Video dialog's Voice picker);
      4. nothing - Fish Speech invents a voice, which is then pinned for this speaker.

    A configured voice that has since been deleted falls through to 4 with a note, never an error:
    the same thing the narration branch already does for a deleted job voice."""
    pinned = pins.get(speaker)
    if pinned:
        return Path(pinned[0]), pinned[1], "pinned"
    if speaker != config.STORY_NARRATOR:
        vid = cast.voice_id_for(job.channel, speaker)
        if vid:
            ref = voices.reference(vid)
            if ref:
                return ref[0], ref[1], f"voice:{vid}"
            notes.append(f"{speaker}'s reference voice no longer exists - the channel's default "
                         "character voice is used instead")
        # the channel's default character voice: on a kids channel this is the one setting that
        # makes every character without a voice of their own sound like a child
        fallback = (job.channel or {}).get("character_voice") or ""
        if fallback:
            ref = voices.reference(fallback)
            if ref:
                return ref[0], ref[1], f"voice:{fallback}"
            notes.append("the channel's default character voice no longer exists - a voice is "
                         "cloned from each character's first line instead")
        return None, "", "cloned"
    if job.voice_id == voices.RANDOM_VOICE:
        # a different narrator every line is the one thing a story must never do
        notes.append("the random voice cannot narrate a story - one voice is cloned and kept")
        return None, "", "cloned"
    ref = voices.reference(job.voice_id)
    if ref:
        return ref[0], ref[1], f"voice:{job.voice_id}"
    if job.voice_id:
        notes.append(f"reference voice {job.voice_id} no longer exists - a narrator voice is cloned")
    return None, "", "cloned"


def _story_audio(job: Job, plan: dict) -> tuple:
    """Speak every line of the story, each in its speaker's own voice, and lay the lines out on the
    clip grid. Returns (narration_file, durs, line_seconds).

    Nothing is transcribed and nothing is aligned: we wrote the words and we built the track, so
    every timestamp is known exactly. The clip each line gets is the job's preset length, or longer
    when the line needs the room - never shorter, so a run of terse lines still paces normally.

    Resume-safe at two levels: each line's wav is kept, and once the track exists the whole grid is
    read back from plan["story"]["clip_seconds"] instead of being recomputed, so a Settings change
    between runs can never re-cut a job and strand its scene prompts."""
    d = job.dir
    st = plan["story"]
    lines, beats = st["lines"], st["beats"]
    n = len(lines)
    narration_file = d / "narration.wav"
    if (narration_file.exists() and narration_file.stat().st_size > 1000
            and len(st.get("clip_seconds") or []) == n):
        job.add_log("resume: reusing the spoken story")
        return narration_file, [float(x) for x in st["clip_seconds"]], st.get("line_seconds") or []

    vdir = d / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    pins = {k: (str(vdir / v["file"]), v["text"])
            for k, v in (st.get("voice_pins") or {}).items() if (vdir / v["file"]).exists()}
    notes, client = [], None
    for i, ln in enumerate(lines, 1):
        _check_cancel(job)
        text = (ln.get("text") or "").strip()
        wav = vdir / f"line_{i:02d}.wav"
        if not text:
            ln["seconds"] = 0.0
            continue
        if wav.exists() and wav.stat().st_size > 1000:
            ln.setdefault("seconds", round(tts.audio_duration(wav), 3))
        else:
            speaker = ln.get("speaker") or config.STORY_NARRATOR
            ref_audio, ref_text, label = _story_reference(job, speaker, pins, notes)
            for note in notes:
                job.add_log(note)
            notes.clear()
            job.set("tts", f"speaking line {i}/{n} ({speaker})")
            _gpu_gate(job, window=False)   # Fish shares the GPU with ComfyUI: one job speaks at a time
            client = client or tts.connect()
            ln["seconds"] = round(tts.synthesize(text, wav, on_progress=_progress(job, "tts"),
                                                 ref_audio=ref_audio, ref_text=ref_text,
                                                 client=client), 3)
            ln["voice"] = label
        speaker = ln.get("speaker") or config.STORY_NARRATOR
        if speaker not in pins and ln.get("voice") in ("cloned", None):
            # the first line this speaker says becomes their voice for the rest of the video
            pins[speaker] = (str(wav), text)
            st.setdefault("voice_pins", {})[speaker] = {"file": wav.name, "text": text}
        _save_plan(job, plan)

    line_seconds = [float(ln.get("seconds") or 0.0) for ln in lines]
    durs = story.clip_durations(line_seconds, _clip_seconds(job), job.video_model)
    long = [i for i, dur in enumerate(durs, 1) if dur > config.STORY_MAX_CLIP_SECONDS]
    if long:
        job.add_log(f"line(s) {', '.join(map(str, long))} speak longer than "
                    f"{config.STORY_MAX_CLIP_SECONDS:g}s, so their scenes run long")
    parts, at = [], 0.0
    for i, (ln, dur) in enumerate(zip(lines, durs), 1):
        if float(ln.get("seconds") or 0.0) > 0:
            parts.append((vdir / f"line_{i:02d}.wav", at + config.STORY_LINE_LEAD))
        at += dur
    job.set("tts", "laying the spoken story out on the clips")
    total = tts.concat_timed(parts, narration_file, total=at)
    st["clip_seconds"] = durs
    st["line_seconds"] = line_seconds
    st["audio"] = {"duration": round(total, 3), "lead": config.STORY_LINE_LEAD,
                   "gap": config.STORY_LINE_GAP, "spoken": sum(1 for s in line_seconds if s > 0)}
    _save_plan(job, plan)
    job.add_log(f"spoken story: {st['audio']['spoken']}/{n} lines, {total:.1f}s")
    return narration_file, durs, line_seconds


def _thumb_variants(job: Job, plan: dict, select=None) -> int:
    """Draw the thumbnail text variants from thumb_base.jpg and make one of them active.

    Pure Pillow and cheap, so it runs on every pass rather than being frozen: an edited line or a
    new accent colour then applies without another GPU render. The active one is COPIED to
    thumbnail.jpg, which keeps its exact existing meaning - so the download route, the publish
    bundle and the YouTube upload need no knowledge of variants at all.

    Non-fatal throughout: the clean render is always a perfectly good thumbnail."""
    d = job.dir
    base = d / "thumb_base.jpg"
    if not base.exists():
        # a job made before variants existed: adopt its finished thumbnail as the base, so the
        # feature works retroactively on every video ever made
        if (d / "thumbnail.jpg").exists():
            shutil.copyfile(d / "thumbnail.jpg", base)
        else:
            return 0
    # A video with a presenter gets a SECOND base as well: the same thumbnail with the
    # presenter cut out of its own clip and composited in the corner it occupies in the video.
    # Both are then offered, because which one earns the click is a judgement call - the face
    # draws the eye, the clean render shows more of the picture - and neither costs a render.
    pres = _presenter_base(job, plan, base)
    bases = [(base, False)] + ([(pres, True)] if pres else [])
    recs = []
    texts = (plan.get("thumbnail_texts") or [])[:config.THUMB_TEXT_VARIANTS]
    for src, is_pres in bases:
        if is_pres:        # the presenter base is itself a choice, with no text on it at all
            recs.append({"file": src.name, "text": "", "layout": "presenter", "presenter": True})
        # Same master-switch rule as the captions: THUMB_TEXT_ENABLED is read here, not frozen
        # into the job, so turning it off in Settings strips the text from an existing video's
        # thumbnail the next time its variants are built. job.thumb_text is this video's choice.
        if config.THUMB_TEXT_ENABLED and job.thumb_text and config.THUMB_TEXT_VARIANTS > 0:
            try:
                for r in thumbtext.variants(src, d, texts, offset=len(recs)):
                    recs.append({**r, "presenter": is_pres})
            except Exception as e:  # noqa: BLE001
                job.add_log(f"thumbnail text failed ({e}); the plain thumbnail is kept")
    thumbtext.drop_stale(d, len(recs))
    plan["thumbnail_variants"] = recs
    k = thumbtext.activate(d, recs, job.thumb_variant if select is None else select)
    job.set(thumb_variant=k)
    return k


def _presenter_base(job: Job, plan: dict, base: Path) -> Path | None:
    """thumb_base_presenter.jpg - the thumbnail with this video's presenter composited in - or
    None when the video has no presenter, or building it fails.

    Rebuilt whenever the segment clip is newer, so a re-rendered presenter reaches the
    thumbnail too. Non-fatal like everything else about the thumbnail: the worst case is that
    the clean render is the only one offered."""
    segs = ((plan.get("avatar") or {}).get("segments")) or []
    if not segs:
        return None
    d = job.dir
    clip = d / segs[0]["file"]
    if not clip.exists():
        return None
    dst = d / "thumb_base_presenter.jpg"
    if dst.exists() and dst.stat().st_mtime >= max(clip.stat().st_mtime, base.stat().st_mtime):
        return dst
    try:
        # `d / ""` is d itself and a directory passes .exists(), so an empty matte name has
        # to be caught before the join - otherwise the bubble look hands ffmpeg the job
        # DIRECTORY as its alpha video, and thumbnail_cutout never reaches the single-frame
        # u2net fallback that is the whole reason it accepts None. Same shape as _avatar_overlay.
        matte = d / segs[0]["matte"] if segs[0].get("matte") else None
        cut = avatar.thumbnail_cutout(clip, d / "thumb_presenter.png",
                                      matte if matte and matte.exists() else None)
        return thumbtext.with_presenter(base, cut, dst)
    except Exception as e:  # noqa: BLE001 - a thumbnail without the presenter is still fine
        job.add_log(f"presenter thumbnail skipped ({e})")
        return None


def _render_thumbnail(job: Job, plan: dict, dest: Path, on_progress=None, should_stop=None) -> tuple:
    """Render plan["thumbnail_prompt"] into dest (JPG at the job's thumbnail size, logo overlaid).

    The canonical looks of every character in the video are always prepended, so the thumbnail
    shows them even when the LLM's prompt only says "the little fox". When the story's opening
    character has an approved portrait (cast_opening.jpg, prepared before the scenes), the render
    is seeded from that portrait like clip 1: a short clip that moves from the reference pose into
    the thumbnail composition, whose LAST frame becomes the thumbnail - face, colours and outfit
    then match the reference. Without a portrait it is the 5-frame text-to-video render."""
    d = job.dir
    client = comfy.ComfyClient()
    parts = [plan.get("style", ""), cast.looks_prefix(job.channel, plan.get("cast_used")),
             plan["thumbnail_prompt"]]
    prompt = "\n\n".join(p.strip() for p in parts if p and p.strip())
    opening = plan.get("opening") or {}
    portrait = d / "cast_opening.jpg"
    first_frame, seconds, template = None, None, config.T2V_WORKFLOW
    if opening.get("name") and portrait.exists():
        first_frame = client.upload_image(portrait, f"videogen_{job.id}_cast.jpg")
        seconds, template = config.THUMB_SEED_SECONDS, config.I2V_WORKFLOW
        prompt += (f"\n\nThe shot starts from the provided reference image of {opening['name']}. "
                   f"{opening['name']} moves into the thumbnail composition described above, keeping this "
                   "exact appearance - same face, colours, clothes and marks. End on that composition, "
                   "held still.")
    wf = client.build_thumbnail(client.load_workflow(template), prompt, f"videogen/{job.id}/thumb",
                                aspect_ratio=job.aspect_ratio, first_frame=first_frame, seconds=seconds)
    raw = client.run(wf, d / "thumb_raw.mp4", on_progress=on_progress, should_stop=should_stop,
                     cleanup=config.COMFY_CLEANUP, on_cleanup_failed=_cleanup_warn(job))
    tw, th = config.THUMB_SIZES.get(job.aspect_ratio, (1280, 720))
    video.make_thumbnail(raw, dest, tw, th, logo=_logo_for(job), last=first_frame is not None)
    if first_frame is not None:
        job.add_log(f"thumbnail seeded from {opening['name']}'s portrait")
        if config.COMFY_CLEANUP:
            client.scrub_file("input", "", first_frame)
    return tw, th


def _repo_source_text(job: Job) -> str:
    """README + facts of the job's GitHub / Hugging Face URL, cached on disk.

    Runs in the draft thread, so the user reviews the script before any GPU time is spent. The
    cache is not just a speed-up: redraft_script re-enters the same draft, and a transient 403
    there must not throw away a good draft."""
    d = job.dir
    d.mkdir(parents=True, exist_ok=True)
    tfile, rfile = d / "transcript.txt", d / "repo.json"
    if tfile.exists() and tfile.stat().st_size > 50:
        job.add_log("reusing the fetched README")
        return tfile.read_text(encoding="utf-8")
    job.set("drafting", "fetching the repository page")
    info = repo.fetch(job.url or job.topic)
    tfile.write_text(info["text"], encoding="utf-8")
    rfile.write_text(json.dumps(info["meta"], indent=2), encoding="utf-8")
    job.add_log(f"{info['kind']}: {info['id']} - {info['words']} words of README and facts")
    return info["text"]


def _page_screenshot(job: Job) -> int:
    """Screenshot the repo / model page for the scrolling panel; returns where its README starts.

    Non-fatal by design: without page.png the video is finished from the README text alone."""
    png = job.dir / "page.png"
    meta = job.dir / "page.json"
    if png.exists() and png.stat().st_size > 5000:
        job.add_log("resume: reusing the page screenshot")
        # the offset is kept beside the picture, not only in the plan: this job's screenshot
        # survived a failed draft, and returning 0 here once started the panel on the page
        # header for the rest of the job's life
        try:
            return int(json.loads(meta.read_text(encoding="utf-8")).get("content_y") or 0)
        except Exception:  # noqa: BLE001 - captured before the sidecar existed, or half-written
            return 0
    url = job.url
    rfile = job.dir / "repo.json"
    if rfile.exists():
        try:
            url = json.loads(rfile.read_text(encoding="utf-8")).get("page_url") or url
        except Exception:  # noqa: BLE001
            pass
    try:
        job.set("drafting", "screenshotting the page")
        w, h, content_y = browser.screenshot(url, png)
        meta.write_text(json.dumps({"width": w, "height": h, "content_y": int(content_y)}),
                        encoding="utf-8")
        job.add_log(f"page screenshot: {w}x{h}")
        return content_y
    except Exception as e:  # noqa: BLE001 - the video is made from the README alone
        job.add_log(f"page screenshot failed ({e}); the video will have no page overlay")
        return 0


def _storyboard_assets(job: Job, plan: dict, n: int):
    """Plan the story board cards (LLM) and render their related images (ComfyUI).
    Non-fatal: on failure the job continues without the overlay for this assembly."""
    job.set("storyboard", "planning the story board")
    try:
        _gpu_gate(job)
        storyboard.ensure_assets(job, plan, n, on_progress=_progress(job, "storyboard"),
                                 add_log=job.add_log,
                                 should_stop=lambda: job.cancel_requested or not schedule.is_open())
        _gpu_gate(job)   # window closed between images: park; the missing images render on resume
        _save_plan(job, plan)
        job.add_log("story board ready")
    except JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001
        job.add_log(f"story board failed ({e}); continuing without the overlay")


def upscale_filename(short_side: int) -> str:
    """File name of an on-demand upscale of final.mp4, e.g. final_1440p.mp4."""
    return f"final_{int(short_side)}p.mp4"


def _drop_upscales(job: Job):
    """Delete the on-demand upscales of the previous final video (called on reassembly)."""
    if not job.upscales:
        return
    for u in job.upscales:
        try:
            (job.dir / u["file"]).unlink(missing_ok=True)
        except OSError:
            pass
    job.upscales = []
    job.add_log("previous upscaled versions removed - they were made from the old final video")


def _upscale_joined(job: Job, joined: Path) -> Path:
    """Upscale the joined clips to the job's output resolution (ffmpeg lanczos) before the story
    board cards are drawn, so the cards render at the final frame size. Native / already large
    enough: the joined file is returned untouched. Never downscales."""
    short = config.OUTPUT_RESOLUTIONS.get(job.resolution or "", 0)
    if not short:
        return joined
    w, h = video.frame_size(joined)
    size = video.target_size(w, h, short)
    if size is None:
        job.add_log(f"clips are {w}x{h}: already at or above {job.resolution}, no upscale")
        return joined
    job.set("assembling", f"upscaling to {size[0]}x{size[1]}")
    out = video.upscale(joined, joined.parent / "clips_upscaled.mp4", *size)
    job.add_log(f"upscaled {w}x{h} -> {size[0]}x{size[1]} ({job.resolution})")
    return out


def _polish_windows(job: Job, total: float) -> tuple:
    """(hook_end, end_start) in seconds. Both collapse to "no card" when the video is too short to
    give those seconds up."""
    if not job.polish:
        return 0.0, float(total)
    hook = config.HOOK_CARD_SECONDS
    tail = config.END_CARD_SECONDS
    hook_end = round(0.2 + hook, 3) if hook > 0 and total > 2 * hook else 0.0
    end_start = round(total - tail, 3) if tail > 0 and total > 3 * tail else float(total)
    return hook_end, end_start


def _polish_layers(job: Job, plan: dict, size: tuple, total: float, hook_end: float,
                   end_start: float) -> tuple:
    """The hook card, end card and progress bar as composite layers, plus the times a whoosh
    should sound. Each piece is guarded on its own, so a broken end card still leaves the rest."""
    if not job.polish:
        return [], []
    d = job.dir
    out, hits = [], []
    if hook_end > 0:
        try:
            text = job.hook or plan.get("seo_title") or plan.get("title") or ""
            if text.strip():
                png = cardtemplates.render_hook(text, size, d / "hook_card.png",
                                                accent=config.STORYBOARD_ACCENT)
                out.append({"png": png, "start": 0.2, "end": hook_end, "fade": 0.35})
                hits.append(0.2)
                job.add_log(f"hook card 0.2s-{hook_end:.1f}s")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"hook card failed ({e}); assembling without it")
    if end_start < total:
        try:
            png = cardtemplates.render_end(size, d / "end_card.png",
                                           channel=(job.channel or {}).get("name", ""),
                                           logo=_logo_for(job), line=config.END_CARD_TEXT,
                                           accent=config.STORYBOARD_ACCENT)
            out.append({"png": png, "start": end_start, "end": total, "fade": 0.4})
            hits.append(end_start)
            job.add_log(f"end card {end_start:.1f}s-{total:.1f}s")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"end card failed ({e}); assembling without it")
    if config.PROGRESS_BAR_HEIGHT > 0:
        try:
            out += video.progress_layers(size, total, d, color=config.STORYBOARD_ACCENT,
                                         height_pct=config.PROGRESS_BAR_HEIGHT)
        except Exception as e:  # noqa: BLE001
            job.add_log(f"progress bar failed ({e}); assembling without it")
    return out, hits


def _sfx_track(job: Job, hits: list, total: float):
    """One full-length wav with a whoosh at each hit, or None."""
    if not (job.polish and config.SFX_VOLUME > 0 and hits):
        return None
    try:
        return sfx.build_track(hits, job.dir / "sfx.wav", total)
    except Exception as e:  # noqa: BLE001 - never lose the mix over a sound effect
        job.add_log(f"sound effects skipped ({e})")
        return None


def _song_words(plan: dict) -> list:
    """The word timings music mode already produced for the lyric-synced scenes."""
    final = plan.get("music_final") or {}
    tries = plan.get("music_tries") or []
    i = int(final.get("try") or 0)
    if 1 <= i <= len(tries) and tries[i - 1].get("words"):
        return tries[i - 1]["words"]
    return (plan.get("music_words") or {}).get("words") or []


def _caption_assets(job: Job, plan: dict):
    """Word timings for the burned-in captions, cached once in plan["caption_words"].

    Only the voiceover path costs anything. A story video's timings are exact by construction
    (every line is spoken alone and laid on the clip grid) and a music video was already
    transcribed for its lyric-synced scenes; a voiceover needs one whisper pass over
    narration.wav. It runs BEFORE the clip loop so a failure surfaces early and a resume finds it
    cached rather than paying for whisper twice.

    Non-fatal: without timings the video is simply assembled with no captions."""
    if plan.get("caption_words"):
        job.add_log("resume: reusing the caption word timings")
        return
    d = job.dir
    try:
        if _is_story(job) and (plan.get("story") or {}).get("lines"):
            st = plan["story"]
            words = captions.words_from_lines(st["lines"], st.get("clip_seconds") or [],
                                              lead=config.STORY_LINE_LEAD)
            src = "story"
        elif _song_words(plan):
            words, src = captions.words_from_timeline(plan.get("timed_segments")) or [], "song"
            if not words:
                words = _song_words(plan)
        else:
            nar = d / "narration.wav"
            if not (nar.exists() and nar.stat().st_size > 1000):
                return
            job.set("tts", "timing the words for captions (whisper)")
            heard = lyrics.transcribe_words(nar, initial_prompt=plan.get("narration", "")[:600],
                                            on_progress=_progress(job, "tts"))
            # align_to_lyrics already solves "snap whisper onto text we wrote": it keeps our
            # spelling and punctuation and interpolates words whisper missed, so the captions
            # show the script rather than a mis-hearing of it
            aligned, ratio = lyrics.align_to_lyrics(heard, plan.get("narration", ""))
            words = aligned if ratio >= config.LYRIC_ALIGN_MIN_RATIO else heard
            src = "script" if ratio >= config.LYRIC_ALIGN_MIN_RATIO else "whisper"
        if not words:
            return
        plan["caption_words"] = {"words": words, "source": src,
                                 "whisper_model": config.WHISPER_MODEL}
        _save_plan(job, plan)
        job.add_log(f"captions: {len(words)} words timed ({src})")
    except JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001 - a video with no captions is still a video
        job.add_log(f"caption timing failed ({e}); assembling without captions")


def _avatar_pending(job: Job, plan: dict) -> bool:
    """True when the job wants a presenter and the plan has none - the one thing that makes a
    reassembly expensive, because the segments have to be rendered on the GPU first."""
    return bool((job.avatar_coverage or "none") != "none" and job.avatar_id
                and not (plan.get("avatar") or {}).get("segments"))


def _avatar_face_summary(segments: list) -> dict:
    """The likeness across the chain, as plan fields. Empty when nothing was measured."""
    scored = [s["face"] for s in segments if (s.get("face") or {}).get("faces")]
    if not scored:
        return {}
    return {"face_first": scored[0]["mean"], "face_last": scored[-1]["mean"],
            "face_min": min(f["min"] for f in scored),
            "face_means": [f["mean"] for f in scored]}


def _log_avatar_faces(job: Job, segments: list):
    """Say what happened to the likeness, in one line, for a chain long enough to drift.

    First chunk to last is the number that matters: the portrait is the reference image for every
    chunk, but each one is conditioned on the pixels of the one before it, so a long chain walks
    away from the face it started on. Read the TREND - the absolute score is a blunt instrument,
    and a presenter who has merely wandered still scores well clear of AVATAR_FACE_MIN_COSINE."""
    scored = [(i, s["face"]) for i, s in enumerate(segments, 1)
              if (s.get("face") or {}).get("faces")]
    if len(scored) < 2:
        return
    first, last = scored[0][1]["mean"], scored[-1][1]["mean"]
    worst = min(f["min"] for _i, f in scored)
    line = (f"presenter avatar: likeness to the portrait {first:.2f} -> {last:.2f} "
            f"over {len(scored)} chunk(s), worst frame {worst:.2f}")
    blind = [i for i, s in enumerate(segments, 1)
             if (s.get("face") or {}).get("frames") and not s["face"]["faces"]]
    if blind:
        line += f"; no face found at all in chunk(s) {', '.join(str(i) for i in blind)}"
    if worst < config.AVATAR_FACE_MIN_COSINE:
        line += f" - under {config.AVATAR_FACE_MIN_COSINE:.3f} is a different person"
    job.add_log(line)


def _avatar_assets(job: Job, plan: dict):
    """Render the presenter segments: one lip-synced clip (and its matte) per window.

    Resumable like every other artifact - a clip already on disk is never re-rendered, which
    matters more here than anywhere else because this is the most expensive stage in the
    pipeline. Non-fatal throughout: a video with no presenter is still a video, so every
    failure degrades with a log line instead of killing the job."""
    if (job.avatar_coverage or "none") == "none" or not job.avatar_id:
        return
    d = job.dir
    nar = d / "narration.wav"
    if not (nar.exists() and nar.stat().st_size > 1000):
        job.add_log("presenter avatar: no narration to lip-sync; skipped")
        return
    src = channels.avatar_path(job.channel_id, job.avatar_id)
    if not src.exists():
        job.add_log("presenter avatar: the channel's image is gone; assembling without it")
        return
    engine = job.avatar_model or config.DEFAULT_AVATAR_MODEL
    style = config.DEFAULT_AVATAR_STYLE
    done = plan.get("avatar") or {}
    if (done.get("engine") == engine and done.get("segments")
            and all((d / s["file"]).exists() for s in done["segments"])):
        job.add_log(f"resume: reusing {len(done['segments'])} presenter segment(s)")
        return
    try:
        total = tts.audio_duration(nar)
        words = ((plan.get("caption_words") or {}).get("words")) or []
        wins = avatar.segment_windows(job.avatar_coverage, total, engine, words)
        if not wins:
            job.add_log(f"presenter avatar: no room for a segment in {total:.0f}s; skipped")
            return
        asked = config.avatar_coverage_seconds(job.avatar_coverage, engine, total)
        got = sum(e - s for s, e in wins)
        est = config.avatar_estimate_seconds(got, engine) / 60.0
        if got + 0.25 < asked:
            # the budget bit: say so here rather than leaving it to be noticed in the timings
            job.add_log(f"presenter avatar: trimmed to {got:.1f}s (budget); "
                        f"{job.avatar_coverage} on {engine} wanted {asked:.1f}s")
        job.set("avatar", f"rendering {len(wins)} presenter segment(s), about {est:.0f} min")
        client = comfy.ComfyClient()
        # Fit to the engine's square HERE: ComfyUI's own fit is a centred cover-crop, which
        # takes the top off a tall headshot and animates the clipped hair for the whole clip.
        # Framing is an improvement, not a requirement, so a failure falls back to the
        # original rather than costing the presenter.
        try:
            ref = avatar.portrait_square(src, d / "avatar_ref.png", engine)
        except Exception as e:  # noqa: BLE001
            job.add_log(f"presenter avatar: could not re-frame the portrait ({e}); using it as is")
            ref = src
        img_name = client.upload_image(ref, name=f"videogen_avatar_{job.id}.png")
        segments = []
        # Which appearance each window belongs to. Every chunk of one appearance renders on that
        # appearance's own seed: the portrait is the reference either way, but fresh noise per
        # chunk makes the model re-interpret the same face at every seam.
        apps = avatar.appearances(wins)
        # ...and which of them may continue the chunk before it. Every AVATAR_CHAIN_MAXth chunk
        # is rendered from the portrait alone instead, so the drift the chain accumulates cannot
        # run the whole length of a long appearance. `anchors` is how many of those resets this
        # setting actually bought, measured against leaving the chain unbroken.
        may_chain = avatar.chain_plan(wins, engine, config.AVATAR_CHAIN_MAX)
        anchors = sum(1 for unbroken, now in zip(avatar.chain_plan(wins, engine, 0), may_chain)
                      if unbroken and not now)
        # prev_raw is the chunk as it came off the box, kept one iteration so the next chunk's
        # motion can be cut from it rather than from trim_lead_frame's re-encode. Tracked in
        # memory and never looked for on disk, because a pause, a cancel or a crash can abort a
        # chunk mid-download and leave a truncated one behind - so anything already there is
        # swept now rather than risked as conditioning, which also keeps the job directory (which
        # app.py lists) from showing a second copy of a segment after a failed run.
        for stale in d.glob("avatar_*_raw.mp4"):
            stale.unlink(missing_ok=True)
        prev_end, prev_clip, prev_raw, chained = None, None, None, 0
        for i, (start, end) in enumerate(wins, 1):
            clip = d / f"avatar_{i:02d}.mp4"
            matte = d / f"avatar_{i:02d}_matte.mp4"
            clip_existed = clip.exists()
            if not clip_existed:
                wav = d / f"avatar_{i:02d}.wav"
                secs = avatar.slice_audio(nar, start, end, wav, engine)
                _gpu_gate(job)
                job.set("avatar", f"presenter segment {i}/{len(wins)} ({secs:.1f}s)")
                audio_name = client.upload_file(wav, name=f"videogen_avatar_{job.id}_{i}.wav")
                # A window that butts up against the one before it is one appearance rendered in
                # chunks, so the chunk before it is what this one should continue. A bookend pair
                # is two separate appearances and gets no motion. prev_clip comes off disk, so a
                # resumed job chains exactly as a fresh one does - which is also why the tail
                # falls back to it: a chunk that was reused rather than rendered has no raw.
                motion_name, tail = "", d / f"avatar_{i:02d}_motion.mp4"
                if may_chain[i - 1] and prev_clip is not None and prev_clip.exists():
                    try:
                        avatar.motion_tail(prev_raw or prev_clip, tail, engine)
                        motion_name = client.upload_file(
                            tail, name=f"videogen_avatar_{job.id}_{i}_motion.mp4", mime="video/mp4")
                        chained += 1
                    except Exception as e:  # noqa: BLE001 - a seam is worse than a pose reset
                        job.add_log(f"presenter avatar: segment {i} could not continue "
                                    f"segment {i - 1} ({e}); rendering it from the portrait")
                raw = d / f"avatar_{i:02d}_raw.mp4"
                avatar.render_segment(
                    client, engine, img_name, audio_name, secs, raw,
                    prefix=f"video/videogen_{job.id}_avatar{i}",
                    on_progress=_progress(job, "avatar"), should_stop=lambda: job.cancel_requested,
                    on_cleanup_failed=_cleanup_warn(job), motion_name=motion_name,
                    seed=avatar.chunk_seed(job.id, apps[i - 1]))
                avatar.trim_lead_frame(raw, clip, engine)
                # the PREVIOUS render has served its last purpose (this chunk's motion); this one
                # is kept until the next chunk has cut its own tail from it
                if prev_raw is not None:
                    prev_raw.unlink(missing_ok=True)
                prev_raw = raw
                tail.unlink(missing_ok=True)
                # the narration slice has served its purpose the moment the clip is down
                if config.COMFY_CLEANUP:
                    client.scrub_file("input", "", audio_name)
                    if motion_name:
                        client.scrub_file("input", "", motion_name)
            if style == "cutout" and not matte.exists():
                job.set("avatar", f"cutting out presenter segment {i}/{len(wins)}")
                avatar.matte_segment(clip, matte, on_progress=_progress(job, "avatar"))
            # where the figure sits inside the square frame, so the overlay can draw the
            # FIGURE at the size that was asked for rather than the frame around it
            bbox = avatar.matte_bounds(matte) if style == "cutout" and matte.exists() else None
            # How far this chunk's face has drifted from the portrait. Measured HERE rather than
            # inside the render branch, and outside the cut-out branch, so it covers a reused
            # chunk on a resume and a bubble-look video too. None whenever the two CPU models
            # are not staged, which costs the measurement and nothing else.
            face = avatar.face_identity(clip, ref) if avatar.face_models_ready() else None
            segments.append({"start": start, "end": end, "file": clip.name,
                             "matte": matte.name if style == "cutout" else "",
                             "bbox": list(bbox) if bbox else [],
                             "face": face or {},
                             # rendered from the portrait alone rather than continuing the chunk
                             # before it - true for the first chunk of every appearance, and for
                             # each deliberate re-anchor after that
                             "anchored": not may_chain[i - 1]})
            prev_end, prev_clip = end, clip
            if clip_existed:
                # a reused chunk leaves no render to cut a tail from, so the next one falls back
                # to the trimmed clip rather than continuing some earlier chunk's render
                prev_raw = None
        if prev_raw is not None:
            prev_raw.unlink(missing_ok=True)
        # the portrait is the reference image for EVERY segment, so it goes only once they are
        # all rendered - not inside the loop with the audio
        if config.COMFY_CLEANUP:
            client.scrub_file("input", "", img_name)
        plan["avatar"] = {"engine": engine, "style": style, "coverage": job.avatar_coverage,
                          "seconds": round(got, 2), "segments": segments,
                          # what was asked for next to what the budgets paid for, so the job page
                          # can explain a trim without anyone reading the log
                          "asked_seconds": round(asked, 2),
                          "coverage_label": config.AVATAR_COVERAGES.get(job.avatar_coverage, ""),
                          "delivered": avatar.describe(wins, total),
                          "chain_max": int(config.AVATAR_CHAIN_MAX or 0),
                          # the likeness across the chain, so a drift can be read off a finished
                          # job instead of re-measured (scripts/avatar_identity.py does that)
                          **_avatar_face_summary(segments)}
        _save_plan(job, plan)
        chain = len(segments) > 1 and all(b["start"] - a["end"] < 0.05
                                          for a, b in zip(segments, segments[1:]))
        job.add_log(f"presenter avatar: {len(segments)} segment(s)"
                    f"{' end to end' if chain else ''}, {got:.1f}s on {engine}"
                    f"{f'; {chained} continued the chunk before' if chained else ''}"
                    f"{f', {anchors} re-anchored to the portrait' if anchors else ''}")
        _log_avatar_faces(job, segments)
    except (JobCancelled, JobPaused, JobWaitingWindow):
        raise
    except Exception as e:  # noqa: BLE001 - a video with no presenter is still a video
        job.add_log(f"presenter avatar failed ({e}); assembling without it")


def _avatar_rect(job: Job, plan: dict, size: tuple):
    """The part of the frame the presenter covers, as (x0, y0, x1, y1), or None for no presenter.

    What the story board cards are kept clear of. It is a CORNER box - a fifth of a 9:16 frame as
    a cut-out, a fortieth as a bubble - which is why the cards are trimmed against this rect
    rather than against the whole frame for the time the presenter speaks.

    avatar_box and not video._fill_box: _fill_box deliberately overshoots the box with the
    engine's transparent padding, so it would reserve screen the presenter does not cover."""
    info = plan.get("avatar") or {}
    if not (info.get("segments") or []):
        return None
    # the same style and caption reserve _avatar_overlay draws with, so the two cannot disagree
    clear = config.CAPTION_BOTTOM_MARGIN if job.captions else 0.0
    x, y, w, h = video.avatar_box(size, style=info.get("style") or "", clear_bottom=clear)
    return (x, y, x + w, y + h)


def _avatar_overlay(job: Job, plan: dict, size: tuple, total: float) -> list:
    """composite() layers for the presenter, or []. Rebuilt on every assembly so a resolution
    or style change applies on reassembly without re-rendering the expensive part."""
    info = plan.get("avatar") or {}
    segs = info.get("segments") or []
    if not segs:
        return []
    d = job.dir
    ready = []
    for s in segs:
        clip = d / s["file"]
        if not clip.exists():
            continue
        matte = d / s["matte"] if s.get("matte") else None
        matte = matte if matte and matte.exists() else None
        # a plan written before the subject was measured is measured now, which is what lets
        # a finished video grow its presenter on a reassembly instead of a re-render
        bbox = s.get("bbox") or (avatar.matte_bounds(matte) if matte else None)
        ready.append({"start": float(s["start"]), "end": min(float(s["end"]), total),
                      "video": clip, "matte": matte, "bbox": list(bbox) if bbox else []})
    if not ready:
        return []
    try:
        # captions own the bottom of the frame; the bubble sits above them, and for the cut-out
        # the captions move instead (see video.avatar_box)
        clear = config.CAPTION_BOTTOM_MARGIN if job.captions else 0.0
        return video.avatar_layers(size, ready, d, style=info.get("style") or "",
                                   accent=config.STORYBOARD_ACCENT, clear_bottom=clear)
    except Exception as e:  # noqa: BLE001
        job.add_log(f"presenter overlay failed ({e}); assembling without it")
        return []


def _caption_track(job: Job, plan: dict, size: tuple, cards: list, total: float):
    """Write captions.ass for this frame size, or None. Redone on every assembly, so a settings
    change applies on reassembly and the track always matches the final resolution."""
    # config.CAPTIONS_ENABLED is a MASTER switch, checked here rather than when the job was
    # created: turning it off in Settings and reassembling must strip the captions from a video
    # that already chose them, which is the whole point of a master switch.
    if not (config.CAPTIONS_ENABLED and job.captions
            and (plan.get("caption_words") or {}).get("words")):
        return None
    try:
        job.set("assembling", "writing the captions")
        phrases = captions.clamp(
            captions.group(plan["caption_words"]["words"], max_words=config.CAPTION_MAX_WORDS,
                           max_chars=config.CAPTION_MAX_CHARS), total)
        if not phrases:
            return None
        # a card owns the lower third for whole clips at a time, so the caption moves above it
        # for that window rather than being suppressed
        lifts = captions.lift_spans(
            [(c["start"], c["end"], c.get("rect")) for c in cards], size)
        family, _fdir = captions.font_family(bold=True)
        ass = captions.write_ass(phrases, size, job.dir / "captions.ass",
                                 accent=config.CAPTION_ACCENT, font=family,
                                 bottom_margin=config.CAPTION_BOTTOM_MARGIN,
                                 highlight=config.CAPTION_HIGHLIGHT, lifts=lifts)
        if not video.subtitles_ok(ass):
            job.add_log("this ffmpeg build cannot burn subtitles; assembling without captions")
            return None
        job.add_log(f"captions: {len(phrases)} on-screen groups")
        return ass
    except Exception as e:  # noqa: BLE001 - never lose the video over the overlay
        job.add_log(f"captions failed ({e}); assembling without them")
        return None


def _chapters(job: Job, plan: dict, bounds: list):
    """Work out the YouTube chapter list from the real clip cut points.

    Recomputed on every assembly rather than frozen: regenerating a clip changes the boundaries,
    and the chapters have to follow them. Non-fatal - a video with no chapters is just a video."""
    if not config.YT_CHAPTERS:
        plan.pop("chapters", None)
        return
    try:
        plan["chapters"] = chapters.build(
            bounds, cards=plan.get("storyboard"), segments=plan.get("segments"),
            beats=(plan.get("story") or {}).get("beats"),
            total=bounds[-1] if bounds else 0.0, min_gap=config.YT_CHAPTER_MIN_SECONDS)
        _save_plan(job, plan)
        if plan["chapters"]:
            job.add_log(f"{len(plan['chapters'])} chapters for the description")
        else:
            job.add_log("too short for YouTube chapters (it needs 3, each at least 10 s); none added")
    except Exception as e:  # noqa: BLE001 - the description is fine without them
        job.add_log(f"chapters failed ({e}); the description gets none")


def upload_description(plan: dict) -> str:
    """The description actually uploaded: the SEO text plus the chapter block.

    The chapters go last and are never merged into plan["seo_description"], which has a carefully
    ordered tail (call to action, then the hashtag line, then any viewer resource - see
    services/resource.py) and is rewritten wholesale when the resource is regenerated. YouTube
    scans the whole description for timestamps, so appending costs nothing.

    When the two together overflow the limit, the DESCRIPTION is trimmed rather than the chapters:
    chapters drive retention and navigation, and the resource is a file in the bundle anyway."""
    desc = str(plan.get("seo_description") or "")
    block = chapters.format_block(plan.get("chapters") or [])
    if not block:
        return desc[:config.YT_DESCRIPTION_LIMIT]
    room = config.YT_DESCRIPTION_LIMIT - len(block) - 2
    if room < 0:
        return block[:config.YT_DESCRIPTION_LIMIT]
    return f"{desc[:room].rstrip()}\n\n{block}"


def _assemble(job: Job, plan: dict, narration_file, music_file) -> Path:
    """Concat the clips, overlay the story board cards (if enabled) and mix the audio
    for the job's audio mode into final.mp4."""
    d = job.dir
    clips = [d / c for c in job.clips]
    joined = video.concat_clips(clips, d / "clips_joined.mp4")
    bounds = video.clip_boundaries(clips)
    _chapters(job, plan, bounds)
    joined = _upscale_joined(job, joined)
    # Both overlays and the captions go into ONE encode: a second full-size libx264 pass costs as
    # much as the first (~190 s for a 3-minute 4K video) and leaves another copy on disk.
    size = video.frame_size(joined)
    total = bounds[-1] if bounds else video.duration(joined)
    hook_end, end_start = _polish_windows(job, total)
    cards, layers = [], []
    # the hook and the end card own their windows outright: they are composited last, so whatever
    # they cover is simply not visible, and unlike a fact card they cannot be moved
    hook_win = [(0.0, hook_end)] if hook_end else []
    end_win = [(end_start, total)] if end_start < total else []
    # The presenter owns only its CORNER for these windows. The cards are trimmed against that
    # rect (and moved above it where they hit it); the page panel, which is wide enough that a
    # corner is most of it, still reserves the window outright the way the hook and end cards do.
    av_win = [(float(x["start"]), min(float(x["end"]), total))
              for x in ((plan.get("avatar") or {}).get("segments") or [])]
    if job.storyboard and isinstance(plan.get("storyboard"), list):
        job.set("assembling", "drawing the story board cards")
        try:
            cards = storyboard.render_all(d, plan, clips, size,
                                          accent=config.STORYBOARD_ACCENT)
            # a fact card must never share the screen with the hook or the end card
            cards = storyboard.trim_schedule(cards, hook_win, size=size)
            cards = storyboard.trim_schedule(cards, end_win, size=size)
            if av_win:
                # a rect per window, so a card clear of the presenter keeps its whole window and
                # one that is not rises above it - see storyboard.trim_schedule
                before = len(cards)
                av_rect = _avatar_rect(job, plan, size)
                cards = storyboard.trim_schedule(
                    cards, [(s, e, av_rect) for s, e in av_win], size=size)
                moved = sum(1 for c in cards if c.get("lifted"))
                if moved or len(cards) < before:
                    job.add_log(f"presenter: {moved} of {before} story board cards lifted clear "
                                f"of it, {before - len(cards)} dropped")
        except Exception as e:  # noqa: BLE001 - never lose the video over the overlay
            job.add_log(f"story board overlay failed ({e}); assembling without it")
            cards = []
    # The panel is settled BEFORE the cards go into `layers`: when they leave it no room it
    # reserves a window and they are trimmed out of it, so `cards` can still change here.
    page = d / "page.png"
    scroll = None
    if job.page_scroll and page.exists():
        try:
            # the fact cards can be moved for the panel, the hook and end card cannot
            soft = [(c["start"], c["end"]) for c in cards]
            win = video.scroll_window(total, busy=soft + hook_win + end_win + av_win)
            if not win:
                # The cards left no gap - about 29% of three-card schedules on a 40s video, and
                # the evenly spread ones at that. Reserve the panel a window and move the cards
                # out of it, rather than dropping the overlay that was explicitly asked for.
                win = video.scroll_reserve(total, soft=soft, hard=hook_win + end_win + av_win)
                if win:
                    kept = storyboard.trim_schedule(cards, [win], size=size)
                    job.add_log(f"page panel reserved {win[0]:.1f}s-{win[1]:.1f}s; "
                                f"{len(kept)} of {len(cards)} story board cards kept clear of it")
                    cards = kept
            if win:
                scroll = video.scroll_layer(joined, page, win[0], win[1],
                                            content_y=int(plan.get("page_content_y") or 0))
                job.add_log(f"page scroll {win[0]:.1f}s-{win[1]:.1f}s")
            elif total < video.SCROLL_MIN_TOTAL:
                job.add_log(f"video shorter than {video.SCROLL_MIN_TOTAL:.0f}s; "
                            "page overlay skipped")
            else:
                job.add_log("no room in the middle for the page overlay; skipped")
        except Exception as e:  # noqa: BLE001 - never lose the video over the overlay
            job.add_log(f"page scroll overlay failed ({e}); assembling without it")
    elif job.page_scroll:
        # the screenshot is only ever taken while the script is drafted, so a job reassembled
        # without one can never grow a panel - say so rather than skipping in silence
        job.add_log("no page screenshot on disk; the page overlay was skipped")
    if scroll is not None:
        layers.append(scroll)   # first in the list = composited first = the cards draw on top
    layers += cards
    # last in the list = composited last = on top of the cards and the page panel
    polish, hits = _polish_layers(job, plan, size, total, hook_end, end_start)
    layers += polish
    layers += _avatar_overlay(job, plan, size, total)
    hits += [c["start"] for c in cards]
    ass = _caption_track(job, plan, size, cards, total)
    source = joined
    if layers or ass is not None:
        job.set("assembling", f"overlaying {len(layers)} layers"
                + (" and the captions" if ass is not None else ""))
        try:
            source = video.composite(joined, layers, d / "clips_overlay.mp4", subtitles=ass)
            job.add_log(f"overlays: {len(cards)} story board cards"
                        + (" + page scroll" if scroll is not None else "")
                        + (" + captions" if ass is not None else ""))
        except Exception as e:  # noqa: BLE001 - never lose the video over the overlay
            if ass is not None:
                # subtitles_ok() already caught a missing libass and a malformed file, so this is
                # a last resort rather than a routine cost
                job.add_log(f"overlay failed ({e}); retrying without the captions")
                try:
                    source = video.composite(joined, layers, d / "clips_overlay.mp4")
                except Exception as e2:  # noqa: BLE001
                    job.add_log(f"overlay failed ({e2}); assembling without it")
                    source = joined
            else:
                job.add_log(f"overlay failed ({e}); assembling without it")
                source = joined
    job.set("assembling", "mixing audio")
    hit_track = _sfx_track(job, hits, total)
    fx = {"sfx": hit_track, "sfx_volume": config.SFX_VOLUME} if hit_track else {}
    mode = job.audio_mode or "narration+music"
    if mode == "music":
        return video.mix_audio(source, d / "final.mp4", music=music_file,
                               music_volume=1.0, clip_volume=0, **fx)
    if mode == "narration+music":
        # the clips' own sound is dropped by default: voice + music + per-clip effects clash
        clip_vol = config.CLIP_AUDIO_VOLUME if config.CLIP_AUDIO_WITH_MUSIC else 0
        return video.mix_audio(source, d / "final.mp4", narration=narration_file,
                               music=music_file, clip_volume=clip_vol, clip_boundaries=bounds, **fx)
    return video.mix_audio(source, d / "final.mp4", narration=narration_file,
                           clip_boundaries=bounds, **fx)


def _compilation_parts(job: Job) -> list:
    """The part job directories, in play order, or RuntimeError naming the one that is gone.

    Read off disk rather than out of JobManager.jobs so the stitch does not depend on the
    in-memory index, which is what a resume after a restart rebuilds."""
    out = []
    for pid in job.parts or []:
        d = config.OUTPUT_DIR / str(pid)
        rec = d / "job.json"
        if not rec.exists():
            raise RuntimeError(f"the video {pid} this compilation is made from has been deleted")
        part = Job.load(rec)
        final = d / (part.final or "final.mp4")
        if not final.exists():
            raise RuntimeError(f"the video {pid} this compilation is made from has no final video")
        out.append((part, final))
    return out


def _compilation_thumbnail(job: Job, plan: dict, parts: list):
    """Give the compilation a thumbnail by adopting one of its parts'.

    Every part already has a thumbnail that cost a GPU render; re-rendering one for the
    compilation would be the only GPU work in an otherwise local job, and it would have nothing
    new to draw. The part's CLEAN base (thumb_base.jpg - the render with the channel logo already
    on it but no overlay text) is preferred over its finished thumbnail, so the compilation's own
    lines are drawn over an unlettered picture rather than on top of the part's.

    Every part has the compilation's aspect ratio, which validate_parts guaranteed, so the base is
    already the right size. Non-fatal: without a thumbnail it is still an uploadable video."""
    d = job.dir
    for part, _final in parts:
        for name in ("thumb_base.jpg", "thumbnail.jpg"):
            src = part.dir / name
            if not src.exists():
                continue
            try:
                shutil.copyfile(src, d / "thumb_base.jpg")
                _thumb_variants(job, plan)
                job.add_log(f"thumbnail taken from {part.title or part.id}")
                return
            except Exception as e:  # noqa: BLE001 - a compilation without one is still a video
                job.add_log(f"thumbnail from {part.id} failed ({e})")
    job.add_log("no part had a thumbnail to adopt; the compilation has none")


def _compilation_seo(job: Job, plan: dict, titles: list, total: float):
    """Title, description, keywords and thumbnail lines for the compilation.

    One vLLM call, and deliberately non-fatal: the stitch is the expensive, irreplaceable part and
    must not be lost because the LLM is down. A failure falls back to compilation.default_title,
    which states the length and the lead subject - publishable, just not optimised."""
    length = compilation.fmt_length(total)
    try:
        seo = llm.write_compilation_seo(titles, length, channel=job.channel,
                                        used=plan.get("cast_used"))
    except Exception as e:  # noqa: BLE001
        job.add_log(f"SEO failed ({e}); using a plain title")
        seo = {}
    plan["seo_title"] = seo.get("seo_title") or compilation.default_title(
        titles, total, (job.channel or {}).get("name", ""))
    plan["seo_keywords"] = seo.get("seo_keywords") or []
    plan["seo_description"] = seo.get("seo_description") or ""
    plan["thumbnail_texts"] = seo.get("thumbnail_texts") or []
    plan["title"] = plan["seo_title"]
    job.set(title=plan["seo_title"])


def _run_compilation(job: Job):
    """Stitch the job's parts into one long final.mp4. Local ffmpeg only - no GPU, no queue.

    Every step is idempotent in the same way the render pipeline is: a part already normalized on
    disk is reused, and a finished final.mp4 is not rebuilt, so a restart mid-stitch costs at most
    the one encode that was running."""
    d = job.dir
    d.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(job)
    parts = _compilation_parts(job)
    titles = [p.title or p.id for p, _ in parts]

    final = d / "final.mp4"
    if not final.exists():
        job.set("assembling", f"preparing {len(parts)} videos")
        size = compilation.target_size([video.frame_size(f) for _, f in parts])
        files = []
        for k, (part, src) in enumerate(parts, start=1):
            dest = d / compilation.part_file(k)
            if not _clip_on_disk(dest):
                job.set("assembling", f"preparing video {k} of {len(parts)}")
                _check_cancel(job)
                video.normalize_for_concat(src, dest, size[0], size[1])
            files.append(dest)
        _check_cancel(job)
        job.set("assembling", "joining the videos")
        video.concat_clips(files, final)
        plan["parts"] = [{"job_id": p.id, "title": p.title, "seconds": video.duration(f)}
                         for (p, _), f in zip(parts, files)]

    # The per-part lengths are measured once and kept on the plan, so a resume that finds
    # final.mp4 already there can still place the chapters without the normalized copies, which
    # are deleted at the end of a successful run.
    durations = [float(p.get("seconds") or 0.0) for p in (plan.get("parts") or [])]
    if not durations:
        durations = [video.duration(f) for _, f in parts]
    bounds = config.boundaries(durations)
    total = video.duration(final)
    plan["chapters"] = []
    if config.YT_CHAPTERS:
        plan["chapters"] = chapters.build(bounds, total=total,
                                          marks=compilation.marks(titles, bounds),
                                          min_gap=config.YT_CHAPTER_MIN_SECONDS)
    _save_plan(job, plan)
    if plan["chapters"]:
        job.add_log(f"{len(plan['chapters'])} chapters, one per video")

    if not plan.get("seo_title"):
        job.set("assembling", "writing the title and description")
        _compilation_seo(job, plan, titles, total)
        _save_plan(job, plan)
    if not (d / "thumbnail.jpg").exists():
        _compilation_thumbnail(job, plan, parts)
        _save_plan(job, plan)

    # The normalized copies are a transcode of videos that still exist: keeping them would double
    # a twenty-minute compilation's disk for nothing, and a re-stitch can make them again.
    for k in range(1, len(parts) + 1):
        (d / compilation.part_file(k)).unlink(missing_ok=True)
    (d / "final.txt").unlink(missing_ok=True)      # the concat demuxer list
    job.set("done", "finished", final=final.name)
    job.add_log(f"compilation of {len(parts)} videos finished ({compilation.fmt_length(total)})")


def _do_assemble(job: Job):
    """Re-concat the job's clips, redo the story board overlay and remix its audio into a
    new final.mp4 (story board assets are generated first if they are missing)."""
    d = job.dir
    job.set("assembling", "re-merging clips and mixing audio")
    plan = _load_plan(job)
    if job.storyboard and not storyboard.assets_ready(d, plan):
        _storyboard_assets(job, plan, len(job.clips))
        job.set("assembling", "re-merging clips and mixing audio")
    if job.captions and not plan.get("caption_words"):
        _caption_assets(job, plan)
        job.set("assembling", "re-merging clips and mixing audio")
    if _avatar_pending(job, plan):
        _avatar_assets(job, plan)
        job.set("assembling", "re-merging clips and mixing audio")
    music_file = d / "music.mp3" if (d / "music.mp3").exists() else None
    narration_file = d / "narration.wav" if (d / "narration.wav").exists() else None
    if (job.audio_mode or "") == "music":
        narration_file = None
    final = _assemble(job, plan, narration_file, music_file)
    _drop_upscales(job)
    job.set(final=final.name, assemble_requested=False, needs_assemble=False)
    job.add_log("final video reassembled")


def _run_maintenance(job: Job):
    """Regenerate requested clips and/or reassemble the final video of a finished job.

    Never leaves the job in a failed state: on cancel or error the job returns to done
    with an explanatory detail."""
    d = job.dir
    try:
        total = len(job.regen_queue)
        done_count = 0
        if job.regen_queue:
            plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
            client = comfy.ComfyClient()
            t2v = client.load_workflow(config.model_workflow(job.video_model))
            i2v = client.load_workflow(config.model_workflow(job.video_model, True))
            while job.regen_queue:
                _check_cancel(job)
                i = job.regen_queue[0]
                job.set("clips", f"regenerating clip {i} ({done_count + 1}/{total})")
                _generate_clip(job, client, t2v, i2v, plan, i, d / f"clip_{i:02d}.mp4",
                               f"regenerating clip {i}")
                job.regen_queue.pop(0)
                done_count += 1
                job.set(needs_assemble=True,
                        stale_clips=[c for c in job.stale_clips if c != i])
                job.add_log(f"clip {i} regenerated (new seed)")
        if job.assemble_requested:
            _check_cancel(job)
            _do_assemble(job)
        job.set("done", "finished")
    except JobCancelled:
        job.regen_queue = []
        job.cancel_requested = False
        job.set("done", "regeneration cancelled", assemble_requested=False)
        job.add_log("regeneration cancelled by user")
    except Exception as e:  # noqa: BLE001
        job.add_log(f"REGEN FAILED: {e}")
        job.add_log(traceback.format_exc()[-800:])
        job.set("done", f"regeneration failed: {e}", assemble_requested=False)


def _run(job: Job):
    d = job.dir
    d.mkdir(parents=True, exist_ok=True)
    mode = job.audio_mode or "narration+music"

    # 1. source text: transcript for YouTube jobs, the user's prompt otherwise
    tfile = d / "transcript.txt"
    if job.source in PROMPT_SOURCES:
        text = job.topic
    elif job.source == REPO_SOURCE:
        # written by the draft thread; a non-YouTube job must never reach get_transcript_info
        if not tfile.exists() or tfile.stat().st_size <= 50:
            raise RuntimeError("the repository README was not fetched - resume the job to retry")
        text = tfile.read_text(encoding="utf-8")
        job.add_log("reusing the fetched README")
    else:
        text = _youtube_transcript(job)

    _check_cancel(job)
    # 2. story plan: title, visual style, story text (reused on resume)
    pfile = d / "plan.json"
    plan = None
    if pfile.exists():
        try:
            old_plan = json.loads(pfile.read_text(encoding="utf-8"))
            if old_plan.get("narration") and old_plan.get("style"):
                plan = old_plan
                job.add_log("resume: reusing story plan")
        except Exception:  # noqa: BLE001
            plan = None
    if plan is None:
        job.set("planning", "asking vLLM for the story plan")
        target_words = _length_params(job)[0]
        plan = llm.write_narration(text, words=target_words,
                                   art_style=config.ART_STYLES.get(job.art_style, ""),
                                   channel=job.channel)
        _fit_narration(job, plan, target_words)
    plan["audio_mode"] = mode
    if "cast_used" not in plan and cast.active(job.channel):   # URL jobs and pre-selection plans
        job.add_log("cast in this video: " + (", ".join(_set_cast_used(job, plan)) or "none"))
    else:
        plan.setdefault("cast_used", [])   # never leave it unset: None means "legacy plan, all cast"
    if not plan.get("seo_title"):
        job.set("planning", "writing SEO title and description")
        try:
            plan.update(llm.write_seo(plan.get("title", ""), plan["narration"], text,
                                      channel=job.channel, used=plan.get("cast_used")))
            job.add_log(f"seo title: {plan.get('seo_title', '')}")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"seo metadata failed ({e}); continuing without it")
    # The video only owes the viewer something when the script promised it - decided from the final
    # narration, never from the channel or the topic. Key PRESENCE is the guard, not truthiness:
    # "promises nothing" is a legitimate falsy answer that must be remembered, or every resume
    # re-decides (and _chat runs at temperature 0.7, so it could decide differently).
    # "music" mode is skipped: what the viewer hears there is the song, and its lyrics are not
    # written until further down. A story video is skipped too - a kids story hands the viewer
    # nothing, and _draft_story already wrote the key away so no resume re-decides it.
    if "resource" not in plan and mode != "music" and not _is_story(job):
        job.set("planning", "checking whether the script promises the viewer anything")
        try:
            res, origin, url = _viewer_document(job, plan, text)
            plan["resource"], logs = resource.attach(
                d, plan, res, channel=(job.channel or {}).get("name", ""), logo=_logo_for(job),
                origin=origin, source_url=url)
            for line in logs:
                job.add_log(line)
        except Exception as e:  # noqa: BLE001 - a missing hand-out never costs a video
            job.add_log(f"viewer resource failed ({e}); continuing without one")
    elif resource.fold_into_description(plan):
        # run 1 had SEO fail, so there was no description to append to; run 2 regenerated it
        job.add_log("resource appended to the regenerated description")
    (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job.set(title=plan.get("title", ""))
    job.add_log(f"plan: '{plan.get('title')}', {len(plan['narration'].split())} story words")

    narration_file = None
    music_file = None
    beats = None

    if _is_story(job):
        # 3. the spoken story: every line in its speaker's own voice, laid out on a clip grid built
        #    around the lines. Nothing is transcribed and nothing is aligned - we wrote the words
        #    and we built the track, so every timestamp is already exact.
        _check_cancel(job)
        if not (plan.get("story") or {}).get("lines"):
            raise RuntimeError("this story has no spoken lines - request a new draft")
        narration_file, durs, line_seconds = _story_audio(job, plan)
        n = len(durs)
        beats = (plan["story"]["beats"] + [{}] * n)[:n]
        timed = story.timeline(plan["story"]["lines"], line_seconds, durs)
        segments = [t["text"] for t in timed]
        timelines = [story.format_timeline(t) for t in timed]
        aligned = True                      # we wrote every word: there is nothing to mis-hear
        kind = "story being told"
        sync_text = plan["narration"]
        plan["timed_segments"] = timed
        plan["lyric_align"] = {"source": "story", "aligned": True, "ratio": 1.0,
                               "words": len(plan["narration"].split()), "inferred": 0,
                               "wps": round(len(plan["narration"].split())
                                            / max(0.1, sum(line_seconds)), 2)}
    elif mode == "music":
        # 3. music first: the actual track length drives the clip count, and the
        #    lyrics drive the per-clip scenes so the song matches each clip.
        _check_cancel(job)
        budget = _song_budget(job)          # sung words per second the lyrics were sized for
        given = (job.lyrics or "").strip()  # the user's own words: sung verbatim, and they set the length
        _lw, l_target, l_max, l_range = _length_params(job)
        if given:
            # frozen on the first pass: KIDS_SONG_WPS is editable at runtime, so a resume after a
            # Settings change must not hand the model a different ceiling than try 1 got
            fz = plan.get("lyric_length")
            if not fz:
                fz = {"words": len(llm.clean_lyrics(given).split()), "wps": budget,
                      "target": round(l_target, 2), "max": round(l_max, 2)}
                plan["lyric_length"] = fz
                _save_plan(job, plan)
            l_target, l_max, l_range = float(fz["target"]), float(fz["max"]), None
            job.add_log(f"song length from your lyrics: {fz['words']} sung words, "
                        f"about {l_target:.0f}s (hard cut {l_max:.0f}s)")
        elif not l_range:
            l_target = config.TARGET_VIDEO_SECONDS
            l_max = (min(360.0, config.TARGET_VIDEO_SECONDS * 2 + 30)
                     if config.TARGET_VIDEO_SECONDS else 360.0)
        # song candidates (resume-safe; each is rendered, whisper-timed and paced at most once):
        # [{caption, lyrics, file, reason, duration, words, ratio, wps}]; the chosen one is copied
        # to music.mp3. try 2 exists when try 1 was sung too fast or ended short of the target range.
        tries = plan.setdefault("music_tries", [])
        music_file = d / "music.mp3"
        if job.music_upload:
            # the user's own track, copied to music.mp3 when the job was created. Nothing is
            # written, rendered or paced: the recording is the song and it is the length.
            if not (music_file.exists() and music_file.stat().st_size > 1000):
                raise RuntimeError("the uploaded music track is missing from the job folder - "
                                   "create the video again with the track uploaded")
            job.add_log(f"music: your uploaded track ({job.music_upload})")
        elif music_file.exists() and music_file.stat().st_size > 1000 and (plan.get("music_final") or not tries):
            job.add_log("resume: reusing music track")
        else:
            why = ""        # why try 2 exists: "fast" | "short" | "" (one try only)
            for k in (1, 2):
                _check_cancel(job)
                if k == 2:
                    # frozen once try 2 exists - a resume must never re-decide the branch
                    why = ((tries[1].get("reason") or "fast") if len(tries) > 1
                           else llm.retry_reason(tries[0], budget, l_range))
                    if not why:
                        break
                if len(tries) < k:
                    if k == 1 and (plan.get("music") or {}).get("caption"):
                        mp = plan["music"]
                    else:
                        job.set("music", ("writing the music for your lyrics with vLLM" if given else
                                          "writing music caption and lyrics with vLLM") if k == 1 else
                                "song too fast - writing a slower version with vLLM" if why == "fast" else
                                "song too short - writing a longer version with vLLM")
                        slow = ({"measured": tries[0]["wps"], "budget": budget,
                                 "previous_lyrics": tries[0]["lyrics"]} if why == "fast" else None)
                        longer = ({"measured": tries[0]["duration"], "low": l_range[0],
                                   "target": l_target, "previous_lyrics": tries[0]["lyrics"]}
                                  if why == "short" else None)
                        mp = llm.write_music(plan.get("title", ""), plan["narration"], text,
                                             standalone=True, seconds=l_target,
                                             max_seconds=l_max, seconds_range=l_range,
                                             channel=job.channel, used=plan.get("cast_used"),
                                             song_voice=job.song_voice, slow_down=slow,
                                             lengthen=longer, given_lyrics=given)
                        if mp.get("lyrics_kept") == "verbatim-fallback":
                            job.add_log("the LLM altered your lyrics; your own words are used instead")
                        plan["music"] = mp
                    tries.append({"caption": mp["caption"], "lyrics": mp.get("lyrics", ""),
                                  "file": f"music_try{k}.mp3", "reason": why or "first"})
                    _save_plan(job, plan)
                t = tries[k - 1]
                f = d / t["file"]
                if not (f.exists() and f.stat().st_size > 1000):
                    _gpu_gate(job)
                    job.set("music", "generating music with ComfyUI" if k == 1 else
                            "generating a slower version of the song with ComfyUI" if why == "fast" else
                            "generating a longer version of the song with ComfyUI")
                    mclient = comfy.ComfyClient()
                    mwf = mclient.build_music(mclient.load_workflow(config.MUSIC_WORKFLOW),
                                              t["caption"], t["lyrics"], l_max,
                                              f"videogen/{job.id}/music{k}")
                    mclient.run(mwf, f, on_progress=lambda m: job.set("music", f"music: {m}"),
                                should_stop=lambda: job.cancel_requested,
                                cleanup=config.COMFY_CLEANUP, on_cleanup_failed=_cleanup_warn(job))
                if "wps" not in t:
                    job.set("music", "timing the sung lyrics (whisper)")
                    heard = lyrics.transcribe_words(f, initial_prompt=llm.clean_lyrics(t["lyrics"])[:600],
                                                    on_progress=_progress(job, "music"))
                    aligned_words, ratio = lyrics.align_to_lyrics(heard, t["lyrics"])
                    t.update(duration=round(tts.audio_duration(f), 3), words=aligned_words,
                             ratio=round(ratio, 3), wps=lyrics.words_per_second(aligned_words),
                             whisper_model=config.WHISPER_MODEL)
                    _save_plan(job, plan)
                    job.add_log(f"song try {k}: {t['duration']:.1f}s, {t['wps']:.2f} sung words/s "
                                f"(budget {budget:g}), lyric match {ratio:.0%}")
                if k == 2:
                    job.add_log(("song too fast: re-rendered once with the same words, sung slower"
                                 if given else
                                 "song too fast: re-rendered once with a slower caption and fewer words")
                                if why == "fast" or not l_range else
                                f"song too short: re-rendered once with longer lyrics (try 1 was "
                                f"{tries[0]['duration']:.1f}s, target {l_range[0]:g}-{l_range[1]:g}s)")
            best = tries[0]
            if len(tries) > 1 and "wps" in tries[1]:        # try 2 was rendered and timed
                if why == "short":
                    # the longer re-write wins only if it lands closer to the target range AND the
                    # extra words were not crammed in over the pace budget
                    if (_range_miss(tries[1], l_range) < _range_miss(tries[0], l_range)
                            and not llm.needs_slowdown(tries[1]["wps"], budget)):
                        best = tries[1]
                elif (0 < tries[1]["wps"] < tries[0].get("wps", 0.0)
                        and not (given and _truncated(tries[1], l_max))):
                    # same words sung slower runs longer, so try 2 is the one that can hit the cut
                    best = tries[1]
            shutil.copyfile(d / best["file"], music_file)
            plan["music"] = {"caption": best["caption"], "lyrics": best["lyrics"]}
            plan["music_final"] = {"try": tries.index(best) + 1, "wps": best.get("wps", 0.0),
                                   "duration": best.get("duration", 0.0), "reason": why}
            _save_plan(job, plan)
            if len(tries) > 1:
                job.add_log(f"song: keeping try {tries.index(best) + 1} "
                            + ("(slowest singing)" if why == "fast" or not l_range
                               else f"(closest to {l_range[0]:g}-{l_range[1]:g}s)"))
        duration = tts.audio_duration(music_file)
        job.add_log(f"music track: {duration:.1f}s")
        if not job.music_upload and given and duration >= l_max - 1.0:
            job.add_log(f"the song hit the {l_max:.0f}s ceiling - its last lines may be missing")
        if job.music_upload and not plan.get("music_final"):
            # try 0 on purpose: `chosen` below stays None, so the track is transcribed with no
            # reference lyrics - there are none, the user brought a finished recording
            plan["music"] = {"caption": "", "lyrics": "", "uploaded": job.music_upload}
            plan["music_final"] = {"try": 0, "wps": 0.0, "duration": round(duration, 3),
                                   "reason": "upload"}
            _save_plan(job, plan)
        durs = config.clip_durations(duration, _clip_seconds(job, duration), job.video_model)
        n = len(durs)
        music_plan = plan.get("music") or {}
        final = plan.get("music_final") or {}
        chosen = tries[final["try"] - 1] if final.get("try") and final["try"] <= len(tries) else None
        if chosen is not None and "words" in chosen:
            words, ratio = chosen["words"], chosen.get("ratio", 0.0)
        elif plan.get("music_words"):   # an uploaded track (or a legacy job) timed on an earlier pass
            words = plan["music_words"]["words"]
            ratio = plan["music_words"].get("ratio", 0.0)
            job.add_log("resume: reusing the word timing of the track")
        else:   # an uploaded track, or a legacy job: music.mp3 without cached timing
            job.set("music", "timing the words in your track (whisper)" if job.music_upload
                    else "timing the sung lyrics (whisper)")
            heard = lyrics.transcribe_words(music_file, initial_prompt=llm.clean_lyrics(music_plan.get("lyrics", ""))[:600],
                                            on_progress=_progress(job, "music"))
            words, ratio = lyrics.align_to_lyrics(heard, music_plan.get("lyrics", ""))
            # cached so a resume never pays for whisper twice
            plan["music_words"] = {"words": words, "ratio": round(ratio, 3),
                                   "whisper_model": config.WHISPER_MODEL}
            _save_plan(job, plan)
        timed = lyrics.timeline(words, config.boundaries(durs))
        segments = [t["text"] for t in timed]
        timelines = [lyrics.format_timeline(t) for t in timed]
        aligned = ratio >= config.LYRIC_ALIGN_MIN_RATIO
        plan["timed_segments"] = timed
        plan["lyric_align"] = {"ratio": round(ratio, 3), "aligned": aligned, "words": len(words),
                               "inferred": sum(1 for w in words if w.get("src") == "inferred"),
                               "wps": lyrics.words_per_second(words), "whisper_model": config.WHISPER_MODEL,
                               "try": final.get("try", 1),
                               "source": "upload" if job.music_upload else "generated"}
        sync_text = llm.clean_lyrics(music_plan.get("lyrics", "")) or plan["narration"]
        kind = "song being played"
        if job.music_upload and len(words) < config.MIN_HEARD_WORDS:
            # An instrumental upload. Empty segments would hand every clip the "no words are
            # heard" prompt and the story would never reach a single scene, so the story is
            # split across the clips instead (llm.write_scenes does that when segments is None).
            # The threshold is not `not words`: whisper hallucinates a stray word or two over music.
            segments, timelines, aligned = None, None, True
            kind = "instrumental music track"
            job.add_log(f"no singing heard in your track - the {n} scenes follow the story instead")
        else:
            sung = sum(1 for s in segments if s.strip())
            job.add_log(f"words timed from your track: heard in {sung}/{n} clip windows"
                        if job.music_upload else
                        f"lyrics timed: words heard in {sung}/{n} clip windows"
                        + ("" if aligned else " (whisper words only - they did not match the written lyrics)"))
    else:
        # 3. narration voiceover
        _check_cancel(job)
        narration_file = d / "narration.wav"
        if narration_file.exists() and narration_file.stat().st_size > 1000:
            duration = tts.audio_duration(narration_file)
            job.add_log("resume: reusing narration audio")
        else:
            ref_audio, ref_text, keep_voice = None, "", True
            if job.voice_id == voices.RANDOM_VOICE:
                keep_voice = False
                job.add_log("narration voice: default, a new random voice per segment")
            elif job.voice_id:
                v = voices.get_voice(job.voice_id)
                vpath = voices.voice_audio_path(job.voice_id)
                if v and vpath.exists():
                    ref_audio, ref_text = vpath, v.get("text", "")
                    job.add_log(f"narration voice: {v['name']}")
                else:
                    job.add_log(f"reference voice {job.voice_id} no longer exists - using the default voice")
            else:
                job.add_log("narration voice: default, kept the same for the whole video")
            job.set("tts", "synthesizing narration with Fish Speech")
            _gpu_gate(job, window=False)   # Fish shares the GPU with ComfyUI: one job speaks at a time
            duration = tts.synthesize(plan["narration"], narration_file,
                                      on_progress=_progress(job, "tts"),
                                      ref_audio=ref_audio, ref_text=ref_text, keep_voice=keep_voice)
            job.add_log(f"narration: {duration:.1f}s")
        durs = config.clip_durations(duration, _clip_seconds(job), job.video_model)
        n = len(durs)
        sync_text = plan["narration"]
        segments = None
        timelines = None
        aligned = True
        kind = "voiceover narration"

    # 4. scenes: one per audio segment so each clip matches what is heard
    _check_cancel(job)
    # cast: the first story character with an approved portrait opens clip 1 (resumable)
    if not (d / "cast_opening.jpg").exists():
        oc = cast.opening_character(job.channel, job.channel_id, used=plan.get("cast_used"))
        if oc is not None:
            try:
                ow, oh = config.THUMB_SIZES.get(job.aspect_ratio, (1280, 720))
                video.fit_image(channels.portrait_path(job.channel_id, oc["id"]), d / "cast_opening.jpg", ow, oh)
                plan["opening"] = {"name": oc["name"], "look": oc.get("look", "")}
                job.add_log(f"opening frame: {oc['name']}'s portrait")
            except Exception as e:  # noqa: BLE001 - fall back to text-to-video for clip 1
                job.add_log(f"cast portrait unusable ({e}); clip 1 starts from text")
    # scenes are written for these exact clip lengths; a legacy plan (no clip_seconds) keeps its scenes
    same_clips = "clip_seconds" not in plan or plan["clip_seconds"] == durs
    plan["clip_seconds"] = durs
    have = list(plan.get("scenes") or []) if same_clips else []
    if len(have) == n:
        job.add_log("resume: reusing scene prompts")
    else:
        job.set("scenes", f"writing {n} scene prompts with vLLM")
        if have:
            job.add_log(f"resume: reusing {len(have)} of {n} scene prompts")
        def scene_progress(m):
            _check_cancel(job)
            job.set("scenes", f"writing {m}")
        def review_issue(idx, issues):
            job.add_log(f"review fixed scene {idx}: " + "; ".join(issues)[:160])
        def review_failed(idx, err):
            job.add_log(f"scene {idx} kept unreviewed ({err})")
        def keep(written):
            # one LLM call per scene, two with the review on: an engine that dies at scene 18 of
            # 22 used to cost every one of them. Saved as they land, so a resume carries on.
            plan["scenes"] = written
            _save_plan(job, plan)
        plan["scenes"] = llm.write_scenes(sync_text, plan["style"], n, kind=kind, segments=segments,
                                          on_progress=scene_progress, on_review_issue=review_issue,
                                          on_review_failed=review_failed, on_scene=keep, done=have,
                                          channel=job.channel, used=plan.get("cast_used"),
                                          opening=plan.get("opening") if (d / "cast_opening.jpg").exists() else None,
                                          timelines=timelines, clip_secs=durs, aligned=aligned,
                                          beats=beats, model=job.video_model)
    plan["segments"] = segments if segments is not None else llm.split_narration(sync_text, n)
    scenes = plan["scenes"]
    (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job.add_log(f"{n} scenes written, one per audio segment")

    # 5. thumbnail: LLM prompt -> single-frame render -> 1280x720 JPG (non-fatal), then the text
    # variants drawn over copies of it
    thumb = d / "thumb_base.jpg"
    if thumb.exists() and thumb.stat().st_size > 1000:
        job.add_log("resume: reusing thumbnail")
    else:
        try:
            _check_cancel(job)
            job.set("thumbnail", "designing the thumbnail with vLLM")
            if not plan.get("thumbnail_prompt"):
                plan.update(llm.write_thumbnail(plan.get("title", ""), plan["narration"], plan["style"],
                                                channel=job.channel, used=plan.get("cast_used"),
                                                idea=plan.get("thumbnail_idea", "")))
                (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            _gpu_gate(job)
            job.set("thumbnail", "rendering the thumbnail frame")
            tw, th = _render_thumbnail(job, plan, thumb, on_progress=_progress(job, "thumbnail"),
                                       should_stop=lambda: job.cancel_requested)
            job.add_log(f"thumbnail ready ({tw}x{th})")
        except JobCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            job.add_log(f"thumbnail generation failed ({e}); continuing without it")
    _thumb_variants(job, plan)

    # 6. secondary music track (narration+music mode only; non-fatal; reused on resume)
    if mode == "narration+music" and (d / "music.mp3").exists() and (d / "music.mp3").stat().st_size > 1000:
        music_file = d / "music.mp3"
        job.add_log("resume: reusing music track")
    elif mode == "narration+music":
        try:
            job.set("music", "writing music caption and lyrics with vLLM")
            music_plan = llm.write_music(plan.get("title", ""), plan["narration"], text,
                                         channel=job.channel, used=plan.get("cast_used"))
            music_plan["lyrics"] = "[Instrumental]"  # background bed: never sing over the narration
            plan["music"] = music_plan
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            _gpu_gate(job)
            job.set("music", "generating music with ComfyUI")
            mclient = comfy.ComfyClient()
            mwf = mclient.build_music(mclient.load_workflow(config.MUSIC_WORKFLOW),
                                      music_plan["caption"], music_plan.get("lyrics", ""),
                                      sum(plan.get("clip_seconds")
                                          or [config.actual_clip_seconds(model=job.video_model)] * n) * 1.05 + 2,
                                      f"videogen/{job.id}/music")
            music_file = mclient.run(mwf, d / "music.mp3",
                                     on_progress=lambda m: job.set("music", f"music: {m}"),
                                     should_stop=lambda: job.cancel_requested,
                                     cleanup=config.COMFY_CLEANUP, on_cleanup_failed=_cleanup_warn(job))
            job.add_log("music track generated")
        except Exception as e:  # noqa: BLE001
            music_file = None
            job.add_log(f"music generation failed ({e}); continuing without music")

    # 6b. story board cards + related images (optional overlay; non-fatal)
    if job.storyboard and not storyboard.assets_ready(d, plan):
        _storyboard_assets(job, plan, n)
    elif job.storyboard:
        job.add_log("resume: reusing story board assets")

    # 6c. caption word timings (before the clips, so a failure surfaces early; non-fatal)
    if job.captions:
        _caption_assets(job, plan)

    # 6d. the presenter avatar. After the word timings on purpose: the segment windows are
    # snapped to gaps between spoken words, and those timings are already on the plan.
    _avatar_assets(job, plan)

    # 7. clips
    mp = job.megapixels if job.megapixels is not None else config.DEFAULT_MEGAPIXELS
    model = job.video_model or config.DEFAULT_VIDEO_MODEL
    if any(not _clip_on_disk(d / f"clip_{i:02d}.mp4") for i in range(1, n + 1)):
        _gpu_gate(job)   # park in the previous status: the `clips` timing starts with the first real render
    job.set("clips", f"generating {n} clips ({model}, {job.aspect_ratio}, {mp:g} MP)",
            clips_total=n, clips_done=0)
    client = comfy.ComfyClient()
    t2v = client.load_workflow(config.model_workflow(job.video_model))
    i2v = client.load_workflow(config.model_workflow(job.video_model, True))
    prev_clip = None
    job.clips = []
    for i, scene in enumerate(scenes, 1):
        _check_cancel(job)
        dest = d / f"clip_{i:02d}.mp4"
        if _clip_on_disk(dest):
            prev_clip = dest
            job.clips.append(dest.name)
            job.set(clips_done=i)
            job.add_log(f"resume: clip {i}/{n} already generated")
            continue
        _generate_clip(job, client, t2v, i2v, plan, i, dest, f"clip {i}/{n}")
        prev_clip = dest
        job.clips.append(dest.name)
        job.set(clips_done=i)
        how = ("first-frame (cast portrait)" if i == 1 and plan.get("opening") and (d / "cast_opening.jpg").exists()
               else "text-to-video" if i == 1 else "first-frame")
        job.add_log(f"clip {i}/{n} done ({how})")

    _check_cancel(job)
    # 8. assemble
    # every render this job needed is on disk and assembly is local ffmpeg, so the GPU goes back
    # now rather than at the end of the run: the next job starts rendering while this one encodes
    gpu.release(job.id)
    job.set("assembling", "concatenating clips and mixing audio")
    final = _assemble(job, plan, narration_file, music_file)
    job.set("done", "finished", final=final.name)
    job.add_log("final video ready")


# ---------------------------------------------------------------------------
JOB_DIR_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")   # output/<job id> folders


def dir_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except OSError:
        return 0


def remove_job_dir(path: Path) -> tuple:
    """Delete a job folder. Returns (bytes_freed, warning_or_None).

    job.json goes first so a half-deleted folder can never resurrect the job at the next
    start; read-only files are made writable; files still held open (Windows keeps a video
    locked while the browser streams it) are retried for a few seconds. Whatever stays
    locked is reported in the warning and removed by the startup sweep."""
    path = Path(path)
    if not path.exists():
        return 0, None
    size = dir_size(path)
    (path / "job.json").unlink(missing_ok=True)

    def onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    for attempt in range(4):          # ~5 s in total before giving the leftovers to the sweep
        shutil.rmtree(path, onerror=onerror)
        if not path.exists():
            return size, None
        time.sleep(0.5 * (attempt + 1))
    left = dir_size(path)
    n = sum(1 for f in path.rglob("*") if f.is_file())
    return size - left, (f"{n} file(s) ({left / 1e6:.1f} MB) are still in use and could not be "
                         f"removed yet - they are cleaned up automatically at the next server start")


def sweep_deleted_dirs() -> int:
    """Remove job folders whose job.json is gone (a delete that hit locked files)."""
    freed = 0
    for d in config.OUTPUT_DIR.glob("*"):
        if d.is_dir() and JOB_DIR_RE.match(d.name) and not (d / "job.json").exists():
            got, warn = remove_job_dir(d)
            freed += got
            print(f"leftover job folder {d.name}: " + (warn or f"removed ({got / 1e6:.1f} MB)"))
    return freed


class JobManager:
    """Two lanes over one GPU: generation is serialized by the lease in services/gpu.py, while a
    second job gets on with the work that needs no GPU - its transcript, its vLLM calls, whisper
    timing, ffmpeg assembly. A job that reaches a generation step while another holds the lease is
    parked back in the queue with everything it has done kept, and the lane moves to a job that
    still has work it can do."""

    LANES = 2

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.running: set[str] = set()
        self.lock = threading.Lock()
        self.workers = [threading.Thread(target=self._loop, daemon=True) for _ in range(self.LANES)]
        self._load_existing()
        for w in self.workers:
            w.start()
        threading.Thread(target=self._sweep_comfy, daemon=True).start()

    @property
    def current(self) -> str | None:
        """The job that owns the GPU, or any running job when nobody does. What "in progress"
        means to a user watching the queue: the one whose renders they are waiting behind."""
        held = gpu.holder()
        if held and held in self.running:
            return held
        return next(iter(self.running), None)

    def _sweep_comfy(self):
        """Scrub orphaned videogen assets on the ComfyUI server (retries/crashes leave some)."""
        if not config.COMFY_CLEANUP:
            return
        try:
            with self.lock:
                skip = [j for j in (list(self.running) + list(self.queue)) if j]
            n = comfy.ComfyClient().sweep_orphans(skip_ids=skip)
            if n:
                print(f"comfy sweep: scrubbed {n} orphaned file(s)")
        except Exception as e:  # noqa: BLE001
            print(f"comfy sweep failed: {e}")

    def _load_existing(self):
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        sweep_deleted_dirs()
        for p in sorted(config.OUTPUT_DIR.glob("*/job.json")):
            try:
                j = Job.load(p)
                if j.status in ("awaiting_approval", "planned"):   # waiting for the user, not for us
                    self.jobs[j.id] = j
                    continue
                if j.status == "drafting":
                    self.jobs[j.id] = j
                    self._start_draft(j)
                    continue
                # A compilation is local ffmpeg work: restart its own thread rather than put it in
                # the GPU queue, where it would wait behind renders it needs nothing from.
                if j.source == compilation.SOURCE and j.status not in ("done", "failed"):
                    j.close_running(j.updated)
                    j.add_log("server restarted - the compilation continues where it stopped")
                    self.jobs[j.id] = j
                    self._start_compilation(j)
                    continue
                if j.status not in ("done", "failed"):
                    j.pause_requested = False
                    if j.cancel_requested:
                        j.set("failed", "cancelled by user", error="cancelled by user")
                        self.jobs[j.id] = j
                        continue
                    j.close_running(j.updated)   # the downtime is not active stage time
                    j.set("queued", "resuming after server restart")
                    j.add_log("server restarted - job requeued, finished steps will be reused")
                    self.queue.append(j.id)
                self.jobs[j.id] = j
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _resolve_choices(aspect_ratio: str, art_style: str, video_length: str,
                         audio_mode: str, voice_id: str, channel: dict | None,
                         storyboard_flag=None, megapixels=None, resolution=None, song_voice: str = "",
                         captions_flag=None, polish_flag=None, video_model=None):
        """Per-video choices: request value -> channel default -> global default; validated.
        Megapixels, the video model and the output resolution have no channel level: request value
        -> Settings default.
        Real/tech-video channels always use the built-in live-action look (art_style stays "");
        the story board overlay and the burned-in captions default to the channel setting (both on
        for tech channels)."""
        ch = channel or {}
        defaults = ch.get("defaults") or {}

        def _flag(given, key):
            if given is None:
                given = defaults.get(key)
                if given is None:
                    given = channels.bool_default(key, ch.get("video_type"))
            return bool(given)

        storyboard_flag = _flag(storyboard_flag, "storyboard")
        captions_flag = _flag(captions_flag, "captions")
        polish_flag = _flag(polish_flag, "polish")
        audio_mode = audio_mode or defaults.get("audio_mode") or config.AUDIO_MODE
        if audio_mode not in config.AUDIO_MODES:
            raise ValueError(f"unsupported audio mode: {audio_mode}")
        voice_id = "" if audio_mode == "music" else (voice_id or defaults.get("voice_id") or "")
        song_voice = (song_voice or defaults.get("song_voice")
                      or channels.default_song_voice(ch.get("audience")))
        if song_voice not in config.SONG_VOICES:
            raise ValueError(f"unsupported song voice: {song_voice}")
        aspect_ratio = aspect_ratio or defaults.get("aspect_ratio") or config.DEFAULT_ASPECT_RATIO
        if aspect_ratio not in config.ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")
        if megapixels in (None, "", 0):
            mp = float(config.DEFAULT_MEGAPIXELS)
        else:
            try:
                mp = float(megapixels)
            except (TypeError, ValueError):
                raise ValueError(f"unsupported megapixels: {megapixels}")
        if not any(abs(mp - o) < 1e-6 for o in config.MEGAPIXELS_OPTIONS):
            raise ValueError(f"unsupported megapixels: {megapixels}")
        mp = round(mp, 1)
        resolution = resolution or config.DEFAULT_RESOLUTION
        if resolution not in config.OUTPUT_RESOLUTIONS:
            raise ValueError(f"unsupported output resolution: {resolution}")
        video_model = video_model or config.DEFAULT_VIDEO_MODEL
        if video_model not in config.VIDEO_MODELS:
            raise ValueError(f"unsupported video model: {video_model}")
        if ch.get("video_type") in config.REALISTIC_VIDEO_TYPES:
            art_style = ""
        else:
            art_style = art_style or defaults.get("art_style") or config.DEFAULT_ART_STYLE
            if art_style not in config.ART_STYLES:
                raise ValueError(f"unsupported art style: {art_style}")
        video_length = video_length or defaults.get("video_length") or config.DEFAULT_VIDEO_LENGTH
        if video_length not in config.VIDEO_LENGTHS:
            raise ValueError(f"unsupported video length: {video_length}")
        snap = channels.snapshot(ch) if ch.get("id") else {}
        # captions sits beside storyboard and the video model before the megapixels, NOT at the
        # end: callers and tests index [-1] for the output resolution
        return (aspect_ratio, art_style, video_length, audio_mode, voice_id, song_voice, ch.get("id", ""), snap,
                storyboard_flag, captions_flag, polish_flag, video_model, mp, resolution)

    @staticmethod
    def _resolve_thumb_text(thumb_text_flag, channel: dict | None) -> bool:
        """Thumbnail text: request value -> channel default -> on.

        A separate resolver for the same reason as _resolve_avatar: _resolve_choices returns a
        positional tuple that callers unpack and index, so appending to it is a trap."""
        defaults = (channel or {}).get("defaults") or {}
        given = thumb_text_flag
        if given is None:
            given = defaults.get("thumb_text")
            if given is None:
                given = channels.bool_default("thumb_text", (channel or {}).get("video_type"))
        return bool(given)

    @staticmethod
    def _resolve_avatar(avatar_id, avatar_model, avatar_coverage, channel: dict | None):
        """Presenter choices: request value -> channel default -> global default.

        Deliberately NOT folded into _resolve_choices: that function returns a positional
        tuple several callers unpack and index ([-1] is the output resolution), so appending
        to it is a trap. Returns (avatar_id, avatar_model, avatar_coverage).

        A coverage that names an avatar the channel does not have is demoted to "none" rather
        than raising: the channel can be edited (or its avatar deleted) between planning a
        video and setting it up, and losing the whole job over a missing portrait would be
        the wrong trade."""
        ch = channel or {}
        defaults = ch.get("defaults") or {}
        avatars = [a for a in (ch.get("avatars") or []) if isinstance(a, dict) and a.get("id")]

        coverage = (avatar_coverage or defaults.get("avatar_coverage")
                    or config.DEFAULT_AVATAR_COVERAGE)
        if coverage not in config.AVATAR_COVERAGES:
            raise ValueError(f"unsupported avatar coverage: {coverage}")
        model = avatar_model if avatar_model is not None else defaults.get("avatar_model") or ""
        model = str(model or "")
        if model and model not in config.AVATAR_MODELS:
            raise ValueError(f"unsupported avatar model: {model}")
        chosen = str(avatar_id or defaults.get("avatar_id") or "")
        if chosen and not any(a["id"] == chosen for a in avatars):
            chosen = ""
        if not chosen:
            fallback = channels.default_avatar(ch)
            chosen = fallback["id"] if fallback else ""
        # no avatar to animate, or the budget is switched off entirely -> no presenter
        if not chosen or config.avatar_budget_seconds(model) <= 0:
            return "", model, "none"
        return chosen, model, coverage

    def submit(self, url: str, aspect_ratio: str = "",
               art_style: str = "", video_length: str = "",
               audio_mode: str = "", voice_id: str = "",
               channel: dict | None = None, storyboard_flag=None, megapixels=None,
               owner: str = "", resolution: str = "", song_voice: str = "",
               lyrics: str = "", music_upload_id: str = "", video_mode: str = "",
               captions_flag=None, polish_flag=None, video_model: str = "",
               avatar_id: str = "", avatar_model=None, avatar_coverage: str = "",
               thumb_text_flag=None) -> Job:
        video_mode = resolve_video_mode(video_mode, channel, "youtube", lyrics, music_upload_id)
        if video_mode == "story":
            # forced BEFORE the resolve: that blanks voice_id whenever the mode lands on "music",
            # and a kids channel usually defaults to music - the narrator would lose their voice
            audio_mode, storyboard_flag = "narration+music", False
        aspect_ratio, art_style, video_length, audio_mode, voice_id, sv, cid, snap, sb, cap, pol, vm, mp, res = \
            self._resolve_choices(aspect_ratio, art_style, video_length, audio_mode, voice_id, channel,
                                  storyboard_flag, megapixels, resolution, song_voice, captions_flag,
                                  polish_flag, video_model)
        av_id, av_model, av_cov = self._resolve_avatar(avatar_id, avatar_model, avatar_coverage, channel)
        thumb_text = self._resolve_thumb_text(thumb_text_flag, channel)
        lyrics = _resolve_lyrics(lyrics, channel, audio_mode)   # audio_mode is defaulted by the resolve above
        up = _resolve_music_upload(music_upload_id, audio_mode, lyrics)
        if lyrics or up:
            video_length = "Auto"      # the lyrics / the track set the length: keep no unused preset
        job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6], url=url.strip(),
                  aspect_ratio=aspect_ratio, megapixels=mp, video_model=vm,
                  resolution=res, audio_mode=audio_mode, voice_id=voice_id,
                  song_voice=sv, art_style=art_style, video_length=video_length, channel_id=cid, channel=snap,
                  storyboard=sb, captions=cap, polish=pol, thumb_text=thumb_text, lyrics=lyrics,
                  avatar_id=av_id, avatar_model=av_model, avatar_coverage=av_cov,
                  owner=owner or (channel or {}).get("owner", ""),
                  video_mode=video_mode,
                  status="drafting" if video_mode == "story" else "queued",
                  detail="writing the story with the LLM" if video_mode == "story" else "",
                  music_upload=(up or {}).get("name", ""),
                  music_seconds=float((up or {}).get("seconds", 0.0)))
        job.save()
        _log_avatar_plan(job)
        _install_upload(job, up)
        with self.lock:
            self.jobs[job.id] = job
            if video_mode != "story":
                self.queue.append(job.id)
        # a story is always reviewed before anything renders, whatever it was created from, so even
        # a URL story job drafts in its own thread instead of going straight into the GPU queue
        if video_mode == "story":
            self._start_draft(job)
        return job

    def _next_job(self) -> str | None:
        """Pop the first job this lane can actually move forward (caller holds the lock).

        A job parked at a generation step stays put while the thing it waited for is still true -
        the window shut, or the GPU held by someone else - so a lane is never spent picking one up
        only to park it again. A job already running in the other lane is skipped too: a priority
        regeneration puts the job it paused back in the queue while it is still finishing its
        current step."""
        blocked = not schedule.is_open() or gpu.busy()
        for idx, cand in enumerate(self.queue):
            j = self.jobs.get(cand)
            if j is not None and (j.gpu_wait and blocked or cand in self.running):
                continue
            return self.queue.pop(idx)
        return None

    def _loop(self):
        while True:
            with self.lock:
                jid = self._next_job()
            if jid is None:
                time.sleep(1)
                continue
            job = self.jobs.get(jid)
            if job is None or (job.status in ("done", "failed")
                               and not job.regen_queue and not job.assemble_requested):
                continue
            job.pause_requested = False
            with self.lock:
                self.running.add(jid)
            try:
                run_job(job)
            finally:
                with self.lock:
                    self.running.discard(jid)
                    # parked: back to the queue, so the other lane keeps its LLM/TTS work moving
                    if job.gpu_wait and jid not in self.queue:
                        self.queue.append(jid)
                threading.Thread(target=self._sweep_comfy, daemon=True).start()

    def cancel(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in ("done", "failed"):
                raise ValueError("job already finished")
            if job.status == "planned":   # never queued and never started: discard it outright
                job.set("failed", "discarded before setup", error="discarded before setup")
                job.add_log("discarded by user before setup")
                return
            if job_id in self.queue:
                self.queue.remove(job_id)
                if job.regen_queue or job.assemble_requested:
                    job.regen_queue = []
                    job.set("done", "request cancelled", assemble_requested=False)
                else:
                    job.set("failed", "cancelled by user", error="cancelled by user")
                return
            if job.status in ("drafting", "awaiting_approval"):
                # Neither state has a worker that would see the flag: a draft is blocked in an LLM
                # request no other thread can interrupt, and a job awaiting approval is not running
                # at all. Finish the cancel here rather than leaving it on "cancelling..." forever.
                # The flag stays set so a draft that lands afterwards drops its result; the two
                # calls that start the next phase, approve_script and redraft_script, clear it.
                drafting = job.status == "drafting"
                job.cancel_requested = True
                if drafting and _drafted(job):
                    job.set("awaiting_approval", "rewrite cancelled - the earlier script is kept")
                    job.add_log("rewrite cancelled by user - the earlier script is kept")
                    return
                job.set("failed", "cancelled by user", error="cancelled by user")
                job.add_log("cancelled by user while the script was being written" if drafting
                            else "cancelled by user before the script was approved")
                return
            job.cancel_requested = True
            job.set(detail="cancelling...")
        try:  # abort the in-flight ComfyUI render if it is ours (outside the lock)
            comfy.ComfyClient().interrupt_if_ours(f"videogen/{job_id}")
        except Exception:  # noqa: BLE001
            pass

    def resume(self, job_id: str):
        """Requeue a failed job (connection loss, service down, crash): it re-enters the
        queue and continues from the step it failed at - artifacts on disk are reused."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "failed":
                raise ValueError("only failed jobs can be resumed")
            job.cancel_requested = False
            job.pause_requested = False
            if job.plan_id and not job.setup_done:   # discarded before setup: back to the setup form
                job.set("planned", "needs setup - open the job to choose length and resolution",
                        error="", archived=False)
                job.add_log("restored by user - waiting to be set up")
                return job
            restart_draft = job.source in DRAFT_SOURCES and not job.script_approved
            # a compilation needs nothing from the GPU, so it resumes in its own thread rather
            # than queueing behind renders - the same routing its submit uses
            restart_stitch = job.source == compilation.SOURCE
            if restart_draft:
                job.set("drafting", "resumed - writing script with the LLM", error="", archived=False)
                job.add_log("resumed by user - restarting the script draft")
            elif restart_stitch:
                job.set("assembling", "resumed - joining the videos", error="", archived=False)
                job.add_log("resumed by user - finished steps will be reused")
            else:
                job.set("queued", "resumed - waiting for the queue", error="", archived=False)
                job.add_log("resumed by user - finished steps will be reused")
                self.queue.append(job_id)
        if restart_draft:
            self._start_draft(job)
        elif restart_stitch:
            self._start_compilation(job)
        return job

    def submit_prompt(self, text: str, source_kind: str,
                      aspect_ratio: str = "", art_style: str = "",
                      video_length: str = "",
                      audio_mode: str = "", voice_id: str = "",
                      channel: dict | None = None, storyboard_flag=None, megapixels=None,
                      owner: str = "", resolution: str = "", song_voice: str = "",
                      start: bool = True, title: str = "", plan: dict | None = None,
                      page_scroll=None, lyrics: str = "", music_upload_id: str = "",
                      video_mode: str = "", captions_flag=None, polish_flag=None,
                      video_model: str = "", avatar_id: str = "", avatar_model=None,
                      avatar_coverage: str = "", thumb_text_flag=None) -> Job:
        """Create a prompt job. start=False parks it at "planned" instead: the job exists and is
        listed, but nothing runs and no LLM is called until `setup_planned` hands it the user's
        parameters. plan: {plan_id, day, date, hook, thumb_idea} when it came from a weekly plan."""
        text = (text or "").strip()
        if source_kind not in DRAFT_SOURCES:
            raise ValueError("invalid source kind")
        if source_kind == REPO_SOURCE:
            if (channel or {}).get("video_type") != "tech":
                raise ValueError("repository videos are only available on Tech channels")
            text = repo.normalize_url(text)      # ValueError here becomes a 400 at submit time
        elif len(text) < 3:
            raise ValueError("topic/description/transcript is too short")
        # only a repo job ever produces a page.png to scroll
        page_scroll = bool(source_kind == REPO_SOURCE
                           and (True if page_scroll is None else page_scroll))
        video_mode = resolve_video_mode(video_mode, channel, source_kind, lyrics, music_upload_id)
        if video_mode == "story":       # see the note in submit(): forced before the resolve
            audio_mode, storyboard_flag = "narration+music", False
        aspect_ratio, art_style, video_length, audio_mode, voice_id, sv, cid, snap, sb, cap, pol, vm, mp, res = \
            self._resolve_choices(aspect_ratio, art_style, video_length, audio_mode, voice_id, channel,
                                  storyboard_flag, megapixels, resolution, song_voice, captions_flag,
                                  polish_flag, video_model)
        av_id, av_model, av_cov = self._resolve_avatar(avatar_id, avatar_model, avatar_coverage, channel)
        thumb_text = self._resolve_thumb_text(thumb_text_flag, channel)
        lyrics = _resolve_lyrics(lyrics, channel, audio_mode)   # audio_mode is defaulted by the resolve above
        up = _resolve_music_upload(music_upload_id, audio_mode, lyrics)
        if lyrics or up:
            video_length = "Auto"      # the lyrics / the track set the length: keep no unused preset
        plan = plan or {}
        job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
                  url=text if source_kind == REPO_SOURCE else "",
                  source=source_kind, topic=text, title=title.strip(), page_scroll=page_scroll,
                  video_mode=video_mode,
                  status="drafting" if start else "planned",
                  detail=(("writing the story with the LLM" if video_mode == "story"
                           else "writing script with the LLM") if start else
                          "needs setup - open the job to choose length and resolution"),
                  aspect_ratio=aspect_ratio, megapixels=mp, video_model=vm,
                  resolution=res, song_voice=sv,
                  audio_mode=audio_mode, voice_id=voice_id, art_style=art_style,
                  video_length=video_length, channel_id=cid, channel=snap, storyboard=sb,
                  # captions and polish were resolved above and then dropped on the floor here,
                  # so every topic/description/transcript/repo video was assembled with both
                  # off whatever the New Video dialog said. Only storyboard was carried.
                  captions=cap, polish=pol, thumb_text=thumb_text,
                  avatar_id=av_id, avatar_model=av_model, avatar_coverage=av_cov,
                  lyrics=lyrics, owner=owner or (channel or {}).get("owner", ""),
                  music_upload=(up or {}).get("name", ""),
                  music_seconds=float((up or {}).get("seconds", 0.0)),
                  plan_id=str(plan.get("plan_id", "")), planned_day=int(plan.get("day", 0) or 0),
                  planned_for=str(plan.get("date", "")), hook=str(plan.get("hook", "")),
                  thumb_idea=str(plan.get("thumb_idea", "")))
        job.save()
        if start:          # a planned job logs it at setup instead, once it has its engine
            _log_avatar_plan(job)
        _install_upload(job, up)
        if source_kind == "transcript":   # shown by the job page's Transcript card
            job.dir.mkdir(parents=True, exist_ok=True)
            (job.dir / "transcript.txt").write_text(text, encoding="utf-8")
        with self.lock:
            self.jobs[job.id] = job
        if start:
            self._start_draft(job)
        return job

    def submit_compilation(self, part_ids: list, channel: dict | None = None, owner: str = "",
                           title: str = "") -> Job:
        """Create a compilation of finished videos and start stitching it immediately.

        Not queued. The whole job is local ffmpeg over files that already exist, so making it wait
        behind a render would be an hour of nothing for work that takes a minute - the same
        reasoning as request_assemble and request_upscale, which both run in their own thread.

        The parts' own parameters decide the output: they all share one aspect ratio (validated),
        and their largest frame size becomes the compilation's. Nothing here is a per-video choice,
        so _resolve_choices is not involved."""
        with self.lock:
            parts = []
            for pid in part_ids or []:
                j = self.jobs.get(str(pid))
                if not j:
                    raise ValueError(f"unknown video {pid}")
                parts.append(j)
            parts = compilation.validate_parts(parts)
            cid = str(parts[0].channel_id or "")
            if channel and cid and str(channel.get("id") or "") != cid:
                raise ValueError("the videos do not belong to the chosen channel")
            snap = channels.snapshot(channel) if channel else dict(parts[0].channel or {})
            job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
                      url="", source=compilation.SOURCE, parts=[p.id for p in parts],
                      title=title.strip() or compilation.default_title(
                          [p.title for p in parts], 0.0, snap.get("name", "")),
                      status="assembling", detail=f"joining {len(parts)} videos",
                      aspect_ratio=parts[0].aspect_ratio,
                      # inherited so the job card, the bundle and any later upscale read right;
                      # none of them is a choice a compilation makes
                      resolution=parts[0].resolution, video_model=parts[0].video_model,
                      megapixels=parts[0].megapixels, audio_mode=parts[0].audio_mode,
                      channel_id=cid, channel=snap,
                      owner=owner or (channel or {}).get("owner", "") or parts[0].owner,
                      thumb_text=self._resolve_thumb_text(None, channel))
            job.save()
            self.jobs[job.id] = job
        self._start_compilation(job)
        return job

    def _start_compilation(self, job: Job):
        threading.Thread(target=run_job, args=(job,), daemon=True).start()

    def setup_planned(self, job_id: str, aspect_ratio: str = "", art_style: str = "",
                      video_length: str = "", audio_mode: str = "", voice_id: str = "",
                      storyboard_flag=None, megapixels=None, resolution: str = "",
                      song_voice: str = "", lyrics: str = "",
                      music_upload_id: str = "", video_mode: str = "", captions_flag=None,
                      polish_flag=None, video_model: str = "", avatar_id: str = "",
                      avatar_model=None, avatar_coverage: str = "",
                      thumb_text_flag=None) -> Job:
        """The Start button of a planned job: apply the parameters the user chose and hand the job
        to the normal drafting flow.

        Resolved against the channel snapshot frozen at plan time (the channel may have been edited
        or deleted since) with the job's own values as the fallback layer, so a field the form left
        blank keeps what the job was created with instead of snapping back to the global default.
        Nothing is mutated until every value validates."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "planned":
                raise ValueError("job is not waiting for setup")
            ch = {**job.channel, "id": job.channel_id,
                  "defaults": {"aspect_ratio": job.aspect_ratio, "art_style": job.art_style,
                               "video_length": job.video_length, "audio_mode": job.audio_mode,
                               "song_voice": job.song_voice, "storyboard": job.storyboard,
                               "captions": job.captions, "polish": job.polish,
                               "thumb_text": job.thumb_text,
                               "voice_id": job.voice_id, "video_mode": job.video_mode,
                               "avatar_id": job.avatar_id, "avatar_model": job.avatar_model,
                               "avatar_coverage": job.avatar_coverage}}
            # resolved against the snapshot frozen at plan time, exactly as the lyrics are below
            vmode = resolve_video_mode(video_mode or job.video_mode, ch, job.source,
                                        lyrics, music_upload_id)
            if vmode == "story":        # see the note in submit(): forced before the resolve
                audio_mode, storyboard_flag = "narration+music", False
            ar, style, vlen, mode, voice, sv, _cid, _snap, sb, cap, pol, vm, mp, res = self._resolve_choices(
                aspect_ratio, art_style, video_length, audio_mode, voice_id, ch, storyboard_flag,
                megapixels if megapixels not in (None, "", 0) else job.megapixels,
                resolution or job.resolution, song_voice, captions_flag, polish_flag,
                video_model or job.video_model)
            av_id, av_model, av_cov = self._resolve_avatar(avatar_id, avatar_model,
                                                           avatar_coverage, ch)
            thumb_text = self._resolve_thumb_text(thumb_text_flag, ch)
            # _cid/_snap are dropped on purpose: the channel id and snapshot stay as taken at plan time
            # lyrics are the one field a blank does NOT inherit from the job: an empty box means
            # the user cleared it (and switching to a voiceover clears it in the form too)
            words = _resolve_lyrics(lyrics, ch, mode)
            # a blank music_upload_id means no upload, not "inherit": a planned job never carries one
            up = _resolve_music_upload(music_upload_id, mode, words)
            if words or up:
                vlen = "Auto"          # the lyrics / the track set the length: keep no unused preset
            if up:      # the job folder exists since plan time; nothing was mutated before this
                music_upload.install(up["id"], job.dir / "music.mp3")
            job.set("drafting",
                    "writing the story with the LLM" if vmode == "story" else "writing script with the LLM",
                    aspect_ratio=ar, art_style=style,
                    video_length=vlen, audio_mode=mode, voice_id=voice, song_voice=sv,
                    storyboard=sb, captions=cap, polish=pol, thumb_text=thumb_text,
                    megapixels=mp, resolution=res,
                    video_model=vm, lyrics=words,
                    avatar_id=av_id, avatar_model=av_model, avatar_coverage=av_cov,
                    setup_done=True,
                    video_mode=vmode,
                    music_upload=(up or {}).get("name", ""),
                    music_seconds=float((up or {}).get("seconds", 0.0)))
            job.add_log(f"setup confirmed ({'your uploaded track' if up else 'your lyrics' if words else vlen}, "
                        f"{vm}, {ar}, {mp} MP, {res}, "
                        f"{'story' if vmode == 'story' else 'song' if mode == 'music' else 'voiceover'})"
                        " - writing the script")
            _log_avatar_plan(job)
        self._start_draft(job)
        return job

    def _start_draft(self, job: Job, feedback: str = ""):
        threading.Thread(target=self._draft_script, args=(job, feedback), daemon=True).start()

    def _draft_story(self, job: Job, feedback: str = ""):
        """The two LLM calls a story video is drafted from, in this order:

        1. the BEATS - the video content: what happens, and therefore what is drawn;
        2. the LINES - the voiceover, written FROM those beats.

        That order is the whole point of the format. A writer asked for narration first has to
        imagine a shot and then describe it, which is how you get "Ellie the elephant, wearing a
        yellow dress, is riding a blue train"; a writer handed a beat that already exists can simply
        say what happens. The lines are written with the cast sheet in spoken mode, so the writer is
        never even shown the looks it must not recite."""
        pfile = job.dir / "plan.json"
        previous = None
        if feedback and pfile.exists():
            try:
                previous = (json.loads(pfile.read_text(encoding="utf-8")).get("story") or {}).get("beats")
            except Exception:  # noqa: BLE001
                previous = None
        text = job.topic if job.source in PROMPT_SOURCES else _youtube_transcript(job, "drafting")
        clip_s = _clip_seconds(job)
        target = _length_params(job)[1] or float(config.TARGET_VIDEO_SECONDS or 0) or 60.0
        n = story.beat_count(target, clip_s, job.video_model)
        job.set("drafting", f"planning {n} story scenes with the LLM")
        plan = llm.write_story_beats(text, job.source, n, clip_s,
                                     art_style=config.ART_STYLES.get(job.art_style, ""),
                                     feedback=feedback, previous=previous,
                                     channel=job.channel, hook=job.hook,
                                     model=job.video_model)
        beats = plan.pop("beats")
        plan["video_mode"] = "story"
        plan["audio_mode"] = job.audio_mode
        plan["narration"] = ""
        plan["story"] = {"premise": plan.pop("premise", ""), "beats": beats, "lines": [],
                         "issues": [], "target_seconds": round(target, 2),
                         "clip_seconds_req": clip_s}
        if job.thumb_idea:
            plan["thumbnail_idea"] = job.thumb_idea
        # a kids story hands the viewer nothing: write the key away so no resume re-decides it
        plan["resource"] = None
        used = _set_cast_used(job, plan, _story_blob(plan))
        job.add_log("cast in this story: " + (", ".join(used) or "none"))

        if _draft_dropped(job):   # cancelled during the beats call: do not pay for the lines too
            return
        max_words = story.line_words(clip_s, job.video_model)
        job.set("drafting", f"writing what is said over the {len(beats)} scenes")
        lines, issues = llm.write_story_lines(beats, used, channel=job.channel, used=used,
                                              max_words=max_words)
        issues += _clean_story_lines(job, beats, lines, used, max_words)
        plan["story"]["lines"] = lines
        plan["story"]["issues"] = issues
        plan["narration"] = "\n".join(ln["text"] for ln in lines if ln.get("text", "").strip())
        used = _set_cast_used(job, plan, _story_blob(plan))
        if _draft_dropped(job):
            return
        job.dir.mkdir(parents=True, exist_ok=True)
        pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        job.set("awaiting_approval", "story ready - review and approve",
                title=plan.get("title", ""))
        for line in issues:
            job.add_log(line)
        spoken = sum(1 for ln in lines if ln.get("text", "").strip())
        job.add_log(f"story drafted: {len(beats)} scenes, {spoken} spoken lines "
                    f"- awaiting approval")

    def _draft_script(self, job: Job, feedback: str = ""):
        """LLM-only work: runs in its own thread, never in the GPU queue."""
        try:
            if _is_story(job):
                self._draft_story(job, feedback)
                return
            job.set("drafting", "writing script with the LLM")
            previous = None
            pfile = job.dir / "plan.json"
            if feedback and pfile.exists():
                try:
                    previous = json.loads(pfile.read_text(encoding="utf-8")).get("narration")
                except Exception:  # noqa: BLE001
                    previous = None
            target_words = _length_params(job)[0]
            source_text, content_y = job.topic, 0
            if job.lyrics:
                # the story seeds the title, SEO, thumbnail and story board cards, while the visuals
                # follow the sung words - without this they can end up about different things
                source_text = (f"{source_text}\n\nThe song's lyrics (the video IS this song):\n"
                               f'\"\"\"{job.lyrics}\"\"\"').strip()
            if job.source == REPO_SOURCE:
                source_text = _repo_source_text(job)
                if job.page_scroll:
                    content_y = _page_screenshot(job)
                job.set("drafting", "writing script with the LLM")
            plan = llm.write_script_from_prompt(
                source_text, job.source, art_style=config.ART_STYLES.get(job.art_style, ""),
                words=target_words, feedback=feedback, previous=previous,
                channel=job.channel, hook=job.hook)
            if _draft_dropped(job):   # the user cancelled while the writer was working
                return
            _fit_narration(job, plan, target_words)   # still pre-approval: the user reviews the result
            plan["audio_mode"] = job.audio_mode
            plan["page_content_y"] = content_y   # where the README starts in page.png
            if job.thumb_idea:      # the thumbnail the weekly plan asked for (used at the thumbnail step)
                plan["thumbnail_idea"] = job.thumb_idea
            if cast.active(job.channel):
                used = _set_cast_used(job, plan)
                job.add_log("cast in this story: " + (", ".join(used) or "none"))
                missing = [n for n in cast.names(cast.used_characters(job.channel, job.topic)) if n not in used]
                if missing:
                    job.add_log(f"note: the request named {', '.join(missing)} but the draft does not - "
                                "edit the script or request a new draft")
            else:
                plan.setdefault("cast_used", [])
            job.dir.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            job.set("awaiting_approval", "script ready - review and approve",
                    title=plan.get("title", ""))
            job.add_log("script drafted - awaiting approval")
        except Exception as e:  # noqa: BLE001
            if job.cancel_requested:   # cancel() already failed it: keep its reason, not this one
                job.add_log(f"script drafting stopped by the cancel: {e}")
                return
            job.add_log(f"script drafting failed: {e}")
            job.set("failed", f"script drafting failed: {e}", error=str(e))

    def approve_script(self, job_id: str, title: str | None = None,
                       narration: str | None = None, lines: list | None = None) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "awaiting_approval":
                raise ValueError("job is not awaiting script approval")
            pfile = job.dir / "plan.json"
            plan = json.loads(pfile.read_text(encoding="utf-8"))
            if title and title.strip():
                plan["title"] = title.strip()
            extra = ()
            if _is_story(job):
                # the spoken lines ARE the narration here, so the free-text box is never applied:
                # an edited paragraph would contradict the lines the clips are timed to
                if lines is not None:
                    st = plan.setdefault("story", {})
                    st["lines"] = _validate_story_lines(lines, st.get("lines") or [],
                                                        plan.get("cast_used") or [])
                    _drop_stale_lines(job, st["lines"])
                plan["narration"] = "\n".join(
                    ln["text"] for ln in (plan.get("story") or {}).get("lines") or []
                    if ln.get("text", "").strip())
                extra = (_story_blob(plan),)
            elif narration and narration.strip():
                plan["narration"] = narration.strip()
            if cast.active(job.channel):   # the user may have written characters in or out
                job.add_log("cast in this video: "
                            + (", ".join(_set_cast_used(job, plan, *extra)) or "none"))
            pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            job.set("queued", "script approved - waiting for video generation",
                    title=plan.get("title", ""), script_approved=True, cancel_requested=False)
            job.add_log("script approved - queued for video generation")
            self.queue.append(job_id)
        return job

    def redraft_script(self, job_id: str, feedback: str = "") -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "awaiting_approval":
                raise ValueError("job is not awaiting script approval")
            job.set("drafting", "rewriting script with the LLM", cancel_requested=False)
        self._start_draft(job, feedback=feedback)
        return job

    def start_youtube_upload(self, job_id: str) -> Job:
        from services import youtube
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status != "done":
                raise ValueError("only finished videos can be uploaded")
            if job.yt_status == "uploading":
                raise ValueError("an upload is already in progress for this job")
            if not (job.dir / "final.mp4").exists():
                raise ValueError("final video not found")
            if not youtube.has_secret():
                raise ValueError("client_secret.json not found - place your OAuth client file in the project folder")
            token_file = channels.token_path(job.channel_id) if job.channel_id else config.YT_TOKEN_FILE
            if not youtube.is_connected(token_file):
                raise ValueError("this channel is not connected to YouTube - connect it on the Settings page first"
                                 if job.channel_id else
                                 "YouTube is not connected - connect it on the Settings page first")
            job.set(yt_status="uploading", yt_progress=0, yt_error="")
        threading.Thread(target=self._yt_upload, args=(job,), daemon=True).start()
        return job

    def _yt_upload(self, job: Job):
        from services import youtube
        try:
            plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
            title = plan.get("seo_title") or plan.get("title") or job.title
            desc = upload_description(plan)
            cat = (config.YT_CATEGORY_BY_TYPE.get((job.channel or {}).get("video_type", ""))
                   or config.YT_CATEGORY_ID)
            tags = youtube.clean_tags(plan.get("seo_keywords") or [], title,
                                      max_tags=config.YT_MAX_TAGS) if config.YT_MAX_TAGS else []
            token_file = channels.token_path(job.channel_id) if job.channel_id else config.YT_TOKEN_FILE
            made_for_kids = job.channel.get("made_for_kids",
                                            job.channel.get("audience") == "kids") \
                if job.channel else config.YT_MADE_FOR_KIDS
            target = f' to "{job.channel["name"]}"' if job.channel.get("name") else ""
            job.add_log(f"YouTube upload started{target}")
            vid, thumb_err, tag_err = youtube.upload(
                job.dir / "final.mp4", job.dir / "thumbnail.jpg", title, desc,
                config.YT_PRIVACY, made_for_kids,
                on_progress=lambda p: job.set(yt_progress=p), token_file=token_file,
                tags=tags, category_id=cat)
            job.set(yt_status="uploaded", yt_progress=100, yt_video_id=vid)
            job.add_log(f"uploaded to YouTube ({config.YT_PRIVACY}): https://youtu.be/{vid}")
            job.add_log(f"filed under {config.YT_CATEGORIES.get(cat, cat)}, {len(tags)} tags, "
                        f"{len(plan.get('chapters') or [])} chapters")
            if tag_err:
                job.add_log(f"note: tags rejected by YouTube ({tag_err[:100]}); uploaded without them")
            if thumb_err:
                job.add_log(f"note: custom thumbnail not set ({thumb_err[:100]})")
            else:
                job.add_log("custom thumbnail set on YouTube")
        except Exception as e:  # noqa: BLE001
            job.set(yt_status="failed", yt_error=str(e)[:200])
            job.add_log(f"YouTube upload failed: {e}")

    def regen_resource(self, job_id: str) -> Job:
        """Re-decide and rewrite the viewer resource of a finished job.

        The pipeline only runs the check once per job, so a video made before this existed - or one
        whose script was edited afterwards - never gets the hand-out it promised. This is the way
        back. LLM + file writes only, so it runs in a side thread rather than the GPU queue."""
        with self.lock:
            job = self._maintenance_target(job_id)
            job.set("planning", "rewriting the viewer resource")
        threading.Thread(target=self._regen_resource, args=(job,), daemon=True).start()
        return job

    def _regen_resource(self, job: Job):
        d = job.dir
        try:
            plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
            # start from the writer's own description so the block is composed once, not twice
            base = plan.pop("seo_description_base", None)
            if base is not None:
                plan["seo_description"] = base
            plan.pop("resource", None)
            tfile = d / "transcript.txt"
            source = tfile.read_text(encoding="utf-8", errors="replace") if tfile.exists() else ""
            res, origin, url = _viewer_document(job, plan, source)
            plan["resource"], logs = resource.attach(
                d, plan, res, channel=(job.channel or {}).get("name", ""), logo=_logo_for(job),
                origin=origin, source_url=url)
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            for line in logs:
                job.add_log(line)
            job.set("done", "viewer resource regenerated")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"viewer resource regeneration failed: {e}")
            job.set("done", "viewer resource unchanged")

    def regen_thumbnail(self, job_id: str) -> Job:
        """Re-render the thumbnail with a fresh concept and seed. Runs in a side thread -
        it is a single ~30 s render, so it does not go through the job queue."""
        with self.lock:
            job = self._maintenance_target(job_id)
            job.set("thumbnail", "regenerating the thumbnail")
        threading.Thread(target=self._regen_thumbnail, args=(job,), daemon=True).start()
        return job

    def _regen_thumbnail(self, job: Job):
        d = job.dir
        try:
            plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
            job.set("thumbnail", "designing a new thumbnail with vLLM")
            plan.update(llm.write_thumbnail(plan.get("title", ""), plan.get("narration", ""),
                                            plan.get("style", ""), channel=job.channel,
                                            used=plan.get("cast_used"),
                                            idea=plan.get("thumbnail_idea", "")))
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            schedule.wait_until_open(on_wait=lambda s: job.set("thumbnail", schedule.describe_wait()))
            gpu.wait_until_free(job.id, on_wait=lambda h: job.set(
                "thumbnail", "waiting for the GPU - another video is rendering"))
            job.set("thumbnail", "rendering the new thumbnail")
            tw, th = _render_thumbnail(job, plan, d / "thumb_base.jpg", on_progress=_progress(job, "thumbnail"))
            _thumb_variants(job, plan, select=thumbtext.AUTO)   # a fresh concept, a fresh best line
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            job.set("done", "thumbnail regenerated")
            job.add_log(f"thumbnail regenerated ({tw}x{th})")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"thumbnail regeneration failed: {e}")
            job.set("done", f"thumbnail regeneration failed: {e}")
        finally:
            gpu.release(job.id)

    def set_thumbnail(self, job_id: str, variant=None, texts=None) -> Job:
        """Pick which thumbnail the video ships with, and/or replace the overlay lines.

        Pillow only - three draws on a 1280x720 JPG - so it runs inline rather than in a thread or
        the GPU queue."""
        with self.lock:
            job = self._maintenance_target(job_id)
        d = job.dir
        plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        if texts is not None:
            plan["thumbnail_texts"] = [thumbtext.normalize(t) for t in texts if str(t).strip()]
        _thumb_variants(job, plan, select=variant)
        (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        job.add_log(f"thumbnail variant {job.thumb_variant or 'none (plain render)'}")
        return job

    def _maintenance_target(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.archived:
            raise ValueError("unarchive the job first")
        if job.status != "done":
            raise ValueError("only finished jobs can be modified")
        if job_id in self.running or job_id in self.queue:
            raise ValueError("job is already queued or running")
        if job.upscaling:
            raise ValueError("an upscale of the final video is in progress - wait for it to finish")
        return job

    def update_scenes(self, job_id: str, updates: dict):
        """Apply edited scene prompts to plan.json and mark the affected clips stale."""
        with self.lock:
            job = self._maintenance_target(job_id)
            pfile = job.dir / "plan.json"
            plan = json.loads(pfile.read_text(encoding="utf-8"))
            scenes = plan.get("scenes") or []
            clean = {}
            for k, text in (updates or {}).items():
                try:
                    i = int(k)
                except (TypeError, ValueError):
                    raise ValueError(f"invalid clip number {k!r}")
                if not (1 <= i <= len(scenes)):
                    raise ValueError(f"clip {i} does not exist")
                if not str(text).strip():
                    raise ValueError(f"scene {i} must not be empty")
                clean[i] = str(text).strip()
            if not clean:
                raise ValueError("no scene updates given")
            changed = [i for i, t in clean.items() if t != scenes[i - 1]]
            for i, t in clean.items():
                scenes[i - 1] = t
            plan["scenes"] = scenes
            pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            stale = sorted(set(job.stale_clips) | set(changed))
            plural = "s" if len(stale) != 1 else ""
            job.set(stale_clips=stale,
                    detail=(f"scene{plural} updated - regenerate clip{plural} "
                            f"{', '.join(map(str, stale))}" if stale else "finished"))
            if changed:
                job.add_log(f"scene prompt{plural} edited for clip{plural} "
                            f"{', '.join(map(str, changed))}")
        return job

    def request_regen(self, job_id: str, clips: list):
        with self.lock:
            job = self._maintenance_target(job_id)
            try:
                clips = sorted({int(c) for c in clips})
            except (TypeError, ValueError):
                raise ValueError("invalid clip numbers")
            if not clips:
                raise ValueError("no clips given")
            for c in clips:
                if not (1 <= c <= len(job.clips)) or not (job.dir / f"clip_{c:02d}.mp4").exists():
                    raise ValueError(f"clip {c} does not exist")
            job.regen_queue = clips
            plural = "s" if len(clips) > 1 else ""
            job.set("queued", f"regeneration of clip{plural} {', '.join(map(str, clips))} requested")
            self.queue.insert(0, job_id)          # priority: front of the queue
            cur_id = self.current
            if cur_id and cur_id != job_id and cur_id in self.jobs:
                cur = self.jobs[cur_id]
                cur.pause_requested = True
                cur.add_log("pausing after the current step - a clip regeneration takes priority")
                if cur_id not in self.queue:
                    self.queue.insert(1, cur_id)  # resume right after the regeneration
        return job

    def request_assemble(self, job_id: str):
        """Reassembly is local ffmpeg work (no GPU): run it immediately in its own
        thread instead of waiting behind rendering jobs in the queue - unless something it
        needs has still to be generated, which goes through the GPU queue (front of the
        queue, like a clip regeneration).

        Two things can be owed: the story board's LLM cards / ComfyUI images, and a
        presenter whose segments never rendered - which is the state a video is left in
        when the matting model is missing, and the reason this is offered on every
        finished job rather than only on one with regenerated clips."""
        with self.lock:
            job = self._maintenance_target(job_id)
            if not job.clips:
                raise ValueError("job has no clips to assemble")
            job.assemble_requested = True
            plan = _load_plan(job)
            pending = ("story board assets"
                       if job.storyboard and not storyboard.assets_ready(job.dir, plan)
                       else "presenter" if _avatar_pending(job, plan) else "")
            if pending:
                job.set("queued", f"{pending} + reassembly requested")
                self.queue.insert(0, job_id)
                return job
            job.set("assembling", "re-merging clips and mixing audio")
        threading.Thread(target=self._assemble_now, args=(job,), daemon=True).start()
        return job

    def set_storyboard(self, job_id: str, enabled: bool, regenerate: bool = False):
        """Switch the story board overlay of a finished job on/off (regenerate=True also
        discards the planned cards and rendered images so they are created afresh). The
        change is applied by the next reassembly."""
        with self.lock:
            job = self._maintenance_target(job_id)
            if regenerate:
                plan = _load_plan(job)
                plan.pop("storyboard", None)
                _save_plan(job, plan)
                storyboard.clear_assets(job.dir)
            changed = job.storyboard != bool(enabled) or regenerate
            job.storyboard = bool(enabled)
            if changed:
                what = ("story board will be regenerated" if regenerate and enabled
                        else "story board enabled" if enabled else "story board disabled")
                job.set(needs_assemble=True, detail=f"{what} - reassemble the final video to apply")
                job.add_log(what)
        return job

    def _assemble_now(self, job: Job):
        try:
            _do_assemble(job)
            job.set("done", "finished")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"REASSEMBLY FAILED: {e}")
            job.set("done", f"reassembly failed: {e}", assemble_requested=False)

    def request_upscale(self, job_id: str, resolution: str) -> Job:
        """Upscale the finished final.mp4 to `resolution` (an OUTPUT_RESOLUTIONS label other than
        Native) into final_<short>p.mp4, kept next to the original. Local ffmpeg work: runs in a
        side thread, not the GPU queue. Re-running the same target replaces that file; other
        targets are kept, so one video can carry 1080p, 2K and 4K versions at once."""
        short = config.OUTPUT_RESOLUTIONS.get(resolution or "", 0)
        if not short:
            raise ValueError(f"unsupported upscale target: {resolution}")
        with self.lock:
            job = self._maintenance_target(job_id)
            final = job.dir / (job.final or "final.mp4")
            if not job.final or not final.exists():
                raise ValueError("the final video has not been generated yet")
            w, h = video.frame_size(final)
            size = video.target_size(w, h, short)
            if size is None:
                raise ValueError(f"the final video is {w}x{h}: already at or above {resolution}")
            job.set(upscaling=resolution, detail=f"upscaling the final video to {size[0]}x{size[1]} ({resolution})")
        threading.Thread(target=self._upscale_now, args=(job, resolution, short, size), daemon=True).start()
        return job

    def _upscale_now(self, job: Job, resolution: str, short: int, size: tuple):
        d = job.dir
        name = upscale_filename(short)
        tmp = d / f".{name}.part.mp4"
        try:
            video.upscale(d / (job.final or "final.mp4"), tmp, *size)
            tmp.replace(d / name)
            entry = {"label": resolution, "file": name, "width": size[0], "height": size[1],
                     "created": time.time()}
            job.upscales = [u for u in job.upscales if u.get("label") != resolution] + [entry]
            job.upscales.sort(key=lambda u: u.get("height", 0) * u.get("width", 0))
            job.set(upscaling="", detail=f"upscaled version ready: {resolution}")
            job.add_log(f"final video upscaled to {size[0]}x{size[1]} ({resolution}) -> {name}")
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            job.add_log(f"upscale to {resolution} failed: {e}")
            job.set(upscaling="", detail=f"upscale to {resolution} failed: {e}")

    def set_archived(self, job_id: str, archived: bool):
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status not in ("done", "failed"):
            raise ValueError("only finished jobs can be archived")
        job.set(archived=archived)
        return job

    def delete(self, job_id: str) -> tuple:
        """Remove the job and its whole output folder. Returns (bytes_freed, warning):
        warning is set when some files stayed locked (they are swept at the next start)."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job_id in self.running:
                raise ValueError("job is currently running and cannot be deleted")
            if job.status in ("assembling", "thumbnail", "drafting"):
                raise ValueError(f"job is busy ({job.status}) - wait until it finishes, then delete it")
            if job.yt_status == "uploading":
                raise ValueError("job is uploading to YouTube - wait until the upload finishes")
            if job.upscaling:
                raise ValueError("job is upscaling its final video - wait until it finishes, then delete it")
            if job_id in self.queue:
                self.queue.remove(job_id)
            self.jobs.pop(job_id)
        return remove_job_dir(job.dir)

    def list(self, owner: str | None = None):
        """All jobs newest first, or only those owned by the given user id."""
        jobs = self.jobs.values()
        if owner is not None:
            jobs = [j for j in jobs if j.owner == owner]
        return sorted(jobs, key=lambda j: j.created, reverse=True)
