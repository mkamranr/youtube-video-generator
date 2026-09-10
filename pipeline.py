"""Job model + the end-to-end generation pipeline."""
import json
import math
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
from services import cast, channels, comfy, llm, lyrics, schedule, storyboard, transcript, tts, video


PROMPT_SOURCES = ("topic", "description", "transcript")   # user text drafted by the LLM, then approved


# statuses that are real pipeline work; `queued`, `awaiting_approval`, `done`, `failed` only close the open one
STAGE_STATUSES = frozenset({"drafting", "transcript", "planning", "tts", "music", "scenes",
                            "thumbnail", "storyboard", "clips", "assembling"})


@dataclass
class Job:
    id: str
    url: str
    aspect_ratio: str = config.DEFAULT_ASPECT_RATIO
    megapixels: float | None = None  # clip resolution; None (legacy jobs) = the current Settings default
    resolution: str = ""            # final-video upscale target (config.OUTPUT_RESOLUTIONS); "" (legacy) = native
    audio_mode: str = ""
    voice_id: str = ""              # reference voice for Fish Speech cloning ("" = default voice)
    art_style: str = ""
    video_length: str = "Auto"
    source: str = "youtube"         # youtube | topic | description | transcript
    topic: str = ""                 # the topic, description or pasted transcript text
    channel_id: str = ""            # channel this video belongs to ("" = legacy pre-channel job)
    owner: str = ""                 # user id of the job's owner (the channel's owner); "" only before the ownership migration
    channel: dict = field(default_factory=dict)  # snapshot: {name, purpose, rules, audience, video_type, cast, ...}
    storyboard: bool = False        # story board overlay: styled fact cards + related images on the video
    script_approved: bool = False
    status: str = "queued"          # queued|transcript|planning|tts|scenes|music|storyboard|clips|assembling|done|failed
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
class JobCancelled(Exception):
    pass


class JobPaused(BaseException):
    """Raised at a checkpoint when a priority regeneration wants the worker."""
    pass


class JobWaitingWindow(BaseException):
    """Raised right before a ComfyUI submission when the GPU schedule window is closed.
    BaseException so the non-fatal `except Exception` wrappers cannot swallow it."""
    pass


def _check_cancel(job: Job):
    if job.cancel_requested:
        raise JobCancelled()
    if job.pause_requested:
        raise JobPaused()


def _gpu_gate(job: Job):
    """Checkpoint placed immediately before every ComfyUI submission of the worker path."""
    _check_cancel(job)
    if not schedule.is_open():
        raise JobWaitingWindow()


def _cleanup_warn(job: Job):
    def cb():
        if "cleanup not supported" not in " ".join(job.log[-50:]):
            job.add_log("note: ComfyUI file cleanup not supported by this server build")
    return cb


def _length_params(job: Job):
    """Per-job soft length targets: (words, target_seconds, max_seconds, seconds_range).
    All None for Auto - the global settings apply."""
    rng = config.video_length_range(job.video_length or "Auto")
    if not rng:
        return None, None, None, None
    lo, hi = rng
    mid = (lo + hi) / 2.0
    return int(mid * 2.5), mid, min(360.0, hi * 1.5 + 20), (lo, hi)


def _logo_for(job: Job):
    """The job's channel logo for the thumbnail overlay; legacy global logo as fallback."""
    if job.channel_id:
        p = channels.logo_path(job.channel_id)
        if p.exists():
            return p
    return config.LOGO_FILE if config.LOGO_FILE.exists() else None


def _progress(job: Job, status: str):
    """Progress callback that also honors a pending cancel request."""
    def cb(msg):
        _check_cancel(job)
        job.set(status, msg)
    return cb


def run_job(job: Job):
    job.gpu_wait = False
    try:
        if job.regen_queue or job.assemble_requested:
            _run_maintenance(job)
            return
        _run(job)
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

    def progress(msg):
        job.set("clips", f"{label}: {msg}")

    for attempt in range(config.CLIP_RETRIES + 1):
        _gpu_gate(job)
        try:
            wf = client.build(template, prompt, f"videogen/{job.id}/clip_{i:02d}",
                              first_frame=first_frame, aspect_ratio=job.aspect_ratio,
                              megapixels=job.megapixels)
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


def _set_cast_used(job: Job, plan: dict) -> list:
    """plan["cast_used"]: the channel characters that are in this video, in first-mention order.

    Every roster name found in the final title/narration counts. The names the script writer
    reported choosing ("cast_chosen") are added when the text itself names nobody - a pasted
    transcript never mentions who plays in the visuals, and a script may say "the little fox"
    without naming Finn - otherwise the text (which the user may have edited) is the authority."""
    used = cast.names(cast.used_characters(job.channel, plan.get("title", ""), plan.get("narration", "")))
    if not used or job.source == "transcript":
        for n in plan.get("cast_chosen") or []:
            if n not in used:
                used.append(n)
    plan["cast_used"] = used
    return used


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


def _assemble(job: Job, plan: dict, narration_file, music_file) -> Path:
    """Concat the clips, overlay the story board cards (if enabled) and mix the audio
    for the job's audio mode into final.mp4."""
    d = job.dir
    clips = [d / c for c in job.clips]
    joined = video.concat_clips(clips, d / "clips_joined.mp4")
    bounds = video.clip_boundaries(clips)
    joined = _upscale_joined(job, joined)
    source = joined
    if job.storyboard and isinstance(plan.get("storyboard"), list):
        job.set("assembling", "drawing the story board cards")
        try:
            cards = storyboard.render_all(d, plan, clips, video.frame_size(joined),
                                          accent=config.STORYBOARD_ACCENT)
            job.set("assembling", f"overlaying {len(cards)} story board cards")
            source = video.overlay_cards(joined, cards, d / "clips_cards.mp4")
            job.add_log(f"story board: {len(cards)} cards overlaid")
        except Exception as e:  # noqa: BLE001 - never lose the video over the overlay
            job.add_log(f"story board overlay failed ({e}); assembling without it")
            source = joined
    job.set("assembling", "mixing audio")
    mode = job.audio_mode or "narration+music"
    if mode == "music":
        return video.mix_audio(source, d / "final.mp4", music=music_file,
                               music_volume=1.0, clip_volume=0)
    if mode == "narration+music":
        # the clips' own sound is dropped by default: voice + music + per-clip effects clash
        clip_vol = config.CLIP_AUDIO_VOLUME if config.CLIP_AUDIO_WITH_MUSIC else 0
        return video.mix_audio(source, d / "final.mp4", narration=narration_file,
                               music=music_file, clip_volume=clip_vol, clip_boundaries=bounds)
    return video.mix_audio(source, d / "final.mp4", narration=narration_file,
                           clip_boundaries=bounds)


def _do_assemble(job: Job):
    """Re-concat the job's clips, redo the story board overlay and remix its audio into a
    new final.mp4 (story board assets are generated first if they are missing)."""
    d = job.dir
    job.set("assembling", "re-merging clips and mixing audio")
    plan = _load_plan(job)
    if job.storyboard and not storyboard.assets_ready(d, plan):
        _storyboard_assets(job, plan, len(job.clips))
        job.set("assembling", "re-merging clips and mixing audio")
    music_file = d / "music.mp3" if (d / "music.mp3").exists() else None
    narration_file = d / "narration.wav" if (d / "narration.wav").exists() else None
    if (job.audio_mode or "") == "music":
        narration_file = None
    final = _assemble(job, plan, narration_file, music_file)
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
            t2v = client.load_workflow(config.T2V_WORKFLOW)
            i2v = client.load_workflow(config.I2V_WORKFLOW)
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
    elif tfile.exists() and tfile.stat().st_size > 50:
        text = tfile.read_text(encoding="utf-8")
        job.add_log("resume: reusing transcript")
    else:
        job.set("transcript", "fetching YouTube transcript")
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
        tfile.write_text(text, encoding="utf-8")
        job.add_log(f"transcript: {words} words covering {coverage:.0f}s of video")

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
        plan = llm.write_narration(text, words=_length_params(job)[0],
                                   art_style=config.ART_STYLES.get(job.art_style, ""),
                                   channel=job.channel)
    plan["audio_mode"] = mode
    if "cast_used" not in plan and cast.all_characters(job.channel):   # URL jobs and pre-selection plans
        job.add_log("cast in this video: " + (", ".join(_set_cast_used(job, plan)) or "none"))
    if not plan.get("seo_title"):
        job.set("planning", "writing SEO title and description")
        try:
            plan.update(llm.write_seo(plan.get("title", ""), plan["narration"], text,
                                      channel=job.channel, used=plan.get("cast_used")))
            job.add_log(f"seo title: {plan.get('seo_title', '')}")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"seo metadata failed ({e}); continuing without it")
    (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job.set(title=plan.get("title", ""))
    job.add_log(f"plan: '{plan.get('title')}', {len(plan['narration'].split())} story words")

    narration_file = None
    music_file = None

    if mode == "music":
        # 3. music first: the actual track length drives the clip count, and the
        #    lyrics drive the per-clip scenes so the song matches each clip.
        _check_cancel(job)
        _lw, l_target, l_max, l_range = _length_params(job)
        if not l_range:
            l_target = config.TARGET_VIDEO_SECONDS
            l_max = (min(360.0, config.TARGET_VIDEO_SECONDS * 2 + 30)
                     if config.TARGET_VIDEO_SECONDS else 360.0)
        music_plan = plan.get("music") or {}
        if not music_plan.get("caption"):
            job.set("music", "writing music caption and lyrics with vLLM")
            music_plan = llm.write_music(plan.get("title", ""), plan["narration"], text,
                                         standalone=True, seconds=l_target,
                                         max_seconds=l_max, seconds_range=l_range,
                                         channel=job.channel, used=plan.get("cast_used"))
            plan["music"] = music_plan
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if (d / "music.mp3").exists() and (d / "music.mp3").stat().st_size > 1000:
            music_file = d / "music.mp3"
            job.add_log("resume: reusing music track")
        else:
            _gpu_gate(job)
            job.set("music", "generating music with ComfyUI")
            mclient = comfy.ComfyClient()
            mwf = mclient.build_music(mclient.load_workflow(config.MUSIC_WORKFLOW),
                                      music_plan["caption"], music_plan.get("lyrics", ""),
                                      l_max,
                                      f"videogen/{job.id}/music")
            music_file = mclient.run(mwf, d / "music.mp3",
                                     on_progress=lambda m: job.set("music", f"music: {m}"),
                                     should_stop=lambda: job.cancel_requested,
                                     cleanup=config.COMFY_CLEANUP, on_cleanup_failed=_cleanup_warn(job))
        duration = tts.audio_duration(music_file)
        job.add_log(f"music track: {duration:.1f}s")
        n = max(1, math.ceil(duration / config.CLIP_SECONDS))
        job.set("music", "timing the sung lyrics (whisper)")
        segments = lyrics.transcribe_windows(music_file, n, config.CLIP_SECONDS,
                                             on_progress=_progress(job, "music"))
        sung = sum(1 for s in segments if s.strip())
        job.add_log(f"lyrics timed: words heard in {sung}/{n} clip windows")
        sync_text = llm.clean_lyrics(music_plan.get("lyrics", "")) or plan["narration"]
        kind = "song being played"
    else:
        # 3. narration voiceover
        _check_cancel(job)
        narration_file = d / "narration.wav"
        if narration_file.exists() and narration_file.stat().st_size > 1000:
            duration = tts.audio_duration(narration_file)
            job.add_log("resume: reusing narration audio")
        else:
            from services import voices
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
            duration = tts.synthesize(plan["narration"], narration_file,
                                      on_progress=_progress(job, "tts"),
                                      ref_audio=ref_audio, ref_text=ref_text, keep_voice=keep_voice)
            job.add_log(f"narration: {duration:.1f}s")
        n = max(1, math.ceil(duration / config.CLIP_SECONDS))
        sync_text = plan["narration"]
        segments = None
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
    if len(plan.get("scenes") or []) == n:
        job.add_log("resume: reusing scene prompts")
    else:
        job.set("scenes", f"writing {n} scene prompts with vLLM")
        def scene_progress(m):
            _check_cancel(job)
            job.set("scenes", f"writing {m}")
        def review_issue(idx, issues):
            job.add_log(f"review fixed scene {idx}: " + "; ".join(issues)[:160])
        plan["scenes"] = llm.write_scenes(sync_text, plan["style"], n, kind=kind, segments=segments,
                                          on_progress=scene_progress, on_review_issue=review_issue,
                                          channel=job.channel, used=plan.get("cast_used"),
                                          opening=plan.get("opening") if (d / "cast_opening.jpg").exists() else None)
    plan["segments"] = segments if segments is not None else llm.split_narration(sync_text, n)
    scenes = plan["scenes"]
    (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job.add_log(f"{n} scenes written, one per audio segment")

    # 5. thumbnail: LLM prompt -> single-frame render -> 1280x720 JPG (non-fatal)
    thumb = d / "thumbnail.jpg"
    if thumb.exists() and thumb.stat().st_size > 1000:
        job.add_log("resume: reusing thumbnail")
    else:
        try:
            _check_cancel(job)
            job.set("thumbnail", "designing the thumbnail with vLLM")
            if not plan.get("thumbnail_prompt"):
                plan.update(llm.write_thumbnail(plan.get("title", ""), plan["narration"], plan["style"],
                                                channel=job.channel, used=plan.get("cast_used")))
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
                                      n * config.CLIP_SECONDS * 1.05 + 2, f"videogen/{job.id}/music")
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

    # 7. clips
    mp = job.megapixels if job.megapixels is not None else config.DEFAULT_MEGAPIXELS
    if any(not _clip_on_disk(d / f"clip_{i:02d}.mp4") for i in range(1, n + 1)):
        _gpu_gate(job)   # park in the previous status: the `clips` timing starts with the first real render
    job.set("clips", f"generating {n} clips ({job.aspect_ratio}, {mp:g} MP)", clips_total=n, clips_done=0)
    client = comfy.ComfyClient()
    t2v = client.load_workflow(config.T2V_WORKFLOW)
    i2v = client.load_workflow(config.I2V_WORKFLOW)
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
    """Serial worker: one job at a time so the GPU is not contended."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.current: str | None = None
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self._load_existing()
        self.worker.start()
        threading.Thread(target=self._sweep_comfy, daemon=True).start()

    def _sweep_comfy(self):
        """Scrub orphaned videogen assets on the ComfyUI server (retries/crashes leave some)."""
        if not config.COMFY_CLEANUP:
            return
        try:
            with self.lock:
                skip = [j for j in ([self.current] + list(self.queue)) if j]
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
                if j.status == "awaiting_approval":
                    self.jobs[j.id] = j
                    continue
                if j.status == "drafting":
                    self.jobs[j.id] = j
                    self._start_draft(j)
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
                         storyboard_flag=None, megapixels=None, resolution=None):
        """Per-video choices: request value -> channel default -> global default; validated.
        Megapixels and the output resolution have no channel level: request value -> Settings default.
        Real/tech-video channels always use the built-in live-action look (art_style stays "");
        the story board overlay defaults to the channel setting (on for tech channels)."""
        ch = channel or {}
        defaults = ch.get("defaults") or {}
        if storyboard_flag is None:
            storyboard_flag = defaults.get("storyboard")
            if storyboard_flag is None:
                storyboard_flag = ch.get("video_type") == "tech"
        storyboard_flag = bool(storyboard_flag)
        audio_mode = audio_mode or defaults.get("audio_mode") or config.AUDIO_MODE
        if audio_mode not in config.AUDIO_MODES:
            raise ValueError(f"unsupported audio mode: {audio_mode}")
        voice_id = "" if audio_mode == "music" else (voice_id or defaults.get("voice_id") or "")
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
        return (aspect_ratio, art_style, video_length, audio_mode, voice_id, ch.get("id", ""), snap,
                storyboard_flag, mp, resolution)

    def submit(self, url: str, aspect_ratio: str = "",
               art_style: str = "", video_length: str = "",
               audio_mode: str = "", voice_id: str = "",
               channel: dict | None = None, storyboard_flag=None, megapixels=None,
               owner: str = "", resolution: str = "") -> Job:
        aspect_ratio, art_style, video_length, audio_mode, voice_id, cid, snap, sb, mp, res = \
            self._resolve_choices(aspect_ratio, art_style, video_length, audio_mode, voice_id, channel,
                                  storyboard_flag, megapixels, resolution)
        job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6], url=url.strip(),
                  aspect_ratio=aspect_ratio, megapixels=mp, resolution=res, audio_mode=audio_mode, voice_id=voice_id,
                  art_style=art_style, video_length=video_length, channel_id=cid, channel=snap, storyboard=sb,
                  owner=owner or (channel or {}).get("owner", ""))
        job.save()
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
        return job

    def _loop(self):
        while True:
            jid = None
            with self.lock:
                # first runnable job in order; jobs parked for the GPU window stay put while it is closed
                closed = not schedule.is_open()
                for idx, cand in enumerate(self.queue):
                    j = self.jobs.get(cand)
                    if j is not None and j.gpu_wait and closed:
                        continue
                    jid = self.queue.pop(idx)
                    break
            if jid is None:
                time.sleep(1)
                continue
            job = self.jobs.get(jid)
            if job is None or (job.status in ("done", "failed")
                               and not job.regen_queue and not job.assemble_requested):
                continue
            job.pause_requested = False
            self.current = jid
            try:
                run_job(job)
            finally:
                self.current = None
                if job.gpu_wait:        # parked: back to the queue, other jobs get their LLM/TTS work done
                    with self.lock:
                        if jid not in self.queue:
                            self.queue.append(jid)
                threading.Thread(target=self._sweep_comfy, daemon=True).start()

    def cancel(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in ("done", "failed"):
                raise ValueError("job already finished")
            if job_id in self.queue:
                self.queue.remove(job_id)
                if job.regen_queue or job.assemble_requested:
                    job.regen_queue = []
                    job.set("done", "request cancelled", assemble_requested=False)
                else:
                    job.set("failed", "cancelled by user", error="cancelled by user")
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
            restart_draft = job.source in PROMPT_SOURCES and not job.script_approved
            if restart_draft:
                job.set("drafting", "resumed - writing script with the LLM", error="", archived=False)
                job.add_log("resumed by user - restarting the script draft")
            else:
                job.set("queued", "resumed - waiting for the queue", error="", archived=False)
                job.add_log("resumed by user - finished steps will be reused")
                self.queue.append(job_id)
        if restart_draft:
            self._start_draft(job)
        return job

    def submit_prompt(self, text: str, source_kind: str,
                      aspect_ratio: str = "", art_style: str = "",
                      video_length: str = "",
                      audio_mode: str = "", voice_id: str = "",
                      channel: dict | None = None, storyboard_flag=None, megapixels=None,
                      owner: str = "", resolution: str = "") -> Job:
        text = (text or "").strip()
        if len(text) < 3:
            raise ValueError("topic/description/transcript is too short")
        if source_kind not in PROMPT_SOURCES:
            raise ValueError("invalid source kind")
        aspect_ratio, art_style, video_length, audio_mode, voice_id, cid, snap, sb, mp, res = \
            self._resolve_choices(aspect_ratio, art_style, video_length, audio_mode, voice_id, channel,
                                  storyboard_flag, megapixels, resolution)
        job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6], url="",
                  source=source_kind, topic=text, status="drafting",
                  detail="writing script with the LLM", aspect_ratio=aspect_ratio, megapixels=mp,
                  resolution=res,
                  audio_mode=audio_mode, voice_id=voice_id, art_style=art_style,
                  video_length=video_length, channel_id=cid, channel=snap, storyboard=sb,
                  owner=owner or (channel or {}).get("owner", ""))
        job.save()
        if source_kind == "transcript":   # shown by the job page's Transcript card
            job.dir.mkdir(parents=True, exist_ok=True)
            (job.dir / "transcript.txt").write_text(text, encoding="utf-8")
        with self.lock:
            self.jobs[job.id] = job
        self._start_draft(job)
        return job

    def _start_draft(self, job: Job, feedback: str = ""):
        threading.Thread(target=self._draft_script, args=(job, feedback), daemon=True).start()

    def _draft_script(self, job: Job, feedback: str = ""):
        """LLM-only work: runs in its own thread, never in the GPU queue."""
        try:
            job.set("drafting", "writing script with the LLM")
            previous = None
            pfile = job.dir / "plan.json"
            if feedback and pfile.exists():
                try:
                    previous = json.loads(pfile.read_text(encoding="utf-8")).get("narration")
                except Exception:  # noqa: BLE001
                    previous = None
            plan = llm.write_script_from_prompt(
                job.topic, job.source, art_style=config.ART_STYLES.get(job.art_style, ""),
                words=_length_params(job)[0], feedback=feedback, previous=previous,
                channel=job.channel)
            plan["audio_mode"] = job.audio_mode
            if cast.all_characters(job.channel):
                used = _set_cast_used(job, plan)
                job.add_log("cast in this story: " + (", ".join(used) or "none"))
                missing = [n for n in cast.names(cast.used_characters(job.channel, job.topic)) if n not in used]
                if missing:
                    job.add_log(f"note: the request named {', '.join(missing)} but the draft does not - "
                                "edit the script or request a new draft")
            job.dir.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            job.set("awaiting_approval", "script ready - review and approve",
                    title=plan.get("title", ""))
            job.add_log("script drafted - awaiting approval")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"script drafting failed: {e}")
            job.set("failed", f"script drafting failed: {e}", error=str(e))

    def approve_script(self, job_id: str, title: str | None = None,
                       narration: str | None = None) -> Job:
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
            if narration and narration.strip():
                plan["narration"] = narration.strip()
            if cast.all_characters(job.channel):   # the user may have written characters in or out
                job.add_log("cast in this video: " + (", ".join(_set_cast_used(job, plan)) or "none"))
            pfile.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            job.set("queued", "script approved - waiting for video generation",
                    title=plan.get("title", ""), script_approved=True)
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
            job.set("drafting", "rewriting script with the LLM")
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
            desc = plan.get("seo_description", "")
            token_file = channels.token_path(job.channel_id) if job.channel_id else config.YT_TOKEN_FILE
            made_for_kids = job.channel.get("made_for_kids",
                                            job.channel.get("audience") == "kids") \
                if job.channel else config.YT_MADE_FOR_KIDS
            target = f' to "{job.channel["name"]}"' if job.channel.get("name") else ""
            job.add_log(f"YouTube upload started{target}")
            vid, thumb_err = youtube.upload(
                job.dir / "final.mp4", job.dir / "thumbnail.jpg", title, desc,
                config.YT_PRIVACY, made_for_kids,
                on_progress=lambda p: job.set(yt_progress=p), token_file=token_file)
            job.set(yt_status="uploaded", yt_progress=100, yt_video_id=vid)
            job.add_log(f"uploaded to YouTube ({config.YT_PRIVACY}): https://youtu.be/{vid}")
            if thumb_err:
                job.add_log(f"note: custom thumbnail not set ({thumb_err[:100]})")
            else:
                job.add_log("custom thumbnail set on YouTube")
        except Exception as e:  # noqa: BLE001
            job.set(yt_status="failed", yt_error=str(e)[:200])
            job.add_log(f"YouTube upload failed: {e}")

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
                                            used=plan.get("cast_used")))
            (d / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            schedule.wait_until_open(on_wait=lambda s: job.set("thumbnail", schedule.describe_wait()))
            job.set("thumbnail", "rendering the new thumbnail")
            tw, th = _render_thumbnail(job, plan, d / "thumbnail.jpg", on_progress=_progress(job, "thumbnail"))
            job.set("done", "thumbnail regenerated")
            job.add_log(f"thumbnail regenerated ({tw}x{th})")
        except Exception as e:  # noqa: BLE001
            job.add_log(f"thumbnail regeneration failed: {e}")
            job.set("done", f"thumbnail regeneration failed: {e}")

    def _maintenance_target(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.archived:
            raise ValueError("unarchive the job first")
        if job.status != "done":
            raise ValueError("only finished jobs can be modified")
        if job_id == self.current or job_id in self.queue:
            raise ValueError("job is already queued or running")
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
        thread instead of waiting behind rendering jobs in the queue - unless the story
        board still needs its LLM cards / ComfyUI images, which goes through the GPU queue
        (front of the queue, like a clip regeneration)."""
        with self.lock:
            job = self._maintenance_target(job_id)
            if not job.clips:
                raise ValueError("job has no clips to assemble")
            job.assemble_requested = True
            if job.storyboard and not storyboard.assets_ready(job.dir, _load_plan(job)):
                job.set("queued", "story board assets + reassembly requested")
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
            if job_id == self.current:
                raise ValueError("job is currently running and cannot be deleted")
            if job.status in ("assembling", "thumbnail", "drafting"):
                raise ValueError(f"job is busy ({job.status}) - wait until it finishes, then delete it")
            if job.yt_status == "uploading":
                raise ValueError("job is uploading to YouTube - wait until the upload finishes")
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
