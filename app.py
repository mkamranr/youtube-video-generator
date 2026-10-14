"""FastAPI web app: submit YouTube URLs, watch progress, download the generated video."""
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from urllib.parse import parse_qs, quote, urlsplit

import requests
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline import JobManager, resolve_video_mode, upload_description
from services import (auth, avatar, bible, browser, cast, channels, compilation, llm,
                      music_upload, ownership, planner, schedule, users, video, voices)

app = FastAPI(title="VideoGen")
channels.ensure_default_channel()   # migrate the legacy single-channel setup on first start
manager = JobManager()
music_upload.sweep()                # drop staged music nobody turned into a video
# channels/jobs written before ownership existed go to the oldest admin (no-op until one exists,
# then repeated right after the first admin is created in /api/auth/setup)
_migrated = ownership.migrate_unowned(manager.jobs.values())
if _migrated["channels"] or _migrated["jobs"]:
    print(f"ownership: assigned {_migrated['channels']} channel(s) and {_migrated['jobs']} job(s) to the first admin")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """UI assets change often during development: force browsers to revalidate."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) \
            or path in ("/", "/settings", "/archive", "/planner", "/login", "/setup") \
            or path.startswith("/job/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---- authentication gate -----------------------------------------------------
# Everything needs a signed-in user except the login/setup pages, the auth endpoints,
# the static assets those pages need, the container health probe and Google's OAuth
# callback (which lands in a fresh browser tab without our cookie).
PUBLIC_PATHS = {"/login", "/setup", "/api/ping", "/api/auth/status", "/api/auth/login",
                "/api/auth/setup", "/api/auth/logout", "/api/youtube/callback"}
PUBLIC_PREFIXES = ("/css/", "/js/")


def _wants_html(request: Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("accept", "")


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    user = auth.current_user(request)
    if path in ("/login", "/setup"):
        if user is not None:
            return RedirectResponse("/", status_code=302)
        setup = not users.has_users()
        if path == "/setup" and not setup:
            return RedirectResponse("/login", status_code=302)
        if path == "/login" and setup:
            return RedirectResponse("/setup", status_code=302)
        return await call_next(request)
    if user is not None or path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    if not users.has_users():   # first run: nobody can sign in yet
        if _wants_html(request):
            return RedirectResponse("/setup", status_code=302)
        return JSONResponse({"detail": "setup required"}, status_code=401)
    if _wants_html(request):
        nxt = path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(nxt, safe='')}", status_code=302)
    return JSONResponse({"detail": "not authenticated"}, status_code=401)


# what the Video type dropdown reads, for the New Video dialog, the planned-job setup card and
# the channel editor. The values themselves live in config.VIDEO_MODES.
VIDEO_MODE_LABELS = {"song": "Song based", "story": "Story based"}


class Submit(BaseModel):
    urls: list[str] = []
    topic: str = ""
    description: str = ""
    transcript: str = ""     # a pasted transcript or song lyrics used as the script itself
    channel_id: str = ""       # required: a channel the caller owns (admins: any channel)
    aspect_ratio: str = ""
    megapixels: float | None = None   # clip resolution (0.2-1.0); None -> Settings default
    video_model: str = ""     # which video model renders the clips (GET /options video_models);
                              # "" -> Settings default
    resolution: str = ""      # final-video upscale target (GET /options resolutions); "" -> Settings default
    art_style: str = ""
    video_length: str = ""
    audio_mode: str = ""       # "" -> channel/global default; music = song, narration(+music) = voiceover
    voice_id: str = ""         # reference voice for the voiceover ("" = default voice)
    song_voice: str = ""       # singer of a generated song (config.SONG_VOICES); "" -> channel default
    lyrics: str = ""           # kids channels + audio_mode "music": sung verbatim, and they set the length
    music_upload_id: str = ""  # id from POST /api/music-uploads: the user's own track is used instead of
                               # a generated song and sets the video length (audio_mode "music" only)
    storyboard: bool | None = None   # story board overlay; None -> channel default (on for tech channels)
    captions: bool | None = None     # burned-in captions; None -> channel default (on for tech channels)
    polish: bool | None = None       # hook/end card, progress bar, whoosh; None -> channel default
    thumb_text: bool | None = None   # draw the headline on the thumbnail; None -> channel default (on)
    video_mode: str = ""       # "" -> channel default; "story" = a kids story video (re-validated here)
    repo_url: str = ""         # one GitHub repo or Hugging Face model/dataset/space URL (tech channels)
    page_scroll: bool | None = None   # scroll the page screenshot over the video; None -> on for repo jobs
    avatar_id: str = ""        # which of the channel's presenter avatars speaks ("" -> channel default)
    avatar_model: str | None = None   # avatar engine (GET /options avatar_models); None -> channel default
    avatar_coverage: str = ""  # none|intro|outro|both|full; "" -> channel default


def _queued_detail(job) -> str:
    """Live, meaningful detail for jobs waiting in the queue."""
    if job.gpu_wait and not schedule.is_open():
        return schedule.describe_wait()
    job_id = job.id
    running = manager.current
    try:
        pos = manager.queue.index(job_id) + 1
    except ValueError:
        pos = None
    if running and running in manager.jobs:
        rj = manager.jobs[running]
        name = rj.title or rj.id
        base = f'waiting in queue - "{name}" is in progress'
        return f"{base} (position {pos})" if pos and pos > 1 else base
    if pos and pos > 1:
        return f"waiting in queue (position {pos})"
    return "waiting to start"


def _owner_names() -> dict:
    """user id -> username, for the owner labels admins see on channels and jobs."""
    return {u["id"]: u["username"] for u in users.list_users()}


def _job_json(job, names: dict | None = None):
    d = asdict(job)
    d["log"] = d["log"][-40:]
    d["has_lyrics"] = bool(d.pop("lyrics", ""))   # the text itself comes from /jobs/{id}/details
    d["files"] = [f.name for f in sorted(job.dir.glob("*"))
                  if f.is_file() and f.name not in ("job.json", "bundle.zip") and not f.name.startswith(".")
                  and not f.name.startswith("music_try")]   # song candidates; the chosen one is music.mp3
    if job.status == "queued":
        d["detail"] = _queued_detail(job)
    d["owner_name"] = (_owner_names() if names is None else names).get(job.owner, "")
    return d


# ---- ownership checks: users reach only their own records, admins everything. Foreign records
# answer 404 (not 403) so ids cannot be probed.
def _owned_job(job_id: str, user: dict):
    job = manager.jobs.get(job_id)
    if not job or not auth.can_access(user, job.owner):
        raise HTTPException(404, "job not found")
    return job


def _owned_upload(upload_id: str, user: dict) -> dict:
    """A staged music track the caller may use. Checked here, not in the pipeline: the uploader
    is the caller while the job's owner is the channel's owner, and an admin may submit to
    another user's channel. Unknown and foreign ids answer alike so ids cannot be probed."""
    rec = music_upload.get(upload_id)
    if not rec or not auth.can_access(user, rec.get("owner", "")):
        raise HTTPException(400, "unknown or expired music upload")
    return rec


def _owned_channel(cid: str, user: dict) -> dict:
    ch = channels.get_channel(cid)
    if not ch or not auth.can_access(user, ch.get("owner", "")):
        raise HTTPException(404, "unknown channel")
    return ch


def _visible_channels(user: dict) -> list:
    return channels.list_channels(owner=None if auth.is_admin(user) else user["id"])


# ---- jobs -----------------------------------------------------------------
@app.post("/jobs")
def create_jobs(body: Submit, user: dict = Depends(auth.require_user)):
    urls = [u.strip() for u in body.urls if u.strip()]
    prompt_kind = ("topic" if body.topic.strip() else "description" if body.description.strip()
                   else "transcript" if body.transcript.strip()
                   else "repo" if body.repo_url.strip() else "")
    if not urls and not prompt_kind:
        raise HTTPException(400, "provide urls, a topic, a description, a transcript or a repository URL")
    if not body.channel_id:
        raise HTTPException(400, "no channel selected - create a channel first (Settings > Channels)")
    ch = channels.get_channel(body.channel_id)
    if not ch or not auth.can_access(user, ch.get("owner", "")):
        raise HTTPException(400, "unknown channel")
    # hiding the option in the UI is not enough: /jobs is a public API
    if prompt_kind == "repo" and ch.get("video_type") != "tech":
        raise HTTPException(400, "repository videos are only available on Tech channels")
    if body.lyrics.strip() and len(urls) > 1:
        raise HTTPException(400, "song lyrics apply to one video - submit one URL at a time")
    owner = ch.get("owner") or user["id"]   # the video belongs to the channel's owner
    vid = body.voice_id.strip()
    if vid and vid != voices.RANDOM_VOICE and not voices.get_voice(vid):
        raise HTTPException(400, "unknown reference voice")
    if body.music_upload_id.strip():
        if len(urls) > 1:
            raise HTTPException(400, "an uploaded music track applies to one video - "
                                     "submit one URL at a time")
        _owned_upload(body.music_upload_id, user)
    try:
        if prompt_kind:
            text = {"topic": body.topic, "description": body.description,
                    "transcript": body.transcript, "repo": body.repo_url}[prompt_kind]
            return [_job_json(manager.submit_prompt(text, prompt_kind, body.aspect_ratio,
                                                    body.art_style, body.video_length,
                                                    body.audio_mode, vid, channel=ch,
                                                    storyboard_flag=body.storyboard,
                                                    captions_flag=body.captions,
                                                    polish_flag=body.polish,
                                                    thumb_text_flag=body.thumb_text,
                                                    megapixels=body.megapixels, owner=owner,
                                                    video_model=body.video_model,
                                                    resolution=body.resolution,
                                                    song_voice=body.song_voice,
                                                    lyrics=body.lyrics,
                                                    music_upload_id=body.music_upload_id,
                                                    page_scroll=body.page_scroll,
                                                    avatar_id=body.avatar_id,
                                                    avatar_model=body.avatar_model,
                                                    avatar_coverage=body.avatar_coverage,
                                                    video_mode=body.video_mode))]
        return [_job_json(manager.submit(u, body.aspect_ratio, body.art_style, body.video_length,
                                         body.audio_mode, vid, channel=ch,
                                         storyboard_flag=body.storyboard,
                                         captions_flag=body.captions,
                                         polish_flag=body.polish,
                                         thumb_text_flag=body.thumb_text,
                                         megapixels=body.megapixels, owner=owner,
                                         video_model=body.video_model,
                                         resolution=body.resolution,
                                         song_voice=body.song_voice,
                                         lyrics=body.lyrics,
                                         music_upload_id=body.music_upload_id,
                                         avatar_id=body.avatar_id,
                                         avatar_model=body.avatar_model,
                                         avatar_coverage=body.avatar_coverage,
                                         video_mode=body.video_mode)) for u in urls]
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/jobs")
def list_jobs(archived: bool = False, user: dict = Depends(auth.require_user)):
    names = _owner_names()
    jobs = manager.list(owner=None if auth.is_admin(user) else user["id"])
    return [_job_json(j, names) for j in jobs if bool(j.archived) == archived]


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(auth.require_user)):
    return _job_json(_owned_job(job_id, user))


class RegenBody(BaseModel):
    clips: list[int]


class SceneUpdates(BaseModel):
    updates: dict[int, str]


class ApproveBody(BaseModel):
    title: str = ""
    narration: str = ""
    # story videos: the reviewed spoken lines, one per scene. None = leave them as drafted; an
    # empty list is NOT the same thing and is rejected, so a broken form cannot blank a story.
    lines: list[dict] | None = None


class RedraftBody(BaseModel):
    feedback: str = ""


class JobSetup(BaseModel):
    """The parameters a planned job is still missing; empty fields fall back to the channel."""
    aspect_ratio: str = ""
    megapixels: float | None = None
    video_model: str = ""      # "" inherits the planned job's own model
    resolution: str = ""
    art_style: str = ""
    video_length: str = ""
    audio_mode: str = ""
    voice_id: str = ""
    song_voice: str = ""
    lyrics: str = ""           # kids channels + Generate Song; "" means none (clearing the box removes them)
    music_upload_id: str = ""  # the user's own track for this video; "" means none (never inherited)
    video_mode: str = ""       # "" inherits the planned job's own format (unlike lyrics)
    storyboard: bool | None = None
    captions: bool | None = None
    polish: bool | None = None
    thumb_text: bool | None = None
    avatar_id: str = ""
    avatar_model: str | None = None
    avatar_coverage: str = ""


@app.post("/jobs/{job_id}/setup")
def setup_job(job_id: str, body: JobSetup, user: dict = Depends(auth.require_user)):
    """Start a job created from a weekly plan: apply its parameters, then draft the script."""
    _owned_job(job_id, user)
    vid = body.voice_id.strip()
    if vid and vid != voices.RANDOM_VOICE and not voices.get_voice(vid):
        raise HTTPException(400, "unknown reference voice")
    if body.music_upload_id.strip():
        _owned_upload(body.music_upload_id, user)
    try:
        return _job_json(manager.setup_planned(job_id, body.aspect_ratio, body.art_style,
                                               body.video_length, body.audio_mode, vid,
                                               storyboard_flag=body.storyboard,
                                               captions_flag=body.captions,
                                               polish_flag=body.polish,
                                               thumb_text_flag=body.thumb_text,
                                               megapixels=body.megapixels,
                                               video_model=body.video_model,
                                               resolution=body.resolution,
                                               song_voice=body.song_voice,
                                               lyrics=body.lyrics,
                                               music_upload_id=body.music_upload_id,
                                               avatar_id=body.avatar_id,
                                               avatar_model=body.avatar_model,
                                               avatar_coverage=body.avatar_coverage,
                                               video_mode=body.video_mode))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/approve")
def approve_script(job_id: str, body: ApproveBody, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.approve_script(job_id, body.title, body.narration, body.lines))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/redraft")
def redraft_script(job_id: str, body: RedraftBody, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.redraft_script(job_id, body.feedback))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.put("/jobs/{job_id}/scenes")
def update_scenes(job_id: str, body: SceneUpdates, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.update_scenes(job_id, body.updates))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/regen")
def regen_clips(job_id: str, body: RegenBody, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.request_regen(job_id, body.clips))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


class StoryboardBody(BaseModel):
    enabled: bool = True
    regenerate: bool = False   # also discard the planned cards/images so they are made afresh


@app.put("/jobs/{job_id}/storyboard")
def set_storyboard(job_id: str, body: StoryboardBody, user: dict = Depends(auth.require_user)):
    """Enable/disable the story board overlay on a finished job; applied by the next reassembly."""
    _owned_job(job_id, user)
    try:
        return _job_json(manager.set_storyboard(job_id, body.enabled, body.regenerate))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/assemble")
def assemble_job(job_id: str, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.request_assemble(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


class UpscaleBody(BaseModel):
    resolution: str


@app.post("/jobs/{job_id}/upscale")
def upscale_job(job_id: str, body: UpscaleBody, user: dict = Depends(auth.require_user)):
    """Upscale the finished final video to one of GET /options' resolutions (not Native); the
    result is kept beside the original as final_<short>p.mp4 and listed in the job's `upscales`."""
    _owned_job(job_id, user)
    try:
        return _job_json(manager.request_upscale(job_id, body.resolution))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/archive")
def archive_job(job_id: str, user: dict = Depends(auth.require_user)):
    return _toggle_archive(job_id, True, user)


@app.post("/jobs/{job_id}/unarchive")
def unarchive_job(job_id: str, user: dict = Depends(auth.require_user)):
    return _toggle_archive(job_id, False, user)


def _toggle_archive(job_id: str, flag: bool, user: dict):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.set_archived(job_id, flag))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, user: dict = Depends(auth.require_user)):
    """Remove the job and its whole output folder (clips, audio, final video, cards)."""
    _owned_job(job_id, user)
    try:
        freed, warning = manager.delete(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"deleted": job_id, "freed_mb": round(freed / 1e6, 1), "warning": warning}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str, user: dict = Depends(auth.require_user)):
    """Requeue a failed job; it continues from the step it failed at (artifacts reused)."""
    _owned_job(job_id, user)
    try:
        return _job_json(manager.resume(job_id))
    except KeyError:
        raise HTTPException(404, "unknown job")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        manager.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"cancelling": job_id}


@app.get("/jobs/{job_id}/details")
def job_details(job_id: str, user: dict = Depends(auth.require_user)):
    """Transcript text, LLM plan and full log for the detail page."""
    job = _owned_job(job_id, user)
    d = {"transcript": None, "plan": None, "log": job.log, "lyrics": job.lyrics}
    t = job.dir / "transcript.txt"
    if t.exists():
        d["transcript"] = t.read_text(encoding="utf-8", errors="replace")
    p = job.dir / "plan.json"
    if p.exists():
        try:
            d["plan"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            d["plan"] = {"raw": p.read_text(encoding="utf-8", errors="replace")}
    return d


@app.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, user: dict = Depends(auth.require_user)):
    job = _owned_job(job_id, user)
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    path = job.dir / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


def _upscale_tag(u: dict) -> str:
    """Download-name suffix of an upscaled version, e.g. '2K 1440p' from '2K (1440p)'."""
    return re.sub(r"[()]", "", u.get("label", "")).strip() or f"{u.get('width')}x{u.get('height')}"


def _safe_filename(title: str, fallback: str) -> str:
    """Filesystem-safe download name derived from the video title."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title or "").strip()
    name = re.sub(r"\s+", " ", name).rstrip(". ")
    return name[:120] or fallback


@app.get("/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, user: dict = Depends(auth.require_user)):
    """Downloads named after the video title: video, video-<short>p (an upscaled version),
    thumbnail, or the full ZIP bundle (native video + every upscaled version)."""
    job = _owned_job(job_id, user)
    d = job.dir
    title = _safe_filename(job.title, job.id)
    if kind == "video":
        f = d / "final.mp4"
        if not f.exists():
            raise HTTPException(404, "the final video has not been generated yet")
        return FileResponse(f, filename=f"{title}.mp4", media_type="video/mp4")
    if kind.startswith("video-"):
        up = next((u for u in job.upscales if u.get("file") == f"final_{kind[6:]}.mp4"), None)
        if not up or not (d / up["file"]).exists():
            raise HTTPException(404, "that upscaled version has not been generated")
        return FileResponse(d / up["file"], filename=f"{title} - {_upscale_tag(up)}.mp4", media_type="video/mp4")
    if kind == "thumbnail":
        f = d / "thumbnail.jpg"
        if not f.exists():
            raise HTTPException(404, "the thumbnail has not been generated yet")
        return FileResponse(f, filename=f"{title}.jpg", media_type="image/jpeg")
    if kind == "bundle":
        final = d / "final.mp4"
        if not final.exists():
            raise HTTPException(404, "the final video has not been generated yet")
        plan = {}
        try:
            plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        zpath = d / "bundle.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            z.write(final, f"{title}.mp4")
            for up in job.upscales:   # every on-demand upscaled version, next to the native one
                if (d / up["file"]).exists():
                    z.write(d / up["file"], f"{title} - {_upscale_tag(up)}.mp4")
            if (d / "thumbnail.jpg").exists():
                z.write(d / "thumbnail.jpg", f"{title}.jpg")
            if (d / "music.mp3").exists():
                z.write(d / "music.mp3", f"{title} - music.mp3")
            if (d / "narration.wav").exists():
                z.write(d / "narration.wav", f"{title} - narration.wav")
            res = plan.get("resource") or {}
            if res.get("offered"):   # the thing the script promised the viewer
                rname = _safe_filename(res.get("title", ""), "resource")
                for ext in ("md", "txt", "pdf"):
                    if (d / f"resource.{ext}").exists():
                        z.write(d / f"resource.{ext}", f"{title} - {rname}.{ext}")
            # the same description the YouTube upload sends, chapter block included, so what the
            # user pastes by hand matches what one click would have published
            text = (f"Title: {plan.get('seo_title') or plan.get('title') or job.title}\r\n\r\n"
                    f"Description:\r\n{upload_description(plan)}\r\n")
            z.writestr(f"{title}.txt", text)
        return FileResponse(zpath, filename=f"{title}.zip", media_type="application/zip")
    raise HTTPException(404, "unknown download kind")


# ---- authentication -----------------------------------------------------------
class Credentials(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserUpdate(BaseModel):
    role: str | None = None       # admin | user
    password: str | None = None   # set a new password (admin reset)


@app.get("/api/ping")
def ping():
    """Unauthenticated liveness probe (container health checks)."""
    return {"ok": True}


@app.get("/api/auth/status")
def auth_status(request: Request):
    u = auth.current_user(request)
    return {"setup_required": not users.has_users(), "authenticated": u is not None,
            "user": users.public(u) if u else None}


@app.post("/api/auth/setup")
def auth_setup(body: Credentials, request: Request):
    """First run only: create the administrator account and sign in."""
    if users.has_users():
        raise HTTPException(409, "setup already completed - sign in instead")
    try:
        rec = users.add_user(body.username, body.password, "admin")
    except ValueError as e:
        raise HTTPException(400, str(e))
    ownership.migrate_unowned(manager.jobs.values())   # the legacy seed channel / old jobs become theirs
    resp = JSONResponse({"user": rec})
    auth.set_cookie(resp, users.get_user(rec["id"]), request)
    return resp


@app.post("/api/auth/login")
def auth_login(body: Credentials, request: Request):
    wait = auth.locked_for(request, body.username)
    if wait:
        raise HTTPException(429, f"too many failed attempts - try again in {wait} s")
    user = users.verify_password(body.username, body.password)
    if user is None:
        auth.record_failure(request, body.username)
        raise HTTPException(401, "wrong username or password")
    auth.clear_failures(request, body.username)
    resp = JSONResponse({"user": users.public(user)})
    auth.set_cookie(resp, user, request)
    return resp


@app.post("/api/auth/logout")
def auth_logout():
    resp = JSONResponse({"ok": True})
    auth.clear_cookie(resp)
    return resp


@app.get("/api/auth/me")
def auth_me(user: dict = Depends(auth.require_user)):
    return {"user": users.public(user)}


@app.put("/api/auth/password")
def change_own_password(body: PasswordChange, request: Request, user: dict = Depends(auth.require_user)):
    if not users.check_password(body.current_password, user["pw_hash"]):
        raise HTTPException(400, "current password is incorrect")
    try:
        users.set_password(user["id"], body.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    resp = JSONResponse({"ok": True})   # other sessions are invalidated; this one is renewed
    auth.set_cookie(resp, users.get_user(user["id"]), request)
    return resp


# ---- users (admin only) ---------------------------------------------------------
@app.get("/api/users")
def list_users_api(admin: dict = Depends(auth.require_admin)):
    return {"users": users.list_users(), "roles": list(users.ROLES)}


@app.post("/api/users", status_code=201)
def create_user(body: UserCreate, admin: dict = Depends(auth.require_admin)):
    try:
        return users.add_user(body.username, body.password, body.role)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/users/{uid}")
def update_user(uid: str, body: UserUpdate, admin: dict = Depends(auth.require_admin)):
    """Change a user's role and/or reset their password."""
    if body.role is not None and body.role not in users.ROLES:
        raise HTTPException(400, f"role must be one of {list(users.ROLES)}")
    if body.role is None and not body.password:
        raise HTTPException(400, "nothing to update")
    try:
        if body.role is not None:
            users.set_role(uid, body.role)
        if body.password:
            users.set_password(uid, body.password)
    except KeyError:
        raise HTTPException(404, "unknown user")
    except ValueError as e:
        raise HTTPException(409 if "last admin" in str(e) else 400, str(e))
    rec = users.get_user(uid)
    if rec is None:
        raise HTTPException(404, "unknown user")
    return users.public(rec)


@app.delete("/api/users/{uid}")
def delete_user_api(uid: str, admin: dict = Depends(auth.require_admin)):
    """Remove an account; its channels and videos move to the admin performing the deletion."""
    try:
        users.delete_user(uid, acting_uid=admin["id"])
    except KeyError:
        raise HTTPException(404, "unknown user")
    except ValueError as e:
        raise HTTPException(409, str(e))
    moved = ownership.reassign(uid, admin["id"], manager.jobs.values())
    return {"ok": True, "reassigned": moved}


# ---- options / settings / health ------------------------------------------
def _channel_json(c: dict, names: dict | None = None) -> dict:
    from services import youtube
    return {**c, "has_logo": channels.logo_path(c["id"]).exists(),
            "has_secret": channels.secret_path(c["id"]).exists(),
            "yt_connected": youtube.is_connected(channels.token_path(c["id"])),
            "owner_name": (_owner_names() if names is None else names).get(c.get("owner", ""), ""),
            "cast": [{**x, "has_portrait": channels.portrait_path(c["id"], x["id"]).exists()}
                     for x in (c.get("cast") or []) if x.get("id")],
            "avatars": [{**x, "has_image": channels.avatar_path(c["id"], x["id"]).exists()}
                        for x in (c.get("avatars") or []) if x.get("id")]}


def _channel_secret(cid: str):
    """The channel's own OAuth client file if uploaded, else the shared one."""
    p = channels.secret_path(cid)
    return p if p.exists() else config.YT_CLIENT_SECRET_FILE


@app.get("/options")
def options(user: dict = Depends(auth.require_user)):
    names = _owner_names()
    chans = [_channel_json(c, names) for c in _visible_channels(user)]
    own = [c for c in chans if c.get("owner") == user["id"]]
    return {"me": users.public(user),
            "aspect_ratios": config.ASPECT_RATIOS, "default_aspect_ratio": config.DEFAULT_ASPECT_RATIO,
            "megapixels_options": config.MEGAPIXELS_OPTIONS, "default_megapixels": config.DEFAULT_MEGAPIXELS,
            "video_models": list(config.VIDEO_MODELS), "default_video_model": config.DEFAULT_VIDEO_MODEL,
            "resolutions": list(config.OUTPUT_RESOLUTIONS), "default_resolution": config.DEFAULT_RESOLUTION,
            "art_styles": list(config.ART_STYLES), "default_art_style": config.DEFAULT_ART_STYLE,
            "video_lengths": list(config.VIDEO_LENGTHS), "default_video_length": config.DEFAULT_VIDEO_LENGTH,
            "default_source": config.DEFAULT_SOURCE,
            "default_audio_mode": "music" if config.AUDIO_MODE == "music" else "narration+music",
            "video_modes": config.VIDEO_MODES, "default_video_mode": config.DEFAULT_VIDEO_MODE,
            "video_mode_labels": VIDEO_MODE_LABELS,
            "realistic_video_types": list(config.REALISTIC_VIDEO_TYPES),
            "avatar_models": list(config.AVATAR_MODELS),
            "default_avatar_model": config.DEFAULT_AVATAR_MODEL,
            "avatar_coverages": config.AVATAR_COVERAGES,
            "default_avatar_coverage": config.DEFAULT_AVATAR_COVERAGE,
            "avatar_estimates": _avatar_estimates(),
            "channels": chans,
            # the caller's own first channel; "" when they have none yet (admins pick explicitly)
            "default_channel_id": own[0]["id"] if own else ""}


@app.get("/api/settings")
def get_settings(admin: dict = Depends(auth.require_admin)):
    meta = {k: {"label": v[2], "help": v[3], "type": ("bool" if v[0] is bool else "number" if v[0] in (int, float) else "text")}
            for k, v in config.EDITABLE_SETTINGS.items()}
    return {"settings": config.current_settings(), "meta": meta, "aspect_ratios": config.ASPECT_RATIOS,
            "megapixels_options": config.MEGAPIXELS_OPTIONS, "resolutions": list(config.OUTPUT_RESOLUTIONS),
            "video_models": list(config.VIDEO_MODELS),
            "art_styles": list(config.ART_STYLES), "video_lengths": list(config.VIDEO_LENGTHS),
            "storyboard_template_sets": list(config.STORYBOARD_TEMPLATE_SETS),
            "yt_categories": config.YT_CATEGORIES,
            "sources": config.SOURCES}


@app.get("/api/schedule")
def get_schedule(user: dict = Depends(auth.require_user)):
    """GPU schedule window state (server local time) for the Jobs-page banner and the Settings card."""
    return schedule.status()


@app.put("/api/settings")
def put_settings(body: dict, admin: dict = Depends(auth.require_admin)):
    try:
        settings = config.save_settings(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"settings": settings}


# ---- compilations ----------------------------------------------------------
class CompilationCreate(BaseModel):
    channel_id: str = ""
    parts: list[str] = []      # job ids of the finished videos, in the order they play
    title: str = ""            # optional; the SEO step writes one when this is empty


@app.get("/api/compilations/candidates")
def compilation_candidates_api(channel_id: str = "", user: dict = Depends(auth.require_user)):
    """The finished videos that may go into a compilation, newest first.

    Grouped by nothing here - the picker groups by aspect ratio, because a compilation may only
    mix videos that share one (services.compilation.validate_parts)."""
    owner = None if auth.is_admin(user) else user["id"]
    out = []
    for job in manager.list(owner=owner):
        if not compilation.eligible(job):
            continue
        if channel_id and job.channel_id != channel_id:
            continue
        final = job.dir / (job.final or "final.mp4")
        out.append({"id": job.id, "title": job.title, "channel_id": job.channel_id,
                    "channel_name": (job.channel or {}).get("name", ""),
                    "aspect_ratio": job.aspect_ratio, "created": job.created,
                    "archived": job.archived,
                    "seconds": round(video.duration(final), 1) if final.exists() else 0.0,
                    "thumbnail": (job.dir / "thumbnail.jpg").exists()})
    return {"videos": out, "min_parts": compilation.MIN_PARTS,
            "max_parts": compilation.MAX_PARTS}


@app.post("/api/compilations", status_code=201)
def create_compilation_api(body: CompilationCreate, user: dict = Depends(auth.require_user)):
    """Stitch several finished videos into one long one. Starts at once - no GPU work."""
    for pid in body.parts:
        _owned_job(pid, user)        # 404s a foreign or unknown id before anything is created
    ch = _owned_channel(body.channel_id, user) if body.channel_id else None
    try:
        job = manager.submit_compilation(body.parts, channel=ch,
                                         owner=(ch or {}).get("owner") or user["id"],
                                         title=body.title)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _job_json(job)


# ---- weekly plans ----------------------------------------------------------
class PlanCreate(BaseModel):
    channel_id: str = ""
    week_start: str = ""       # ISO date of day 1; today when empty
    notes: str = ""            # optional steer for the week
    video_mode: str = ""       # the format every video of this week is made in; "" -> song
    videos: int = planner.DEFAULT_VIDEOS   # how many videos the week holds, spread across it


class PlanRegen(BaseModel):
    days: list[int] = []       # day numbers to rewrite; empty = the whole week
    feedback: str = ""


class PlanDay(BaseModel):
    actors: list[str] = []
    title: str = ""
    description: str = ""
    thumbnail_idea: str = ""
    hook: str = ""


class PlanJobs(BaseModel):
    days: list[int] = []       # day numbers to turn into jobs
    video_mode: str = ""       # "" -> the format stored on the plan; a value changes it and sticks


def _owned_plan(pid: str, user: dict) -> dict:
    p = planner.get_plan(pid)
    if not p or not auth.can_access(user, p.get("owner", "")):
        raise HTTPException(404, "plan not found")
    return p


def _plan_channel(plan: dict, user: dict) -> dict:
    """The plan's channel, or {} when it was deleted - a plan outlives its channel."""
    ch = channels.get_channel(plan.get("channel_id", ""))
    return ch if ch and auth.can_access(user, ch.get("owner", "")) else {}


@app.get("/api/plans")
def list_plans_api(channel_id: str = "", user: dict = Depends(auth.require_user)):
    """The caller's weekly plans (admins: everyone's), newest first."""
    owner = None if auth.is_admin(user) else user["id"]
    names = _owner_names()
    plans = planner.list_plans(owner=owner, channel_id=channel_id or None)
    return {"plans": [{**p, "owner_name": names.get(p.get("owner", ""), "")} for p in plans],
            "days": planner.DAYS, "min_videos": planner.MIN_VIDEOS,
            "max_videos": planner.MAX_VIDEOS, "default_videos": planner.DEFAULT_VIDEOS}


@app.post("/api/plans")
def create_plan_api(body: PlanCreate, user: dict = Depends(auth.require_user)):
    """Write a week of video ideas with the LLM and store it. No job is created here."""
    ch = _owned_channel(body.channel_id, user)
    week_start = planner.parse_week_start(body.week_start)
    # the planner always writes from a description, so the source never blocks a story
    try:
        mode = resolve_video_mode(body.video_mode, ch, "description")
    except ValueError as e:
        raise HTTPException(400, str(e))
    videos = planner.clamp_videos(body.videos)
    try:
        rows = llm.write_week_plan(channel=channels.snapshot(ch), days=videos,
                                   week_start=week_start, notes=body.notes,
                                   avoid=planner.previous_titles(ch["id"]), video_mode=mode)
    except Exception as e:  # noqa: BLE001  (vLLM down / bad reply)
        raise HTTPException(502, f"the LLM could not write the plan: {e}")
    days = planner.normalize_days(rows, week_start, ch, days=videos)
    return planner.create_plan(ch["id"], ch.get("owner") or user["id"], week_start, days,
                               body.notes, video_mode=mode, videos=videos)


@app.post("/api/plans/{plan_id}/regenerate")
def regenerate_plan_api(plan_id: str, body: PlanRegen, user: dict = Depends(auth.require_user)):
    """Rewrite one or more days in place. Rows that already became a job are left alone."""
    plan = _owned_plan(plan_id, user)
    ch = _plan_channel(plan, user)
    snap = channels.snapshot(ch) if ch else plan.get("channel_snapshot") or {}
    wanted = [d for d in (body.days or [r["day"] for r in plan["days"]])]
    out = plan
    for day in wanted:
        row = next((r for r in plan["days"] if r["day"] == day), None)
        if not row or row.get("job_id"):
            continue
        others = [r for r in plan["days"] if r["day"] != day]
        try:
            new = llm.write_plan_day(channel=snap, day=day, others=others, feedback=body.feedback,
                                     days=plan.get("videos") or planner.DAYS,
                                     week_start=plan["week_start"],
                                     notes=plan.get("notes", ""),
                                     avoid=planner.previous_titles(plan["channel_id"]),
                                     video_mode=plan.get("video_mode") or config.DEFAULT_VIDEO_MODE)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"the LLM could not rewrite day {day}: {e}")
        out = planner.set_day(plan_id, day,
                              planner.validate_day(new, day, plan["week_start"], ch or None, row,
                                                   videos=plan.get("videos") or planner.DAYS))
        plan = out
    return out


@app.put("/api/plans/{plan_id}/days/{day}")
def edit_plan_day_api(plan_id: str, day: int, body: PlanDay, user: dict = Depends(auth.require_user)):
    """Save a row the user edited by hand."""
    plan = _owned_plan(plan_id, user)
    row = next((r for r in plan["days"] if r["day"] == day), None)
    if not row:
        raise HTTPException(404, "unknown day")
    ch = _plan_channel(plan, user)
    clean = planner.validate_day(body.model_dump(), day, plan["week_start"], ch or None, row,
                                 videos=plan.get("videos") or planner.DAYS)
    return planner.set_day(plan_id, day, clean)


@app.post("/api/plans/{plan_id}/jobs")
def create_plan_jobs_api(plan_id: str, body: PlanJobs, user: dict = Depends(auth.require_user)):
    """Confirm rows: one paused job per day. Nothing is generated until the job is set up."""
    plan = _owned_plan(plan_id, user)
    ch = _plan_channel(plan, user)
    if not ch:
        raise HTTPException(400, "the channel of this plan no longer exists")
    owner = ch.get("owner") or user["id"]
    # resolved ONCE, before anything is created: set_job runs after each submit, so a failure
    # half-way through the loop would leave the earlier days created and stamped on disk
    try:
        mode = resolve_video_mode(body.video_mode or plan.get("video_mode"), ch, "description")
    except ValueError as e:
        raise HTTPException(400, str(e))
    if mode != plan.get("video_mode"):
        plan = planner.set_video_mode(plan_id, mode)
    made = []
    for day in body.days or []:
        row = next((r for r in plan["days"] if r["day"] == day), None)
        if not row or row.get("job_id") or not row.get("description"):
            continue
        try:
            job = manager.submit_prompt(planner.job_seed(row), "description", channel=ch,
                                        owner=owner, start=False, title=row.get("title", ""),
                                        video_mode=mode,
                                        plan={"plan_id": plan_id, "day": day,
                                              "date": row.get("date", ""), "hook": row.get("hook", ""),
                                              "thumb_idea": row.get("thumbnail_idea", "")})
        except ValueError as e:
            raise HTTPException(400, str(e))
        plan = planner.set_job(plan_id, day, job.id)
        made.append(_job_json(job))
    return {"plan": plan, "jobs": made}


@app.delete("/api/plans/{plan_id}")
def delete_plan_api(plan_id: str, user: dict = Depends(auth.require_user)):
    """Forget a plan. Jobs already created from it stay - they are real work."""
    _owned_plan(plan_id, user)
    planner.delete_plan(plan_id)
    return {"ok": True}


# ---- channels --------------------------------------------------------------
@app.get("/api/channels")
def list_channels_api(user: dict = Depends(auth.require_user)):
    from services import youtube
    names = _owner_names()
    return {"me": users.public(user),
            "channels": [_channel_json(c, names) for c in _visible_channels(user)],
            "secret": youtube.has_secret(),
            "audiences": config.AUDIENCES, "video_types": config.VIDEO_TYPES,
            "realistic_video_types": list(config.REALISTIC_VIDEO_TYPES),
            "art_styles": list(config.ART_STYLES), "aspect_ratios": config.ASPECT_RATIOS,
            "video_lengths": list(config.VIDEO_LENGTHS), "audio_modes": config.AUDIO_MODES,
            "song_voices": config.SONG_VOICES, "song_voice_labels": config.SONG_VOICE_LABELS,
            "video_modes": config.VIDEO_MODES, "video_mode_labels": VIDEO_MODE_LABELS,
            "max_cast": channels.MAX_CAST, "max_rules_chars": channels.MAX_RULES_CHARS,
            "max_avatars": channels.MAX_AVATARS,
            "avatar_models": list(config.AVATAR_MODELS),
            "avatar_model_default": config.DEFAULT_AVATAR_MODEL,
            "avatar_coverages": config.AVATAR_COVERAGES,
            "avatar_coverage_default": config.DEFAULT_AVATAR_COVERAGE,
            "avatar_estimates": _avatar_estimates()}


def _avatar_estimates() -> dict:
    """Per engine, how much presenter each coverage buys and what it costs to render.

    The UI shows this next to the picker because the two engines differ by ~33x and the
    render budget silently trims coverage - better to see the cost while choosing than to
    find it in a job log afterwards."""
    out = {}
    for name in config.AVATAR_MODELS:
        secs, mins = {}, {}
        for cov in config.AVATAR_COVERAGES:
            s = config.avatar_coverage_seconds(cov, name)
            secs[cov] = round(s, 1)
            mins[cov] = round(config.avatar_estimate_seconds(s, name) / 60.0, 1) if s else 0.0
        out[name] = {"seconds": secs, "render_minutes": mins}
    return out


class BibleBody(BaseModel):
    markdown: str


@app.post("/api/channels/parse-bible")
def parse_bible_api(body: BibleBody, user: dict = Depends(auth.require_user)):
    """Preview a pasted character bible (markdown) as channel data: {rules, cast, sections,
    warnings}. Nothing is stored - the Settings page merges the result into the channel being
    edited and the user saves through PUT /api/channels/{id}."""
    if len(body.markdown) > 200_000:
        raise HTTPException(400, "bible too large (max 200 KB)")
    out = bible.parse_bible(body.markdown, channels.MAX_CAST_KIND_CHARS, channels.MAX_CAST_NAME_CHARS)
    if not out["cast"] and not out["rules"]:
        raise HTTPException(400, "no character sections (### Name with - **Look:** bullets) or rule sections found")
    if len(out["rules"]) > channels.MAX_RULES_CHARS:
        out["rules"] = out["rules"][: channels.MAX_RULES_CHARS].rstrip()
        out["warnings"].append(f"style & rules text trimmed to {channels.MAX_RULES_CHARS} characters")
    if len(out["cast"]) > channels.MAX_CAST:
        out["warnings"].append(f"{len(out['cast'])} characters found; a channel holds at most {channels.MAX_CAST}")
    return out


@app.post("/api/channels")
def create_channel(body: dict, user: dict = Depends(auth.require_user)):
    """Create a channel owned by the caller (the owner cannot be set through the body)."""
    try:
        return _channel_json(channels.add_channel(body or {}, owner=user["id"]))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/channels/{cid}")
def update_channel(cid: str, body: dict, user: dict = Depends(auth.require_user)):
    _owned_channel(cid, user)
    try:
        return _channel_json(channels.update_channel(cid, body or {}))
    except KeyError:
        raise HTTPException(404, "unknown channel")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/channels/{cid}")
def delete_channel(cid: str, user: dict = Depends(auth.require_user)):
    _owned_channel(cid, user)
    try:
        channels.delete_channel(cid)
    except KeyError:
        raise HTTPException(404, "unknown channel")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/api/channels/{cid}/logo")
async def upload_channel_logo(cid: str, file: UploadFile, user: dict = Depends(auth.require_user)):
    """Store a channel's logo (converted to PNG with alpha); overlaid on its thumbnails."""
    _owned_channel(cid, user)
    from PIL import Image
    import io as _io
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "logo too large (max 5 MB)")
    try:
        img = Image.open(_io.BytesIO(data))
        img.load()
        img = img.convert("RGBA")
    except Exception:
        raise HTTPException(400, "not a valid image file")
    if img.width > 1024:
        img = img.resize((1024, max(1, int(img.height * 1024 / img.width))), Image.LANCZOS)
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(channels.logo_path(cid), "PNG")
    return {"ok": True, "size": [img.width, img.height]}


@app.delete("/api/channels/{cid}/logo")
def delete_channel_logo(cid: str, user: dict = Depends(auth.require_user)):
    _owned_channel(cid, user)
    try:
        channels.logo_path(cid).unlink(missing_ok=True)
    except OSError as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.get("/api/channels/{cid}/logo")
def get_channel_logo(cid: str, user: dict = Depends(auth.require_user)):
    _owned_channel(cid, user)
    p = channels.logo_path(cid)
    if not p.exists():
        raise HTTPException(404, "no logo uploaded for this channel")
    return FileResponse(p, media_type="image/png")


# ---- channel cast portraits -------------------------------------------------
def _cast_member(cid: str, char_id: str, user: dict) -> dict:
    ch = _owned_channel(cid, user)
    c = next((x for x in (ch.get("cast") or []) if x.get("id") == char_id), None)
    if c is None:
        raise HTTPException(404, "unknown cast character")
    return c


class CastGenerate(BaseModel):
    count: int = 3


class CastApprove(BaseModel):
    k: int


@app.get("/api/channels/{cid}/cast/{char_id}/portrait")
def get_cast_portrait(cid: str, char_id: str, user: dict = Depends(auth.require_user)):
    _cast_member(cid, char_id, user)
    p = channels.portrait_path(cid, char_id)
    if not p.exists():
        raise HTTPException(404, "no portrait for this character")
    return FileResponse(p, media_type="image/png")


@app.post("/api/channels/{cid}/cast/{char_id}/portrait")
async def upload_cast_portrait(cid: str, char_id: str, file: UploadFile,
                               user: dict = Depends(auth.require_user)):
    """Upload a reference portrait (any image; stored as PNG, downscaled to <=1280 px wide)."""
    _cast_member(cid, char_id, user)
    from PIL import Image
    import io as _io
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(400, "image too large (max 8 MB)")
    try:
        img = Image.open(_io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "not a valid image file")
    if img.width > 1280:
        img = img.resize((1280, max(1, int(img.height * 1280 / img.width))), Image.LANCZOS)
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(channels.portrait_path(cid, char_id), "PNG")
    return {"ok": True, "size": [img.width, img.height]}


@app.delete("/api/channels/{cid}/cast/{char_id}/portrait")
def delete_cast_portrait(cid: str, char_id: str, user: dict = Depends(auth.require_user)):
    _cast_member(cid, char_id, user)
    channels.portrait_path(cid, char_id).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/channels/{cid}/cast/{char_id}/portrait/generate")
def generate_cast_portrait(cid: str, char_id: str, body: CastGenerate | None = None,
                           user: dict = Depends(auth.require_user)):
    """Render portrait candidates with ComfyUI in the channel's art style (background thread)."""
    _cast_member(cid, char_id, user)
    if not cast.start_candidates(cid, char_id, (body or CastGenerate()).count):
        raise HTTPException(409, "portrait generation is already running for this character")
    return {"started": True}


@app.get("/api/channels/{cid}/cast/{char_id}/portrait/candidates")
def cast_portrait_candidates(cid: str, char_id: str, user: dict = Depends(auth.require_user)):
    _cast_member(cid, char_id, user)
    st = cast.candidates_state(cid, char_id)
    st["candidates"] = [f"/api/channels/{cid}/cast/{char_id}/portrait/candidate/{k}" for k in st["candidates"]]
    return st


@app.get("/api/channels/{cid}/cast/{char_id}/portrait/candidate/{k}")
def cast_portrait_candidate(cid: str, char_id: str, k: int, user: dict = Depends(auth.require_user)):
    _cast_member(cid, char_id, user)
    p = channels.candidate_path(cid, char_id, k)
    if not p.exists():
        raise HTTPException(404, "candidate not found")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/channels/{cid}/cast/{char_id}/portrait/approve")
def approve_cast_portrait(cid: str, char_id: str, body: CastApprove, user: dict = Depends(auth.require_user)):
    _cast_member(cid, char_id, user)
    try:
        cast.approve_candidate(cid, char_id, body.k)
    except FileNotFoundError:
        raise HTTPException(404, "candidate not found")
    return {"ok": True}


# ---- channel avatars (talking-head presenters) ------------------------------
# A channel may keep several presenter portraits and flag one default; a video picks which
# one speaks it. The image itself is uploaded here, the name/default live on the channel
# record, so creating one is upload-then-save exactly like a cast portrait.
def _avatar_entry(cid: str, avatar_id: str, user: dict) -> dict:
    _owned_channel(cid, user)
    ch = channels.get_channel(cid)
    a = channels.find_avatar(ch, avatar_id)
    if a is None:
        raise HTTPException(404, "unknown avatar")
    return a


@app.get("/api/channels/{cid}/avatars/{avatar_id}/image")
def get_channel_avatar(cid: str, avatar_id: str, user: dict = Depends(auth.require_user)):
    _avatar_entry(cid, avatar_id, user)
    p = channels.avatar_path(cid, avatar_id)
    if not p.exists():
        raise HTTPException(404, "no image for this avatar")
    return FileResponse(p, media_type="image/png")


@app.post("/api/channels/{cid}/avatars/{avatar_id}/image")
async def upload_channel_avatar(cid: str, avatar_id: str, file: UploadFile,
                                user: dict = Depends(auth.require_user)):
    """Store an avatar portrait, prepared for the engines: opaque, square, <=1024 px.

    A head-and-shoulders portrait on a plain background works best: the engine animates it
    and the matting step cuts the background away, so a busy one costs quality at both ends.

    The preparation happens HERE, once, rather than on every render. Squaring needs a matte to
    find the head (see avatar.square), and paying ~3 s of model load and inference per video
    for a result that only ever depends on the image is waste. It also makes the settings
    preview honest: what the channel shows is what the engine will animate."""
    _avatar_entry(cid, avatar_id, user)
    from PIL import Image
    import io as _io
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(400, "image too large (max 8 MB)")
    try:
        img = Image.open(_io.BytesIO(data))
        img.load()
        if img.mode in ("RGBA", "LA", "P"):
            # Composite onto grey rather than letting convert() drop the alpha channel: a
            # cut-out PNG would otherwise arrive with whatever RGB sat under the transparency,
            # which is usually black, and the engine would animate a portrait on a black slab.
            img = img.convert("RGBA")
            img = Image.alpha_composite(Image.new("RGBA", img.size, (128, 128, 128, 255)), img)
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "not a valid image file")
    img = avatar.square(img)
    if img.width > config.AVATAR_STORE_PX:
        side = config.AVATAR_STORE_PX
        img = img.resize((side, side), Image.LANCZOS)
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(channels.avatar_path(cid, avatar_id), "PNG")
    return {"ok": True, "size": [img.width, img.height]}


@app.delete("/api/channels/{cid}/avatars/{avatar_id}/image")
def delete_channel_avatar_image(cid: str, avatar_id: str, user: dict = Depends(auth.require_user)):
    _avatar_entry(cid, avatar_id, user)
    channels.avatar_path(cid, avatar_id).unlink(missing_ok=True)
    return {"ok": True}


# ---- reference voices ------------------------------------------------------
# A shared library: admins manage it, every user can list the voices and pick one per video.
@app.get("/api/voices")
def list_voices(user: dict = Depends(auth.require_user)):
    return {"voices": voices.list_voices()}


@app.post("/api/voices")
async def create_voice(file: UploadFile, name: str = Form(...), text: str = Form(...),
                       admin: dict = Depends(auth.require_admin)):
    """Add a reference voice for Fish Speech cloning: audio sample + the text spoken in it."""
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "audio too large (max 20 MB)")
    try:
        return voices.add_voice(name, text, data, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str, admin: dict = Depends(auth.require_admin)):
    try:
        voices.delete_voice(voice_id)
    except KeyError:
        raise HTTPException(404, "unknown reference voice")
    return {"ok": True}


@app.get("/api/voices/{voice_id}/audio")
def voice_audio(voice_id: str, user: dict = Depends(auth.require_user)):
    path = voices.voice_audio_path(voice_id)
    if not path.exists():
        raise HTTPException(404, "unknown reference voice")
    return FileResponse(path, media_type="audio/wav")


# ---- uploaded music tracks -------------------------------------------------
# Per-user staging for ONE video: the id returned here goes into POST /jobs (or /setup), which
# copies the track to that job's music.mp3 and consumes the upload. Unlike the shared reference
# voices this is not an admin library - anyone may bring their own soundtrack for their own video.
@app.post("/api/music-uploads")
async def create_music_upload(file: UploadFile, user: dict = Depends(auth.require_user)):
    """Validate and stage a music track to use instead of a generated song."""
    data = await file.read()
    if len(data) > music_upload.MAX_BYTES:
        raise HTTPException(400, f"music file too large (max {music_upload.MAX_BYTES // (1024 * 1024)} MB)")
    try:
        return music_upload.stage(data, user["id"], file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/music-uploads/{upload_id}")
def delete_music_upload(upload_id: str, user: dict = Depends(auth.require_user)):
    """Drop a staged track the user decided not to use."""
    _owned_upload(upload_id, user)
    music_upload.discard(upload_id)
    return {"ok": True}


@app.post("/jobs/{job_id}/regen-resource")
def regen_resource(job_id: str, user: dict = Depends(auth.require_user)):
    """Re-decide what the script promised the viewer and rewrite the hand-out."""
    _owned_job(job_id, user)
    try:
        return _job_json(manager.regen_resource(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


class ThumbBody(BaseModel):
    variant: int | None = None       # 1-based; 0 = the clean render with no text
    texts: list[str] | None = None   # replace the overlay lines and redraw


@app.put("/jobs/{job_id}/thumbnail")
def set_thumbnail(job_id: str, body: ThumbBody, user: dict = Depends(auth.require_user)):
    """Pick the thumbnail text variant, and/or edit the lines. Pillow only, so it returns at once."""
    _owned_job(job_id, user)
    try:
        return _job_json(manager.set_thumbnail(job_id, variant=body.variant, texts=body.texts))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/regen-thumbnail")
def regen_thumbnail(job_id: str, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.regen_thumbnail(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _oauth_redirect(request: Request) -> tuple:
    """(redirect_uri, manual) for the Google consent flow.

    Desktop-type OAuth clients accept only loopback redirects, so by default the flow
    returns to http://localhost:<port>/api/youtube/callback. That completes automatically
    when the browser runs on the server machine; from any other computer the page cannot
    load and the user pastes its address into Settings instead (manual=True). A configured
    PUBLIC_BASE_URL (Web-type client on an HTTPS domain) is used verbatim."""
    if config.PUBLIC_BASE_URL:
        return config.PUBLIC_BASE_URL.rstrip("/") + "/api/youtube/callback", False
    host = (request.url.hostname or "").lower()
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    suffix = "" if port in (80, 443) else f":{port}"
    return f"http://localhost{suffix}/api/youtube/callback", host not in LOOPBACK_HOSTS


def _parse_oauth_return(text: str) -> tuple:
    """(code, state) from the pasted callback address, its query string, or a bare code."""
    text = (text or "").strip()
    if not text:
        raise ValueError("paste the address of the page Google sent you to")
    if "code=" in text:
        query = urlsplit(text).query if "://" in text else text.lstrip("?")
        q = parse_qs(query)
        code, state = (q.get("code") or [""])[0].strip(), (q.get("state") or [""])[0].strip()
        if not code:
            raise ValueError("no authorization code found in the pasted address")
        return code, state
    if " " in text or "/" in text:
        raise ValueError("that does not look like a Google callback address or authorization code")
    return text, ""


class OAuthFinish(BaseModel):
    url: str


@app.get("/api/channels/{cid}/youtube/auth")
def channel_youtube_auth(cid: str, request: Request, user: dict = Depends(auth.require_user)):
    from services import youtube
    _owned_channel(cid, user)
    secret = _channel_secret(cid)
    if not youtube.has_secret(secret):
        raise HTTPException(409, "no OAuth client secret - upload one for this channel "
                                 "or a shared one on the Settings page")
    redirect, manual = _oauth_redirect(request)
    try:
        return {"url": youtube.auth_url(redirect, channels.token_path(cid), secret_file=secret),
                "redirect_uri": redirect, "manual": manual}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"could not start authorization: {e}")


@app.post("/api/channels/{cid}/youtube/finish")
def channel_youtube_finish(cid: str, body: OAuthFinish, user: dict = Depends(auth.require_user)):
    """Complete a consent flow whose loopback redirect could not reach the server: the user
    pastes the address Google sent the browser to (or the bare authorization code)."""
    from services import youtube
    _owned_channel(cid, user)
    try:
        code, state = _parse_oauth_return(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not state:
        state = youtube.pending_state_for(channels.token_path(cid))
        if not state:
            raise HTTPException(409, "no authorization in progress - click Connect YouTube again")
    try:
        youtube.finish_auth(code, state)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Google rejected the code: {str(e)[:200]}")
    return _channel_json(channels.get_channel(cid))


async def _read_secret_upload(file: UploadFile) -> bytes:
    from services import youtube
    data = await file.read()
    if len(data) > 64 * 1024:
        raise HTTPException(400, "file too large for an OAuth client file (max 64 KB)")
    try:
        youtube.validate_secret(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return data


@app.post("/api/channels/{cid}/secret")
async def upload_channel_secret(cid: str, file: UploadFile, user: dict = Depends(auth.require_user)):
    """Store a channel's own OAuth client file (overrides the shared one for this channel)."""
    _owned_channel(cid, user)
    data = await _read_secret_upload(file)
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    channels.secret_path(cid).write_bytes(data)
    return {"ok": True}


@app.delete("/api/channels/{cid}/secret")
def delete_channel_secret(cid: str, user: dict = Depends(auth.require_user)):
    _owned_channel(cid, user)
    channels.secret_path(cid).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/youtube/secret")
async def upload_shared_secret(file: UploadFile, admin: dict = Depends(auth.require_admin)):
    """Store the shared OAuth client file used by channels without their own (admin only)."""
    data = await _read_secret_upload(file)
    config.YT_CLIENT_SECRET_FILE.write_bytes(data)
    return {"ok": True}


@app.delete("/api/youtube/secret")
def delete_shared_secret(admin: dict = Depends(auth.require_admin)):
    config.YT_CLIENT_SECRET_FILE.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/youtube/callback")
def youtube_callback(code: str = "", state: str = "", error: str = ""):
    from services import youtube
    if error or not code:
        return HTMLResponse(f"<h3>Authorization failed: {error or 'no code returned'}</h3>")
    try:
        youtube.finish_auth(code, state)
    except Exception as e:  # noqa: BLE001
        return HTMLResponse(f"<h3>Authorization failed: {e}</h3>")
    return HTMLResponse("<h3>YouTube connected. You can close this tab and return to VideoGen.</h3>")


@app.delete("/api/channels/{cid}/youtube/auth")
def channel_youtube_disconnect(cid: str, user: dict = Depends(auth.require_user)):
    from services import youtube
    _owned_channel(cid, user)
    youtube.disconnect(channels.token_path(cid))
    return {"ok": True}


@app.post("/jobs/{job_id}/youtube-upload")
def youtube_upload(job_id: str, user: dict = Depends(auth.require_user)):
    _owned_job(job_id, user)
    try:
        return _job_json(manager.start_youtube_upload(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/health")
def health(admin: dict = Depends(auth.require_admin)):
    targets = {
        "comfyui": f"{config.COMFY_URL.rstrip('/')}/system_stats",
        "vllm": f"{config.VLLM_URL.rstrip('/')}/models",
        "fish": f"{config.FISH_URL.rstrip('/')}/gradio_api/info",
    }

    def ping(url):
        t0 = time.time()
        try:
            r = requests.get(url, timeout=5)
            return {"ok": r.status_code == 200, "ms": int((time.time() - t0) * 1000),
                    "error": None if r.status_code == 200 else f"HTTP {r.status_code}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "ms": None, "error": str(e)[:150]}

    with ThreadPoolExecutor(len(targets)) as ex:
        results = dict(zip(targets, ex.map(ping, targets.values())))
    # local, not a service: without it repository videos still work, they just get no page overlay
    binary = browser.resolve()
    results["browser"] = {"ok": bool(binary), "ms": None,
                          "error": None if binary else "no Chrome/Chromium found (page scroll overlay "
                                                       "disabled); set CHROME_BIN or install chromium"}
    return results


# ---- art style previews ----------------------------------------------------
PREVIEW_SUBJECT = ("A cheerful young fox explorer waving happily in a sunny meadow, holding a small "
                   "red balloon. No text, subtitles, logos or watermarks of any kind.")
_preview_state = {"generating": False, "current": "", "errors": 0, "waiting": False}
_preview_lock = threading.Lock()


def _style_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _preview_dir():
    d = config.STATIC_DIR / "style_previews"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.get("/api/style-previews")
def style_previews(user: dict = Depends(auth.require_user)):
    d = _preview_dir()
    previews = {}
    for name in config.ART_STYLES:
        f = d / f"{_style_slug(name)}.jpg"
        previews[name] = f"/style_previews/{f.name}" if f.exists() and f.stat().st_size > 1000 else None
    return {"previews": previews, "generating": _preview_state["generating"],
            "done": sum(1 for v in previews.values() if v), "total": len(previews),
            "current": _preview_state["current"], "errors": _preview_state["errors"],
            "waiting": _preview_state["waiting"]}


@app.post("/api/style-previews")
def generate_style_previews(admin: dict = Depends(auth.require_admin)):
    with _preview_lock:
        if _preview_state["generating"]:
            raise HTTPException(409, "preview generation is already running")
        _preview_state["generating"] = True
        _preview_state["errors"] = 0
    threading.Thread(target=_generate_previews, daemon=True).start()
    return {"started": True}


def _generate_previews():
    """Render one sample image per art style through the video model (skips existing)."""
    from services.comfy import ComfyClient
    from services import video as videosvc
    d = _preview_dir()
    try:
        client = ComfyClient()
        template = client.load_workflow(config.T2V_WORKFLOW)
        for name, desc in config.ART_STYLES.items():
            slug = _style_slug(name)
            out = d / f"{slug}.jpg"
            if out.exists() and out.stat().st_size > 1000:
                continue
            _preview_state["current"] = name
            raw = d / f"_{slug}_raw.mp4"
            try:
                if not schedule.is_open():   # admin renders obey the GPU window too
                    _preview_state["waiting"] = True
                    schedule.wait_until_open()
                    _preview_state["waiting"] = False
                wf = client.build_thumbnail(template, f"{desc}\n\n{PREVIEW_SUBJECT}",
                                            f"videogen/_style_previews/{slug}")
                client.run(wf, raw, cleanup=config.COMFY_CLEANUP)
                videosvc.make_thumbnail(raw, out, 640, 360)
            except Exception:  # noqa: BLE001 - skip this style, keep going
                _preview_state["errors"] += 1
            finally:
                try:
                    raw.unlink(missing_ok=True)
                except OSError:
                    pass
    finally:
        _preview_state["generating"] = False
        _preview_state["current"] = ""
        _preview_state["waiting"] = False


# ---- pages ----------------------------------------------------------------
@app.get("/job/{job_id}")
def job_page(job_id: str):
    return FileResponse(config.STATIC_DIR / "job.html")


@app.get("/settings")
def settings_page():
    return FileResponse(config.STATIC_DIR / "settings.html")


@app.get("/login")
def login_page():
    return FileResponse(config.STATIC_DIR / "login.html")


@app.get("/setup")
def setup_page():
    return FileResponse(config.STATIC_DIR / "setup.html")


@app.get("/archive")
def archive_page():
    return FileResponse(config.STATIC_DIR / "archive.html")


@app.get("/planner")
def planner_page():
    return FileResponse(config.STATIC_DIR / "planner.html")


app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
