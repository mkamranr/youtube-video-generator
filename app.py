"""FastAPI web app: submit YouTube URLs, watch progress, download the generated video."""
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline import JobManager
from services import channels, voices

app = FastAPI(title="VideoGen")
channels.ensure_default_channel()   # migrate the legacy single-channel setup on first start
manager = JobManager()


@app.middleware("http")
async def no_cache_static(request, call_next):
    """UI assets change often during development: force browsers to revalidate."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path in ("/", "/settings", "/archive") or path.startswith("/job/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Submit(BaseModel):
    urls: list[str] = []
    topic: str = ""
    description: str = ""
    channel_id: str = ""       # "" -> the first channel (backcompat for scripted API use)
    aspect_ratio: str = ""
    art_style: str = ""
    video_length: str = ""
    audio_mode: str = ""       # "" -> channel/global default; music = song, narration(+music) = voiceover
    voice_id: str = ""         # reference voice for the voiceover ("" = default voice)


def _queued_detail(job_id: str) -> str:
    """Live, meaningful detail for jobs waiting in the queue."""
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


def _job_json(job):
    d = asdict(job)
    d["log"] = d["log"][-40:]
    d["files"] = [f.name for f in sorted(job.dir.glob("*"))
                  if f.is_file() and f.name not in ("job.json", "bundle.zip")]
    if job.status == "queued":
        d["detail"] = _queued_detail(job.id)
    return d


# ---- jobs -----------------------------------------------------------------
@app.post("/jobs")
def create_jobs(body: Submit):
    urls = [u.strip() for u in body.urls if u.strip()]
    prompt_kind = "topic" if body.topic.strip() else "description" if body.description.strip() else ""
    if not urls and not prompt_kind:
        raise HTTPException(400, "provide urls, a topic, or a description")
    if body.channel_id:
        ch = channels.get_channel(body.channel_id)
        if not ch:
            raise HTTPException(400, "unknown channel")
    else:  # backcompat: scripted API calls without a channel use the first one
        chans = channels.list_channels()
        ch = chans[0] if chans else None
    vid = body.voice_id.strip()
    if vid and not voices.get_voice(vid):
        raise HTTPException(400, "unknown reference voice")
    try:
        if prompt_kind:
            text = body.topic if prompt_kind == "topic" else body.description
            return [_job_json(manager.submit_prompt(text, prompt_kind, body.aspect_ratio,
                                                    body.art_style, body.video_length,
                                                    body.audio_mode, vid, channel=ch))]
        return [_job_json(manager.submit(u, body.aspect_ratio, body.art_style, body.video_length,
                                         body.audio_mode, vid, channel=ch)) for u in urls]
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/jobs")
def list_jobs(archived: bool = False):
    return [_job_json(j) for j in manager.list() if bool(j.archived) == archived]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return _job_json(job)


class RegenBody(BaseModel):
    clips: list[int]


class SceneUpdates(BaseModel):
    updates: dict[int, str]


class ApproveBody(BaseModel):
    title: str = ""
    narration: str = ""


class RedraftBody(BaseModel):
    feedback: str = ""


@app.post("/jobs/{job_id}/approve")
def approve_script(job_id: str, body: ApproveBody):
    try:
        return _job_json(manager.approve_script(job_id, body.title, body.narration))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/redraft")
def redraft_script(job_id: str, body: RedraftBody):
    try:
        return _job_json(manager.redraft_script(job_id, body.feedback))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.put("/jobs/{job_id}/scenes")
def update_scenes(job_id: str, body: SceneUpdates):
    try:
        return _job_json(manager.update_scenes(job_id, body.updates))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/regen")
def regen_clips(job_id: str, body: RegenBody):
    try:
        return _job_json(manager.request_regen(job_id, body.clips))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/assemble")
def assemble_job(job_id: str):
    try:
        return _job_json(manager.request_assemble(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/archive")
def archive_job(job_id: str):
    return _toggle_archive(job_id, True)


@app.post("/jobs/{job_id}/unarchive")
def unarchive_job(job_id: str):
    return _toggle_archive(job_id, False)


def _toggle_archive(job_id: str, flag: bool):
    try:
        return _job_json(manager.set_archived(job_id, flag))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        manager.delete(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"deleted": job_id}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Requeue a failed job; it continues from the step it failed at (artifacts reused)."""
    try:
        return _job_json(manager.resume(job_id))
    except KeyError:
        raise HTTPException(404, "unknown job")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        manager.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"cancelling": job_id}


@app.get("/jobs/{job_id}/details")
def job_details(job_id: str):
    """Transcript text, LLM plan and full log for the detail page."""
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    d = {"transcript": None, "plan": None, "log": job.log}
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
def job_file(job_id: str, name: str):
    job = manager.jobs.get(job_id)
    if not job or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    path = job.dir / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


def _safe_filename(title: str, fallback: str) -> str:
    """Filesystem-safe download name derived from the video title."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title or "").strip()
    name = re.sub(r"\s+", " ", name).rstrip(". ")
    return name[:120] or fallback


@app.get("/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str):
    """Downloads named after the video title: video, thumbnail, or the full ZIP bundle."""
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    d = job.dir
    title = _safe_filename(job.title, job.id)
    if kind == "video":
        f = d / "final.mp4"
        if not f.exists():
            raise HTTPException(404, "the final video has not been generated yet")
        return FileResponse(f, filename=f"{title}.mp4", media_type="video/mp4")
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
            if (d / "thumbnail.jpg").exists():
                z.write(d / "thumbnail.jpg", f"{title}.jpg")
            if (d / "music.mp3").exists():
                z.write(d / "music.mp3", f"{title} - music.mp3")
            if (d / "narration.wav").exists():
                z.write(d / "narration.wav", f"{title} - narration.wav")
            text = (f"Title: {plan.get('seo_title') or plan.get('title') or job.title}\r\n\r\n"
                    f"Description:\r\n{plan.get('seo_description', '')}\r\n")
            z.writestr(f"{title}.txt", text)
        return FileResponse(zpath, filename=f"{title}.zip", media_type="application/zip")
    raise HTTPException(404, "unknown download kind")


# ---- options / settings / health ------------------------------------------
def _channel_json(c: dict) -> dict:
    from services import youtube
    return {**c, "has_logo": channels.logo_path(c["id"]).exists(),
            "has_secret": channels.secret_path(c["id"]).exists(),
            "yt_connected": youtube.is_connected(channels.token_path(c["id"]))}


def _channel_secret(cid: str):
    """The channel's own OAuth client file if uploaded, else the shared one."""
    p = channels.secret_path(cid)
    return p if p.exists() else config.YT_CLIENT_SECRET_FILE


@app.get("/options")
def options():
    chans = [_channel_json(c) for c in channels.list_channels()]
    return {"aspect_ratios": config.ASPECT_RATIOS, "default_aspect_ratio": config.DEFAULT_ASPECT_RATIO,
            "art_styles": list(config.ART_STYLES), "default_art_style": config.DEFAULT_ART_STYLE,
            "video_lengths": list(config.VIDEO_LENGTHS), "default_video_length": config.DEFAULT_VIDEO_LENGTH,
            "default_source": config.DEFAULT_SOURCE,
            "default_audio_mode": "music" if config.AUDIO_MODE == "music" else "narration+music",
            "channels": chans, "default_channel_id": chans[0]["id"] if chans else ""}


@app.get("/api/settings")
def get_settings():
    meta = {k: {"label": v[2], "help": v[3], "type": ("bool" if v[0] is bool else "number" if v[0] in (int, float) else "text")}
            for k, v in config.EDITABLE_SETTINGS.items()}
    return {"settings": config.current_settings(), "meta": meta, "aspect_ratios": config.ASPECT_RATIOS,
            "art_styles": list(config.ART_STYLES), "video_lengths": list(config.VIDEO_LENGTHS),
            "sources": config.SOURCES}


@app.put("/api/settings")
def put_settings(body: dict):
    try:
        settings = config.save_settings(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"settings": settings}


# ---- channels --------------------------------------------------------------
@app.get("/api/channels")
def list_channels_api():
    from services import youtube
    return {"channels": [_channel_json(c) for c in channels.list_channels()],
            "secret": youtube.has_secret(),
            "audiences": config.AUDIENCES, "video_types": config.VIDEO_TYPES,
            "art_styles": list(config.ART_STYLES), "aspect_ratios": config.ASPECT_RATIOS,
            "video_lengths": list(config.VIDEO_LENGTHS), "audio_modes": config.AUDIO_MODES}


@app.post("/api/channels")
def create_channel(body: dict):
    try:
        return _channel_json(channels.add_channel(body or {}))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/channels/{cid}")
def update_channel(cid: str, body: dict):
    try:
        return _channel_json(channels.update_channel(cid, body or {}))
    except KeyError:
        raise HTTPException(404, "unknown channel")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/channels/{cid}")
def delete_channel(cid: str):
    try:
        channels.delete_channel(cid)
    except KeyError:
        raise HTTPException(404, "unknown channel")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/api/channels/{cid}/logo")
async def upload_channel_logo(cid: str, file: UploadFile):
    """Store a channel's logo (converted to PNG with alpha); overlaid on its thumbnails."""
    if not channels.get_channel(cid):
        raise HTTPException(404, "unknown channel")
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
def delete_channel_logo(cid: str):
    try:
        channels.logo_path(cid).unlink(missing_ok=True)
    except OSError as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.get("/api/channels/{cid}/logo")
def get_channel_logo(cid: str):
    p = channels.logo_path(cid)
    if not p.exists():
        raise HTTPException(404, "no logo uploaded for this channel")
    return FileResponse(p, media_type="image/png")


# ---- reference voices ------------------------------------------------------
@app.get("/api/voices")
def list_voices():
    return {"voices": voices.list_voices()}


@app.post("/api/voices")
async def create_voice(file: UploadFile, name: str = Form(...), text: str = Form(...)):
    """Add a reference voice for Fish Speech cloning: audio sample + the text spoken in it."""
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "audio too large (max 20 MB)")
    try:
        return voices.add_voice(name, text, data, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    try:
        voices.delete_voice(voice_id)
    except KeyError:
        raise HTTPException(404, "unknown reference voice")
    return {"ok": True}


@app.get("/api/voices/{voice_id}/audio")
def voice_audio(voice_id: str):
    path = voices.voice_audio_path(voice_id)
    if not path.exists():
        raise HTTPException(404, "unknown reference voice")
    return FileResponse(path, media_type="audio/wav")


@app.post("/jobs/{job_id}/regen-thumbnail")
def regen_thumbnail(job_id: str):
    try:
        return _job_json(manager.regen_thumbnail(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/channels/{cid}/youtube/auth")
def channel_youtube_auth(cid: str, request: Request):
    from services import youtube
    if not channels.get_channel(cid):
        raise HTTPException(404, "unknown channel")
    secret = _channel_secret(cid)
    if not youtube.has_secret(secret):
        raise HTTPException(409, "no OAuth client secret - upload one for this channel "
                                 "or a shared one on the Settings page")
    redirect = str(request.base_url).rstrip("/") + "/api/youtube/callback"
    try:
        return {"url": youtube.auth_url(redirect, channels.token_path(cid), secret_file=secret)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"could not start authorization: {e}")


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
async def upload_channel_secret(cid: str, file: UploadFile):
    """Store a channel's own OAuth client file (overrides the shared one for this channel)."""
    if not channels.get_channel(cid):
        raise HTTPException(404, "unknown channel")
    data = await _read_secret_upload(file)
    config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
    channels.secret_path(cid).write_bytes(data)
    return {"ok": True}


@app.delete("/api/channels/{cid}/secret")
def delete_channel_secret(cid: str):
    channels.secret_path(cid).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/youtube/secret")
async def upload_shared_secret(file: UploadFile):
    """Store the shared OAuth client file used by channels without their own."""
    data = await _read_secret_upload(file)
    config.YT_CLIENT_SECRET_FILE.write_bytes(data)
    return {"ok": True}


@app.delete("/api/youtube/secret")
def delete_shared_secret():
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
def channel_youtube_disconnect(cid: str):
    from services import youtube
    youtube.disconnect(channels.token_path(cid))
    return {"ok": True}


@app.post("/jobs/{job_id}/youtube-upload")
def youtube_upload(job_id: str):
    try:
        return _job_json(manager.start_youtube_upload(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/health")
def health():
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
    return results


# ---- art style previews ----------------------------------------------------
PREVIEW_SUBJECT = ("A cheerful young fox explorer waving happily in a sunny meadow, holding a small "
                   "red balloon. No text, subtitles, logos or watermarks of any kind.")
_preview_state = {"generating": False, "current": "", "errors": 0}
_preview_lock = threading.Lock()


def _style_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _preview_dir():
    d = config.STATIC_DIR / "style_previews"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.get("/api/style-previews")
def style_previews():
    d = _preview_dir()
    previews = {}
    for name in config.ART_STYLES:
        f = d / f"{_style_slug(name)}.jpg"
        previews[name] = f"/style_previews/{f.name}" if f.exists() and f.stat().st_size > 1000 else None
    return {"previews": previews, "generating": _preview_state["generating"],
            "done": sum(1 for v in previews.values() if v), "total": len(previews),
            "current": _preview_state["current"], "errors": _preview_state["errors"]}


@app.post("/api/style-previews")
def generate_style_previews():
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


# ---- pages ----------------------------------------------------------------
@app.get("/job/{job_id}")
def job_page(job_id: str):
    return FileResponse(config.STATIC_DIR / "job.html")


@app.get("/settings")
def settings_page():
    return FileResponse(config.STATIC_DIR / "settings.html")


@app.get("/archive")
def archive_page():
    return FileResponse(config.STATIC_DIR / "archive.html")


app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
