"""FastAPI web app: submit YouTube URLs, watch progress, download the generated video."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline import JobManager

app = FastAPI(title="VideoGen")
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
    urls: list[str]
    aspect_ratio: str = ""
    art_style: str = ""


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
    d["files"] = [f.name for f in sorted(job.dir.glob("*")) if f.is_file() and f.name != "job.json"]
    if job.status == "queued":
        d["detail"] = _queued_detail(job.id)
    return d


# ---- jobs -----------------------------------------------------------------
@app.post("/jobs")
def create_jobs(body: Submit):
    urls = [u.strip() for u in body.urls if u.strip()]
    if not urls:
        raise HTTPException(400, "no urls given")
    ar = body.aspect_ratio or config.DEFAULT_ASPECT_RATIO
    if ar not in config.ASPECT_RATIOS:
        raise HTTPException(400, f"aspect_ratio must be one of {config.ASPECT_RATIOS}")
    style = body.art_style or config.DEFAULT_ART_STYLE
    if style not in config.ART_STYLES:
        raise HTTPException(400, f"art_style must be one of {list(config.ART_STYLES)}")
    return [_job_json(manager.submit(u, ar, style)) for u in urls]


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


# ---- options / settings / health ------------------------------------------
@app.get("/options")
def options():
    return {"aspect_ratios": config.ASPECT_RATIOS, "default_aspect_ratio": config.DEFAULT_ASPECT_RATIO,
            "art_styles": list(config.ART_STYLES), "default_art_style": config.DEFAULT_ART_STYLE}


@app.get("/api/settings")
def get_settings():
    meta = {k: {"label": v[2], "help": v[3], "type": ("bool" if v[0] is bool else "number" if v[0] in (int, float) else "text")}
            for k, v in config.EDITABLE_SETTINGS.items()}
    return {"settings": config.current_settings(), "meta": meta, "aspect_ratios": config.ASPECT_RATIOS,
            "art_styles": list(config.ART_STYLES)}


@app.put("/api/settings")
def put_settings(body: dict):
    try:
        settings = config.save_settings(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"settings": settings}


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
