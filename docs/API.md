# HTTP API Reference

Base URL: `http://localhost:8080`. All request/response bodies are JSON. Errors return `{"detail": "<message>"}` with an appropriate status code (`400` invalid input, `404` unknown job, `409` state conflict).

## Jobs

### `POST /jobs`
Submit one job per URL.

```json
{ "urls": ["https://www.youtube.com/watch?v=..."], "aspect_ratio": "16:9 (Widescreen)", "art_style": "Pixar-like 3D" }
```
Or create a job from a prompt instead of URLs (one job per request):

```json
{ "topic": "How bees make honey" }
{ "description": "A gentle bedtime story about ..." }
```
Prompt jobs draft their script immediately in a background thread (never in the GPU queue) and rest at `awaiting_approval` until approved. `aspect_ratio`, `art_style` and `video_length` (`Auto`, `30-45 seconds`, `45-60 seconds`, `1-2 minutes`, `2-3 minutes` — a soft target guiding script and audio length) are optional and validated against `GET /options`. Returns a list of job objects.

### `GET /jobs?archived=false`
List jobs, newest first. `archived=true` lists archived jobs instead. Queued jobs carry a live human-readable `detail` (queue position / what is currently running).

### `GET /jobs/{id}`
One job object:

```json
{
  "id": "20260826-060951-2e6720",
  "url": "...", "title": "...", "status": "clips",
  "detail": "clip 12/31: sampling 8/20",
  "aspect_ratio": "16:9 (Widescreen)", "art_style": "Pixar-like 3D", "audio_mode": "music",
  "clips_total": 31, "clips_done": 11, "clips": ["clip_01.mp4", "..."],
  "final": "final.mp4", "error": "",
  "stale_clips": [5], "regen_queue": [], "needs_assemble": false, "archived": false,
  "files": ["clip_01.mp4", "music.mp3", "thumbnail.jpg", "..."],
  "log": ["..."], "created": 0, "updated": 0
}
```

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → clips → assembling → done | failed`.
Prompt-based jobs (`source` = `topic`/`description`) start with `drafting → awaiting_approval` and enter the queue only when approved.

### `GET /jobs/{id}/details`
Large payload for the job page: full `transcript` text, the parsed `plan` (narration/story, style, scenes, per-clip audio `segments`, `music` caption+lyrics, `seo_title`, `seo_description`, `thumbnail_prompt`), and the full `log`.

### `GET /jobs/{id}/files/{name}`
Download any artifact from the job folder (clips, `final.mp4`, `thumbnail.jpg`, `music.mp3`, `narration.wav`, `plan.json`, `transcript.txt`, …).

### `DELETE /jobs/{id}`
Delete the job and all its local files. `409` if it is currently running.

### `POST /jobs/{id}/cancel`
Cancel a queued or running job. Takes effect within seconds; the in-flight ComfyUI render is aborted (only if it belongs to this job). A cancelled *revision* request returns the job to `done`; a cancelled full run becomes `failed: cancelled by user`.

## Revision workflow

### `PUT /jobs/{id}/scenes`
Edit scene prompts of a **finished** job. Keys are 1-based clip numbers.

```json
{ "updates": { "5": "New scene prompt text ..." } }
```
Changed clips are added to `stale_clips` (shown badged in the UI). `409` if the job is running/queued/archived, the index is out of range, or a prompt is empty.

### `POST /jobs/{id}/regen`
Regenerate specific clips of a finished job with their (possibly edited) scene prompts and a fresh seed.

```json
{ "clips": [5, 6] }
```
The request goes to the **front of the queue**; a currently running job pauses at its next clip boundary and automatically resumes afterwards. Regenerated clips clear their stale flag and set `needs_assemble`.

### `POST /jobs/{id}/assemble`
Re-concatenate all clips and remix the audio into a new `final.mp4`. Runs **immediately in a background thread** (pure ffmpeg, no GPU) — it never waits behind rendering jobs. Clears `needs_assemble`.

### `POST /jobs/{id}/approve`
Approve the drafted script of a job in `awaiting_approval`, optionally with inline edits; the job then joins the video-generation queue.

```json
{ "title": "optional edited title", "narration": "optional edited script" }
```

### `POST /jobs/{id}/redraft`
Ask the LLM for a new draft (job returns to `drafting`, then `awaiting_approval`).

```json
{ "feedback": "make it rhyme" }
```

## Archive

- `POST /jobs/{id}/archive` — hide a finished job from the main list.
- `POST /jobs/{id}/unarchive` — bring it back.

## Settings & meta

### `GET /options`
`{"aspect_ratios": [...], "default_aspect_ratio": "...", "art_styles": [...], "default_art_style": "...", "video_lengths": [...], "default_video_length": "...", "default_source": "url"}` — the valid submit-form values and configured defaults.

### `GET /api/settings`
`{"settings": {...}, "meta": {name: {label, help, type}}, "aspect_ratios": [...], "art_styles": [...]}` — current values plus field metadata.

### `PUT /api/settings`
Partial update; values are validated (range/enum) and persisted to `settings.json`, applying to new jobs immediately.

```json
{ "TARGET_VIDEO_SECONDS": 0, "CHANNEL_NAME": "Habibi Kids Channel" }
```

### `GET /api/health`
Pings ComfyUI, vLLM and Fish Speech concurrently: `{"comfyui": {"ok": true, "ms": 10, "error": null}, ...}`.

## Pages (HTML)

`/` (jobs + submit) · `/job/{id}` (job detail) · `/archive` (archived jobs) · `/settings`.
