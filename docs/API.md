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
Prompt jobs draft their script immediately in a background thread (never in the GPU queue) and rest at `awaiting_approval` until approved. `aspect_ratio`, `art_style` and `video_length` (`Auto`, `30-45 seconds`, `45-60 seconds`, `1-2 minutes`, `2-3 minutes` — a soft target guiding script and audio length) are optional and validated against `GET /options`. `audio_mode` (`music` = Generate Song, `narration` / `narration+music` = Generate Voiceover; default: the `AUDIO_MODE` setting) picks the audio pipeline per job, and `voice_id` optionally selects a reference voice from `GET /api/voices` for the voiceover (`""` = default voice; ignored when `audio_mode` is `music`). Returns a list of job objects.

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

### `GET /jobs/{id}/download/{kind}`
Publish-ready downloads named after the video title. `kind`: `video` (`<Title>.mp4`), `thumbnail` (`<Title>.jpg`), or `bundle` — a ZIP (`<Title>.zip`) built on request containing the video, thumbnail, audio track(s), and `<Title>.txt` with the SEO title and description. `404` while the underlying artifact doesn't exist yet.

### `DELETE /jobs/{id}`
Delete the job and all its local files. `409` if it is currently running.

### `POST /jobs/{id}/cancel`
Cancel a queued or running job. Takes effect within seconds; the in-flight ComfyUI render is aborted (only if it belongs to this job). A cancelled *revision* request returns the job to `done`; a cancelled full run becomes `failed: cancelled by user`.

### `POST /jobs/{id}/resume`
Requeue a **failed** job (e.g. after a connection timeout to ComfyUI/vLLM/Fish). Once the service is reachable again, one call re-enters the job into the queue and it continues from the exact step it failed at - every artifact already on disk (plan, music, narration, finished clips) is reused. A job that failed while drafting its script returns to `drafting` and waits for approval again. `409` if the job is not in the `failed` state.

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

### `POST /jobs/{id}/regen-thumbnail`
Generate a brand-new thumbnail for a finished job (fresh LLM concept + render, ~1 minute; runs in a side thread, not the job queue). The channel logo, if uploaded, is overlaid top-right. `409` if the job is running, queued, or archived.

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

## YouTube upload

- `GET /api/youtube/status` — `{secret, connected, privacy, made_for_kids}` (secret = client_secret.json present).
- `GET /api/youtube/auth` — returns the Google consent URL (scope: `youtube.upload` only); the loopback callback `GET /api/youtube/callback` stores the token locally. `DELETE /api/youtube/auth` disconnects.
- `POST /jobs/{id}/youtube-upload` — uploads a finished job's video (resumable, live `yt_progress` on the job) with the SEO title/description, the configured privacy (default **private** — an unpublished draft reviewable in YouTube Studio) and made-for-kids flag, then sets the custom thumbnail (best-effort). Job fields: `yt_status` (`uploading`/`uploaded`/`failed`), `yt_progress`, `yt_video_id`, `yt_error`.

## Channel logo

- `POST /api/channel-logo` — multipart upload (`file`); stored as PNG (max 5 MB, downscaled to <=1024 px wide) and overlaid on the top-right corner of every generated thumbnail.
- `GET /channel-logo` — the current logo (404 if none). `DELETE /api/channel-logo` — remove it.

## Reference voices

Voice samples for Fish Speech voice cloning, managed on the Settings page; pick one per voiceover job via `voice_id` on `POST /jobs`.

- `GET /api/voices` — `{"voices": [{"id", "name", "text", "file", "duration", "created"}]}`.
- `POST /api/voices` — multipart upload: `file` (audio, max 20 MB, 1–120 s; normalized to 44.1 kHz mono wav), `name`, `text` (the exact words spoken in the sample). Returns the new voice record.
- `GET /api/voices/{id}/audio` — the normalized sample (`audio/wav`, 404 if unknown).
- `DELETE /api/voices/{id}` — remove the voice; queued jobs that referenced it fall back to the default voice.

## Settings & meta

### `GET /options`
`{"aspect_ratios": [...], "default_aspect_ratio": "...", "art_styles": [...], "default_art_style": "...", "video_lengths": [...], "default_video_length": "...", "default_source": "url", "default_audio_mode": "music"}` — the valid submit-form values and configured defaults (`default_audio_mode` maps the `AUDIO_MODE` setting onto the dialog's two choices: `music` or `narration+music`).

### `GET /api/settings`
`{"settings": {...}, "meta": {name: {label, help, type}}, "aspect_ratios": [...], "art_styles": [...]}` — current values plus field metadata.

### `PUT /api/settings`
Partial update; values are validated (range/enum) and persisted to `settings.json`, applying to new jobs immediately.

```json
{ "TARGET_VIDEO_SECONDS": 0, "CHANNEL_NAME": "My YouTube Channel" }
```

### `GET /api/health`
Pings ComfyUI, vLLM and Fish Speech concurrently: `{"comfyui": {"ok": true, "ms": 10, "error": null}, ...}`.

## Pages (HTML)

`/` (jobs + submit) · `/job/{id}` (job detail) · `/archive` (archived jobs) · `/settings`.
