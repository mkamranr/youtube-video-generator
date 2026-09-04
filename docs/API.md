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
Prompt jobs draft their script immediately in a background thread (never in the GPU queue) and rest at `awaiting_approval` until approved. `channel_id` selects the channel the video is generated for (see `GET /api/channels`; omitted = the first channel) — its name, purpose, audience and video type steer every LLM prompt, and the job snapshots them so later channel edits don't affect it. `aspect_ratio`, `art_style` and `video_length` (`Auto`, `30-45 seconds`, `45-60 seconds`, `1-2 minutes`, `2-3 minutes` — a soft target guiding script and audio length) are optional; unset fields fall back to the channel's defaults, then the global ones. For a `real` or `tech` video-type channel `art_style` is ignored (a built-in realistic live-action look is used). `storyboard` (bool, optional) switches the on-screen story board overlay (fact cards + related images) for this video; omitted = the channel's default (on for `tech` channels). `audio_mode` (`music` = Generate Song, `narration` / `narration+music` = Generate Voiceover) picks the audio pipeline per job — voiceover jobs get a gentle low-volume **instrumental** background bed, never sung lyrics — and `voice_id` optionally selects a reference voice from `GET /api/voices` for the voiceover (`""` = default voice; ignored when `audio_mode` is `music`). Returns a list of job objects.

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

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → [storyboard] → clips → assembling → done | failed` (`storyboard` only for jobs with the overlay enabled: the LLM plans the cards and ComfyUI renders their images).
Prompt-based jobs (`source` = `topic`/`description`) start with `drafting → awaiting_approval` and enter the queue only when approved.

### `GET /jobs/{id}/details`
Large payload for the job page: full `transcript` text, the parsed `plan` (narration/story, style, scenes, per-clip audio `segments`, `music` caption+lyrics, `seo_title`, `seo_description`, `thumbnail_prompt`, `storyboard` — the overlay cards as `[{start_clip, end_clip, kicker, headline, body, image_prompt}]`), and the full `log`.

### `GET /jobs/{id}/files/{name}`
Download any artifact from the job folder (clips, `final.mp4`, `thumbnail.jpg`, `music.mp3`, `narration.wav`, `plan.json`, `transcript.txt`, …).

### `GET /jobs/{id}/download/{kind}`
Publish-ready downloads named after the video title. `kind`: `video` (`<Title>.mp4`), `thumbnail` (`<Title>.jpg`), or `bundle` — a ZIP (`<Title>.zip`) built on request containing the video, thumbnail, audio track(s), and `<Title>.txt` with the SEO title and description. `404` while the underlying artifact doesn't exist yet.

### `DELETE /jobs/{id}`
Delete the job and all its local files. `409` if it is currently running.

### `PUT /jobs/{id}/storyboard`
Switch the story board overlay of a **finished** job: `{"enabled": true|false, "regenerate": false}`. `regenerate: true` also discards the planned cards and rendered images so they are created afresh. The change is applied by the next `POST /jobs/{id}/assemble` (the job gets `needs_assemble: true`); if cards or images still have to be generated, that reassembly goes through the GPU queue (front of the queue) instead of running immediately. `409` unless the job is `done` and idle.

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

- `POST /jobs/{id}/youtube-upload` — uploads a finished job's video (resumable, live `yt_progress` on the job) to the YouTube account connected for the job's channel, with the SEO title/description and the configured privacy (default **private** — an unpublished draft reviewable in YouTube Studio); made-for-kids comes from the channel's own `made_for_kids` toggle (snapshotted onto the job at submit). Then sets the custom thumbnail (best-effort). Job fields: `yt_status` (`uploading`/`uploaded`/`failed`), `yt_progress`, `yt_video_id`, `yt_error`. Returns `409` if the channel is not connected.

## Channels

Channels are managed on the Settings page; each has a name, purpose, audience (`kids`/`family`/`general`), video type (`animated`/`real`/`tech`), per-video defaults, an optional logo and its own YouTube connection. A default channel is created automatically on first start, migrated from the legacy single-channel settings (name, defaults, logo and YouTube token are carried over).

- `GET /api/channels` — `{"channels": [{id, name, purpose, audience, video_type, made_for_kids, defaults, created, has_logo, has_secret, yt_connected}], "secret": bool, "audiences": [...], "video_types": [...], "realistic_video_types": ["real", "tech"], "art_styles": [...], "aspect_ratios": [...], "video_lengths": [...], "audio_modes": [...]}` (`secret` = client_secret.json present).
- `POST /api/channels` / `PUT /api/channels/{id}` — create / update (partial updates allowed). `made_for_kids` (bool) is the channel's COPPA self-declaration for uploads, independent of `audience`. `defaults` holds `art_style`, `audio_mode`, `voice_id`, `aspect_ratio`, `video_length` and `storyboard` (bool; unset = on for `tech` channels) and pre-fills the New Video dialog. Real and tech channels always keep `art_style: ""` — the built-in realistic live-action look is used instead; `tech` channels also steer the LLM towards concrete, technically accurate explainer content.
- `DELETE /api/channels/{id}` — remove a channel and its logo/token (`409` for the last remaining channel).
- `POST /api/channels/{id}/logo` — multipart upload (`file`); stored as PNG (max 5 MB, downscaled to <=1024 px wide) and overlaid on the top-right corner of that channel's thumbnails. `GET /api/channels/{id}/logo` fetches it (404 if none); `DELETE` removes it.
- `GET /api/channels/{id}/youtube/auth` — Google consent URL for connecting this channel's own YouTube account (scope: `youtube.upload` only); the loopback callback `GET /api/youtube/callback` stores the token per channel. `DELETE /api/channels/{id}/youtube/auth` disconnects the channel.
- `POST /api/channels/{id}/secret` — multipart upload (`file`) of the channel's **own** OAuth `client_secret.json` (validated: JSON with an `installed`/`web` client, max 64 KB); it overrides the shared client for this channel (`has_secret: true`). `DELETE /api/channels/{id}/secret` removes it (the channel falls back to the shared client). Secrets are never served back.
- `POST /api/youtube/secret` / `DELETE /api/youtube/secret` — upload/remove the **shared** OAuth client (stored as `client_secret.json` in the project folder; placing the file there manually is equivalent). Resolution order when connecting a channel: its own secret → the shared one.

## Reference voices

Voice samples for Fish Speech voice cloning, managed on the Settings page; pick one per voiceover job via `voice_id` on `POST /jobs`.

- `GET /api/voices` — `{"voices": [{"id", "name", "text", "file", "duration", "created"}]}`.
- `POST /api/voices` — multipart upload: `file` (audio, max 20 MB, 1–120 s; normalized to 44.1 kHz mono wav), `name`, `text` (the exact words spoken in the sample). Returns the new voice record.
- `GET /api/voices/{id}/audio` — the normalized sample (`audio/wav`, 404 if unknown).
- `DELETE /api/voices/{id}` — remove the voice; queued jobs that referenced it fall back to the default voice.

## Settings & meta

### `GET /options`
`{"aspect_ratios": [...], "default_aspect_ratio": "...", "art_styles": [...], "default_art_style": "...", "video_lengths": [...], "default_video_length": "...", "default_source": "url", "default_audio_mode": "music", "realistic_video_types": ["real", "tech"], "channels": [...], "default_channel_id": "..."}` — the valid submit-form values, global fallbacks, and the channel list (same objects as `GET /api/channels`); the UI applies the selected channel's `defaults` client-side.

### `GET /api/settings`
`{"settings": {...}, "meta": {name: {label, help, type}}, "aspect_ratios": [...], "art_styles": [...]}` — current values plus field metadata.

### `PUT /api/settings`
Partial update; values are validated (range/enum) and persisted to `settings.json`, applying to new jobs immediately.

```json
{ "TARGET_VIDEO_SECONDS": 0, "YT_PRIVACY": "unlisted" }
```

### `GET /api/health`
Pings ComfyUI, vLLM and Fish Speech concurrently: `{"comfyui": {"ok": true, "ms": 10, "error": null}, ...}`.

## Pages (HTML)

`/` (jobs + submit) · `/job/{id}` (job detail) · `/archive` (archived jobs) · `/settings`.
