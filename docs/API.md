# HTTP API Reference

Base URL: `http://localhost:8080`. All request/response bodies are JSON. Errors return `{"detail": "<message>"}` with an appropriate status code (`400` invalid input, `401` not signed in, `403` admin required, `404` unknown job or channel — including one that belongs to another user, `409` state conflict, `429` login throttled).

## Authentication

Every endpoint and page requires a signed-in user, except the login/setup pages, `/api/auth/*`, `/api/ping`, `/api/youtube/callback` and the `/css/*`, `/js/*` assets. Sessions are HMAC-signed `vg_session` cookies (HttpOnly, 30 days; invalidated when the user's password changes). Browser navigations without a session are redirected to `/login?next=…` (or `/setup` while no account exists); API calls get `401`.

- `GET /api/auth/status` — `{"setup_required": bool, "authenticated": bool, "user": {id, username, role}|null}`.
- `POST /api/auth/setup` `{username, password}` — first run only: creates the administrator and signs in (`409` once any user exists).
- `POST /api/auth/login` `{username, password}` — sets the session cookie (`401` wrong credentials; after 5 failures per client+username within 10 minutes, `429` for 60 s). `POST /api/auth/logout` clears it. `GET /api/auth/me` returns the current user.
- `PUT /api/auth/password` `{current_password, new_password}` — change your own password (other sessions are signed out, this one is renewed).

### Ownership

Channels and jobs belong to a user. A channel's `owner` is the account that created it (server-assigned; an `owner` field in a request body is ignored) and every job inherits the owner of the channel it was created for. Users with role `user` only see and act on their own channels and jobs — a foreign id answers `404`, exactly like a non-existent one. Admins see and act on everything; list responses carry `owner` (user id) and `owner_name` (username) on every channel and job so the UI can label them, and `GET /options` / `GET /api/channels` include `me: {id, username, role}`.

Admin-only endpoints (`403` for users): `GET/PUT /api/settings`, `GET /api/health`, `POST/DELETE /api/voices`, `POST/DELETE /api/youtube/secret`, `POST /api/style-previews` and everything under `/api/users`. `GET /api/schedule`, `GET /api/voices`, `GET /api/voices/{id}/audio` and `GET /api/style-previews` stay readable by every signed-in user.

Migration: channels and jobs written before ownership existed (and the channel seeded from a legacy single-channel install, which is created before any account) have no owner; at every startup, and once more right after the first admin is created, they are assigned to the oldest admin account. Deleting a user (`DELETE /api/users/{id}`) moves their channels and jobs to the admin performing the deletion.

Scripted use: sign in once and reuse the cookie jar:

```bash
curl -c cookies.txt -H 'Content-Type: application/json' -d '{"username":"admin","password":"…"}' http://localhost:8080/api/auth/login
curl -b cookies.txt http://localhost:8080/jobs
```

### Users (admin only)

Roles: `user` (their own channels and videos, their own password) and `admin` (everything: all users' channels and videos, the settings, reference voices, style previews and user management). There is always at least one admin.

- `GET /api/users` — `{"users": [{id, username, role, created}], "roles": ["admin", "user"]}`.
- `POST /api/users` `{username, password, role}` — `201` with the new user. Usernames: 3–40 characters `[a-z0-9._-]` (lower-cased); passwords: at least 8 characters.
- `PUT /api/users/{id}` `{role?, password?}` — change the role and/or reset the password (`409` when demoting the last admin).
- `DELETE /api/users/{id}` — `409` for the last admin or your own account. The deleted user's channels and jobs move to you; the response carries `reassigned: {channels, jobs}`.

## Jobs

### `POST /jobs`
Submit one job per URL.

```json
{ "urls": ["https://www.youtube.com/watch?v=..."], "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.6, "art_style": "Pixar-like 3D" }
```
Or create a job from a prompt instead of URLs (one job per request):

```json
{ "topic": "How bees make honey" }
{ "description": "A gentle bedtime story about ..." }
{ "transcript": "Twinkle, twinkle, little star ..." }
```
`transcript` is a pasted transcript or song lyrics used as the script itself: the LLM keeps the text and only fixes typos, timestamps, cue/section tags and TTS-unreadable bits (the `Video length` word target is not applied).
Prompt jobs draft their script immediately in a background thread (never in the GPU queue) and rest at `awaiting_approval` until approved. `channel_id` is required and must be a channel you own (admins: any channel; `400` "create a channel first" when missing, `400` "unknown channel" when not yours) — the video belongs to that channel's owner, and the channel's name, purpose, audience and video type steer every LLM prompt; the job snapshots them so later channel edits don't affect it. `aspect_ratio`, `art_style` and `video_length` (`Auto`, `30-45 seconds`, `45-60 seconds`, `1-2 minutes`, `2-3 minutes` — a soft target guiding script and audio length) are optional; unset fields fall back to the channel's defaults, then the global ones. `megapixels` (optional, one of `GET /options`' `megapixels_options`: 0.2–1.0 in 0.1 steps) sets the clip resolution for this video; omitted = the `DEFAULT_MEGAPIXELS` setting (there is no channel level). For a `real` or `tech` video-type channel `art_style` is ignored (a built-in realistic live-action look is used). `storyboard` (bool, optional) switches the on-screen story board overlay (fact cards + related images) for this video; omitted = the channel's default (on for `tech` channels). `audio_mode` (`music` = Generate Song, `narration` / `narration+music` = Generate Voiceover) picks the audio pipeline per job — voiceover jobs get a gentle low-volume **instrumental** background bed, never sung lyrics — and `voice_id` optionally selects a reference voice from `GET /api/voices` for the voiceover (`""` = the built-in voice kept the same for the whole video, `"random"` = the built-in voice with a new random speaker per narration chunk; ignored when `audio_mode` is `music`). Returns a list of job objects.

### `GET /jobs?archived=false`
List your jobs, newest first (admins: every user's jobs, each with `owner` and `owner_name`). `archived=true` lists archived jobs instead. Queued jobs carry a live human-readable `detail` (queue position / what is currently running; or *waiting for the GPU window (…)* when the job is parked by the GPU schedule — `gpu_wait: true`).

### `GET /jobs/{id}`
One job object:

```json
{
  "id": "20260826-060951-2e6720",
  "url": "...", "source": "youtube", "topic": "", "title": "...", "status": "clips",
  "detail": "clip 12/31: sampling 8/20",
  "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4, "art_style": "Pixar-like 3D", "audio_mode": "music",
  "clips_total": 31, "clips_done": 11, "clips": ["clip_01.mp4", "..."],
  "final": "final.mp4", "error": "",
  "stale_clips": [5], "regen_queue": [], "needs_assemble": false, "archived": false,
  "files": ["clip_01.mp4", "music.mp3", "thumbnail.jpg", "..."],
  "log": ["..."], "created": 0, "updated": 0
}
```

`source` is `youtube` for URL jobs, else `topic` / `description` / `transcript`; `topic` holds the submitted prompt text for prompt jobs (empty for URL jobs). `script_approved` is `true` once a prompt job's draft has been approved.

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → [storyboard] → clips → assembling → done | failed` (`storyboard` only for jobs with the overlay enabled: the LLM plans the cards and ComfyUI renders their images).
Prompt-based jobs (`source` = `topic`/`description`/`transcript`) start with `drafting → awaiting_approval` and enter the queue only when approved.

### `GET /jobs/{id}/details`
Large payload for the job page: full `transcript` text, the parsed `plan` (narration/story, style, scenes, per-clip audio `segments`, `music` caption+lyrics, `seo_title`, `seo_description`, `thumbnail_prompt`, `storyboard` — the overlay cards as `[{start_clip, end_clip, kicker, headline, body, image_prompt}]`), and the full `log`.

### `GET /jobs/{id}/files/{name}`
Download any artifact from the job folder (clips, `final.mp4`, `thumbnail.jpg`, `music.mp3`, `narration.wav`, `plan.json`, `transcript.txt`, …).

### `GET /jobs/{id}/download/{kind}`
Publish-ready downloads named after the video title. `kind`: `video` (`<Title>.mp4`), `thumbnail` (`<Title>.jpg`), or `bundle` — a ZIP (`<Title>.zip`) built on request containing the video, thumbnail, audio track(s), and `<Title>.txt` with the SEO title and description. `404` while the underlying artifact doesn't exist yet.

### `DELETE /jobs/{id}`
Delete the job and its whole `output/<id>/` folder (clips, audio, story board cards, final video, bundle). Returns `{"deleted": id, "freed_mb": 52.3, "warning": null}`. `409` while the job is running, assembling, regenerating its thumbnail, drafting, or uploading to YouTube. If a file is still held open (a video being streamed to a browser on Windows), everything else is removed, `warning` says what is left, and the remainder is swept automatically at the next server start.

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

- `GET /api/channels` — `{"me": {id, username, role}, "channels": [{id, owner, owner_name, name, purpose, rules, audience, video_type, made_for_kids, defaults, cast, cast_opening, created, has_logo, has_secret, yt_connected}], "secret": bool, "audiences": [...], "video_types": [...], "realistic_video_types": ["real", "tech"], "art_styles": [...], "aspect_ratios": [...], "video_lengths": [...], "audio_modes": [...], "max_cast": 24, "max_rules_chars": 6000}` (`secret` = client_secret.json present). `rules` (max 6000 chars, optional) is the channel's fixed **style & rules** text: it is added to the system prompt of every script and scene call (not SEO/music) as a fixed reference — typically the non-character sections of a character bible.
- `POST /api/channels` / `PUT /api/channels/{id}` — create (owned by you) / update (partial updates allowed; `owner` in the body is ignored). Users only see and reach their own channels (`404` otherwise); admins reach every channel. Deleting your last channel is allowed — `POST /jobs` then asks you to create one first. `made_for_kids` (bool) is the channel's COPPA self-declaration for uploads, independent of `audience`. `defaults` holds `art_style`, `audio_mode`, `voice_id`, `aspect_ratio`, `video_length` and `storyboard` (bool; unset = on for `tech` channels) and pre-fills the New Video dialog. Real and tech channels always keep `art_style: ""` — the built-in realistic live-action look is used instead; `tech` channels also steer the LLM towards concrete, technically accurate explainer content.
- `DELETE /api/channels/{id}` — remove a channel and its logo/token (`409` for the last remaining channel).
- `POST /api/channels/{id}/logo` — multipart upload (`file`); stored as PNG (max 5 MB, downscaled to <=1024 px wide) and overlaid on the top-right corner of that channel's thumbnails. `GET /api/channels/{id}/logo` fetches it (404 if none); `DELETE` removes it.
- **Cast** (recurring characters): the channel record carries `cast: [{id, name, kind, role: "main"|"supporting", look, personality, has_portrait}]` (max 24; `look` is required and, like `personality`, has no length cap) and `cast_opening` (bool, default true = the first character of the story that has an approved portrait is the first frame of clip 1). Send `cast` in `POST`/`PUT` to replace the list; keep an entry's `id` to keep its portrait, omit it for a new character (the server assigns the id). The cast is frozen onto each job's `channel` snapshot. Not every character is in every video: the script call gets the whole roster and picks the characters the story needs (names written in the topic/description are mandatory; `role` is only a hint — mains are the usual protagonists); the result is stored on the plan as `cast_used: [names]` (recomputed from the final title/narration on approval, plus the writer's reported choice for pasted transcripts) and every later LLM call only carries those characters' looks. `cast.with_prefix` prepends each named character's look to the ComfyUI prompts that mention it, restricted to `cast_used`.
- `POST /api/channels/parse-bible` `{markdown}` — preview a pasted **character bible** as channel data: `{rules, cast: [{name, kind, role, look, personality}], sections: [rule section titles], warnings: [...]}`. Every `### Name` block with `- **Field:** value` bullets (Kind, Role, Precise look, Details, Clothes, Accessories, Distinguishing marks, Personality) becomes a cast entry whose `look` is the labelled fields joined as sentences; every other `## section` becomes the `rules` text, except the document preamble and a "scene/prompt format" section. Nothing is stored — the Settings page merges the result into the channel being edited (matching existing characters by name keeps their ids and portraits) and saves it with `PUT /api/channels/{id}`. `400` for a >200 KB body or when neither characters nor rules are found.
- `GET|POST|DELETE /api/channels/{id}/cast/{char_id}/portrait` — the approved reference portrait (PNG; upload is multipart `file`, max 8 MB, downscaled to <=1280 px wide). `POST …/portrait/generate` `{count: 3}` renders candidates with ComfyUI in the channel's art style in a background thread (409 if already running; honours the GPU schedule window); `GET …/portrait/candidates` → `{generating, waiting, done, total, errors, candidates: [url, …]}`; `GET …/portrait/candidate/{k}` → JPEG; `POST …/portrait/approve` `{k}` promotes candidate k to the portrait and deletes the others.
- `GET /api/channels/{id}/youtube/auth` — Google consent URL for connecting this channel's own YouTube account (scope: `youtube.upload` only); the loopback callback `GET /api/youtube/callback` stores the token per channel. `DELETE /api/channels/{id}/youtube/auth` disconnects the channel.
- `POST /api/channels/{id}/secret` — multipart upload (`file`) of the channel's **own** OAuth `client_secret.json` (validated: JSON with an `installed`/`web` client, max 64 KB); it overrides the shared client for this channel (`has_secret: true`). `DELETE /api/channels/{id}/secret` removes it (the channel falls back to the shared client). Secrets are never served back.
- `POST /api/youtube/secret` / `DELETE /api/youtube/secret` — upload/remove the **shared** OAuth client (stored as `client_secret.json` in the project folder; placing the file there manually is equivalent). Resolution order when connecting a channel: its own secret → the shared one.

## Reference voices

Voice samples for Fish Speech voice cloning: a shared library managed by admins on the Settings page (`POST`/`DELETE` are admin-only); every user can list them and pick one per voiceover job via `voice_id` on `POST /jobs`.

- `GET /api/voices` — `{"voices": [{"id", "name", "text", "file", "duration", "created"}]}`.
- `POST /api/voices` — multipart upload: `file` (audio, max 20 MB, 1–120 s; normalized to 44.1 kHz mono wav), `name`, `text` (the exact words spoken in the sample). Returns the new voice record.
- `GET /api/voices/{id}/audio` — the normalized sample (`audio/wav`, 404 if unknown).
- `DELETE /api/voices/{id}` — remove the voice; queued jobs that referenced it fall back to the default voice.

## Settings & meta

### `GET /options`
`{"aspect_ratios": [...], "default_aspect_ratio": "...", "megapixels_options": [0.2, ..., 1.0], "default_megapixels": 0.4, "art_styles": [...], "default_art_style": "...", "video_lengths": [...], "default_video_length": "...", "default_source": "url", "default_audio_mode": "music", "realistic_video_types": ["real", "tech"], "channels": [...], "default_channel_id": "...", "me": {id, username, role}}` — the valid submit-form values, global fallbacks, and the caller's channels (admins: all channels; same objects as `GET /api/channels`, incl. `owner`/`owner_name`); `default_channel_id` is the caller's own first channel or `""` when they have none. The UI applies the selected channel's `defaults` client-side. `GET /api/settings` and `PUT /api/settings` below are admin-only.

### `GET /api/settings`
`{"settings": {...}, "meta": {name: {label, help, type}}, "aspect_ratios": [...], "megapixels_options": [...], "art_styles": [...]}` — current values plus field metadata.

### `PUT /api/settings`
Partial update; values are validated (range/enum) and persisted to `settings.json`, applying to new jobs immediately.

```json
{ "TARGET_VIDEO_SECONDS": 0, "YT_PRIVACY": "unlisted" }
```

### `GET /api/health`
Pings ComfyUI, vLLM and Fish Speech concurrently: `{"comfyui": {"ok": true, "ms": 10, "error": null}, ...}`.

### `GET /api/schedule`
GPU schedule window state (settings `GPU_SCHEDULE_ENABLED`, `GPU_WINDOW_START`, `GPU_WINDOW_END`; server local time): `{"enabled": true, "start": "22:00", "end": "06:00", "open": false, "now": "14:32", "next_open": "22:00", "next_close": null, "opens_in_min": 448, "closes_in_min": 0}`. While closed, ComfyUI renders are not submitted: queued jobs run their transcript/LLM/TTS steps and then park until the window opens; LLM drafting is never affected.

## Pages (HTML)

`/` (jobs + submit) · `/job/{id}` (job detail) · `/archive` (archived jobs) · `/settings` · `/login` · `/setup` (first run only).
