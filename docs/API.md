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
{ "urls": ["https://www.youtube.com/watch?v=..."], "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.6, "video_model": "LTX 2.5", "resolution": "2K (1440p)", "art_style": "Pixar-like 3D" }
```
Or create a job from a prompt instead of URLs (one job per request):

```json
{ "topic": "How bees make honey" }
{ "description": "A gentle bedtime story about ..." }
{ "transcript": "Twinkle, twinkle, little star ..." }
```
`lyrics` (optional, **kids channels with `audio_mode` `music` only** — `400` otherwise) is a song sheet of your own: the words are sung **verbatim** (the LLM only writes the music caption and may add `[Verse]`/`[Chorus]` tags; a deterministic guard restores your words if it edits any), and **they set the length** — `video_length` is ignored and stored as `Auto`, the target is the sung-word count divided by `KIDS_SONG_WPS`, and the “song came out short” rewrite never fires because rewriting your words is not allowed. Rejected with `400` when shorter than `MIN_LYRIC_WORDS` (8) or longer than `MAX_LYRIC_WORDS` (250) sung words or `MAX_LYRICS_CHARS` (2000) characters — never silently trimmed — and when sent with more than one URL (one song cannot serve several videos). Independent of the `transcript` source below: `transcript` is where the *script* comes from, `lyrics` is what is *sung*.
`music_upload_id` (optional, **`audio_mode` `music` only** — `400` otherwise) is an id from `POST /api/music-uploads`: your own recording is used as the soundtrack **as-is** instead of a generated song, and **it sets the length** — `video_length` is ignored and stored as `Auto`, and the clip count comes from the track's measured duration, so the video is exactly as long as the music. No caption or lyrics are written and ComfyUI is never asked for a song. The track is transcribed once with Whisper (no reference lyrics — there are none) so the scenes can follow what is actually sung; a track with fewer than `MIN_HEARD_WORDS` (8) words heard counts as instrumental and its scenes follow the script instead. Mutually exclusive with `lyrics` (`400` — the track you uploaded *is* the song) and rejected with `400` when sent with more than one URL, when the id is unknown, expired or someone else's, or when it was already used (an upload serves exactly one video). Unlike `lyrics` it is **not** restricted to kids channels.
`video_mode` (`song` by default, or `story`) picks the **content format**. `story` is a kids-channel feature and is rejected with `400` on any other audience, on a channel whose characters are off or whose cast is empty (the format is built on named speakers), from a `transcript` or `repo_url` source (those words *are* the script, and a story writes its own), and together with `lyrics` or `music_upload_id` (a story is spoken, not sung). It forces `audio_mode` to `narration+music` and `storyboard` off, and it keeps `voice_id`, which becomes the **narrator** voice. A story job always drafts and waits at `awaiting_approval`, **including a `urls` job** - the transcript is fetched in the drafting thread rather than the GPU queue - so a story is never rendered unreviewed. The draft writes `plan["story"] = {premise, beats, lines, issues}`: one beat per clip (`{setting, action, characters}`, what is seen) and one spoken line per beat (`{speaker, text}`, what is heard, derived from the beats and never describing them). Each line is spoken separately in its speaker's voice - the character's own `voice_id` from the channel cast, the channel's `character_voice` default, the job's `voice_id` for the narrator, or one invented once and reused for that character - and laid out on a clip grid sized from the lines, so picture and words are in sync with no transcription step.
`transcript` is a pasted transcript or script used as the script itself: the LLM keeps the text and only fixes typos, timestamps, cue/section tags and TTS-unreadable bits (the `Video length` word target is not applied).
Prompt jobs draft their script immediately in a background thread (never in the GPU queue) and rest at `awaiting_approval` until approved. `channel_id` is required and must be a channel you own (admins: any channel; `400` "create a channel first" when missing, `400` "unknown channel" when not yours) — the video belongs to that channel's owner, and the channel's name, purpose, audience and video type steer every LLM prompt; the job snapshots them so later channel edits don't affect it. `aspect_ratio`, `art_style` and `video_length` (`Auto`, `30-45 seconds`, `45-60 seconds`, `1-2 minutes`, `2-3 minutes`, `3-4 minutes`, `4-5 minutes` — a soft target guiding script and audio length, and the clip length: longer targets render longer clips so the clip count stays around 27) are optional; unset fields fall back to the channel's defaults, then the global ones. `megapixels` (optional, one of `GET /options`' `megapixels_options`: 0.2–1.0 in 0.1 steps) sets the clip resolution for this video; omitted = the `DEFAULT_MEGAPIXELS` setting (there is no channel level). `video_model` (optional, one of `GET /options`' `video_models`) picks the ComfyUI workflow that renders the clips; omitted = the `DEFAULT_VIDEO_MODEL` setting (no channel level), and an unknown name is rejected with `400`. The models quantise clip length differently (a requested 5 s renders as 5.167 s on `MiniMax H3`, 5.042 s on `LTX 2.5`) and those real lengths are frozen into `plan["clip_seconds"]` at plan time, so the model is fixed once the video is planned. Thumbnails, cast portraits and story board images always render on MiniMax H3 regardless. `resolution` (optional, one of `GET /options`' `resolutions`: `Native (clip size)`, `1080p (Full HD)`, `2K (1440p)`, `4K (2160p)`) is the final video's size: during assembly the joined clips are upscaled with ffmpeg to that short side (never downscaled) before the story board cards are drawn; omitted = the `DEFAULT_RESOLUTION` setting (no channel level). For a `real` or `tech` video-type channel `art_style` is ignored (a built-in realistic live-action look is used). `storyboard` (bool, optional) switches the on-screen story board overlay (fact cards + related images) for this video; omitted = the channel's default (on for `tech` channels). `captions` (bool, optional) switches burned-in word-synced subtitles, and `polish` (bool, optional) the hook card, end card, progress bar and card whoosh; both default to the channel's setting, which is on for `tech` channels. `repo_url` (optional, **Tech channels only** - `400` otherwise) is one GitHub repository or Hugging Face model / dataset / space URL: the README or model card plus the project's real facts (stars, downloads, licence, task, library) become the script source, and the job drafts and waits for approval like a topic job. A URL that is not one of those four kinds is rejected with `400` at submit time. `page_scroll` (bool, optional; omitted = on for `repo_url` jobs, always off otherwise) screenshots the live page with a local headless Chrome and scrolls it as a panel covering ~70% of the frame across a window in the middle of the video, leaving the opening and ending clear. If no browser is installed (`GET /api/health` reports `browser.ok: false`) the job logs a warning and finishes from the README alone. `audio_mode` (`music` = Generate Song, `narration` / `narration+music` = Generate Voiceover) picks the audio pipeline per job — voiceover jobs get a gentle low-volume **instrumental** background bed, never sung lyrics — and `voice_id` optionally selects a reference voice from `GET /api/voices` for the voiceover (`""` = the built-in voice kept the same for the whole video, `"random"` = the built-in voice with a new random speaker per narration chunk; ignored when `audio_mode` is `music`). Returns a list of job objects.

### `GET /jobs?archived=false`
List your jobs, newest first (admins: every user's jobs, each with `owner` and `owner_name`). `archived=true` lists archived jobs instead. Queued jobs carry a live human-readable `detail` (queue position / what is currently running; or *waiting for the GPU window (…)* when the job is parked by the GPU schedule — `gpu_wait: true`).

### `GET /jobs/{id}`
One job object:

```json
{
  "id": "20260826-060951-2e6720",
  "url": "...", "source": "youtube", "topic": "", "title": "...", "status": "clips",
  "detail": "clip 12/31: sampling 8/20",
  "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4, "video_model": "MiniMax H3", "resolution": "2K (1440p)", "art_style": "Pixar-like 3D", "audio_mode": "music", "video_mode": "song",
  "music_upload": "", "music_seconds": 0.0,
  "clips_total": 31, "clips_done": 11, "clips": ["clip_01.mp4", "..."],
  "final": "final.mp4", "error": "",
  "stale_clips": [5], "regen_queue": [], "needs_assemble": false, "archived": false,
  "files": ["clip_01.mp4", "music.mp3", "thumbnail.jpg", "..."],
  "log": ["..."], "created": 0, "updated": 0,
  "timings": {
    "planning": { "start": 0, "end": 0, "seconds": 12.4, "running": false, "last_start": 0 },
    "clips":    { "start": 0, "end": 0, "seconds": 1384.0, "running": false, "last_start": 0 }
  }
}
```

`timings` records every pipeline stage the job has run (`drafting`, `transcript`, `planning`, `tts`, `music`, `scenes`, `thumbnail`, `storyboard`, `clips`, `assembling`), keyed by status name, all values epoch seconds: `start` is written once when the stage first did real work (for `clips`: the moment the first clip was submitted for rendering), `end` is the last time the stage stopped for any reason (next stage, GPU-window park, pause, failure — for `clips`, when the last clip finished), `seconds` accumulates only active time across those runs, and `running` is `true` while the stage is in progress. Clip regenerations and reassemblies of a finished job are timed under `regen_clips` / `regen_assembling` so the original render's times stay intact. Jobs created before this field existed have `{}`.

`source` is `youtube` for URL jobs, else `topic` / `description` / `transcript`; `topic` holds the submitted prompt text for prompt jobs (empty for URL jobs). `script_approved` is `true` once a prompt job's draft has been approved.

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → [storyboard] → clips → assembling → done | failed` (`storyboard` only for jobs with the overlay enabled: the LLM plans the cards and ComfyUI renders their images).
Prompt-based jobs (`source` = `topic`/`description`/`transcript`) start with `drafting → awaiting_approval` and enter the queue only when approved.
A job confirmed from a weekly plan starts one step earlier, at `planned`: it exists and is listed, but nothing runs and no LLM is called until `POST /jobs/{id}/setup` supplies the missing parameters. `plan_id`, `planned_day`, `planned_for`, `hook` and `thumb_idea` name the plan row it came from; `setup_done` turns true when it is started. A server restart never resumes a `planned` job, `POST /jobs/{id}/cancel` discards it (to `failed`, restorable with `resume`), and it cannot be archived until it has run.

### `POST /jobs/{id}/setup`
The Start button of a `planned` job (one confirmed from a weekly plan): `{aspect_ratio, megapixels, video_model, resolution, art_style, video_length, audio_mode, video_mode, voice_id, song_voice, lyrics, storyboard}`, every field optional — a blank one keeps the value the job was created with (its channel's default at plan time), never the global default. `lyrics` and `music_upload_id` are the exception: a blank one means *none*, because clearing the box is how you remove them — a planned job never carries either. A blank `video_mode` **does** inherit, and it is resolved against the channel snapshot frozen at plan time, so a planned story job stays a story even if its channel has changed since. Applies them and hands the job to the normal prompt flow (`drafting → awaiting_approval`). `409` when the job is not waiting for setup, `400` for an unknown `voice_id`. A rejected `lyrics` value answers `409` here where `POST /jobs` answers `400`; a `music_upload_id` that is unknown or not yours still answers `400`, and one refused for any other reason answers `409` with the job left untouched at `planned`.

### `GET /jobs/{id}/details`
Large payload for the job page: the user's own song `lyrics` (`""` when they wrote none — the job list carries only a `has_lyrics` flag, so the sheet is not repeated in every polled row), full `transcript` text, the parsed `plan` (narration/story, style, scenes, per-clip audio `segments`, `music` caption+lyrics, `seo_title`, `seo_description` (the publish-ready text, with the viewer resource already appended), `resource` (`{offered, origin, kind, title, markdown, files, in_description, pdf_error}` — the document that ships with the video, absent from videos that have none. `origin` is `promise`, the script offered it, plus a `promise` field quoting the offer, and `in_description` is then true once it has been appended; or `repo-guide`, the project's setup guide that every repo/Hugging Face video ships, plus a `source_url` field, never appended to the description), `thumbnail_prompt`, `thumbnail_texts` and `thumbnail_variants` (`[{file, text, layout}]`, the drawn candidates; the active one is `thumb_variant` on the job), `chapters` (`[{t, title}]`, the timestamp list appended to the uploaded description), `caption_words` (the cached word timings the burned-in captions are drawn from), `storyboard` — the overlay cards as `[{start_clip, end_clip, kicker, headline, body, image_prompt, template}]` (`template` is the card's frozen layout, e.g. `big_stat`, `split`, `full_bleed`; assigned once when the cards are planned so a reassembly never restyles the video)), and the full `log`.

### `GET /jobs/{id}/files/{name}`
Download any artifact from the job folder (clips, `final.mp4`, `thumbnail.jpg`, `music.mp3`, `narration.wav`, `plan.json`, `transcript.txt`, `resource.md`/`.txt`/`.pdf`, …). Owner-checked — there is no public URL, so a resource or guide a video points viewers at is yours to upload or paste.

### `GET /jobs/{id}/download/{kind}`
Publish-ready downloads named after the video title. `kind`: `video` (`<Title>.mp4`), `video-<short>p` (an on-demand upscaled version, e.g. `video-1440p` → `<Title> - 2K 1440p.mp4`), `thumbnail` (`<Title>.jpg`), or `bundle` — a ZIP (`<Title>.zip`) built on request containing the native video, every upscaled version, thumbnail, audio track(s), the viewer resource or setup guide (`.md`/`.txt`/`.pdf`) when the video has one, and `<Title>.txt` with the SEO title and description. `404` while the underlying artifact doesn't exist yet.

### `POST /jobs/{id}/upscale`
Body `{"resolution": "2K (1440p)"}` — any `GET /options` `resolutions` entry except Native. Upscales the finished `final.mp4` with ffmpeg (lanczos, libx264 CRF 18) into `final_<short>p.mp4` next to the original and lists it in the job's `upscales` (`[{label, file, width, height, created}]`); `upscaling` holds the label while it renders (status stays `done`, `detail` shows the progress). Repeating a target replaces that file; different targets accumulate (1080p, 2K and 4K can coexist). Runs in a side thread, not the GPU queue. `409` when the job is not finished, archived, busy, already upscaling, or when the final video is already at or above the target. Reassembling the final video deletes the upscaled versions (they were made from the old file).

### `DELETE /jobs/{id}`
Delete the job and its whole `output/<id>/` folder (clips, audio, story board cards, final video, bundle). Returns `{"deleted": id, "freed_mb": 52.3, "warning": null}`. `409` while the job is running, assembling, regenerating its thumbnail, drafting, or uploading to YouTube. If a file is still held open (a video being streamed to a browser on Windows), everything else is removed, `warning` says what is left, and the remainder is swept automatically at the next server start.

### `PUT /jobs/{id}/thumbnail`
Body `{"variant": 2}` and/or `{"texts": ["40% cheaper", "Nobody tells you this"]}` on a **finished** job. `variant` picks which thumbnail ships: `0` is the clean render with no text (a real choice, not just the absence of one), `1..n` the text variants in `plan.thumbnail_variants`. `texts` replaces the overlay lines and redraws the variants. Pillow only — no GPU, no queue, no thread — so it returns with the job already updated; the chosen file is copied over `thumbnail.jpg`, which is what `GET /jobs/{id}/download/thumbnail`, the publish bundle and the YouTube upload all use. `409` unless the job is `done` and idle.

### `PUT /jobs/{id}/storyboard`
Switch the story board overlay of a **finished** job: `{"enabled": true|false, "regenerate": false}`. `regenerate: true` also discards the planned cards and rendered images so they are created afresh — which is also how a changed **card layout set** is applied to an existing job, since each card's layout is frozen when it is planned. The change is applied by the next `POST /jobs/{id}/assemble` (the job gets `needs_assemble: true`); if cards or images still have to be generated, that reassembly goes through the GPU queue (front of the queue) instead of running immediately. `409` unless the job is `done` and idle.

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

### `POST /jobs/{id}/regen-resource`
Re-decide and rewrite the document that ships with a finished video (LLM + file writes only, ~10 seconds; runs in a side thread, not the job queue). Re-reads the final narration: if it promises the viewer an artifact, that artifact is written; otherwise, for a repo/Hugging Face video, the project's setup guide is written from the cached `transcript.txt`. Rewrites `resource.md`/`.txt`/`.pdf` and the `resource` record in `plan.json`, restoring the writer's own `seo_description` first so the description block is never appended twice. This is the way back for a video made before the feature existed, or one whose script was edited afterwards. `409` if the job is running, queued, or archived.

### `POST /jobs/{id}/assemble`
Re-concatenate all clips and remix the audio into a new `final.mp4`. Runs **immediately in a background thread** (pure ffmpeg, no GPU) — it never waits behind rendering jobs. Clears `needs_assemble`.

### `POST /jobs/{id}/approve`
Approve the drafted script of a job in `awaiting_approval`, optionally with inline edits; the job then joins the video-generation queue. For a **story video** send `lines` instead of `narration` — `[{speaker, text}]`, exactly one per scene, as reviewed on the job page. `narration` is ignored there: the spoken lines *are* the narration (the server rebuilds it from them), and a stale paragraph would contradict the lines the clips are timed to. A different number of lines, a non-object entry, or a line over `MAX_STORY_LINE_CHARS` (400) characters is rejected with `409` and the job stays at `awaiting_approval`; an unknown `speaker` becomes the narrator rather than an error. Omitting `lines` keeps the draft exactly as written.

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

## Weekly plans (Planner)

A plan is seven video ideas for one channel, written by the LLM and reviewed before anything is generated. Plans are kept per channel (a history of the weeks already covered) and are owner-scoped like channels and jobs: `404` for someone else's. Stored in `plans/plans.json` (`VIDEOGEN_PLANS_DIR`).

A plan row is `{day, date, actors: [cast names], title, description, thumbnail_idea, hook, job_id}`; `job_id` is filled once the row has been confirmed into a job. The prompts use the **spoken** cast sheet, so a description names the characters without describing how they look.

- `GET /api/plans?channel_id=` — `{"plans": [{id, channel_id, owner, owner_name, created, updated, week_start, notes, days: [...]}], "days": 7}`, newest first (admins see everyone's).
- `POST /api/plans` `{channel_id, week_start, notes, video_mode}` — writes the week with the LLM and stores it. `video_mode` (`song` by default, or `story`) is the format **every** video confirmed from this week is created in; it is stored on the plan, and it is also told to the plan writer, which then keeps each story in one or two connected places (a story is rendered as clips that each continue from the previous one). It is validated exactly as `POST /jobs` validates it — `400` on a non-kids channel or one whose characters are off. `week_start` is the ISO date of day 1 (today when missing); `notes` optionally steers the week. Titles from the channel's earlier plans are sent along so a new week does not repeat them. `502` when the LLM is unreachable or returns nothing usable.
- `POST /api/plans/{id}/regenerate` `{days: [1, 3], feedback}` — rewrites those days in place, each told what the other days are so the new idea stays different. Rows that already became a job are skipped; empty `days` rewrites the whole week.
- `PUT /api/plans/{id}/days/{day}` `{actors, title, description, thumbnail_idea, hook}` — saves a hand-edited row (text clamped; `actors` kept to names in the channel's cast when it has one, and always empty for a channel with `characters_enabled: false`). The day number, its date and an existing `job_id` are preserved.
- `POST /api/plans/{id}/jobs` `{days: [1, 2], video_mode}` — confirms rows: one **paused** job per day (status `planned`, see `POST /jobs/{id}/setup`), built from the row's description, with its title, hook and thumbnail idea carried along. `video_mode` is optional: blank uses the format stored on the plan, and a value both overrides it and is saved back onto the plan, so the rest of the week follows. It is resolved **once before any job is created**, so a plan whose channel is no longer eligible fails cleanly instead of half-confirming a week. Returns `{"plan": ..., "jobs": [...]}`. `400` when the plan's channel no longer exists, or when the format does not suit it.
- `DELETE /api/plans/{id}` — forgets the plan. Jobs already created from it stay.

## Channels

Channels are managed on the Settings page; each has a name, purpose, audience (`kids`/`family`/`general`), video type (`animated`/`real`/`tech`), per-video defaults, an optional logo and its own YouTube connection. A default channel is created automatically on first start, migrated from the legacy single-channel settings (name, defaults, logo and YouTube token are carried over).

- `GET /api/channels` — `{"me": {id, username, role}, "channels": [{id, owner, owner_name, name, purpose, rules, audience, video_type, made_for_kids, defaults, cast, cast_opening, characters_enabled, created, has_logo, has_secret, yt_connected}], "secret": bool, "audiences": [...], "video_types": [...], "realistic_video_types": ["real", "tech"], "art_styles": [...], "aspect_ratios": [...], "video_lengths": [...], "audio_modes": [...], "max_cast": 24, "max_rules_chars": 6000}` (`secret` = client_secret.json present). `rules` (max 6000 chars, optional) is the channel's fixed **style & rules** text: it is added to the system prompt of every script and scene call (not SEO/music) as a fixed reference — typically the non-character sections of a character bible.
- `POST /api/channels` / `PUT /api/channels/{id}` — create (owned by you) / update (partial updates allowed; `owner` in the body is ignored). Users only see and reach their own channels (`404` otherwise); admins reach every channel. Deleting your last channel is allowed — `POST /jobs` then asks you to create one first. `made_for_kids` (bool) is the channel's COPPA self-declaration for uploads, independent of `audience`. `characters_enabled` (bool) decides the content shape: on = the channel's recurring cast drives the scripts, scenes, thumbnails and weekly plans; off = informative explainer videos with no protagonist and no named people (only generic unnamed figures on screen, and the narration addresses the viewer as "you"). It is independent of `audience`, so a grown-up channel can have characters too. New channels and migrated records derive it from whether a cast is present; turning it off keeps the cast and its portraits and only stops the prompts from using them. A job freezes the flag in its `channel` snapshot, so a video already queued finishes in the mode it started in. `defaults` holds `art_style`, `audio_mode`, `voice_id`, `aspect_ratio`, `video_length` and `storyboard` (bool; unset = on for `tech` channels) and pre-fills the New Video dialog. Real and tech channels always keep `art_style: ""` — the built-in realistic live-action look is used instead; `tech` channels also steer the LLM towards concrete, technically accurate explainer content.
- `DELETE /api/channels/{id}` — remove a channel and its logo/token (`409` for the last remaining channel).
- `POST /api/channels/{id}/logo` — multipart upload (`file`); stored as PNG (max 5 MB, downscaled to <=1024 px wide) and overlaid on the top-right corner of that channel's thumbnails. `GET /api/channels/{id}/logo` fetches it (404 if none); `DELETE` removes it.
- **Cast** (recurring characters): the channel record carries `cast: [{id, name, kind, role: "main"|"supporting", look, personality, has_portrait}]` (max 24; `look` is required and, like `personality`, has no length cap) and `cast_opening` (bool, default true = the first character of the story that has an approved portrait is the first frame of clip 1). The whole cast is ignored by every prompt while `characters_enabled` is false, though it stays stored and editable (portraits included). Send `cast` in `POST`/`PUT` to replace the list; keep an entry's `id` to keep its portrait, omit it for a new character (the server assigns the id). The cast is frozen onto each job's `channel` snapshot. Not every character is in every video: the script call gets the whole roster and picks the characters the story needs (names written in the topic/description are mandatory; `role` is only a hint — mains are the usual protagonists); the result is stored on the plan as `cast_used: [names]` (recomputed from the final title/narration on approval, plus the writer's reported choice for pasted transcripts) and every later LLM call only carries those characters' looks. `cast.with_prefix` prepends each named character's look to the ComfyUI prompts that mention it, restricted to `cast_used`.
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

## Uploaded music

Your own track for **one** video, used instead of a generated song (`audio_mode` `music`). Per user and one-shot: unlike the shared reference-voice library this is not an admin resource, and the upload is consumed by the job that uses it.

- `POST /api/music-uploads` — multipart upload (`file`, max 60 MB, 10 seconds to 6 minutes). Decoded and normalized with ffmpeg to 44.1 kHz stereo MP3 with cover art and tags stripped; anything ffmpeg cannot decode is `400` "not a valid audio file". A track outside the length bounds is **rejected, never trimmed** — a clipped track is not the song you picked. Returns `{"id", "owner", "name", "seconds", "bytes", "created"}`; `seconds` is what the finished video will be. Pass `id` as `music_upload_id` on `POST /jobs` or `POST /jobs/{id}/setup`.
- `DELETE /api/music-uploads/{id}` — drop a staged track you decided not to use (`400` if unknown or not yours).

Staged files live in `output/_uploads/<id>.mp3` with an `<id>.json` sidecar, are deleted the moment a job is created from them (so the same id cannot serve two videos), are capped at 5 pending per user, and are swept after 24 hours. The track itself is then served like any other job artifact: `GET /jobs/{id}/files/music.mp3`.

## Settings & meta

### `GET /options`
`{"aspect_ratios": [...], "default_aspect_ratio": "...", "megapixels_options": [0.2, ..., 1.0], "default_megapixels": 0.4, "video_models": ["MiniMax H3", "LTX 2.5"], "default_video_model": "MiniMax H3", "resolutions": ["Native (clip size)", "1080p (Full HD)", "2K (1440p)", "4K (2160p)"], "default_resolution": "Native (clip size)", "art_styles": [...], "default_art_style": "...", "video_lengths": [...], "default_video_length": "...", "default_source": "url", "default_audio_mode": "music", "realistic_video_types": ["real", "tech"], "channels": [...], "default_channel_id": "...", "me": {id, username, role}}` — the valid submit-form values, global fallbacks, and the caller's channels (admins: all channels; same objects as `GET /api/channels`, incl. `owner`/`owner_name`); `default_channel_id` is the caller's own first channel or `""` when they have none. The UI applies the selected channel's `defaults` client-side. `GET /api/settings` and `PUT /api/settings` below are admin-only.

### `GET /api/settings`
`{"settings": {...}, "meta": {name: {label, help, type}}, "aspect_ratios": [...], "megapixels_options": [...], "resolutions": [...], "video_models": [...], "art_styles": [...], "storyboard_template_sets": [...], "video_lengths": [...], "sources": [...]}` — current values plus field metadata.

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

`/` (jobs + submit) · `/job/{id}` (job detail) · `/planner` (weekly plans) · `/archive` (archived jobs) · `/settings` · `/login` · `/setup` (first run only).
