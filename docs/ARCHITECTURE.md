# Architecture

## Module map

```
app.py                 FastAPI app: HTTP API, page routes, no-cache middleware, settings/health endpoints
pipeline.py            Job model, serial worker (JobManager), the generation pipeline and the
                       maintenance pipeline (clip regeneration / reassembly), pause & resume logic
config.py              Defaults + runtime-editable settings (EDITABLE_SETTINGS, persisted to settings.json),
                       art-style presets, workflow node ids
services/
  transcript.py        YouTube captions via youtube-transcript-api with yt-dlp fallback; cue-tag cleaning;
                       caption coverage seconds for quality gating
  repo.py              GitHub repos and Hugging Face models/datasets/spaces: URL parsing, the two REST
                       APIs, a stdlib markdown stripper and the fact block that becomes the script source
  browser.py           Locates a local Chrome/Chromium (like ffbin.py locates ffmpeg) and drives it over
                       the DevTools protocol to screenshot a page at a phone-width viewport, plus the
                       offset where its README starts
  llm.py               All vLLM prompting: story plan, per-clip scenes, music caption/lyrics, SEO metadata,
                       two content shapes - cast-driven stories, or informative explainers for a channel with
                       characters off (no protagonist, second person, generic unnamed people only);
                       thumbnail prompt; the audience safety rules, channel purpose and branding live in the system prompt, built per job from its channel snapshot
  tts.py               Fish Speech synthesis (chunked <=TTS_CHUNK_CHARS/request, GPU-OOM aware,
                       optional reference-voice cloning), audio duration
  cast.py              Channel cast: enabled()/active() gate the roster the PROMPTS see (a channel can
                       switch characters off and keep its cast; all_characters() stays raw for the editor and
                       the portrait renderer); character sheet for the LLM system prompt (whole roster while the script is
                       written, only this video's `cast_used` afterwards), detection of the characters a text
                       names, deterministic look prefix for ComfyUI prompts that name a character, portrait
                       candidate rendering/approval, and the "opening character" whose portrait seeds clip 1
  resource.py          The document that ships with a video - the thing the script promised, or the
                       setup guide every repo/Hugging Face video gets: markdown -> plain text, the
                       branded PDF (channel logo, headings, code blocks, footer) and the once-only
                       fold into the YouTube description, which a guide never takes
  bible.py             Character-bible import: parses a markdown bible into cast entries + the channel's
                       style & rules text (pure text processing, previewed via POST /api/channels/parse-bible)
  ownership.py         Who owns channels and jobs: startup migration of unowned records to the oldest
                       admin, reassignment when an account is deleted
  voices.py            Reference-voice library (reference_audio/): normalized samples + transcripts
                       for Fish Speech voice cloning, JSON index, add/list/delete
  story.py             story videos: the clip grid built around the spoken lines, their timing in
                       lyrics.timeline's shape (no whisper - we wrote the words and built the
                       track, so every timestamp is known), and the deterministic check that
                       catches a voiceover line describing the picture instead of telling the story
  lyrics.py            faster-whisper word timestamps, aligned to the written lyrics (difflib), bucketed against
                       the real clip boundaries -> per-clip lyric timelines (which line is sung when) + singing pace
  comfy.py             ComfyUI client: queue/wait (websocket + polling), cancel-aware waiting, output
                       download with node auto-detection, targeted interrupt, server-side scrubbing
  video.py             ffmpeg helpers: last-frame extraction, thumbnail crop (1280x720), clip concat,
                       lanczos upscale to the job's output resolution (1080p/2K/4K, before the cards
                       are drawn so they render at the final size), timed RGBA card overlays, N-way audio mix (narration / ambience / music) with
                       side-chain ducking, music looping/fade-out and clip-cut ambience dips
  storyboard.py        Story board overlay: LLM card planning (via llm.py), related-image rendering
                       through ComfyUI, Pillow card drawing (landscape/portrait/square layouts), schedule
                       aligned to the real clip cut points
  ffbin.py             Locates modern ffmpeg/ffprobe (static-ffmpeg bundle, env override, PATH fallback)
  users.py             Login accounts (users/users.json): username, role, scrypt password hash;
                       last-admin guards
  auth.py              Signed session cookies (HMAC-SHA256, stdlib only), current_user /
                       require_user / require_admin dependencies, login throttling
  planner.py           Weekly content plans (plans/plans.json): seven dated video ideas per
                       channel, normalization and the store; rows become paused jobs
static/                Web UI: index (submit + job list), job (detail/revision), planner (weekly
                       plans), archive, settings, login, setup; shared css/app.css and
                       js/common.js (nav + user slot, status pills, mode-aware stepper,
                       401 -> login redirect)
workflows/             ComfyUI API-format workflow JSONs (text-to-video, first-frame video, music)
start_server.bat       Watchdog: relaunches uvicorn whenever it exits; logs to server.log
```

## Authentication

A single `auth_gate` HTTP middleware in `app.py` runs outermost: it resolves the session cookie once per request (`auth.current_user`, cached on `request.state`) and lets through only signed-in users, the public paths (`/login`, `/setup`, `/api/auth/*`, `/api/ping`, `/api/youtube/callback`, `/css/*`, `/js/*`) — the `StaticFiles` mount and job file downloads are therefore covered too. While no account exists every protected request is sent to the one-time `/setup` page, which creates the first admin; afterwards `/setup` redirects to `/login`. Sessions are stateless: the cookie payload `{uid, exp, ps}` is HMAC-signed with a key from `VIDEOGEN_SECRET_KEY` or `users/session_secret` (generated once); `ps` fingerprints the password hash so a password change invalidates old sessions. Every channel and job handler takes `Depends(auth.require_user)` and resolves its target through `_owned_channel` / `_owned_job` in `app.py`, which return `404` unless `auth.can_access(user, owner)` (admin, or the record's `owner` equals the user id); admin-only routes (settings, health, voice and shared-OAuth mutations, style-preview generation, users) use `Depends(auth.require_admin)`. Channels store `owner` (set by `channels.add_channel(data, owner)`, never taken from the request); jobs carry `Job.owner`, inherited from the channel at submit time. `services/ownership.py` repairs ownership: `migrate_unowned` hands records without an owner to the oldest admin (run at startup and after `/api/auth/setup`, idempotent), `reassign` moves a deleted user's channels and jobs to the acting admin. `GET /jobs`, `/options` and `/api/channels` are filtered to the caller for role `user`; admins get everything plus `owner`/`owner_name` labels and `me`.

## Job lifecycle

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → [storyboard] → clips → assembling → done | failed`.

- One background worker thread processes the queue serially (single GPU). Queued jobs' `detail` is computed live at request time (position, what is running).
- **Prompt jobs** (`job.source` = `topic` / `description` / `transcript`, `job.topic` = the submitted text) skip the transcript fetch. `submit_prompt` creates the job at `drafting` and `_draft_script` runs `llm.write_script_from_prompt` in its own thread — never in the GPU queue — then parks the job at `awaiting_approval`; `POST /jobs/{id}/approve` (optionally with an edited title/script) enqueues it, and *Request new draft* re-runs the draft with the user's feedback and the previous narration. The three sources differ only in the drafting prompt: `topic` invents a script of about the target word count, `description` expands the user's text faithfully, and `transcript` keeps the text as the narration verbatim — only typos, timestamps, cue/section tags and speaker labels are fixed, and no word target is applied. For `transcript` jobs the pasted text is also written to `transcript.txt` at submit time so the job page's Transcript card can show it.
- **Planner jobs** (the Planner page): `llm.write_week_plan` drafts seven ideas for a channel, the user reviews and rewrites rows, and a confirmed row calls `submit_prompt(..., start=False)` — the job is created at `planned` with the channel's defaults already resolved onto it, but nothing runs and no LLM is called. `_load_existing` keeps a `planned` job in memory instead of re-queueing it, so a restart never starts a video nobody set up. `POST /jobs/{id}/setup` (`setup_planned`) re-resolves the parameters against the channel snapshot frozen at plan time, with the job's own values as the fallback layer, then hands over to the normal `_start_draft`. The row's `hook` reaches `write_script_from_prompt` as the opening line and its `thumbnail_idea` is written into `plan.json` for `write_thumbnail`.
- **music mode** (default): the song is generated *first*; its measured duration decides the clip lengths (`config.clip_durations`: full clips at the length the video model really renders — the H3 workflow quantises 5 s to 124 frames = 5.167 s — plus one shorter trailing clip, so the video ends with the song; frozen as `plan["clip_seconds"]`). The full-clip length comes from the job's target length (`config.video_length_clip_seconds`: 5 s up to `1-2 minutes`, 6/8/10 s for the longer presets, `CLIP_SECONDS` for Auto), so a 5-minute video is ~27 clips rather than ~59. faster-whisper (`WHISPER_MODEL`, primed with the written lyrics) gives word timestamps, `lyrics.align_to_lyrics` snaps them onto the lyric text (so mis-hearings take the written words) and `lyrics.timeline` buckets them against the real clip boundaries into per-clip lines with times relative to the clip start (`plan["timed_segments"]`). Scenes are then written **one LLM call per clip** from exactly those words and that timeline, so the storyboard beats land where the line is sung (one-call-per-scene prevents alignment drift on long videos). Kids channels get a nursery-rhyme caption (`KIDS_SONG_BPM` forced into the caption, `KIDS_SONG_WPS` sizing the lyrics) sung by the channel's `song_voice` (child by default); the measured singing pace and duration are logged, and the song is rendered **at most once more** (`music_try1.mp3` / `music_try2.mp3`, the kept one copied to `music.mp3`) for one of two reasons: `"fast"` — the pace exceeded the budget by 40 % (`SONG_PACE_TOLERANCE`), rewritten with fewer words; or `"short"` — the recording ended below the target range's low bound (`SONG_SHORT_TOLERANCE`), rewritten with more verses at the same pace. A song *longer* than the range is deliberately never retried: the render is already hard-cut at `max_seconds` and the clip count simply follows the measured track. A job carrying the user's own lyrics (`job.lyrics`, kids channels) has **no range at all**: its target is the sung-word count over `KIDS_SONG_WPS` and its ceiling `2x+20`, both frozen into `plan["lyric_length"]` on the first pass because `KIDS_SONG_WPS` is editable at runtime. `"short"` therefore can never fire (rewriting the user's words is not allowed), `"fast"` re-renders the *same* words with a slower caption, and a try 2 that ran into the hard cut can never win; the clip length then comes from `config.clip_seconds_for_duration(measured)` instead of the preset ladder. The reason is frozen in `plan["music_tries"][1]["reason"]` when try 2 is created, so a resume (or a settings edit between runs) can never re-decide the branch and strand a finished render; the winner is the slower try for `"fast"`, and for `"short"` the try closest to the range that did not also break the pace budget. The final mix is the song alone.
  **Uploaded track** (`job.music_upload`, `services/music_upload.py`): the user can bring their own recording instead. It is validated and normalized to MP3 at `POST /api/music-uploads`, staged in `output/_uploads/`, and copied to the job's `music.mp3` when the job is *created* — so by the time the worker reaches the music stage the track is simply there. The whole caption/lyrics/render/retry path above is skipped (`_gpu_gate` is never reached in this stage), `video_length` is forced to `Auto`, and the measured track decides the clip count exactly as a generated song does, with the rung from `config.clip_seconds_for_duration(measured)` since there is no preset. Whisper still runs once, with **no** reference lyrics (`lyrics.align_to_lyrics(heard, "")` returns its raw words), and the result is cached in `plan["music_words"]` so a resume never pays for it twice; `plan["music_final"]["reason"]` is `"upload"` and `plan["lyric_align"]["source"]` is `"upload"`. A track with fewer than `MIN_HEARD_WORDS` words heard is treated as instrumental: the per-clip segments are dropped entirely (`segments=None`) so `llm.write_scenes` splits the *story* across the clips instead — all-empty segments would hand every clip the "no words are heard" prompt and the story would never reach a single scene. Mutually exclusive with `job.lyrics`: lyrics are an instruction to the singer model, and an uploaded recording has no singer.
- **narration modes**: Fish TTS narration length decides the clip count (a draft that came out below 85 % of the job's word target is extended once by the LLM first — before any TTS and before the approval step, so nothing the user approved is ever rewritten; pasted transcripts are skipped); scenes come from evenly split narration segments; final mix = narration (full) + an optional gentle instrumental music bed (the music model is sent `[Instrumental]` as lyrics, so nothing is ever sung over the narration). The clips' own generated sound is dropped in narration+music mode unless `CLIP_AUDIO_WITH_MUSIC` is on (three independent layers — voice, music, per-clip effects that change at every cut — sounded like a mess); in pure narration mode it stays, dipped to silence around each clip cut. Beds are side-chain compressed under the narration; a music track the model ended early is looped with a cross-fade, and always fades out at the end. The result is trimmed to the video length and soft-limited. The audio mode and an optional reference voice (`voice_id`) are chosen per job in the New Video dialog; a job whose reference voice was deleted logs a note and falls back to the default voice.
- **story mode** (`job.video_mode` = `story`, kids channels with a cast): the inverse of music mode. `_draft_story` runs two LLM calls off the GPU queue - `llm.write_story_beats` writes exactly `story.beat_count(target, clip_seconds)` beats (`{setting, action, characters}`, the VIDEO content), then `llm.write_story_lines` derives one spoken line per beat FROM those beats. Both are written with the cast sheet in look-free mode (`llm._system(brief=True)` / `spoken=True`), so the writer that produces the voiceover has never seen the appearance it must not recite; `story.describes_look` then catches what slips through and `llm.rewrite_story_line` fixes that one line - all before the user reviews it, so nothing approved is ever rewritten. The job parks at `awaiting_approval` whatever its source (a URL story job drafts in its own thread and fetches its transcript there), and `POST /jobs/{id}/approve` carries the edited `lines`. At the audio stage `_story_audio` speaks each line on its own in its speaker's voice - the character's own reference voice, the channel's `character_voice` default, the job's voice for the narrator, or one invented once and pinned in `plan["story"]["voice_pins"]` - then `story.clip_durations` sizes each clip (the preset as a floor, longer when the line needs the room) and `tts.concat_timed` lays the lines onto that grid. The result is exact A/V sync by construction: no whisper pass and no alignment. `plan["story"]["clip_seconds"]` is frozen, so a resume never re-cuts the video. The mix is the normal `narration+music` one.
- Clips: clip 1 uses the T2V workflow; clips 2..N use the first-frame workflow seeded with the previous clip's last frame (ffmpeg extraction → ComfyUI image upload), giving visual continuity.
- **Cast** (`services/cast.py`, per-channel `cast` list and `rules` text frozen onto the job's channel snapshot): the channel's style & rules text and the character sheet are appended to the LLM system prompt (`llm._system(channel, used, rules)`). While the script is written the sheet lists the whole roster and asks the writer to pick only the characters the story needs (names found in the user's topic/description are mandatory, `llm._cast_selection_rule`; the writer also reports its choice as `characters`). `pipeline._set_cast_used` then stores `plan["cast_used"]` — every roster name found in the final title/narration in first-mention order (case-sensitive, so "a penny" is not Penny), plus the reported choice when the text names nobody or is a verbatim transcript — and recomputes it when the user approves an edited script. Every later call — scenes, both review passes, thumbnail, story board, SEO, music — gets `used=cast_used`, so only those characters' looks are resent (SEO and music skip the rules text). At render time `cast.with_prefix(..., only=cast_used)` prepends each named character's canonical look to the clip, thumbnail and story board image prompts, independent of how the LLM phrased the scene. When `cast_opening` is on and the first character of `cast_used` with an approved portrait exists (`channels/<id>_cast_<char>.png`), `_run` fits it to the job's aspect (`cast_opening.jpg`, resumable), `write_scenes` is told scene 1 starts from that image, and `_generate_clip` renders clip 1 through the first-frame workflow seeded with it — the video opens on the real character and the chain of last frames carries it forward. `services/bible.py` turns a pasted markdown character bible into cast entries and the rules text (Settings page → `POST /api/channels/parse-bible`, merged client-side, saved with the channel).
- **Story board** (optional, `job.storyboard`, default on for `tech` channels): before the clips, the LLM plans the cards (`plan.json["storyboard"]`: clip range, kicker, headline, body, optional image prompt) and ComfyUI renders each requested related image as a 5-frame clip whose first frame is cropped to `sbimg_NN.jpg` (capped by `STORYBOARD_MAX_IMAGES`); the picture's shape follows the video orientation (`storyboard.image_spec`: square for portrait videos, 4:3 for square, 16:9 for landscape), and the card layout uses the same ratio for its picture slot, so older 16:9 images are simply cover-cropped. Both are resumable artifacts; failures degrade to text-only cards or to no overlay, never to a failed job.
- Assembly: ffmpeg concat (stream-copy with re-encode fallback); with the story board on, the cards are drawn with Pillow at the real frame size (`card_NN.png`) and composited with alpha fades over `clips_joined.mp4` → `clips_cards.mp4`, each card timed to the measured clip cut points (0.35 s inset); then the filtergraph audio mix; video length ≈ audio length (count is the ceiling of the audio duration).

## Resume & idempotency

Every pipeline step checks for its artifact before doing work: `transcript.txt`, `plan.json` (story, scenes, music plan, SEO, thumbnail prompt), `music.mp3`, `narration.wav`, `thumbnail.jpg`, each `clip_NN.mp4`. On server start, non-terminal jobs are re-queued and re-run `_run()`, which skips everything already done — a crash or restart costs at most the in-flight render. Job state persists in `output/<id>/job.json` after every mutation. `job.timings` records per-stage `start` / `end` / active `seconds` (hooked into every status transition in `Job.set`), so a stage parked by the GPU window, paused for a priority regen or cut by a restart keeps its first start and accumulates only the time it actually ran; the `clips` stage is entered only once the GPU gate passes, so its `start` is the first real render.

## Cancel, pause, priority

- **Cancel** sets a flag checked at every step boundary, inside every progress callback, and continuously inside the ComfyUI wait loop (`should_stop`); the in-flight ComfyUI prompt is interrupted only after verifying it belongs to the job (filename-prefix match). Cancelled full runs become `failed`; cancelled revision runs return to `done`.
- **Priority regeneration**: a regen request is inserted at the *front* of the queue and the running job gets `pause_requested`. `JobPaused` deliberately subclasses `BaseException` so no broad `except Exception` handler can swallow it; the paused job stops at its next checkpoint (clip boundary — no GPU work wasted), is re-inserted right after the regen job, and resumes via the normal resume path.
- **GPU schedule window** (`services/schedule.py`, settings `GPU_SCHEDULE_ENABLED` / `GPU_WINDOW_START` / `GPU_WINDOW_END`, server local time, overnight windows supported): `_gpu_gate(job)` sits immediately before every ComfyUI submission of the worker path (clips, both music renders, thumbnail, story board images) and raises `JobWaitingWindow` — another `BaseException`, for the same reason as `JobPaused` — when the window is closed. `run_job` marks the job `gpu_wait` and `queued` with a "waiting for the GPU window (opens at …)" detail; the worker loop re-appends it and, while the window is closed, skips `gpu_wait` jobs when picking the next one, so other queued jobs still get their transcript/LLM/TTS phases done and everything is ready to render the moment the window opens. An in-flight render is never interrupted by the schedule (the gate is pre-submit). The side threads that bypass the queue (thumbnail regeneration, art-style previews) block in `schedule.wait_until_open` instead.
- **Reassembly** (`POST /assemble`) is pure-local ffmpeg, so it runs immediately in a dedicated thread rather than the GPU queue — except when the story board was just enabled/regenerated and its cards or images still have to be produced: that reassembly is inserted at the front of the GPU queue instead.

## Revision (maintenance) pipeline

`run_job` dispatches to `_run_maintenance` when `regen_queue` or `assemble_requested` is set on a finished job. It re-renders each requested clip with the current (possibly edited) scene prompt via the shared `_generate_clip` helper (same retry/cancel/cleanup behavior as the main loop), clears the clip's `stale_clips` entry, sets `needs_assemble`, and optionally reassembles. Errors or cancellation during maintenance always return the job to `done` with an explanatory detail — a finished video is never flipped to `failed` by a revision.

## ComfyUI server cleanup

The ComfyUI build in use has no file-deletion API (its files route is GET-only), so cleanup **scrubs**: each rendered output and uploaded chaining frame is overwritten with a 73-byte PNG stub through the upload API (storage reclaimed; a zero-size name remains listed), and the prompt's history entry is deleted. This runs per artifact immediately after download (`COMFY_CLEANUP` setting) — long jobs never accumulate data on the render server. GPU-VRAM sharing is also managed: before TTS the app asks ComfyUI to unload models (`POST /free`), since Fish/vLLM/ComfyUI share one GPU.

## Transcript quality gate

Before any generation, captions are cleaned of cue tags (`[Music]` etc.) and validated: minimum word count and minimum words-per-minute of caption coverage (computed from caption timestamps, so it works even when video metadata is unavailable). Failing videos are refused with a clear user-facing message and consume no GPU time.

## Storage

```
output/<job_id>/
  job.json            persisted job state (resume source of truth)
  transcript.txt      source captions, or the pasted text for transcript jobs
  plan.json           music.mp3           narration.wav
  clip_01..NN.mp4     frame_NN.jpg        thumbnail.jpg      thumb_raw.mp4
  sbimg_NN.jpg        card_NN.png         clips_joined.mp4   clips_overlay.mp4  final.mp4
  resource.md         resource.txt        resource.pdf       (the promised thing, or a repo setup guide)
output/_uploads/
  <id>.mp3            music a user uploaded instead of generating a song, with an <id>.json sidecar
                      (owner, name, duration); consumed when the job is created, swept after 24 h
users/
  users.json          login accounts (scrypt hashes)     session_secret   cookie signing key
```

`settings.json` (repo root, untracked; `VIDEOGEN_SETTINGS_FILE` relocates it — the Docker compose file puts it in the bind-mounted `/app/settings/`) holds runtime setting overrides applied on top of `config.py` defaults at import and on every save. If the relocated file does not exist yet, the repo-root one is read as a one-time carry-over.
