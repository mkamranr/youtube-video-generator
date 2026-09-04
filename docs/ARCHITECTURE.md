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
  llm.py               All vLLM prompting: story plan, per-clip scenes, music caption/lyrics, SEO metadata,
                       thumbnail prompt; the audience safety rules, channel purpose and branding live in the system prompt, built per job from its channel snapshot
  tts.py               Fish Speech synthesis (chunked <=TTS_CHUNK_CHARS/request, GPU-OOM aware,
                       optional reference-voice cloning), audio duration
  voices.py            Reference-voice library (reference_audio/): normalized samples + transcripts
                       for Fish Speech voice cloning, JSON index, add/list/delete
  lyrics.py            faster-whisper word-timestamp transcription -> words heard per clip window
  comfy.py             ComfyUI client: queue/wait (websocket + polling), cancel-aware waiting, output
                       download with node auto-detection, targeted interrupt, server-side scrubbing
  video.py             ffmpeg helpers: last-frame extraction, thumbnail crop (1280x720), clip concat,
                       timed RGBA card overlays, N-way audio mix (narration / ambience / music) with
                       side-chain ducking, music looping/fade-out and clip-cut ambience dips
  storyboard.py        Story board overlay: LLM card planning (via llm.py), related-image rendering
                       through ComfyUI, Pillow card drawing (landscape/portrait/square layouts), schedule
                       aligned to the real clip cut points
  ffbin.py             Locates modern ffmpeg/ffprobe (static-ffmpeg bundle, env override, PATH fallback)
static/                Web UI: index (submit + job list), job (detail/revision), archive, settings;
                       shared css/app.css and js/common.js (nav, status pills, mode-aware stepper)
workflows/             ComfyUI API-format workflow JSONs (text-to-video, first-frame video, music)
start_server.bat       Watchdog: relaunches uvicorn whenever it exits; logs to server.log
```

## Job lifecycle

Statuses: `queued → transcript → planning → [music | tts] → scenes → thumbnail → [storyboard] → clips → assembling → done | failed`.

- One background worker thread processes the queue serially (single GPU). Queued jobs' `detail` is computed live at request time (position, what is running).
- **music mode** (default): the song is generated *first*; its measured duration decides the clip count (`ceil(duration / CLIP_SECONDS)`); faster-whisper assigns the sung words to each clip's time window; scenes are then written **one LLM call per clip** from exactly those words (one-call-per-scene prevents alignment drift on long videos). The final mix is the song alone.
- **narration modes**: Fish TTS narration length decides the clip count; scenes come from evenly split narration segments; final mix = narration (full) + an optional gentle instrumental music bed (the music model is sent `[Instrumental]` as lyrics, so nothing is ever sung over the narration). The clips' own generated sound is dropped in narration+music mode unless `CLIP_AUDIO_WITH_MUSIC` is on (three independent layers — voice, music, per-clip effects that change at every cut — sounded like a mess); in pure narration mode it stays, dipped to silence around each clip cut. Beds are side-chain compressed under the narration; a music track the model ended early is looped with a cross-fade, and always fades out at the end. The result is trimmed to the video length and soft-limited. The audio mode and an optional reference voice (`voice_id`) are chosen per job in the New Video dialog; a job whose reference voice was deleted logs a note and falls back to the default voice.
- Clips: clip 1 uses the T2V workflow; clips 2..N use the first-frame workflow seeded with the previous clip's last frame (ffmpeg extraction → ComfyUI image upload), giving visual continuity.
- **Story board** (optional, `job.storyboard`, default on for `tech` channels): before the clips, the LLM plans the cards (`plan.json["storyboard"]`: clip range, kicker, headline, body, optional image prompt) and ComfyUI renders each requested related image as a 5-frame clip whose first frame is cropped to `sbimg_NN.jpg` (capped by `STORYBOARD_MAX_IMAGES`); the picture's shape follows the video orientation (`storyboard.image_spec`: square for portrait videos, 4:3 for square, 16:9 for landscape), and the card layout uses the same ratio for its picture slot, so older 16:9 images are simply cover-cropped. Both are resumable artifacts; failures degrade to text-only cards or to no overlay, never to a failed job.
- Assembly: ffmpeg concat (stream-copy with re-encode fallback); with the story board on, the cards are drawn with Pillow at the real frame size (`card_NN.png`) and composited with alpha fades over `clips_joined.mp4` → `clips_cards.mp4`, each card timed to the measured clip cut points (0.35 s inset); then the filtergraph audio mix; video length ≈ audio length (count is the ceiling of the audio duration).

## Resume & idempotency

Every pipeline step checks for its artifact before doing work: `transcript.txt`, `plan.json` (story, scenes, music plan, SEO, thumbnail prompt), `music.mp3`, `narration.wav`, `thumbnail.jpg`, each `clip_NN.mp4`. On server start, non-terminal jobs are re-queued and re-run `_run()`, which skips everything already done — a crash or restart costs at most the in-flight render. Job state persists in `output/<id>/job.json` after every mutation.

## Cancel, pause, priority

- **Cancel** sets a flag checked at every step boundary, inside every progress callback, and continuously inside the ComfyUI wait loop (`should_stop`); the in-flight ComfyUI prompt is interrupted only after verifying it belongs to the job (filename-prefix match). Cancelled full runs become `failed`; cancelled revision runs return to `done`.
- **Priority regeneration**: a regen request is inserted at the *front* of the queue and the running job gets `pause_requested`. `JobPaused` deliberately subclasses `BaseException` so no broad `except Exception` handler can swallow it; the paused job stops at its next checkpoint (clip boundary — no GPU work wasted), is re-inserted right after the regen job, and resumes via the normal resume path.
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
  transcript.txt      plan.json           music.mp3          narration.wav
  clip_01..NN.mp4     frame_NN.jpg        thumbnail.jpg      thumb_raw.mp4
  sbimg_NN.jpg        card_NN.png         clips_joined.mp4   clips_cards.mp4    final.mp4
```

`settings.json` (repo root, untracked) holds runtime setting overrides applied on top of `config.py` defaults at import and on every save.
