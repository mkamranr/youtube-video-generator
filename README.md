# VideoGen — AI Video Generator for Habibi Kids Channel

VideoGen turns any YouTube video into a brand-new, kid-safe animated video — complete with an original soundtrack **or a voiceover narration in a voice you choose (Fish Speech voice cloning)**, scene-synced visuals, a YouTube-ready thumbnail, and SEO metadata — then publishes it to YouTube with one click, using locally hosted AI services end to end.

Paste a YouTube URL, pick an art style and aspect ratio, and the pipeline produces a finished `final.mp4` plus everything needed to publish it.

## Features

- **Three ways to create a video** — a YouTube URL (transcript-driven), a **topic** (the LLM invents the script), or your own **description / short script** (the LLM builds on it faithfully). Topic/description jobs draft their script immediately — outside the video queue — and wait for your **approval** before any video generation starts.
- **30 selectable art styles** — Anime, Western Cartoon, Pixar-like 3D, Claymation, Watercolor, and more; each drives the visual direction of every generated shot and the thumbnail. A **preview image** of each style (rendered once by your own video model via Settings → "Generate style previews") is shown while picking.
- **Kid-safe by design** — every LLM prompt enforces family-friendly, gentle content; even "dark" styles render as playful and reassuring.
- **Channel branding** — a configurable channel name (default: *Habibi Kids Channel*) replaces any channel/brand names found in source content, across narration, lyrics, and SEO.
- **Song or voiceover, chosen per video** — the New Video dialog offers **Generate Song** (an original song is generated **first**; its actual length sets the clip count and its lyrics drive the visuals) or **Generate Voiceover** (the LLM writes the narration, Fish Speech speaks it, and a quiet music bed is mixed underneath). The `AUDIO_MODE` setting only preselects the dialog; a pure `narration` mode (no music bed) remains available via the setting/API.
- **Voice cloning with reference voices** — upload short speech samples (plus the exact text spoken in them) on the Settings page, then pick a voice per voiceover video; Fish Speech clones it for the narration. No selection = the model's default voice.
- **Lyric-accurate visuals** — the generated song is transcribed with word timestamps (faster-whisper); each 5-second clip's scene is written from the words *actually sung during that clip's window*, so what you hear is what you see. Instrumental passages get atmosphere/transition scenes.
- **Continuous clips** — clip 1 is text-to-video; every following clip is seeded with the last frame of the previous one for smooth visual flow.
- **Thumbnail generation** — an LLM-designed scene that instantly shows what the video is about, rendered through the video model as a JPG matching the video aspect ratio (1280×720 for 16:9, 720×1280 for portrait, etc.), with your **channel logo** (uploaded in Settings) overlaid top-right. One-click regeneration on any finished job.
- **SEO metadata** — a keyword-optimized YouTube title and description (with hashtags and a subscribe call-to-action) generated for every video.
- **LLM review pass** — every generated scene and every drafted script is reviewed a second time by the LLM against the exact audio/content it must match, and inconsistencies (a wrong letter, a mismatched object, an unsafe detail) are fixed automatically. The reviewer is transcription-aware: if the speech recognizer misheard a letter, it trusts the word ("fairy" belongs to F). Toggleable via `SCRIPT_REVIEW`.
- **Per-job target length** — pick Auto, 30-45 s, 45-60 s, 1-2 min, or 2-3 min when creating a job; a soft target that guides the script length and the generated song/narration duration.
- **One-click YouTube upload** — connect your channel once (OAuth, `youtube.upload` scope only) and push any finished video straight to YouTube with live progress: uploaded **private** by default (review in YouTube Studio before publishing), with the SEO title/description, made-for-kids flag, and custom thumbnail applied. One-time Google Cloud setup guide: **docs/YOUTUBE.md**.
- **Title-named downloads & bundle** — the video and thumbnail download as `<Title>.mp4` / `<Title>.jpg`, and one click grabs the whole publish **bundle**: a ZIP with the video, thumbnail, audio track(s), and the SEO title/description in a text file.
- **Full revision workflow** — edit any scene prompt in the UI, regenerate exactly the clips you want (they're clearly badged), then reassemble the final video with one click. Regenerations **jump the queue**, pausing the running job at a clip boundary and resuming it afterwards.
- **Robust operations** — crash-safe resume (every pipeline step is idempotent), instant cancel (including aborting the in-flight GPU render), a serial GPU queue with live status messages, automatic ComfyUI server cleanup (per artifact, plus an **orphan sweep** at startup and after every job that catches renders left behind by retries or crashes), transcript quality gating, and a self-restarting server watchdog.
- **Polished web UI** — dark/light theme, a **New Video** dialog behind a *Create new video* button, a type-to-search art-style picker with live model-rendered previews, search across the Jobs and Archived Jobs lists, configurable dialog defaults, live progress with a pipeline stepper, rich per-job detail pages, archive/cancel/delete, and a runtime Settings page with service health checks.

## Screenshots

**Creating a video** — the New Video dialog: pick a source (URL / topic / description), target length, aspect ratio, and a searchable art style with a live preview rendered by your own model:

![New Video dialog](screenshots/video_1.JPG)

**A finished job** — pipeline stepper, job metadata, and the final portrait video ("Jack and Jill's Pencil Adventure", Pencil Sketch style, 9:16):

![Job page with finished video](screenshots/video_2.JPG)

**The clip grid** — every 5-second clip with checkbox selection for targeted regeneration:

![Clip grid with regeneration selection](screenshots/video_3.JPG)

## Architecture

```
YouTube URL
    │
    ▼
┌─────────────┐   captions    ┌──────────────┐  title/style/story  ┌──────────────┐
│ Transcript   │──────────────▶│  Story plan  │────────────────────▶│  Soundtrack  │
│ (yt captions │  quality gate │  (vLLM LLM)  │   + SEO metadata    │              │
│  + yt-dlp)   │               └──────────────┘                     │ music mode:  │
└─────────────┘                                                     │  MiniMax     │
                                                                    │  Music3 song │
                                                                    │ narration:   │
                                                                    │  Fish Speech │
                                                                    └──────┬───────┘
                                                                           │ actual duration
                                                                           ▼
┌──────────────┐   1280x720   ┌──────────────┐  1 scene per clip  ┌──────────────┐
│  Thumbnail   │◀─────────────│    Scenes    │◀───────────────────│ Lyric timing │
│ (LLM prompt  │              │ (vLLM, one   │   audio window     │(faster-      │
│  + 5-frame   │              │  call per    │                    │ whisper word │
│  T2V render) │              │  clip)       │                    │ timestamps)  │
└──────┬───────┘              └──────┬───────┘                    └──────────────┘
       │                             ▼
       │                      ┌──────────────┐  last frame chains  ┌──────────────┐
       │                      │    Clips     │────────────────────▶│   Assemble   │
       └─────────────────────▶│ (ComfyUI     │   clip N → N+1      │ (ffmpeg      │
                              │  MiniMax H3  │                     │  concat +    │
                              │  T2V / I2V)  │                     │  audio mix)  │
                              └──────────────┘                     └──────┬───────┘
                                                                          ▼
                                              final.mp4 + thumbnail.jpg + SEO metadata
```

### External services

| Service | Purpose | Default endpoint |
|---|---|---|
| ComfyUI (MiniMax H3 T2V/I2V + Music3 workflows) | Video clips, music, thumbnail renders | `http://<gpu-host>:8188` |
| vLLM (OpenAI-compatible) | Story, scenes, lyrics, SEO, thumbnail prompts | `http://<gpu-host>:8000/v1` |
| Fish Speech (Gradio) | TTS narration (narration modes) | `http://<gpu-host>:7860` |
| faster-whisper (local, CPU) | Word-timestamped lyric transcription | in-process |
| ffmpeg/ffprobe (bundled via `static-ffmpeg`) | Frame extraction, concat, audio mixing | local |

See **docs/ARCHITECTURE.md** for module-level detail, **docs/API.md** for the HTTP API, **docs/YOUTUBE.md** for the YouTube upload setup, and **docs/DOCKER.md** for containerized deployment.

## Installation

Requirements: Windows or Linux, Python 3.10+, network access to the three GPU services.

```bash
pip install -r requirements.txt
```

`static-ffmpeg` downloads modern ffmpeg/ffprobe binaries on first use; `faster-whisper` downloads its `base` model on first transcription. The three ComfyUI workflow JSONs ship in `workflows/` and require the corresponding MiniMax H3 / Music3 models installed on the ComfyUI server.

Point the app at your services in `config.py` (defaults) or later at runtime via the Settings page.

## Running

```bash
# with auto-restart watchdog (recommended, Windows)
start_server.bat

# or directly
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open **http://localhost:8080**.

### Docker

A ready-to-use `Dockerfile` and `docker-compose.yml` are included; every setting can be supplied as a `VIDEOGEN_*` environment variable. See **docs/DOCKER.md** for the full deployment guide.

## Usage

1. **Create** — click **Create new video**. In the dialog, choose the source in *Create from* (YouTube URLs, one per line; a **topic**; or a **description / short script** — one job per topic/description), a **Video length** (Auto … 2-3 min, a soft target for script and audio duration), an **Aspect ratio**, an **Audio** choice (**Generate Song** or **Generate Voiceover**; picking Voiceover reveals a **Voice** dropdown listing your reference voices, or the default voice), and an **Art style** (type to search; a preview image shows how each style looks). The fields pre-select your defaults from Settings. Video jobs run one at a time on the GPU; script drafting runs immediately in parallel.
   - **Topic/description jobs**: the LLM writes the script right away and the job waits at *review script*. Open it, edit the title/script if you like, and click **Approve & generate video** (or **Request new draft** with feedback). Only approved jobs enter the video-generation queue.
2. **Watch** — the job page shows a live pipeline stepper, progress, per-clip previews as they render, the generated song with caption/lyrics, the narration/story, per-clip scene prompts, transcript, log, and all raw files.
3. **Publish** — when done, push the video straight to your channel with **Upload to YouTube** (private by default, with SEO title/description, thumbnail and made-for-kids flag applied — see docs/YOUTUBE.md for the one-time setup), or download the video and thumbnail (named after the video's title), or grab the **bundle**: one ZIP with the video, thumbnail, audio track, and the SEO title/description in a text file.

### Revising a finished video

1. On the job page, click **Edit** on any scene prompt, change it, **Save scene** — the matching clip is badged *needs regeneration*.
2. Click **Regenerate** on the badged clip (or checkbox-select any clips and *Regenerate selected*). Regenerations take priority: a running job pauses at its next clip boundary and resumes afterwards.
3. When satisfied, click **Reassemble final video** — a new `final.mp4` is produced in seconds (reassembly never waits behind GPU work).

Note: a regenerated clip keeps its opening continuity (seeded from the previous clip's last frame) but its new ending may differ slightly from the next clip's opening — regenerate the following clip too if that matters.

### Managing jobs

- **Cancel** a queued or running job (takes effect within seconds; the in-flight GPU render is aborted).
- **Archive** finished jobs to keep the main list clean (Archived Jobs page; unarchive anytime).
- **Delete** removes a job and all its local files permanently.

## Configuration

All settings are editable at runtime on the **Settings** page (persisted to `settings.json`, applied to new jobs immediately). Defaults live in `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `COMFY_URL` | `http://…:8188` | ComfyUI base URL |
| `VLLM_URL` | `http://…:8000/v1` | OpenAI-compatible LLM endpoint |
| `VLLM_MODEL` | `agents-a1` | Model id served by vLLM |
| `FISH_URL` | `http://…:7860/` | Fish Speech Gradio URL |
| `TTS_TEMPERATURE` | `0.8` | Fish Speech sampling temperature (0.1–2) |
| `TTS_TOP_P` | `0.8` | Fish Speech nucleus sampling (0.1–1) |
| `TTS_REPETITION_PENALTY` | `1.1` | Fish Speech penalty against repeated tokens (0.9–2) |
| `TTS_CHUNK_CHARS` | `300` | Max characters per Fish request; longer narration is split and concatenated (100–500) |
| `CHANNEL_NAME` | `Habibi Kids Channel` | Channel branding; replaces source channel/brand names |
| `DEFAULT_ART_STYLE` | `Pixar-like 3D` | Preselected art style for new jobs |
| `DEFAULT_ASPECT_RATIO` | `16:9 (Widescreen)` | Preselected aspect ratio |
| `DEFAULT_SOURCE` | `url` | Preselected creation mode in the New Video dialog |
| `DEFAULT_VIDEO_LENGTH` | `Auto` | Preselected target length for new jobs |
| `AUDIO_MODE` | `music` | Default preselection for the dialog's Audio choice: `music` / `narration` / `narration+music` |
| `TARGET_VIDEO_SECONDS` | `90` | Music mode song-length target; `0` = no target (song covers full content, up to 6 min) |
| `CLIP_SECONDS` | `5` | Length of each generated clip (2–15 s) |
| `NARRATION_WORDS` | `170` | Narration length target (narration modes) |
| `CLIP_AUDIO_VOLUME` | `0.3` | Clips' own ambience level under narration (0–1) |
| `MUSIC_VOLUME` | `0.25` | Music level under narration (narration+music mode) |
| `LIGHTNING_LORA` | `off` | 8-step turbo sampling: ~3× faster, lower quality |
| `CLIP_RETRIES` | `1` | Retries per failed clip before the job fails |
| `MIN_TRANSCRIPT_WORDS` | `50` | Refuse videos with fewer transcript words |
| `MIN_TRANSCRIPT_WPM` | `20` | Refuse transcripts too sparse for the video length |
| `SCRIPT_REVIEW` | `on` | LLM double-checks every generated scene and drafted script and fixes inconsistencies |
| `COMFY_CLEANUP` | `on` | Scrub rendered files/uploads/history from ComfyUI once downloaded |

Code-only constants (`config.py`): workflow file paths and node ids, `WHISPER_MODEL` (`base`), `MAX_TRANSCRIPT_CHARS` (60000), `COMFY_TIMEOUT` (1 h per render).

The Settings page also hosts three non-setting tools: **Channel logo** (upload once; overlaid on the top-right corner of every generated thumbnail; stored locally, never committed), **Reference voices** (upload named speech samples with their transcripts for Fish Speech voice cloning; stored in `reference_audio/`, never committed), and **Generate style previews** (renders the art-style sample images with your own model).

## Output layout

Each job writes to `output/<job_id>/`:

```
transcript.txt      cleaned source captions
plan.json           title, story, style, scenes, per-clip audio segments, music caption/lyrics, SEO
music.mp3           generated song            narration.wav   TTS voiceover (narration modes)
clip_01..NN.mp4     generated clips           frame_NN.jpg    chaining frames
thumbnail.jpg       YouTube thumbnail matching the video aspect (channel logo overlaid if uploaded)
clips_joined.mp4    concatenated clips        final.mp4       finished video with mixed audio
job.json            job state (enables crash-safe resume)
```

`output/` and `settings.json` are intentionally untracked.

## Troubleshooting

- **"This video cannot be processed: no proper transcript available"** — the source video has no captions, or too few words for its length (music videos, ambience). Adjust `MIN_TRANSCRIPT_WORDS` / `MIN_TRANSCRIPT_WPM` if needed.
- **Fish TTS CUDA out-of-memory** — Fish, vLLM and ComfyUI share the GPU. The app pre-empts this by asking ComfyUI to unload models before synthesis and retries on OOM; long texts are chunked automatically.
- **Server died / machine rebooted** — just start it again (`start_server.bat` restarts automatically). Interrupted jobs are re-queued and **resume**: every completed artifact (transcript, plan, song, scenes, finished clips) is reused; at most the in-flight render is lost.
- **A song mentions another channel** — shouldn't happen (channel branding rule), but resubmit the video or regenerate the affected clips.
- **Service health** — the Settings page pings ComfyUI, vLLM and Fish Speech and shows latency or the exact error.
