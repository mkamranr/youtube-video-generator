# VideoGen — AI Video Generator for Your YouTube Channel

VideoGen turns any YouTube video into a brand-new AI video tailored to one of your channels — animated, realistic or tech-explainer, with content rules matching the channel's audience — complete with an original soundtrack **or a voiceover narration in a voice you choose (Fish Speech voice cloning)**, scene-synced visuals, a YouTube-ready thumbnail, and SEO metadata — then publishes it to YouTube with one click, using locally hosted AI services end to end.

Paste a YouTube URL, pick an art style and aspect ratio, and the pipeline produces a finished `final.mp4` plus everything needed to publish it.

## Features

- **Four ways to create a video** — a YouTube URL (transcript-driven), a **topic** (the LLM invents the script), your own **description / short script** (the LLM builds on it faithfully), or a pasted **transcript / song lyrics** (used as the script as-is; the LLM only fixes typos, cue tags and the like). Topic/description/transcript jobs draft their script immediately — outside the video queue — and wait for your **approval** before any video generation starts.
- **30 selectable art styles** — Anime, Western Cartoon, Pixar-like 3D, Claymation, Watercolor, and more; each drives the visual direction of every generated shot and the thumbnail. A **preview image** of each style (rendered once by your own video model via Settings → "Generate style previews") is shown while picking.
- **Multiple channels** — manage any number of channels in Settings, each with its own name, purpose, logo, audience (Kids / Family / General), video type (Animated / Real / Tech) and per-video defaults. Every video is generated to match the channel picked in the New Video dialog, and uploads go to that channel's own connected YouTube account. Real and Tech channels skip the art-style picker and use a built-in realistic live-action look; **Tech** channels additionally get concrete, technically accurate scripts (devices, screens, code, data-centre visuals) and the on-screen story board switched on by default.
- **On-screen story board** — an optional overlay (per video, default on for Tech channels) of styled fact cards: the LLM plans a kicker (KEY FACT, HOW IT WORKS, BY THE NUMBERS…), a headline and a one-line body for the right moments of the narration, some with a **related still image** rendered by the video model in a shape matching the video (square for portrait, 4:3 for square, 16:9 for landscape); the cards are drawn with Pillow (rounded lower-third panel, accent bar, framed picture — layouts for landscape, portrait and square) and composited over the clips with fades, timed to the actual clip cuts. Toggle or regenerate it on any finished job, then reassemble.
- **Clean voiceover mix** — voiceover videos mix narration + an instrumental bed only: the clips' own generated sound effects are dropped by default (they clashed with the voice and the music and jumped at every cut; `CLIP_AUDIO_WITH_MUSIC` brings them back). The music bed is **side-chain ducked** under the voice, looped with a cross-fade when the model ends the track early, faded out at the end, and the mix is trimmed to the video and soft-limited. When ambience is kept it is dipped to silence around every clip cut.
- **Audience-aware safety** — the channel's audience preset drives the LLM content rules: kids channels stay strictly gentle and family-friendly (even "dark" styles render playful and reassuring), family channels allow mild peril, general channels follow platform guidelines only.
- **Channel branding** — each channel's name replaces any channel/brand names found in source content, across narration, lyrics, and SEO; its purpose text steers the tone of every script.
- **Character videos on or off** — a per-channel switch on the Settings page. On, the channel's recurring cast drives every script, scene, thumbnail and weekly plan. Off, the channel makes **informative explainers** instead: hook, the viewer's problem, the concrete steps, the payoff, spoken straight to the viewer as *"you"* — no protagonist, no story, no invented persona. Generic unnamed people (a project manager at a laptop, hands typing, a presenter) can still appear on screen, they are just never named or recurring. It is independent of the audience preset, so a grown-up channel can have characters and a kids channel could go character-free. Switching it off keeps the cast and its portraits, so it is reversible.
- **Videos from a GitHub repo or a Hugging Face model** — on a Tech channel, paste a `github.com` repository or a `huggingface.co` model, dataset or space URL instead of a YouTube link. The README or model card and the project's real facts (stars, downloads, licence, task, library) become the script, which you review and approve before anything renders. Optionally the live page is screenshotted with a locally installed headless Chrome and scrolled slowly as a large panel over the middle of the video, so viewers see the actual repo. No extra Python packages; if no browser is found the video is still made from the README.
- **Recurring cast** — give a channel up to 24 named characters (an elephant, a kid, …) with a precise look and personality, or **import a Character Bible** (a markdown document with a `### Name` block per character plus style, rules and checklist sections) in one paste on the Settings page: the character blocks become cast entries, the other sections become the channel's **Style & rules** text that is sent as a fixed reference with every script and scene prompt. Not every character is in every video: the LLM picks the 1-3 the story needs (or exactly the ones you name in the topic, e.g. *"Use Finn as the main character and Pip as the supporting character"*); the full roster is only sent while the script is written, afterwards every prompt carries just this video's characters, and each character's canonical look is prepended to every scene, thumbnail and story board prompt that names them, so the video model gets the same description every time. Approve a **reference portrait** per character (rendered by your video model in the channel's art style, or uploaded): the first character of the story that has one becomes the first frame of clip 1, so the video opens on the real character and the following clips chain from it. Looks and colours stay consistent; faces can still drift later in long videos, since the model has no identity-lock beyond the first frame.
- **Job dashboard** — the jobs list shows every job's parameters at a glance (channel, audio choice and voice, art style, target length, aspect ratio, clip progress), with a per-channel filter, text search and pagination; the archive page gets the same filter, search and pagination.
- **Song or voiceover, chosen per video** — the New Video dialog offers **Generate Song** (an original song is generated **first**; its actual length sets the clip count and its lyrics drive the visuals) or **Generate Voiceover** (the LLM writes the narration, Fish Speech speaks it, and a gentle **instrumental** music bed — never sung lyrics — is mixed underneath at low volume). The channel's default audio preselects the dialog; a pure `narration` mode (no music bed) remains available via the API.
- **Voice cloning with reference voices** — upload short speech samples (plus the exact text spoken in them) on the Settings page, then pick a voice per voiceover video; Fish Speech clones it for the narration. No selection = the model's default voice, **kept the same for the whole video** (the first narration chunk becomes the reference for the rest, so long narrations no longer switch speakers between chunks); a *random voice per segment* option keeps the old behaviour.
- **Lyric-accurate visuals** — the generated song is transcribed with word timestamps (faster-whisper); each clip's scene is written from the words *actually sung during that clip's window*, so what you hear is what you see. Instrumental passages get atmosphere/transition scenes.
- **Continuous clips** — clip 1 is text-to-video; every following clip is seeded with the last frame of the previous one for smooth visual flow.
- **Thumbnail generation** — an LLM-designed scene that instantly shows what the video is about, rendered through the video model as a JPG matching the video aspect ratio (1280×720 for 16:9, 720×1280 for portrait, etc.), with the **channel's logo** overlaid top-right. One-click regeneration on any finished job.
- **SEO metadata** — a keyword-optimized YouTube title and description (with hashtags and a subscribe call-to-action) generated for every video.
- **LLM review pass** — every generated scene and every drafted script is reviewed a second time by the LLM against the exact audio/content it must match, and inconsistencies (a wrong letter, a mismatched object, an unsafe detail) are fixed automatically. The reviewer is transcription-aware: if the speech recognizer misheard a letter, it trusts the word ("fairy" belongs to F). Toggleable via `SCRIPT_REVIEW`.
- **Per-job target length** — pick Auto, 30-45 s, 45-60 s, 1-2 min, 2-3 min, 3-4 min or 4-5 min when creating a job; a soft target that guides the script length and the generated song/narration duration. A song that lands short of the range is rewritten longer and re-rendered once, and a script drafted well under its word target is extended before it is spoken. Longer targets also render longer clips (up to 10 s), so a 5-minute video is about 27 clips rather than 59.
- **One-click YouTube upload** — connect each channel to its own YouTube account (OAuth, `youtube.upload` scope only) and push any finished video straight to that channel with live progress: uploaded **private** by default (review in YouTube Studio before publishing), with the SEO title/description, the made-for-kids flag (a per-channel toggle), and custom thumbnail applied. Connecting works from a remote server too: Google's localhost redirect is completed by pasting its address into a dialog, or set a *Public base URL* with a Web-type client on an HTTPS domain for a fully automatic sign-in. One-time Google Cloud setup guide: **docs/YOUTUBE.md**.
- **Title-named downloads & bundle** — the video and thumbnail download as `<Title>.mp4` / `<Title>.jpg`, and one click grabs the whole publish **bundle**: a ZIP with the video, any upscaled versions, thumbnail, audio track(s), and the SEO title/description in a text file.
- **Output resolution & on-demand upscaling** — pick 1080p / 2K / 4K per video (or as the Settings default) to have the final video upscaled during assembly, or upscale a finished video later from its job page; every version is kept separately.
- **Full revision workflow** — edit any scene prompt in the UI, regenerate exactly the clips you want (they're clearly badged), then reassemble the final video with one click. Regenerations **jump the queue**, pausing the running job at a clip boundary and resuming it afterwards.
- **Robust operations** — crash-safe resume (every pipeline step is idempotent), instant cancel (including aborting the in-flight GPU render), a serial GPU queue with live status messages, automatic ComfyUI server cleanup (per artifact, plus an **orphan sweep** at startup and after every job that catches renders left behind by retries or crashes), transcript quality gating, and a self-restarting server watchdog.
- **Login and user accounts** — the whole app sits behind a sign-in screen. On the very first start a one-time **setup** page creates the administrator account; it never appears again. Channels and videos are **per user**: each user creates and manages their own channels and sees only their own videos; admins see every user's channels and videos (tagged with the owner, with an owner filter), are the only ones who can change the service settings, GPU schedule, reference voices, style previews and users, and inherit the channels and videos of accounts they delete. Everyone can change their own password. Sessions are signed cookies (30 days); passwords are stored as scrypt hashes in `users/` (never committed).
- **Polished web UI** — dark/light theme, a **New Video** dialog behind a *Create new video* button, a type-to-search art-style picker with live model-rendered previews, search across the Jobs and Archived Jobs lists, configurable dialog defaults, live progress with a pipeline stepper, rich per-job detail pages, archive/cancel/delete, and a runtime Settings page with service health checks.

## Screenshots

**Creating a video** — the New Video dialog: pick a source (URL / topic / description / transcript), target length, aspect ratio, and a searchable art style with a live preview rendered by your own model:

![New Video dialog](screenshots/video_1.JPG)

**A finished job** — pipeline stepper, job metadata, and the final portrait video ("Jack and Jill's Pencil Adventure", Pencil Sketch style, 9:16):

![Job page with finished video](screenshots/video_2.JPG)

**The clip grid** — every clip with checkbox selection for targeted regeneration:

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

See **docs/ARCHITECTURE.md** for module-level detail, **docs/API.md** for the HTTP API (including sign-in for scripts), **docs/YOUTUBE.md** for the YouTube upload setup, **docs/DEPLOYMENT.md** for Windows / Linux / Docker deployments, and **docs/DOCKER.md** for the container details.

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

Open **http://localhost:8080** — the first visit shows the one-time **Create the admin account** screen (see *First run* below), every later visit the sign-in page.

### Deploying for real use

**docs/DEPLOYMENT.md** covers every supported deployment: Windows (interactive watchdog, Task Scheduler, NSSM service), Linux (virtualenv + systemd unit), Docker (compose or plain `docker run`), HTTPS via Caddy/nginx, first-run accounts, updating, backups and troubleshooting. The container specifics — image, volumes, every `VIDEOGEN_*` variable — are in **docs/DOCKER.md**.

## Usage

1. **Create** — click **Create new video**. In the dialog, choose the source in *Create from* (YouTube URLs, one per line; a **topic**; a **description / short script**; or a **transcript / lyrics** you paste, kept as the script with only fixes — one job per topic/description/transcript), a **Video length** (Auto … 4-5 min, a soft target for script and audio duration), an **Aspect ratio**, a **Resolution** in megapixels (0.2–1.0; higher = sharper clips but slower renders), an **Output resolution** (Native, 1080p, 2K or 4K — the joined clips are upscaled with ffmpeg to that size during assembly; upscaling adds pixels, not detail), an **Audio** choice (**Generate Song** or **Generate Voiceover**; picking Voiceover reveals a **Voice** dropdown listing your reference voices, or the default voice), an **Art style** (type to search; a preview image shows how each style looks — hidden for Real and Tech channels, which use the built-in live-action look), and the **Add story board cards** switch (on by default for Tech channels): styled fact cards with related images drawn over the video. The fields pre-select your defaults from Settings. You need to be signed in; every user sees the same job list. Video jobs run one at a time on the GPU; script drafting runs immediately in parallel.
   - **Topic/description/transcript jobs**: the LLM writes the script right away and the job waits at *review script*. Open it, edit the title/script if you like, and click **Approve & generate video** (or **Request new draft** with feedback). Only approved jobs enter the video-generation queue.
   - **Transcript / lyrics jobs** keep your text as the script: the LLM only fixes typos and grammar and strips timestamps, cue tags like `[Music]`, speaker labels and section tags like `[Verse]`/`[Chorus]` — it never expands, shortens or reorders your words, and the *Video length* target is not applied (the audio is as long as your text). The pasted text is shown in the job page's **Transcript** card. Pair it with **Generate Voiceover** for a read-aloud script, or **Generate Song** to have your lyrics sung.
2. **Watch** — the job page shows a live pipeline stepper, progress, per-clip previews as they render, the generated song with caption/lyrics, the narration/story, per-clip scene prompts, transcript, log, and all raw files.
3. **Upscale (optional)** — on a finished job, **Upscale this video** makes a 1080p / 2K / 4K copy of the final video with ffmpeg. The original stays; each version is shown below the main player with its own download button, goes into the bundle, and you can add further sizes later (e.g. 2K first, then 4K).
4. **Publish** — when done, push the video straight to your channel with **Upload to YouTube** (private by default, with SEO title/description, thumbnail and made-for-kids flag applied — see docs/YOUTUBE.md for the one-time setup), or download the video and thumbnail (named after the video's title), or grab the **bundle**: one ZIP with the video, thumbnail, audio track, and the SEO title/description in a text file. Finished jobs also show a **Bundle** button directly on the jobs list and the archive, so you can download without opening the job.

### Revising a finished video

1. On the job page, click **Edit** on any scene prompt, change it, **Save scene** — the matching clip is badged *needs regeneration*.
2. Click **Regenerate** on the badged clip (or checkbox-select any clips and *Regenerate selected*). Regenerations take priority: a running job pauses at its next clip boundary and resumes afterwards.
3. When satisfied, click **Reassemble final video** — a new `final.mp4` is produced in seconds (reassembly never waits behind GPU work).

Note: a regenerated clip keeps its opening continuity (seeded from the previous clip's last frame) but its new ending may differ slightly from the next clip's opening — regenerate the following clip too if that matters.

### Managing jobs

- **Cancel** a queued or running job (takes effect within seconds; the in-flight GPU render is aborted).
- **Resume** a failed job (connection timeout, service down, crash) with one click once the services are back: it re-enters the queue and continues from the exact step it failed at - the plan, music, narration and every finished clip are reused.
- **Archive** finished jobs to keep the main list clean (Archived Jobs page; unarchive anytime).
- **Delete** removes a job and all its local files permanently.
- **GPU schedule** (Settings → GPU schedule, off by default): restrict ComfyUI renders to a daily window such as 22:00–06:00. Jobs still fetch transcripts, write scripts and narrate at any time; when they reach their first render outside the window they park in the queue with the detail *waiting for the GPU window (opens at …)* and continue where they stopped once it opens. A render already running when the window closes finishes first. The jobs page shows a banner with the window state.

## First run

Open the app in a browser: it shows the **Create the admin account** screen until the first user exists. That account is the administrator — sign in with it, then add more users under Settings → Users. Accounts and the session signing key live in `users/`; delete that folder to start over (everyone is signed out and the setup screen returns). For scripts, sign in with `POST /api/auth/login` and reuse the cookie (see docs/API.md).

## Configuration

All settings are editable at runtime on the **Settings** page (persisted to `settings.json` in the project root, or wherever `VIDEOGEN_SETTINGS_FILE` points — the Docker setup keeps it in the bind-mounted `data/settings/` so it survives rebuilds; applied to new jobs immediately). Defaults live in `config.py`.

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
| `DEFAULT_SOURCE` | `url` | Preselected creation mode in the New Video dialog |
| `TARGET_VIDEO_SECONDS` | `90` | Music mode song-length target; `0` = no target (song covers full content, up to 6 min) |
| `CLIP_SECONDS` | `5` | Length of each generated clip (2–15 s) |
| `DEFAULT_MEGAPIXELS` | `0.4` | Default clip resolution in megapixels (0.2–1.0, 0.1 steps); each video can override it in the New Video dialog. Thumbnails always render at 1 MP |
| `DEFAULT_RESOLUTION` | `Native (clip size)` | Final-video size: `1080p (Full HD)`, `2K (1440p)` or `4K (2160p)` upscale the joined clips with ffmpeg (lanczos, libx264 CRF 18) before the story board cards are drawn; Native keeps the clips' own size. Never downscales; each video can override it in the New Video dialog |
| `NARRATION_WORDS` | `170` | Narration length target (narration modes) |
| `CLIP_AUDIO_VOLUME` | `0.3` | Clips' own ambience level under narration (0–1); dipped at clip cuts, ducked under the voice |
| `CLIP_AUDIO_WITH_MUSIC` | `off` | Voiceover + music videos also keep the clips' own sound (off = narration + music bed only) |
| `MUSIC_VOLUME` | `0.25` | Music level under narration (narration+music mode); ducked further while the voice speaks |
| `STORYBOARD_MAX_IMAGES` | `6` | Story board: max related images rendered per video (one short ComfyUI render each; 0 = text cards only) |
| `STORYBOARD_ACCENT` | `#e9a13b` | Story board: accent colour of the cards (kicker text, side bar) |
| `LIGHTNING_LORA` | `off` | 8-step turbo sampling: ~3× faster, lower quality |
| `CLIP_RETRIES` | `1` | Retries per failed clip before the job fails |
| `MIN_TRANSCRIPT_WORDS` | `50` | Refuse videos with fewer transcript words |
| `MIN_TRANSCRIPT_WPM` | `20` | Refuse transcripts too sparse for the video length |
| `SCRIPT_REVIEW` | `on` | LLM double-checks every generated scene and drafted script and fixes inconsistencies |
| `COMFY_CLEANUP` | `on` | Scrub rendered files/uploads/history from ComfyUI once downloaded |
| `GPU_SCHEDULE_ENABLED` | `off` | Only submit ComfyUI renders (clips, music, thumbnails, story board images, style previews) inside the window below; script writing and narration run any time |
| `GPU_WINDOW_START` | `22:00` | Window start, server local time (HH:MM) |
| `GPU_WINDOW_END` | `06:00` | Window end (HH:MM); before the start = overnight window, equal to the start = always open |

Code-only constants (`config.py`): workflow file paths and node ids, `WHISPER_MODEL` (`base`), `MAX_TRANSCRIPT_CHARS` (60000), `COMFY_TIMEOUT` (1 h per render).

The Settings page also hosts four non-setting tools: **Users** (admins only: add users, reset passwords, grant or revoke admin rights; everyone gets a *Your account* card to change their own password), **Channels** (each user's own channels — name, purpose, audience, video type, per-video defaults, logo, whether its videos have **characters** at all, the recurring **cast** with its reference portraits, and each channel's own YouTube connection; admins see everyone's; stored in `channels/`, never committed), **Reference voices** (admins only: upload named speech samples with their transcripts for Fish Speech voice cloning, shared by all users; stored in `reference_audio/`, never committed), and **Generate style previews** (admins only: renders the art-style sample images with your own model). The settings form itself is admin-only; regular users see just Channels and Your account.

## Output layout

Each job writes to `output/<job_id>/`:

```
transcript.txt      cleaned source captions (URL jobs) or the pasted transcript / lyrics (transcript jobs)
plan.json           title, story, style, scenes, per-clip audio segments, music caption/lyrics, SEO,
                    story board cards (kicker/headline/body, clip range, image prompt)
music.mp3           generated song            narration.wav   TTS voiceover (narration modes)
clip_01..NN.mp4     generated clips           frame_NN.jpg    chaining frames
thumbnail.jpg       YouTube thumbnail matching the video aspect (channel logo overlaid if uploaded)
sbimg_NN.jpg        story board related images   card_NN.png   rendered story board overlays
clips_joined.mp4    concatenated clips        clips_cards.mp4 clips with the story board overlaid
final.mp4           finished video with mixed audio
job.json            job state (enables crash-safe resume)
```

`output/` and `settings.json` are intentionally untracked.

## Troubleshooting

- **"This video cannot be processed: no proper transcript available"** — the source video has no captions, or too few words for its length (music videos, ambience). Adjust `MIN_TRANSCRIPT_WORDS` / `MIN_TRANSCRIPT_WPM` if needed.
- **Fish TTS CUDA out-of-memory** — Fish, vLLM and ComfyUI share the GPU. The app pre-empts this by asking ComfyUI to unload models before synthesis and retries on OOM; long texts are chunked automatically.
- **Server died / machine rebooted** — just start it again (`start_server.bat` restarts automatically). Interrupted jobs are re-queued and **resume**: every completed artifact (transcript, plan, song, scenes, finished clips) is reused; at most the in-flight render is lost.
- **A song mentions another channel** — shouldn't happen (channel branding rule), but resubmit the video or regenerate the affected clips.
- **Service health** — the Settings page pings ComfyUI, vLLM and Fish Speech and shows latency or the exact error.
