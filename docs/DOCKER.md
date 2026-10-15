# Docker Deployment Guide

Run VideoGen as a container. The container hosts the web app and pipeline worker only — the GPU services (ComfyUI, vLLM, Fish Speech) run elsewhere and are reached over the network; lyric transcription (faster-whisper) runs on the container's CPU.

## Prerequisites

- Docker Engine 20.10+ with the Compose plugin (`docker compose version`)
- Network reachability from the Docker host to your ComfyUI (`:8188`), vLLM (`:8000`) and Fish Speech (`:7860`) services
- Outbound internet access from the container (YouTube transcripts, first-run whisper model download)

## Quick start

```bash
git clone https://github.com/mkamranr/youtube-video-generator.git
cd youtube-video-generator

# 1. Tell compose where your GPU services run (never "localhost" - inside the
#    container that is the container itself). .env is gitignored.
cp .env.example .env
#    GPU_HOST=<LAN IP of the ComfyUI/vLLM/Fish machine>   or host.docker.internal
#    when they run on the Docker host itself (Docker Desktop).

# 2. Build and start
docker compose up -d --build

# 3. Open the UI
#    http://localhost:8080
```

## Configuration

Every setting from the Configuration table in the README can be supplied as an environment variable with the `VIDEOGEN_` prefix. Precedence: **environment variable → `settings.json` (Settings page) → `config.py` default**. Environment values re-apply on every container start, so treat them as the deployment's source of truth.

| Variable | Example |
|---|---|
| `VIDEOGEN_COMFY_URL` | `http://192.168.1.50:8188` |
| `VIDEOGEN_VLLM_URL` | `http://192.168.1.50:8000/v1` |
| `VIDEOGEN_VLLM_MODEL` | `agents-a1` |
| `VIDEOGEN_FISH_URL` | `http://192.168.1.50:7860/` |
| `VIDEOGEN_TTS_TEMPERATURE` | `0.8` |
| `VIDEOGEN_TTS_TOP_P` | `0.8` |
| `VIDEOGEN_TTS_REPETITION_PENALTY` | `1.1` |
| `VIDEOGEN_TTS_CHUNK_CHARS` | `300` |
| `VIDEOGEN_TARGET_VIDEO_SECONDS` | `90` (`0` = no target) |
| `VIDEOGEN_COMFY_CLEANUP` | `true` / `false` |
| `VIDEOGEN_SECRET_KEY` | any long random string — signs the login session cookies (optional; without it a random key is generated once into `/app/users/session_secret`) |
| `VIDEOGEN_SETTINGS_FILE`, `VIDEOGEN_OUTPUT_DIR`, `VIDEOGEN_CHANNELS_DIR`, `VIDEOGEN_USERS_DIR` | relocate the settings file / the output, channels and users directories (the compose file uses bind mounts instead; these mainly serve tests and custom layouts) |
| `TZ` | `Asia/Dubai` — the container's local clock, which the GPU schedule window (Settings → GPU schedule) is compared against; default `UTC` |
| …and every other `EDITABLE_SETTINGS` key | |

Booleans accept `1/true/yes/on` (case-insensitive); numbers are parsed and range-validated — invalid values are ignored with the previous value kept.

## Persistence

| Path in container | Purpose | Recommended mount |
|---|---|---|
| `/app/output` | All generated videos, thumbnails, and per-job state (**required** for resume across restarts and for keeping your videos) | `./data/output` bind mount |
| `/root/.cache/huggingface` | faster-whisper model cache (~150 MB, downloaded once) | named volume `whisper-cache` |
| `/app/reference_audio` | Reference voices for Fish Speech cloning (Settings page) | `./data/reference_audio` bind mount |
| `/app/channels` | Channels: index, logos and per-channel YouTube tokens (Settings page) | `./data/channels` bind mount |
| `/app/users` | Login accounts (username, role, password hash) and the session signing key — **required**, or every container recreation returns to the first-run setup screen and signs everyone out | `./data/users` bind mount |
| `/app/settings` | Settings-page changes (`settings.json`: GPU schedule window, clip length, service URLs ...) — the container is told to keep the file here via `VIDEOGEN_SETTINGS_FILE=/app/settings/settings.json` | `./data/settings` bind mount |
| `/app/models` | Presenter avatar: the CPU matting model `u2net_human_seg.onnx` (176 MB). Too large for the repository, so it is **not** in the image — put the file in `./data/models/` by hand (docs/DEPLOYMENT.md section 2.1). Missing it costs only the presenter overlay, with the reason logged | `./data/models` bind mount |

The compose file ships with all of them configured. Without the `settings` mount every image rebuild or container recreation resets the Settings page (including the GPU schedule) to the `config.py` defaults.

## Operations

```bash
docker compose logs -f videogen      # live logs
docker compose restart videogen      # restart (interrupted jobs auto-resume)
docker compose up -d --build         # upgrade after a git pull
docker compose down                  # stop (output/ data persists on the host)
```

- **Health**: the container healthcheck polls the app every 30 s (`docker ps` shows `healthy`). Note it checks the app itself; use the Settings page's Service health panel to check ComfyUI/vLLM/Fish connectivity.
- **Restart safety**: `restart: unless-stopped` plus the pipeline's resume design means a crash or host reboot costs at most the in-flight render — jobs re-queue and reuse every completed artifact.
- **Port**: change the host port by editing `ports:` (e.g. `"9090:8080"`).

## Troubleshooting

- **Jobs fail instantly with connection errors** — the service URLs point at `localhost`/`127.0.0.1`, which inside the container is the container itself. Use `host.docker.internal` or the host's real IP.
- **First music-mode job pauses at "timing the sung lyrics"** — the whisper model is downloading into the cache volume (one-time, ~150 MB).
- **Transcript fetch fails for all videos** — the container has no outbound internet access, or YouTube requires a newer `yt-dlp` (`docker compose build --no-cache` picks up the latest).
- **A job fails on the overlay or upscale pass with no ffmpeg error text** (`ffmpeg killed by signal 9`) — the container ran out of memory. The overlay pass is the pipeline's peak: it opens one input per layer, each looped for the whole video, with the page screenshot and every card held as a full-frame RGBA buffer, so a 1440x2580 video with a page panel, three cards, hook and end cards, a progress bar and two presenter segments wants a couple of GB on its own. Confirm with `docker inspect videogen --format '{{.State.OOMKilled}}'` and give the container headroom — on Docker Desktop that is WSL2's memory (`.wslconfig`), on Linux it is only a limit if you set one. A killed pass is retried once automatically, which is usually enough; a container that is permanently short will fail both attempts.
- **Settings-page changes disappear after `docker compose down && up`** — mount `/app/settings.json` (see Persistence) or move those values into `VIDEOGEN_*` env vars.
