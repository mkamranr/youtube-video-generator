# Deployment Guide

How to run VideoGen for real use — on Windows, on Linux (as a systemd service, optionally behind HTTPS), or as a Docker container — plus what to back up, how to update, and how the first sign-in works.

## 1. What you are deploying

```
 browser ──HTTP(S)──▶  VideoGen (this app)  ──▶ ComfyUI      :8188  (video + music + images, GPU)
                        FastAPI + worker     ──▶ vLLM         :8000  (scripts, scenes, story board, SEO)
                        port 8080            ──▶ Fish Speech  :7860  (voiceover TTS, GPU)
                                             ──▶ YouTube / Google APIs (transcripts, uploads)
```

VideoGen itself is a single Python process: the web UI/API and one background worker that runs jobs serially. It needs **no GPU**. The three AI services run wherever your GPU is (same machine or another host) and are reached over HTTP; their URLs are set on the Settings page or via environment variables. Whisper (lyric timing for song videos) runs on the app's CPU.

Everything the app owns lives in its folder:

| Path | Contents | Persist? |
|---|---|---|
| `output/` | every job: clips, audio, story board cards, `final.mp4`, `job.json` (resume state) | yes |
| `channels/` | channel index, logos, per-channel YouTube tokens and OAuth clients | yes |
| `users/` | login accounts (scrypt hashes) and the session-cookie signing key | yes |
| `reference_audio/` | reference voices for voice cloning | yes |
| `settings.json` | Settings-page changes (GPU schedule window, clip length, URLs). Repo root by default; `VIDEOGEN_SETTINGS_FILE` relocates it — Docker keeps it in `data/settings/` | yes |
| `client_secret.json` | shared Google OAuth client (YouTube upload) | yes |
| `server.log` | server output when started via the watchdog / service | rotate |

None of these are committed to git.

## 2. Prerequisites (all platforms)

- **Python 3.10 or newer** (3.10 and 3.11 are what the project is run and built with).
- **git** to clone and update.
- **ComfyUI** reachable over HTTP with the MiniMax H3 video model and the MiniMax Music3 model installed; the three API-format workflows in `workflows/` must load there unchanged.
- **vLLM** (or any OpenAI-compatible chat endpoint) serving an instruction model that returns JSON reliably; the default model id is `agents-a1`.
- **Fish Speech** (Gradio app) for voiceover narration.
- Outbound internet from the app host: YouTube transcripts, the one-time ffmpeg download (`static-ffmpeg`), the one-time faster-whisper model download, Google APIs for uploads.
- Fonts for the story board cards: Windows uses Segoe UI automatically; on Linux install **DejaVu** (`fonts-dejavu-core`) or drop any `.ttf` into a `fonts/` folder inside the project (`*Bold*.ttf` + a regular face). Without any font the cards still render with Pillow's built-in face, just less polished.

`pip install -r requirements.txt` installs everything else, including ffmpeg/ffprobe binaries (downloaded on first use) and Pillow.

### 2.1 Presenter avatar (optional) — model files to stage by hand

The talking-head presenter needs weights on both machines, and neither set is in git: one is
too large for a repository, the other is 24 GB on a box that is air-gapped by design. Skip
this whole section if you are not using the feature — a job logs the reason and assembles
without a presenter rather than failing.

**On the app host — the matting model.** `models/u2net_human_seg.onnx`, 176 MB, which cuts
the background away for the cut-out look and finds the head when a portrait is squared on
upload. It runs on the CPU through `onnxruntime` (already in `requirements.txt`); it is
deliberately not done on the GPU box, which is shared, serial, and cannot `pip install`
`rembg`. Download `u2net_human_seg.onnx` from the `danielgatis/rembg` releases and drop it in
`models/`. `VIDEOGEN_MODELS_DIR` moves that folder elsewhere. **Under Docker** put it in
`./data/models/` on the host instead — compose bind-mounts that to `/app/models`, and the file
is excluded from the build context by `.dockerignore`, so it is never baked into the image.

**On the app host — the two face models, for measuring the likeness.** Optional even by the
standards of this section: without them a presenter renders identically, it is simply not
measured. They are what puts a number on how far a long chained presenter has drifted off the
uploaded portrait, per chunk, in the job log and in `plan.json` — and what
`scripts/avatar_identity.py` reads back. Both come from the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) and are read through the `opencv-python`
already in `requirements.txt`, so they add no dependency:

```
models/face_detection_yunet_2023mar.onnx      0.2 MB   opencv_zoo/models/face_detection_yunet/
models/face_recognition_sface_2021dec.onnx     37 MB   opencv_zoo/models/face_recognition_sface/
```

Same folder, same `VIDEOGEN_MODELS_DIR` and same `./data/models/` under Docker as the matting
model. Read the scores as a coarse instrument: a different person measures about 0.44 below the
real one, which is decisive, but the metric will not adjudicate a subtle resemblance, and a
move smaller than about 0.04 between chunks is inside its own error.

**On the ComfyUI box — the Wan 2.2 S2V weights.** Four files, ~24 GB, listed with their
destination folders in `scripts/install_wan_s2v.sh`. Run that script on the box if it has
internet; it ends with a staging route for when it does not, which is the usual case here.
ComfyUI 0.37.0 already ships the `WanSoundImageToVideo` nodes, so this is weights only.
Restart ComfyUI afterwards so it rescans `models/`. The other engine, LTX 2.5, needs nothing
new — it is the stack the clips already render on.

## 3. Configuration model

Precedence, highest first:

1. `VIDEOGEN_<SETTING>` environment variables (e.g. `VIDEOGEN_COMFY_URL`) — the deployment's source of truth, re-applied at every start.
2. `settings.json` — written by the Settings page (path from `VIDEOGEN_SETTINGS_FILE`, default repo root).
3. Defaults in `config.py`.

Every row of the Configuration table in the README can be set this way. Two extra variables are deployment-only and never appear in the UI:

| Variable | Purpose |
|---|---|
| `VIDEOGEN_SECRET_KEY` | Signs the login cookies. Optional: without it a random key is generated once into `users/session_secret`. Set it explicitly when you run more than one instance or want sessions to survive a wiped `users/` folder. |
| `VIDEOGEN_USERS_DIR` | Where accounts and the signing key live (default `<project>/users`). |

The app binds to **127.0.0.1** in the provided scripts. To reach it from other machines either bind to `0.0.0.0` (LAN only, behind your firewall) or, preferably, put a reverse proxy with HTTPS in front (section 7).

## 4. Windows

### 4.1 Install

```bat
git clone https://github.com/mkamranr/youtube-video-generator.git
cd youtube-video-generator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Run interactively

```bat
start_server.bat
```

The batch file is a watchdog: it starts `uvicorn` on `http://127.0.0.1:8080`, appends output to `server.log`, and restarts the server whenever it exits (crash, `Ctrl+C` inside the process, `Stop-Process`). Close the window to stop it. If you use a virtual environment, activate it before running the batch file, or edit the `python` line in it to `.venv\Scripts\python.exe`.

Plain alternative without the watchdog:

```bat
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

### 4.3 Run at logon / boot (Task Scheduler)

1. Open **Task Scheduler → Create Task**.
2. *General*: name `VideoGen`, tick **Run whether user is logged on or not**, optionally **Run with highest privileges** is *not* needed.
3. *Triggers*: **At startup** (or *At log on* if you want the log to show in a console).
4. *Actions*: **Start a program** → Program: `C:\path\to\youtube-video-generator\start_server.bat`, *Start in*: `C:\path\to\youtube-video-generator`.
5. *Settings*: untick **Stop the task if it runs longer than**; tick **If the task fails, restart every 1 minute**.

Stop / restart: end the task in Task Scheduler (or `taskkill /IM python.exe /F` if it is the only Python process).

### 4.4 Run as a Windows service (NSSM)

For a real service (starts before logon, restarts on failure, managed with `sc`/Services):

```bat
nssm install VideoGen "C:\path\to\youtube-video-generator\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8080
nssm set VideoGen AppDirectory "C:\path\to\youtube-video-generator"
nssm set VideoGen AppStdout "C:\path\to\youtube-video-generator\server.log"
nssm set VideoGen AppStderr "C:\path\to\youtube-video-generator\server.log"
nssm set VideoGen AppEnvironmentExtra VIDEOGEN_COMFY_URL=http://192.168.1.50:8188 VIDEOGEN_VLLM_URL=http://192.168.1.50:8000/v1 VIDEOGEN_FISH_URL=http://192.168.1.50:7860/
nssm start VideoGen
```

(NSSM: https://nssm.cc — a single `nssm.exe`, no installer.) Use `nssm restart VideoGen` after updates.

### 4.5 Reaching it from other PCs

Bind to all interfaces (`--host 0.0.0.0`) **and** open the port: `netsh advfirewall firewall add rule name="VideoGen" dir=in action=allow protocol=TCP localport=8080`. Because the app now has logins, make sure the port is only reachable on your LAN/VPN, or terminate HTTPS in front of it with Caddy (section 7) so passwords are not sent in clear text.

## 5. Linux

### 5.1 Install

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv git fonts-dejavu-core   # Debian/Ubuntu
# Fedora/RHEL: sudo dnf install -y python3 git dejavu-sans-fonts

sudo useradd --system --create-home --shell /usr/sbin/nologin videogen   # optional dedicated user
sudo -u videogen -H bash -c '
  cd ~ && git clone https://github.com/mkamranr/youtube-video-generator.git
  cd youtube-video-generator
  python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
  .venv/bin/python -c "from static_ffmpeg import run; run.get_or_fetch_platform_executables_else_raise()"
'
```

The last line pre-downloads ffmpeg/ffprobe so the first job does not wait for it.

### 5.2 Run interactively

```bash
cd ~videogen/youtube-video-generator
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

### 5.3 systemd service

`/etc/systemd/system/videogen.service`:

```ini
[Unit]
Description=VideoGen - AI video generator
After=network-online.target
Wants=network-online.target

[Service]
User=videogen
Group=videogen
WorkingDirectory=/home/videogen/youtube-video-generator
Environment=PYTHONUNBUFFERED=1
# Point at your GPU services (or set them later on the Settings page):
Environment=VIDEOGEN_COMFY_URL=http://192.168.1.50:8188
Environment=VIDEOGEN_VLLM_URL=http://192.168.1.50:8000/v1
Environment=VIDEOGEN_FISH_URL=http://192.168.1.50:7860/
# Optional, recommended: fixed cookie-signing key
# Environment=VIDEOGEN_SECRET_KEY=change-me-to-a-long-random-string
ExecStart=/home/videogen/youtube-video-generator/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8080 --proxy-headers --forwarded-allow-ips=127.0.0.1
Restart=always
RestartSec=3
# Long renders: give the worker time to persist state on stop
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now videogen
sudo systemctl status videogen
journalctl -u videogen -f          # live log
```

`--proxy-headers` lets uvicorn trust `X-Forwarded-Proto` from a local reverse proxy, so the app knows it is served over HTTPS and marks the login cookie `Secure`. Drop both flags if you expose port 8080 directly.

Interrupted jobs are safe: on restart every non-finished job is re-queued and resumes from its last completed step.

### 5.4 Firewall

Keep 8080 closed to the outside (`sudo ufw deny 8080`) and publish only the reverse proxy's 443, or `sudo ufw allow from 192.168.1.0/24 to any port 8080` for LAN-only direct access.

## 6. Docker

The image contains the app only; the GPU services stay outside the container. Full details, every variable and the volume table are in **docs/DOCKER.md** — this section is the operational summary.

### 6.1 Compose (recommended)

```bash
git clone https://github.com/mkamranr/youtube-video-generator.git
cd youtube-video-generator
cp .env.example .env   # set GPU_HOST to the machine running ComfyUI/vLLM/Fish (host.docker.internal or a LAN IP, never localhost)
docker compose up -d --build
docker compose logs -f
```

Persisted through bind mounts under `./data/`: `output/`, `channels/`, `users/` (accounts + signing key), `reference_audio/` (uploaded voices) and `settings/` (the Settings page's `settings.json`, incl. the GPU schedule window — `VIDEOGEN_SETTINGS_FILE` points the app there); the whisper model lives in a named volume. **Mount `users/`** — without it every container recreation returns to the first-run setup screen and signs everyone out. Set `VIDEOGEN_SECRET_KEY` in the `environment:` block for a stable key across rebuilds.

On Linux hosts `host.docker.internal` needs `extra_hosts: ["host.docker.internal:host-gateway"]` under the service, or use the host's LAN IP.

### 6.2 Plain `docker run`

```bash
docker build -t videogen .
docker run -d --name videogen --restart unless-stopped -p 8080:8080 \
  -e VIDEOGEN_COMFY_URL=http://192.168.1.50:8188 \
  -e VIDEOGEN_VLLM_URL=http://192.168.1.50:8000/v1 \
  -e VIDEOGEN_FISH_URL=http://192.168.1.50:7860/ \
  -e VIDEOGEN_SECRET_KEY=change-me \
  -v "$PWD/data/output:/app/output" -v "$PWD/data/channels:/app/channels" \
  -v "$PWD/data/users:/app/users" -v "$PWD/data/reference_audio:/app/reference_audio" \
  -v videogen-whisper:/root/.cache/huggingface \
  videogen
```

### 6.3 Health, update, logs

- Health: the image's `HEALTHCHECK` calls `GET /api/ping` (public; everything else needs a login). `docker inspect --format '{{.State.Health.Status}}' videogen`.
- Update: `git pull && docker compose up -d --build` (jobs in progress resume after the restart).
- Logs: `docker compose logs -f videogen`. The container writes to stdout, not `server.log`.
- Publish only through a reverse proxy in production (`ports: - "127.0.0.1:8080:8080"` then proxy to it), see section 7.

## 7. HTTPS with a reverse proxy (any platform)

Logins are sent as passwords, so terminate TLS in front of the app whenever it is reachable beyond your own machine.

**Caddy** (Windows, Linux, Docker — automatic certificates):

```
videogen.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

**nginx** (Linux):

```nginx
server {
    listen 443 ssl http2;
    server_name videogen.example.com;
    ssl_certificate     /etc/letsencrypt/live/videogen.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/videogen.example.com/privkey.pem;
    client_max_body_size 50m;          # logo / voice-sample uploads
    proxy_read_timeout 600s;           # slow first requests while models download

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Run uvicorn with `--proxy-headers --forwarded-allow-ips=127.0.0.1` (already in the systemd unit) so the cookie gets the `Secure` flag. Video files are served by the app itself; if `final.mp4` downloads stall behind a proxy, raise its buffering/timeouts.

**YouTube OAuth on a remote host:** with the default Desktop-type Google client the consent flow always returns to `http://localhost:<port>/api/youtube/callback` (the only kind of redirect Google allows for such clients). On the server machine that completes by itself; from any other computer the page cannot load and you paste its address into the *Finish the YouTube connection* dialog in Settings — no Google Cloud changes needed. If you have an HTTPS domain, you can instead create a Web-type client with `https://your-domain/api/youtube/callback` registered and set **Settings → Public base URL** to `https://your-domain` for a fully automatic sign-in. Details: docs/YOUTUBE.md.

## 8. First run and accounts

1. Open the app. With no account on file every page redirects to **Create the admin account**. Choose a username (3–40 characters, letters/digits/`.`/`-`/`_`) and a password (8+ characters). This screen never appears again.
2. Sign in. **Settings → Users** lets admins add users, reset passwords, grant/revoke admin rights and delete accounts; there is always at least one admin. Everyone can change their own password under **Settings → Your account**.
3. Users see and manage only their own channels and jobs (a new user starts with no channel and creates one under **Settings → Channels** before making videos). Admins see everyone's channels and jobs, tagged with the owner, and alone manage the service settings, GPU schedule, reference voices, style previews and users. Channels and jobs created before this rule existed belong to the oldest admin; a deleted user's channels and jobs move to the admin who deleted them.
4. Scripts and automation: `POST /api/auth/login` once and reuse the cookie (`curl -c cookies.txt … ; curl -b cookies.txt …`); details in docs/API.md.

Locked out? Stop the server, delete `users/users.json` (keep `session_secret` if you like), start again: the setup screen returns and you create a new admin. Existing jobs are untouched.

## 9. Updating

```bash
git pull
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
# then restart: systemctl restart videogen | nssm restart VideoGen | close & rerun start_server.bat
```

Docker: `git pull && docker compose up -d --build`. Restarting while a job renders loses at most the in-flight clip; the job re-queues and reuses every finished step.

## 10. Backups

Back up the folders in section 1 (`output/`, `channels/`, `users/`, `reference_audio/`, `settings.json`, `client_secret.json`). `channels/*_yt_token.json` and `users/` contain credentials — encrypt backups or restrict access. `output/` is by far the largest; per-job folders can be deleted freely once the video is published (or use **Archive/Delete** in the UI).

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Browser shows the setup screen again after a restart | `users/` was not persisted (Docker: mount `./data/users:/app/users`; check the folder is writable by the service user). |
| Signed out after every restart although users exist | The signing key changed: set `VIDEOGEN_SECRET_KEY` or persist `users/session_secret`. |
| Settings → Service health shows a service unreachable | From the app host (or inside the container: `docker exec videogen python -c "import urllib.request;print(urllib.request.urlopen('http://HOST:8188/system_stats').status)"`) check the URL; in Docker never use `localhost` for a service on the host. |
| Story board text looks plain / boxy on Linux | Install `fonts-dejavu-core` (or add `.ttf` files to `fonts/`) and restart. |
| `ffmpeg killed by signal 9` / a job fails on the overlay or upscale pass with no error text | The container ran out of memory; the overlay pass is the peak. Check `docker inspect videogen --format '{{.State.OOMKilled}}'` and `docker stats` during an assembly, and give the container a few GB (Docker Desktop: WSL2 memory; Linux: no `mem_limit` needed unless you set one). A killed pass is retried once automatically. |
| Jobs hang at "writing script with the LLM" | vLLM is wedged: it answers `/v1/models` but decodes no tokens (`vllm:num_requests_running` stuck at 1 in `/metrics`). Restart vLLM on the render box, then **Resume** the job. The job fails by itself after `LLM_TIMEOUT` (default 300 s). |
| First job hangs at "timing the sung lyrics" | faster-whisper is downloading its model (once, ~150 MB); needs outbound internet. |
| `ffmpeg failed` on first assembly | `static-ffmpeg` could not download its binaries; run the pre-fetch line from 5.1 or set `FFMPEG_BIN`/`FFPROBE_BIN` to system binaries. |
| YouTube "redirect_uri_mismatch" | Add the exact `https://your-host/api/youtube/callback` (and the port, if any) to the OAuth client's redirect URIs. |
| Port 8080 already in use | Change `--port` (and the proxy / compose `ports:`); the UI has no hard-coded port. |
| Job stuck in *queued* after a restart | Normal for a few seconds: interrupted jobs are re-queued and resume; check the job log for `server restarted - job requeued`. |
