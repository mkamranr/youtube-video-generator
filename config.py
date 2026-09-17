"""Configuration: constants + user-editable settings (persisted to settings.json).

Values in EDITABLE_SETTINGS can be changed at runtime from the Settings page; they are
stored in settings.json and applied on top of the defaults below at import and on save.
The file lives in the project root by default; VIDEOGEN_SETTINGS_FILE moves it (the Docker
setup points it into a bind-mounted directory so Settings survive image rebuilds).
"""
import json
import math
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = BASE_DIR / "workflows"
OUTPUT_DIR = Path(os.environ.get("VIDEOGEN_OUTPUT_DIR") or BASE_DIR / "output")
STATIC_DIR = BASE_DIR / "static"
LEGACY_SETTINGS_FILE = BASE_DIR / "settings.json"
SETTINGS_FILE = Path(os.environ.get("VIDEOGEN_SETTINGS_FILE") or LEGACY_SETTINGS_FILE)
LOGO_FILE = BASE_DIR / "channel_logo.png"   # uploaded via Settings; overlaid on thumbnails
VOICES_DIR = BASE_DIR / "reference_audio"   # reference voices for Fish Speech cloning (Settings)
VOICES_INDEX = VOICES_DIR / "voices.json"
YT_CLIENT_SECRET_FILE = BASE_DIR / "client_secret.json"   # Google OAuth client (Desktop type)
YT_TOKEN_FILE = BASE_DIR / "youtube_token.json"           # legacy global OAuth token (pre-channels)
CHANNELS_DIR = Path(os.environ.get("VIDEOGEN_CHANNELS_DIR") or BASE_DIR / "channels")  # index, logos, tokens, portraits
CHANNELS_INDEX = CHANNELS_DIR / "channels.json"
PLANS_DIR = Path(os.environ.get("VIDEOGEN_PLANS_DIR") or BASE_DIR / "plans")  # weekly content plans
PLANS_INDEX = PLANS_DIR / "plans.json"
USERS_DIR = Path(os.environ.get("VIDEOGEN_USERS_DIR") or BASE_DIR / "users")  # login accounts
USERS_INDEX = USERS_DIR / "users.json"
SESSION_SECRET_FILE = USERS_DIR / "session_secret"   # random key signing the session cookies
SESSION_DAYS = 30                                    # how long a login stays valid

# ---- service endpoints (placeholders) -------------------------------------
# Set your real service URLs via the Settings page (persisted to settings.json)
# or VIDEOGEN_* environment variables (see docs/DOCKER.md).
COMFY_URL = "http://localhost:8188"
VLLM_URL = "http://localhost:8000/v1"
VLLM_MODEL = "agents-a1"
FISH_URL = "http://localhost:7860/"

# ---- workflows and node ids (fixed) ---------------------------------------
T2V_WORKFLOW = WORKFLOWS_DIR / "video_minimax_h3_t2v.json"
I2V_WORKFLOW = WORKFLOWS_DIR / "video_minimax_h3_t2v_first_frame.json"
MUSIC_WORKFLOW = WORKFLOWS_DIR / "audio_minimax_music_3.json"

NODE_PROMPT = "140:131"
NODE_SEED = "140:129"
NODE_DURATION = "140:133"
NODE_SAVE = "92"
NODE_LOAD_IMAGE = "141"
NODE_LIGHTNING = "140:139"
NODE_RESOLUTION = "115"

# music workflow node ids
NODE_MUSIC_ENCODE = "37:13"
NODE_MUSIC_SEED = "37:38"
NODE_MUSIC_SAVE = "35"

# Art styles the user can pick per job; each maps to the art direction fed to the LLM.
ART_STYLES = {
    "Anime": "Japanese anime style: cel-shaded characters, clean bold line art, big expressive eyes, vibrant colors, dynamic 2D motion.",
    "Retro Anime": "1980s retro anime: grainy cel animation, muted vintage palette, soft film glow, hand-painted backgrounds, nostalgic charm.",
    "Western Cartoon": "Classic western TV cartoon: bold outlines, flat bright colors, exaggerated squash-and-stretch motion, playful character designs.",
    "Classic Hand-Drawn": "Classic hand-drawn animation: soft pencil-and-ink lines, painterly backgrounds, warm storybook feel, gentle frame-by-frame motion.",
    "Modern 2D": "Modern 2D animation: clean vector shapes, flat design with subtle gradients, smooth motion, bright contemporary palette.",
    "Chibi": "Chibi style: tiny cute characters with oversized heads and eyes, pastel colors, bouncy playful motion, adorable simplicity.",
    "Motion Comic": "Motion comic: illustrated comic panels with parallax depth, dramatic ink shading, dynamic camera moves across still-art poses.",
    "Comic Book": "Comic book style: bold ink outlines, halftone dot shading, saturated primary colors, action-panel energy.",
    "Manga": "Manga style: black-and-white ink art, screentone shading, expressive speed lines, emotive character close-ups.",
    "Pixar-like 3D": "Polished 3D animation like a family film: soft global illumination, rounded appealing characters, rich warm colors, cinematic depth of field.",
    "Stylized 3D": "Stylized 3D: simplified sculpted shapes, hand-painted textures, exaggerated proportions, warm playful lighting.",
    "Realistic 3D": "Realistic 3D CGI: detailed textures, physically-based lighting, lifelike materials, cinematic rendering.",
    "Low-Poly 3D": "Low-poly 3D: faceted geometric shapes, flat-shaded pastel surfaces, minimalist charming world.",
    "Game Cinematic": "Video-game cinematic: heroic 3D characters, dramatic rim lighting, epic camera sweeps, polished render look.",
    "Claymation": "Claymation: hand-molded clay characters with visible fingerprints, stop-motion jitter, miniature set feel, warm studio lighting.",
    "Stop-Motion": "Stop-motion animation: handcrafted puppets and props, slightly jittery charming motion, tactile miniature sets.",
    "Paper Cutout": "Paper cutout animation: layered construction-paper shapes, visible edges and textures, flat 2D puppet motion, craft-project charm.",
    "Puppet Animation": "Puppet animation: soft felt and fabric puppets, expressive hand-operated movement, cozy theatrical stage lighting.",
    "Watercolor": "Watercolor animation: soft translucent washes, bleeding pigment edges, gentle paper texture, dreamy pastel palette.",
    "Oil Painting": "Oil-painting animation: thick visible brushstrokes, rich impasto texture, classical warm palette, living-painting motion.",
    "Ink Wash": "Ink-wash animation: flowing black ink gradients on rice paper, minimal elegant strokes, meditative negative space.",
    "Pencil Sketch": "Pencil-sketch animation: loose graphite lines, cross-hatched shading, sketchbook paper texture, hand-drawn wobble.",
    "Minimalist": "Minimalist animation: simple geometric characters, limited flat palette, generous negative space, clean elegant motion.",
    "Motion Graphics": "Motion graphics: crisp geometric shapes, bold flat colors, snappy synchronized transitions, kinetic design energy.",
    "Cyberpunk": "Kid-friendly cyberpunk: neon-glow city colors, holographic gadgets, playful futuristic tech, rain-slicked reflections, bright not grim.",
    "Dark Fantasy": "Gentle dark-fantasy storybook: moonlit enchanted forests, mysterious but friendly creatures, deep blues and purples, magical glow.",
    "Surreal": "Surreal dreamscape: impossible playful geometry, floating whimsical objects, candy-colored skies, gentle dream logic.",
    "Horror": "Spooky-but-friendly Halloween cartoon: cute ghosts and pumpkins, moonlit haunted-house silhouettes, playful shivers, nothing frightening.",
    "Retro/VHS": "Retro VHS look: soft scanlines, slight chroma bleed, faded 80s palette, nostalgic camcorder warmth.",
    "Experimental": "Experimental mixed-media animation: shifting art styles, collage textures, inventive transitions, playful visual surprises.",
}
DEFAULT_ART_STYLE = "Pixar-like 3D"
CHANNEL_NAME = "My YouTube Channel"    # legacy fallback (channels are managed in Settings now)

# Channel presets: the audience decides the content-safety rules fed to the LLM (and the
# YouTube made-for-kids flag); the video type decides animated vs realistic look.
AUDIENCES = ["kids", "family", "general"]
# animated = art-style driven; real = built-in live-action look; tech = live-action tech
# explainer look with the story-board text/image overlay switched on by default.
VIDEO_TYPES = ["animated", "real", "tech"]
REALISTIC_VIDEO_TYPES = ("real", "tech")     # no art-style picker, built-in realistic look

# Per-job soft target length presets (None = Auto, use the global settings).
# (low_seconds, high_seconds, clip_seconds): longer videos render longer clips, so the clip
# count - and with it the scene LLM calls and the GPU renders - stays around 25-30.
VIDEO_LENGTHS = {
    "Auto": None,
    "30-45 seconds": (30, 45, 5),
    "45-60 seconds": (45, 60, 5),
    "1-2 minutes": (60, 120, 5),
    "2-3 minutes": (120, 180, 6),
    "3-4 minutes": (180, 240, 8),
    "4-5 minutes": (240, 300, 10),
}
DEFAULT_VIDEO_LENGTH = "Auto"
SOURCES = ["url", "topic", "description", "transcript"]
DEFAULT_SOURCE = "url"


def video_length_range(name: str):
    v = VIDEO_LENGTHS.get(name)
    return (v[0], v[1]) if v else None


def video_length_clip_seconds(name: str) -> float:
    """Requested clip length for a target preset; CLIP_SECONDS for Auto and unknown names."""
    v = VIDEO_LENGTHS.get(name)
    return float(v[2]) if v and len(v) > 2 else float(CLIP_SECONDS)

# Thumbnail dimensions per aspect ratio (long side <= 1280)
THUMB_SEED_SECONDS = 3.0   # thumbnail render length when it is seeded from a cast portrait (last frame is kept)
THUMB_SIZES = {
    "16:9 (Widescreen)": (1280, 720),
    "9:16 (Portrait Widescreen)": (720, 1280),
    "1:1 (Square)": (1080, 1080),
    "4:3 (Standard)": (1280, 960),
    "3:4 (Portrait Standard)": (960, 1280),
    "3:2 (Photo)": (1200, 800),
    "2:3 (Portrait Photo)": (800, 1200),
    "21:9 (Ultrawide)": (1260, 540),
}

ASPECT_RATIOS = [
    "16:9 (Widescreen)", "9:16 (Portrait Widescreen)", "1:1 (Square)", "4:3 (Standard)",
    "3:4 (Portrait Standard)", "3:2 (Photo)", "2:3 (Portrait Photo)", "21:9 (Ultrawide)",
]
MEGAPIXELS_OPTIONS = [round(0.2 + 0.1 * i, 1) for i in range(9)]   # 0.2 … 1.0 (ResolutionSelector)
# Final-video output resolution: label -> short side in pixels the joined clips are upscaled to with
# ffmpeg (lanczos) during assembly; 0 = keep the clips' own size. Never downscales.
OUTPUT_RESOLUTIONS = {
    "Native (clip size)": 0,
    "1080p (Full HD)": 1080,
    "2K (1440p)": 1440,
    "4K (2160p)": 2160,
}

# ---- generation defaults (editable in Settings) ---------------------------
DEFAULT_ASPECT_RATIO = ASPECT_RATIOS[0]
DEFAULT_MEGAPIXELS = 0.4        # clip resolution in megapixels; per-video override in the New Video dialog (thumbnails: 1.0)
DEFAULT_RESOLUTION = "Native (clip size)"   # final-video upscale target (OUTPUT_RESOLUTIONS); per-video override
CLIP_SECONDS = 5                # length of each generated clip
LIGHTNING_LORA = False          # True -> 8-step turbo LoRA (faster, lower quality)
NARRATION_WORDS = 170           # target narration length (~65-75 s of speech)
CLIP_AUDIO_VOLUME = 0.3         # model-generated ambience level under the narration
CLIP_AUDIO_WITH_MUSIC = False   # narration+music: also keep the clips' own sound (off = voice + music only)
CLIP_RETRIES = 1
GPU_SCHEDULE_ENABLED = False    # ComfyUI renders only between GPU_WINDOW_START and GPU_WINDOW_END (server local time)
GPU_WINDOW_START = "22:00"      # HH:MM; end before start = overnight window
GPU_WINDOW_END = "06:00"
AUDIO_MODES = ["music", "narration", "narration+music"]
AUDIO_MODE = "music"            # default preselection for the New Video dialog audio choice
MUSIC_VOLUME = 0.25             # music level under the narration (narration+music mode, auto-ducked)
STORYBOARD_MAX_IMAGES = 6       # story board overlay: max related images rendered per video (0 = text only)
STORYBOARD_ACCENT = "#e9a13b"   # story board overlay accent colour (kicker text, side bar)
TARGET_VIDEO_SECONDS = 90       # music mode: requested song length (the model may end earlier)
MIN_TRANSCRIPT_WORDS = 50       # refuse videos whose transcript has fewer words
MIN_TRANSCRIPT_WPM = 20         # refuse when words-per-minute of caption coverage is below this
COMFY_CLEANUP = True            # delete each artifact from the ComfyUI server once downloaded
SCRIPT_REVIEW = True            # LLM double-checks every generated scene and drafted script
YT_DESCRIPTION_LIMIT = 4900     # YouTube caps descriptions at 5000; uploads are cut here
YT_PRIVACY = "private"          # privacy of uploaded YouTube videos
PUBLIC_BASE_URL = ""            # https://host users open the app with; "" = localhost OAuth redirect
YT_MADE_FOR_KIDS = False        # selfDeclaredMadeForKids on uploads (default: not made for kids)
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
WHISPER_MODEL = "small"         # faster-whisper model used to time the sung lyrics per clip
KIDS_SONG_BPM = 66              # tempo forced into every kids-channel song caption (nursery-rhyme pace)
KIDS_SONG_WPS = 0.9             # sung-words-per-second budget for kids songs (lyrics length + pace check)
SONG_VOICES = ["auto", "child", "girl", "boy", "female", "male"]   # per-channel default song voice
SONG_VOICE_LABELS = {"auto": "Auto (child for kids channels)", "child": "Young child",
                     "girl": "Young girl", "boy": "Young boy", "female": "Adult female", "male": "Adult male"}
TTS_TEMPERATURE = 0.8           # Fish Speech sampling temperature
TTS_TOP_P = 0.8                 # Fish Speech nucleus sampling
TTS_REPETITION_PENALTY = 1.1    # Fish Speech penalty against repeated tokens
TTS_CHUNK_CHARS = 300           # max characters per Fish Speech request (long text is split)

# ---- fixed limits ---------------------------------------------------------
MAX_TRANSCRIPT_CHARS = 60000    # ~15k tokens, leaves room in the 32k context
COMFY_TIMEOUT = 60 * 60         # seconds to wait for a single clip
LYRIC_ALIGN_MIN_RATIO = 0.55    # below this whisper/lyrics token match ratio the raw whisper words are kept
SONG_PACE_TOLERANCE = 1.4       # a song measured faster than budget * this is re-rendered once, slower
SONG_SHORT_TOLERANCE = 0.9      # a song shorter than the target range's low bound * this is re-rendered once, longer

# ---- clip length as the video model really renders it ----------------------
# The MiniMax H3 workflow (node 140:132) quantises the requested seconds to a frame count
# f = max(5, round(s * 24)) rounded UP to f ≡ 5 (mod 17): 5 s -> 124 frames = 5.1667 s.
# Everything that lines audio up with clips (lyric windows, scene timelines, the clip count)
# must use these real lengths, not CLIP_SECONDS.
VIDEO_FPS = 24
FRAME_QUANTUM = 17
MIN_LAST_CLIP_SECONDS = 2.0     # shortest trailing clip when the last clip is trimmed to the audio


def clip_frames(seconds: float) -> int:
    """Frame count the H3 workflow renders for a requested clip length (mirrors node 140:132)."""
    f = max(5, round(float(seconds) * VIDEO_FPS))
    return f + (5 - f % FRAME_QUANTUM) % FRAME_QUANTUM


def actual_clip_seconds(seconds: float | None = None) -> float:
    """Real length of a clip requested at `seconds` (default CLIP_SECONDS)."""
    return clip_frames(CLIP_SECONDS if seconds is None else seconds) / VIDEO_FPS


def clip_durations(total_seconds: float, clip_seconds: float | None = None) -> list:
    """Per-clip real lengths covering an audio track of total_seconds: full clips plus one
    trailing clip quantised from the remainder (at least MIN_LAST_CLIP_SECONDS), so the video
    ends with the audio instead of a silent tail.

    clip_seconds: requested length of a full clip (default CLIP_SECONDS); the per-job target
    length picks it (video_length_clip_seconds) so long videos do not become 60 clips."""
    full = actual_clip_seconds(clip_seconds)
    total = max(0.0, float(total_seconds))
    n = max(1, math.ceil(total / full - 1e-6))
    rem = total - (n - 1) * full
    if n > 1 and rem < 0.5:     # a few hundred ms of fade-out are not worth a whole extra clip
        return [full] * (n - 1)
    last = actual_clip_seconds(max(MIN_LAST_CLIP_SECONDS, rem))
    return [full] * (n - 1) + [last]


def boundaries(durations: list) -> list:
    """Cumulative end times of consecutive clips (the last entry is the total length)."""
    out, t = [], 0.0
    for d in durations:
        t += float(d)
        out.append(round(t, 4))
    return out

# ---------------------------------------------------------------------------
EDITABLE_SETTINGS = {
    # name: (caster, validator, label, help)
    "COMFY_URL": (str, lambda v: v.startswith("http"), "ComfyUI URL", "Base URL of the ComfyUI server"),
    "VLLM_URL": (str, lambda v: v.startswith("http"), "vLLM URL", "OpenAI-compatible base URL (…/v1)"),
    "VLLM_MODEL": (str, lambda v: len(v) > 0, "vLLM model id", "Model name served by vLLM"),
    "FISH_URL": (str, lambda v: v.startswith("http"), "Fish TTS URL", "Gradio URL of Fish Speech"),
    "TTS_TEMPERATURE": (float, lambda v: 0.1 <= v <= 2.0, "TTS temperature",
                        "Fish Speech sampling temperature (0.1-2)"),
    "TTS_TOP_P": (float, lambda v: 0.1 <= v <= 1.0, "TTS top-p",
                  "Fish Speech nucleus sampling (0.1-1)"),
    "TTS_REPETITION_PENALTY": (float, lambda v: 0.9 <= v <= 2.0, "TTS repetition penalty",
                               "Fish Speech penalty against repeated tokens (0.9-2)"),
    "TTS_CHUNK_CHARS": (int, lambda v: 100 <= v <= 500, "TTS chunk length (chars)",
                        "Narration is split into chunks of at most this many characters per Fish request (100-500)"),
    "DEFAULT_SOURCE": (str, lambda v: v in SOURCES, "Default source (Create from)",
                       "Preselected creation mode in the New Video dialog: url, topic, description or transcript"),
    "CLIP_SECONDS": (float, lambda v: 2 <= v <= 15, "Clip length (s)",
                     "Requested duration of each generated clip (2-15 s). The video model rounds it to an "
                     "allowed frame count (5 s renders as 5.17 s); lyric timing and scenes use that real length"),
    "WHISPER_MODEL": (str, lambda v: v in WHISPER_MODELS, "Whisper model (lyric timing)",
                      "faster-whisper size used on the CPU to time the sung lyrics: small is a good default, "
                      "medium is slower but more accurate. The first use downloads the model"),
    "KIDS_SONG_BPM": (int, lambda v: 50 <= v <= 110, "Kids song tempo (BPM)",
                      "Nursery-rhyme tempo forced into every kids-channel song caption (60-72 recommended)"),
    "KIDS_SONG_WPS": (float, lambda v: 0.4 <= v <= 2.0, "Kids song words/second",
                      "Sung-words-per-second budget for kids songs (sizes the lyrics); a song measured 40% faster "
                      "than this is re-rendered once with a slower caption and fewer words"),
    "DEFAULT_MEGAPIXELS": (float, lambda v: any(abs(v - o) < 1e-6 for o in MEGAPIXELS_OPTIONS),
                           "Clip resolution (megapixels)",
                           "Default ResolutionSelector megapixels for every clip (0.2-1.0 in 0.1 steps); "
                           "can be changed per video in the New Video dialog. Thumbnails always render at 1.0"),
    "DEFAULT_RESOLUTION": (str, lambda v: v in OUTPUT_RESOLUTIONS, "Output resolution",
                           "Size of the final video: after the clips are joined they are upscaled with ffmpeg "
                           "(lanczos) to this short side - 1080p, 2K (1440p) or 4K (2160p); Native keeps the clips' "
                           "own size. Upscaling adds no detail, only pixels, and the 2K/4K encode takes a few extra "
                           "minutes on the CPU. Can be changed per video in the New Video dialog"),
    "NARRATION_WORDS": (int, lambda v: 30 <= v <= 500, "Narration words",
                        "Target narration length; ~2.5 words/s of speech decides the clip count"),
    "CLIP_AUDIO_VOLUME": (float, lambda v: 0 <= v <= 1, "Clip ambience volume",
                          "Volume of the clips' own audio under the narration (0-1); it is dipped at every "
                          "clip boundary and ducked while the voice speaks"),
    "CLIP_AUDIO_WITH_MUSIC": (bool, lambda v: True, "Keep clip ambience under music",
                              "Voiceover + music videos also mix in the clips' own generated sound. Off (default) = "
                              "narration + music bed only, which avoids the three-layer clash"),
    "LIGHTNING_LORA": (bool, lambda v: True, "Lightning LoRA (fast preview)",
                       "8-step turbo sampling: ~3x faster, lower quality"),
    "CLIP_RETRIES": (int, lambda v: 0 <= v <= 5, "Clip retries",
                     "Retries per failed clip before failing the job"),
    "GPU_SCHEDULE_ENABLED": (bool, lambda v: True, "GPU schedule",
                             "Only submit ComfyUI renders (clips, music, thumbnails, story board images) between "
                             "GPU_WINDOW_START and GPU_WINDOW_END; script writing and narration run any time"),
    "GPU_WINDOW_START": (str, lambda v: bool(re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v)),
                         "GPU window start (HH:MM)",
                         "Server local time the render window opens, e.g. 22:00"),
    "GPU_WINDOW_END": (str, lambda v: bool(re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v)),
                       "GPU window end (HH:MM)",
                       "Server local time the render window closes, e.g. 06:00 (before the start = overnight; "
                       "equal to the start = always open)"),
    "MUSIC_VOLUME": (float, lambda v: 0 <= v <= 1, "Music volume",
                     "Music level under the narration (narration+music mode, 0-1). The bed is ducked further "
                     "while the voice speaks, looped if shorter than the video and faded out at the end"),
    "STORYBOARD_MAX_IMAGES": (int, lambda v: 0 <= v <= 40, "Story board: max images",
                              "Story board overlay: at most this many related images are rendered per video "
                              "(each is one short ComfyUI render); 0 = text cards only"),
    "STORYBOARD_ACCENT": (str, lambda v: bool(re.fullmatch(r"#[0-9a-fA-F]{6}", v)),
                          "Story board: accent colour",
                          "Hex colour of the story board cards' accent (kicker label and side bar), e.g. #e9a13b"),
    "TARGET_VIDEO_SECONDS": (float, lambda v: v == 0 or 15 <= v <= 300, "Target video length (s)",
                             "music mode: requested song length; 0 = no target, the song covers the full content"),
    "MIN_TRANSCRIPT_WORDS": (int, lambda v: 0 <= v <= 2000, "Min transcript words",
                             "Refuse videos whose transcript has fewer words than this"),
    "MIN_TRANSCRIPT_WPM": (float, lambda v: 0 <= v <= 200, "Min transcript words/min",
                           "Refuse when the transcript is too sparse for the video length (music/no speech)"),
    "YT_PRIVACY": (str, lambda v: v in ("private", "unlisted", "public"), "YouTube privacy",
                   "Privacy of uploaded videos: private (review in YouTube Studio first), unlisted or public"),
    "PUBLIC_BASE_URL": (str, lambda v: v == "" or v.startswith("http"), "Public base URL (YouTube OAuth)",
                        "Leave empty (default): Google returns to http://localhost:<port>/api/youtube/callback, "
                        "which Desktop-type OAuth clients allow; from another computer you paste that address "
                        "into Settings to finish. Set to the HTTPS address users open the app with (e.g. "
                        "https://videogen.example.com) together with a Web-type OAuth client that has "
                        "<url>/api/youtube/callback registered for a fully automatic sign-in"),
    "SCRIPT_REVIEW": (bool, lambda v: True, "LLM review pass",
                      "After generating each scene (and each drafted script), the LLM reviews it once more against the audio/content and fixes inconsistencies (wrong letters, mismatched objects, safety issues)"),
    "COMFY_CLEANUP": (bool, lambda v: True, "Clean up ComfyUI",
                      "Delete each job's rendered files, uploaded frames and history from the ComfyUI server as soon as they are downloaded"),
}


def current_settings() -> dict:
    return {k: globals()[k] for k in EDITABLE_SETTINGS}


def load_overrides():
    # a relocated settings file that does not exist yet falls back to the repo-root one
    # (one-time carry-over; the next save writes to SETTINGS_FILE)
    path = SETTINGS_FILE if SETTINGS_FILE.exists() else LEGACY_SETTINGS_FILE
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    for k, v in data.items():
        spec = EDITABLE_SETTINGS.get(k)
        if not spec:
            continue
        try:
            v = spec[0](v)
            if spec[1](v):
                globals()[k] = v
        except Exception:  # noqa: BLE001
            pass


def save_settings(data: dict) -> dict:
    """Validate + persist the given subset of settings, apply them, return the full set."""
    clean = {}
    for k, v in data.items():
        spec = EDITABLE_SETTINGS.get(k)
        if not spec:
            continue
        caster, valid = spec[0], spec[1]
        try:
            v = caster(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k}: invalid value {v!r}")
        if not valid(v):
            raise ValueError(f"{k}: value {v!r} out of range")
        clean[k] = v
    existing = {}
    src = SETTINGS_FILE if SETTINGS_FILE.exists() else LEGACY_SETTINGS_FILE
    if src.is_file():
        try:
            existing = json.loads(src.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    existing.update(clean)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    load_overrides()
    return current_settings()


def _apply_env_overrides():
    """VIDEOGEN_<SETTING> environment variables override settings.json and defaults
    (useful for Docker deployments)."""
    for k, spec in EDITABLE_SETTINGS.items():
        raw = os.environ.get(f"VIDEOGEN_{k}")
        if raw is None:
            continue
        caster, valid = spec[0], spec[1]
        try:
            v = raw.strip().lower() in ("1", "true", "yes", "on") if caster is bool else caster(raw)
            if valid(v):
                globals()[k] = v
        except Exception:  # noqa: BLE001
            pass


load_overrides()
_apply_env_overrides()
