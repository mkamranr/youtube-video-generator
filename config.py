"""Configuration: constants + user-editable settings (persisted to settings.json).

Values in EDITABLE_SETTINGS can be changed at runtime from the Settings page; they are
stored in settings.json and applied on top of the defaults below at import and on save.
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = BASE_DIR / "workflows"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
SETTINGS_FILE = BASE_DIR / "settings.json"

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
CHANNEL_NAME = "Habibi Kids Channel"   # the channel these videos are published on

# Per-job soft target length presets (None = Auto, use the global settings)
VIDEO_LENGTHS = {
    "Auto": None,
    "30-45 seconds": (30, 45),
    "45-60 seconds": (45, 60),
    "1-2 minutes": (60, 120),
    "2-3 minutes": (120, 180),
}
DEFAULT_VIDEO_LENGTH = "Auto"


def video_length_range(name: str):
    return VIDEO_LENGTHS.get(name)

ASPECT_RATIOS = [
    "16:9 (Widescreen)", "9:16 (Portrait Widescreen)", "1:1 (Square)", "4:3 (Standard)",
    "3:4 (Portrait Standard)", "3:2 (Photo)", "2:3 (Portrait Photo)", "21:9 (Ultrawide)",
]

# ---- generation defaults (editable in Settings) ---------------------------
DEFAULT_ASPECT_RATIO = ASPECT_RATIOS[0]
CLIP_SECONDS = 5                # length of each generated clip
LIGHTNING_LORA = False          # True -> 8-step turbo LoRA (faster, lower quality)
NARRATION_WORDS = 170           # target narration length (~65-75 s of speech)
CLIP_AUDIO_VOLUME = 0.3         # model-generated ambience level under the narration
CLIP_RETRIES = 1
AUDIO_MODES = ["music", "narration", "narration+music"]
AUDIO_MODE = "music"            # music: the song is the only audio and drives the video length
MUSIC_VOLUME = 0.25             # music level under the narration (narration+music mode)
TARGET_VIDEO_SECONDS = 90       # music mode: requested song length (the model may end earlier)
MIN_TRANSCRIPT_WORDS = 50       # refuse videos whose transcript has fewer words
MIN_TRANSCRIPT_WPM = 20         # refuse when words-per-minute of caption coverage is below this
COMFY_CLEANUP = True            # delete each artifact from the ComfyUI server once downloaded
SCRIPT_REVIEW = True            # LLM double-checks every generated scene and drafted script
WHISPER_MODEL = "base"          # faster-whisper model used to time the sung lyrics per clip

# ---- fixed limits ---------------------------------------------------------
MAX_TRANSCRIPT_CHARS = 60000    # ~15k tokens, leaves room in the 32k context
COMFY_TIMEOUT = 60 * 60         # seconds to wait for a single clip

# ---------------------------------------------------------------------------
EDITABLE_SETTINGS = {
    # name: (caster, validator, label, help)
    "COMFY_URL": (str, lambda v: v.startswith("http"), "ComfyUI URL", "Base URL of the ComfyUI server"),
    "VLLM_URL": (str, lambda v: v.startswith("http"), "vLLM URL", "OpenAI-compatible base URL (…/v1)"),
    "VLLM_MODEL": (str, lambda v: len(v) > 0, "vLLM model id", "Model name served by vLLM"),
    "FISH_URL": (str, lambda v: v.startswith("http"), "Fish TTS URL", "Gradio URL of Fish Speech"),
    "DEFAULT_ASPECT_RATIO": (str, lambda v: v in ASPECT_RATIOS, "Default aspect ratio",
                             "Preselected aspect ratio for new jobs"),
    "DEFAULT_ART_STYLE": (str, lambda v: v in ART_STYLES, "Default art style",
                          "Preselected art style for new jobs"),
    "CHANNEL_NAME": (str, lambda v: 0 < len(v.strip()) <= 80, "Channel name",
                     "Videos are branded for this channel; any channel or brand name in the source content is replaced with it"),
    "CLIP_SECONDS": (float, lambda v: 2 <= v <= 15, "Clip length (s)",
                     "Duration of each generated clip (2-15 s)"),
    "NARRATION_WORDS": (int, lambda v: 30 <= v <= 500, "Narration words",
                        "Target narration length; ~2.5 words/s of speech decides the clip count"),
    "CLIP_AUDIO_VOLUME": (float, lambda v: 0 <= v <= 1, "Clip ambience volume",
                          "Volume of the clips' own audio under the narration (0-1)"),
    "LIGHTNING_LORA": (bool, lambda v: True, "Lightning LoRA (fast preview)",
                       "8-step turbo sampling: ~3x faster, lower quality"),
    "CLIP_RETRIES": (int, lambda v: 0 <= v <= 5, "Clip retries",
                     "Retries per failed clip before failing the job"),
    "AUDIO_MODE": (str, lambda v: v in AUDIO_MODES, "Audio mode",
                   "music: song only, synced per clip; narration: voiceover only; narration+music: both"),
    "MUSIC_VOLUME": (float, lambda v: 0 <= v <= 1, "Music volume",
                     "Music level under the narration (narration+music mode, 0-1)"),
    "TARGET_VIDEO_SECONDS": (float, lambda v: v == 0 or 15 <= v <= 300, "Target video length (s)",
                             "music mode: requested song length; 0 = no target, the song covers the full content"),
    "MIN_TRANSCRIPT_WORDS": (int, lambda v: 0 <= v <= 2000, "Min transcript words",
                             "Refuse videos whose transcript has fewer words than this"),
    "MIN_TRANSCRIPT_WPM": (float, lambda v: 0 <= v <= 200, "Min transcript words/min",
                           "Refuse when the transcript is too sparse for the video length (music/no speech)"),
    "SCRIPT_REVIEW": (bool, lambda v: True, "LLM review pass",
                      "After generating each scene (and each drafted script), the LLM reviews it once more against the audio/content and fixes inconsistencies (wrong letters, mismatched objects, safety issues)"),
    "COMFY_CLEANUP": (bool, lambda v: True, "Clean up ComfyUI",
                      "Delete each job's rendered files, uploaded frames and history from the ComfyUI server as soon as they are downloaded"),
}


def current_settings() -> dict:
    return {k: globals()[k] for k in EDITABLE_SETTINGS}


def load_overrides():
    if not SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
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
    if SETTINGS_FILE.exists():
        try:
            existing = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    existing.update(clean)
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
