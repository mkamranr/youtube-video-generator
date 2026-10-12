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
MODELS_DIR = Path(os.environ.get("VIDEOGEN_MODELS_DIR") or BASE_DIR / "models")  # local CPU models (avatar matting)
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

# ---- video models ---------------------------------------------------------
def _quantise(frames, quantum: int, rest: int) -> int:
    """Smallest f >= frames with f % quantum == rest % quantum."""
    f = int(frames)
    return f + (rest - f) % quantum


# One entry per ComfyUI video workflow family the app can render clips with. A model carries its
# own frame arithmetic - how its graph turns requested seconds into a frame count, and what its
# duration node wants to be handed - plus a node map per variant: "t2v" for a clip rendered from
# text alone and "i2v" for one seeded with a first frame. The two variants are separate graphs
# with their own node numbering, so the map belongs to the variant, not the model.
#
#   file       - template file under WORKFLOWS_DIR
#   prompt     - (node, input field) carrying the positive prompt
#   negative   - the same for the negative prompt, or None when the graph has none
#   seeds      - every noise node, in sampling order; each is given its own seed
#   duration   - (node, input field) carrying the clip length
#   save       - SaveVideo node (filename_prefix)
#   resolution - ResolutionSelector node (aspect_ratio + megapixels)
#   load_image - LoadImage node, or None for a text-to-video graph
#   lightning  - Lightning LoRA boolean, or None when the model has no such switch
#
# Nothing model-specific lives outside this dict: a third model is one new entry plus its JSON.
VIDEO_MODELS = {
    "MiniMax H3": {
        "fps": 24,
        # node 140:132: max(5, round(a * 24)), raised to the next f = 5 (mod 17)
        "frames": lambda s, fps: _quantise(max(5, round(float(s) * fps)), 17, 5),
        # a float of seconds: the graph quantises it itself, so it round-trips unchanged
        "duration_value": lambda s, frames, fps: float(s),
        "t2v": {"file": "video_minimax_h3_t2v.json",
                "prompt": ("140:131", "prompt"), "negative": None,
                "seeds": (("140:129", "noise_seed"),),
                "duration": ("140:133", "value"),
                "save": "92", "resolution": "115",
                "load_image": None, "lightning": "140:139"},
        "i2v": {"file": "video_minimax_h3_t2v_first_frame.json",
                "prompt": ("140:131", "prompt"), "negative": None,
                "seeds": (("140:129", "noise_seed"),),
                "duration": ("140:133", "value"),
                "save": "92", "resolution": "115",
                "load_image": "141", "lightning": "140:139"},
    },
    "LTX 2.5": {
        "fps": 24,
        # node 398:378 / 405:378: duration * 24 + 1, i.e. the next f = 1 (mod 24)
        "frames": lambda s, fps: _quantise(max(fps + 1, math.ceil(float(s) * fps - 1)), fps, 1),
        # a PrimitiveInt of whole seconds, derived from the frame count rather than passed
        # straight through, so the node reproduces exactly the length the planner froze
        "duration_value": lambda s, frames, fps: int(round((frames - 1) / fps)),
        "t2v": {"file": "video_ltx_2_5_t2v.json",
                "prompt": ("405:376", "value"),       # PrimitiveStringMultiline
                "negative": ("405:373", "text"),      # CLIPTextEncode
                "seeds": (("405:339", "noise_seed"), ("405:338", "noise_seed")),
                "duration": ("405:362", "value"),     # PrimitiveInt: whole seconds only
                "save": "75", "resolution": "409",
                "load_image": None, "lightning": None},
        "i2v": {"file": "video_ltx_2_5_i2v.json",
                "prompt": ("398:376", "value"),
                "negative": ("398:373", "text"),
                "seeds": (("398:339", "noise_seed"), ("398:338", "noise_seed")),
                "duration": ("398:362", "value"),
                "save": "75", "resolution": "403",
                "load_image": "395", "lightning": None},
    },
}


def video_model(name: str = "") -> dict:
    """A model's registry entry; the Settings default for "" (legacy jobs) and unknown names."""
    return VIDEO_MODELS.get(name or DEFAULT_VIDEO_MODEL) or VIDEO_MODELS[DEFAULT_VIDEO_MODEL]


def model_variant(name: str = "", first_frame: bool = False) -> dict:
    """The node map for the graph this render needs: the first-frame one when a seed frame was
    given, the text-to-video one otherwise."""
    return video_model(name)["i2v" if first_frame else "t2v"]


def model_workflow(name: str = "", first_frame: bool = False) -> Path:
    return WORKFLOWS_DIR / model_variant(name, first_frame)["file"]


# Thumbnails, cast portraits, story board images and the art-style previews are 5-frame stills at
# 1 MP - a length only the H3 graph can express, since LTX's duration node is whole seconds - so
# they stay on MiniMax H3 whatever model a video picked. These three are for those stills only.
STILLS_VIDEO_MODEL = "MiniMax H3"
T2V_WORKFLOW = WORKFLOWS_DIR / VIDEO_MODELS[STILLS_VIDEO_MODEL]["t2v"]["file"]
I2V_WORKFLOW = WORKFLOWS_DIR / VIDEO_MODELS[STILLS_VIDEO_MODEL]["i2v"]["file"]

# ---- avatar (talking-head presenter) engines -------------------------------
# A channel can upload presenter portraits; the chosen one is animated to lip-sync the
# voiceover and composited over the video. Two engines, with very different economics:
#
#   Wan 2.2 S2V - purpose-built audio-to-video. Holds the uploaded likeness. Measured at
#                 9.16 s per output frame at 640x640/20 steps (warm; a cold run was the
#                 same, so the weight load is not the cost). Renders in fixed 77-frame
#                 chunks at 16 fps, chained with WanSoundImageToVideoExtend, and its first
#                 frame is the ref-image conditioning slot and must be dropped.
#   LTX 2.5     - the clip model, run as audio-to-video by pinning our narration into the
#                 audio half of its joint AV latent (noise_mask 0 = keep) and masking the
#                 video half for generation. ~33x faster at 0.28 s/frame, 24 fps, one pass
#                 for the whole segment - but the face drifts off the uploaded portrait.
#
# As with VIDEO_MODELS, no node id lives outside this dict: ComfyClient.build_avatar reads
# everything from here, so a third engine is one entry plus its JSON.
AVATAR_MODELS = {
    "Wan 2.2 S2V": {
        "file": "avatar_wan_s2v.json",
        "fps": 16,
        # Frames per pass. The model was trained on this length and the graph renders exactly
        # one chunk, so a segment is capped here rather than chained: WanSoundImageToVideoExtend
        # exists for longer takes, but its latent stitching is unverified and each experiment
        # costs a ~6 min render. 4.81 s of presenter per appearance is plenty for an intro.
        "chunk": 77,
        "max_segment": 77 / 16.0,
        "square": 640,          # the size it was trained at; also what the cut-out look needs
        "seconds_per_frame": 9.16,
        "drop_first": True,     # frame 0 carries the ref image, not generated picture
        "audio": ("5", "audio"),
        "image": ("7", "image"),
        "prompt": ("8", "text"),
        "seeds": (("12", "seed"),),
        "steps": ("12", "steps"),
        # 20, not 10. Halving the steps does halve the cost (359 s vs 705 s for the same seed),
        # but it also halves the lip sync: correlation against the speech envelope falls from
        # +0.559 to +0.319 and mouth-open-in-speech-vs-silence from 1.47x to 1.16x. Sync is the
        # entire point of the feature, so this is the one knob not to turn.
        "default_steps": 20,
        "length": ("11", "length"),          # in frames
        "width": ("11", "width"), "height": ("11", "height"),
        "save": "15",
    },
    "LTX 2.5": {
        "file": "avatar_ltx_2_5_a2v.json",
        "fps": 24,
        "chunk": 0,             # 0 = no chunking, one pass covers the segment
        "max_segment": 0.0,     # 0 = unlimited; LTX renders the whole segment at once
        "square": 512,
        "seconds_per_frame": 0.28,
        "drop_first": False,
        "audio": ("900", "audio"),
        "image": ("395", "image"),
        "prompt": ("398:376", "value"),
        "negative": ("398:373", "text"),
        "seeds": (("398:339", "noise_seed"), ("398:338", "noise_seed")),
        "steps": None,          # distilled: the step count is baked into its sigma schedule
        "default_steps": 0,
        "duration": ("398:362", "value"),    # PrimitiveInt, whole seconds
        "resolution": "403",                 # ResolutionSelector (aspect_ratio + megapixels)
        "save": "75",
    },
}
DEFAULT_AVATAR_MODEL = "Wan 2.2 S2V"

# Where the presenter appears. "none" is the off switch - one control rather than a separate
# on/off flag that could contradict it.
AVATAR_COVERAGES = {
    "none": "No avatar",
    "intro": "Intro only",
    "outro": "Outro only",
    "both": "Intro and outro",
    "full": "Whole video",
}
DEFAULT_AVATAR_COVERAGE = "both"
AVATAR_STYLES = {"bubble": "Circle bubble", "cutout": "Cut-out (no background)"}
DEFAULT_AVATAR_STYLE = "cutout"
AVATAR_INTRO_SECONDS = 12.0     # how long the presenter speaks at the top
AVATAR_OUTRO_SECONDS = 12.0     # ...and at the end
# Two caps, because they guard different things. MAX_SECONDS is how much presenter a viewer
# should see; MAX_RENDER_MINUTES is how much of the shared GPU a video may spend getting it.
# They are not interchangeable: 24 s of footage costs ~70 min on Wan and ~2.5 min on LTX.
AVATAR_MAX_SECONDS = 30.0       # footage cap per video
AVATAR_MAX_RENDER_MINUTES = 45.0  # wall-clock cap; coverage trims to fit, with the reason logged
AVATAR_SIZE_PCT = 0.22          # bubble diameter as a fraction of the frame's SHORT side
AVATAR_CUTOUT_PCT = 0.58        # cut-out height as a fraction of the frame HEIGHT
AVATAR_MAX_WIDTH_PCT = 0.60     # ...but never wider than this much of the frame WIDTH, which is
                                # what keeps a 9:16 Short from being filled by the presenter.
                                # On a portrait frame this is the setting that actually sizes
                                # the cut-out: the engines render a square, so bw == bh and the
                                # height above is always the wider of the two - AVATAR_CUTOUT_PCT
                                # only bites on 16:9.
AVATAR_MARGIN_PCT = 0.035       # inset from the frame edge, fraction of the short side
AVATAR_CORNER = "bottom-right"
AVATAR_CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")
AVATAR_MATTE_MODEL = "u2net_human_seg.onnx"   # CPU ONNX matting, in MODELS_DIR
# Portraits are stored square and at most this wide: the squaring is done once at upload (it
# needs a matte to find the head, which is too slow to repeat per video) and 1024 leaves room
# above the 640 the biggest engine renders, so a better model later costs no re-upload.
AVATAR_STORE_PX = 1024
# What the engine is told the presenter is doing. Deliberately generic: the portrait carries
# the likeness, and describing a face the model can already see only fights the reference.
AVATAR_PROMPT = ("A person speaks directly to the camera, talking naturally and expressively "
                 "to the viewer, slight natural head movement, steady framing, plain background.")


def avatar_model(name: str = "") -> dict:
    """An engine's registry entry; the Settings default for "" and unknown names."""
    return AVATAR_MODELS.get(name or DEFAULT_AVATAR_MODEL) or AVATAR_MODELS[DEFAULT_AVATAR_MODEL]


def avatar_workflow(name: str = "") -> Path:
    return WORKFLOWS_DIR / avatar_model(name)["file"]


def avatar_frames(seconds: float, name: str = "") -> int:
    """Frames this engine renders for `seconds` of speech, rounded to whole chunks where the
    engine has a fixed chunk size (Wan), so the estimate matches what actually runs."""
    spec = avatar_model(name)
    want = max(1, int(round(float(seconds) * spec["fps"])))
    chunk = spec.get("chunk") or 0
    if chunk:
        return max(chunk, -(-want // chunk) * chunk)
    return want


def avatar_estimate_seconds(seconds: float, name: str = "") -> float:
    """Wall-clock estimate for rendering `seconds` of presenter on this engine."""
    spec = avatar_model(name)
    return avatar_frames(seconds, name) * float(spec["seconds_per_frame"])


def avatar_coverage_seconds(coverage: str, name: str = "", total: float = 0.0) -> float:
    """Seconds of presenter a coverage asks for, clamped to what the engine can afford.

    `total` is the finished video's length, known only once the narration exists; without it
    "full" is quoted at the budget, which is what it will be trimmed to anyway on any video
    longer than the budget."""
    budget = avatar_budget_seconds(name)
    if coverage == "none" or budget <= 0:
        return 0.0
    if coverage == "full":
        want = float(total) if total else budget
    elif coverage == "intro":
        want = float(AVATAR_INTRO_SECONDS)
    elif coverage == "outro":
        want = float(AVATAR_OUTRO_SECONDS)
    elif coverage == "both":
        want = float(AVATAR_INTRO_SECONDS) + float(AVATAR_OUTRO_SECONDS)
    else:
        return 0.0
    return max(0.0, min(want, budget))


def avatar_budget_seconds(name: str = "") -> float:
    """How many seconds of presenter this engine can render inside both caps.

    The footage cap is a product decision and the render cap is a resource one, so the
    affordable length is whichever binds first. On a slow engine that is the render cap,
    which is exactly how a 70-minute job becomes a 45-minute one without anyone noticing
    too late."""
    spec = avatar_model(name)
    spf, fps, chunk = float(spec["seconds_per_frame"]), spec["fps"], spec.get("chunk") or 0
    frames = int(float(AVATAR_MAX_RENDER_MINUTES) * 60.0 / max(1e-6, spf))
    if chunk:
        # whole chunks only, rounded DOWN: avatar_frames() rounds a request up to the next
        # chunk, so budgeting a partial one would quietly overshoot the cap it just enforced
        frames = (frames // chunk) * chunk
    return max(0.0, min(float(AVATAR_MAX_SECONDS), frames / float(fps)))


# ---- music workflow and its node ids (fixed) ------------------------------
MUSIC_WORKFLOW = WORKFLOWS_DIR / "audio_minimax_music_3.json"
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


def ltx_negative(video_type: str = "") -> str:
    """The LTX negative prompt for a channel: the anti-cartoon terms only where they help."""
    return LTX_NEGATIVE_REAL if video_type in REALISTIC_VIDEO_TYPES else LTX_NEGATIVE_ANIMATED


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
DEFAULT_VIDEO_MODEL = "MiniMax H3"   # which VIDEO_MODELS entry renders the clips; per-video
                                     # override in the New Video dialog. Stills (thumbnails, cast
                                     # portraits, story board images) always use MiniMax H3.
# LTX 2.5 takes a negative prompt (MiniMax H3 has no such input). The anti-cartoon terms help a
# live-action channel and actively fight an animated one, so the two are kept apart.
LTX_NEGATIVE_ANIMATED = ("blurry, low quality, distorted, deformed, watermark, text, subtitles, "
                         "logo, black frames, static image")
LTX_NEGATIVE_REAL = LTX_NEGATIVE_ANIMATED + ", cartoon, anime, childish, video game, overly CG look"
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
# Per-video CONTENT FORMAT, offered on kids channels. "song" is every video made so far; "story"
# tells a story whose voiceover is nothing but the spoken lines of that story (services/story.py).
# Not the same thing as a channel's VIDEO_TYPES (animated / real / tech), which is how it looks.
VIDEO_MODES = ["song", "story"]
DEFAULT_VIDEO_MODE = "song"
STORY_WPS = 2.2                 # spoken words per second each story line is sized for
STORY_LINE_LEAD = 0.2           # silence before every line, so no word starts exactly on a clip cut
STORY_LINE_GAP = 0.6            # silence held after every line, before its clip ends
STORY_MAX_CLIP_SECONDS = 12.0   # a line speaking longer than this is shortened once and re-spoken
STORY_MIN_BEATS = 4             # fewest scenes a story is told in
STORY_MAX_BEATS = 40            # most scenes a story is told in: one LLM call and one GPU
                                # render per scene, so this is a cost ceiling, not a style
STORY_NARRATOR = "Narrator"     # speaker of a line that no cast character says out loud
MAX_STORY_LINE_CHARS = 400      # one spoken line, as edited on the approval screen (above
                                # TTS_CHUNK_CHARS, so a long one is at most two requests)
MUSIC_VOLUME = 0.25             # music level under the narration (narration+music mode, auto-ducked)
STORYBOARD_MAX_IMAGES = 6       # story board overlay: max related images rendered per video (0 = text only)
STORYBOARD_ACCENT = "#e9a13b"   # story board overlay accent colour (kicker text, side bar)
# Which card layouts a video may draw from. () = every template in services/cardtemplates.TEMPLATES;
# otherwise the allowed template names. The set is read at PLANNING time only - each card's pick is
# frozen into plan.json, so changing this never restyles a finished video (regenerate its story
# board for that). Names, not families: a family-level set could not express "classic only", since
# the classic family still alternates two variants.
STORYBOARD_TEMPLATE_SETS = {
    "Mixed (all layouts)": (),
    "Panels only (no full-bleed)": ("lower_third", "ribbon", "big_stat", "split", "photo_stack"),
    "Classic lower third only": ("lower_third",),
}
STORYBOARD_TEMPLATE_SET = "Mixed (all layouts)"
STORYBOARD_MOTION = True        # cards ease into place instead of only fading in
# Thumbnail overlay text. The generated picture never contains letters (image models spell badly,
# and the prompt forbids typography), so the line is drawn with Pillow afterwards - which is what
# makes the variants free: one GPU render, several text treatments over copies of it.
THUMB_TEXT_ENABLED = True       # master switch, like CAPTIONS_ENABLED; per channel and per video
THUMB_TEXT_VARIANTS = 3         # text versions offered on the job page (0 = the clean render only)
THUMB_TEXT_ACCENT = ""          # "" = use STORYBOARD_ACCENT
# Burned-in captions. The words come from the script itself: the narration is timed with whisper
# and snapped back onto the written text, so captions show what was written, not what whisper
# thought it heard. Story and music videos are already timed and pay nothing.
CAPTION_HIGHLIGHTS = ["off", "word"]
# MASTER switch: read at assembly time, not frozen into a job, so turning it off and reassembling
# strips the captions from a video that already chose them. Per channel and per video below it.
CAPTIONS_ENABLED = True
CAPTION_MAX_WORDS = 3
CAPTION_MAX_CHARS = 26
CAPTION_HIGHLIGHT = "word"
CAPTION_ACCENT = "#2ba1e9"
CAPTION_FONT_SCALE = 1.0
CAPTION_BOTTOM_MARGIN = 0.10    # fraction of the frame height; portrait never goes below 0.14
# Retention furniture: a hook card over the opening, a subscribe card at the end, a progress bar
# and a whoosh as each card appears. All of them are layers in the overlay pass that already runs
# plus one audio input, so none of them costs an extra encode.
POLISH = True                   # default for a new channel's videos (per channel, per video)
HOOK_CARD_SECONDS = 3.0
END_CARD_SECONDS = 4.0
END_CARD_TEXT = "Subscribe for more"
PROGRESS_BAR_HEIGHT = 0.6       # percent of the frame height (0 = off)
SFX_VOLUME = 0.3                # card whoosh level (0 = silent)
SFX_FILE = ""                   # your own whoosh; empty = synthesized with ffmpeg once and cached
LOUDNESS_LUFS = -14.0           # EBU R128 target; YouTube normalises to about this (0 = off)
TARGET_VIDEO_SECONDS = 90       # music mode: requested song length (the model may end earlier)
MIN_TRANSCRIPT_WORDS = 50       # refuse videos whose transcript has fewer words
MIN_TRANSCRIPT_WPM = 20         # refuse when words-per-minute of caption coverage is below this
COMFY_CLEANUP = True            # delete each artifact from the ComfyUI server once downloaded
SCRIPT_REVIEW = True            # LLM double-checks every generated scene and drafted script
YT_DESCRIPTION_LIMIT = 4900     # YouTube caps descriptions at 5000; uploads are cut here
# The category an upload is filed under. 24 (Entertainment) used to be hardcoded, which mis-files
# every technical video; a channel whose video type is "tech" now goes to 28 automatically.
YT_CATEGORIES = {"1": "Film & Animation", "10": "Music", "17": "Sport", "20": "Gaming",
                 "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
                 "26": "Howto & Style", "27": "Education", "28": "Science & Technology"}
YT_CATEGORY_ID = "24"
YT_CATEGORY_BY_TYPE = {"tech": "28"}
YT_MAX_TAGS = 20                # how many SEO keywords are uploaded as video tags (0 = none)
YT_CHAPTERS = True              # add a timestamp list to the description
YT_CHAPTER_MIN_SECONDS = 20.0   # shortest chapter; YouTube's own floor is 10 s
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
MIN_HEARD_WORDS = 8             # an uploaded track with fewer words heard in the whole song counts as
                                # instrumental (whisper hallucinates a word or two over music), and its
                                # scenes follow the story instead of the empty per-clip lyric windows
SONG_PACE_TOLERANCE = 1.4       # a song measured faster than budget * this is re-rendered once, slower
SONG_SHORT_TOLERANCE = 0.9      # a song shorter than the target range's low bound * this is re-rendered once, longer
# User-supplied song lyrics (kids channels): the sheet is sung verbatim, so it is rejected rather
# than trimmed - a clamped sheet would lose a verse while the feature promises the user's own words.
MAX_SONG_SECONDS = 360.0        # MiniMax Music3 max_duration ceiling (mirrors services/comfy.py)
MAX_LYRICS_CHARS = 2000         # outer sanity guard on the request body
MIN_LYRIC_WORDS = 8             # fewer sung words than this is not a song
MAX_LYRIC_WORDS = 250           # ~278 s at KIDS_SONG_WPS: 1.3x headroom under the hard cut

# ---- clip length as the video model really renders it ----------------------
# No model renders exactly the length it is asked for; each quantises it to a frame count its
# graph can produce (see the "frames" lambdas in VIDEO_MODELS). MiniMax H3 rounds to
# f ≡ 5 (mod 17), so 5 s comes out as 124 frames = 5.1667 s; LTX 2.5 renders whole seconds plus
# one frame, so 5 s comes out as 121 frames = 5.0417 s. Everything that lines audio up with clips
# (lyric windows, scene timelines, the clip count) must use these real lengths, not CLIP_SECONDS -
# and must pass the job's own model, because the two disagree.
VIDEO_FPS = 24                  # both models render at 24 fps (VIDEO_MODELS[...]["fps"])
FRAME_QUANTUM = 17              # MiniMax H3's frame quantum
MIN_LAST_CLIP_SECONDS = 2.0     # shortest trailing clip when the last clip is trimmed to the audio


def clip_frames(seconds: float, model: str = "") -> int:
    """Frame count `model` renders for a requested clip length (mirrors its length node)."""
    m = video_model(model)
    return m["frames"](seconds, m["fps"])


def actual_clip_seconds(seconds: float | None = None, model: str = "") -> float:
    """Real length of a clip `model` renders when asked for `seconds` (default CLIP_SECONDS).

    A fixed point: feeding its own result back in returns the same value, which is what lets a
    frozen plan["clip_seconds"] be re-quantised at render time without drifting."""
    secs = CLIP_SECONDS if seconds is None else seconds
    return clip_frames(secs, model) / video_model(model)["fps"]


def model_duration_value(seconds: float, model: str = ""):
    """What `model`'s duration node wants to be handed for a clip of `seconds`: a float of
    seconds for H3, whose graph quantises them itself, or the whole seconds behind the frame
    count for LTX, whose duration node is a PrimitiveInt."""
    m = video_model(model)
    return m["duration_value"](seconds, clip_frames(seconds, model), m["fps"])


def clip_durations(total_seconds: float, clip_seconds: float | None = None, model: str = "") -> list:
    """Per-clip real lengths covering an audio track of total_seconds: full clips plus one
    trailing clip quantised from the remainder (at least MIN_LAST_CLIP_SECONDS), so the video
    ends with the audio instead of a silent tail.

    clip_seconds: requested length of a full clip (default CLIP_SECONDS); the per-job target
    length picks it (video_length_clip_seconds) so long videos do not become 60 clips.

    model: the job's video model, which decides what the requested lengths really render as."""
    full = actual_clip_seconds(clip_seconds, model)
    total = max(0.0, float(total_seconds))
    n = max(1, math.ceil(total / full - 1e-6))
    rem = total - (n - 1) * full
    if n > 1 and rem < 0.5:     # a few hundred ms of fade-out are not worth a whole extra clip
        return [full] * (n - 1)
    last = actual_clip_seconds(max(MIN_LAST_CLIP_SECONDS, rem), model)
    return [full] * (n - 1) + [last]


def clip_seconds_for_duration(seconds: float) -> float:
    """Full-clip length for a track whose length was NOT picked from VIDEO_LENGTHS (a song sung
    from the user's own lyrics): the same ladder as video_length_clip_seconds, selected by the
    measured duration, so the clip count stays around 25-30 however long the song came out.

    A pure function of `seconds` and the ladder - a resume recomputes the same value, so
    plan["clip_seconds"] does not change and the scene prompts are not rewritten."""
    hit = [float(v[2]) for v in VIDEO_LENGTHS.values()
           if v and len(v) > 2 and float(seconds or 0.0) >= v[0]]
    return max(hit) if hit else float(CLIP_SECONDS)


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
    "DEFAULT_VIDEO_MODEL": (str, lambda v: v in VIDEO_MODELS, "Video model",
                            "Which ComfyUI workflow renders the clips; can be changed per video in "
                            "the New Video dialog. The models quantise clip length differently "
                            "(5 s renders as 5.17 s on MiniMax H3, 5.04 s on LTX 2.5), so the "
                            "choice is frozen when the video is planned. Thumbnails, cast "
                            "portraits and story board images always render on MiniMax H3"),
    "LTX_NEGATIVE_ANIMATED": (str, lambda v: len(v) <= 600, "LTX negative prompt (animated)",
                              "What LTX 2.5 should steer away from on an animated channel. MiniMax "
                              "H3 has no negative prompt and ignores this"),
    "LTX_NEGATIVE_REAL": (str, lambda v: len(v) <= 600, "LTX negative prompt (real / tech)",
                          "The same for a live-action or tech channel, where naming cartoon and "
                          "video-game looks helps instead of fighting the art style"),
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
                       "8-step turbo sampling: ~3x faster, lower quality. MiniMax H3 only - "
                       "LTX 2.5 is already a distilled model and has no such switch"),
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
    "DEFAULT_AVATAR_MODEL": (str, lambda v: v in AVATAR_MODELS, "Avatar: default engine",
                             "Which engine animates a channel's presenter portrait when the channel does not "
                             "pick one. Wan 2.2 S2V holds the uploaded likeness and is the quality choice; "
                             "LTX 2.5 is about 33x faster but the face drifts off the photo"),
    "DEFAULT_AVATAR_COVERAGE": (str, lambda v: v in AVATAR_COVERAGES, "Avatar: default coverage",
                                "Where the presenter appears in a new video: " +
                                ", ".join(f"{k} ({v})" for k, v in AVATAR_COVERAGES.items()) +
                                ". Per-channel and per-video overrides both exist"),
    "DEFAULT_AVATAR_STYLE": (str, lambda v: v in AVATAR_STYLES, "Avatar: look",
                             "cutout floats the presenter on the video with the background matted away; "
                             "bubble puts them in a circle with a ring and shadow"),
    "AVATAR_INTRO_SECONDS": (float, lambda v: 2 <= v <= 120, "Avatar: intro seconds",
                             "How long the presenter speaks at the start of a video"),
    "AVATAR_OUTRO_SECONDS": (float, lambda v: 2 <= v <= 120, "Avatar: outro seconds",
                             "How long the presenter speaks at the end of a video"),
    "AVATAR_MAX_SECONDS": (float, lambda v: 0 <= v <= 3600, "Avatar: footage budget (seconds)",
                           "Hard cap on how much presenter footage a video may contain. 0 disables the "
                           "avatar everywhere"),
    "AVATAR_MAX_RENDER_MINUTES": (float, lambda v: 0 <= v <= 1440, "Avatar: render budget (minutes)",
                                  "Hard cap on GPU wall clock spent on the presenter per video. Lip sync is "
                                  "the most expensive step in the pipeline - Wan is ~9.2 s of render per "
                                  "output frame, so 24 s of footage is ~70 min - and coverage is trimmed to "
                                  "fit this with the reason written to the job log, rather than a job quietly "
                                  "taking hours on a queue other videos are waiting in"),
    "AVATAR_SIZE_PCT": (float, lambda v: 0.05 <= v <= 0.6, "Avatar: bubble size",
                        "Bubble diameter as a fraction of the frame's short side, so it scales with every "
                        "resolution and aspect ratio"),
    "AVATAR_CUTOUT_PCT": (float, lambda v: 0.1 <= v <= 1.0, "Avatar: cut-out height",
                          "Cut-out presenter height as a fraction of the frame height"),
    "AVATAR_MAX_WIDTH_PCT": (float, lambda v: 0.1 <= v <= 1.0, "Avatar: max width",
                             "The presenter is never wider than this fraction of the frame. On a "
                             "9:16 Short, sizing by height alone would make it as wide as the "
                             "video, so this is the cap that actually binds there"),
    "AVATAR_MARGIN_PCT": (float, lambda v: 0 <= v <= 0.3, "Avatar: edge margin",
                          "Inset from the frame edge, as a fraction of the short side"),
    "AVATAR_CORNER": (str, lambda v: v in AVATAR_CORNERS, "Avatar: corner",
                      "Which corner the presenter sits in: " + ", ".join(AVATAR_CORNERS)),
    "STORYBOARD_MAX_IMAGES": (int, lambda v: 0 <= v <= 40, "Story board: max images",
                              "Story board overlay: at most this many related images are rendered per video "
                              "(each is one short ComfyUI render); 0 = text cards only"),
    "STORYBOARD_ACCENT": (str, lambda v: bool(re.fullmatch(r"#[0-9a-fA-F]{6}", v)),
                          "Story board: accent colour",
                          "Hex colour of the story board cards' accent (kicker label and side bar), e.g. #e9a13b"),
    "STORYBOARD_TEMPLATE_SET": (str, lambda v: v in STORYBOARD_TEMPLATE_SETS,
                                "Story board: card layouts",
                                "Which card layouts new videos draw from. Mixed uses all eight: the card's own "
                                "content picks the family (a figure, a picture, a closing takeaway) and a "
                                "per-video seed picks the variant, never the same layout twice in a row. "
                                "Classic uses only the lower-third panel. Applies to newly planned story "
                                "boards; regenerate a job's story board to restyle it"),
    "POLISH": (bool, lambda v: True, "Hook card, end card and progress bar",
               "Default for new videos: a full-screen hook card over the opening seconds, a subscribe "
               "card at the end, a thin progress bar and a short whoosh as each story board card "
               "appears. Can be changed per channel and per video. All of it rides in the overlay "
               "pass that already runs, so it costs no extra encode"),
    "HOOK_CARD_SECONDS": (float, lambda v: 0 <= v <= 6, "Hook card (s)",
                          "Seconds the opening line is shown as a full-screen card (0 = off). It is "
                          "already the first spoken line; this puts it on screen too, for the viewer "
                          "who has not unmuted yet"),
    "END_CARD_SECONDS": (float, lambda v: 0 <= v <= 10, "End card (s)",
                         "Seconds of subscribe card at the end (0 = off). Skipped on videos shorter "
                         "than three times this, and kept clear of the lower-right quarter where "
                         "YouTube draws its own end-screen elements"),
    "END_CARD_TEXT": (str, lambda v: len(v) <= 60, "End card line",
                      "The line on the end card, e.g. 'Subscribe for more'"),
    "PROGRESS_BAR_HEIGHT": (float, lambda v: 0 <= v <= 2.0, "Progress bar height (% of frame)",
                            "A thin bar across the bottom that fills as the video plays (0 = off). "
                            "0.6 is about 6 pixels on a 1080p frame"),
    "SFX_VOLUME": (float, lambda v: 0 <= v <= 1, "Card whoosh volume",
                   "A short whoosh when an overlay card appears (0 = silent). It is deliberately NOT "
                   "ducked under the narration: a whoosh you cannot hear while the voice talks is "
                   "pointless"),
    "SFX_FILE": (str, lambda v: v == "" or Path(v).exists(), "Card whoosh file",
                 "Your own sound file for the card whoosh; empty = one is synthesized with ffmpeg on "
                 "first use and cached, so the repository ships no audio binary"),
    "LOUDNESS_LUFS": (float, lambda v: v == 0 or -24 <= v <= -8, "Loudness target (LUFS)",
                      "EBU R128 normalisation of the final mix (0 = off). YouTube normalises "
                      "everything to about -14, so a quieter master is simply played quiet"),
    "CAPTIONS_ENABLED": (bool, lambda v: True, "Burned-in captions",
                         "MASTER switch for word-synced subtitles burned into the picture. Off means no "
                         "video is captioned, whatever a channel or a video asks for, and reassembling an "
                         "existing video strips them. On, the per-channel and per-video switches decide. "
                         "The words come from the script itself - the narration is timed with whisper and "
                         "snapped back onto the written text, so captions show your words, not what "
                         "whisper thought it heard"),
    "CAPTION_MAX_WORDS": (int, lambda v: 1 <= v <= 6, "Captions: words per group",
                          "How many words appear on screen at once (1-6). Two or three reads fastest "
                          "on a phone"),
    "CAPTION_MAX_CHARS": (int, lambda v: 10 <= v <= 48, "Captions: characters per group",
                          "A group also breaks at this many characters (10-48), so one line always fits "
                          "the frame"),
    "CAPTION_HIGHLIGHT": (str, lambda v: v in CAPTION_HIGHLIGHTS, "Captions: word highlight",
                          "word = the word being spoken is coloured as the line plays (karaoke); off = "
                          "the whole group is one colour. Only the colour changes - resizing the spoken "
                          "word would reflow the line and make it jump on every word"),
    "CAPTION_ACCENT": (str, lambda v: bool(re.fullmatch(r"#[0-9a-fA-F]{6}", v)),
                       "Captions: highlight colour", "Hex colour of the spoken word, e.g. #2ba1e9"),
    "CAPTION_FONT_SCALE": (float, lambda v: 0.6 <= v <= 1.8, "Captions: text size",
                           "Multiplier on the caption text size (0.6-1.8); 1.0 is about 5% of the "
                           "frame height"),
    "CAPTION_BOTTOM_MARGIN": (float, lambda v: 0.03 <= v <= 0.35, "Captions: bottom margin",
                              "Distance from the bottom of the frame as a fraction of its height "
                              "(0.03-0.35). Portrait videos never go below 0.14, which is what keeps "
                              "the text clear of the Shorts UI"),
    "THUMB_TEXT_ENABLED": (bool, lambda v: True, "Thumbnail text",
                           "MASTER switch for the short punchy line drawn over the generated thumbnail. "
                           "Off means no thumbnail carries text, whatever a channel or a video asks for. "
                           "On, the per-channel and per-video switches decide. The picture itself is "
                           "always generated without letters (image models spell badly), so the text is "
                           "drawn afterwards with the same font the story board cards use"),
    "THUMB_TEXT_VARIANTS": (int, lambda v: 0 <= v <= 3, "Thumbnail: text variants",
                            "How many text versions of the thumbnail to offer on the job page (0-3). They "
                            "cost no GPU time: one picture is rendered and the text is drawn over copies "
                            "of it, so switching between them is instant"),
    "THUMB_TEXT_ACCENT": (str, lambda v: v == "" or bool(re.fullmatch(r"#[0-9a-fA-F]{6}", v)),
                          "Thumbnail: text accent colour",
                          "Hex colour of the thumbnail text's highlight; empty = use the story board "
                          "accent colour"),
    "STORYBOARD_MOTION": (bool, lambda v: True, "Story board: card motion",
                          "Cards ease into place (a short slide or rise) instead of only fading in. Costs "
                          "nothing: the movement is a per-frame position in the overlay pass that already runs"),
    "TARGET_VIDEO_SECONDS": (float, lambda v: v == 0 or 15 <= v <= 300, "Target video length (s)",
                             "music mode: requested song length; 0 = no target, the song covers the full content"),
    "MIN_TRANSCRIPT_WORDS": (int, lambda v: 0 <= v <= 2000, "Min transcript words",
                             "Refuse videos whose transcript has fewer words than this"),
    "MIN_TRANSCRIPT_WPM": (float, lambda v: 0 <= v <= 200, "Min transcript words/min",
                           "Refuse when the transcript is too sparse for the video length (music/no speech)"),
    "YT_CATEGORY_ID": (str, lambda v: v in YT_CATEGORIES, "YouTube category",
                       "Category every upload is filed under. 24 (Entertainment) was the old fixed value "
                       "and it buries technical videos; a channel whose video type is 'tech' uses "
                       "28 (Science & Technology) automatically, whatever this is set to"),
    "YT_MAX_TAGS": (int, lambda v: 0 <= v <= 30, "YouTube: tags per video",
                    "How many of the generated SEO keywords are uploaded as video tags (0 = none). "
                    "YouTube caps all tags together at 500 characters, so fewer may actually be sent"),
    "YT_CHAPTERS": (bool, lambda v: True, "Chapters in the description",
                    "Add a timestamp list so YouTube splits the player into chapters. It needs at least "
                    "3 chapters, the first at 0:00, each at least 10 seconds - a video too short for "
                    "that simply gets none, because YouTube ignores a partial list entirely"),
    "YT_CHAPTER_MIN_SECONDS": (float, lambda v: 10 <= v <= 120, "Chapters: shortest chapter (s)",
                               "Chapters are coarser than the story board cards on purpose: a clip is "
                               "about 5 seconds, so one chapter per card would be under YouTube's "
                               "10-second floor. 20 gives four to six chapters on a typical video"),
    "YT_PRIVACY": (str, lambda v: v in ("private", "unlisted", "public"), "YouTube privacy",
                   "Privacy of uploaded videos: private (review in YouTube Studio first), unlisted or public"),
    "PUBLIC_BASE_URL": (str, lambda v: v == "" or v.startswith("http"), "Public base URL (YouTube OAuth)",
                        "Leave empty (default): Google returns to http://localhost:<port>/api/youtube/callback, "
                        "which Desktop-type OAuth clients allow; from another computer you paste that address "
                        "into Settings to finish. Set to the HTTPS address users open the app with (e.g. "
                        "https://videogen.example.com) together with a Web-type OAuth client that has "
                        "<url>/api/youtube/callback registered for a fully automatic sign-in"),
    "STORY_WPS": (float, lambda v: 1.0 <= v <= 4.0, "Story words/second",
                  "Spoken-words-per-second budget for a story video: it sizes every spoken line so the "
                  "line fits the clip that shows its scene (1-4)"),
    "STORY_LINE_LEAD": (float, lambda v: 0 <= v <= 2.0, "Story line lead-in (s)",
                        "Silence held at the start of each story clip before the line is spoken, so no "
                        "word begins exactly on a cut (0-2 s)"),
    "STORY_LINE_GAP": (float, lambda v: 0 <= v <= 3.0, "Story line tail (s)",
                       "Silence held after each spoken story line before its clip ends (0-3 s)"),
    "STORY_MAX_CLIP_SECONDS": (float, lambda v: 6.0 <= v <= 15.0, "Story: longest clip (s)",
                               "A story line that speaks longer than this stretches its clip too far, so "
                               "the LLM is asked once for a shorter line and it is spoken again (6-15 s)"),
    "STORY_MIN_BEATS": (int, lambda v: 2 <= v <= 20, "Story: fewest scenes",
                        "A story is always told in at least this many scenes, however short the target "
                        "length is (2-20)"),
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
