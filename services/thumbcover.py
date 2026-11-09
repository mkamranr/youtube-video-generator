"""Reel cover: a dense, poster-style thumbnail layout for portrait (9:16) videos.

Where services/thumbtext.py draws ONE short line, a cover stacks several elements the way tech
reel covers do: a headline with accent words, a black-on-yellow brush strip, one or two follow-up
lines, a tilted figure badge, a short checklist and a handwritten tag. The LLM writes the copy as
a small dict (services/llm.write_thumbnail, key "thumbnail_cover") and normalize() below cleans it.

Same contract as the other layouts: the picture is the letterless GPU render, everything here is
Pillow, and the channel logo in the top-right corner is never drawn over. The layout is designed
on a 1080x1920 canvas and scaled back to the base's size, so it only applies to frames close to
9:16 - see eligible().

Fonts are the open-licence ones in fonts/cover/ (Poppins, Permanent Marker); a missing file falls
back to the story board's bold font so a cover still renders on a stripped-down install.
"""
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config
from services import storyboard

DW, DH = 1080, 1920                     # design canvas
MIN_RATIO = 1.6                         # H/W at or above which a frame counts as 9:16-ish

YELLOW = (255, 222, 0, 255)
ORANGE = (255, 140, 0, 255)
WHITE = (255, 255, 255, 255)
INK = (14, 14, 20, 255)
RED = (230, 30, 40, 255)
GREEN = (40, 200, 70, 255)

BLACK_FONT = "Poppins-Black.ttf"
XBOLD_FONT = "Poppins-ExtraBold.ttf"
BOLD_FONT = "Poppins-Bold.ttf"
HAND_FONT = "PermanentMarker-Regular.ttf"

# (max_words, max_chars) per field; the headline's chars exclude the *accent* markers
LIMITS = {"headline": (3, 22), "strip": (4, 20), "line": (3, 18), "check": (3, 18),
          "tag": (6, 28), "badge_big": (1, 6), "badge_small": (2, 14)}


# ---- copy ---------------------------------------------------------------------------
def _clip(text, field: str) -> str:
    """One clean field: collapsed spaces, no wrapping quotes, no trailing full stop, capped."""
    words_max, chars_max = LIMITS[field]
    t = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”‘’").strip()
    t = re.sub(r"[.…]+$", "", t).strip()
    words = t.split()[:words_max]
    plain = lambda ws: len(" ".join(ws).replace("*", ""))   # noqa: E731
    while len(words) > 1 and plain(words) > chars_max:
        words.pop()
    t = " ".join(words)
    if plain(words) > chars_max:                 # one over-long word: cut it, drop stray markers
        t = t.replace("*", "")[:chars_max].strip()
    if t.count("*") % 2:                         # an unclosed accent marker means nothing
        t = t.replace("*", "")
    return t


def normalize(raw) -> dict | None:
    """The cover dict the LLM returned, cleaned and capped - or None when it is unusable.

    A cover needs at least a headline or a strip line; every other element is optional and the
    layout simply closes up around a missing one."""
    if not isinstance(raw, dict):
        return None
    lines = raw.get("lines") or []
    if isinstance(lines, str):
        lines = [lines]
    checks = raw.get("checks") or []
    if isinstance(checks, str):
        checks = [checks]
    badge = raw.get("badge") if isinstance(raw.get("badge"), dict) else {}
    out = {
        "headline": _clip(raw.get("headline"), "headline"),
        "strip": _clip(raw.get("strip"), "strip"),
        "lines": [x for x in (_clip(v, "line") for v in list(lines)[:2]) if x],
        "badge": None,
        "checks": [x for x in (_clip(v, "check") for v in list(checks)[:3]) if x],
        "tag": _clip(raw.get("tag"), "tag"),
    }
    big, small = _clip(badge.get("big"), "badge_big"), _clip(badge.get("small"), "badge_small")
    if big:
        out["badge"] = {"big": big.replace("*", ""), "small": small.replace("*", "").upper()}
    if not (out["headline"] or out["strip"]):
        return None
    return out


def summary(cover: dict) -> str:
    """The short label the job page shows under this variant."""
    return " ".join(x for x in (cover.get("headline", "").replace("*", ""), cover.get("strip", "")) if x)


def eligible(size: tuple) -> bool:
    w, h = size
    return w > 0 and h / w >= MIN_RATIO


def eligible_aspect(aspect_ratio: str) -> bool:
    w, h = config.THUMB_SIZES.get(aspect_ratio, (1280, 720))
    return eligible((w, h))


# ---- drawing primitives ----------------------------------------------------------------
_FONT_DIR = config.BASE_DIR / "fonts" / "cover"


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), int(size))
    except OSError:
        return storyboard.load_font(int(size), bold=True)


def _fit(text: str, name: str, max_w: float, max_size: int, min_size: int = 36):
    for s in range(max_size, min_size, -2):
        f = _font(name, s)
        if f.getlength(text) <= max_w:
            return f
    return _font(name, min_size)


def _runs(text: str, accent) -> list:
    """'Xiaomi-*OCR-0*' -> [('Xiaomi-', None), ('OCR-0', accent)]."""
    out = []
    for i, part in enumerate(text.split("*")):
        if part:
            out.append((part, accent if i % 2 else None))
    return out


def _shadow_under(canvas: Image.Image, layer: Image.Image, pos=(0, 0), blur=10, offset=(6, 8),
                  opacity=190):
    a = layer.getchannel("A").point(lambda v: v * opacity // 255)
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 255))
    sh.putalpha(a)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh, (pos[0] + offset[0], pos[1] + offset[1]))
    canvas.alpha_composite(layer, pos)


def _brush_strip(d, box, color, rng, rough=14):
    """A highlighter / paint-brush strip: jagged ends, wobbly edges, dry-brush streaks."""
    x0, y0, x1, y1 = box
    n = 28
    top = [(x0 + (x1 - x0) * i / n, y0 + rng.uniform(-rough, rough) * 0.5) for i in range(n + 1)]
    bot = [(x0 + (x1 - x0) * i / n, y1 + rng.uniform(-rough, rough) * 0.5) for i in range(n + 1)]
    left = [(x0 - rng.uniform(0, rough * 2), y0 + (y1 - y0) * k / 6) for k in range(7)]
    right = [(x1 + rng.uniform(0, rough * 2), y0 + (y1 - y0) * k / 6) for k in range(7)]
    d.polygon(top + right + bot[::-1] + left[::-1], fill=color)
    for _ in range(5):
        for edge, sgn in ((x1, 1), (x0, -1)):
            yy = rng.uniform(y0 + 6, y1 - 6)
            d.line([(edge - 10 * sgn, yy), (edge + sgn * rng.uniform(20, 60), yy + rng.uniform(-4, 4))],
                   fill=color, width=rng.randint(3, 7))


def _swoosh(d, x0, x1, y, color, width=8, lift=14):
    pts = [(x0 + (x1 - x0) * t, y - math.sin(t * math.pi) * lift + t * 6)
           for t in [i / 30 for i in range(31)]]
    d.line(pts, fill=color, width=width, joint="curve")


def _line(d, y, text, name, max_w, max_size, fill, accent=None, stroke=True) -> tuple:
    """One centred line, *accent* words in `accent`. Returns (bottom_y, font, width)."""
    runs = _runs(text, accent)
    full = "".join(t for t, _ in runs)
    f = _fit(full, name, max_w, max_size)
    total = f.getlength(full)
    x = DW / 2 - total / 2
    sw = max(3, f.size // 16) if stroke else 0
    for t, col in runs:
        d.text((x, y), t, font=f, fill=col or fill, stroke_width=sw, stroke_fill=INK)
        x += f.getlength(t)
    asc, desc = f.getmetrics()
    return y + asc + desc, f, total


def _check(d, cx, cy, r):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GREEN, outline=WHITE, width=3)
    d.line([(cx - r * .45, cy + r * .02), (cx - r * .1, cy + r * .38), (cx + r * .5, cy - r * .38)],
           fill=WHITE, width=max(4, r // 4), joint="curve")


def _arrow(d, p0, p1, ctrl, color, width=16):
    pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t * t * p1[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t * t * p1[1])
           for t in [i / 40 for i in range(41)]]
    ang = math.atan2(pts[-1][1] - pts[-4][1], pts[-1][0] - pts[-4][0])
    tip = (p1[0] + math.cos(ang) * 18, p1[1] + math.sin(ang) * 18)
    head = [tip,
            (tip[0] - 46 * math.cos(ang - 0.5), tip[1] - 46 * math.sin(ang - 0.5)),
            (tip[0] - 46 * math.cos(ang + 0.5), tip[1] - 46 * math.sin(ang + 0.5))]
    d.line(pts, fill=INK, width=width + 10, joint="curve")
    d.polygon(head, fill=INK, outline=INK, width=10)
    d.line(pts, fill=color, width=width, joint="curve")
    d.polygon(head, fill=color)


def _rotated(canvas, draw_fn, size, angle, center):
    local = Image.new("RGBA", size)
    draw_fn(ImageDraw.Draw(local), local)
    local = local.rotate(angle, resample=Image.BICUBIC, expand=True)
    pos = (int(center[0] - local.width / 2), int(center[1] - local.height / 2))
    _shadow_under(canvas, local, pos, blur=14, offset=(8, 12), opacity=200)
    return pos[1] + local.height


def _vramp(alpha_at) -> Image.Image:
    """A DWxDH dark-violet layer whose alpha at row y is alpha_at(y)."""
    col = Image.new("L", (1, DH))
    col.putdata([max(0, min(255, int(alpha_at(y)))) for y in range(DH)])
    layer = Image.new("RGBA", (DW, DH), (10, 8, 40, 0))
    layer.putalpha(col.resize((DW, DH)))
    return layer


def _backdrop(img: Image.Image) -> Image.Image:
    """Brand wash (blue -> purple), scrims behind the text bands and two soft edge glows."""
    tint = Image.new("RGBA", (DW, DH))
    grad = Image.new("RGB", (1, DH))
    grad.putdata([(int(70 + 60 * y / DH), 20, int(200 - 60 * y / DH)) for y in range(DH)])
    tint.paste(grad.resize((DW, DH)))
    tint.putalpha(70)
    img = Image.alpha_composite(img, tint)
    img = Image.alpha_composite(img, _vramp(
        lambda y: min(235, max(0.0, 1 - y / 980) ** 1.4 * 215 + max(0.0, (y - 1250) / 670) ** 1.3 * 225)))
    glow = Image.new("RGBA", (DW, DH))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-420, 120, 260, 900), fill=(190, 40, 255, 110))
    gd.ellipse((820, 500, 1500, 1300), fill=(40, 120, 255, 90))
    return Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(140)))


# ---- layout -----------------------------------------------------------------------------
def draw(base: Image.Image, cover: dict, seed: int = 7) -> Image.Image:
    """The cover drawn over `base` (any size); returns a new RGBA image of the same size."""
    size = base.size
    rng = random.Random(seed)
    img = _backdrop(base.convert("RGBA").resize((DW, DH), Image.LANCZOS))
    y = 205

    # headline + the follow-up lines, one shadowed layer
    text = Image.new("RGBA", (DW, DH))
    td = ImageDraw.Draw(text)
    if cover.get("headline"):
        y, _, _ = _line(td, y, cover["headline"], BLACK_FONT, 900, 130, WHITE, accent=ORANGE)
        _swoosh(td, 90, DW - 180, y - 6, ORANGE, width=6, lift=8)
        y += 12
    _shadow_under(img, text)

    if cover.get("strip"):
        f = _fit(cover["strip"], BLACK_FONT, 860, 128)
        asc, desc = f.getmetrics()
        tw = f.getlength(cover["strip"])
        strip = Image.new("RGBA", (DW, DH))
        _brush_strip(ImageDraw.Draw(strip), (DW / 2 - tw / 2 - 36, y + 22, DW / 2 + tw / 2 + 36,
                                             y + asc + desc - 14), YELLOW, rng)
        strip = strip.rotate(-1.5, resample=Image.BICUBIC, center=(DW / 2, y + (asc + desc) / 2))
        img = Image.alpha_composite(img, strip)
        ImageDraw.Draw(img).text((DW / 2 - tw / 2, y), cover["strip"], font=f, fill=INK)
        y += asc + desc + 14

    lines = cover.get("lines") or []
    if lines:
        text = Image.new("RGBA", (DW, DH))
        td = ImageDraw.Draw(text)
        for i, ln in enumerate(lines):
            last = i == len(lines) - 1
            if last:                                  # the payoff line: big and yellow
                y, f, w = _line(td, y - 10, ln, BLACK_FONT, 880, 140, YELLOW, accent=ORANGE)
            else:
                y, f, w = _line(td, y, ln, XBOLD_FONT, 860, 112, WHITE, accent=YELLOW)
            if last:                                  # spark marks off the last line's corner
                asc, desc = f.getmetrics()
                ex, ey = min(DW - 70, DW / 2 + w / 2 + 4), y - asc - desc + 40
                for a in (-80, -45, -10):
                    r = math.radians(a)
                    td.line([(ex + 14 * math.cos(r), ey + 14 * math.sin(r)),
                             (ex + 54 * math.cos(r), ey + 54 * math.sin(r))], fill=YELLOW, width=8)
        _shadow_under(img, text)

    # badge on the left of the middle band, an arrow from the right edge into the scene
    checks = cover.get("checks") or []
    checks_top = 1310 if checks else DH
    badge = cover.get("badge")
    if badge:
        def _badge(d, canvas):
            w, h = canvas.size
            d.rounded_rectangle((12, 12, w - 12, h - 12), radius=44, fill=INK)
            d.rounded_rectangle((22, 22, w - 22, h - 22), radius=36, fill=YELLOW,
                                outline=(255, 245, 160, 255), width=4)
            big = _fit(badge["big"], BLACK_FONT, w - 100, 150)
            if badge.get("small"):
                d.text((w / 2, 128), badge["big"], font=big, fill=INK, anchor="mm")
                small = _fit(badge["small"], BLACK_FONT, w - 90, 60)
                d.text((w / 2, 252), badge["small"], font=small, fill=INK, anchor="mm")
                _swoosh(d, 70, w - 70, 300, RED, width=10, lift=10)
            else:
                d.text((w / 2, h / 2), badge["big"], font=big, fill=INK, anchor="mm")
        cy = min(max(y + 245, 1000), checks_top - 235)
        _rotated(img, _badge, (520, 350), 6, (300, cy))
        arr = Image.new("RGBA", (DW, DH))
        _arrow(ImageDraw.Draw(arr), (1010, cy - 145), (760, cy - 65), (960, cy - 15), YELLOW)
        _shadow_under(img, arr, blur=10, offset=(5, 8), opacity=170)

    if checks:
        text = Image.new("RGBA", (DW, DH))
        td = ImageDraw.Draw(text)
        cy = checks_top
        for i, c in enumerate(checks):
            hero = i == len(checks) - 1 and len(checks) > 1
            f = _fit(c, BLACK_FONT if hero else BOLD_FONT, DW - 220, 76 if hero else 66)
            _check(td, 92, cy + f.size * 0.62, 36)
            td.text((150, cy), c, font=f, fill=YELLOW if hero else WHITE, stroke_width=3, stroke_fill=INK)
            cy += f.size + 16
        _shadow_under(img, text, blur=8, offset=(4, 6))

    if cover.get("tag"):
        words = cover["tag"].split()
        cut = (len(words) + 1) // 2
        tag_lines = [" ".join(words[:cut]), " ".join(words[cut:])] if len(words) > 2 else [cover["tag"]]

        def _tag(d, canvas):
            w, h = canvas.size
            _brush_strip(d, (40, 30, w - 40, h - 30), YELLOW, rng, rough=22)
            f = min((_fit(t, HAND_FONT, w - 130, 66) for t in tag_lines), key=lambda f: f.size)
            for i, t in enumerate(tag_lines):
                yy = h * (0.33 + 0.35 * i) if len(tag_lines) > 1 else h * 0.48
                d.text((w / 2, yy), t, font=f, fill=INK, anchor="mm")
            _swoosh(d, 70, w - 60, h - 52, RED, width=9, lift=-10)
        _rotated(img, _tag, (560, 250), 6, (DW / 2 + 110, 1745))

    return img.resize(size, Image.LANCZOS)


def render(base_jpg, out_jpg, cover: dict, logo=None) -> Path:
    """Draw `cover` over a copy of base_jpg and save it as out_jpg (same size, JPEG).

    The base already carries the channel logo, but the backdrop wash darkens it, so the logo is
    pasted again on top - with its own alpha, at the same place - when its file is known.
    Without it the logo corner is restored from the original, as thumbtext.render does."""
    cover = normalize(cover)
    if not cover:
        raise ValueError("no usable cover copy")
    with Image.open(base_jpg) as src:
        original = src.convert("RGBA")
    if not eligible(original.size):
        raise ValueError(f"a cover needs a 9:16 frame, not {original.width}x{original.height}")
    out = draw(original, cover)
    if logo and Path(logo).exists():
        from services import video
        video.overlay_logo(out, logo)
    else:
        W = out.width
        side = int(0.19 * W)
        box = (W - side, 0, W, side)
        out.paste(original.crop(box), box[:2])
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(out_jpg, "JPEG", quality=92)
    return out_jpg
