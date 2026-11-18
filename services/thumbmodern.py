"""Modern thumbnail: a premium tech-advert poster for portrait (9:16) videos.

The "classic" thumbnails are the GPU render with one overlay line (services/thumbtext.py) or the
reel cover (services/thumbcover.py). A modern thumbnail is the dense ad-poster look: the channel
brand top-left, a stacked headline (white title, black-on-yellow brush strip with a check mark, a
white line, a yellow + white split line with a red underline), a red-neon feature badge, the
rendered scene in the middle, a row of four icon cards and a yellow brush banner at the bottom.

Same contract as the other layouts: the picture is a letterless GPU render composed for this
layout (services/llm.write_thumbnail_modern), and every word and icon here is Pillow, so the text is
always spelled right. Icons come from a small fixed vocabulary (ICONS) drawn as vector shapes - the
LLM picks names from it - so nothing depends on an emoji font being installed.

Designed on a 1080x1920 canvas and scaled to the base's size, so it only applies to frames close to
9:16 (thumbcover.eligible).
"""
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from services import thumbcover
from services.thumbcover import BLACK_FONT, BOLD_FONT, XBOLD_FONT, _brush_strip, _fit, _font

DW, DH = thumbcover.DW, thumbcover.DH

NAVY = (0, 19, 47, 255)
NAVY2 = (0, 41, 91, 255)
CYAN = (0, 191, 255, 255)
WHITE = (255, 255, 255, 255)
YELLOW = (255, 217, 0, 255)
ORANGE = (255, 75, 0, 255)
GREEN = (0, 220, 60, 255)
RED = (235, 30, 40, 255)
INK = (8, 8, 14, 255)
BLUE_INK = (6, 69, 192, 255)
PURPLE = (170, 90, 255, 255)

ICONS = ("shield", "lock", "gear", "check", "terminal", "wifi_off", "cloud", "rocket", "bolt",
         "chip", "server", "code", "globe", "clock", "star", "spark", "download", "chart")

# (max_words, max_chars) per field
LIMITS = {"title": (3, 20), "strip": (3, 16), "line": (3, 16), "split": (2, 12),
          "badge": (4, 22), "card": (4, 24), "banner": (5, 26), "tagline": (7, 32)}


# ---- copy ---------------------------------------------------------------------------
def _clip(text, field: str) -> str:
    words_max, chars_max = LIMITS[field]
    t = re.sub(r"\s+", " ", str(text or "")).replace("*", "").strip().strip("\"'“”‘’").strip()
    t = re.sub(r"[.…]+$", "", t).strip()
    words = t.split()[:words_max]
    while len(words) > 1 and len(" ".join(words)) > chars_max:
        words.pop()
    t = " ".join(words)
    return t[:chars_max].strip()


def _icon(name, default: str) -> str:
    n = re.sub(r"[^a-z_]", "", str(name or "").lower().replace("-", "_").replace(" ", "_"))
    return n if n in ICONS else default


def normalize(raw) -> dict | None:
    """The modern copy the LLM returned, cleaned and capped - or None when it is unusable.

    Needs a title or a strip; everything else is optional and the layout closes up around it."""
    if not isinstance(raw, dict):
        return None
    split = raw.get("split") or []
    if isinstance(split, str):
        split = split.split("|") if "|" in split else [split]
    split = [x for x in (_clip(v, "split") for v in list(split)[:2]) if x]
    badge = raw.get("badge") if isinstance(raw.get("badge"), dict) else {}
    cards = []
    for c in list(raw.get("cards") or [])[:4]:
        if isinstance(c, dict) and _clip(c.get("text"), "card"):
            cards.append({"icon": _icon(c.get("icon"), "check"), "text": _clip(c.get("text"), "card")})
    banner = raw.get("banner") or []
    if isinstance(banner, str):
        banner = [banner]
    out = {
        "title": _clip(raw.get("title"), "title"),
        "title_icon": _icon(raw.get("title_icon"), "") if raw.get("title_icon") else "",
        "strip": _clip(raw.get("strip"), "strip"),
        "line": _clip(raw.get("line"), "line"),
        "split": split,
        "badge": None,
        "cards": cards if len(cards) >= 2 else [],
        "banner": [x for x in (_clip(v, "banner") for v in list(banner)[:2]) if x],
        "banner_icon": _icon(raw.get("banner_icon"), "rocket"),
        "tagline": _clip(raw.get("tagline"), "tagline"),
    }
    if _clip(badge.get("text"), "badge"):
        out["badge"] = {"icon": _icon(badge.get("icon"), "bolt"), "text": _clip(badge.get("text"), "badge")}
    if not (out["title"] or out["strip"]):
        return None
    return out


def summary(copy: dict) -> str:
    return "Modern: " + " ".join(x for x in (copy.get("title", ""), copy.get("strip", "")) if x)


# ---- icons ------------------------------------------------------------------------------
def draw_icon(d: ImageDraw.ImageDraw, name: str, cx: float, cy: float, s: float):
    """A colourful flat icon of size ~s centred on (cx, cy)."""
    r = s / 2
    w = max(3, int(s / 12))
    if name == "shield":
        d.polygon([(cx - r * .8, cy - r * .75), (cx, cy - r), (cx + r * .8, cy - r * .75),
                   (cx + r * .75, cy + r * .1), (cx, cy + r), (cx - r * .75, cy + r * .1)],
                  fill=(30, 140, 255, 255), outline=(150, 220, 255, 255), width=w)
        _lock(d, cx, cy + r * .05, r * .5, (230, 245, 255, 255))
    elif name == "lock":
        _lock(d, cx, cy + r * .1, r, (255, 200, 40, 255))
    elif name == "gear":
        teeth = 8
        pts = []
        for i in range(teeth * 2):
            a = math.pi * i / teeth
            rr = r if i % 2 == 0 else r * .74
            for da in (-.17, .17):
                pts.append((cx + rr * math.cos(a + da), cy + rr * math.sin(a + da)))
        d.polygon(pts, fill=PURPLE)
        d.ellipse((cx - r * .32, cy - r * .32, cx + r * .32, cy + r * .32), fill=NAVY)
    elif name == "check":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GREEN, outline=WHITE, width=w)
        d.line([(cx - r * .45, cy + r * .02), (cx - r * .1, cy + r * .38), (cx + r * .5, cy - r * .38)],
               fill=INK if s < 120 else WHITE, width=max(4, int(r / 4)), joint="curve")
    elif name == "terminal":
        d.rounded_rectangle((cx - r, cy - r * .75, cx + r, cy + r * .75), radius=r * .15,
                            outline=WHITE, width=w)
        d.line([(cx - r * .6, cy - r * .35), (cx - r * .15, cy), (cx - r * .6, cy + r * .35)],
               fill=WHITE, width=w + 2, joint="curve")
        d.line([(cx, cy + r * .4), (cx + r * .6, cy + r * .4)], fill=WHITE, width=w + 2)
    elif name == "wifi_off":
        for k, rr in enumerate((r * .95, r * .65, r * .35)):
            d.arc((cx - rr, cy - rr + r * .35, cx + rr, cy + rr + r * .35), 220, 320, fill=WHITE, width=w + 2)
        d.ellipse((cx - r * .12, cy + r * .23, cx + r * .12, cy + r * .47), fill=WHITE)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=RED, width=w + 2)
        d.line([(cx - r * .7, cy - r * .7), (cx + r * .7, cy + r * .7)], fill=RED, width=w + 3)
    elif name == "cloud":
        c = (200, 230, 255, 255)
        d.ellipse((cx - r, cy - r * .1, cx - r * .1, cy + r * .6), fill=c)
        d.ellipse((cx - r * .55, cy - r * .6, cx + r * .35, cy + r * .4), fill=c)
        d.ellipse((cx, cy - r * .3, cx + r, cy + r * .6), fill=c)
        d.rectangle((cx - r * .55, cy + r * .1, cx + r * .55, cy + r * .6), fill=c)
    elif name == "rocket":
        _rocket(d, cx, cy, r)
    elif name == "bolt":
        d.polygon([(cx + r * .15, cy - r), (cx - r * .6, cy + r * .1), (cx - r * .05, cy + r * .1),
                   (cx - r * .2, cy + r), (cx + r * .6, cy - r * .15), (cx + r * .05, cy - r * .15)],
                  fill=YELLOW, outline=(255, 150, 0, 255))
    elif name == "chip":
        d.rounded_rectangle((cx - r * .6, cy - r * .6, cx + r * .6, cy + r * .6), radius=r * .1,
                            fill=(40, 60, 90, 255), outline=CYAN, width=w)
        for i in range(4):
            o = -r * .42 + i * r * .28
            for a, b in (((cx + o, cy - r * .6), (cx + o, cy - r * .95)), ((cx + o, cy + r * .6), (cx + o, cy + r * .95)),
                         ((cx - r * .6, cy + o), (cx - r * .95, cy + o)), ((cx + r * .6, cy + o), (cx + r * .95, cy + o))):
                d.line([a, b], fill=CYAN, width=max(2, w - 1))
        d.rectangle((cx - r * .25, cy - r * .25, cx + r * .25, cy + r * .25), fill=CYAN)
    elif name == "server":
        for i in range(3):
            y0 = cy - r + i * r * .7
            d.rounded_rectangle((cx - r * .85, y0, cx + r * .85, y0 + r * .58), radius=r * .08,
                                fill=(30, 40, 60, 255), outline=CYAN, width=max(2, w - 1))
            d.ellipse((cx + r * .5, y0 + r * .2, cx + r * .68, y0 + r * .38), fill=GREEN)
    elif name == "code":
        d.line([(cx - r * .35, cy - r * .6), (cx - r * .9, cy), (cx - r * .35, cy + r * .6)], fill=CYAN, width=w + 2, joint="curve")
        d.line([(cx + r * .35, cy - r * .6), (cx + r * .9, cy), (cx + r * .35, cy + r * .6)], fill=CYAN, width=w + 2, joint="curve")
        d.line([(cx + r * .2, cy - r * .75), (cx - r * .2, cy + r * .75)], fill=WHITE, width=w + 1)
    elif name == "globe":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 110, 220, 255), outline=WHITE, width=w)
        d.ellipse((cx - r * .45, cy - r, cx + r * .45, cy + r), outline=WHITE, width=max(2, w - 1))
        d.line([(cx - r, cy), (cx + r, cy)], fill=WHITE, width=max(2, w - 1))
        d.line([(cx, cy - r), (cx, cy + r)], fill=WHITE, width=max(2, w - 1))
    elif name == "clock":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 170, 30, 255), outline=WHITE, width=w)
        d.line([(cx, cy), (cx, cy - r * .6)], fill=INK, width=w + 1)
        d.line([(cx, cy), (cx + r * .45, cy + r * .2)], fill=INK, width=w + 1)
    elif name == "star":
        pts = [(cx + (r if i % 2 == 0 else r * .42) * math.cos(-math.pi / 2 + math.pi * i / 5),
                cy + (r if i % 2 == 0 else r * .42) * math.sin(-math.pi / 2 + math.pi * i / 5)) for i in range(10)]
        d.polygon(pts, fill=YELLOW, outline=(255, 150, 0, 255))
    elif name == "spark":
        for i in range(12):
            a = math.pi * i / 6 + .13
            d.line([(cx + r * .18 * math.cos(a), cy + r * .18 * math.sin(a)),
                    (cx + r * math.cos(a), cy + r * math.sin(a))], fill=(255, 110, 60, 255), width=int(s / 7))
    elif name == "download":
        d.line([(cx, cy - r), (cx, cy + r * .3)], fill=CYAN, width=w + 3)
        d.polygon([(cx - r * .5, cy - r * .05), (cx + r * .5, cy - r * .05), (cx, cy + r * .5)], fill=CYAN)
        d.line([(cx - r * .85, cy + r * .45), (cx - r * .85, cy + r * .85), (cx + r * .85, cy + r * .85),
                (cx + r * .85, cy + r * .45)], fill=WHITE, width=w + 1, joint="curve")
    elif name == "chart":
        for i, h in enumerate((.5, .9, 1.4)):
            x0 = cx - r * .85 + i * r * .6
            d.rectangle((x0, cy + r * .85 - r * h, x0 + r * .42, cy + r * .85),
                        fill=((0, 200, 255, 255), (120, 120, 255, 255), GREEN)[i])


def _lock(d, cx, cy, r, color):
    d.arc((cx - r * .45, cy - r * .95, cx + r * .45, cy - r * .05), 180, 360, fill=color, width=max(3, int(r / 5)))
    d.rounded_rectangle((cx - r * .65, cy - r * .45, cx + r * .65, cy + r * .55), radius=r * .12, fill=color)
    d.ellipse((cx - r * .12, cy - r * .1, cx + r * .12, cy + r * .14), fill=NAVY)


def _rocket(d, cx, cy, r):
    """A rocket flying up-right: body, window, fins and a flame, rotated 45 degrees."""
    a = math.radians(-45)

    def p(x, y):
        return (cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a))
    d.polygon([p(-r * .9, -r * .22), p(-r * 1.3, 0), p(-r * .9, r * .22)], fill=(255, 140, 0, 255))
    d.polygon([p(-r * .9, -r * .12), p(-r * 1.1, 0), p(-r * .9, r * .12)], fill=YELLOW)
    d.polygon([p(-r * .55, -r * .25), p(-r * .95, -r * .6), p(-r * .8, -r * .2)], fill=RED)
    d.polygon([p(-r * .55, r * .25), p(-r * .95, r * .6), p(-r * .8, r * .2)], fill=RED)
    d.polygon([p(-r * .85, -r * .27), p(r * .35, -r * .3), p(r, 0), p(r * .35, r * .3), p(-r * .85, r * .27)],
              fill=(235, 240, 250, 255), outline=(150, 160, 180, 255))
    d.polygon([p(r * .55, -r * .2), p(r, 0), p(r * .55, r * .2)], fill=RED)
    wx, wy = p(r * .15, 0)
    d.ellipse((wx - r * .17, wy - r * .17, wx + r * .17, wy + r * .17), fill=(40, 140, 255, 255),
              outline=(90, 100, 120, 255), width=max(2, int(r / 14)))


# ---- drawing helpers ----------------------------------------------------------------------
def _italic(layer: Image.Image, shear: float = 0.18) -> Image.Image:
    """Fake italic: shear the layer right about its vertical centre (Poppins ships no italic)."""
    return layer.transform(layer.size, Image.AFFINE, (1, shear, -shear * layer.height / 2, 0, 1, 0),
                           resample=Image.BICUBIC)


def _text_layer(text, font, fill, stroke=0, stroke_fill=INK, pad=24):
    """A tight RGBA layer with `text` on it; returns (layer, text_width)."""
    w = int(font.getlength(text)) + 2 * pad
    asc, desc = font.getmetrics()
    layer = Image.new("RGBA", (w, asc + desc + 2 * pad))
    ImageDraw.Draw(layer).text((pad, pad), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return layer, w - 2 * pad


def _paste_shadow(canvas, layer, pos, blur=10, offset=(5, 8), opacity=200):
    thumbcover._shadow_under(canvas, layer, (int(pos[0]), int(pos[1])), blur=blur, offset=offset, opacity=opacity)


def _glow_rect(canvas, box, radius, color, width=4, glow=16, fill=None):
    """A rounded rectangle with a neon outline that glows."""
    g = Image.new("RGBA", canvas.size)
    ImageDraw.Draw(g).rounded_rectangle(box, radius=radius, outline=color, width=width + 6)
    canvas.alpha_composite(g.filter(ImageFilter.GaussianBlur(glow)))
    s = Image.new("RGBA", canvas.size)
    ImageDraw.Draw(s).rounded_rectangle(box, radius=radius, fill=fill, outline=color, width=width)
    canvas.alpha_composite(s)


def _underline(d, x0, x1, y, color, width=10, rng=None):
    rng = rng or random.Random(3)
    pts = [(x0 + (x1 - x0) * t, y + math.sin(t * math.pi) * -6 + t * 8 + rng.uniform(-1.5, 1.5))
           for t in [i / 30 for i in range(31)]]
    d.line(pts, fill=color, width=width, joint="curve")
    d.line([(x1 - 6, y + 6), (x1 + 30, y + 2)], fill=color, width=max(3, width // 2))


def _vramp(alpha_at, rgb=(0, 12, 34)) -> Image.Image:
    col = Image.new("L", (1, DH))
    col.putdata([max(0, min(255, int(alpha_at(y)))) for y in range(DH)])
    layer = Image.new("RGBA", (DW, DH), rgb + (0,))
    layer.putalpha(col.resize((DW, DH)))
    return layer


def _backdrop(img: Image.Image) -> Image.Image:
    """Navy/electric-blue grade, dark scrims behind the headline and the bottom panels, glows."""
    tint = Image.new("RGBA", (DW, DH), (0, 40, 110, 60))
    img = Image.alpha_composite(img, tint)
    img = Image.alpha_composite(img, _vramp(
        lambda y: max(0.0, 1 - y / 860) ** 1.2 * 225 + max(0.0, (y - 1300) / 620) ** 1.1 * 215))
    glow = Image.new("RGBA", (DW, DH))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((700, 300, 1400, 1100), fill=(0, 140, 255, 70))
    gd.ellipse((-300, 900, 380, 1500), fill=(0, 90, 255, 55))
    return Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(150)))


# ---- layout -----------------------------------------------------------------------------
def draw(base: Image.Image, copy: dict, brand: str = "", logo=None, seed: int = 11) -> Image.Image:
    """The modern poster drawn over `base` (any size); returns a new RGBA image of the same size."""
    size = base.size
    rng = random.Random(seed)
    img = _backdrop(base.convert("RGBA").resize((DW, DH), Image.LANCZOS))

    # -- brand header, top-left
    x = 56
    if logo and Path(logo).exists():
        with Image.open(logo) as lg:
            lg = lg.convert("RGBA")
            side = 128
            lg = lg.resize((side, max(1, int(lg.height * side / lg.width))), Image.LANCZOS)
        ring = Image.new("RGBA", (DW, DH))
        ImageDraw.Draw(ring).ellipse((x - 8, 44 - 8, x + side + 8, 44 + side + 8), outline=CYAN, width=5)
        img.alpha_composite(ring.filter(ImageFilter.GaussianBlur(8)))
        img.alpha_composite(lg, (x, 44 + (side - lg.height) // 2))
        x += side + 28
    if brand:
        d = ImageDraw.Draw(img)
        f = _fit(brand.upper(), XBOLD_FONT, DW - x - 60, 48, 26)
        d.text((x, 62 if copy.get("tagline") else 86), brand.upper(), font=f, fill=WHITE)
        if copy.get("tagline"):
            tag = " • ".join(p.strip().upper() for p in re.split(r"[•·|,]", copy["tagline"]) if p.strip())
            ft = _fit(tag, BOLD_FONT, DW - x - 60, 26, 16)
            spaced = " ".join(tag) if ft.getlength(" ".join(tag)) < DW - x - 60 else tag
            d.text((x + 2, 126), spaced, font=ft, fill=(190, 225, 255, 255))
    y = 205

    # -- title (white, optional icon on the left)
    if copy.get("title"):
        icon = copy.get("title_icon")
        max_w = DW - 120 - (150 if icon else 0)
        f = _fit(copy["title"], BLACK_FONT, max_w, 140, 60)
        layer, tw = _text_layer(copy["title"], f, WHITE)
        total = tw + (150 if icon else 0)
        x0 = DW / 2 - total / 2
        if icon:
            ic = Image.new("RGBA", (180, 180))
            draw_icon(ImageDraw.Draw(ic), icon, 90, 90, min(150, f.size * 1.05))
            _paste_shadow(img, ic, (x0 - 30, y + f.size * .62 - 90 + 24))
        _paste_shadow(img, layer, (x0 + (150 if icon else 0) - 24, y - 24))
        asc, desc = f.getmetrics()
        y += asc + desc - 10

    # -- the strip: black italic on a yellow brush stroke, green check on the right
    if copy.get("strip"):
        f = _fit(copy["strip"], BLACK_FONT, DW - 280, 168, 70)
        asc, desc = f.getmetrics()
        tw = f.getlength(copy["strip"])
        x0 = DW / 2 - tw / 2 - 40
        strip = Image.new("RGBA", (DW, DH))
        _brush_strip(ImageDraw.Draw(strip), (max(30, x0 - 50), y + 26, min(DW - 30, x0 + tw + 90),
                                             y + asc + desc - 6), YELLOW, rng, rough=18)
        strip = strip.rotate(-2, resample=Image.BICUBIC, center=(DW / 2, y + (asc + desc) / 2))
        img = Image.alpha_composite(img, strip)
        layer, _ = _text_layer(copy["strip"], f, INK)
        img.alpha_composite(_italic(layer), (int(x0 - 24), int(y - 24)))
        ck = Image.new("RGBA", (150, 150))
        draw_icon(ImageDraw.Draw(ck), "check", 75, 75, 104)
        _paste_shadow(img, ck, (min(DW - 150, x0 + tw + 50), y + (asc + desc) / 2 - 55))
        y += asc + desc - 4

    # -- the follow-up line (white italic)
    if copy.get("line"):
        f = _fit(copy["line"], BLACK_FONT, DW - 160, 130, 56)
        layer, tw = _text_layer(copy["line"], f, WHITE, stroke=3)
        _paste_shadow(img, _italic(layer), (DW / 2 - tw / 2 - 24, y - 24))
        asc, desc = f.getmetrics()
        y += asc + desc - 14

    # -- the split line: yellow left part, white right part with a red underline
    split = copy.get("split") or []
    if split:
        left, right = (split + [""])[:2]
        gap = 30 if right else 0
        f = _fit(f"{left} {right}".strip(), BLACK_FONT, DW - 140, 120, 52)
        lw, rw = f.getlength(left), f.getlength(right) if right else 0
        x0 = DW / 2 - (lw + gap + rw) / 2
        asc, desc = f.getmetrics()
        under = Image.new("RGBA", (DW, DH))
        if right:
            _underline(ImageDraw.Draw(under), x0 + lw + gap + 10, x0 + lw + gap + rw + 10, y + asc + desc + 4,
                       RED, width=11, rng=rng)
        img.alpha_composite(under)
        ll, _ = _text_layer(left, f, YELLOW, stroke=3)
        _paste_shadow(img, _italic(ll), (x0 - 24, y - 24))
        if right:
            rl, _ = _text_layer(right, f, WHITE, stroke=3)
            _paste_shadow(img, _italic(rl), (x0 + lw + gap - 24, y - 24))
        y += asc + desc + 20

    # -- the badge, right side under the headline
    badge = copy.get("badge")
    if badge:
        words = badge["text"].split()
        cut = (len(words) + 1) // 2
        lines = [" ".join(words[:cut]), " ".join(words[cut:])] if len(words) > 1 else [badge["text"]]
        bw, bh = 470, 160
        bx, by = DW - bw - 50, max(y + 20, 800)
        _glow_rect(img, (bx, by, bx + bw, by + bh), 34, RED, width=6, glow=14, fill=(6, 6, 12, 235))
        d = ImageDraw.Draw(img)
        d.ellipse((bx + 26, by + 26, bx + 134, by + 134), fill=(20, 20, 30, 255))
        draw_icon(d, badge["icon"], bx + 80, by + 80, 96)
        tl = Image.new("RGBA", (bw - 150, bh))
        tdr = ImageDraw.Draw(tl)
        fs = min((_fit(ln, XBOLD_FONT, bw - 190, 54, 28) for ln in lines), key=lambda f: f.size)
        for i, ln in enumerate(lines):
            yy = bh / 2 + (i - (len(lines) - 1) / 2) * fs.size * 1.1
            tdr.text(((bw - 150) / 2, yy), ln, font=fs, fill=WHITE, anchor="mm")
        img.alpha_composite(_italic(tl, 0.14), (bx + 140, by))

    # -- the four feature cards
    cards = copy.get("cards") or []
    banner = copy.get("banner") or []
    cards_top = 1400 if banner else 1560
    if cards:
        n = len(cards)
        gap, margin = 18, 36
        cw = (DW - 2 * margin - gap * (n - 1)) / n
        ch = 210
        for i, c in enumerate(cards):
            x0 = margin + i * (cw + gap)
            _glow_rect(img, (x0, cards_top, x0 + cw, cards_top + ch), 24, CYAN, width=4, glow=10,
                       fill=(0, 26, 66, 205))
            d = ImageDraw.Draw(img)
            draw_icon(d, c["icon"], x0 + cw / 2, cards_top + 62, 78)
            words = c["text"].split()
            cut = (len(words) + 1) // 2
            lines = [" ".join(words[:cut]), " ".join(words[cut:])] if len(words) > 1 else [c["text"]]
            f = min((_fit(ln, BOLD_FONT, cw - 26, 34, 18) for ln in lines), key=lambda f: f.size)
            for k, ln in enumerate(lines):
                d.text((x0 + cw / 2, cards_top + 138 + k * (f.size + 4) - (len(lines) - 1) * (f.size + 4) / 2 + 8),
                       ln, font=f, fill=WHITE, anchor="mm")

    # -- the bottom banner: yellow brush, icon on the left, two-tone italic lines
    if banner:
        top = 1640
        bh = 230
        layer = Image.new("RGBA", (DW, DH))
        _brush_strip(ImageDraw.Draw(layer), (60, top + 18, DW - 60, top + bh - 18), YELLOW, rng, rough=26)
        layer = layer.rotate(-1.5, resample=Image.BICUBIC, center=(DW / 2, top + bh / 2))
        _paste_shadow(img, layer, (0, 0), blur=16, offset=(0, 10), opacity=150)
        ic = Image.new("RGBA", (220, 220))
        draw_icon(ImageDraw.Draw(ic), copy.get("banner_icon") or "rocket", 110, 110, 150)
        _paste_shadow(img, ic, (30, top + bh / 2 - 120))
        tx0, tx1 = 230, DW - 70
        colors = (INK, BLUE_INK)
        fs = min((_fit(ln, BLACK_FONT, tx1 - tx0, 70, 30) for ln in banner), key=lambda f: f.size)
        line_h = fs.size * 1.12
        yy = top + bh / 2 - line_h * len(banner) / 2
        for i, ln in enumerate(banner):
            tl, tw = _text_layer(ln, fs, colors[i % 2])
            img.alpha_composite(_italic(tl, 0.16), (int((tx0 + tx1) / 2 - tw / 2 - 24), int(yy - 24)))
            if i == len(banner) - 1:
                ul = Image.new("RGBA", (DW, DH))
                _underline(ImageDraw.Draw(ul), (tx0 + tx1) / 2 - tw / 2 + 20, (tx0 + tx1) / 2 + tw / 2 - 20,
                           yy + fs.size * 1.2, RED, width=8, rng=rng)
                img.alpha_composite(ul)
            yy += line_h

    return img.resize(size, Image.LANCZOS)


def render(base_jpg, out_jpg, copy: dict, brand: str = "", logo=None) -> Path:
    """Draw the modern poster over a copy of base_jpg and save it as out_jpg (same size, JPEG).

    The base is rendered WITHOUT the channel logo: here the logo sits in the brand header,
    top-left, next to the channel name."""
    copy = normalize(copy)
    if not copy:
        raise ValueError("no usable modern thumbnail copy")
    with Image.open(base_jpg) as src:
        original = src.convert("RGBA")
    if not thumbcover.eligible(original.size):
        raise ValueError(f"a modern thumbnail needs a 9:16 frame, not {original.width}x{original.height}")
    out = draw(original, copy, brand=brand, logo=logo)
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(out_jpg, "JPEG", quality=93)
    return out_jpg
