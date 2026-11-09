"""Thumbnail overlay text: the punchy line drawn over the generated thumbnail.

The picture is deliberately generated WITHOUT letters - image models spell badly, and the prompt
in services/llm.py forbids typography outright - so the text is drawn here afterwards, with the
same Pillow primitives the story board cards use. That is also what makes variants free: one GPU
render, several text treatments over copies of it.

Text only ever sits at the TOP of the frame or in the MIDDLE - the bottom is where a phone's
progress bar, the duration badge and the presenter's corner are, and the picture prompt keeps the
top third quiet on purpose so a headline has somewhere to go. Every layout keeps out of the
top-right corner, where services/video.make_thumbnail pastes the channel logo at 16% of the width.
"""
import re
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

import config
from services import storyboard

LAYOUTS = ("punch", "burst", "sticker")
TOP_LAYOUTS = ("punch", "sticker")   # the presenter stands in a bottom corner: keep clear of the middle
LOGO_RESERVE = 0.26          # fraction of the width the logo corner owns, left untouched
SCRIM_RGB = (8, 10, 14)
INK = (12, 14, 20, 255)
WHITE = (255, 255, 255, 255)

# (max_words, max_chars) of the overlay line, by channel audience. A kids thumbnail is scanned by
# an adult holding a phone for a child who cannot read at all, so the line has to work as a shape:
# two or three big words fill the frame, five small ones are a caption nobody finishes. Every other
# audience keeps the original caps, which assume a viewer who does read the line.
CAPS = {"kids": (3, 18)}
DEFAULT_CAPS = (5, 24)


def caps(channel: dict | None = None) -> tuple:
    """(max_words, max_chars) for this channel's thumbnail line.

    One source of truth: services.llm.write_thumbnail asks the model for a line of this length and
    normalize() then enforces it, so the prompt and the cap can never drift apart."""
    return CAPS.get(str((channel or {}).get("audience") or ""), DEFAULT_CAPS)


def normalize(text: str, max_words: int = DEFAULT_CAPS[0], max_chars: int = DEFAULT_CAPS[1]) -> str:
    """One clean overlay line: no quotes, no trailing full stop, at most a few short words."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.strip("\"'“”‘’").strip()
    t = re.sub(r"[.…]+$", "", t).strip()
    words = t.split()
    if len(words) > max_words:
        words = words[:max_words]
    t = " ".join(words)
    while len(t) > max_chars and len(words) > 1:
        words.pop()
        t = " ".join(words)
    return t[:max_chars].strip()


def variant_path(job_dir, k: int) -> Path:
    return Path(job_dir) / f"thumb_var_{k:02d}.jpg"


def _accent(accent: str | None) -> tuple:
    return storyboard._hex(accent or config.THUMB_TEXT_ACCENT or config.STORYBOARD_ACCENT)


def _logo_box(W: int, H: int) -> tuple:
    """(x0, y1): the logo corner is x >= x0 and y < y1 - the box render() restores afterwards."""
    return W - int(LOGO_RESERVE * W), min(H, int(LOGO_RESERVE * W))


def _keyword(text: str) -> str:
    """The word drawn in the accent colour: the first one carrying a figure, else the longest.
    Nothing for a one-word line - a single word in the accent is just a different colour."""
    words = text.split()
    if len(words) < 2:
        return ""
    for w in words:
        if any(ch.isdigit() for ch in w):
            return w
    return max(words, key=len)


def _stroke(font) -> int:
    return max(2, int(font.size * 0.075))


def _fit_lines(draw, text: str, max_w: int, max_lines: int, start: float, floor: float,
               max_h: float | None = None, line_h: float = 1.05):
    """(font, lines) at the largest size where `text` wraps into at most max_lines whole-word
    lines that fit max_w with their stroke, and the block fits max_h. Bigger and two lines beats
    smaller and one: a thumbnail headline is a shape first and a sentence second."""
    size, step = float(start), max(1.0, start * 0.05)
    while True:
        f = storyboard.load_font(size, bold=True)
        w = max_w - 2 * _stroke(f)
        lines = storyboard._wrap(draw, text, f, w, max_lines)
        whole = " ".join(lines) == text            # no word split, no ellipsis
        if (whole and (max_h is None or f.size * line_h * len(lines) <= max_h)) or size <= floor:
            return f, lines
        size = max(floor, size - step)


def _words(draw, xy: tuple, line: str, font, fill, accent, keyword: str, used: list):
    """Draw one stroked line word by word, the keyword (first occurrence only) in the accent."""
    x, y = xy
    sw = _stroke(font)
    space = draw.textlength(" ", font=font)
    for w in line.split():
        hit = keyword and not used and w == keyword
        if hit:
            used.append(w)
        draw.text((x, y), w, font=font, fill=accent if hit else fill,
                  stroke_width=sw, stroke_fill=INK)
        x += draw.textlength(w, font=font) + space


def _line_w(draw, line: str, font) -> int:
    return int(draw.textlength(line, font=font)) + 2 * _stroke(font)


def _ramp(size: tuple, stops: tuple, horizontal: bool) -> Image.Image:
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    n = len(stops)
    vals = []
    for i in range(256):
        p = i * (n - 1) / 255.0
        k = min(n - 2, int(p))
        vals.append(int(stops[k] + (stops[k + 1] - stops[k]) * (p - k)))
    r = Image.new("L", (256, 1) if horizontal else (1, 256))
    r.putdata(vals)
    return r.resize((w, h), Image.BILINEAR)


def _scrim(size: tuple, vstops: tuple, hstops: tuple | None = None) -> Image.Image:
    """A dark layer whose alpha ramps through vstops top->bottom, times hstops left->right."""
    alpha = _ramp(size, vstops, False)
    if hstops:
        alpha = ImageChops.multiply(alpha, _ramp(size, hstops, True))
    layer = Image.new("RGBA", alpha.size, SCRIM_RGB + (0,))
    layer.putalpha(alpha)
    return layer


def _punch(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """Huge stroked capitals across the top-left, one word in the accent, over a scrim that fades
    downwards and away from the logo corner. The classic big-headline thumbnail."""
    W, H = base.size
    d = ImageDraw.Draw(base)
    pad = int(0.045 * W)
    x0, _ = _logo_box(W, H)
    col = x0 - 2 * pad
    text = text.upper()
    tall = H > W * 1.05                       # a narrow column: more lines, not smaller letters
    f, lines = _fit_lines(d, text, col, 3 if tall else 2, 0.19 * min(W, H), 0.06 * min(W, H),
                          max_h=(0.3 if tall else 0.36) * H)
    lh = int(f.size * 1.05)
    top = int(0.045 * H)
    block = lh * len(lines)
    sh = min(H, int(top + block + 0.2 * H))
    base.alpha_composite(_scrim((W, sh), (215, 170, 0), (255, 255, 120, 0)), (0, 0))
    y, used, key = top, [], _keyword(text)
    for ln in lines:
        _words(d, (pad, y), ln, f, WHITE, accent, key, used)
        y += lh
    return base


def _burst(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """A tilted block of giant stroked capitals in the middle of the frame, lines alternating
    white and accent, on a heavy shadow. Loud on purpose - it is the one that stops a scroll."""
    W, H = base.size
    pad = int(0.05 * W)
    x0, y1 = _logo_box(W, H)
    text = text.upper()
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    max_h = 0.34 * H
    centre_y = int(0.47 * H)
    # a block that would reach up into the logo band moves its column left of the logo instead
    reaches_logo = centre_y - max_h / 2 - 0.04 * H < y1
    left, right = pad, (x0 - pad) if reaches_logo else (W - pad)
    col = int(0.94 * (right - left))                   # room for the tilt
    f, lines = _fit_lines(probe, text, col, 3, 0.2 * min(W, H), 0.06 * min(W, H), max_h=max_h)
    lh = int(f.size * 1.05)
    sw = _stroke(f)
    bw = max(_line_w(probe, ln, f) for ln in lines)
    bh = lh * len(lines) + sw + int(0.25 * f.size)
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    y = 0
    for i, ln in enumerate(lines):
        lw = _line_w(ld, ln, f)
        ld.text(((bw - lw) // 2 + sw, y), ln, font=f, fill=accent if i % 2 else WHITE,
                stroke_width=sw, stroke_fill=INK)
        y += lh
    layer = layer.rotate(4, resample=Image.BICUBIC, expand=True)
    x = left + max(0, (right - left - layer.width) // 2)
    y = max(int(0.2 * H), centre_y - layer.height // 2)
    storyboard._shadowed(base, layer, (x, y), blur=int(0.012 * W), offset=int(0.008 * W), opacity=200)
    return base


def _sticker(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """The line on a tilted accent sticker with a white border, top-left - the hand-slapped label
    look ("WAIT, WHAT?") that reads as a person talking rather than a caption."""
    W, H = base.size
    pad = int(0.05 * W)
    x0, _ = _logo_box(W, H)
    text = text.upper()
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    inner = int(0.025 * min(W, H))
    border = max(3, int(0.008 * min(W, H)))
    col = int(0.9 * (x0 - 2 * pad)) - 2 * (inner + border)
    f, lines = _fit_lines(probe, text, col, 2, 0.15 * min(W, H), 0.05 * min(W, H), max_h=0.3 * H,
                          line_h=1.1)
    lh = int(f.size * 1.1)
    tw = max(int(probe.textlength(ln, font=f)) for ln in lines)
    sw_, sh_ = tw + 2 * (inner + border), lh * len(lines) + 2 * (inner + border) + int(0.15 * f.size)
    slab = Image.new("RGBA", (sw_, sh_), (0, 0, 0, 0))
    sd = ImageDraw.Draw(slab)
    r = int(0.18 * f.size)
    sd.rounded_rectangle((0, 0, sw_ - 1, sh_ - 1), radius=r, fill=WHITE)
    sd.rounded_rectangle((border, border, sw_ - 1 - border, sh_ - 1 - border),
                         radius=max(0, r - border), fill=accent)
    y = border + inner
    for ln in lines:
        sd.text(((sw_ - sd.textlength(ln, font=f)) // 2, y), ln, font=f, fill=INK)
        y += lh
    slab = slab.rotate(5, resample=Image.BICUBIC, expand=True)
    storyboard._shadowed(base, slab, (pad, int(0.05 * H)), blur=int(0.012 * W),
                         offset=int(0.01 * W), opacity=190)
    return base


_DRAW = {"punch": _punch, "burst": _burst, "sticker": _sticker}


def render(base_jpg, out_jpg, text: str, layout: str = "punch", accent: str | None = None,
           logo_reserve: float = LOGO_RESERVE) -> Path:
    """Draw `text` over a copy of base_jpg and save it as out_jpg (same size, JPEG).

    The top-right logo corner is restored from the original afterwards, so no layout can ever
    creep under the channel logo however the text wraps."""
    text = normalize(text)
    if not text:
        raise ValueError("no thumbnail text")
    with Image.open(base_jpg) as src:
        original = src.convert("RGBA")
    base = original.copy()
    _DRAW.get(layout, _punch)(base, text, _accent(accent))
    if logo_reserve > 0:                      # keep the logo corner pristine
        W, H = base.size
        box = (W - int(logo_reserve * W), 0, W, int(logo_reserve * W))
        base.paste(original.crop(box), box[:2])
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_jpg, "JPEG", quality=92)
    return out_jpg


def with_presenter(base_jpg, cutout_png, out_jpg, style: str = "", corner: str = "") -> Path:
    """Composite the presenter cut-out onto a copy of the thumbnail base.

    Placed with video.avatar_box at the thumbnail's own size, which is the same function the
    video overlay uses, so the presenter lands in the same corner at the same fraction of the
    frame - a viewer who clicks the thumbnail sees the picture it promised. The caption
    reserve is zero here: a thumbnail has no subtitles to clear.

    The logo corner is left alone by construction, because the presenter is anchored to the
    bottom of the frame and the logo sits top-right."""
    from PIL import Image
    from services import video

    base_jpg, out_jpg = Path(base_jpg), Path(out_jpg)
    base = Image.open(base_jpg).convert("RGB")
    cut = Image.open(cutout_png).convert("RGBA")
    x, y, w, h = video.avatar_box(base.size, style or config.DEFAULT_AVATAR_STYLE, corner=corner)
    if w < 2 or h < 2:
        raise ValueError("no room for the presenter on this thumbnail")
    sc = max(w / cut.width, h / cut.height)      # cover the box, then centre-crop to it
    cut = cut.resize((max(1, round(cut.width * sc)), max(1, round(cut.height * sc))),
                     Image.LANCZOS)
    left, top = (cut.width - w) // 2, (cut.height - h) // 2
    cut = cut.crop((left, top, left + w, top + h))
    out = base.convert("RGBA")
    out.alpha_composite(cut, (x, y))
    out.convert("RGB").save(out_jpg, "JPEG", quality=92)
    return out_jpg


def variants(base_jpg, job_dir, texts, accent: str | None = None, offset: int = 0,
             layouts: tuple = LAYOUTS) -> list:
    """Render one variant per text and return [{file, text, layout}].

    A layout that raises is skipped rather than fatal, so one bad treatment never costs the
    others.

    `offset` is how many variants already exist in the caller's list, so a second pass over a
    different base (the presenter thumbnail) numbers its files after the first rather than
    overwriting them. Clearing leftovers is therefore the caller's job - see drop_stale.

    `layouts` is the rotation to draw from - TOP_LAYOUTS for the presenter base, whose bottom
    corner figure would collide with a headline in the middle."""
    out = []
    layouts = tuple(layouts) or LAYOUTS
    for i, text in enumerate(list(texts or [])):
        layout = layouts[i % len(layouts)]
        k = offset + len(out) + 1
        try:
            path = render(base_jpg, variant_path(job_dir, k), text, layout=layout, accent=accent)
            out.append({"file": path.name, "text": normalize(text), "layout": layout})
        except Exception as e:  # noqa: BLE001 - the clean thumbnail is still perfectly usable
            print(f"thumbnail variant {layout} skipped: {e}")
    return out


def drop_stale(job_dir, keep: int):
    """Delete variant files past `keep`, left by a pass that produced more of them.

    Generous upper bound rather than a glob: every layout, for both the clean and the presenter
    base, plus room for the presenter's own text-free entry and the reel cover."""
    for stale in range(keep + 1, 2 * len(LAYOUTS) + 5):     # +1 for the reel cover
        variant_path(job_dir, stale).unlink(missing_ok=True)


AUTO = -1    # "nobody has chosen yet" - distinct from 0, which is a deliberate "no text please"


def activate(job_dir, recs: list, k: int) -> int:
    """Copy variant k over thumbnail.jpg and return the variant actually used.

    k is 1-based; 0 means the clean render with no text, and AUTO means nobody has picked yet, in
    which case the model's own first line wins. Keeping 0 and AUTO distinct is what lets a user
    turn the text off and have that stick across a reassembly.

    thumbnail.jpg keeps its exact existing meaning - it IS the active thumbnail - so the download
    route, the publish bundle and the YouTube upload need no knowledge of variants at all."""
    d = Path(job_dir)
    base = d / "thumb_base.jpg"
    recs = recs or []
    k = int(k if k is not None else AUTO)
    if k < 0 or k > len(recs):                       # auto, or a stale index after a re-plan
        k = 1 if recs else 0
    if k and not (d / recs[k - 1]["file"]).exists():  # the chosen file was deleted
        k = 1 if recs and (d / recs[0]["file"]).exists() else 0
    if k and not (d / recs[k - 1]["file"]).exists():
        k = 0
    src = (d / recs[k - 1]["file"]) if k else base
    if src.exists():
        shutil.copyfile(src, d / "thumbnail.jpg")
    return k
