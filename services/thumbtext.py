"""Thumbnail overlay text: the punchy line drawn over the generated thumbnail.

The picture is deliberately generated WITHOUT letters - image models spell badly, and the prompt
in services/llm.py forbids typography outright - so the text is drawn here afterwards, with the
same Pillow primitives the story board cards use. That is also what makes variants free: one GPU
render, several text treatments over copies of it.

Every layout keeps out of the top-right corner, where services/video.make_thumbnail pastes the
channel logo at 16% of the width.
"""
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

import config
from services import storyboard

LAYOUTS = ("band", "stack", "strip")
LOGO_RESERVE = 0.26          # fraction of the width the logo corner owns, left untouched
SCRIM_RGB = (8, 10, 14)

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


def _fit(draw, text: str, max_w: int, start: float, floor: float):
    size, step = float(start), max(1.0, start * 0.05)
    while size > floor:
        f = storyboard.load_font(size, bold=True)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= step
    return storyboard.load_font(floor, bold=True)


def _scrim(size: tuple, a0: int, a1: int) -> Image.Image:
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    ramp = Image.new("L", (1, 256))
    ramp.putdata([int(a0 + (a1 - a0) * i / 255) for i in range(256)])
    layer = Image.new("RGBA", (w, h), SCRIM_RGB + (0,))
    layer.putalpha(ramp.resize((w, h), Image.BILINEAR))
    return layer


def _lines(draw, text: str, font, max_w: int) -> list:
    return storyboard._wrap(draw, text, font, max_w, 2) or [text]


def _band(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """A full-width gradient strip across the bottom with the line on it. Works on any picture."""
    W, H = base.size
    d = ImageDraw.Draw(base)
    pad = int(0.04 * W)
    f = _fit(d, text, W - 2 * pad, 0.135 * H, 0.055 * H)
    lines = _lines(d, text, f, W - 2 * pad)
    lh = int(f.size * 1.12)
    rule = max(3, int(0.012 * H))
    # the gap has to clear the descenders of the last line, which reach below lh at this size -
    # too tight and the accent rule strikes through the text instead of underlining it
    gap = int(0.045 * H)
    block = lh * len(lines) + gap + rule
    band_h = min(int(block + 2 * pad), H)
    base.alpha_composite(_scrim((W, band_h), 0, 235), (0, H - band_h))
    y = H - band_h + (band_h - block) // 2
    for ln in lines:
        d.text(((W - d.textlength(ln, font=f)) // 2, y), ln, font=f, fill=(255, 255, 255, 255))
        y += lh
    rw = int(0.18 * W)
    d.rectangle(((W - rw) // 2, y + gap - rule, (W + rw) // 2, y + gap), fill=accent)
    return base


def _stack(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """Each word-group on its own accent slab, down the left. Needs a quiet left side."""
    W, H = base.size
    d = ImageDraw.Draw(base)
    pad = int(0.045 * W)
    col = int(0.56 * W)
    f = _fit(d, text, col, 0.155 * H, 0.06 * H)
    lines = _lines(d, text, f, col)
    lh = int(f.size * 1.1)
    slab_pad = int(0.018 * W)
    total = (lh + 2 * slab_pad) * len(lines) + int(0.012 * H) * (len(lines) - 1)
    y = (H - total) // 2
    for ln in lines:
        tw = int(d.textlength(ln, font=f))
        sw, sh = tw + 2 * slab_pad, lh + 2 * slab_pad
        slab = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        ImageDraw.Draw(slab).rounded_rectangle((0, 0, sw - 1, sh - 1),
                                               radius=int(0.012 * W), fill=accent)
        ImageDraw.Draw(slab).text((slab_pad, slab_pad - int(f.size * 0.04)), ln, font=f,
                                  fill=(12, 14, 20, 255))
        storyboard._shadowed(base, slab, (pad, y), blur=int(0.012 * W), offset=int(0.008 * W), opacity=170)
        y += sh + int(0.012 * H)
    return base


def _strip(base: Image.Image, text: str, accent: tuple) -> Image.Image:
    """A thin accent ribbon low-left under one big line. Reads well on portrait."""
    W, H = base.size
    d = ImageDraw.Draw(base)
    pad = int(0.05 * W)
    col = W - 2 * pad
    f = _fit(d, text, col, 0.145 * H, 0.055 * H)
    lines = _lines(d, text, f, col)
    lh = int(f.size * 1.12)
    block = lh * len(lines)
    bar = max(4, int(0.016 * H))
    y = H - int(0.09 * H) - block - bar
    base.alpha_composite(_scrim((W, int(block + bar + 0.16 * H)), 0, 215),
                         (0, max(0, y - int(0.05 * H))))
    d.rectangle((pad, y, pad + int(0.22 * W), y + bar), fill=accent)
    y += bar + int(0.025 * H)
    for ln in lines:
        d.text((pad, y), ln, font=f, fill=(255, 255, 255, 255))
        y += lh
    return base


_DRAW = {"band": _band, "stack": _stack, "strip": _strip}


def render(base_jpg, out_jpg, text: str, layout: str = "band", accent: str | None = None,
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
    _DRAW.get(layout, _band)(base, text, _accent(accent))
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


def variants(base_jpg, job_dir, texts, accent: str | None = None, offset: int = 0) -> list:
    """Render one variant per text and return [{file, text, layout}].

    A layout that raises is skipped rather than fatal, so one bad treatment never costs the
    others.

    `offset` is how many variants already exist in the caller's list, so a second pass over a
    different base (the presenter thumbnail) numbers its files after the first rather than
    overwriting them. Clearing leftovers is therefore the caller's job - see drop_stale."""
    out = []
    for i, text in enumerate(list(texts or [])):
        layout = LAYOUTS[i % len(LAYOUTS)]
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
    base, plus room for the presenter's own text-free entry."""
    for stale in range(keep + 1, 2 * len(LAYOUTS) + 4):
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
