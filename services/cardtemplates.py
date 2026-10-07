"""Story board card layouts: eight templates in four families, and the rules that pick one.

services/storyboard.py owns everything shared - fonts, wrapping, shadows, rounded images, the
asset lifecycle and the overlay schedule. This module owns only the look: one small function per
template, all taking (card, ctx, image) and drawing onto a full-frame transparent canvas that
video.composite lays over the clips.

Which template a card gets is decided once and frozen into plan["storyboard"][i]["template"], so a
reassemble - which happens for reasons that have nothing to do with the overlay, like toggling the
page scroll or changing the output resolution - can never restyle a finished video. The card's own
content picks the FAMILY (a card carrying a number gets a stat layout, a card with a rendered still
gets a visual one, the opener and the closer get a banner); a seeded RNG picks the VARIANT inside
it. That is what makes one video show several different cards while the same job always renders
identically.

Every template must keep its content inside the safe box (ctx.m / ctx.top_safe / ctx.bottom_safe -
portrait frames reserve room for the Shorts UI top and bottom) and must survive image=None: a
ComfyUI render can fail, and a card that loses its picture still has to read.
"""
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

import config
from services import storyboard

PANEL_FILL = (12, 14, 20, 196)
PANEL_EDGE = (255, 255, 255, 38)
BAND_FILL = (10, 12, 18, 210)
HEAD_FILL = (255, 255, 255, 255)
BODY_FILL = (214, 219, 228, 255)
NUM_FILL = (150, 156, 168, 255)
DARK_TEXT = (12, 14, 20, 255)
SCRIM_RGB = (8, 10, 14)

MOTION_SECONDS = 0.45   # longest a card takes to ease into place (the alpha fade does the rest)


# ---- the drawing context ---------------------------------------------------
@dataclass
class CardCtx:
    """Everything that is the same for every template: the frame, the scale unit, the safe box,
    the accent and the canvas. Built once by storyboard.render_card."""
    W: int
    H: int
    s: float                     # scale unit: 1.0 at 864x480 / 480x864 (storyboard.py rule)
    portrait: bool
    square: bool
    accent: tuple                # RGBA
    m: int                       # side margin
    top_safe: int
    bottom_safe: int
    gap: int
    pad: int
    bar_w: int
    radius: int
    index: int
    total: int
    img_ratio: float
    canvas: Image.Image
    draw: ImageDraw.ImageDraw
    boxes: list = field(default_factory=list)   # every placed rect, for the layout tests

    def px(self, v: float) -> int:
        return int(v * self.s)

    @property
    def safe_w(self) -> int:
        return self.W - 2 * self.m

    @property
    def safe_h(self) -> int:
        return self.H - self.top_safe - self.bottom_safe

    @property
    def floor(self) -> int:
        """y of the lowest pixel a card may touch."""
        return self.H - self.bottom_safe


def context(size: tuple, accent: str | None, index: int, total: int) -> CardCtx:
    W, H = int(size[0]), int(size[1])
    portrait = H > W * 1.05
    square = abs(W - H) <= W * 0.05
    # square frames scale by width: their panel is only ~56% wide, so the short-side rule would
    # make the text far too large (the original render_card comment, kept)
    s = W / 864.0 if square else min(W, H) / 480.0
    m = int((0.06 if portrait else 0.045) * W)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return CardCtx(W=W, H=H, s=s, portrait=portrait, square=square,
                   accent=storyboard._hex(accent or config.STORYBOARD_ACCENT),
                   m=m,
                   top_safe=int(0.08 * H) if portrait else m,
                   bottom_safe=int(0.14 * H) if portrait else m,
                   gap=int(14 * s), pad=int(18 * s), bar_w=max(3, int(5 * s)),
                   radius=int(14 * s), index=index, total=total,
                   img_ratio=storyboard.image_spec(W, H)[1],
                   canvas=canvas, draw=ImageDraw.Draw(canvas))


# ---- shared drawing helpers ------------------------------------------------
def _fonts(ctx: CardCtx, head: float = 0.0, body: float = 0.0, kick: float = 13.0,
           num: float = 12.0) -> tuple:
    """(kicker, headline, body, index-badge) fonts at template-chosen sizes."""
    head = head or (28.0 if ctx.portrait else 30.0)
    body = body or (18.0 if ctx.portrait else 19.0)
    return (storyboard.load_font(kick * ctx.s, bold=True),
            storyboard.load_font(head * ctx.s, bold=True),
            storyboard.load_font(body * ctx.s, bold=False),
            storyboard.load_font(num * ctx.s, bold=True))


def _spaced_width(draw, text: str, font, spacing: float) -> int:
    """Width storyboard._spaced will actually draw (it has no measuring counterpart)."""
    if not text:
        return 0
    return int(sum(draw.textlength(ch, font=font) for ch in text) + spacing * (len(text) - 1))


def _measure(ctx: CardCtx, card: dict, fonts: tuple, text_w: int,
             max_head: int = 2, max_body: int = 3) -> tuple:
    """(kicker, head_lines, body_lines, height) for a text stack of this width."""
    f_kick, f_head, f_body, _ = fonts
    d = ctx.draw
    kicker = (card.get("kicker") or "").strip().upper()[:28]
    head_lines = storyboard._wrap(d, card.get("headline", ""), f_head, text_w, max_head)
    body_lines = storyboard._wrap(d, card.get("body", ""), f_body, text_w, max_body)
    h = 0
    if kicker:
        h += int(f_kick.size * 1.3) + ctx.px(8)
    h += int(f_head.size * 1.15) * len(head_lines)
    if body_lines:
        h += ctx.px(8) + int(f_body.size * 1.35) * len(body_lines)
    return kicker, head_lines, body_lines, h


def _text_block(pd, ctx: CardCtx, x: int, y: int, kicker, head_lines, body_lines, fonts: tuple,
                text_w: int, align: str = "left", kicker_fill=None, head_fill=HEAD_FILL,
                body_fill=BODY_FILL, badge: str = "") -> int:
    """Draw the kicker / headline / body stack. Returns the y just below it."""
    f_kick, f_head, f_body, f_num = fonts
    kfill = ctx.accent if kicker_fill is None else kicker_fill
    spacing = 1.6 * ctx.s
    if kicker:
        kx = x + (text_w - _spaced_width(pd, kicker, f_kick, spacing)) // 2 if align == "center" else x
        storyboard._spaced(pd, (kx, y), kicker, f_kick, kfill, spacing)
        if badge:
            bw = pd.textlength(badge, font=f_num)
            pd.text((x + text_w - bw, y + 1), badge, font=f_num, fill=NUM_FILL)
        y += int(f_kick.size * 1.3) + ctx.px(8)
    for ln in head_lines:
        lx = x + int((text_w - pd.textlength(ln, font=f_head)) // 2) if align == "center" else x
        pd.text((lx, y), ln, font=f_head, fill=head_fill)
        y += int(f_head.size * 1.15)
    if body_lines:
        y += ctx.px(8)
        for ln in body_lines:
            lx = x + int((text_w - pd.textlength(ln, font=f_body)) // 2) if align == "center" else x
            pd.text((lx, y), ln, font=f_body, fill=body_fill)
            y += int(f_body.size * 1.35)
    return y


def _panel(size: tuple, radius: int, fill=PANEL_FILL, outline=PANEL_EDGE) -> Image.Image:
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle((0, 0, w - 1, h - 1), radius=max(0, radius),
                                            fill=fill, outline=outline, width=1 if outline else 0)
    return panel


def _scrim(size: tuple, stops: list, horizontal: bool = False, rgb=SCRIM_RGB) -> Image.Image:
    """A rectangle whose alpha ramps through `stops` (top->bottom, or left->right).

    Used instead of a solid panel where the footage should still show through.
    """
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    n = max(2, len(stops))
    vals = []
    for i in range(256):
        p = i * (n - 1) / 255.0
        k = min(n - 2, int(p))
        vals.append(int(round(stops[k] + (stops[k + 1] - stops[k]) * (p - k))))
    ramp = Image.new("L", (256, 1) if horizontal else (1, 256))
    ramp.putdata(vals)
    layer = Image.new("RGBA", (w, h), tuple(rgb) + (0,))
    layer.putalpha(ramp.resize((w, h), Image.BILINEAR))
    return layer


def _fit_font(draw, text: str, max_w: int, start: float, floor: float, bold: bool = True):
    """Largest font at or below `start` whose `text` fits in max_w (never below `floor`)."""
    size, step = float(start), max(1.0, start * 0.06)
    while size > floor:
        f = storyboard.load_font(size, bold=bold)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= step
    return storyboard.load_font(floor, bold=bold)


NUM_RE = re.compile(r"([€$£]?\d[\d.,]*\s?(?:%|x|×|k|K|M|B|bn|GB|TB|ms|fps|W|kW)?)")


def _pull_number(card: dict):
    """(number, remaining text) of the first figure in the headline, else the body; None if the
    card carries no number at all."""
    for key in ("headline", "body"):
        text = str(card.get(key) or "")
        mt = NUM_RE.search(text)
        if mt:
            rest = (text[:mt.start()] + " " + text[mt.end():]).strip(" ,.;:-")
            return mt.group(1).strip(), rest
    return None


def _place(ctx: CardCtx, layer: Image.Image, pos: tuple, blur: int, offset: int,
           opacity: int = 150) -> None:
    """Composite a layer with its drop shadow and record where it landed.

    ctx.boxes is what the layout tests assert on: Pillow silently clips at the canvas edge, so a
    template that overflows still writes a valid PNG - only the intended geometry reveals it."""
    x, y = int(pos[0]), int(pos[1])
    storyboard._shadowed(ctx.canvas, layer, (x, y), blur, offset, opacity)
    ctx.boxes.append((x, y, x + layer.width, y + layer.height))


def _badge(ctx: CardCtx) -> str:
    return f"{ctx.index:02d} / {ctx.total:02d}"


def _fit_image(ctx: CardCtx, image, size: tuple) -> Image.Image:
    return storyboard._rounded_image(image, size, radius=ctx.px(12), border=max(1, ctx.px(2)))


# ---- family: classic -------------------------------------------------------
def _tpl_lower_third(card: dict, ctx: CardCtx, image=None) -> None:
    """The original layout: a rounded lower-third panel with an accent side bar, and the related
    picture beside it (landscape/square) or above it (portrait).

    This is the default and the "classic only" escape hatch, so its output must not drift."""
    fonts = _fonts(ctx)
    f_kick, f_head, f_body, _ = fonts
    if ctx.portrait:
        panel_w, img_w = ctx.safe_w, int(0.8 * ctx.W)
    elif ctx.square:
        panel_w, img_w = int(0.56 * ctx.W), int(0.34 * ctx.W)
    else:
        panel_w, img_w = int(0.58 * ctx.W), int(0.30 * ctx.W)
    img_h = int(img_w / ctx.img_ratio)

    text_w = panel_w - 2 * ctx.pad - ctx.bar_w - ctx.px(12)
    kicker, head_lines, body_lines, used = _measure(ctx, card, fonts, text_w)
    panel_h = used + 2 * ctx.pad
    panel_x, panel_y = ctx.m, ctx.floor - panel_h

    panel = _panel((panel_w, panel_h), ctx.radius)
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle((ctx.pad, ctx.pad, ctx.pad + ctx.bar_w, panel_h - ctx.pad),
                         radius=ctx.bar_w // 2, fill=ctx.accent)
    _text_block(pd, ctx, ctx.pad + ctx.bar_w + ctx.px(12), ctx.pad, kicker, head_lines,
                body_lines, fonts, text_w, badge=_badge(ctx))
    _place(ctx, panel, (panel_x, panel_y), blur=ctx.px(10), offset=ctx.px(6))

    if image is not None:
        if ctx.portrait:
            room = panel_y - ctx.gap - ctx.top_safe
            if img_h > room > 0:          # shrink rather than overflow the frame
                img_h, img_w = int(room), int(room * ctx.img_ratio)
            pos = ((ctx.W - img_w) // 2, panel_y - ctx.gap - img_h)
        else:
            pos = (ctx.W - ctx.m - img_w, ctx.floor - img_h)
        if img_w > 0 and img_h > 0:
            _place(ctx, _fit_image(ctx, image, (img_w, img_h)), pos,
                   blur=ctx.px(12), offset=ctx.px(8), opacity=180)


def _tpl_ribbon(card: dict, ctx: CardCtx, image=None) -> None:
    """An edge-to-edge band across the bottom with an accent rule along its top edge - no rounded
    corners, no side bar. The picture becomes a square thumbnail at the band's trailing end."""
    fonts = _fonts(ctx, head=26.0 if ctx.portrait else 28.0)
    f_kick, f_head, f_body, f_num = fonts
    inset = ctx.m + ctx.pad
    text_w = ctx.W - 2 * inset
    thumb = 0
    if image is not None:
        thumb = ctx.px(96 if ctx.portrait else 110)
        text_w -= thumb + ctx.gap
    kicker, head_lines, body_lines, used = _measure(ctx, card, fonts, text_w, max_body=2)
    band_h = max(used + 2 * ctx.pad, thumb + 2 * ctx.pad if thumb else 0)
    band_y = ctx.floor - band_h

    band = Image.new("RGBA", (ctx.W, band_h), BAND_FILL)
    bd = ImageDraw.Draw(band)
    bd.rectangle((0, 0, ctx.W - 1, max(1, ctx.px(3))), fill=ctx.accent)
    _text_block(bd, ctx, inset, ctx.pad, kicker, head_lines, body_lines, fonts, text_w,
                badge=_badge(ctx) if not thumb else "")
    if thumb:
        side = min(thumb, band_h - 2 * ctx.pad)
        if side > 0:
            band.alpha_composite(_fit_image(ctx, image, (side, side)),
                                 (ctx.W - inset - side, (band_h - side) // 2))
    _place(ctx, band, (0, band_y), blur=ctx.px(12), offset=ctx.px(6))


# ---- family: stat ----------------------------------------------------------
def _tpl_big_stat(card: dict, ctx: CardCtx, image=None) -> None:
    """The figure is the point: a hero number in the accent colour in its own column, the rest of
    the card beside it (stacked above it in portrait)."""
    fonts = _fonts(ctx, head=22.0, body=17.0)
    f_kick, f_head, f_body, f_num = fonts
    panel_w = ctx.safe_w if ctx.portrait else int((0.70 if ctx.square else 0.62) * ctx.W)
    found = _pull_number(card)
    number = found[0] if found else (str(card.get("headline") or "").split() or [""])[0]
    rest = found[1] if found else str(card.get("headline") or "")
    side = dict(card, headline=rest or card.get("headline", ""))

    if ctx.portrait:
        col_w = panel_w - 2 * ctx.pad
        text_w = col_w
    else:
        col_w = int(0.34 * panel_w)
        text_w = panel_w - col_w - 3 * ctx.pad
    f_num_big = _fit_font(ctx.draw, number, col_w - ctx.px(6), 64 * ctx.s, 26 * ctx.s)
    kicker, head_lines, body_lines, used = _measure(ctx, side, fonts, text_w)
    num_h = int(f_num_big.size * 1.2)
    panel_h = (num_h + ctx.px(10) + used + 2 * ctx.pad if ctx.portrait
               else max(num_h, used) + 2 * ctx.pad)
    panel_y = ctx.floor - panel_h
    panel = _panel((panel_w, panel_h), ctx.radius)
    pd = ImageDraw.Draw(panel)

    if ctx.portrait:
        nx = ctx.pad + (col_w - int(pd.textlength(number, font=f_num_big))) // 2
        pd.text((nx, ctx.pad), number, font=f_num_big, fill=ctx.accent)
        rule_w = int(0.5 * panel_w)
        ry = ctx.pad + num_h + ctx.px(4)
        pd.rectangle(((panel_w - rule_w) // 2, ry, (panel_w + rule_w) // 2, ry + 1), fill=PANEL_EDGE)
        _text_block(pd, ctx, ctx.pad, ctx.pad + num_h + ctx.px(10), kicker, head_lines,
                    body_lines, fonts, text_w, align="center")
    else:
        pd.text((ctx.pad, ctx.pad), number, font=f_num_big, fill=ctx.accent)
        rx = ctx.pad + col_w + ctx.pad // 2
        pd.rectangle((rx, ctx.pad, rx + 1, panel_h - ctx.pad), fill=PANEL_EDGE)
        _text_block(pd, ctx, rx + ctx.pad // 2 + ctx.px(6), ctx.pad, kicker, head_lines,
                    body_lines, fonts, text_w, badge=_badge(ctx))
    _place(ctx, panel, (ctx.m, panel_y), blur=ctx.px(10), offset=ctx.px(6))

    if image is not None and not ctx.portrait:
        iw = int(0.22 * ctx.W)
        ih = int(iw / ctx.img_ratio)
        if ctx.m + panel_w + ctx.gap + iw <= ctx.W - ctx.m:
            _place(ctx, _fit_image(ctx, image, (iw, ih)),
                   (ctx.W - ctx.m - iw, ctx.floor - ih), blur=ctx.px(12), offset=ctx.px(8),
                   opacity=180)


def _tpl_stat_chip(card: dict, ctx: CardCtx, image=None) -> None:
    """No panel at all: an accent chip holding the number, the text beside it over a soft
    horizontal scrim so the footage still reads through."""
    fonts = _fonts(ctx, head=24.0, body=17.0)
    f_kick, f_head, f_body, _ = fonts
    found = _pull_number(card)
    number = found[0] if found else (str(card.get("headline") or "").split() or ["#"])[0]
    rest = found[1] if found else str(card.get("headline") or "")
    side = dict(card, headline=rest or card.get("headline", ""))

    chip = int(2.4 * f_head.size)
    block_w = ctx.safe_w if ctx.portrait else int(0.66 * ctx.W)
    text_w = block_w if ctx.portrait else block_w - chip - ctx.gap
    kicker, head_lines, body_lines, used = _measure(ctx, side, fonts, text_w)
    block_h = chip + ctx.gap + used if ctx.portrait else max(chip, used)

    scrim_h = block_h + 2 * ctx.pad
    scrim_w = min(ctx.W, block_w + ctx.m + 2 * ctx.pad)
    scrim_y = ctx.floor - scrim_h
    ctx.canvas.alpha_composite(_scrim((scrim_w, scrim_h), [190, 150, 0], horizontal=True), (0, scrim_y))
    ctx.boxes.append((0, scrim_y, scrim_w, scrim_y + scrim_h))

    tile = Image.new("RGBA", (chip, chip), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    td.rounded_rectangle((0, 0, chip - 1, chip - 1), radius=ctx.px(10), fill=ctx.accent)
    f_big = _fit_font(ctx.draw, number, chip - ctx.px(10), chip * 0.5, chip * 0.22)
    td.text(((chip - td.textlength(number, font=f_big)) // 2, (chip - f_big.size * 1.25) // 2),
            number, font=f_big, fill=DARK_TEXT)

    top = scrim_y + ctx.pad
    _place(ctx, tile, (ctx.m, top), blur=ctx.px(8), offset=ctx.px(4), opacity=120)
    tx = ctx.m if ctx.portrait else ctx.m + chip + ctx.gap
    ty = top + chip + ctx.gap if ctx.portrait else top + max(0, (chip - used) // 2)
    _text_block(ctx.draw, ctx, tx, ty, kicker, head_lines, body_lines, fonts, text_w,
                kicker_fill=(255, 255, 255, 180))


# ---- family: visual --------------------------------------------------------
def _tpl_split(card: dict, ctx: CardCtx, image=None) -> None:
    """The picture gets a true half of the frame, the text panel the other half."""
    if image is None:
        return _tpl_lower_third(card, ctx, None)
    fonts = _fonts(ctx, head=24.0 if ctx.portrait else 25.0, body=17.0)
    if ctx.portrait:
        img_w, img_h = ctx.safe_w, int(0.42 * ctx.H)
        img_pos = (ctx.m, ctx.top_safe)
        panel_w = ctx.safe_w
    else:
        half = ctx.W // 2 + ctx.gap // 2
        img_w, img_h = ctx.W - ctx.m - half, ctx.safe_h
        img_pos = (half, ctx.top_safe)
        panel_w = ctx.W // 2 - ctx.m - ctx.gap // 2
    _place(ctx, _fit_image(ctx, image, (max(1, img_w), max(1, img_h))), img_pos,
           blur=ctx.px(12), offset=ctx.px(8), opacity=180)

    text_w = panel_w - 2 * ctx.pad - ctx.bar_w - ctx.px(12)
    kicker, head_lines, body_lines, used = _measure(ctx, card, fonts, text_w, max_body=4)
    panel_h = used + 2 * ctx.pad
    panel = _panel((panel_w, panel_h), ctx.radius)
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle((ctx.pad, ctx.pad, ctx.pad + ctx.bar_w, panel_h - ctx.pad),
                         radius=ctx.bar_w // 2, fill=ctx.accent)
    _text_block(pd, ctx, ctx.pad + ctx.bar_w + ctx.px(12), ctx.pad, kicker, head_lines,
                body_lines, fonts, text_w, badge=_badge(ctx))
    _place(ctx, panel, (ctx.m, ctx.floor - panel_h), blur=ctx.px(10), offset=ctx.px(6))


def _tpl_photo_stack(card: dict, ctx: CardCtx, image=None) -> None:
    """Magazine look: the text panel overlaps the corner of the picture tile."""
    if image is None:
        return _tpl_lower_third(card, ctx, None)
    fonts = _fonts(ctx, head=24.0, body=17.0)
    if ctx.portrait:
        tile_w = ctx.safe_w
        tile_h = min(int(0.40 * ctx.H), ctx.safe_h // 2)
        tile_pos = (ctx.m, ctx.top_safe)
        panel_w = ctx.safe_w
    else:
        tile_w = int((0.38 if ctx.square else 0.42) * ctx.W)
        tile_h = min(int(tile_w / ctx.img_ratio), ctx.safe_h)
        tile_pos = (ctx.m, ctx.floor - tile_h)
        panel_w = int((0.56 if ctx.square else 0.52) * ctx.W)
    _place(ctx, _fit_image(ctx, image, (max(1, tile_w), max(1, tile_h))), tile_pos,
           blur=ctx.px(12), offset=ctx.px(8), opacity=180)

    text_w = panel_w - 2 * ctx.pad
    kicker, head_lines, body_lines, used = _measure(ctx, card, fonts, text_w, max_body=3)
    panel_h = used + 2 * ctx.pad
    panel = _panel((panel_w, panel_h), ctx.radius)
    pd = ImageDraw.Draw(panel)
    _text_block(pd, ctx, ctx.pad, ctx.pad, kicker, head_lines, body_lines, fonts, text_w,
                badge=_badge(ctx))
    if ctx.portrait:
        pos = (ctx.m, min(ctx.floor - panel_h, tile_pos[1] + tile_h - int(0.08 * panel_h)))
    else:
        overlap = ctx.px(24)
        px = min(ctx.m + tile_w - overlap, ctx.W - ctx.m - panel_w)
        pos = (max(ctx.m, px), ctx.floor - panel_h)
    _place(ctx, panel, pos, blur=ctx.px(10), offset=ctx.px(6))


# ---- family: banner --------------------------------------------------------
def _tpl_full_bleed(card: dict, ctx: CardCtx, image=None) -> None:
    """A statement, centred across the frame over a soft vertical scrim. No panel, no index badge -
    a banner is not an item in a list."""
    fonts = _fonts(ctx, head=30.0 if ctx.portrait else 34.0, body=18.0)
    f_kick, f_head, f_body, _ = fonts
    text_w = int(0.78 * ctx.safe_w)
    kicker, head_lines, body_lines, _used = _measure(ctx, card, fonts, text_w, max_body=2)
    rule_h = max(2, ctx.px(2))
    rule_w = int(0.12 * ctx.W)
    block_h = 0
    if kicker:
        block_h += int(f_kick.size * 1.3) + ctx.px(8) + rule_h + ctx.px(10)
    block_h += int(f_head.size * 1.15) * len(head_lines)
    if body_lines:
        block_h += ctx.px(10) + int(f_body.size * 1.35) * len(body_lines)

    scrim_h = min(block_h + 4 * ctx.pad, ctx.safe_h)
    scrim_y = ctx.top_safe + (ctx.safe_h - scrim_h) // 2
    ctx.canvas.alpha_composite(_scrim((ctx.W, scrim_h), [0, 200, 200, 0]), (0, scrim_y))
    ctx.boxes.append((0, scrim_y, ctx.W, scrim_y + scrim_h))

    pd = ctx.draw
    x = (ctx.W - text_w) // 2
    y = scrim_y + (scrim_h - block_h) // 2
    if kicker:
        kw = _spaced_width(pd, kicker, f_kick, 1.6 * ctx.s)
        storyboard._spaced(pd, ((ctx.W - kw) // 2, y), kicker, f_kick, ctx.accent, 1.6 * ctx.s)
        y += int(f_kick.size * 1.3) + ctx.px(8)
        pd.rectangle(((ctx.W - rule_w) // 2, y, (ctx.W + rule_w) // 2, y + rule_h), fill=ctx.accent)
        y += rule_h + ctx.px(10)
    for ln in head_lines:
        pd.text(((ctx.W - pd.textlength(ln, font=f_head)) // 2, y), ln, font=f_head, fill=HEAD_FILL)
        y += int(f_head.size * 1.15)
    if body_lines:
        y += ctx.px(10)
        for ln in body_lines:
            pd.text(((ctx.W - pd.textlength(ln, font=f_body)) // 2, y), ln, font=f_body,
                    fill=BODY_FILL)
            y += int(f_body.size * 1.35)
    ctx.boxes.append((x, scrim_y + (scrim_h - block_h) // 2, x + text_w, y))


def _tpl_quote_slab(card: dict, ctx: CardCtx, image=None) -> None:
    """A pulled-out line: a big accent quote mark behind the text, on a slab that sits higher than
    the usual lower third. The quote mark replaces the kicker."""
    fonts = _fonts(ctx, head=28.0 if ctx.portrait else 30.0, body=18.0)
    f_kick, f_head, f_body, f_num = fonts
    panel_w = ctx.safe_w if ctx.portrait else int((0.56 if ctx.square else 0.60) * ctx.W)
    text_w = panel_w - 2 * ctx.pad
    rule_h = max(2, ctx.px(4))
    plain = dict(card, kicker="")
    _k, head_lines, body_lines, used = _measure(ctx, plain, fonts, text_w, max_body=3)
    panel_h = used + 2 * ctx.pad + rule_h + ctx.px(10)
    panel_x = ctx.m + (0 if ctx.portrait else int(0.08 * ctx.W))
    panel_x = min(panel_x, ctx.W - ctx.m - panel_w)
    panel_y = min(int(0.62 * ctx.H) - panel_h, ctx.floor - panel_h)
    panel_y = max(ctx.top_safe, panel_y)

    f_quote = storyboard.load_font(96 * ctx.s, bold=True)
    qx, qy = panel_x + ctx.px(6), max(0, panel_y - ctx.px(30))
    ctx.draw.text((qx, qy), "“", font=f_quote, fill=ctx.accent[:3] + (90,))

    panel = _panel((panel_w, panel_h), ctx.radius)
    pd = ImageDraw.Draw(panel)
    y = ctx.pad
    for ln in head_lines:
        pd.text((ctx.pad, y), ln, font=f_head, fill=HEAD_FILL)
        y += int(f_head.size * 1.15)
    y += ctx.px(6)
    pd.rectangle((ctx.pad, y, ctx.pad + int(0.28 * panel_w), y + rule_h), fill=ctx.accent)
    y += rule_h + ctx.px(10)
    for ln in body_lines:
        pd.text((ctx.pad, y), ln, font=f_body, fill=BODY_FILL)
        y += int(f_body.size * 1.35)
    badge = _badge(ctx)
    bw = pd.textlength(badge, font=f_num)
    pd.text((panel_w - ctx.pad - bw, panel_h - ctx.pad - f_num.size * 1.2), badge, font=f_num,
            fill=NUM_FILL)
    _place(ctx, panel, (panel_x, panel_y), blur=ctx.px(10), offset=ctx.px(6))


# ---- the registry ----------------------------------------------------------
# Order is the RNG's domain and must stay stable: a set anywhere in the pick would silently
# destroy reproducibility across processes.
FAMILIES = {
    "classic": ("lower_third", "ribbon"),
    "stat": ("big_stat", "stat_chip"),
    "visual": ("split", "photo_stack"),
    "banner": ("full_bleed", "quote_slab"),
}
TEMPLATES = {
    "lower_third": _tpl_lower_third,
    "ribbon": _tpl_ribbon,
    "big_stat": _tpl_big_stat,
    "stat_chip": _tpl_stat_chip,
    "split": _tpl_split,
    "photo_stack": _tpl_photo_stack,
    "full_bleed": _tpl_full_bleed,
    "quote_slab": _tpl_quote_slab,
}
DEFAULT_TEMPLATE = "lower_third"

# The kickers llm.write_storyboard asks for (services/llm.py). An unknown kicker is fine - it just
# falls through to the content rules in _family().
KICKER_FAMILY = {
    "BY THE NUMBERS": "stat", "THE NUMBERS": "stat", "STATS": "stat",
    "TAKEAWAY": "banner", "IN SHORT": "banner", "QUICK RECAP": "banner",
    "WHY IT MATTERS": "banner", "THE POINT": "banner",
    "KEY FACT": "classic", "DID YOU KNOW": "classic", "HOW IT WORKS": "classic",
    "PRO TIP": "classic", "WHAT'S NEW": "classic",
}

# template -> (axis, direction, fraction of the frame dimension it travels)
MOTION = {
    "lower_third": ("y", 1, 0.05),     # rises into place
    "ribbon": ("x", -1, 0.10),         # slides in from the leading edge
    "big_stat": ("y", 1, 0.04),
    "stat_chip": ("x", -1, 0.08),
    "split": ("x", 1, 0.08),
    "photo_stack": ("y", 1, 0.05),
    "full_bleed": None,                # a full-width scrim that slides reads as a glitch
    "quote_slab": ("y", 1, 0.06),
}


# ---- the two cards that are not story board cards --------------------------
def render_hook(text: str, size: tuple, out_png: Path, accent: str | None = None,
                kicker: str = "") -> Path:
    """The opening card: the hook in big type over a soft scrim, centred.

    The hook is already the first spoken line (services/llm.py forces it), so this is not new
    information - it is the same line for the viewer who has not pressed unmute yet, which on a
    silent autoplay feed is most of them."""
    ctx = context(size, accent, 1, 1)
    fonts = _fonts(ctx, head=38.0 if ctx.portrait else 42.0, body=18.0)
    f_kick, f_head, _f_body, _ = fonts
    text_w = int(0.84 * ctx.safe_w)
    lines = storyboard._wrap(ctx.draw, text or "", f_head, text_w, 3)
    lh = int(f_head.size * 1.15)
    block = lh * len(lines)
    kick = (kicker or "").strip().upper()[:28]
    if kick:
        block += int(f_kick.size * 1.3) + ctx.px(10)
    scrim_h = min(block + 6 * ctx.pad, ctx.safe_h)
    scrim_y = ctx.top_safe + (ctx.safe_h - scrim_h) // 2
    ctx.canvas.alpha_composite(_scrim((ctx.W, scrim_h), [0, 215, 215, 0]), (0, scrim_y))
    ctx.boxes.append((0, scrim_y, ctx.W, scrim_y + scrim_h))
    pd = ctx.draw
    y = scrim_y + (scrim_h - block) // 2
    if kick:
        kw = _spaced_width(pd, kick, f_kick, 1.6 * ctx.s)
        storyboard._spaced(pd, ((ctx.W - kw) // 2, y), kick, f_kick, ctx.accent, 1.6 * ctx.s)
        y += int(f_kick.size * 1.3) + ctx.px(10)
    for ln in lines:
        pd.text(((ctx.W - pd.textlength(ln, font=f_head)) // 2, y), ln, font=f_head, fill=HEAD_FILL)
        y += lh
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ctx.canvas.save(out_png, "PNG")
    return out_png


def render_end(size: tuple, out_png: Path, channel: str = "", logo=None, line: str = "",
               accent: str | None = None) -> Path:
    """The closing card: logo, channel name and a subscribe line.

    Kept clear of the lower-right quarter, where YouTube draws its own end-screen elements."""
    ctx = context(size, accent, 1, 1)
    fonts = _fonts(ctx, head=30.0, body=19.0)
    _f_kick, f_head, f_body, _ = fonts
    ctx.canvas.alpha_composite(_scrim((ctx.W, ctx.H), [150, 215]), (0, 0))
    ctx.boxes.append((0, 0, ctx.W, ctx.H))
    pd = ctx.draw
    # the safe box for this card stops before YouTube's end-screen area
    avail_h = int(ctx.H * (0.62 if not ctx.portrait else 0.74))
    logo_img = None
    if logo is not None and Path(logo).exists():
        try:
            lw = int(0.16 * ctx.W)
            with Image.open(logo) as lg:
                lg = lg.convert("RGBA")
                logo_img = lg.resize((lw, max(1, int(lg.height * lw / lg.width))), Image.LANCZOS)
        except Exception as e:  # noqa: BLE001 - the card is fine without the logo
            print(f"end card logo skipped: {e}")
    name = (channel or "").strip()
    line = (line or "").strip()
    block = (logo_img.height + ctx.px(16) if logo_img else 0)
    block += int(f_head.size * 1.2) if name else 0
    block += int(f_body.size * 1.5) if line else 0
    y = ctx.top_safe + max(0, (avail_h - ctx.top_safe - block) // 2)
    if logo_img:
        ctx.canvas.alpha_composite(logo_img, ((ctx.W - logo_img.width) // 2, y))
        ctx.boxes.append(((ctx.W - logo_img.width) // 2, y,
                          (ctx.W + logo_img.width) // 2, y + logo_img.height))
        y += logo_img.height + ctx.px(16)
    if name:
        pd.text(((ctx.W - pd.textlength(name, font=f_head)) // 2, y), name, font=f_head,
                fill=HEAD_FILL)
        y += int(f_head.size * 1.2)
    if line:
        pd.text(((ctx.W - pd.textlength(line, font=f_body)) // 2, y), line, font=f_body,
                fill=ctx.accent)
        y += int(f_body.size * 1.5)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ctx.canvas.save(out_png, "PNG")
    return out_png


def _family(card: dict, index: int, total: int) -> str:
    """Which layout family this card's CONTENT asks for. The order below is the priority order."""
    if (card.get("image_prompt") or "").strip():      # a rendered still is the strongest signal
        return "visual"
    kick = (card.get("kicker") or "").strip().upper()
    fam = KICKER_FAMILY.get(kick, "")
    if fam == "banner" or index in (1, total):        # the opener names the topic, the closer sums up
        return "banner"
    if fam == "stat" or _pull_number(card):
        return "stat"
    return fam or "classic"


def enabled_templates() -> tuple:
    """Template names the current Settings template set allows, in TEMPLATES order."""
    allow = config.STORYBOARD_TEMPLATE_SETS.get(config.STORYBOARD_TEMPLATE_SET) or ()
    return tuple(t for t in TEMPLATES if not allow or t in allow)


def pick_templates(cards: list, seed: str) -> list:
    """One template name per card: the content picks the family, a per-card RNG picks the variant
    inside it, and a variant equal to the previous card's rotates on so no layout is used twice in
    a row. Deterministic in (cards, seed, enabled set).

    The RNG is seeded per CARD rather than walked across the list, so re-planning that adds or
    removes one card does not reshuffle every other card's layout."""
    allowed = enabled_templates() or (DEFAULT_TEMPLATE,)
    out, prev, n = [], "", len(cards or [])
    for i, c in enumerate(cards or [], 1):
        fam = _family(c if isinstance(c, dict) else {}, i, n)
        opts = tuple(t for t in FAMILIES.get(fam, ()) if t in allowed) or allowed
        pick = opts[random.Random(f"{seed}|{i}|{fam}").randrange(len(opts))]
        if pick == prev and len(opts) > 1:
            pick = opts[(opts.index(pick) + 1) % len(opts)]
        out.append(pick)
        prev = pick
    return out


def motion_xy(template: str, size: tuple, start: float, end: float) -> tuple:
    """(x, y) for a video.composite layer: ffmpeg per-frame expressions easing the card into place
    over at most MOTION_SECONDS from its start, or (0, 0) for a card that only fades.

    `t` in an overlay position is the main stream's timestamp - the same clock the enable window
    and the alpha fade already use - so the movement, the fade-in and the card's window stay in
    phase without any extra bookkeeping. The travel is small on purpose (4-10% of a frame
    dimension): the card pops into place while the existing fade does the appearing."""
    spec = MOTION.get(template) if config.STORYBOARD_MOTION else None
    if not spec:
        return 0, 0
    axis, sign, frac = spec
    start, end = float(start), float(end)
    # mirror composite()'s own fade clamp so a short card is never still moving as it fades out
    d = min(MOTION_SECONDS, max(0.1, (end - start) / 3))
    px = int(sign * frac * (size[0] if axis == "x" else size[1]))
    if not px:
        return 0, 0
    expr = f"{px}*pow(1-clip((t-{start:.3f})/{d:.3f},0,1),3)"   # ease-out cubic
    return (expr, 0) if axis == "x" else (0, expr)
