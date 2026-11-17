"""Render the repository's PNG brand assets (logo + GitHub social preview card) with Pillow.

    python scripts/make_brand_assets.py

The vector logo lives in docs/assets/logo.svg; this draws the same mark so the PNGs need no SVG
renderer. Output: docs/assets/logo.png (512x512) and docs/assets/social-preview.png (1280x640).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
FONTS = ROOT / "fonts" / "cover"
STOPS = [(0.0, (109, 91, 255)), (0.55, (177, 77, 255)), (1.0, (255, 107, 107))]


def _lerp(t: float):
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return tuple(round(a + (b - a) * k) for a, b in zip(c0, c1))
    return STOPS[-1][1]


def gradient(w: int, h: int) -> Image.Image:
    """Diagonal gradient, top-left to bottom-right."""
    small = Image.new("RGB", (256, 256))
    px = small.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = _lerp((x + y) / 510)
    return small.resize((w, h), Image.BICUBIC)


def star(cx, cy, r, inner=0.3):
    k = r * inner
    return [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k),
            (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)]


def logo(size: int) -> Image.Image:
    """The mark at `size` px, drawn at 4x and downsampled for clean edges."""
    S = size * 4
    u = S / 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([8 * u, 8 * u, 248 * u, 248 * u], radius=56 * u, fill=255)
    img.paste(gradient(S, S), (0, 0), mask)
    over = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for y in (52, 88, 124, 160, 196):
        d.rounded_rectangle([34 * u, y * u, 52 * u, (y + 14) * u], radius=4 * u, fill=(255, 255, 255, 72))
    d.polygon([(100 * u, 66 * u), (194 * u, 117 * u), (100 * u, 168 * u)], fill=(255, 255, 255, 255))
    d.polygon(star(196 * u, 185 * u, 27 * u), fill=(255, 255, 255, 255))
    d.polygon(star(160 * u, 209 * u, 13 * u), fill=(255, 255, 255, 205))
    img = Image.alpha_composite(img, over)
    return img.resize((size, size), Image.LANCZOS)


def social_card() -> Image.Image:
    W, H = 1280, 640
    card = Image.new("RGB", (W, H), (13, 14, 22))
    d = ImageDraw.Draw(card)
    # soft gradient glow behind the mark
    glow = gradient(W, H).convert("RGBA")
    glow.putalpha(Image.linear_gradient("L").rotate(90).resize((W, H)).point(lambda v: v * 0.22))
    card.paste(glow, (0, 0), glow)
    mark = logo(300)
    card.paste(mark, (110, (H - 300) // 2), mark)
    title = ImageFont.truetype(str(FONTS / "Poppins-ExtraBold.ttf"), 112)
    sub = ImageFont.truetype(str(FONTS / "Poppins-Bold.ttf"), 34)
    small = ImageFont.truetype(str(FONTS / "Poppins-Bold.ttf"), 24)
    x = 470
    d.text((x, 158), "VideoGen", font=title, fill=(255, 255, 255))
    d.text((x, 300), "Self-hosted AI video studio", font=sub, fill=(225, 220, 255))
    d.text((x, 346), "for YouTube, Shorts and Reels", font=sub, fill=(225, 220, 255))
    tags = ["Script", "Voice", "Clips", "Thumbnail", "Upload"]
    tx = x
    for t in tags:
        w = d.textlength(t, font=small) + 36
        d.rounded_rectangle([tx, 430, tx + w, 474], radius=22, outline=(177, 77, 255), width=2,
                            fill=(30, 24, 52))
        d.text((tx + 18, 437), t, font=small, fill=(255, 255, 255))
        tx += w + 14
    return card


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    logo(512).save(OUT / "logo.png")
    social_card().save(OUT / "social-preview.png")
    print("wrote", OUT / "logo.png", "and", OUT / "social-preview.png")
