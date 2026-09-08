"""Story board overlay: LLM-planned fact cards (kicker / headline / body plus an optional
related image) rendered with Pillow as full-frame RGBA PNGs and composited over the video.

Assets live in the job folder: plan.json["storyboard"] holds the cards, sbimg_NN.jpg the
rendered related images (one short ComfyUI render each) and card_NN.png the final overlays,
which are (re)drawn at assembly time so they always match the video frame size.
"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config
from services import cast, llm, video

CARD_FADE = 0.5      # seconds of alpha fade in/out per card
CARD_INSET = 0.35    # seconds a card stays clear of the clip cut on each side
MAX_LINES = {"headline": 2, "body": 3}

KICKERS = ("KEY FACT", "DID YOU KNOW", "HOW IT WORKS", "TAKEAWAY", "BY THE NUMBERS",
           "PRO TIP", "WHY IT MATTERS", "QUICK RECAP", "WHAT'S NEW", "IN SHORT")


# ---- fonts -----------------------------------------------------------------
def _font_candidates(bold: bool) -> list:
    env = os.environ.get("VIDEOGEN_FONT_BOLD" if bold else "VIDEOGEN_FONT_REGULAR")
    out = [env] if env else []
    local = config.BASE_DIR / "fonts"
    if local.is_dir():
        pat = ("*bold*", "*Bold*", "*BOLD*") if bold else ("*regular*", "*Regular*", "*.ttf")
        for p in pat:
            out += [str(f) for f in sorted(local.glob(p)) if f.suffix.lower() in (".ttf", ".otf")]
    if sys.platform.startswith("win"):
        fd = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        out += [str(fd / f) for f in (("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf") if bold
                                      else ("segoeui.ttf", "arial.ttf", "calibri.ttf"))]
    out += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"]
    try:  # matplotlib ships DejaVu - a decent last resort on any platform
        import matplotlib
        mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        out.append(str(mpl / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")))
    except Exception:  # noqa: BLE001
        pass
    return out


def load_font(size: int, bold: bool = False):
    size = max(8, int(size))
    for cand in _font_candidates(bold):
        try:
            if cand and Path(cand).exists():
                return ImageFont.truetype(cand, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1: scalable built-in font
    except TypeError:
        return ImageFont.load_default()


# ---- text helpers ----------------------------------------------------------
def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int) -> list:
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1].rstrip()
        lines[-1] = last + "…"
    return lines


def _spaced(draw, xy, text, font, fill, spacing):
    """Draw text with extra letter spacing (Pillow has no tracking option)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing
    return x


def image_spec(width: int, height: int) -> tuple:
    """Shape of the related picture for a frame of this size:
    (ComfyUI aspect name, width/height ratio, crop_w, crop_h).
    Portrait videos get a square picture, square videos 4:3, landscape 16:9."""
    if height > width * 1.05:
        return "1:1 (Square)", 1.0, 640, 640
    if abs(width - height) <= width * 0.05:
        return "4:3 (Standard)", 4 / 3, 768, 576
    return "16:9 (Widescreen)", 16 / 9, 768, 432


def _hex(color: str, alpha: int = 255) -> tuple:
    c = (color or "#e9a13b").lstrip("#")
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha
    except ValueError:
        return 233, 161, 59, alpha


def _shadowed(canvas: Image.Image, layer: Image.Image, pos: tuple, blur: int, offset: int,
              opacity: int = 150):
    """Paste `layer` (RGBA) on canvas at pos with a soft drop shadow underneath."""
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = layer.split()[3].point(lambda a: a * opacity // 255)
    black = Image.new("RGBA", layer.size, (0, 0, 0, 255))
    black.putalpha(alpha)
    sh.paste(black, (pos[0], pos[1] + offset), black)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh)
    canvas.alpha_composite(layer, pos)


def _rounded_image(img: Image.Image, size: tuple, radius: int, border: int) -> Image.Image:
    """Cover-crop img to size, round the corners, add a thin light border."""
    w, h = size
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                     Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    img = img.crop((left, top, left + w, top + h)).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    img.putalpha(mask)
    if border > 0:
        ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius,
                                              outline=(255, 255, 255, 190), width=border)
    return img


# ---- card rendering --------------------------------------------------------
def render_card(card: dict, size: tuple, out_png: Path, image: Path | None = None,
                accent: str | None = None, index: int = 1, total: int = 1) -> Path:
    """Draw one full-frame transparent card: a lower-third text panel (kicker, headline,
    body) and, if given, a framed related image. Layout adapts to the frame orientation."""
    W, H = size
    accent = accent or config.STORYBOARD_ACCENT
    portrait, square = H > W * 1.05, abs(W - H) <= W * 0.05
    # scale unit: 1.0 at 864x480 / 480x864; square frames scale by width (their panel is
    # only ~56% wide, so the short-side rule would make the text far too large)
    s = W / 864.0 if square else min(W, H) / 480.0
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    f_kick = load_font(13 * s, bold=True)
    f_head = load_font((28 if portrait else 30) * s, bold=True)
    f_body = load_font((18 if portrait else 19) * s, bold=False)
    f_num = load_font(12 * s, bold=True)

    m = int((0.06 if portrait else 0.045) * W)
    gap = int(14 * s)
    pad = int(18 * s)
    bar_w = max(3, int(5 * s))
    radius = int(14 * s)
    bottom_safe = int(0.14 * H) if portrait else m   # keep clear of the Shorts UI

    # geometry of the two blocks; the picture's shape follows the frame orientation
    _, img_ratio, _, _ = image_spec(W, H)
    if portrait:
        panel_w = W - 2 * m
        img_w = int(0.8 * W)
    elif square:
        panel_w = int(0.56 * W)
        img_w = int(0.34 * W)
    else:
        panel_w = int(0.58 * W)
        img_w = int(0.30 * W)
    img_h = int(img_w / img_ratio)
    has_img = image is not None and Path(image).exists()

    # text layout inside the panel
    text_w = panel_w - 2 * pad - bar_w - int(12 * s)
    kicker = (card.get("kicker") or "").strip().upper()[:28]
    head_lines = _wrap(draw, card.get("headline", ""), f_head, text_w, MAX_LINES["headline"])
    body_lines = _wrap(draw, card.get("body", ""), f_body, text_w, MAX_LINES["body"])
    lh_head = int(f_head.size * 1.15)
    lh_body = int(f_body.size * 1.35)
    y_needed = pad
    if kicker:
        y_needed += int(f_kick.size * 1.3) + int(8 * s)
    y_needed += lh_head * len(head_lines)
    if body_lines:
        y_needed += int(8 * s) + lh_body * len(body_lines)
    panel_h = y_needed + pad

    panel_x = m
    panel_y = H - bottom_safe - panel_h
    if portrait:
        room = panel_y - gap - m          # vertical space between the top margin and the panel
        if img_h > room > 0:              # shrink the picture rather than overflow the frame
            img_h = int(room)
            img_w = int(img_h * img_ratio)
        img_x, img_y = (W - img_w) // 2, panel_y - gap - img_h
    else:
        img_x, img_y = W - m - img_w, H - bottom_safe - img_h

    # panel
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle((0, 0, panel_w - 1, panel_h - 1), radius=radius, fill=(12, 14, 20, 196),
                         outline=(255, 255, 255, 38), width=1)
    pd.rounded_rectangle((pad, pad, pad + bar_w, panel_h - pad), radius=bar_w // 2, fill=_hex(accent))
    tx = pad + bar_w + int(12 * s)
    ty = pad
    if kicker:
        _spaced(pd, (tx, ty), kicker, f_kick, _hex(accent), spacing=1.6 * s)
        num = f"{index:02d} / {total:02d}"
        nw = pd.textlength(num, font=f_num)
        pd.text((panel_w - pad - nw, ty + 1), num, font=f_num, fill=(150, 156, 168, 255))
        ty += int(f_kick.size * 1.3) + int(8 * s)
    for line in head_lines:
        pd.text((tx, ty), line, font=f_head, fill=(255, 255, 255, 255))
        ty += lh_head
    if body_lines:
        ty += int(8 * s)
        for line in body_lines:
            pd.text((tx, ty), line, font=f_body, fill=(214, 219, 228, 255))
            ty += lh_body
    _shadowed(canvas, panel, (panel_x, panel_y), blur=int(10 * s), offset=int(6 * s))

    if has_img:
        try:
            img = _rounded_image(Image.open(image), (img_w, img_h), radius=int(12 * s),
                                 border=max(1, int(2 * s)))
            _shadowed(canvas, img, (img_x, img_y), blur=int(12 * s), offset=int(8 * s), opacity=180)
        except Exception as e:  # noqa: BLE001 - a broken image must not lose the text card
            print(f"storyboard image skipped: {e}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, "PNG")
    return out_png


# ---- planning / assets -----------------------------------------------------
def normalize_cards(cards, n: int, max_images: int | None = None) -> list:
    """Validate LLM cards: clip ranges inside 1..n, ordered, non-overlapping; text trimmed;
    images capped to max_images (the first ones win)."""
    max_images = config.STORYBOARD_MAX_IMAGES if max_images is None else max_images
    out, last_end, images = [], 0, 0
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        try:
            a = int(c.get("start_clip") or c.get("clip") or 0)
            b = int(c.get("end_clip") or a)
        except (TypeError, ValueError):
            continue
        a, b = max(1, a), min(n, max(a, b))
        if a > n or a <= last_end:
            continue
        head = str(c.get("headline") or "").strip()
        body = str(c.get("body") or "").strip()
        if not head and not body:
            continue
        if not head:
            head, body = body, ""
        kicker = str(c.get("kicker") or "").strip().upper()
        prompt = str(c.get("image_prompt") or "").strip()
        if prompt and images < max_images:
            images += 1
        else:
            prompt = ""
        out.append({"start_clip": a, "end_clip": b, "kicker": kicker[:28],
                    "headline": head[:90], "body": body[:200], "image_prompt": prompt})
        last_end = b
    return out


def image_path(job_dir: Path, i: int) -> Path:
    return Path(job_dir) / f"sbimg_{i:02d}.jpg"


def card_path(job_dir: Path, i: int) -> Path:
    return Path(job_dir) / f"card_{i:02d}.png"


def assets_ready(job_dir: Path, plan: dict) -> bool:
    """True when the cards are planned and every requested image is rendered."""
    cards = plan.get("storyboard")
    if not isinstance(cards, list):
        return False
    return all(image_path(job_dir, i).exists() for i, c in enumerate(cards, 1) if c.get("image_prompt"))


def clear_assets(job_dir: Path):
    for p in list(Path(job_dir).glob("sbimg_*.jpg")) + list(Path(job_dir).glob("card_*.png")):
        p.unlink(missing_ok=True)


def ensure_assets(job, plan: dict, n: int, on_progress=None, add_log=None, should_stop=None):
    """Plan the cards with the LLM (once) and render their related images through ComfyUI
    (skipping the ones already on disk). Image failures are logged and skipped; the card then
    shows text only. Returns the (possibly updated) card list."""
    from services import comfy
    d = job.dir
    cards = plan.get("storyboard")
    if not isinstance(cards, list):
        if on_progress:
            on_progress("planning the story board cards with vLLM")
        raw = llm.write_storyboard(plan.get("title", ""), plan.get("narration", ""),
                                   plan.get("segments") or [], n, channel=job.channel,
                                   max_images=config.STORYBOARD_MAX_IMAGES)
        cards = normalize_cards(raw, n)
        plan["storyboard"] = cards
        if add_log:
            imgs = sum(1 for c in cards if c["image_prompt"])
            add_log(f"story board: {len(cards)} cards planned, {imgs} with a related image")
    todo = [(i, c) for i, c in enumerate(cards, 1)
            if c.get("image_prompt") and not image_path(d, i).exists()]
    if not todo:
        return cards
    client = comfy.ComfyClient()
    template = client.load_workflow(config.T2V_WORKFLOW)
    # the clips may not exist yet: derive the frame orientation from the job's aspect ratio
    fw, fh = config.THUMB_SIZES.get(job.aspect_ratio, (1280, 720))
    img_aspect, _, crop_w, crop_h = image_spec(fw, fh)
    for i, c in todo:
        if should_stop and should_stop():
            break
        if on_progress:
            on_progress(f"rendering story board image {i}/{len(cards)}")
        raw = d / f"sbimg_{i:02d}_raw.mp4"
        try:
            prompt = (f"{plan.get('style', '')}\n\n{cast.with_prefix(job.channel, c['image_prompt'])}\n\n"
                      "A single clean still image. No text, captions, letters, logos or watermarks.")
            wf = client.build_thumbnail(template, prompt, f"videogen/{job.id}/sbimg_{i:02d}",
                                        aspect_ratio=img_aspect)
            client.run(wf, raw, on_progress=lambda m, i=i: on_progress and on_progress(
                f"story board image {i}: {m}"), should_stop=should_stop,
                       cleanup=config.COMFY_CLEANUP)
            video.make_thumbnail(raw, image_path(d, i), crop_w, crop_h)
        except Exception as e:  # noqa: BLE001 - text-only card instead
            if add_log:
                add_log(f"story board image {i} failed ({e}); card {i} will be text only")
            c["image_prompt"] = ""
        finally:
            raw.unlink(missing_ok=True)
    return cards


def render_all(job_dir: Path, plan: dict, clips: list, size: tuple,
               accent: str | None = None) -> list:
    """Draw every card PNG for the given frame size and return the overlay schedule
    [(png, start_s, end_s), ...] aligned to the actual clip cut points."""
    cards = plan.get("storyboard") or []
    bounds = video.clip_boundaries(clips)
    n = len(bounds)
    out = []
    for i, c in enumerate(cards, 1):
        a, b = int(c.get("start_clip", 1)), int(c.get("end_clip", 1))
        if not (1 <= a <= b <= n):
            continue
        start = (bounds[a - 2] if a > 1 else 0.0) + CARD_INSET
        end = bounds[b - 1] - CARD_INSET
        if end - start < 1.0:
            continue
        img = image_path(job_dir, i)
        png = render_card(c, size, card_path(job_dir, i), image=img if img.exists() else None,
                         accent=accent, index=i, total=len(cards))
        out.append((png, round(start, 3), round(end, 3)))
    return out
