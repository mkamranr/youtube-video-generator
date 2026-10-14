"""ffmpeg helpers: last-frame extraction, concatenation, upscaling, story-board overlays and audio mixing."""
import math
import subprocess
from pathlib import Path

import config
from services.ffbin import ffmpeg, ffprobe


def _run(cmd, cwd=None):
    """Run an ffmpeg command. `cwd` exists for the subtitle filter: inside a filtergraph argument
    '\\' escapes and ':' separates options, so a Windows path like C:\\jobs\\captions.ass cannot be
    written literally. Running from the job folder lets the filter name the bare file instead,
    which needs no escaping and behaves the same on Windows and Linux."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=(str(cwd) if cwd else None))
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def _probe(path: Path, *entries: str) -> str:
    return subprocess.run([ffprobe(), "-v", "error", *entries, str(path)],
                          capture_output=True, text=True).stdout.strip()


def duration(path: Path) -> float:
    """Container duration in seconds (0.0 if it cannot be read)."""
    try:
        return float(_probe(path, "-show_entries", "format=duration", "-of", "csv=p=0"))
    except ValueError:
        return 0.0


def has_audio(path: Path) -> bool:
    return _probe(path, "-select_streams", "a", "-show_entries", "stream=codec_type",
                  "-of", "csv=p=0") != ""


def frame_size(path: Path) -> tuple:
    """(width, height) of the first video stream."""
    out = _probe(path, "-select_streams", "v:0", "-show_entries", "stream=width,height",
                 "-of", "csv=p=0")
    w, h = out.split(",")[:2]
    return int(w), int(h)


def target_size(width: int, height: int, short_side: int) -> tuple | None:
    """(w, h) that upscales a width x height frame so its short side equals `short_side`,
    keeping the aspect and rounding to even numbers (yuv420p). None when the frame's short
    side already is at least `short_side` - the pipeline never downscales."""
    width, height, short_side = int(width), int(height), int(short_side)
    if short_side <= 0 or min(width, height) >= short_side:
        return None
    even = lambda v: max(2, int(round(v / 2)) * 2)  # noqa: E731
    if width >= height:
        return even(width * short_side / height), even(short_side)
    return even(short_side), even(height * short_side / width)


def upscale(video: Path, out_mp4: Path, width: int, height: int) -> Path:
    """Re-encode the video scaled (lanczos) to exactly width x height; the audio is copied as is."""
    out_mp4 = Path(out_mp4)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video),
           "-vf", f"scale={int(width)}:{int(height)}:flags=lanczos",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if has_audio(video):
        cmd += ["-c:a", "copy"]
    cmd += [str(out_mp4)]
    _run(cmd)
    return out_mp4


def clip_boundaries(clips: list) -> list:
    """Cumulative end times of the given clips, i.e. the cut points of their concatenation
    (the last entry is the total length)."""
    out, t = [], 0.0
    for c in clips:
        t += duration(Path(c))
        out.append(round(t, 3))
    return out


def last_frame(video: Path, out_jpg: Path) -> Path:
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    # -sseof seeks relative to the end; -update 1 keeps only the last decoded frame.
    _run([ffmpeg(), "-y", "-v", "error", "-sseof", "-0.3", "-i", str(video),
          "-update", "1", "-q:v", "2", str(out_jpg)])
    if not out_jpg.exists():
        raise RuntimeError("could not extract last frame")
    return out_jpg


def fit_image(src: Path, out_jpg: Path, width: int, height: int) -> Path:
    """Cover-scale and centre-crop a still image to width x height, saved as JPG
    (used to fit a cast portrait to the job's aspect before seeding clip 1)."""
    from PIL import Image
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    scale = max(width / img.width, height / img.height)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                     Image.LANCZOS)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    img.crop((left, top, left + width, top + height)).save(out_jpg, "JPEG", quality=94)
    return out_jpg


def make_thumbnail(video: Path, out_jpg: Path, width: int = 1280, height: int = 720,
                   logo: Path | None = None, last: bool = False) -> Path:
    """First (or, with last=True, last) frame of the video, cover-cropped to width x height;
    optionally overlays the channel logo in the top-right corner. Saved as JPG."""
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    if last:   # -sseof seeks relative to the end; -update 1 keeps only the last decoded frame
        _run([ffmpeg(), "-y", "-v", "error", "-sseof", "-0.3", "-i", str(video),
              "-vf", vf, "-update", "1", "-q:v", "2", str(out_jpg)])
    else:
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(video),
              "-vf", vf, "-frames:v", "1", "-q:v", "2", str(out_jpg)])
    if not out_jpg.exists():
        raise RuntimeError("could not extract thumbnail frame")
    if logo is not None and Path(logo).exists():
        try:
            from PIL import Image
            base = Image.open(out_jpg).convert("RGB")
            lg = Image.open(logo).convert("RGBA")
            lw = max(1, int(base.width * 0.16))
            lh = max(1, int(lg.height * lw / lg.width))
            lg = lg.resize((lw, lh), Image.LANCZOS)
            margin = max(8, int(base.width * 0.025))
            base.paste(lg, (base.width - lw - margin, margin), lg)
            base.save(out_jpg, "JPEG", quality=92)
        except Exception as e:  # noqa: BLE001 - logo overlay must never break the pipeline
            print(f"logo overlay failed: {e}")
    return out_jpg


def concat_clips(clips: list, out_mp4: Path) -> Path:
    out_mp4 = Path(out_mp4)
    listfile = out_mp4.with_suffix(".txt")
    listfile.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    try:
        _run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
              "-c", "copy", str(out_mp4)])
    except RuntimeError:
        _run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
              "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out_mp4)])
    return out_mp4


def normalize_for_concat(src: Path, out_mp4: Path, width: int, height: int,
                         rate: float = config.VIDEO_FPS) -> Path:
    """Re-encode one finished video to exactly width x height at `rate`, ready to be stream-copied
    into a concatenation with others.

    concat_clips uses the concat DEMUXER, which needs every input to agree on codec, resolution,
    pixel format and frame rate - true for the clips of one job, which came out of one workflow,
    and routinely false across jobs (different aspect ratios, different output resolutions, one
    upscaled and one native). Its re-encode fallback does not save that case: the demuxer refuses
    a mid-stream resolution change whatever the output codec is. So each part is brought to one
    shape first and the concatenation itself stays a copy.

    The picture is FITTED and padded rather than cropped - a part is a finished video someone may
    already have published, so losing its edges is worse than a black border. The audio is
    re-encoded too (a part may be mono or 48 kHz), and a part with no audio at all gets silence,
    because one input without a stream the others have breaks the copy just as surely.

    Deliberately no loudnorm: every final.mp4 left mix_audio at about -14 LUFS already, and a
    second pass over the joined video would pump its gain across the seams."""
    src, out_mp4 = Path(src), Path(out_mp4)
    w, h, rate = _even(width), _even(height), float(rate or config.VIDEO_FPS)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(src)]
    if not has_audio(src):
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest"]
    cmd += ["-vf", (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={rate},setsar=1"),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-video_track_timescale", "90000", str(out_mp4)]
    _run(cmd)
    return out_mp4


def fps(path: Path) -> float:
    """Frame rate of the first video stream (25.0 when it cannot be read)."""
    out = _probe(path, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                 "-of", "csv=p=0")
    try:
        num, den = (out.split("/") + ["1"])[:2]
        return float(num) / float(den or 1) or 25.0
    except (ValueError, ZeroDivisionError):
        return 25.0


def _even(v) -> int:
    return max(2, int(round(float(v) / 2)) * 2)


def _pos(v) -> str:
    """An overlay x/y: a plain integer, or an ffmpeg per-frame expression given as a string.

    Expressions are single-quoted so their commas do not split the filtergraph - the same trick
    scroll_layer uses for crop's y, and safe for the same reason (_run passes a list and never
    goes through a shell). An expression must not contain ':' or '[' / ']'."""
    if isinstance(v, str):
        return v if v.startswith("'") else f"'{v}'"
    return str(int(v))


def subtitles_ok(ass_path) -> bool:
    """Can this ffmpeg build burn this subtitle file? A ~50 ms render of a 64x64 colour source.

    Cheap insurance: libass is a build option, and finding out it is missing AFTER a 190-second 4K
    encode would throw the whole encode away. This turns that into a logged skip."""
    p = Path(ass_path)
    try:
        _run([ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
              "-vf", f"subtitles={p.name}", "-frames:v", "1", "-f", "null", "-"], cwd=p.parent)
        return True
    except RuntimeError:
        return False


def composite(video: Path, layers: list, out_mp4: Path, fade: float = 0.5, subtitles=None) -> Path:
    """Composite timed RGBA image layers over the video in ONE encode.

    layers: [{png, start, end, x, y, pre, fade}] - `pre` is extra filter steps applied to the image
    stream before the alpha fades (the scrolling panel uses it for its scale+crop); x/y default
    to 0, which is the full-frame story-board card case, and may each be an ffmpeg per-frame
    expression (a string) instead of a number - that is how a story board card eases into place
    without a second pass. The audio is copied as is.

    A layer may instead be {video, w, h, matte, ...}: a MOVING layer, which is how the talking-head
    presenter is composited. It is resampled to the video's frame rate (the avatar engines render
    at 16 or 24 fps regardless of the job), scaled to w x h, slid to its window with setpts, and -
    when a `matte` grayscale video is given - cut out per frame with alphamerge. Everything after
    that is identical to a PNG layer, so the bubble and cut-out looks share one code path and both
    still fit in the single encode.

    subtitles: an .ass file burned in as the LAST filter, so captions always sit on top. One filter
    and no extra input, which is why captions are a subtitle track rather than a PNG per phrase: a
    3-minute video is 60-120 phrases, and a full-frame RGBA PNG at 4K is ~33 MB decoded.

    One pass matters: at 4K a libx264 re-encode of a 3-minute video costs about 190 s, so a
    second overlay stage would nearly double the assembly time and leave another full-size
    intermediate on disk."""
    out_mp4 = Path(out_mp4)
    layers = [dict(lyr) for lyr in layers if float(lyr["end"]) > float(lyr["start"])]
    if not layers and subtitles is None:
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out_mp4)])
        return out_mp4
    total = duration(video) or (max(float(lyr["end"]) for lyr in layers) if layers else 0.0)
    if total <= 0:
        raise RuntimeError("could not read the video duration")
    rate = fps(video)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    parts, prev, nin = [], "[0:v]", 1
    for i, lyr in enumerate(layers, 1):
        start, end = float(lyr["start"]), min(float(lyr["end"]), total)
        f = max(0.05, min(float(lyr.get("fade", fade)), (end - start) / 2))
        pre = lyr.get("pre") or ""
        x, y = _pos(lyr.get("x", 0)), _pos(lyr.get("y", 0))
        fades = (f"fade=t=in:st={start:.3f}:d={f:.3f}:alpha=1,"
                 f"fade=t=out:st={end - f:.3f}:d={f:.3f}:alpha=1")
        if lyr.get("video"):
            # A moving layer (the talking-head presenter). Its own frame rate is the engine's,
            # not the video's - Wan renders at 16 fps - so fps= resamples it before anything
            # else. setpts slides it to its window; overlay's default eof_action=repeat holds
            # the last frame if the clip is marginally shorter than the window, and the alpha
            # fade has already taken it to zero by then.
            w, h = _even(lyr.get("w") or 0), _even(lyr.get("h") or 0)
            scale = f",scale={w}:{h}" if w and h else ""
            cmd += ["-i", str(Path(lyr["video"]))]
            vi = nin
            nin += 1
            if lyr.get("matte"):
                # Per-frame alpha from a grayscale matte video: this is what makes the cut-out
                # look possible. The circle bubble uses the same alphamerge with a STATIC png
                # mask, which has to be looped like any other still or alphamerge starves after
                # one frame and the layer disappears.
                matte = Path(lyr["matte"])
                still = matte.suffix.lower() == ".png"
                if still:
                    cmd += ["-loop", "1", "-framerate", f"{rate:.4f}", "-t", f"{total:.3f}"]
                cmd += ["-i", str(matte)]
                mi = nin
                nin += 1
                mshift = "" if still else f",setpts=PTS-STARTPTS+{start:.3f}/TB"
                parts.append(f"[{vi}:v]fps={rate:.4f}{scale},format=rgba{pre},"
                             f"setpts=PTS-STARTPTS+{start:.3f}/TB[avc{i}]")
                shadow = float(lyr.get("shadow") or 0)
                if shadow > 0 and not still:
                    # A contact shadow is what stops a cut-out looking pasted on. It is the same
                    # matte, blurred, used as the alpha of a black rectangle - so it costs one
                    # split rather than another input or another encode.
                    dx, dy = int(lyr.get("shadow_dx", 12)), int(lyr.get("shadow_dy", 10))
                    op = float(lyr.get("shadow_alpha", 0.40))
                    parts.append(f"[{mi}:v]fps={rate:.4f}{scale},format=gray{mshift},"
                                 f"split=2[avm{i}][avs{i}]")
                    parts.append(f"[avs{i}]gblur=sigma={shadow:.1f}[avsb{i}]")
                    parts.append(f"color=c=black:s={w}x{h}:r={rate:.4f}:d={total:.3f}[avk{i}]")
                    parts.append(f"[avk{i}][avsb{i}]alphamerge,colorchannelmixer=aa={op:.2f},"
                                 f"{fades}[sh{i}]")
                    # the presenter's x/y are plain numbers (unlike a card's easing expression),
                    # so the offset is arithmetic; _pos still guards the expression case
                    ox, oy = lyr.get("x", 0), lyr.get("y", 0)
                    sx = _pos(ox + dx) if not isinstance(ox, str) else _pos(ox)
                    sy = _pos(oy + dy) if not isinstance(oy, str) else _pos(oy)
                    parts.append(f"{prev}[sh{i}]overlay={sx}:{sy}:eval=frame:"
                                 f"enable='between(t,{start:.3f},{end:.3f})'[vs{i}]")
                    prev = f"[vs{i}]"
                else:
                    parts.append(f"[{mi}:v]fps={rate:.4f}{scale},format=gray{mshift}[avm{i}]")
                parts.append(f"[avc{i}][avm{i}]alphamerge,{fades}[c{i}]")
            else:
                parts.append(f"[{vi}:v]fps={rate:.4f}{scale},format=rgba{pre},"
                             f"setpts=PTS-STARTPTS+{start:.3f}/TB,{fades}[c{i}]")
        else:
            # -framerate: a looped image decodes at 25 fps by default, which makes a moving crop
            # step in 25 Hz jumps no matter what the video itself runs at
            cmd += ["-loop", "1", "-framerate", f"{rate:.4f}", "-t", f"{total:.3f}",
                    "-i", str(Path(lyr["png"]))]
            vi = nin
            nin += 1
            parts.append(f"[{vi}:v]format=rgba{pre},{fades}[c{i}]")
        parts.append(f"{prev}[c{i}]overlay={x}:{y}:eval=frame:"
                     f"enable='between(t,{start:.3f},{end:.3f})'[v{i}]")
        prev = f"[v{i}]"
    if subtitles is not None:
        parts.append(f"{prev}subtitles={Path(subtitles).name}[vsub]")
        prev = "[vsub]"
    cmd += ["-filter_complex", ";".join(parts), "-map", prev,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if has_audio(video):
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-t", f"{total:.3f}", str(out_mp4)]
    # every other path in the command is absolute, so running from the subtitle file's folder
    # changes nothing except letting the filter name it without escaping (see _run)
    _run(cmd, cwd=Path(subtitles).parent if subtitles is not None else None)
    return out_mp4


def progress_layers(size: tuple, total: float, out_dir, color: str = "#e9a13b",
                    height_pct: float = 0.6, alpha: int = 235, track_alpha: int = 55) -> list:
    """Two composite layers: a dim full-width track and a bar that fills left to right.

    The bar is ONE static full-width PNG moved by the overlay x expression - its left edge
    crossing the frame IS the progress. The obvious alternative, a crop that widens per frame,
    does not work: that changes the overlay input's frame size mid-stream, which overlay rejects."""
    from PIL import Image

    W, H = int(size[0]), int(size[1])
    total = float(total or 0.0)
    h = max(2, _even(H * height_pct / 100.0))
    if total <= 0 or h <= 0:
        return []
    c = (color or "#e9a13b").lstrip("#")
    try:
        rgb = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    except ValueError:
        rgb = (233, 161, 59)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track = out_dir / "progress_track.png"
    bar = out_dir / "progress_bar.png"
    Image.new("RGBA", (W, h), rgb + (track_alpha,)).save(track)
    Image.new("RGBA", (W, h), rgb + (alpha,)).save(bar)
    return [{"png": track, "start": 0.0, "end": total, "x": 0, "y": H - h, "fade": 0.3},
            {"png": bar, "start": 0.0, "end": total, "y": H - h, "fade": 0.3,
             "x": f"-{W}+{W}*clip(t/{total:.3f},0,1)"}]


def avatar_box(size: tuple, style: str = "", size_pct=None, cutout_pct=None,
               margin_pct=None, corner: str = "", clear_bottom: float = 0.0) -> tuple:
    """Where the presenter sits: (x, y, w, h).

    Every dimension comes from a FRACTION of the frame, never a pixel constant, so one setting
    covers 1080p through 4K and 16:9 / 9:16 / 1:1 alike. The bubble measures against the SHORT
    side (a circle should not grow just because the video got wider), the cut-out against the
    HEIGHT (it is a standing figure). Both are clamped so a portrait frame cannot end up with a
    presenter wider than the video.

    clear_bottom: fraction of the height to keep free at the bottom - the burned-in caption band,
    so the presenter never sits on the subtitles."""
    W, H = int(size[0]), int(size[1])
    short = min(W, H)
    style = style or config.DEFAULT_AVATAR_STYLE
    margin = int(round((config.AVATAR_MARGIN_PCT if margin_pct is None else margin_pct) * short))
    reserve = int(round(max(0.0, float(clear_bottom)) * H))
    if style == "bubble":
        d = _even((config.AVATAR_SIZE_PCT if size_pct is None else size_pct) * short)
        bw = bh = max(2, min(d, W - 2 * margin, H - 2 * margin - reserve))
    else:
        bh = _even((config.AVATAR_CUTOUT_PCT if cutout_pct is None else cutout_pct) * H)
        bh = max(2, min(bh, H))
        bw = bh                      # every avatar engine renders a square frame
        # A standing figure is sized against the height, but on a tall frame (9:16 Shorts) that
        # alone would make it as wide as the video, so the width cap is what actually binds there.
        cap = _even(config.AVATAR_MAX_WIDTH_PCT * W)
        if bw > cap:
            bw = bh = max(2, cap)
    corner = corner or config.AVATAR_CORNER
    x = (W - bw - margin) if "right" in corner else margin
    if "bottom" in corner:
        # A bubble is inset from the edge; a cut-out is anchored TO it, so the torso runs off
        # frame - a floating head with a cut neck is the thing that looks wrong. The caption
        # reserve therefore applies only to the bubble: lifting a cut-out off the bottom is
        # exactly the mistake it is meant to prevent, so the captions move instead (the
        # pipeline feeds this rect to captions.lift_spans, like a story-board card).
        y = (H - bh - margin - reserve) if style == "bubble" else (H - bh)
    else:
        y = margin if style == "bubble" else 0
    return int(x), int(max(0, y)), int(bw), int(bh)


AVATAR_FILL_MAX = 2.5     # how far measuring the subject may enlarge the frame, so one bad
                          # matte cannot blow the presenter up to fill the video


def _fill_box(x: int, y: int, bw: int, bh: int, bbox, bottom: bool = True) -> tuple:
    """Draw rect that puts the FIGURE in the box, rather than the engine's whole square frame.

    `bbox` is (x0, y0, x1, y1) in 0..1 - where avatar.matte_bounds found the subject. The
    engines render a head-and-shoulders portrait centred in a square, so without this a
    quarter of the box is transparent and the presenter reads as smaller than the settings
    ask for. The frame is enlarged until the subject spans the box and then shifted so it
    lands on it; the transparent remainder hangs off the frame, which overlay clips for
    free. The vertical anchor follows the corner for the reason avatar_box anchors a cut-out
    to the edge - a figure floating above the bottom with a cut torso is what looks wrong.

    No bbox is the identity, so the bubble and every plan written before the subject was
    measured draw exactly as they did."""
    if not bbox or len(bbox) != 4:
        return x, y, bw, bh
    x0, y0, x1, y1 = (float(v) for v in bbox)
    # the smaller of the two enlargements, so the subject fits the box on BOTH axes
    f = min(1.0 / max(1e-6, x1 - x0), 1.0 / max(1e-6, y1 - y0), AVATAR_FILL_MAX)
    if f <= 1.0:
        return x, y, bw, bh
    dw, dh = _even(bw * f), _even(bh * f)
    nx = x + bw / 2.0 - (x0 + x1) / 2.0 * dw                  # centred across the box
    ny = (y + bh - y1 * dh) if bottom else (y - y0 * dh)      # ...anchored to its own edge
    return int(round(nx)), int(round(ny)), dw, dh


def avatar_layers(size: tuple, segments: list, out_dir, style: str = "", corner: str = "",
                  size_pct=None, cutout_pct=None, margin_pct=None, accent: str = "",
                  clear_bottom: float = 0.0, fade: float = 0.4) -> list:
    """composite() layers for the presenter.

    segments: [{start, end, video, matte}] - `matte` is the per-frame cut-out alpha (None for the
    bubble, which gets a drawn circle instead). Returns layers in draw order."""
    from PIL import Image, ImageDraw, ImageFilter

    W, H = int(size[0]), int(size[1])
    style = style or config.DEFAULT_AVATAR_STYLE
    x, y, bw, bh = avatar_box(size, style, size_pct, cutout_pct, margin_pct, corner, clear_bottom)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = [s for s in segments if s.get("video") and float(s["end"]) > float(s["start"])]
    if not segs or bw < 2 or bh < 2:
        return []
    out = []
    if style == "bubble":
        ss = 4      # supersample, then downscale: the circle edge has to be antialiased or it
        mask = Image.new("L", (bw * ss, bh * ss), 0)   # crawls against moving video
        ImageDraw.Draw(mask).ellipse((0, 0, bw * ss - 1, bh * ss - 1), fill=255)
        mask_png = out_dir / "avatar_mask.png"
        mask.resize((bw, bh), Image.LANCZOS).save(mask_png)
        for s in segs:
            out.append({"video": s["video"], "matte": mask_png, "w": bw, "h": bh,
                        "x": x, "y": y, "start": float(s["start"]), "end": float(s["end"]),
                        "fade": fade})
        ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse((x + 6, y + 10, x + bw + 6, y + bh + 10), fill=(0, 0, 0, 130))
        ring.alpha_composite(sh.filter(ImageFilter.GaussianBlur(max(4, bw // 17))))
        rr = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        ImageDraw.Draw(rr).ellipse((x * 2, y * 2, (x + bw) * 2, (y + bh) * 2),
                                   outline=(accent or config.STORYBOARD_ACCENT),
                                   width=max(4, bw // 20))
        ring.alpha_composite(rr.resize((W, H), Image.LANCZOS))
        ring_png = out_dir / "avatar_ring.png"
        ring.save(ring_png)
        # the ring is drawn AFTER the bubble so it sits on the cut edge rather than under it
        for s in segs:
            out.append({"png": ring_png, "x": 0, "y": 0,
                        "start": float(s["start"]), "end": float(s["end"]), "fade": fade})
    else:
        bottom = "bottom" in (corner or config.AVATAR_CORNER)
        for s in segs:
            dx, dy, dw, dh = _fill_box(x, y, bw, bh, s.get("bbox"), bottom)
            out.append({"video": s["video"], "matte": s.get("matte"), "w": dw, "h": dh,
                        "x": dx, "y": dy, "start": float(s["start"]), "end": float(s["end"]),
                        "fade": fade,
                        # scaled with the figure so the shadow reads the same at every resolution
                        "shadow": max(4.0, bw / 45.0),
                        "shadow_dx": max(4, bw // 52), "shadow_dy": max(4, bw // 62),
                        "shadow_alpha": 0.40})
    return out


SCROLL_PANEL = 0.70       # picture-in-picture panel: this much of each frame dimension, centred
SCROLL_SPEED = 55.0       # panel-scale pixels per second; above about 60 the text smears
SCROLL_MIN_TOTAL = 20.0   # shorter videos have no middle worth giving up
SCROLL_MIN_WINDOW = 6.0


def _busy_overlap(start: float, end: float, busy: list) -> float:
    """Seconds of [start, end] that the `busy` ranges already occupy."""
    return sum(max(0.0, min(end, be) - max(start, bs)) for bs, be in busy)


def scroll_window(total: float, busy: list = None, lead: float = 0.22, tail: float = 0.22):
    """(start, end) for the page panel: the middle of the video, leaving the opening and the
    ending clear. `busy` is [(start, end), ...] of story-board cards - the window goes in the
    largest gap between them, because on a tech channel both overlays are on by default.
    None when the video is too short to give up a middle.

    None also when the cards leave no gap of SCROLL_MIN_WINDOW - about 29% of three-card
    schedules on a 40s video, and specifically the evenly spread ones the LLM tends to write.
    `scroll_reserve` is the fallback for that case."""
    total = float(total or 0.0)
    if total < SCROLL_MIN_TOTAL:
        return None
    lo, hi = round(total * lead, 3), round(total * (1.0 - tail), 3)
    if hi - lo < SCROLL_MIN_WINDOW:
        return None
    gaps = [(lo, hi)]
    for bs, be in sorted((float(s), float(e)) for s, e in (busy or [])):
        nxt = []
        for gs, ge in gaps:
            if be <= gs or bs >= ge:
                nxt.append((gs, ge))
                continue
            if bs - gs >= SCROLL_MIN_WINDOW:
                nxt.append((gs, bs))
            if ge - be >= SCROLL_MIN_WINDOW:
                nxt.append((be, ge))
        gaps = nxt
    if not gaps:
        return None
    start, end = max(gaps, key=lambda g: g[1] - g[0])
    return (round(start, 3), round(end, 3))


def scroll_reserve(total: float, soft: list = None, hard: list = None,
                   lead: float = 0.22, tail: float = 0.22, window: float = SCROLL_MIN_WINDOW):
    """(start, end) for the page panel when `scroll_window` found no gap at all: take a `window`
    of the middle anyway and expect the caller to trim the story board cards out of it
    (storyboard.trim_schedule). Without this the overlay was simply dropped, so a video at the
    common 30-45s length with evenly spread cards never showed the page it was asked to show.

    `soft` (the story board cards) is what the panel is placed to OVERLAP LEAST - those cards can
    be moved. `hard` (the hook and end cards) instead CLAMPS the band, because polish layers are
    composited after the panel and therefore on top of it: a panel under one is simply invisible,
    and unlike a fact card it cannot be trimmed out of the way. At the default 3s/4s they never
    reach the band, but both are settings (0-6s and 0-10s), and at END_CARD_SECONDS=10 on a 20s
    video the end card starts at 10.0 - inside the band.

    `window` is SCROLL_MIN_WINDOW rather than anything roomier on purpose. Reserving the whole
    band would drop two cards in three (at 40s the band is 22.4s); even 8s costs 2.3s of card time
    and more than halves one fact card, against 0.3s at 6s, and buys only 110px more travel. The
    band is 0.56*total, so at the SCROLL_MIN_TOTAL floor it is 11.2s wide and this window always
    fits - which is what makes a panel on a >=20s video a guarantee rather than a hope."""
    total = float(total or 0.0)
    if total < SCROLL_MIN_TOTAL:
        return None
    lo, hi = round(total * lead, 3), round(total * (1.0 - tail), 3)
    for hs, he in sorted((float(s), float(e)) for s, e in (hard or [])):
        if he <= lo or hs >= hi:
            continue
        if hs <= lo and he >= hi:
            return None                       # the band is entirely covered
        if hs <= lo:
            lo = he                           # eats the opening of the band
        elif he >= hi:
            hi = hs                           # eats the ending of the band
        elif hs - lo >= hi - he:              # strictly inside: keep the roomier side
            hi = hs
        else:
            lo = he
    if hi - lo < window:
        return None
    soft = [(float(s), float(e)) for s, e in (soft or [])]
    # Overlap as a function of the window's start is piecewise linear, and its breakpoints are
    # only where a card's edge meets the window's start or its end. A piecewise-linear minimum
    # sits at a breakpoint or an interval end, so this candidate set is exact - and a few values
    # rather than a grid sweep, which would also round the answer to its step.
    last = round(hi - window, 3)
    cands = {lo, last}
    for bs, be in soft:
        for c in (bs, be, bs - window, be - window):
            if lo <= c <= last:
                cands.add(round(c, 3))
    mid = (lo + hi) / 2.0
    start = min(sorted(cands),      # sorted() so equal placements resolve the same way every run
                key=lambda c: (round(_busy_overlap(c, c + window, soft), 3),
                               abs(c + window / 2.0 - mid)))
    return (round(start, 3), round(start + window, 3))


def scroll_layer(video: Path, page_png: Path, start: float, end: float, content_y: int = 0,
                 panel: float = SCROLL_PANEL) -> dict:
    """A `composite` layer that slowly scrolls a tall page screenshot inside a centred panel.

    content_y: where the README starts in the screenshot - GitHub stacks a branch picker and the
    whole file tree above it, and scrolling through directory listings is not the point."""
    from PIL import Image
    fw, fh = frame_size(video)
    pw, ph = _even(fw * panel), _even(fh * panel)
    with Image.open(page_png) as im:
        ow, oh = im.width, im.height
    sh = _even(oh * pw / ow)                       # the page scaled to the panel width
    top = max(0, min(int(content_y * sh / oh), max(0, sh - ph)))
    dur = max(0.1, float(end) - float(start))
    travel = int(min(max(0, sh - ph - top), SCROLL_SPEED * dur))
    # crop's x/y are per-frame expressions (ffmpeg -h filter=crop marks them T), so the panel
    # moves without any extra option - this build has no `eval` option on crop at all.
    # `t` on a looped image input is the output timeline, so the expression is written relative
    # to the window. Commas survive inside single quotes because _run passes a list and never
    # goes through a shell.
    ypos = (f"'{top}+{travel}*clip((t-{float(start):.3f})/{dur:.3f},0,1)'" if travel else str(top))
    pre = (f",scale={pw}:{sh}:flags=lanczos"
           f",crop=w={pw}:h={ph}:x=0:y={ypos}")
    return {"png": page_png, "start": float(start), "end": float(end),
            "x": (fw - pw) // 2, "y": (fh - ph) // 2, "pre": pre}


def _dip_expr(boundaries: list, width: float = 0.35) -> str:
    """ffmpeg volume expression that dips to silence at every clip cut (linear over
    `width` seconds on each side) - hides the hard jumps between the clips' own audio."""
    cuts = [b for b in boundaries[:-1] if b > 0]  # the last boundary is the end of the video
    if not cuts:
        return "1"
    expr = f"abs(t-{cuts[0]:.3f})"
    for b in cuts[1:]:
        expr = f"min({expr},abs(t-{b:.3f}))"
    return f"min(1,{expr}/{width})"


def mix_audio(video: Path, out_mp4: Path, narration: Path | None = None,
              music: Path | None = None, clip_volume: float | None = None,
              music_volume: float | None = None, clip_boundaries: list | None = None,
              duck: bool = True, sfx: Path | None = None, sfx_volume: float = 0.0) -> Path:
    """Mix any of: narration (full volume), music, clip ambience - over the video.

    - clip_volume=0 drops the clips' own audio; when kept, it is dipped to silence around
      every clip cut (`clip_boundaries`) so the per-clip sound design does not jump.
    - music shorter than the video is looped with a cross-fade, and always fades out at the end;
      music_volume=1.0 plays it at full level.
    - with a narration, music and ambience are side-chain ducked while the voice speaks (duck=True).
    - sfx is a full-length track of card whooshes, mixed in WITHOUT ducking (see below).
    - the result is trimmed to the video length, loudness-normalised and soft-limited.

    loudnorm matters more than it looks: YouTube normalises everything to about -14 LUFS, so an
    un-normalised mix is played several dB quiet, which reads as amateur before a word is heard."""
    clip_volume = config.CLIP_AUDIO_VOLUME if clip_volume is None else clip_volume
    music_volume = config.MUSIC_VOLUME if music_volume is None else music_volume
    out_mp4 = Path(out_mp4)
    total = duration(video)
    if total <= 0:
        raise RuntimeError("could not read the video duration")
    vid_audio = has_audio(video)
    pad = f"apad,atrim=0:{total:.3f},asetpts=PTS-STARTPTS"

    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    chains, idx = [], 1
    layers = []            # output labels feeding the final mix
    sidechains = []        # labels of narration copies used for ducking

    ducked = []            # (label_in, label_out) of beds to duck under the narration
    if narration is not None:
        cmd += ["-i", str(narration)]
        nar_idx = idx
        idx += 1
    if music is not None:
        mdur = duration(music)
        loops = max(1, math.ceil(total / mdur)) if mdur > 0 else 1
        loops = min(loops, 12)
        first = idx
        for _ in range(loops):
            cmd += ["-i", str(music)]
            idx += 1
        if loops == 1:
            src = f"[{first}:a]"
        else:  # chain cross-fades: music, music, music ... -> one long bed
            xf = min(2.0, mdur / 4)
            prev = f"[{first}:a]"
            for k in range(1, loops):
                chains.append(f"{prev}[{first + k}:a]acrossfade=d={xf:.2f}:c1=tri:c2=tri[mx{k}]")
                prev = f"[mx{k}]"
            src = prev
        fade_out = min(2.5, total / 4)
        chains.append(f"{src}volume={music_volume},{pad},"
                      f"afade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out:.3f}[mu0]")
        ducked.append(("[mu0]", "[mu]"))
    if vid_audio and clip_volume > 0:
        dip = _dip_expr(clip_boundaries or []) if clip_boundaries else "1"
        chains.append(f"[0:a]volume={clip_volume},volume='{dip}':eval=frame,{pad}[bg0]")
        ducked.append(("[bg0]", "[bg]"))
    if sfx is not None and Path(sfx).exists() and sfx_volume > 0:
        cmd += ["-i", str(sfx)]
        chains.append(f"[{idx}:a]volume={sfx_volume},{pad}[sx]")
        idx += 1
        # deliberately NOT in `ducked`: the whoosh exists to mark a card appearing WHILE the
        # narrator is talking, so side-chaining it under the voice would delete exactly the hits
        # it is there for
        layers.append("[sx]")

    if narration is not None:
        n_copies = 1 + (len(ducked) if duck else 0)
        outs = "[vo]" + "".join(f"[sc{i}]" for i in range(len(ducked) if duck else 0))
        chains.append(f"[{nar_idx}:a]{pad}" + (f",asplit={n_copies}{outs}" if n_copies > 1 else "[vo]"))
        layers.append("[vo]")
        sidechains = [f"[sc{i}]" for i in range(len(ducked) if duck else 0)]
    for i, (src, dst) in enumerate(ducked):
        if narration is not None and duck:
            chains.append(f"{src}{sidechains[i]}sidechaincompress=threshold=0.1:ratio=3:attack=100:"
                          f"release=800:makeup=1:level_sc=1{dst}")
        else:
            chains.append(f"{src}anull{dst}")
        layers.append(dst)

    if not layers:  # no audio at all: keep the video silent
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-c:v", "copy", "-an", str(out_mp4)])
        return out_mp4
    norm = f"loudnorm=I={config.LOUDNESS_LUFS}:TP=-1.5:LRA=11," if config.LOUDNESS_LUFS else ""
    if len(layers) == 1:
        chains.append(f"{layers[0]}{norm}alimiter=limit=0.97:level=0[a]")
    else:
        chains.append("".join(layers) + f"amix=inputs={len(layers)}:duration=longest:"
                      f"dropout_transition=0:normalize=0,{norm}alimiter=limit=0.97:level=0[a]")
    cmd += ["-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", str(out_mp4)]
    _run(cmd)
    return out_mp4
