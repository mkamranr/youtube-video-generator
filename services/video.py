"""ffmpeg helpers: last-frame extraction, concatenation, upscaling, story-board overlays and audio mixing."""
import math
import subprocess
from pathlib import Path

import config
from services.ffbin import ffmpeg, ffprobe


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
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


def composite(video: Path, layers: list, out_mp4: Path, fade: float = 0.5) -> Path:
    """Composite timed RGBA image layers over the video in ONE encode.

    layers: [{png, start, end, x, y, pre}] - `pre` is extra filter steps applied to the image
    stream before the alpha fades (the scrolling panel uses it for its scale+crop); x/y default
    to 0, which is the full-frame story-board card case. The audio is copied as is.

    One pass matters: at 4K a libx264 re-encode of a 3-minute video costs about 190 s, so a
    second overlay stage would nearly double the assembly time and leave another full-size
    intermediate on disk."""
    out_mp4 = Path(out_mp4)
    layers = [dict(lyr) for lyr in layers if float(lyr["end"]) > float(lyr["start"])]
    if not layers:
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out_mp4)])
        return out_mp4
    total = duration(video) or max(float(lyr["end"]) for lyr in layers)
    rate = fps(video)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    parts, prev = [], "[0:v]"
    for i, lyr in enumerate(layers, 1):
        start, end = float(lyr["start"]), min(float(lyr["end"]), total)
        f = max(0.05, min(fade, (end - start) / 2))
        pre = lyr.get("pre") or ""
        x, y = int(lyr.get("x", 0)), int(lyr.get("y", 0))
        # -framerate: a looped image decodes at 25 fps by default, which makes a moving crop
        # step in 25 Hz jumps no matter what the video itself runs at
        cmd += ["-loop", "1", "-framerate", f"{rate:.4f}", "-t", f"{total:.3f}",
                "-i", str(Path(lyr["png"]))]
        parts.append(f"[{i}:v]format=rgba{pre},fade=t=in:st={start:.3f}:d={f:.3f}:alpha=1,"
                     f"fade=t=out:st={end - f:.3f}:d={f:.3f}:alpha=1[c{i}]")
        parts.append(f"{prev}[c{i}]overlay={x}:{y}:enable='between(t,{start:.3f},{end:.3f})'[v{i}]")
        prev = f"[v{i}]"
    cmd += ["-filter_complex", ";".join(parts), "-map", prev,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if has_audio(video):
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-t", f"{total:.3f}", str(out_mp4)]
    _run(cmd)
    return out_mp4


def overlay_cards(video: Path, cards: list, out_mp4: Path, fade: float = 0.5) -> Path:
    """Composite timed full-frame RGBA cards over the video.

    cards: [(png_path, start_s, end_s), ...] - each PNG must match the video frame size."""
    return composite(video, [{"png": p, "start": s, "end": e} for p, s, e in cards],
                     out_mp4, fade=fade)


SCROLL_PANEL = 0.70       # picture-in-picture panel: this much of each frame dimension, centred
SCROLL_SPEED = 55.0       # panel-scale pixels per second; above about 60 the text smears
SCROLL_MIN_TOTAL = 20.0   # shorter videos have no middle worth giving up
SCROLL_MIN_WINDOW = 6.0


def scroll_window(total: float, busy: list = None, lead: float = 0.22, tail: float = 0.22):
    """(start, end) for the page panel: the middle of the video, leaving the opening and the
    ending clear. `busy` is [(start, end), ...] of story-board cards - the window goes in the
    largest gap between them, because on a tech channel both overlays are on by default.
    None when the video is too short to give up a middle."""
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
              duck: bool = True) -> Path:
    """Mix any of: narration (full volume), music, clip ambience - over the video.

    - clip_volume=0 drops the clips' own audio; when kept, it is dipped to silence around
      every clip cut (`clip_boundaries`) so the per-clip sound design does not jump.
    - music shorter than the video is looped with a cross-fade, and always fades out at the end;
      music_volume=1.0 plays it at full level.
    - with a narration, music and ambience are side-chain ducked while the voice speaks (duck=True).
    - the result is trimmed to the video length and soft-limited to avoid clipping."""
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
    if len(layers) == 1:
        chains.append(f"{layers[0]}alimiter=limit=0.97:level=0[a]")
    else:
        chains.append("".join(layers) + f"amix=inputs={len(layers)}:duration=longest:"
                      f"dropout_transition=0:normalize=0,alimiter=limit=0.97:level=0[a]")
    cmd += ["-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", str(out_mp4)]
    _run(cmd)
    return out_mp4
