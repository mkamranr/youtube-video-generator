"""Presenter avatar: animate a channel's portrait so it lip-syncs the voiceover.

Three steps, deliberately split by where they run:

  segment_windows()  decides WHERE the presenter speaks, from the caption word timings the
                     pipeline already has, snapped to gaps between words so a segment never
                     starts or stops mid-syllable.
  render_segment()   animates the portrait against one slice of narration, on the render box.
                     This is the expensive half - Wan is ~9.2 s of GPU per output frame.
  matte_segment()    cuts the background away, on the LOCAL CPU. Matting is cheap (~0.7 s a
                     frame) and the render box is a shared, serial resource, so there is no
                     reason to spend it on something a laptop core can do. It also sidesteps
                     the box being air-gapped: `rembg` is not installed there and cannot be.

The matte is written as a plain grayscale video; services/video.composite() alphamerges it
with the colour video at assembly time, which is the same path the circle bubble takes with a
static mask. Nothing here knows how the presenter is finally drawn.
"""
import random
import subprocess
from pathlib import Path

import config
from services.ffbin import ffmpeg

MIN_SEGMENT = 1.5          # shorter than this reads as a glitch rather than a presenter
GAP_SEARCH = 1.2           # how far to hunt for a word gap when snapping a window edge
SEAM = 0.05                # windows closer than this are one appearance, not two. The same
                           # number as video.SEAM, deliberately: what counts as one appearance
                           # has to be the same answer for the chain, the seed and the overlay.


def _duration(path: Path) -> float:
    from services import video
    return video.duration(path)


def describe(wins: list, total: float) -> str:
    """Where the presenter ended up, in the words the dialog offers - so a finished job can say
    what it got next to what was asked for, instead of leaving the user to read the log and work
    it out from two timestamps."""
    if not wins:
        return "nothing"
    total = float(total or 0.0)
    head = wins[0][0] <= 0.05
    tail = total and wins[-1][1] >= total - 0.05
    covered = sum(e - s for s, e in wins)
    chained = all(b[0] - a[1] < 0.05 for a, b in zip(wins, wins[1:]))
    if chained and head and tail and total and covered >= total - 0.5:
        return "the whole video"
    if len(wins) == 1:
        return "the start" if head else "the end" if tail else "the middle"
    if len(wins) == 2 and head and tail:
        return "the start and the end"
    return f"{len(wins)} segments"


def segment_windows(coverage: str, total: float, engine: str = "", words=None) -> list:
    """Windows the presenter speaks in, as [(start, end)] in seconds.

    Clamped three ways, in order: the engine's single-pass limit (Wan renders one 77-frame
    chunk, so 4.81 s), then the per-video budget, then the video's own length. A coverage the
    budget cannot afford degrades rather than failing - "full" on a long video becomes an
    intro and an outro, which is the closest thing to what was asked for."""
    total = float(total or 0.0)
    if coverage == "none" or total <= 0:
        return []
    budget = config.avatar_budget_seconds(engine)
    if budget <= 0:
        return []
    cap = config.avatar_model(engine).get("max_segment") or 0.0

    def _clip(length):
        length = min(length, total)
        return min(length, cap) if cap else length

    intro = _clip(float(config.AVATAR_INTRO_SECONDS))
    outro = _clip(float(config.AVATAR_OUTRO_SECONDS))
    chained = False
    if coverage == "full" and total <= budget:
        # One window where the engine renders the whole thing in a pass (LTX), a chain of
        # cap-sized ones where it cannot (Wan renders a single 77-frame chunk). Laid end to
        # end rather than stitched: the overlay composites every segment at its own start,
        # so a chain costs no extra encode - only a pose reset at each seam, which is what a
        # presenter who never leaves costs on a chunked engine.
        wins = [(0.0, total)] if not cap else _tile(total, cap)
        chained = len(wins) > 1
    elif coverage == "intro":
        wins = [(0.0, intro)]
    elif coverage == "outro":
        wins = [(max(0.0, total - outro), total)]
    else:
        # "both", and "full" on a video the budget cannot cover: the start and the end
        wins = [(0.0, intro), (max(0.0, total - outro), total)]
    # overlapping windows on a very short video collapse to one - but a chain is contiguous
    # by construction, so the same test would collapse a two-chunk full video into one chunk
    if not chained and len(wins) == 2 and wins[1][0] <= wins[0][1] + MIN_SEGMENT:
        wins = [(0.0, min(total, intro))]
    wins = [(s, e) for s, e in wins if e - s >= MIN_SEGMENT]
    # spend the budget front to back, so an intro survives when only one of the two fits
    out, spent = [], 0.0
    for s, e in wins:
        room = budget - spent
        if room < MIN_SEGMENT:
            break
        e = min(e, s + room)
        if e - s < MIN_SEGMENT:
            continue
        # a chain's interior edges stay exact: snapping each to its nearest word boundary
        # would pull them apart and leave the presenter blinking out in the gaps
        out.append((round(s, 3), round(e, 3)) if chained else _snap(s, e, words, total))
        spent += e - s
    return out


def _tile(total: float, cap: float) -> list:
    """[0, total] cut into consecutive cap-length windows, the last one short.

    What "the whole video" means on an engine that renders a fixed chunk: the segments are
    contiguous, so the presenter is on screen from the first frame to the last."""
    out, t = [], 0.0
    while t < total - 0.01:
        e = min(total, t + cap)
        # rounded here, once, so two neighbours share the very same boundary: rounding each
        # window separately later would leave a sub-millisecond seam between them
        out.append((round(t, 3), round(e, 3)))
        t = e
    return out


def appearances(wins: list) -> list:
    """Which appearance each window belongs to, as a 1-based index per window.

    A contiguous run of windows is ONE appearance rendered in chunks; a gap starts another. The
    bookend coverages give two appearances, "full" on a chunked engine gives one of many chunks,
    and the caller needs to tell them apart to decide what may be chained and what shares a
    seed. Derived from the windows rather than tracked through the render loop, so a resumed job
    - which skips the chunks already on disk - gets the same answer as a fresh one."""
    out, n = [], 0
    for i, (s, _e) in enumerate(wins):
        if i == 0 or s - wins[i - 1][1] >= SEAM:
            n += 1
        out.append(n)
    return out


def chain_plan(wins: list, engine: str = "", max_chain: int = 0) -> list:
    """Whether each window may continue the one before it, as a flag per window.

    False means that chunk is rendered from the portrait alone, with no ref_motion - which is how
    the first chunk of every appearance has always rendered, so it needs no new graph.

    The point of saying no on purpose: chaining hands each chunk the PIXELS of the one before it,
    so a chunk eight deep is conditioned on eight generations of small deviations and the face has
    walked off the portrait. `max_chain` is the longest run allowed to share one anchor, so 4
    anchors chunks 1, 5, 9... and no chunk is ever more than three hops from the photograph. 0
    keeps one unbroken chain, which is what this did before.

    Derived from the windows, not counted as the render loop goes: a resumed job skips the chunks
    already on disk, so a counter incremented per render would re-anchor in different places the
    second time round."""
    apps = appearances(wins)
    chainable = bool(config.avatar_model(engine).get("motion"))
    cap = max(0, int(max_chain or 0))
    out, start = [], 0
    for i, app in enumerate(apps):
        if i == 0 or app != apps[i - 1]:
            start = i                       # first chunk of an appearance: nothing to continue
            out.append(False)
        else:
            out.append(chainable and (not cap or (i - start) % cap != 0))
    return out


def chunk_seed(job_id: str, appearance: int = 1) -> int:
    """The sampler seed every chunk of one appearance renders on.

    Fixed per appearance rather than drawn fresh per chunk, which is what ComfyClient.build_avatar
    falls back to. With the portrait pinned as ref_image for the whole chain, fresh noise makes the
    model land on its own reading of the same face in every chunk, so the likeness wobbles across
    the seams on top of whatever the motion chain drifts. It is also the only reason two runs can
    be compared at all: without it every render differs, and no change to the chain can be
    measured.

    Per APPEARANCE, not per video: the intro and the outro of a bookend pair are two separate
    performances and would read as a loop if they rendered identically.

    Derived from the job id the way the card layouts are (cardtemplates.pick_templates), so a
    resume seeds exactly as the first pass did. Kept clear of 2**31 because build_avatar spreads
    `seed + i` over an engine's seed nodes."""
    return random.Random(f"{job_id}|avatar|{appearance}").randrange(2 ** 31 - 16)


def _snap(start: float, end: float, words, total: float) -> tuple:
    """Pull a window's edges onto gaps between spoken words.

    Cutting a lip-synced clip mid-word is the one artefact a viewer always notices, and the
    word timings are already on the plan for the captions, so this costs nothing."""
    ws = [w for w in (words or []) if isinstance(w, dict)
          and w.get("start") is not None and w.get("end") is not None]
    if not ws:
        return (round(start, 3), round(end, 3))
    if start > 0:
        near = [float(w["start"]) for w in ws
                if abs(float(w["start"]) - start) <= GAP_SEARCH]
        if near:
            start = min(near, key=lambda v: abs(v - start))
    if end < total:
        near = [float(w["end"]) for w in ws if abs(float(w["end"]) - end) <= GAP_SEARCH]
        if near:
            end = min(near, key=lambda v: abs(v - end))
    return (round(max(0.0, start), 3), round(min(total, end), 3))


def slice_audio(narration: Path, start: float, end: float, dst: Path, engine: str = "") -> float:
    """Cut [start, end] out of the narration, padded to exactly the length the engine renders.

    The pad matters: Wan produces a whole number of 77-frame chunks whatever it is asked for,
    so an audio slice shorter than that would leave the tail of the clip animating silence."""
    spec = config.avatar_model(engine)
    frames = config.avatar_frames(end - start, engine)
    exact = frames / float(spec["fps"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(narration),
                    "-t", f"{exact:.4f}", "-af", f"apad=whole_dur={exact:.4f}",
                    "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True)
    return exact


def motion_tail(clip: Path, dest: Path, engine: str = "") -> Path:
    """The end of a finished chunk, to be handed to the next one as the motion it continues.

    Wan S2V takes up to `motion_seconds` of ref_motion and keeps the LAST frames of whatever it
    is given, so the tail is both all it can use and all that is worth uploading. Re-encoded
    rather than stream-copied: -sseof lands on the nearest keyframe, and a chunk this short has
    one at the front, so a copy would hand over the whole clip.

    `clip` wants to be the chunk as it came OFF the box rather than trim_lead_frame's output.
    This is a path that feeds itself - chunk N's pixels are chunk N+1's conditioning - so every
    generation of lossy encoding in it compounds over the chain. The frames are the same either
    way: the tail is the last 73 of a 77-frame render and of the 76-frame clip trimmed from it,
    which are both raw frames 4..76, so taking the earlier file costs nothing and saves an
    encode. crf 0 for the same reason, and the pixel format and codec stay what ComfyUI's
    SaveVideo already wrote - h264 yuv420p - so LoadVideo on the box opens the same thing it
    always did. It costs about 8 MB a tail against 1 MB at crf 16, and a second to encode:
    nothing next to the twelve minutes of GPU the chunk it conditions will take."""
    spec = config.avatar_model(engine)
    secs = float(spec.get("motion_seconds") or 0.0)
    if secs <= 0:
        raise RuntimeError(f"{engine or config.DEFAULT_AVATAR_MODEL} cannot be chained")
    fps = float(spec["fps"])
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-sseof", f"-{secs:.4f}", "-i", str(clip),
                    "-an", "-r", f"{fps}", "-c:v", "libx264", "-crf", "0",
                    "-pix_fmt", "yuv420p", str(dest)], check=True)
    if not Path(dest).exists():
        raise RuntimeError("could not cut the motion tail")
    return Path(dest)


def render_segment(client, engine: str, image_name: str, audio_name: str, seconds: float,
                   dest: Path, prefix: str, prompt: str = "", seed=None,
                   on_progress=None, should_stop=None, on_cleanup_failed=None,
                   motion_name: str = "") -> Path:
    """Animate the portrait against one narration slice. Returns the raw clip.

    `motion_name` is an uploaded motion_tail() from the chunk before this one, when there is
    one: it selects the engine's chained graph, and the presenter carries his pose and gesture
    across the seam instead of snapping back to the portrait. An engine with no chained graph
    ignores it, as does the first chunk of any appearance.

    cleanup=COMFY_CLEANUP like every other render: the clip is scrubbed from the server the
    moment it has been downloaded, rather than waiting for the orphan sweep. A presenter
    segment is a 640x640 video and the box is shared, so leaving it there is not free."""
    from services import comfy
    tpl = comfy.ComfyClient.load_workflow(config.avatar_workflow(engine, chain=bool(motion_name)))
    wf = comfy.ComfyClient.build_avatar(tpl, engine, image=image_name, audio=audio_name,
                                        seconds=seconds, filename_prefix=prefix,
                                        prompt=prompt or config.AVATAR_PROMPT, seed=seed,
                                        motion=motion_name)
    # name the node that saves: the chained graph also contains a node that LOADS a video,
    # and every output node is otherwise fair game for the download
    client.run(wf, dest, on_progress=on_progress, should_stop=should_stop,
               cleanup=config.COMFY_CLEANUP, on_cleanup_failed=on_cleanup_failed,
               node_id=config.avatar_model(engine).get("save"))
    return dest


def trim_lead_frame(src: Path, dst: Path, engine: str = "") -> Path:
    """Drop the engine's reference frame, if it has one.

    Wan's frame 0 is the ref-image conditioning slot rather than generated picture, and it
    comes out visibly broken. The official template drops it with LatentCut; doing it here
    costs one cheap re-encode and keeps the graph a straight line."""
    spec = config.avatar_model(engine)
    if not spec.get("drop_first"):
        if src != dst:
            dst.write_bytes(src.read_bytes())
        return dst
    fps = float(spec["fps"])
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src),
                    "-vf", f"select='gt(n\\,0)',setpts=N/{fps}/TB",
                    "-an", "-r", f"{fps}", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


_U2NET_MEAN = (0.485, 0.456, 0.406)
_U2NET_STD = (0.229, 0.224, 0.225)


def _u2net():
    """The CPU matting session and its io names. Shared by the still and the video path."""
    import onnxruntime as ort

    model = config.MODELS_DIR / config.AVATAR_MATTE_MODEL
    if not model.exists():
        raise FileNotFoundError(f"avatar matting model missing: {model}")
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def _alpha(sess, inp, out0, rgb):
    """u2net alpha for one RGB frame, 0..1, back at the frame's own size."""
    import numpy as np
    import cv2

    x = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(_U2NET_MEAN, np.float32)) / np.array(_U2NET_STD, np.float32)
    d = sess.run([out0], {inp: x.transpose(2, 0, 1)[None]})[0][0, 0]
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def _subject_bounds(img, thresh: float = 0.5):
    """(left, top, right, bottom) of the figure in a still, from the cut-out matte.

    None when matting is unavailable, so framing degrades to a fixed anchor rather than
    failing a render over a nicety."""
    try:
        import numpy as np

        sess, inp, out0 = _u2net()
        a = _alpha(sess, inp, out0, np.asarray(img.convert("RGB")))
        ys, xs = np.where(a >= thresh)
        if not len(ys):
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    except Exception:          # noqa: BLE001 - framing must never fail a render
        return None


def _window(span: int, side: int, lead, headroom: float) -> int:
    """Where a `side`-long window starts inside a `span`-long axis.

    `lead` is where the subject starts on that axis, or None when matting is unavailable -
    which anchors at 0, because on a portrait the head is at the top and the top is therefore
    the safe end to keep."""
    start = 0 if lead is None else int(round(lead - headroom * side))
    return max(0, min(start, span - side))


def square(img, headroom: float = 0.07):
    """Crop a portrait to a square, anchored on the figure rather than on the image centre.

    The avatar graphs hand the reference image straight to the sampler, so the fit to the
    model's square is ComfyUI's own - and that is a CENTRED cover-crop. A 944x1133 headshot
    loses 94 px off the top, which comes out as hair clipped flat against the frame edge, and
    the model then animates that flat edge for the whole clip. So the crop happens here
    instead: the same u2net matte the cut-out uses finds where the head starts, and the square
    is placed to leave `headroom` of itself above it. A tall source therefore loses the bottom
    of the torso, which is what the cut-out runs off the frame edge anyway.

    Cover, not letterbox: padding would shrink the face inside the 640 px the model gets, and
    the bars would be picture the sampler is free to invent in.

    An already-square image is returned untouched, and - this is the point - without matting,
    so the guard left in the render path costs nothing once the portrait has been prepared."""
    W, H = img.size
    if W == H:
        return img
    side = min(W, H)
    box = _subject_bounds(img)
    if H > W:
        # vertical window: hold the top of the head, with a little air above it
        left, top = 0, _window(H, side, None if box is None else box[1], headroom)
    else:
        # horizontal window: centre on the figure, not on the middle of the photo
        mid = (box[0] + box[2]) // 2 if box else (W // 2)
        left, top = max(0, min(mid - side // 2, W - side)), 0
    return img.crop((left, top, left + side, top + side))


def portrait_square(src: Path, dst: Path, engine: str = "", headroom: float = 0.07) -> Path:
    """`square()` on a file, then sized to what the engine renders.

    Kept on the render path as a guard for portraits uploaded before squaring moved to upload
    time. For a prepared (already square) avatar it is a resize and nothing else - no matting,
    no model load - so it costs milliseconds rather than the ~2 s it would otherwise add to
    every single video."""
    from PIL import Image

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        out = square(im.convert("RGB"), headroom)
        want = int(config.avatar_model(engine).get("square") or 0)
        if want and want != out.size[0]:
            out = out.resize((want, want), Image.LANCZOS)
        out.save(dst, "PNG")
    return dst


def thumbnail_cutout(clip: Path, dst: Path, matte: Path | None = None, at: float = 0.4) -> Path:
    """One frame of the presenter as an RGBA cut-out PNG, for compositing onto a thumbnail.

    `at` is a fraction of the clip rather than a timestamp: a little past the start, where the
    face has settled and the mouth is mid-sentence. Frame 0 is the worst choice available - on
    Wan it is the reference-image slot, and even trimmed the opening frame has a closed mouth.

    The alpha comes from the segment's own matte video when one was made (the cut-out look
    already paid for it), and otherwise from a single u2net pass on this one frame - which is
    well under a second, and is what lets the bubble look have a cut-out thumbnail."""
    from PIL import Image
    import numpy as np

    clip, dst = Path(clip), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = max(0.0, _duration(clip) * min(max(at, 0.0), 0.99))
    rgb = dst.with_suffix(".rgb.png")
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(clip),
                    "-frames:v", "1", str(rgb)], check=True)
    img = Image.open(rgb).convert("RGB")
    if matte is not None and Path(matte).exists():
        a = dst.with_suffix(".a.png")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(matte),
                        "-frames:v", "1", str(a)], check=True)
        with Image.open(a) as am:
            alpha = am.convert("L").resize(img.size, Image.LANCZOS)
        a.unlink(missing_ok=True)
    else:
        sess, inp, out0 = _u2net()
        m = _alpha(sess, inp, out0, np.asarray(img))
        alpha = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), "L")
    out = img.convert("RGBA")
    out.putalpha(alpha)
    out.save(dst, "PNG")
    rgb.unlink(missing_ok=True)
    return dst


def matte_bounds(matte: Path, thresh: float = 0.5) -> tuple | None:
    """Where the figure sits inside the engine's square frame, as (x0, y0, x1, y1) in 0..1.

    The union over every frame, so the box holds the presenter for the whole segment rather
    than breathing with the gesture. It is worth measuring because the engines centre a
    head-and-shoulders portrait in a square and how much of that square the figure fills is
    up to the render: where it leaves air around the head, drawing the whole frame spends the
    box on transparency and the presenter reads as smaller than the size settings asked for.
    services/video.avatar_layers uses this to put the figure in the box rather than the frame;
    a figure that already fills its frame measures (0, 0, 1, 1) and nothing changes.

    Read back from the finished matte rather than measured during matte_segment, so a video
    rendered before this existed picks it up on a reassembly. None when the matte cannot be
    read or found nothing - the caller then draws the whole frame, exactly as it used to."""
    try:
        import numpy as np
        from services import video

        W, H = video.frame_size(matte)
        raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(matte), "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        peak = np.frombuffer(raw, np.uint8).reshape(-1, H, W).max(axis=0)
        ys, xs = np.where(peak >= int(thresh * 255))
        if not len(ys):
            return None
        return (float(xs.min()) / W, float(ys.min()) / H,
                float(xs.max() + 1) / W, float(ys.max() + 1) / H)
    except Exception:          # noqa: BLE001 - framing must never fail a render
        return None


def _face_models():
    """The YuNet detector and the SFace recogniser. Raises when either file is missing.

    cv2, not onnxruntime: opencv-python ships both of these as first-class objects, so the
    alignment crop SFace needs comes with them instead of having to be reimplemented. Same
    arrangement as the matting either way - a CPU model in MODELS_DIR, kept off the shared box."""
    import cv2

    det = config.MODELS_DIR / config.AVATAR_FACE_DETECT_MODEL
    emb = config.MODELS_DIR / config.AVATAR_FACE_EMBED_MODEL
    for p in (det, emb):
        if not p.exists():
            raise FileNotFoundError(f"avatar face model missing: {p}")
    return (cv2.FaceDetectorYN_create(str(det), "", (320, 320), 0.6, 0.3, 5000),
            cv2.FaceRecognizerSF_create(str(emb), ""))


def _embed(det, rec, bgr):
    """The identity vector of the clearest face in one BGR frame, or None if there is no face."""
    import numpy as np

    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    ok, faces = det.detect(bgr)
    if not ok or faces is None or not len(faces):
        return None
    # by confidence rather than by position: YuNet promises no ordering
    return rec.feature(rec.alignCrop(bgr, faces[int(np.argmax(faces[:, 14]))]))


def face_models_ready() -> bool:
    """Whether the likeness can be measured at all, without loading anything.

    Checked before the work starts so a deployment that never staged the two files pays nothing
    - not an ffmpeg call per chunk, and not a 37 MB model load."""
    return ((config.MODELS_DIR / config.AVATAR_FACE_DETECT_MODEL).exists()
            and (config.MODELS_DIR / config.AVATAR_FACE_EMBED_MODEL).exists())


def _sample_frames(clip: Path, samples: int) -> list:
    """Frame indices to measure a chunk at: evenly spread, past the clip's lead-in.

    The lead-in is skipped because it is not a fair sample of the likeness. Scoring every frame
    of a Wan chunk against its portrait: frame 0 lands at 0.36 against a take mean of 0.81,
    because it is the ref-image conditioning slot rather than generated picture, and frames 1-4
    sit at 0.69-0.71 while the model settles out of the reference. Those five frames alone pull
    the standard deviation of a take from 0.041 to 0.065 - noise the measurement does not need.
    thumbnail_cutout skips the opening for the same reason."""
    from services import video

    fps = video.fps(clip) or 0.0
    total = int(round(float(video.duration(clip) or 0.0) * fps))
    if total < 3:
        return [max(0, total - 1)]
    lead, last = max(1, total // 10), total - 1
    if samples >= last - lead + 1:
        return list(range(lead, last + 1))
    step = (last - lead) / float(samples - 1) if samples > 1 else 0.0
    return sorted({min(last, lead + int(round(k * step))) for k in range(samples)})


def face_identity(clip: Path, portrait: Path, samples: int = 5) -> dict | None:
    """How much like the portrait the presenter still looks, over one chunk.

    {"mean", "min", "faces", "frames"} - cosine similarity against the portrait's own face,
    sampled across the clip rather than taken from one frame, because a single frame mid-blink
    or mid-syllable is noise. `faces` below `frames` is itself the strongest signal available:
    a likeness that has melted stops being detected as a face at all.

    Read this as a COARSE instrument, and read the trend rather than any single number. Measured
    on the spike footage: a different person sits 0.44 below the real one, which is decisive,
    and two takes of the same engine agree to 0.002. But the gap between the engine that holds
    the likeness and the one the notes call "a younger, thinner stranger" is only 0.053, against
    a 5-frame mean's own error of about 0.018 (take sd 0.041). So this will show a chain that
    has collapsed, and a drift that moves a chunk mean by more than ~0.04 - it will not
    adjudicate a subtle resemblance, and a smaller move than that is not evidence of anything.

    None when the models are not staged, the clip cannot be read, or the portrait has no
    findable face. Never raises - the same contract as matte_bounds and _subject_bounds, because
    a measurement must never cost a render that has already been paid for."""
    try:
        import numpy as np
        import cv2
        from services import video

        if not face_models_ready():
            return None
        det, rec = _face_models()
        ref = _embed(det, rec, cv2.imread(str(portrait)))
        if ref is None:
            return None
        W, H = video.frame_size(clip)
        # One decode of a handful of frames, named by index rather than thinned by the fps
        # filter: a whole-clip rawvideo pass (what matting does) would be ~90 MB for a chunk
        # this size, and asking for a RATE would make the frame count depend on the duration
        # probe - which, if it ever came back wrong, would quietly pull in frame 0. Frame 0 is
        # the one frame that must not be measured: on Wan it is the reference-image slot and
        # comes out broken, and even on a trimmed clip the opening frame has a closed mouth
        # (thumbnail_cutout avoids it for the same reason). An index past the end simply yields
        # no frame, so a bad probe costs samples instead of corrupting the score.
        picks = _sample_frames(clip, max(1, int(samples)))
        if not picks:
            return None
        sel = "+".join(rf"eq(n\,{n})" for n in picks)
        raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(clip),
                              "-vf", f"select='{sel}'", "-vsync", "0",
                              "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-"], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
        scores = []
        for f in frames:
            e = _embed(det, rec, np.ascontiguousarray(f))
            if e is not None:
                scores.append(float(rec.match(ref, e, cv2.FaceRecognizerSF_FR_COSINE)))
        if not scores:
            return {"mean": 0.0, "min": 0.0, "faces": 0, "frames": int(len(frames))}
        return {"mean": round(float(np.mean(scores)), 4), "min": round(min(scores), 4),
                "faces": len(scores), "frames": int(len(frames))}
    except Exception:          # noqa: BLE001 - measuring must never fail a render
        return None


def matte_segment(src: Path, dst: Path, on_progress=None) -> Path:
    """Per-frame alpha for the cut-out look, as a grayscale video. CPU only.

    u2net human segmentation through onnxruntime - `rembg` is just a wrapper around these same
    ONNX models, and this way the environment gains no dependency. Three post-steps earn their
    keep: a 3-tap temporal mean (the per-frame matte's edge jitters, which reads as a crawling
    halo once the cut-out sits on moving video), a 1 px erode (pulls the edge inside the colour
    fringe left by the original background) and a small feather."""
    import numpy as np
    import cv2
    from services import video

    sess, inp, out0 = _u2net()
    W, H = video.frame_size(src)
    fps = video.fps(src) or 24.0
    raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(src), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    mattes = []
    for i, f in enumerate(frames):
        mattes.append(_alpha(sess, inp, out0, f))
        if on_progress and (i + 1) % 25 == 0:
            on_progress(f"matting frame {i + 1}/{len(frames)}")
    m = np.stack(mattes)
    m = np.concatenate([m[:1], m, m[-1:]])
    m = (m[:-2] + m[1:-1] + m[2:]) / 3.0
    k = np.ones((3, 3), np.uint8)
    buf = bytearray()
    for f in m:
        a = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        a = cv2.GaussianBlur(cv2.erode(a, k, iterations=1), (5, 5), 0)
        buf += a.tobytes()
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                    "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)],
                   input=bytes(buf), check=True)
    return dst
