"""Automated checks of a finished video, run after every assembly.

They FLAG, they never fail a job: each check is {id, label, status, detail} with status one of
pass / warn / fail / manual, stored in plan["qa"] and shown on the job page next to the
"approve for publish" button. "manual" marks what a machine cannot judge (claims, pronunciation,
another platform's watermark) so the reviewer knows it is still theirs to look at.

Two ffmpeg passes do the media work - one over a downscaled picture (black and frozen stretches),
one over the audio (silence, loudness and true peak) - plus a probe and one frame read.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import config
from services.ffbin import ffmpeg, ffprobe


def _check(cid: str, label: str, status: str, detail: str = "") -> dict:
    return {"id": cid, "label": label, "status": status, "detail": detail}


def _ff_stderr(args: list, timeout: int = 600) -> str:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-nostats", *map(str, args)],
                       capture_output=True, text=True, timeout=timeout)
    return r.stderr or ""


def probe(path: Path) -> dict:
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-show_entries",
         "format=format_name,duration:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate",
         "-of", "default=nw=0", str(path)], capture_output=True, text=True).stdout
    info, cur = {"streams": []}, None
    for line in out.splitlines():
        line = line.strip()
        if line == "[STREAM]":
            cur = {}
        elif line == "[/STREAM]":
            info["streams"].append(cur)
            cur = None
        elif "=" in line:
            k, v = line.split("=", 1)
            (cur if cur is not None else info)[k] = v
    return info


def _rate(v: str) -> float:
    try:
        n, d = v.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _aspect(aspect_ratio: str) -> float | None:
    m = re.match(r"\s*(\d+)\s*:\s*(\d+)", aspect_ratio or "")
    return int(m.group(1)) / int(m.group(2)) if m else None


def video_scan(path: Path, freeze_s: float) -> dict:
    """Black stretches and frozen stretches, on a small copy of the picture."""
    err = _ff_stderr(["-i", path, "-an", "-vf",
                      f"scale=240:-2,blackdetect=d=0.25:pix_th=0.10,freezedetect=n=0.003:d={freeze_s}",
                      "-f", "null", "-"])
    black = [(float(a), float(b)) for a, b in
             re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)", err)]
    starts = [float(x) for x in re.findall(r"freeze_start:\s*([\d.]+)", err)]
    durs = [float(x) for x in re.findall(r"freeze_duration:\s*([\d.]+)", err)]
    freezes = list(zip(starts, durs))
    if len(starts) > len(durs):        # frozen through the very end: no duration line printed
        freezes.append((starts[-1], -1.0))
    return {"black": black, "freeze": freezes}


def audio_scan(path: Path) -> dict:
    """Silences (> 0.3 s under -50 dB), integrated loudness and true peak."""
    err = _ff_stderr(["-i", path, "-vn", "-af", "silencedetect=n=-50dB:d=0.3,ebur128=peak=true",
                      "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", err)]
    summary = err[err.rfind("Summary:"):] if "Summary:" in err else ""
    lufs = re.search(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", summary)
    peak = re.search(r"True peak:\s*Peak:\s*(-?[\d.]+|-inf)\s*dBFS", summary, re.S)

    def num(m):
        if not m:
            return None
        return float("-inf") if m.group(1) == "-inf" else float(m.group(1))
    return {"silence": [(max(0.0, s), ends[i] if i < len(ends) else None) for i, s in enumerate(starts)],
            "lufs": num(lufs), "peak": num(peak)}


def first_frame_stats(path: Path) -> tuple:
    """(mean, stddev) luma of the first frame, 0-255."""
    r = subprocess.run([ffmpeg(), "-v", "error", "-i", str(path), "-frames:v", "1", "-vf",
                        "scale=64:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    px = r.stdout
    if not px:
        return 0.0, 0.0
    mean = sum(px) / len(px)
    var = sum((p - mean) ** 2 for p in px) / len(px)
    return mean, var ** 0.5


def caption_overflow(plan: dict, size: tuple, safe: dict | None) -> tuple:
    """(phrases checked, problems): a caption group that needs more than two lines, or a single
    word wider than the caption area, at the size it is actually drawn."""
    from services import captions, storyboard
    words = (plan.get("caption_words") or {}).get("words")
    if not words:
        return 0, []
    W, H = int(size[0]), int(size[1])
    _mv, px, outline = captions._metrics(size, config.CAPTION_BOTTOM_MARGIN, 0, safe)
    ml, mr = captions.side_margins(size, safe)
    avail = W - ml - mr - 2 * outline
    font = storyboard.load_font(px, bold=True)
    phrases = captions.group(words, max_words=config.CAPTION_MAX_WORDS,
                             max_chars=config.CAPTION_MAX_CHARS)
    bad = []
    for p in phrases:
        toks = [w["word"] for w in p["words"]]
        lines, cur = 1, ""
        for t in toks:
            if font.getlength(t) > avail:
                bad.append(f'"{t}" is wider than the caption area')
                break
            test = (cur + " " + t).strip()
            if cur and font.getlength(test) > avail:
                lines, cur = lines + 1, t
            else:
                cur = test
        if lines > 2:
            bad.append(f'"{" ".join(toks)}" needs {lines} lines')
    return len(phrases), bad


def run(job, plan: dict, reel_safe: dict | None = None) -> dict:
    """Every check over job.dir / job.final. Never raises: a check that cannot run says so."""
    d = Path(job.dir)
    final = d / (job.final or "final.mp4")
    checks = []
    if not final.exists():
        checks.append(_check("file", "Final video exists", "fail", f"{final.name} is missing"))
        return {"at": time.time(), "checks": checks, "summary": _summary(checks)}

    # ---- container, codec, size, rate, duration
    info = probe(final)
    vs = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    aus = next((s for s in info["streams"] if s.get("codec_type") == "audio"), {})
    w, h = int(vs.get("width") or 0), int(vs.get("height") or 0)
    dur = float(info.get("duration") or 0)
    fmt_ok = "mp4" in (info.get("format_name") or "") and vs.get("codec_name") == "h264"
    checks.append(_check("container", "MP4 / H.264 / AAC", "pass" if fmt_ok and aus.get("codec_name") == "aac"
                         else "warn", f"{info.get('format_name')}, {vs.get('codec_name')}, {aus.get('codec_name') or 'no audio'}"))
    want = _aspect(job.aspect_ratio)
    if want and h:
        ok = abs(w / h - want) <= 0.01 * want
        checks.append(_check("aspect", "Aspect ratio", "pass" if ok else "fail",
                             f"{w}x{h} for {job.aspect_ratio}"))
    if want and want < 1:
        hd = w >= 1080 and h >= 1920
        checks.append(_check("resolution", "1080x1920", "pass" if hd else "warn",
                             f"{w}x{h}" + ("" if hd else " - fine if the source is lower; never upscale "
                                                         "poor footage just to hit the number")))
    r, avg = _rate(vs.get("r_frame_rate", "")), _rate(vs.get("avg_frame_rate", ""))
    cfr = r and avg and abs(r - avg) < 0.05
    checks.append(_check("fps", "Constant frame rate", "pass" if cfr else "warn", f"{avg:.2f} fps"))
    expected = sum(float(x) for x in (plan.get("clip_seconds") or []))
    if expected:
        off = dur - expected
        checks.append(_check("duration", "Duration matches the plan",
                             "pass" if abs(off) <= config.QA_DURATION_TOLERANCE else "warn",
                             f"{dur:.1f}s against {expected:.1f}s planned"))

    # ---- picture
    try:
        mean, sd = first_frame_stats(final)
        blank = mean < 16 or sd < 6
        checks.append(_check("first_frame", "First frame shows something", "fail" if blank else "pass",
                             f"luma mean {mean:.0f}, contrast {sd:.0f}" + (" - blank or near-black" if blank else "")))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("first_frame", "First frame shows something", "warn", f"could not read it ({e})"))
    try:
        vscan = video_scan(final, float(config.QA_FREEZE_SECONDS))
        bl = vscan["black"]
        checks.append(_check("black", "No black frames", "warn" if bl else "pass",
                             ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in bl[:5]) or "none"))
        fr = [(s, (dur - s) if dd < 0 else dd) for s, dd in vscan["freeze"]]
        checks.append(_check("static", f"No frozen picture over {config.QA_FREEZE_SECONDS:g}s",
                             "warn" if fr else "pass",
                             ", ".join(f"{s:.1f}s for {dd:.1f}s" for s, dd in fr[:5]) or "none"))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("black", "Picture scan", "warn", f"could not run ({e})"))

    # ---- sound
    if not aus:
        checks.append(_check("audio", "Has audio", "fail", "no audio stream"))
    else:
        try:
            a = audio_scan(final)
            edge, gap = float(config.QA_EDGE_SILENCE), float(config.QA_GAP_SILENCE)
            head = tail = 0.0
            inner = []
            for s, e in a["silence"]:
                e = dur if e is None else e
                if s <= 0.05:
                    head = e
                elif e >= dur - 0.05:
                    tail = dur - s
                elif e - s > gap:
                    inner.append((s, e))
            bad_edges = [f"{head:.1f}s at the start"] if head > edge else []
            bad_edges += [f"{tail:.1f}s at the end"] if tail > edge else []
            checks.append(_check("edge_silence", "No dead air at the start or end",
                                 "warn" if bad_edges else "pass", ", ".join(bad_edges) or "none"))
            checks.append(_check("gap_silence", f"No silence over {gap:g}s inside",
                                 "warn" if inner else "pass",
                                 ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in inner[:5]) or "none"))
            pk = a["peak"]
            if pk is not None:
                st = "fail" if pk > 0.0 else "warn" if pk > -1.0 else "pass"
                checks.append(_check("clipping", "No clipping (true peak)", st, f"{pk:.1f} dBTP"))
            if a["lufs"] is not None and config.LOUDNESS_LUFS:
                off = abs(a["lufs"] - float(config.LOUDNESS_LUFS))
                checks.append(_check("loudness", "Loudness on target", "pass" if off <= 2.0 else "warn",
                                     f"{a['lufs']:.1f} LUFS (target {config.LOUDNESS_LUFS:g})"))
        except Exception as e:  # noqa: BLE001
            checks.append(_check("audio", "Audio scan", "warn", f"could not run ({e})"))
        narr = d / "narration.wav"
        if narr.exists() and (job.audio_mode or "") != "music":
            try:
                from services import tts
                spoken = tts.audio_duration(narr)
                cut = spoken > dur - 0.3
                checks.append(_check("end_cut", "Speech ends before the video does", "fail" if cut else "pass",
                                     f"speech {spoken:.1f}s, video {dur:.1f}s"))
            except Exception as e:  # noqa: BLE001
                checks.append(_check("end_cut", "Speech ends before the video does", "warn", str(e)))

    # ---- captions
    if job.captions and plan.get("caption_words") and w and h:
        try:
            n, bad = caption_overflow(plan, (w, h), reel_safe)
            checks.append(_check("captions", "Captions fit (max two lines)", "warn" if bad else "pass",
                                 "; ".join(bad[:4]) or f"{n} groups checked"))
            if reel_safe:
                from services import captions
                _mv, px, _o = captions._metrics((w, h), config.CAPTION_BOTTOM_MARGIN, 0, reel_safe)
                small = px < float(config.REEL_MIN_CAPTION_FONT) * h - 1
                checks.append(_check("caption_size", "Caption text readable on a phone",
                                     "warn" if small else "pass", f"{px}px on a {h}px frame"))
        except Exception as e:  # noqa: BLE001
            checks.append(_check("captions", "Captions fit", "warn", f"could not check ({e})"))

    # ---- files
    missing = [c for c in (job.clips or []) if not (d / c).exists()]
    for sc in ((plan.get("reel") or {}).get("scenes") or []):
        if sc.get("asset") and not (d / sc["asset"]).exists():
            missing.append(sc["asset"])
    checks.append(_check("assets", "No missing clips or footage", "fail" if missing else "pass",
                         ", ".join(missing[:6]) or f"{len(job.clips or [])} clips present"))

    # ---- what only a person can judge
    checks.append(_check("review", "Watch it once", "manual",
                         "claims match the sources, names and numbers are pronounced right, no other "
                         "platform's watermark, the opening is the strongest moment"))
    return {"at": time.time(), "checks": checks, "summary": _summary(checks)}


def _summary(checks: list) -> dict:
    return {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "warn", "fail", "manual")}
