"""ffmpeg helpers: last-frame extraction, concatenation and narration mixing."""
import subprocess
from pathlib import Path

import config
from services.ffbin import ffmpeg, ffprobe


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def last_frame(video: Path, out_jpg: Path) -> Path:
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    # -sseof seeks relative to the end; -update 1 keeps only the last decoded frame.
    _run([ffmpeg(), "-y", "-v", "error", "-sseof", "-0.3", "-i", str(video),
          "-update", "1", "-q:v", "2", str(out_jpg)])
    if not out_jpg.exists():
        raise RuntimeError("could not extract last frame")
    return out_jpg


def make_thumbnail(video: Path, out_jpg: Path, width: int = 1280, height: int = 720) -> Path:
    """First frame of the video, cover-cropped to exactly width x height, saved as JPG."""
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg(), "-y", "-v", "error", "-i", str(video),
          "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                 f"crop={width}:{height}",
          "-frames:v", "1", "-q:v", "2", str(out_jpg)])
    if not out_jpg.exists():
        raise RuntimeError("could not extract thumbnail frame")
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


def mix_audio(video: Path, out_mp4: Path, narration: Path | None = None,
              music: Path | None = None, clip_volume: float | None = None,
              music_volume: float | None = None) -> Path:
    """Mix any of: narration (full volume), music, clip ambience - over the video.

    clip_volume=0 drops the clips' own audio; music_volume=1.0 plays the music at full level."""
    clip_volume = config.CLIP_AUDIO_VOLUME if clip_volume is None else clip_volume
    music_volume = config.MUSIC_VOLUME if music_volume is None else music_volume
    out_mp4 = Path(out_mp4)
    has_audio = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip() != ""
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    mix, labels, idx = [], "", 1
    if narration is not None:
        cmd += ["-i", str(narration)]
        mix.append(f"[{idx}:a]apad[vo]"); labels += "[vo]"; idx += 1
    if music is not None:
        cmd += ["-i", str(music)]
        mix.append(f"[{idx}:a]volume={music_volume},apad[mu]"); labels += "[mu]"; idx += 1
    if has_audio and clip_volume > 0:
        mix.insert(0, f"[0:a]volume={clip_volume},apad[bg]"); labels = "[bg]" + labels
    n = labels.count("[")
    if n == 0:  # no audio at all: keep the video silent
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-c:v", "copy", "-an", str(out_mp4)])
        return out_mp4
    if n == 1:
        filt = mix[0][:mix[0].rfind("[")] + "[a]"
    else:
        filt = ";".join(mix) + f";{labels}amix=inputs={n}:duration=shortest:dropout_transition=0:normalize=0[a]"
    cmd += ["-filter_complex", filt, "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)]
    _run(cmd)
    return out_mp4
