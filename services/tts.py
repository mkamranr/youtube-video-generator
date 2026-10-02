"""Fish Speech (Gradio) narration synthesis.

Two server quirks are handled here:
- Requests longer than ~400 characters fail, so text is split into sentence chunks of
  <= config.TTS_CHUNK_CHARS, synthesized separately and concatenated with ffmpeg.
A reference voice (audio sample + its transcript) can be passed to clone that voice.
- Fish shares GPU 0 with ComfyUI and vLLM and can hit CUDA OOM. Before synthesis we ask
  ComfyUI to unload its models (it reloads them on demand), and OOM chunks are retried.
"""
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from gradio_client import Client, handle_file

import config
from services.ffbin import ffmpeg, ffprobe

CHUNK_RETRIES = 3


def audio_duration(path) -> float:
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def split_chunks(text: str, limit: int = 0) -> list:
    limit = limit or config.TTS_CHUNK_CHARS
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > limit:  # single overlong sentence: hard-split on a space
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(s[:cut])
            s = s[cut:].strip()
        if len(cur) + len(s) + 1 <= limit:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def free_comfy_vram():
    """Best-effort: ask ComfyUI to unload models so Fish has VRAM headroom."""
    try:
        requests.post(f"{config.COMFY_URL}/free",
                      json={"unload_models": True, "free_memory": True}, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def _clean_err(err) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(err or ""))).strip() or "unknown error"


def _predict_chunk(client: Client, text: str, ref_audio=None, ref_text: str = "") -> str:
    last = None
    for attempt in range(CHUNK_RETRIES):
        try:
            audio, err = client.predict(
                text=text, reference_id="",
                reference_audio=handle_file(str(ref_audio)) if ref_audio else None,
                reference_text=ref_text or "",
                max_new_tokens=0, chunk_length=config.TTS_CHUNK_CHARS,
                top_p=config.TTS_TOP_P, repetition_penalty=config.TTS_REPETITION_PENALTY,
                temperature=config.TTS_TEMPERATURE, seed=0, use_memory_cache="on",
                api_name="/partial")
            if audio:
                return audio["path"] if isinstance(audio, dict) else audio
            last = _clean_err(err)
        except Exception as e:  # noqa: BLE001
            last = _clean_err(e)
        if "out of memory" in last.lower():
            free_comfy_vram()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Fish TTS failed: {last}")


def connect() -> Client:
    """A Fish Speech client, with ComfyUI asked to free VRAM first.

    Split out of synthesize() so a caller that speaks many short pieces in a row (a story video
    speaks one line at a time) can build the client and free the VRAM once instead of per line."""
    free_comfy_vram()
    return Client(config.FISH_URL, verbose=False)


def synthesize(text: str, out_wav: Path, on_progress=None,
               ref_audio: Path = None, ref_text: str = "", keep_voice: bool = True,
               client: Client = None) -> float:
    """Speak `text` into out_wav, chunk by chunk.

    Without a reference sample Fish Speech invents a new voice for every request, so a
    long narration would switch speakers at each chunk. keep_voice=True (default) makes
    the first chunk's audio the reference for all following chunks, so one video keeps one
    voice; keep_voice=False keeps the old behaviour (a random voice per chunk).

    client: reuse an existing connect() client instead of making one (and freeing VRAM) per call.
    Note that keep_voice only holds WITHIN one call - a caller speaking line by line must carry the
    reference across calls itself, or every line gets a different voice."""
    client = client or connect()
    chunks = split_chunks(text)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    with tempfile.TemporaryDirectory() as d:
        for i, chunk in enumerate(chunks, 1):
            if on_progress:
                on_progress(f"narration chunk {i}/{len(chunks)}")
            src = _predict_chunk(client, chunk, ref_audio=ref_audio, ref_text=ref_text)
            part = Path(d) / f"part_{i:03d}.wav"
            # normalize every part to the same format so concat is safe
            subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", src,
                            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(part)], check=True)
            parts.append(part)
            if i == 1 and ref_audio is None and keep_voice and len(chunks) > 1:
                ref_audio, ref_text = part, chunk   # clone the voice we just got for the rest
        if len(parts) == 1:
            shutil.copy(parts[0], out_wav)
        else:
            listfile = Path(d) / "list.txt"
            listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
            subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                            "-i", str(listfile), "-c", "copy", str(out_wav)], check=True)
    return audio_duration(out_wav)


AUDIO_ARGS = ("-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le")   # what synthesize() writes


def silence(seconds: float, out_wav: Path) -> Path:
    """A silent wav in the same format every synthesized part uses, so concat can stream-copy."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=mono", "-t", f"{max(0.001, float(seconds)):.3f}",
                    *AUDIO_ARGS, str(out_wav)], check=True)
    return out_wav


def concat_timed(parts: list, out_wav: Path, total: float = 0.0) -> float:
    """One wav in which each (wav, start_seconds) part begins at its own start time.

    The gaps between parts - and the tail up to `total` - are digital silence. Used by story videos,
    where every spoken line is synthesized on its own and then laid out on the clip grid, so the
    words land exactly where the scene that shows them is rendered. Parts must not overlap; they are
    placed in the order given and a start that has already passed is clamped to the cursor."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        pieces, cursor, n = [], 0.0, 0
        for src, start in parts:
            gap = float(start) - cursor
            if gap > 0.001:
                n += 1
                pieces.append(silence(gap, Path(d) / f"sil_{n:04d}.wav"))
                cursor += gap
            pieces.append(Path(src))
            cursor += audio_duration(src)
        pad = float(total or 0.0) - cursor
        if pad > 0.001:
            n += 1
            pieces.append(silence(pad, Path(d) / f"sil_{n:04d}.wav"))
        if not pieces:
            return audio_duration(silence(0.5, out_wav))
        listfile = Path(d) / "list.txt"
        listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces), encoding="utf-8")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listfile), "-c", "copy", str(out_wav)], check=True)
    return audio_duration(out_wav)


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of the narration voice."
    ref_audio, ref_text = None, ""
    if len(sys.argv) > 2:  # optional reference voice id from the Settings library
        from services import voices
        v = voices.get_voice(sys.argv[2])
        if not v:
            sys.exit(f"unknown voice id: {sys.argv[2]}")
        ref_audio, ref_text = voices.voice_audio_path(v["id"]), v["text"]
        print(f"using reference voice: {v['name']}")
    d = synthesize(text, Path("output/_tts_test.wav"), on_progress=print,
                   ref_audio=ref_audio, ref_text=ref_text)
    print(f"wrote output/_tts_test.wav ({d:.1f}s, {len(split_chunks(text))} chunks)")
