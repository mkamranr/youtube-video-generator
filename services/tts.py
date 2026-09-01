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


def synthesize(text: str, out_wav: Path, on_progress=None,
               ref_audio: Path = None, ref_text: str = "") -> float:
    free_comfy_vram()
    client = Client(config.FISH_URL, verbose=False)
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
        if len(parts) == 1:
            shutil.copy(parts[0], out_wav)
        else:
            listfile = Path(d) / "list.txt"
            listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
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
