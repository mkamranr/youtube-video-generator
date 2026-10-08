"""Minimal ComfyUI API client: upload images, queue a workflow, wait, download the result."""
import copy
import json
import random
import sys
import time
import uuid
from pathlib import Path

import requests
import websocket

import config


# 1x1 transparent PNG used to overwrite (scrub) server-side files: this ComfyUI build has
# no file-delete API, but the upload API can overwrite any output/input file, reclaiming
# its storage (a ~73-byte stub remains listed).
PNG_STUB = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478"
    "9c626001000000ffff03000006000557bfabd40000000049454e44ae426082")


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str = config.COMFY_URL):
        self.url = url.rstrip("/")
        self.ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://")
        self.client_id = uuid.uuid4().hex

    # ---- helpers -------------------------------------------------------
    def upload_file(self, path: Path, name: str | None = None, mime: str = "") -> str:
        """Put a file in ComfyUI's input directory and return the name a Load* node should use.

        /upload/image is the only upload route this build exposes, but it stores whatever it is
        given: a .wav lands in the same input directory and LoadAudio lists it. That is how the
        avatar graphs get their narration slice without a server-side change."""
        path = Path(path)
        name = name or path.name
        mime = mime or ("audio/wav" if path.suffix.lower() in (".wav", ".mp3", ".flac")
                        else "image/jpeg")
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image",
                              files={"image": (name, f, mime)},
                              data={"overwrite": "true", "type": "input"}, timeout=300)
        r.raise_for_status()
        j = r.json()
        return (j.get("subfolder") + "/" if j.get("subfolder") else "") + j["name"]

    def upload_image(self, path: Path, name: str | None = None) -> str:
        return self.upload_file(path, name, "image/jpeg")

    @staticmethod
    def load_workflow(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def build(template: dict, prompt: str, filename_prefix: str, seed: int | None = None,
              first_frame: str | None = None, seconds: float | None = None,
              aspect_ratio: str | None = None, megapixels: float | None = None,
              model: str = "", negative: str | None = None) -> dict:
        """Patch one render's values into a workflow template.

        Which nodes those values go to depends on the model AND on whether a first frame was
        given - the text-to-video and first-frame graphs are separate files with their own node
        numbering - so `template` has to be the graph config.model_workflow(model, bool(
        first_frame)) names. Everything model-specific is read from config.VIDEO_MODELS; nothing
        here knows a node id.

        `negative` is ignored by a model whose graph has no negative prompt (MiniMax H3)."""
        seconds = config.CLIP_SECONDS if seconds is None else seconds
        aspect_ratio = aspect_ratio or config.DEFAULT_ASPECT_RATIO
        megapixels = config.DEFAULT_MEGAPIXELS if megapixels is None else float(megapixels)
        if aspect_ratio not in config.ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")
        spec = config.model_variant(model, first_frame is not None)
        wf = copy.deepcopy(template)
        node, field = spec["prompt"]
        wf[node]["inputs"][field] = prompt
        if spec["negative"] and negative is not None:
            node, field = spec["negative"]
            wf[node]["inputs"][field] = negative
        seed = random.randint(0, 2**53) if seed is None else int(seed)
        # one seed per noise node: a model that samples in two passes (LTX 2.5) would otherwise
        # keep its second pass pinned to whatever the template shipped with
        for i, (node, field) in enumerate(spec["seeds"]):
            wf[node]["inputs"][field] = seed + i
        node, field = spec["duration"]
        wf[node]["inputs"][field] = config.model_duration_value(seconds, model)
        wf[spec["save"]]["inputs"]["filename_prefix"] = filename_prefix
        if spec["lightning"]:
            wf[spec["lightning"]]["inputs"]["value"] = bool(config.LIGHTNING_LORA)
        wf[spec["resolution"]]["inputs"]["aspect_ratio"] = aspect_ratio
        wf[spec["resolution"]]["inputs"]["megapixels"] = megapixels
        if first_frame is not None:
            wf[spec["load_image"]]["inputs"]["image"] = first_frame
        return wf

    @classmethod
    def build_thumbnail(cls, template: dict, prompt: str, filename_prefix: str,
                        seed: int | None = None,
                        aspect_ratio: str = "16:9 (Widescreen)",
                        first_frame: str | None = None, seconds: float | None = None) -> dict:
        """~1 MP render used as a thumbnail source: minimal length (5 frames) by default, or a
        short clip seeded from `first_frame` (an uploaded cast portrait) whose last frame is kept.

        Always config.STILLS_VIDEO_MODEL, whatever video model the job picked: a 5-frame render is
        a length only that graph can express (LTX 2.5's duration node is whole seconds, so its
        shortest clip is a second). Callers pass the matching config.T2V_WORKFLOW /
        config.I2V_WORKFLOW template."""
        wf = cls.build(template, prompt, filename_prefix, seed=seed,
                       seconds=0.1 if seconds is None else seconds, aspect_ratio=aspect_ratio,
                       first_frame=first_frame, model=config.STILLS_VIDEO_MODEL)
        spec = config.model_variant(config.STILLS_VIDEO_MODEL, first_frame is not None)
        wf[spec["resolution"]]["inputs"]["megapixels"] = 1.0
        return wf

    @staticmethod
    def build_music(template: dict, caption: str, lyrics: str, seconds: float,
                    filename_prefix: str, seed: int | None = None) -> dict:
        wf = copy.deepcopy(template)
        enc = wf[config.NODE_MUSIC_ENCODE]["inputs"]
        enc["caption"] = caption
        enc["lyrics"] = lyrics
        enc["max_duration"] = min(360.0, max(10.0, float(seconds)))
        wf[config.NODE_MUSIC_SEED]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**53)
        wf[config.NODE_MUSIC_SAVE]["inputs"]["filename_prefix"] = filename_prefix
        return wf

    @staticmethod
    def build_avatar(template: dict, engine: str, image: str, audio: str, seconds: float,
                     filename_prefix: str, prompt: str = "", seed: int | None = None,
                     frames: int | None = None) -> dict:
        """Patch one presenter render into an avatar workflow.

        Like build(), every node id comes from the registry (config.AVATAR_MODELS), so this
        knows nothing engine-specific. The two engines express length differently - Wan takes a
        frame count it renders in fixed chunks, LTX a whole-second duration - which is why the
        registry carries `length` for one and `duration` for the other and this sets whichever
        is present."""
        spec = config.avatar_model(engine)
        wf = copy.deepcopy(template)

        def _set(key, value):
            ref = spec.get(key)
            if ref:
                node, field = ref
                wf[node]["inputs"][field] = value

        _set("image", image)
        _set("audio", audio)
        if prompt:
            _set("prompt", prompt)
        seed = random.randint(0, 2**31) if seed is None else int(seed)
        for i, (node, field) in enumerate(spec.get("seeds") or ()):
            wf[node]["inputs"][field] = seed + i
        if spec.get("steps"):
            _set("steps", int(spec.get("default_steps") or 20))
        if spec.get("length"):
            _set("length", int(frames if frames else config.avatar_frames(seconds, engine)))
        if spec.get("duration"):
            # LTX's duration node is whole seconds; its graph renders seconds * fps + 1 frames
            _set("duration", max(1, int(round(float(seconds)))))
        if spec.get("width"):
            _set("width", int(spec["square"]))
            _set("height", int(spec["square"]))
        if spec.get("resolution"):
            wf[spec["resolution"]]["inputs"]["aspect_ratio"] = "1:1 (Square)"
            wf[spec["resolution"]]["inputs"]["megapixels"] = round(spec["square"] ** 2 / 1e6, 2)
        wf[spec["save"]]["inputs"]["filename_prefix"] = filename_prefix
        return wf

    # ---- queue / wait --------------------------------------------------
    def queue(self, workflow: dict) -> str:
        r = requests.post(f"{self.url}/prompt",
                          json={"prompt": workflow, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            raise ComfyError(f"ComfyUI rejected the workflow: {r.text[:2000]}")
        j = r.json()
        if j.get("node_errors"):
            raise ComfyError(f"Workflow validation errors: {json.dumps(j['node_errors'])[:2000]}")
        return j["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        return r.json().get(prompt_id)

    def queue_position(self, prompt_id: str) -> int | None:
        """0 = running, n>0 = position in pending queue, None = not queued."""
        q = requests.get(f"{self.url}/queue", timeout=30).json()
        for item in q.get("queue_running", []):
            if item[1] == prompt_id:
                return 0
        for i, item in enumerate(q.get("queue_pending", []), 1):
            if item[1] == prompt_id:
                return i
        return None

    def wait(self, prompt_id: str, on_progress=None, timeout: float = config.COMFY_TIMEOUT,
             should_stop=None) -> dict:
        """Block until the prompt finishes. Uses the websocket for progress, /history for the result."""
        deadline = time.time() + timeout
        ws = None
        try:
            ws = websocket.create_connection(f"{self.ws_url}/ws?clientId={self.client_id}", timeout=10)
        except Exception:  # noqa: BLE001 - fall back to polling
            ws = None
        last_poll = 0.0
        try:
            while time.time() < deadline:
                if should_stop is not None and should_stop():
                    raise ComfyError("cancelled by user")
                if ws is not None:
                    try:
                        msg = ws.recv()
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            t, d = data.get("type"), data.get("data", {})
                            if d.get("prompt_id") == prompt_id:
                                if t == "progress" and on_progress:
                                    on_progress(f"sampling {d.get('value')}/{d.get('max')}")
                                elif t == "executing" and d.get("node") is None:
                                    break  # finished
                                elif t == "execution_error":
                                    raise ComfyError(f"{d.get('node_type')}: {d.get('exception_message')}")
                    except websocket.WebSocketTimeoutException:
                        pass
                    except ComfyError:
                        raise
                    except Exception:  # noqa: BLE001 - websocket dropped, switch to polling
                        ws = None
                else:
                    time.sleep(5)
                if time.time() - last_poll > 5:
                    last_poll = time.time()
                    h = self.history(prompt_id)
                    if h and h.get("status", {}).get("completed") is not None or (h and h.get("outputs")):
                        break
                    if on_progress and ws is None:
                        pos = self.queue_position(prompt_id)
                        on_progress("running" if pos == 0 else f"queued (position {pos})" if pos else "waiting")
        finally:
            if ws is not None:
                ws.close()
        h = None
        for _ in range(10):
            h = self.history(prompt_id)
            if h:
                break
            time.sleep(1)
        if not h:
            raise ComfyError("Timed out waiting for ComfyUI")
        status = h.get("status", {})
        if status.get("status_str") == "error":
            msgs = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
            detail = msgs[0].get("exception_message") if msgs else "unknown error"
            raise ComfyError(f"ComfyUI execution error: {detail}")
        return h

    def download_output(self, history: dict, dest: Path, node_id: str | None = None) -> Path:
        """Download the saved file. If node_id is None (or has no files), scan all output nodes."""
        outputs = history.get("outputs", {})

        def files_of(out):
            return out.get("videos") or out.get("images") or out.get("gifs") or out.get("audio") or []

        files = files_of(outputs.get(node_id, {})) if node_id else []
        if not files:
            for out in outputs.values():
                files = files_of(out)
                if files:
                    break
        if not files:
            raise ComfyError(f"No output file found: {json.dumps(outputs)[:500]}")
        f = files[0]
        r = requests.get(f"{self.url}/view",
                         params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                 "type": f.get("type", "output")}, timeout=600, stream=True)
        r.raise_for_status()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        return dest

    def run(self, workflow: dict, dest: Path, on_progress=None, should_stop=None,
            cleanup: bool = False, on_cleanup_failed=None) -> Path:
        pid = self.queue(workflow)
        if on_progress:
            on_progress(f"queued in ComfyUI ({pid[:8]})")
        h = self.wait(pid, on_progress=on_progress, should_stop=should_stop)
        result = self.download_output(h, dest)
        if cleanup and not self.cleanup_outputs(h, pid) and on_cleanup_failed:
            on_cleanup_failed()
        return result

    def scrub_file(self, dir_type: str, subfolder: str, filename: str) -> bool:
        """Reclaim a server file's storage by overwriting it with a 1x1 PNG stub."""
        try:
            import io as _io
            r = requests.post(f"{self.url}/upload/image",
                              files={"image": (filename, _io.BytesIO(PNG_STUB),
                                               "application/octet-stream")},
                              data={"overwrite": "true", "type": dir_type,
                                    "subfolder": subfolder}, timeout=30)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def forget_history(self, prompt_id: str):
        """Best-effort removal of the prompt's history entry."""
        try:
            requests.post(f"{self.url}/history", json={"delete": [prompt_id]}, timeout=20)
        except Exception:  # noqa: BLE001
            pass

    def cleanup_outputs(self, history: dict, prompt_id: str) -> bool:
        """Scrub every file this prompt produced on the server, then forget its history."""
        ok = True
        for out in history.get("outputs", {}).values():
            for key in ("videos", "images", "gifs", "audio"):
                for f in out.get(key) or []:
                    ok = self.scrub_file(f.get("type", "output"), f.get("subfolder", ""),
                                         f["filename"]) and ok
        self.forget_history(prompt_id)
        return ok

    def sweep_orphans(self, skip_ids=()) -> int:
        """Scrub every videogen output left on the server (and forget its history) plus any
        full-size uploaded frames, except those belonging to the given active job ids.
        Catches renders orphaned by retries, crashes and restarts. Returns files scrubbed."""
        scrubbed = 0
        try:
            hist = requests.get(f"{self.url}/history", timeout=30).json()
        except Exception:  # noqa: BLE001
            return 0
        for pid, h in hist.items():
            files = [f for o in h.get("outputs", {}).values()
                     for k in ("videos", "images", "gifs", "audio") for f in (o.get(k) or [])]
            ours = [f for f in files if str(f.get("subfolder", "")).startswith("videogen")]
            if not ours:
                continue
            if any(sid and sid in f.get("subfolder", "") for f in ours for sid in skip_ids):
                continue
            for f in ours:
                if self.scrub_file(f.get("type", "output"), f.get("subfolder", ""), f["filename"]):
                    scrubbed += 1
            self.forget_history(pid)
        try:
            inp = requests.get(f"{self.url}/internal/files/input", timeout=20).json()
        except Exception:  # noqa: BLE001
            return scrubbed
        for entry in inp:
            name = str(entry).split(" [")[0]
            if not name.startswith("videogen"):
                continue
            if any(sid and sid in name for sid in skip_ids):
                continue
            try:
                r = requests.get(f"{self.url}/view", params={"filename": name, "type": "input"},
                                 stream=True, timeout=15)
                n = int(r.headers.get("content-length", 0))
                r.close()
            except Exception:  # noqa: BLE001
                continue
            if n > 1000 and self.scrub_file("input", "", name):
                scrubbed += 1
        return scrubbed

    def interrupt_if_ours(self, prefix: str) -> bool:
        """Interrupt the currently running ComfyUI prompt ONLY if it belongs to us."""
        try:
            q = requests.get(f"{self.url}/queue", timeout=15).json()
            for item in q.get("queue_running", []):
                for node in item[2].values():
                    if isinstance(node, dict) and str(
                            node.get("inputs", {}).get("filename_prefix", "")).startswith(prefix):
                        requests.post(f"{self.url}/interrupt", timeout=15)
                        return True
        except Exception:  # noqa: BLE001
            pass
        return False


if __name__ == "__main__":
    from services.video import last_frame

    # python -m services.comfy ["LTX 2.5"]  - two clips, the second seeded from the first
    model = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_VIDEO_MODEL
    c = ComfyClient()
    prompt = ("A calm mountain lake at sunrise, mist drifting over the water, a lone wooden canoe. "
              "[0s-5s] slow push-in across the water. Camera: static tripod, slow dolly. "
              "Audio: gentle water, birds. No text, subtitles, logos or watermarks of any kind.")
    out = Path("output/_comfy_test")
    wf = c.build(c.load_workflow(config.model_workflow(model)), prompt, "videogen/_test/clip_01",
                 model=model, negative=config.ltx_negative())
    clip1 = c.run(wf, out / "clip_01.mp4", on_progress=print)
    print("clip 1:", clip1)
    frame = last_frame(clip1, out / "frame_01.jpg")
    name = c.upload_image(frame, "videogen_test_frame_01.jpg")
    wf = c.build(c.load_workflow(config.model_workflow(model, True)),
                 prompt + " The canoe starts to drift.", "videogen/_test/clip_02",
                 first_frame=name, model=model, negative=config.ltx_negative())
    clip2 = c.run(wf, out / "clip_02.mp4", on_progress=print)
    print("clip 2:", clip2)
