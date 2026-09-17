"""Screenshot a web page with a locally installed Chrome, driven over the DevTools protocol.

No pip dependency: the browser binary is located the way services/ffbin.py locates ffmpeg
(CHROME_BIN -> known install paths -> PATH), and it is driven with `websocket-client`, which the
project already depends on for ComfyUI (services/comfy.py).

CDP rather than `--headless --screenshot` for two reasons: --screenshot only captures the
--window-size viewport and new headless clamps very tall sizes (the old headless that honoured
them was removed in Chrome 132), and CDP lets us pick the CSS viewport width, which is what
actually decides whether the text is readable once the page is scaled into a video panel.
"""
import base64
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

LOCAL = os.environ.get("LOCALAPPDATA", "")
HOME = os.path.expanduser("~")

# The page is captured at a phone-width CSS viewport and 3x device pixels: legibility is decided
# by the CSS width, not the pixel width. A 1280-wide desktop capture scaled into a ~340px panel
# puts body text at ~5 output pixels; at 360 CSS px it lands near 1:1. Both github.com and
# huggingface.co have real responsive layouts at this width.
CSS_WIDTH = 360
SCALE = 3
CSS_HEIGHT = 900             # viewport height; captureBeyondViewport grabs the whole page anyway
MAX_RATIO = 8.0              # cap the capture at 8x its width - nothing below that ever gets shown
LAUNCH_TIMEOUT = 25          # seconds to wait for DevToolsActivePort
NAV_TIMEOUT = 30             # seconds to wait for the load event
SETTLE = 1.5                 # seconds after load, for webfonts and late layout

# Where the interesting part of the page starts. GitHub puts the branch picker and the whole file
# tree above the README, so a scroll that begins at y=0 spends its time on directory listings.
_CONTENT_JS = """(() => {
  const sels = ['#readme', 'article.markdown-body', '.markdown-body',
                '[data-target="model-card.content"]', '.model-card-content',
                'main article', 'main'];
  for (const s of sels) {
    const el = document.querySelector(s);
    if (el) {
      const r = el.getBoundingClientRect();
      if (r.height > 200) return Math.max(0, r.top + window.scrollY - 24);
    }
  }
  return 0;
})()"""

_WIN = [
    LOCAL + r"\ms-playwright\chromium_headless_shell-*\chrome-headless-shell-win64\chrome-headless-shell.exe",
    LOCAL + r"\ms-playwright\chromium-*\chrome-win\chrome.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    LOCAL + r"\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
_NIX = [
    HOME + "/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell",
    HOME + "/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable", "/opt/google/chrome/chrome",
    "/usr/bin/microsoft-edge", "/snap/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
_PATH_NAMES = ("chrome-headless-shell", "chromium", "chromium-browser", "google-chrome",
               "google-chrome-stable", "chrome", "msedge", "microsoft-edge")


def _newest(pattern: str) -> str:
    """Highest-numbered match of a versioned glob (the ms-playwright build directories)."""
    hits = [p for p in glob.glob(pattern) if os.path.exists(p)]
    if not hits:
        return ""
    return sorted(hits, key=lambda p: max([int(n) for n in re.findall(r"-(\d+)", p)] or [0]))[-1]


@lru_cache(maxsize=1)
def _resolve() -> str:
    env = os.environ.get("CHROME_BIN") or os.environ.get("BROWSER_BIN")
    if env and Path(env).exists():
        return env
    for pat in (_WIN if sys.platform == "win32" else _NIX):
        hit = _newest(pat) if "*" in pat else (pat if os.path.exists(pat) else "")
        if hit:
            return hit
    for name in _PATH_NAMES:
        hit = shutil.which(name)
        if hit:
            return hit
    return ""


def resolve() -> str:
    """Path of the browser that will be used, or "" when none is installed."""
    return _resolve()


def available() -> bool:
    return bool(_resolve())


# ---- devtools protocol -----------------------------------------------------
class _CDP:
    """A minimal DevTools client: one websocket, synchronous request/response by id."""

    def __init__(self, ws_url: str, timeout: float = NAV_TIMEOUT):
        from websocket import create_connection
        self.ws = create_connection(ws_url, timeout=timeout, max_size=None)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        deadline = time.time() + NAV_TIMEOUT
        while time.time() < deadline:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error'].get('message', msg['error'])}")
                return msg.get("result", {})
        raise RuntimeError(f"{method}: no reply within {NAV_TIMEOUT}s")

    def wait_event(self, name: str, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:  # noqa: BLE001 - a socket timeout just means "not yet"
                break
            if msg.get("method") == name:
                return True
        return False

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def _port_from(profile: Path, proc: subprocess.Popen) -> int:
    """Chrome writes the port it chose into DevToolsActivePort once it is listening."""
    f = profile / "DevToolsActivePort"
    deadline = time.time() + LAUNCH_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode("utf-8", "replace")[-400:] if proc.stderr else ""
            raise RuntimeError(f"browser exited with code {proc.returncode} {err}".strip())
        if f.exists():
            try:
                first = f.read_text(encoding="utf-8").splitlines()[0].strip()
                if first.isdigit():
                    return int(first)
            except (OSError, IndexError):
                pass
        time.sleep(0.1)
    raise RuntimeError(f"browser did not start listening within {LAUNCH_TIMEOUT}s")


def _page_socket(port: int) -> str:
    """The websocket URL of a page target to drive."""
    import urllib.request
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
                for t in json.loads(r.read()):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        return t["webSocketDebuggerUrl"]
        except (OSError, ValueError, socket.timeout):
            pass
        time.sleep(0.2)
    raise RuntimeError("could not reach the browser's DevTools endpoint")


def _flags(profile: str) -> list:
    return [
        "--headless=new",
        # Chrome's process singleton is keyed on the profile: WITHOUT this, launching while the
        # user has Chrome open just forwards the command line to that instance, opens a tab and
        # exits 0 having written nothing.
        f"--user-data-dir={profile}",
        "--remote-debugging-port=0",
        # Chrome 111+ rejects a websocket whose Origin it does not know; ours is 127.0.0.1:<port>
        "--remote-allow-origins=*",
        "--disable-gpu", "--hide-scrollbars", "--mute-audio",
        "--no-sandbox", "--disable-dev-shm-usage",     # required as root in a container
        "--no-first-run", "--no-default-browser-check", "--no-pings",
        "--disable-extensions", "--disable-sync", "--disable-background-networking",
        "--disable-component-update", "--disable-client-side-phishing-detection",
        "--disable-default-apps", "--metrics-recording-only",
        "--disable-features=DialMediaRouteProvider,Translate,BackForwardCache,AcceptCHFrame",
        "about:blank",
    ]


def screenshot(url: str, out_png, css_width: int = CSS_WIDTH, scale: int = SCALE) -> tuple:
    """Full-page screenshot of `url` at a phone-width viewport.

    Returns (width, height, content_y): the pixel size, and the y offset where the README /
    model card begins, so the scroll can skip the nav and file tree above it.
    Raises RuntimeError on any failure; every caller treats that as non-fatal and continues
    without the overlay."""
    binary = _resolve()
    if not binary:
        raise RuntimeError("no Chrome/Chromium binary found - install chromium or set CHROME_BIN "
                           "to a chrome, chromium or chrome-headless-shell executable")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.unlink(missing_ok=True)

    profile = tempfile.mkdtemp(prefix="videogen-chrome-")
    proc, cdp, content_y = None, None, 0
    try:
        proc = subprocess.Popen([binary, *_flags(profile)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        port = _port_from(Path(profile), proc)
        cdp = _CDP(_page_socket(port))
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=css_width, height=CSS_HEIGHT,
                 deviceScaleFactor=scale, mobile=True)
        cdp.call("Page.navigate", url=url)
        cdp.wait_event("Page.loadEventFired", NAV_TIMEOUT)
        time.sleep(SETTLE)
        try:
            res = cdp.call("Runtime.evaluate", expression=_CONTENT_JS, returnByValue=True)
            content_y = int(float(res.get("result", {}).get("value") or 0) * scale)
        except Exception:  # noqa: BLE001 - an odd page just scrolls from the top
            content_y = 0
        shot = cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        data = base64.b64decode(shot.get("data") or "")
        if len(data) < 5000:
            raise RuntimeError("the browser returned an empty image")
        out_png.write_bytes(data)
    finally:
        if cdp:
            cdp.close()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Windows holds the profile lock briefly after exit; cleanup must never fail the job
        shutil.rmtree(profile, ignore_errors=True)

    w, h = _cap(out_png)
    return w, h, min(content_y, max(0, h - 200))


def _cap(png: Path) -> tuple:
    """Cap an extremely tall capture. Returns the final (width, height)."""
    from PIL import Image
    with Image.open(png) as img:
        w, h = img.width, img.height
        limit = int(w * MAX_RATIO)
        if h <= limit:
            return w, h
        img.crop((0, 0, w, limit)).save(png, "PNG", optimize=True)
        return w, limit
