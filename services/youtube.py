"""YouTube upload: installed-app OAuth (loopback redirect) + resumable video upload.

Requires the OAuth client secret JSON (Desktop-type client) saved as client_secret.json
in the project folder; the one client serves every channel - each channel picks its own
Google account on the consent screen and stores its own token file. Only the
youtube.upload scope is requested.
"""
import json
import threading
import time
from pathlib import Path

import config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_flows = {}          # state -> {redirect_uri, token_file, ts}; several connects may be pending
_lock = threading.Lock()
FLOW_TTL = 600       # seconds a pending consent flow stays valid


def has_secret(secret_file=None) -> bool:
    return Path(secret_file or config.YT_CLIENT_SECRET_FILE).exists()


def validate_secret(data: bytes) -> None:
    """Check that data looks like a Google OAuth client file; raise ValueError otherwise."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise ValueError("not a valid OAuth client file (not JSON)")
    client = (obj.get("installed") or obj.get("web")) if isinstance(obj, dict) else None
    if not isinstance(client, dict) or not client.get("client_id"):
        raise ValueError('not a valid OAuth client file (missing "installed"/"web" client_id)')


def is_connected(token_file=None) -> bool:
    return Path(token_file or config.YT_TOKEN_FILE).exists()


def auth_url(redirect_uri: str, token_file, secret_file=None) -> str:
    from google_auth_oauthlib.flow import Flow
    secret_file = str(secret_file or config.YT_CLIENT_SECRET_FILE)
    flow = Flow.from_client_secrets_file(secret_file, scopes=SCOPES,
                                         redirect_uri=redirect_uri)
    url, state = flow.authorization_url(access_type="offline", prompt="consent",
                                        include_granted_scopes="false")
    with _lock:
        now = time.time()
        for k in [k for k, v in _flows.items() if now - v["ts"] > FLOW_TTL]:
            del _flows[k]
        _flows[state] = {"redirect_uri": redirect_uri, "token_file": str(token_file),
                         "secret_file": secret_file, "ts": now}
    return url


def finish_auth(code: str, state: str) -> None:
    from google_auth_oauthlib.flow import Flow
    with _lock:
        pending = _flows.pop(state, None)
    if not pending:
        raise RuntimeError("no authorization in progress - start again from Settings")
    flow = Flow.from_client_secrets_file(pending.get("secret_file") or str(config.YT_CLIENT_SECRET_FILE),
                                         scopes=SCOPES,
                                         redirect_uri=pending["redirect_uri"])
    flow.fetch_token(code=code)
    token_file = Path(pending["token_file"])
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(flow.credentials.to_json(), encoding="utf-8")


def disconnect(token_file=None):
    Path(token_file or config.YT_TOKEN_FILE).unlink(missing_ok=True)


def _credentials(token_file=None):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    token_file = Path(token_file or config.YT_TOKEN_FILE)
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload(video_path, thumb_path, title: str, description: str,
           privacy: str, made_for_kids: bool, on_progress=None, token_file=None):
    """Resumable upload; returns (video_id, thumbnail_error_or_empty)."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    yt = build("youtube", "v3", credentials=_credentials(token_file), cache_discovery=False)
    body = {
        "snippet": {
            "title": (title or "Untitled")[:100],
            "description": (description or "")[:4900],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": bool(made_for_kids),
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024,
                            resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status and on_progress:
            on_progress(int(status.progress() * 100))
    video_id = resp["id"]
    thumb_error = ""
    if thumb_path and Path(thumb_path).exists():
        try:
            yt.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumb_path), mimetype="image/jpeg"),
            ).execute()
        except Exception as e:  # noqa: BLE001 - custom thumbnails may need channel verification
            thumb_error = str(e)
    return video_id, thumb_error
