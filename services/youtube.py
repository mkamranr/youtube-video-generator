"""YouTube upload: installed-app OAuth (loopback redirect) + resumable video upload.

Requires the OAuth client secret JSON (Desktop-type client) saved as client_secret.json
in the project folder. Only the youtube.upload scope is requested.
"""
import threading
from pathlib import Path

import config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_flow_state = {}
_lock = threading.Lock()


def has_secret() -> bool:
    return config.YT_CLIENT_SECRET_FILE.exists()


def is_connected() -> bool:
    return config.YT_TOKEN_FILE.exists()


def auth_url(redirect_uri: str) -> str:
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(str(config.YT_CLIENT_SECRET_FILE), scopes=SCOPES,
                                         redirect_uri=redirect_uri)
    url, state = flow.authorization_url(access_type="offline", prompt="consent",
                                        include_granted_scopes="false")
    with _lock:
        _flow_state["state"] = state
        _flow_state["redirect_uri"] = redirect_uri
    return url


def finish_auth(code: str) -> None:
    from google_auth_oauthlib.flow import Flow
    with _lock:
        redirect_uri = _flow_state.get("redirect_uri")
    if not redirect_uri:
        raise RuntimeError("no authorization in progress - start again from Settings")
    flow = Flow.from_client_secrets_file(str(config.YT_CLIENT_SECRET_FILE), scopes=SCOPES,
                                         redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    config.YT_TOKEN_FILE.write_text(flow.credentials.to_json(), encoding="utf-8")


def disconnect():
    config.YT_TOKEN_FILE.unlink(missing_ok=True)


def _credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials.from_authorized_user_file(str(config.YT_TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        config.YT_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload(video_path, thumb_path, title: str, description: str,
           privacy: str, made_for_kids: bool, on_progress=None):
    """Resumable upload; returns (video_id, thumbnail_error_or_empty)."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    yt = build("youtube", "v3", credentials=_credentials(), cache_discovery=False)
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
