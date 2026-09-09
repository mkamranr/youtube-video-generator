"""Session cookies and request-level auth helpers.

Sessions are stateless: the cookie carries a base64 JSON payload {uid, exp, ps} signed with
HMAC-SHA256. `ps` is a fingerprint of the user's password hash, so changing the password
invalidates every existing session. The signing key comes from VIDEOGEN_SECRET_KEY or a
random value generated once into users/session_secret.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

from fastapi import HTTPException, Request

import config
from services import users

COOKIE = "vg_session"
LOGIN_MAX_FAILURES = 5        # per client IP + username ...
LOGIN_WINDOW = 600            # ... within this many seconds ...
LOGIN_LOCK = 60               # ... locks the pair out for this long

_secret_cache = None
_fail_lock = threading.Lock()
_failures: dict = {}          # (ip, username) -> [timestamps]


# ---- signing key ---------------------------------------------------------------
def secret() -> bytes:
    global _secret_cache
    if _secret_cache is not None:
        return _secret_cache
    env = os.environ.get("VIDEOGEN_SECRET_KEY", "").strip()
    if env:
        _secret_cache = hashlib.sha256(env.encode("utf-8")).digest()
        return _secret_cache
    f = config.SESSION_SECRET_FILE
    try:
        key = bytes.fromhex(f.read_text(encoding="utf-8").strip())
        if len(key) >= 32:
            _secret_cache = key
            return key
    except Exception:  # noqa: BLE001
        pass
    key = secrets.token_bytes(32)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(key.hex(), encoding="utf-8")
    _secret_cache = key
    return key


# ---- tokens ----------------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str) -> str:
    return _b64(hmac.new(secret(), payload.encode("ascii"), hashlib.sha256).digest())


def _stamp(user: dict) -> str:
    return hashlib.sha256(user.get("pw_hash", "").encode("utf-8")).hexdigest()[:12]


def make_token(user: dict) -> str:
    payload = _b64(json.dumps({"uid": user["id"], "exp": int(time.time()) + config.SESSION_DAYS * 86400,
                               "ps": _stamp(user)}, separators=(",", ":")).encode("ascii"))
    return f"{payload}.{_sign(payload)}"


def read_token(token: str):
    """The full user record for a valid, unexpired token whose password is unchanged; else None."""
    try:
        payload, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        data = json.loads(_unb64(payload))
        if int(data.get("exp", 0)) < time.time():
            return None
        user = users.get_user(str(data.get("uid", "")))
        if user is None or not hmac.compare_digest(_stamp(user), str(data.get("ps", ""))):
            return None
        return user
    except Exception:  # noqa: BLE001
        return None


# ---- request helpers ---------------------------------------------------------------
def current_user(request: Request):
    """Cached per request: the signed-in user's full record or None."""
    if hasattr(request.state, "user"):
        return request.state.user
    token = request.cookies.get(COOKIE)
    user = read_token(token) if token else None
    request.state.user = user
    return user


def set_cookie(response, user: dict, request: Request):
    response.set_cookie(COOKIE, make_token(user), max_age=config.SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", path="/",
                        secure=request.url.scheme == "https")


def clear_cookie(response):
    response.delete_cookie(COOKIE, path="/")


def require_user(request: Request) -> dict:
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "not authenticated")
    return user


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if user.get("role") != "admin":
        raise HTTPException(403, "admin access required")
    return user


# ---- ownership ------------------------------------------------------------------------
def is_admin(user: dict | None) -> bool:
    return bool(user) and user.get("role") == "admin"


def can_access(user: dict | None, owner: str) -> bool:
    """Admins reach everything; users only records they own. A record without an owner
    (the window between a legacy install's first start and its admin being created) is
    admin-only."""
    if not user:
        return False
    return is_admin(user) or (bool(owner) and owner == user.get("id"))


# ---- login throttling -----------------------------------------------------------------
def _key(request: Request, username: str) -> tuple:
    ip = request.client.host if request.client else "?"
    return ip, users.normalize_username(username)


def locked_for(request: Request, username: str) -> int:
    """Seconds the client must still wait before another attempt (0 = allowed)."""
    now = time.time()
    with _fail_lock:
        stamps = [t for t in _failures.get(_key(request, username), []) if now - t < LOGIN_WINDOW]
        if len(stamps) >= LOGIN_MAX_FAILURES:
            wait = LOGIN_LOCK - (now - stamps[-1])
            if wait > 0:
                return int(wait) + 1
        return 0


def record_failure(request: Request, username: str):
    now = time.time()
    with _fail_lock:
        k = _key(request, username)
        _failures[k] = [t for t in _failures.get(k, []) if now - t < LOGIN_WINDOW] + [now]
        if len(_failures) > 5000:   # keep the table bounded
            _failures.clear()


def clear_failures(request: Request, username: str):
    with _fail_lock:
        _failures.pop(_key(request, username), None)
