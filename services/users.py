"""User accounts: username, role (admin | user) and a scrypt password hash.

Index: users/users.json (never committed). The first account is created by the one-time
setup screen and is always an admin; admins add further users on the Settings page.
Passwords are hashed with hashlib.scrypt (stdlib, no extra dependency).
"""
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import uuid

import config

_lock = threading.Lock()

ROLES = ("admin", "user")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
MIN_PASSWORD_CHARS = 8
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


# ---- store -------------------------------------------------------------------
def _read_index() -> dict:
    try:
        data = json.loads(config.USERS_INDEX.read_text(encoding="utf-8"))
        if isinstance(data.get("users"), list):
            return data
    except Exception:  # noqa: BLE001  (missing or corrupt index -> empty store)
        pass
    return {"users": []}


def _write_index(data: dict):
    config.USERS_DIR.mkdir(parents=True, exist_ok=True)
    config.USERS_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")


def public(rec: dict) -> dict:
    """The user record without secrets (what the API returns)."""
    return {k: rec[k] for k in ("id", "username", "role", "created") if k in rec}


def list_users() -> list:
    return [public(u) for u in _read_index()["users"]]


def get_user(uid: str):
    for u in _read_index()["users"]:
        if u["id"] == uid:
            return u
    return None


def get_by_username(username: str):
    name = normalize_username(username)
    for u in _read_index()["users"]:
        if u["username"] == name:
            return u
    return None


def has_users() -> bool:
    return bool(_read_index()["users"])


# ---- validation / hashing -------------------------------------------------------
def normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def validate_username(username: str) -> str:
    name = normalize_username(username)
    if not USERNAME_RE.match(name):
        raise ValueError("username must be 3-40 characters: letters, digits, dot, dash or underscore, "
                         "starting with a letter or digit")
    return name


def validate_password(password: str) -> str:
    pw = str(password or "")
    if len(pw) < MIN_PASSWORD_CHARS:
        raise ValueError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(pw) > 200:
        raise ValueError("password is too long")
    return pw


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def check_password(password: str, pw_hash: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = pw_hash.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(str(password or "").encode("utf-8"), salt=bytes.fromhex(salt_hex), **_SCRYPT)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:  # noqa: BLE001
        return False


def verify_password(username: str, password: str):
    """The full user record when the credentials match, else None (constant-ish time)."""
    u = get_by_username(username)
    if u is None:  # burn the same hashing cost so a missing user is not detectable by timing
        check_password(password, "scrypt$00$00")
        return None
    return u if check_password(password, u["pw_hash"]) else None


# ---- mutations -------------------------------------------------------------------
def add_user(username: str, password: str, role: str = "user") -> dict:
    name = validate_username(username)
    pw = validate_password(password)
    if role not in ROLES:
        raise ValueError(f"role must be one of {list(ROLES)}")
    rec = {"id": uuid.uuid4().hex[:8], "username": name, "role": role,
           "pw_hash": hash_password(pw), "created": time.time()}
    with _lock:
        idx = _read_index()
        if any(u["username"] == name for u in idx["users"]):
            raise ValueError(f"a user named {name!r} already exists")
        idx["users"].append(rec)
        _write_index(idx)
    return public(rec)


def set_password(uid: str, password: str) -> dict:
    pw = validate_password(password)
    with _lock:
        idx = _read_index()
        for u in idx["users"]:
            if u["id"] == uid:
                u["pw_hash"] = hash_password(pw)
                _write_index(idx)
                return public(u)
    raise KeyError(uid)


def set_role(uid: str, role: str) -> dict:
    if role not in ROLES:
        raise ValueError(f"role must be one of {list(ROLES)}")
    with _lock:
        idx = _read_index()
        target = next((u for u in idx["users"] if u["id"] == uid), None)
        if target is None:
            raise KeyError(uid)
        admins = [u for u in idx["users"] if u["role"] == "admin"]
        if target["role"] == "admin" and role != "admin" and len(admins) <= 1:
            raise ValueError("cannot demote the last admin - make another user admin first")
        target["role"] = role
        _write_index(idx)
        return public(target)


def delete_user(uid: str, acting_uid: str | None = None):
    with _lock:
        idx = _read_index()
        target = next((u for u in idx["users"] if u["id"] == uid), None)
        if target is None:
            raise KeyError(uid)
        if acting_uid is not None and acting_uid == uid:
            raise ValueError("you cannot delete your own account")
        admins = [u for u in idx["users"] if u["role"] == "admin"]
        if target["role"] == "admin" and len(admins) <= 1:
            raise ValueError("cannot delete the last admin - make another user admin first")
        idx["users"] = [u for u in idx["users"] if u["id"] != uid]
        _write_index(idx)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    config.USERS_DIR = Path(tempfile.mkdtemp())
    config.USERS_INDEX = config.USERS_DIR / "users.json"
    print("empty:", has_users())
    a = add_user("Admin", "correct horse battery", "admin")
    print("added", a, "verify ok:", bool(verify_password("admin", "correct horse battery")),
          "verify bad:", verify_password("admin", "nope"))
    try:
        delete_user(a["id"])
    except ValueError as e:
        print("guard:", e)
    b = add_user("bob", "password123")
    set_role(b["id"], "admin")
    delete_user(a["id"])
    print("remaining:", list_users())
