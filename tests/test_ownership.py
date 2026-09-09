"""User-scoped channels and jobs: run with  python -m unittest discover -s tests -t .  from the repo root.

Route-level tests through FastAPI's TestClient against a throw-away data directory. `config`
paths are patched before `app` is imported (every store reads config.X at call time)."""
import importlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-own-"))
LEGACY_CID = "c0ffee01"
LEGACY_JOB = "20260101-000000-abcdef"

app = None          # the imported module
clients = {}        # username -> TestClient with its own cookie jar
ids = {}            # username -> user id


def _patch_config():
    config.CHANNELS_DIR = TMP / "channels"
    config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"
    config.OUTPUT_DIR = TMP / "output"
    config.USERS_DIR = TMP / "users"
    config.USERS_INDEX = config.USERS_DIR / "users.json"
    config.SESSION_SECRET_FILE = config.USERS_DIR / "session_secret"
    config.SETTINGS_FILE = TMP / "settings.json"
    config.LEGACY_SETTINGS_FILE = TMP / "legacy_settings.json"
    config.YT_CLIENT_SECRET_FILE = TMP / "client_secret.json"
    config.YT_TOKEN_FILE = TMP / "youtube_token.json"
    config.LOGO_FILE = TMP / "channel_logo.png"
    config.VOICES_DIR = TMP / "reference_audio"
    config.VOICES_INDEX = config.VOICES_DIR / "voices.json"
    config.COMFY_CLEANUP = False          # no ComfyUI sweep thread at JobManager start
    os.environ["VIDEOGEN_SECRET_KEY"] = "test-secret"


def _seed_legacy_data():
    """A channel and a finished job written before ownership existed (no `owner`)."""
    config.CHANNELS_DIR.mkdir(parents=True)
    config.CHANNELS_INDEX.write_text(json.dumps({"channels": [{
        "id": LEGACY_CID, "name": "Legacy Channel", "purpose": "", "audience": "kids",
        "video_type": "animated", "made_for_kids": True, "cast": [], "cast_opening": True,
        "created": time.time(), "defaults": {"art_style": "", "audio_mode": "music", "voice_id": "",
                                             "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                                             "video_length": config.DEFAULT_VIDEO_LENGTH, "storyboard": False}}]}))
    jd = config.OUTPUT_DIR / LEGACY_JOB
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({"id": LEGACY_JOB, "url": "", "source": "topic", "topic": "old",
                                             "channel_id": LEGACY_CID, "channel": {"name": "Legacy Channel"},
                                             "status": "done", "title": "Old video",
                                             "created": time.time(), "updated": time.time()}))


def _login(username: str, password: str):
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    ids[username] = r.json()["user"]["id"]
    clients[username] = c
    return c


def setUpModule():
    global app
    _patch_config()
    _seed_legacy_data()
    app = importlib.import_module("app")
    # nobody exists yet -> the startup migration must have left the records unowned
    assert app.channels.get_channel(LEGACY_CID)["owner"] == ""
    assert app.manager.jobs[LEGACY_JOB].owner == ""
    from fastapi.testclient import TestClient
    c = TestClient(app.app)
    r = c.post("/api/auth/setup", json={"username": "admin1", "password": "password1"})
    assert r.status_code == 200, r.text
    ids["admin1"] = r.json()["user"]["id"]
    clients["admin1"] = c
    for name, role in (("alice", "user"), ("bob", "user"), ("admin2", "admin")):
        r = c.post("/api/users", json={"username": name, "password": "password1", "role": role})
        assert r.status_code == 201, r.text
        _login(name, "password1")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def _new_channel(client, name):
    r = client.post("/api/channels", json={"name": name, "audience": "kids", "video_type": "animated",
                                            "owner": "someone-else"})
    assert r.status_code == 200, r.text
    return r.json()


class Migration(unittest.TestCase):
    def test_setup_assigned_legacy_records_to_first_admin(self):
        self.assertEqual(app.channels.get_channel(LEGACY_CID)["owner"], ids["admin1"])
        self.assertEqual(app.manager.jobs[LEGACY_JOB].owner, ids["admin1"])
        saved = json.loads((config.OUTPUT_DIR / LEGACY_JOB / "job.json").read_text())
        self.assertEqual(saved["owner"], ids["admin1"])
        self.assertEqual(app.ownership.oldest_admin_id(), ids["admin1"])

    def test_migration_is_noop_without_admins(self):
        with mock.patch.object(app.users, "list_users", return_value=[]):
            self.assertEqual(app.ownership.migrate_unowned([])["skipped"], True)

    def test_legacy_seed_does_not_rerun_when_index_exists(self):
        self.assertIsNone(app.channels.ensure_default_channel())


class Channels(unittest.TestCase):
    def test_owner_comes_from_the_caller_not_the_body(self):
        ch = _new_channel(clients["alice"], "Alice TV")
        self.assertEqual(ch["owner"], ids["alice"])
        self.assertEqual(ch["owner_name"], "alice")
        r = clients["alice"].put(f"/api/channels/{ch['id']}", json={"name": "Alice TV 2", "owner": ids["bob"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["owner"], ids["alice"])
        clients["alice"].delete(f"/api/channels/{ch['id']}")

    def test_other_users_get_404_admins_get_through(self):
        ch = _new_channel(clients["alice"], "Alice Private")
        cid = ch["id"]
        bob = clients["bob"]
        for method, path in (("get", f"/api/channels/{cid}/logo"),
                             ("put", f"/api/channels/{cid}"),
                             ("delete", f"/api/channels/{cid}"),
                             ("delete", f"/api/channels/{cid}/logo"),
                             ("delete", f"/api/channels/{cid}/secret"),
                             ("delete", f"/api/channels/{cid}/youtube/auth"),
                             ("get", f"/api/channels/{cid}/cast/xx/portrait/candidates"),
                             ("get", f"/api/channels/{cid}/youtube/auth")):
            kwargs = {"json": {"name": "x"}} if method == "put" else {}
            r = getattr(bob, method)(path, **kwargs)
            self.assertEqual(r.status_code, 404, f"{method} {path}: {r.status_code} {r.text}")
        r = clients["admin2"].put(f"/api/channels/{cid}", json={"name": "Renamed by admin"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["owner"], ids["alice"])
        self.assertEqual(clients["alice"].delete(f"/api/channels/{cid}").status_code, 200)

    def test_listings_are_scoped(self):
        ch = _new_channel(clients["alice"], "Alice List")
        try:
            mine = clients["alice"].get("/api/channels").json()
            self.assertEqual([c["id"] for c in mine["channels"]], [ch["id"]])
            self.assertEqual(mine["me"]["role"], "user")
            opts = clients["alice"].get("/options").json()
            self.assertEqual(opts["default_channel_id"], ch["id"])
            self.assertEqual(opts["me"]["username"], "alice")
            self.assertEqual(clients["bob"].get("/options").json()["default_channel_id"], "")
            self.assertEqual(clients["bob"].get("/api/channels").json()["channels"], [])
            everything = clients["admin1"].get("/api/channels").json()["channels"]
            self.assertIn(ch["id"], [c["id"] for c in everything])
            self.assertIn(LEGACY_CID, [c["id"] for c in everything])
            names = {c["id"]: c["owner_name"] for c in everything}
            self.assertEqual(names[ch["id"]], "alice")
            self.assertEqual(names[LEGACY_CID], "admin1")
        finally:
            clients["alice"].delete(f"/api/channels/{ch['id']}")

    def test_a_users_only_channel_can_be_deleted(self):
        ch = _new_channel(clients["bob"], "Bob Only")
        self.assertEqual(clients["bob"].delete(f"/api/channels/{ch['id']}").status_code, 200)
        self.assertEqual(clients["bob"].get("/api/channels").json()["channels"], [])


class Jobs(unittest.TestCase):
    def test_create_requires_an_owned_channel(self):
        r = clients["alice"].post("/jobs", json={"topic": "bees and honey"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("create a channel first", r.json()["detail"])
        r = clients["alice"].post("/jobs", json={"topic": "bees and honey", "channel_id": LEGACY_CID})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "unknown channel")

    def test_jobs_inherit_the_channel_owner_and_are_scoped(self):
        ch = _new_channel(clients["alice"], "Alice Jobs")
        with mock.patch.object(app.JobManager, "_start_draft"):   # no LLM thread
            r = clients["alice"].post("/jobs", json={"topic": "the little fox", "channel_id": ch["id"]})
            self.assertEqual(r.status_code, 200, r.text)
            job = r.json()[0]
            self.assertEqual(job["owner"], ids["alice"])
            self.assertEqual(job["owner_name"], "alice")
            # an admin creating on alice's channel still makes alice's video
            r = clients["admin1"].post("/jobs", json={"topic": "by admin", "channel_id": ch["id"]})
            self.assertEqual(r.json()[0]["owner"], ids["alice"])
            admin_job = r.json()[0]["id"]
        try:
            jid = job["id"]
            bob = clients["bob"]
            self.assertNotIn(jid, [j["id"] for j in bob.get("/jobs").json()])
            for method, path in (("get", f"/jobs/{jid}"), ("get", f"/jobs/{jid}/details"),
                                 ("get", f"/jobs/{jid}/files/plan.json"),
                                 ("get", f"/jobs/{jid}/download/video"),
                                 ("post", f"/jobs/{jid}/cancel"), ("post", f"/jobs/{jid}/archive"),
                                 ("post", f"/jobs/{jid}/resume"), ("delete", f"/jobs/{jid}")):
                r = getattr(bob, method)(path)
                self.assertEqual(r.status_code, 404, f"{method} {path}: {r.status_code} {r.text}")
            mine = [j["id"] for j in clients["alice"].get("/jobs").json()]
            self.assertIn(jid, mine)
            self.assertNotIn(LEGACY_JOB, mine)
            theirs = {j["id"]: j["owner_name"] for j in clients["admin2"].get("/jobs").json()}
            self.assertEqual(theirs[jid], "alice")
            self.assertEqual(theirs[LEGACY_JOB], "admin1")
            self.assertEqual(clients["alice"].get(f"/jobs/{jid}").status_code, 200)
        finally:
            clients["alice"].delete(f"/jobs/{jid}")
            clients["alice"].delete(f"/jobs/{admin_job}")
            clients["alice"].delete(f"/api/channels/{ch['id']}")


class AdminOnly(unittest.TestCase):
    def test_settings_and_shared_resources_need_admin(self):
        alice = clients["alice"]
        self.assertEqual(alice.get("/api/settings").status_code, 403)
        self.assertEqual(alice.put("/api/settings", json={"CLIP_SECONDS": 5}).status_code, 403)
        self.assertEqual(alice.get("/api/health").status_code, 403)
        self.assertEqual(alice.delete("/api/youtube/secret").status_code, 403)
        self.assertEqual(alice.post("/api/style-previews").status_code, 403)
        self.assertEqual(alice.delete("/api/voices/nope").status_code, 403)
        self.assertEqual(alice.post("/api/voices", files={"file": ("a.wav", b"x")},
                                    data={"name": "n", "text": "t"}).status_code, 403)
        self.assertEqual(alice.get("/api/users").status_code, 403)

    def test_shared_reads_stay_open_to_users(self):
        alice = clients["alice"]
        self.assertEqual(alice.get("/api/schedule").status_code, 200)
        self.assertEqual(alice.get("/api/voices").status_code, 200)
        self.assertEqual(alice.get("/api/style-previews").status_code, 200)
        self.assertEqual(clients["admin1"].get("/api/settings").status_code, 200)


class UserDeletion(unittest.TestCase):
    def test_deleted_users_records_move_to_the_acting_admin(self):
        c = clients["admin1"]
        r = c.post("/api/users", json={"username": "carol", "password": "password1", "role": "user"})
        self.assertEqual(r.status_code, 201)
        carol = _login("carol", "password1")
        ch = _new_channel(carol, "Carol TV")
        r = clients["admin2"].delete(f"/api/users/{ids['carol']}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["reassigned"]["channels"], 1)
        self.assertEqual(app.channels.get_channel(ch["id"])["owner"], ids["admin2"])
        self.assertEqual(clients["admin2"].delete(f"/api/channels/{ch['id']}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
