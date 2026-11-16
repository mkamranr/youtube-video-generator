"""Reel post copy per platform, its edits and the bundle text, and the reel-aware cover copy.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-social-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from services import llm, reel  # noqa: E402

CHANNEL = {"name": "Daily Tech Talks", "audience": "general", "video_type": "tech", "cast": [],
           "characters_enabled": False, "reel_format": True}
REEL = {"hooks": [{"text": "Run a huge model on one GPU?"}], "chosen": 0,
        "central_message": "A memory hierarchy fits a big model on one GPU.",
        "audience_benefit": "Try bigger local models.", "cta": {"type": "save", "text": "Save this."}}


class Writer(unittest.TestCase):

    def test_three_platforms_and_clean_hashtags(self):
        seen = {}

        def fake(messages, max_tokens=8000):
            seen["user"] = messages[1]["content"]
            return {"instagram": {"caption": "Run big models at home.", "hashtags": ["LocalLLM", "#AI Tools", "x"]},
                    "facebook": {"caption": "Plain words."},
                    "youtube": {"title": "Big model, one GPU", "description": "Context. #LocalLLM"}}

        with mock.patch.object(llm, "_chat", fake):
            out = llm.write_social_captions("t", "narration text", REEL, channel=CHANNEL)
        self.assertEqual(out["instagram"]["hashtags"], ["#localllm", "#aitools", "#x"])
        self.assertEqual(out["facebook"]["caption"], "Plain words.")
        self.assertIn("#localllm", out["youtube"]["description"])
        self.assertIn("Save this.", seen["user"])
        self.assertIn("never add features", seen["user"])

    def test_cover_copy_sells_the_hook(self):
        seen = {}

        def fake(messages, max_tokens=8000):
            seen["user"] = messages[1]["content"]
            return {"thumbnail_prompt": "a GPU", "thumbnail_texts": ["ONE GPU?"]}

        with mock.patch.object(llm, "_chat", fake):
            llm.write_thumbnail("t", "n", "s", channel=CHANNEL, reel=REEL)
        self.assertIn("Run a huge model on one GPU?", seen["user"])
        self.assertIn("ask it as a question", seen["user"])
        with mock.patch.object(llm, "_chat", fake):
            llm.write_thumbnail("t", "n", "s", channel=CHANNEL)
        self.assertNotIn("This is a REEL", seen["user"])


class Edits(unittest.TestCase):

    def test_edit_and_bundle_text(self):
        mgr = pipeline.JobManager()
        job = pipeline.Job(id="so1", url="", source="topic", channel=dict(CHANNEL), status="done")
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / "plan.json").write_text(json.dumps(
            {"social": {"instagram": {"caption": "old", "hashtags": ["#a"]}}}), encoding="utf-8")
        mgr.jobs[job.id] = job
        mgr.update_social(job.id, {"instagram": {"caption": "new caption", "hashtags": "#ai localllm"},
                                   "youtube": {"title": "T", "description": "D"}})
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["social"]["instagram"], {"caption": "new caption", "hashtags": ["#ai", "#localllm"]})
        text = reel.social_text(plan["social"])
        self.assertIn("==== Instagram ====", text)
        self.assertIn("new caption", text)
        self.assertIn("Title: T", text)
        self.assertEqual(reel.social_text(None), "")


if __name__ == "__main__":
    unittest.main()
