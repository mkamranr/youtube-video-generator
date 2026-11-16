"""Reel format: scored hook candidates, a scene-by-scene script and one CTA.

services/reel.py decides which hook wins and whether the script is fit to speak; the pipeline
stores the structured script in plan["reel"] and speaks plan["narration"], the scenes joined.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-reel-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import channels, llm, reel  # noqa: E402

CHANNEL = {"id": "ch01", "name": "Daily Tech Talks", "purpose": "Practical AI tools", "rules": "",
           "owner": "u1", "audience": "general", "video_type": "tech", "cast": [],
           "characters_enabled": False, "reel_format": True,
           "defaults": {"art_style": "", "audio_mode": "narration+music", "voice_id": "",
                        "aspect_ratio": "9:16 (Portrait Widescreen)", "video_length": "Auto",
                        "storyboard": False}}


def _scores(**kw):
    base = {k: 4 for k in reel.SCORE_KEYS}
    base.update(kw)
    return base


def _vo(n):
    return " ".join(f"word{n}_{i}" for i in range(12)) + "."


RAW = {
    "title": "Run a big model at home",
    "central_message": "A memory hierarchy lets one GPU serve a model that would not fit.",
    "audience_benefit": "Try larger local models without renting a cloud GPU.",
    "hooks": [
        {"text": "Run a huge AI model without renting a cloud GPU?", "category": "outcome",
         "scores": _scores(clarity=5, curiosity=5), "supported": True},
        {"text": "This revolutionary engine changes everything.", "category": "surprise",
         "scores": _scores(clarity=5, relevance=5, curiosity=5, specificity=5), "supported": True},
        {"text": "It runs completely offline on your laptop.", "category": "outcome",
         "scores": _scores(clarity=5, curiosity=5, specificity=5), "supported": False},
        {"text": "Local AI is expensive. This cuts the GPU memory it needs.",
         "category": "problem/solution", "scores": _scores(factual=3, clarity=5), "supported": True},
        {"text": "What happens when the model is bigger than your GPU?", "category": "curiosity",
         "scores": _scores(), "supported": True},
    ],
    "scenes": [
        {"role": "hook", "voiceover": "Run a huge AI model without renting a cloud GPU? " + _vo(1),
         "headline": "BIG MODEL, ONE GPU"},
        {"role": "context", "voiceover": _vo(2), "headline": "WHY IT MATTERS"},
        {"role": "demo", "voiceover": _vo(3), "headline": "THE REAL SETUP", "has_claim": True,
         "asset_needed": "screen recording of the model generating"},
        {"role": "payoff", "voiceover": _vo(4), "headline": "THE TRADE-OFF"},
        {"role": "cta", "voiceover": "Save this if you are experimenting with local AI.",
         "headline": "SAVE THIS"},
    ],
    "cta": {"type": "save", "text": "Save this if you are experimenting with local AI."},
    "style": "clean studio tech look",
}


class HookSelection(unittest.TestCase):

    def test_hype_unsupported_and_weak_factual_hooks_are_rejected(self):
        hooks = reel.normalize_hooks(copy.deepcopy(RAW["hooks"]))
        best = reel.select_hook(hooks)
        self.assertEqual(best, 0)
        self.assertIn("revolutionary", hooks[1]["rejected"])
        self.assertTrue(hooks[2]["rejected"])          # unsupported
        self.assertIn("factual", hooks[3]["rejected"])
        self.assertEqual(hooks[3]["category"], "problem_solution")
        self.assertEqual(hooks[4]["rejected"], "")

    def test_unsupported_claim_words_need_support(self):
        self.assertTrue(reel.hype_reason("It is free and offline", supported=False))
        self.assertEqual(reel.hype_reason("It is free and offline", supported=True), "")

    def test_none_left_gives_minus_one(self):
        hooks = reel.normalize_hooks([{"text": "Unlimited power", "scores": _scores()}])
        self.assertEqual(reel.select_hook(hooks), -1)


class Validation(unittest.TestCase):

    def _reel(self):
        return reel.build(copy.deepcopy(RAW))

    def test_good_script_passes(self):
        r = self._reel()
        words = len(reel.narration(r).split())
        self.assertEqual(reel.validate(r, words), [])

    def test_banned_opener(self):
        r = self._reel()
        r["scenes"][0]["voiceover"] = "Today we're going to look at a new engine. " + _vo(9)
        self.assertTrue(any("introduction" in i for i in reel.validate(r, None)))

    def test_missing_demo_and_two_ctas(self):
        r = self._reel()
        r["scenes"][2]["role"] = "cta"
        issues = reel.validate(r, None)
        self.assertTrue(any("demonstration" in i for i in issues))
        self.assertTrue(any("exactly one call to action" in i for i in issues))

    def test_cta_with_two_asks(self):
        r = self._reel()
        r["scenes"][-1]["voiceover"] = "Save this guide. Check the repository for the flags."
        self.assertTrue(any("several things" in i for i in reel.validate(r, None)))

    def test_too_long(self):
        r = self._reel()
        words = len(reel.narration(r).split())
        self.assertTrue(any("too long" in i for i in reel.validate(r, int(words / 1.5))))

    def test_repeated_line(self):
        r = self._reel()
        r["scenes"][3]["voiceover"] = r["scenes"][2]["voiceover"]
        self.assertTrue(any("repeats" in i for i in reel.validate(r, None)))


class HookSwap(unittest.TestCase):

    def test_swap_replaces_the_opening_line_once(self):
        r = reel.build(copy.deepcopy(RAW))
        rest = _vo(1)
        reel.apply_edits(r, {"chosen": 4})
        self.assertEqual(r["scenes"][0]["voiceover"],
                         "What happens when the model is bigger than your GPU? " + rest)
        reel.apply_edits(r, {"chosen": 0})
        self.assertEqual(r["scenes"][0]["voiceover"],
                         "Run a huge AI model without renting a cloud GPU? " + rest)

    def test_writer_opening_with_another_candidate_is_not_doubled(self):
        raw = copy.deepcopy(RAW)
        raw["scenes"][0]["voiceover"] = "What happens when the model is bigger than your GPU? " + _vo(1)
        r = reel.build(raw)          # hook 0 wins the scoring
        self.assertEqual(r["scenes"][0]["voiceover"],
                         "Run a huge AI model without renting a cloud GPU? " + _vo(1))

    def test_page_edit_after_swap_is_final(self):
        r = reel.build(copy.deepcopy(RAW))
        reel.apply_edits(r, {"chosen": 4, "scenes": [{"n": 1, "voiceover": "My own opener."}],
                             "cta": {"text": "Share this with a developer."}})
        self.assertEqual(r["scenes"][0]["voiceover"], "My own opener.")
        self.assertEqual(r["cta"]["text"], "Share this with a developer.")


class ClipSceneMap(unittest.TestCase):

    def test_each_clip_gets_the_scene_that_fills_it(self):
        scenes = [{"voiceover": "a " * 10}, {"voiceover": "b " * 10}, {"voiceover": "c " * 10}]
        m = reel.clip_scene_map(scenes, [4.0, 4.0, 4.0], audio_seconds=12.0)
        self.assertEqual([s for s, _ in m], [0, 1, 2])
        self.assertEqual(m[0][1].split(), ["a"] * 10)

    def test_music_tail_stays_on_last_scene(self):
        scenes = [{"voiceover": "a " * 10}, {"voiceover": "b " * 10}]
        m = reel.clip_scene_map(scenes, [5.0, 5.0, 5.0], audio_seconds=10.0)
        self.assertEqual([s for s, _ in m], [0, 1, 1])
        self.assertEqual(m[2][1], "")


class ChannelSwitch(unittest.TestCase):

    def test_switch_round_trips_and_is_snapshotted(self):
        rec = channels._validate({"name": "Daily Tech Talks", "video_type": "tech", "reel_format": True})
        self.assertTrue(rec["reel_format"])
        self.assertTrue(channels.snapshot(rec)["reel_format"])
        rec2 = channels._validate({"name": "Other", "video_type": "tech"})
        self.assertFalse(channels.snapshot(rec2)["reel_format"])

    def test_reel_needs_vertical_spoken_drafted_video(self):
        job = pipeline.Job(id="x", url="", aspect_ratio="9:16 (Portrait Widescreen)",
                           audio_mode="narration+music", source="topic", channel=dict(CHANNEL))
        self.assertTrue(pipeline._reel(job))
        job.aspect_ratio = "16:9 (Landscape)"
        self.assertFalse(pipeline._reel(job))
        job.aspect_ratio = "9:16 (Portrait Widescreen)"
        job.source = "transcript"
        self.assertFalse(pipeline._reel(job))
        job.source = "topic"
        job.channel = {**CHANNEL, "reel_format": False}
        self.assertFalse(pipeline._reel(job))


class DraftAndApprove(unittest.TestCase):

    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        p = mock.patch.object(pipeline.JobManager, "_start_draft")
        p.start()
        self.addCleanup(p.stop)
        self.mgr = pipeline.JobManager()

    def _job(self, channel=CHANNEL):
        return self.mgr.submit_prompt("A hierarchical-memory inference engine", "topic",
                                      aspect_ratio="9:16 (Portrait Widescreen)",
                                      channel=channel, owner="u1")

    def test_draft_stores_reel_and_narration_is_the_scenes(self):
        job = self._job()
        words = len(reel.narration(reel.build(copy.deepcopy(RAW))).split())
        with mock.patch.object(pipeline, "_length_params", return_value=(words, None, None, None)), \
             mock.patch.object(llm, "write_reel_script", return_value=copy.deepcopy(RAW)) as w, \
             mock.patch.object(llm, "write_script_from_prompt") as old:
            self.mgr._draft_script(job)
        w.assert_called_once()
        old.assert_not_called()
        self.assertEqual(job.status, "awaiting_approval")
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["narration"], reel.narration(plan["reel"]))
        self.assertEqual(plan["reel"]["chosen"], 0)
        self.assertEqual(len(plan["reel"]["hooks"]), 5)
        self.assertEqual(plan["reel"]["issues"], [])

    def test_failed_check_asks_for_one_rewrite(self):
        job = self._job()
        raw = copy.deepcopy(RAW)
        raw["scenes"][2]["role"] = "context"    # no demo
        fixed = copy.deepcopy(RAW["scenes"])
        with mock.patch.object(pipeline, "_length_params", return_value=(None, None, None, None)), \
             mock.patch.object(llm, "write_reel_script", return_value=raw), \
             mock.patch.object(llm, "tighten_reel_script", return_value=fixed) as t:
            self.mgr._draft_script(job)
        t.assert_called_once()
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertIn("demo", [s["role"] for s in plan["reel"]["scenes"]])
        self.assertEqual(plan["reel"]["issues"], [])

    def test_approve_with_hook_swap_rebuilds_narration(self):
        job = self._job()
        with mock.patch.object(pipeline, "_length_params", return_value=(None, None, None, None)), \
             mock.patch.object(llm, "write_reel_script", return_value=copy.deepcopy(RAW)):
            self.mgr._draft_script(job)
        self.mgr.approve_script(job.id, narration="ignored free text", reel_edits={"chosen": 4})
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertTrue(plan["narration"].startswith("What happens when the model is bigger"))
        self.assertNotIn("ignored free text", plan["narration"])
        self.assertEqual(job.status, "queued")

    def test_non_reel_channel_keeps_the_paragraph_writer(self):
        job = self._job({**CHANNEL, "reel_format": False})
        answer = {"title": "t", "narration": "plain paragraph script", "style": "s", "cast_chosen": []}
        with mock.patch.object(pipeline, "_length_params", return_value=(None, None, None, None)), \
             mock.patch.object(llm, "write_script_from_prompt", return_value=answer), \
             mock.patch.object(llm, "write_reel_script") as w:
            self.mgr._draft_script(job)
        w.assert_not_called()
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertNotIn("reel", plan)


class Prompt(unittest.TestCase):

    def test_reel_prompt_asks_for_hooks_scenes_and_one_cta(self):
        seen = {}

        def fake(messages, max_tokens=8000):
            seen["user"] = messages[1]["content"]
            return copy.deepcopy(RAW)

        with mock.patch.object(llm, "_chat", fake):
            llm.write_reel_script("local LLMs", "topic", words=100, channel=CHANNEL, hook="Plan hook line")
        u = seen["user"]
        self.assertIn("at least 5 candidate", u)
        self.assertIn("exactly ONE \"cta\" scene", u)
        self.assertIn("Never claim something is free", u)
        self.assertIn("Plan hook line", u)


if __name__ == "__main__":
    unittest.main()
