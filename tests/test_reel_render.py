"""Reel format on the render side: scene-aligned clip prompts, the hook card on frame 0, the CTA on
the end card, captions inside the Reels safe area, and narration with its dead air trimmed.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-reelrender-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from services import captions, cardtemplates, llm, reel, tts  # noqa: E402
from services.ffbin import ffmpeg  # noqa: E402

PORTRAIT = "9:16 (Portrait Widescreen)"
CHANNEL = {"name": "Daily Tech Talks", "purpose": "", "rules": "", "audience": "general",
           "video_type": "tech", "cast": [], "characters_enabled": False, "reel_format": True}
REEL = {"hooks": [{"text": "Run a huge model on one GPU?", "category": "outcome",
                   "scores": {}, "supported": True, "rejected": ""}],
        "chosen": 0,
        "scenes": [{"n": 1, "role": "hook", "voiceover": "Run a huge model on one GPU? Here is how.",
                    "headline": "ONE GPU, HUGE MODEL", "visual": "the model generating text"},
                   {"n": 2, "role": "demo", "voiceover": "This is the real setup running.",
                    "headline": "THE SETUP", "visual": "terminal with tokens streaming"},
                   {"n": 3, "role": "cta", "voiceover": "Save this for later.", "headline": "SAVE"}],
        "cta": {"type": "save", "text": "Save this if you are experimenting with local AI."}}


def _job(reel_on=True, polish=True):
    ch = dict(CHANNEL, reel_format=reel_on)
    job = pipeline.Job(id="r" + ("1" if reel_on else "0"), url="", aspect_ratio=PORTRAIT,
                       audio_mode="narration+music", source="topic", channel=ch, polish=polish)
    job.dir.mkdir(parents=True, exist_ok=True)
    return job


class PolishCards(unittest.TestCase):

    def test_reel_hook_card_starts_on_frame_zero_with_the_hook_headline(self):
        job = _job()
        hook_end, end_start = pipeline._polish_windows(job, 40.0)
        self.assertAlmostEqual(hook_end, config.HOOK_CARD_SECONDS)
        seen = {}
        real = cardtemplates.render_hook

        def spy(text, *a, **kw):
            seen["text"] = text
            return real(text, *a, **kw)

        with mock.patch.object(cardtemplates, "render_hook", spy):
            layers, hits = pipeline._polish_layers(job, {"reel": REEL, "title": "t"}, (720, 1280),
                                                   40.0, hook_end, end_start)
        self.assertEqual(seen["text"], "ONE GPU, HUGE MODEL")
        hook = layers[0]
        self.assertEqual(hook["start"], 0.0)
        self.assertEqual(hook["fade_in"], 0.0)
        self.assertEqual(hits[0], 0.0)

    def test_reel_end_card_carries_the_cta(self):
        self.assertEqual(pipeline._end_line(_job(), {"reel": REEL}), REEL["cta"]["text"])
        self.assertEqual(pipeline._end_line(_job(reel_on=False), {"reel": REEL}), config.END_CARD_TEXT)

    def test_other_videos_keep_the_old_timing(self):
        job = _job(reel_on=False)
        self.assertEqual(pipeline._hook_start(job), 0.2)

    def test_long_cta_wraps_inside_the_card(self):
        out = cardtemplates.render_end((720, 1280), TMP / "end.png", channel="Daily Tech Talks",
                                       line="Share this with a developer who would find it useful today.")
        self.assertTrue(out.exists())


class SceneHints(unittest.TestCase):

    def test_clip_hints_carry_opening_rule_role_and_visual(self):
        hints = reel.clip_hints(REEL["scenes"], [0, 1, 1, 2])
        self.assertIn("OPENING clip", hints[0])
        self.assertNotIn("OPENING clip", hints[1])
        self.assertIn("terminal with tokens streaming", hints[1])
        self.assertIn("close enough to read on a phone", hints[2])

    def test_write_scenes_puts_the_hint_in_the_prompt(self):
        prompts = []

        def fake(messages, max_tokens=8000):
            prompts.append(messages[1]["content"])
            return {"scene": "a scene"}

        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("a b c d", "style", 2, segments=["a b", "c d"], channel=CHANNEL,
                             hints=["HINT ONE", ""])
        self.assertIn("REEL DIRECTION for this clip: HINT ONE", prompts[0])
        self.assertNotIn("REEL DIRECTION", prompts[1])

    def test_reel_hints_only_for_reels(self):
        plan = {"reel": REEL, "clip_scene": [0, 1, 2]}
        self.assertEqual(len(pipeline._reel_hints(_job(), plan)), 3)
        self.assertIsNone(pipeline._reel_hints(_job(reel_on=False), plan))


class SafeArea(unittest.TestCase):

    def test_reel_captions_use_asymmetric_margins_and_a_higher_floor(self):
        safe = captions.reel_safe()
        size = (1080, 1920)
        phrases = [{"start": 0.0, "end": 1.0, "text": "hello there",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5},
                              {"word": "there", "start": 0.5, "end": 1.0}]}]
        reel_ass = captions.write_ass(phrases, size, TMP / "reel.ass", safe=safe).read_text(encoding="utf-8")
        plain_ass = captions.write_ass(phrases, size, TMP / "plain.ass").read_text(encoding="utf-8")
        style = next(l for l in reel_ass.splitlines() if l.startswith("Style: Cap"))
        f = style.split(",")
        ml, mr, mv = int(f[-4]), int(f[-3]), int(f[-2])
        self.assertEqual(ml, int(config.REEL_SAFE_LEFT * 1080))
        self.assertEqual(mr, int(config.REEL_SAFE_RIGHT * 1080))
        self.assertEqual(mv, int(config.REEL_SAFE_BOTTOM * 1920))
        plain = next(l for l in plain_ass.splitlines() if l.startswith("Style: Cap")).split(",")
        self.assertEqual(int(plain[-4]), int(plain[-3]))
        self.assertLess(int(plain[-2]), mv)

    def test_band_rises_with_the_reel_bottom(self):
        size = (1080, 1920)
        self.assertLess(captions.band(size, safe=captions.reel_safe())[1], captions.band(size)[1])

    def test_cards_respect_the_safe_area_only_inside_the_context(self):
        size = (1080, 1920)
        plain = cardtemplates.context(size, None, 1, 1)
        with cardtemplates.safe_area(captions.reel_safe()):
            inside = cardtemplates.context(size, None, 1, 1)
        after = cardtemplates.context(size, None, 1, 1)
        self.assertEqual(inside.m, int(config.REEL_SAFE_RIGHT * 1080))
        self.assertGreater(inside.bottom_safe, plain.bottom_safe)
        self.assertEqual(after.m, plain.m)


@unittest.skipUnless(shutil.which(ffmpeg()) or Path(ffmpeg()).exists(), "ffmpeg not available")
class TrimSilence(unittest.TestCase):

    def test_both_ends_are_trimmed_and_the_middle_kept(self):
        wav = TMP / "n.wav"
        # 1.0 s silence, 0.5 s tone, 0.6 s silence, 0.5 s tone, 1.0 s silence = 3.6 s
        graph = ("anullsrc=r=44100:cl=mono:d=1[s1];sine=f=440:r=44100:d=0.5[t1];"
                 "anullsrc=r=44100:cl=mono:d=0.6[s2];sine=f=440:r=44100:d=0.5[t2];"
                 "anullsrc=r=44100:cl=mono:d=1[s3];[s1][t1][s2][t2][s3]concat=n=5:v=0:a=1")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-filter_complex", graph, str(wav)], check=True)
        new = tts.trim_silence(wav, keep=0.15)
        self.assertGreater(new, 1.5)     # tone + inner gap + tone survive
        self.assertLess(new, 2.2)        # about 1.6 s of speech + up to 0.3 s kept at the ends


if __name__ == "__main__":
    unittest.main()
