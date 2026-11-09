"""The DIY build channel type: a fast vertical "watch him build it" short.

One recurring builder makes one project in one place; a quick third-person voiceover (hook question ->
play-by-play -> payoff) runs over light music and the clips' own work sounds; nothing is on screen.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-diy-"))
config.OUTPUT_DIR = TMP / "output"
config.CHANNELS_DIR = TMP / "channels"
config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"

from services import cast, channels, llm, tts  # noqa: E402
from services.ffbin import ffmpeg  # noqa: E402
import pipeline  # noqa: E402


def _ch(**defaults):
    rec = channels._validate({"name": "Backyard Builds", "video_type": "diy", "defaults": dict(defaults)})
    rec["id"] = "c1"
    return rec


class Channel(unittest.TestCase):

    def test_a_diy_channel_validates_as_realistic_with_no_overlays(self):
        rec = channels._validate({"name": "C", "video_type": "diy", "defaults": {"art_style": "Pixar-like 3D"}})
        self.assertEqual(rec["defaults"]["art_style"], "")
        for key in ("storyboard", "captions", "polish"):
            self.assertFalse(rec["defaults"][key], key)

    def test_a_new_diy_channel_starts_from_the_build_short_format(self):
        rec = channels._validate({"name": "C", "video_type": "diy"})
        self.assertEqual(rec["defaults"]["aspect_ratio"], "9:16 (Portrait Widescreen)")
        self.assertEqual(rec["defaults"]["audio_mode"], "narration+music")
        self.assertEqual(rec["defaults"]["video_length"], "30-45 seconds")
        self.assertEqual(rec["cast_portrait"], "reference")

    def test_editing_a_diy_channel_keeps_its_own_choices(self):
        rec = channels._validate({"name": "C", "video_type": "diy", "cast_portrait": "opening",
                                  "defaults": {"aspect_ratio": "1:1 (Square)"}})
        again = channels._validate({"name": "D"}, base=rec)
        self.assertEqual(again["defaults"]["aspect_ratio"], "1:1 (Square)")
        self.assertEqual(again["cast_portrait"], "opening")

    def test_it_files_under_howto_and_style(self):
        self.assertEqual(config.YT_CATEGORY_BY_TYPE["diy"], "26")


class Choices(unittest.TestCase):

    @staticmethod
    def _resolve(channel, **kw):
        return pipeline.JobManager._resolve_choices(
            kw.get("ar", ""), "", kw.get("vlen", ""), kw.get("audio", ""), "", channel,
            None, None, "", "", None, None, kw.get("model"))

    def test_unset_choices_take_the_build_short_format(self):
        r = self._resolve(_ch())
        self.assertEqual(r[0], "9:16 (Portrait Widescreen)")
        self.assertEqual(r[2], "30-45 seconds")
        self.assertEqual(r[3], "narration+music")
        self.assertEqual(r[11], "LTX 2.5")
        self.assertFalse(r[9])      # captions

    def test_the_request_and_the_channel_still_win(self):
        r = self._resolve(_ch(aspect_ratio="1:1 (Square)"), vlen="1-2 minutes", model="MiniMax H3")
        self.assertEqual(r[0], "1:1 (Square)")
        self.assertEqual(r[2], "1-2 minutes")
        self.assertEqual(r[11], "MiniMax H3")

    def test_other_channel_types_are_untouched(self):
        rec = channels._validate({"name": "C", "video_type": "real"})
        rec["id"] = "c2"
        r = self._resolve(rec)
        self.assertEqual(r[0], config.DEFAULT_ASPECT_RATIO)
        self.assertEqual(r[11], config.DEFAULT_VIDEO_MODEL)


class Pacing(unittest.TestCase):

    def _job(self, video_type="diy", length="30-45 seconds"):
        return SimpleNamespace(channel={"video_type": video_type}, video_length=length,
                               lyrics="", music_upload="", music_seconds=0.0)

    def test_diy_clips_are_short_and_not_floored_by_the_global_length(self):
        self.assertEqual(pipeline._clip_seconds(self._job()), float(config.DIY_CLIP_SECONDS))
        self.assertGreaterEqual(pipeline._clip_seconds(self._job("real")), float(config.CLIP_SECONDS))

    def test_a_sped_up_voiceover_is_written_with_more_words(self):
        diy = pipeline._length_params(self._job())[0]
        real = pipeline._length_params(self._job("real"))[0]
        self.assertEqual(diy, int(37.5 * 2.5 * config.DIY_NARRATION_TEMPO))
        self.assertGreater(diy, real)

    def test_change_tempo_shortens_the_file(self):
        wav = TMP / "n.wav"
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=4", str(wav)], check=True)
        self.assertAlmostEqual(tts.change_tempo(wav, 1.25), 3.2, delta=0.1)


class Prompts(unittest.TestCase):

    def test_the_system_prompt_carries_the_build_short_brief(self):
        sysp = llm._system({"name": "B", "video_type": "diy", "audience": "general"})
        self.assertIn("DIY BUILD channel", sysp)
        self.assertIn("curiosity hook", sysp)
        self.assertIn("payoff", sysp)
        self.assertNotIn("DIY BUILD", llm._system({"name": "B", "video_type": "real"}))

    def test_diy_scenes_may_cut_between_angles_and_name_work_sounds(self):
        rules = llm._scene_rules({"video_type": "diy"}, "3")
        self.assertIn("NEW camera angle", rules)
        self.assertIn("WORK SOUNDS", rules)
        self.assertNotIn("Never open with a cut", rules)
        self.assertNotIn("[4s-3s]", rules)
        self.assertIn("People on screen", rules)        # the gaze rules survive the swap

    def test_other_types_keep_the_one_take_contract(self):
        self.assertEqual(llm._scene_rules({"video_type": "real"}, "5"), llm.SCENE_RULES.format(secs="5"))

    def test_the_style_example_is_phone_footage(self):
        self.assertIn("smartphone", llm._style_example({"video_type": "diy"}))


class RandomBuilder(unittest.TestCase):
    """A DIY channel can cast a new person per video instead of a saved character: one description,
    invented once, that every clip of that video carries."""
    SAVED = [{"id": "a1", "name": "Hank", "look": "denim overalls in a cornfield", "role": "main"}]

    def snap(self, builder="random", video_type="diy"):
        rec = channels._validate({"name": "B", "video_type": video_type, "diy_builder": builder,
                                  "characters_enabled": True,
                                  "cast": [dict(c) for c in self.SAVED]})
        return channels.snapshot(rec)

    def test_the_channel_setting_validates_and_travels_with_the_job(self):
        self.assertEqual(channels._validate({"name": "B", "video_type": "diy"})["diy_builder"], "cast")
        self.assertEqual(self.snap()["diy_builder"], "random")
        with self.assertRaises(ValueError):
            channels._validate({"name": "B", "diy_builder": "sometimes"})

    def test_editing_keeps_the_choice(self):
        rec = channels._validate({"name": "B", "video_type": "diy", "diy_builder": "random"})
        self.assertEqual(channels._validate({"name": "C"}, base=rec)["diy_builder"], "random")

    def test_the_same_job_always_gets_the_same_person(self):
        self.assertEqual(cast.random_builder("job-1"), cast.random_builder("job-1"))

    def test_different_jobs_get_different_people(self):
        looks = {cast.random_builder(f"job-{i}")["look"] for i in range(20)}
        self.assertGreater(len(looks), 15)

    def test_the_look_is_a_person_and_never_a_place(self):
        for i in range(50):
            look = cast.random_builder(f"j{i}")["look"]
            self.assertIn("wearing", look)
            for place in ("field", "farm", "backyard", "garage", "workshop"):
                self.assertNotIn(place, look)

    def test_the_random_builder_replaces_the_saved_cast(self):
        ch = self.snap()
        b = cast.cast_random_builder(ch, "job-1")
        self.assertEqual(cast.names(cast.active(ch)), [b["name"]])
        self.assertIsNone(cast.cast_random_builder(ch, "job-1"))     # once per job
        self.assertEqual(cast.random_builder_name(ch), b["name"])

    def test_saved_cast_mode_and_other_types_are_untouched(self):
        for ch in (self.snap(builder="cast"), self.snap(video_type="real")):
            self.assertIsNone(cast.cast_random_builder(ch, "job-1"))
            self.assertEqual(cast.names(cast.active(ch)), ["Hank"])

    def test_the_builder_is_in_the_video_even_when_the_script_never_names_them(self):
        ch = self.snap()
        b = cast.cast_random_builder(ch, "job-1")
        job = SimpleNamespace(channel=ch, source="prompt")
        plan = {"title": "Pallet planter", "narration": "He drills the slats, then fills them."}
        self.assertEqual(pipeline._set_cast_used(job, plan), [b["name"]])
        self.assertIn(b["look"], cast.looks_prefix(ch, plan["cast_used"]))

    def test_no_portrait_is_looked_up_for_the_random_builder(self):
        ch = self.snap()
        cast.cast_random_builder(ch, "job-1")
        self.assertIsNone(cast.opening_character(ch, "c1", used=cast.names(cast.active(ch))))


if __name__ == "__main__":
    unittest.main()
