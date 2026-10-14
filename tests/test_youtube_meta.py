"""YouTube tags, category and chapters: run with  python -m unittest discover -s tests -t .

Pure functions over plan data - no network, no Google client, no ffmpeg.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-ytmeta-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from services import chapters, llm, youtube  # noqa: E402


def cards(times_titles) -> list:
    return [{"start_clip": clip, "end_clip": clip, "kicker": "KEY FACT", "headline": title,
             "body": "", "image_prompt": ""} for clip, title in times_titles]


class Keywords(unittest.TestCase):
    """write_seo used to ask the model for keywords and then drop them on the floor."""

    def _seo(self, out):
        with mock.patch.object(llm, "_chat", return_value=out):
            return llm.write_seo("T", "narration", "transcript")

    def test_keywords_survive_into_the_plan(self):
        got = self._seo({"seo_title": "A title", "seo_description": "Body text",
                         "seo_keywords": ["local llm", "ollama", "gpu"]})
        self.assertEqual(got["seo_keywords"], ["local llm", "ollama", "gpu"])

    def test_keywords_are_bounded_and_blanks_dropped(self):
        got = self._seo({"seo_title": "t", "seo_description": "d",
                         "seo_keywords": [f"kw{i}" for i in range(30)] + ["", "  "]})
        self.assertEqual(len(got["seo_keywords"]), 12)

    def test_missing_keywords_is_not_an_error(self):
        self.assertEqual(self._seo({"seo_title": "t", "seo_description": "d"})["seo_keywords"], [])


class Tags(unittest.TestCase):

    def test_dedupes_case_insensitively_and_strips_brackets(self):
        got = youtube.clean_tags(["Local LLM", "local llm", "<script>", 'say "hi"'])
        self.assertEqual([t.lower() for t in got].count("local llm"), 1)
        self.assertTrue(all("<" not in t and ">" not in t and '"' not in t for t in got))

    def test_respects_the_500_character_budget(self):
        got = youtube.clean_tags([f"a long keyword phrase number {i}" for i in range(40)])
        self.assertLessEqual(sum(len(t) + (2 if " " in t else 0) + 1 for t in got), 500)

    def test_a_sentence_is_not_a_tag(self):
        self.assertEqual(youtube.clean_tags(["x" * 80]), [])

    def test_max_tags_is_honoured(self):
        self.assertEqual(len(youtube.clean_tags([f"kw{i}" for i in range(40)], max_tags=5)), 5)

    def test_empty_input_returns_empty(self):
        self.assertEqual(youtube.clean_tags([]), [])
        self.assertEqual(youtube.clean_tags(None), [])


class Chapters(unittest.TestCase):

    def setUp(self):
        self.bounds = [float(10 * i) for i in range(1, 19)]      # 18 clips of 10 s = 180 s

    def test_the_first_chapter_is_always_zero(self):
        got = chapters.build(self.bounds, cards=cards([(1, "Opening"), (4, "Middle"), (10, "End")]))
        self.assertTrue(got)
        self.assertEqual(got[0]["t"], 0.0)
        self.assertEqual(chapters.fmt_time(got[0]["t"]), "0:00")

    def test_chapters_are_never_closer_than_the_minimum(self):
        got = chapters.build(self.bounds,
                             cards=cards([(1, "A"), (2, "B"), (3, "C"), (8, "D"), (14, "E")]),
                             min_gap=20.0)
        for a, b in zip(got, got[1:]):
            self.assertGreaterEqual(b["t"] - a["t"], 20.0)

    def test_the_last_chapter_still_leaves_ten_seconds(self):
        got = chapters.build(self.bounds,
                             cards=cards([(1, "A"), (5, "B"), (10, "C"), (18, "Too late")]))
        self.assertTrue(all(180.0 - c["t"] >= 10.0 for c in got))

    def test_fewer_than_three_gives_nothing(self):
        """YouTube ignores a partial list entirely, so half a list is worse than none."""
        self.assertEqual(chapters.build(self.bounds, cards=cards([(1, "A"), (10, "B")])), [])

    def test_a_short_video_gives_nothing(self):
        self.assertEqual(chapters.build([10.0, 20.0, 25.0], cards=cards([(1, "A"), (2, "B")])), [])

    def test_falls_back_to_story_beats(self):
        beats = [{"action": f"Beat {i}", "setting": "x"} for i in range(1, 19)]
        got = chapters.build(self.bounds, cards=None, beats=beats)
        self.assertGreaterEqual(len(got), 3)

    def test_falls_back_to_segments(self):
        segs = [f"this is what is heard during clip {i} of the video" for i in range(1, 19)]
        got = chapters.build(self.bounds, cards=None, segments=segs)
        self.assertGreaterEqual(len(got), 3)

    def test_titles_lose_hashes_and_angle_brackets(self):
        """A stray '#' would become a YouTube hashtag and outrank the channel's real ones."""
        self.assertEqual(chapters.clean_title("#1 <fast> setup."), "1 fast setup")

    def test_titles_are_trimmed_on_a_word_boundary(self):
        t = chapters.clean_title("word " * 40)
        self.assertLessEqual(len(t), 55)
        self.assertFalse(t.endswith(" "))

    def test_an_empty_title_is_dropped_not_rendered_blank(self):
        got = chapters.build(self.bounds, cards=cards([(1, ""), (5, "B"), (10, "C"), (14, "D")]))
        self.assertTrue(all(c["title"] for c in got))

    def test_capped_at_max_count(self):
        got = chapters.build([float(30 * i) for i in range(1, 40)],
                             cards=cards([(i, f"Card {i}") for i in range(1, 40)]),
                             max_count=5)
        self.assertLessEqual(len(got), 5)

    def test_fmt_time_minutes_and_hours(self):
        self.assertEqual(chapters.fmt_time(0), "0:00")
        self.assertEqual(chapters.fmt_time(47), "0:47")
        self.assertEqual(chapters.fmt_time(3725), "1:02:05")

    def test_format_block_is_empty_for_no_chapters(self):
        self.assertEqual(chapters.format_block([]), "")


class Description(unittest.TestCase):

    def test_chapters_are_appended_after_the_hashtags_and_resource(self):
        desc = "Hook sentence.\n\nSubscribe to Channel\n#one #two\n\nTHE RESOURCE\nstep one"
        plan = {"seo_description": desc,
                "chapters": [{"t": 0.0, "title": "Intro"}, {"t": 47.0, "title": "Why"}]}
        out = pipeline.upload_description(plan)
        self.assertTrue(out.startswith(desc))
        self.assertIn("\n0:00 Intro\n0:47 Why", out)
        self.assertLess(out.index("#one"), out.index("0:00 Intro"))

    def test_no_chapters_leaves_the_description_byte_identical(self):
        plan = {"seo_description": "Just the text."}
        self.assertEqual(pipeline.upload_description(plan), "Just the text.")

    def test_an_overlong_description_is_trimmed_not_the_chapters(self):
        plan = {"seo_description": "x" * (config.YT_DESCRIPTION_LIMIT + 500),
                "chapters": [{"t": 0.0, "title": "Intro"}, {"t": 47.0, "title": "Why"}]}
        out = pipeline.upload_description(plan)
        self.assertLessEqual(len(out), config.YT_DESCRIPTION_LIMIT)
        self.assertIn("0:00 Intro", out)


class Category(unittest.TestCase):

    def test_a_tech_channel_is_science_and_technology(self):
        self.assertEqual(config.YT_CATEGORY_BY_TYPE.get("tech"), "28")

    def test_other_channels_keep_the_default(self):
        self.assertEqual(config.YT_CATEGORY_BY_TYPE.get("animated", config.YT_CATEGORY_ID), "24")

    def test_the_setting_only_accepts_a_real_category(self):
        ok = config.EDITABLE_SETTINGS["YT_CATEGORY_ID"][1]
        self.assertTrue(ok("28"))
        self.assertFalse(ok("999"))


class TitleRule(unittest.TestCase):
    """A kids title is scanned, not read: by a parent typing two or three words, and by a sidebar
    that truncates around 60 characters. The searched words therefore have to come first."""

    KIDS = {"name": "Bright Minds Town", "purpose": "", "audience": "kids",
            "video_type": "animated", "cast": []}

    def _seo_prompt(self, channel):
        seen = {}

        def grab(msgs, **kw):
            seen["user"] = msgs[-1]["content"]
            return {"seo_title": "t", "seo_description": "d"}

        with mock.patch.object(llm, "_chat", side_effect=grab):
            llm.write_seo("T", "narration", "transcript", channel=channel)
        return seen["user"]

    def test_a_kids_channel_front_loads_the_searched_words(self):
        user = self._seo_prompt(self.KIDS)
        self.assertIn("max 60 characters", user)
        self.assertIn("the first 40 must already carry the words a parent would type", user)
        self.assertIn("never open with the channel name", user)

    def test_every_other_audience_keeps_the_original_rule(self):
        for audience in ("general", "family"):
            user = self._seo_prompt({**self.KIDS, "audience": audience})
            self.assertIn("max 70 characters", user)
            self.assertNotIn("a parent would type", user)
        # a job from before channels existed carries no audience and must not change either
        self.assertIn("max 70 characters", self._seo_prompt(None))

    def test_the_rule_is_chosen_by_audience_alone(self):
        self.assertEqual(llm._title_rule(self.KIDS), llm._SEO_TITLE_RULE_KIDS)
        self.assertEqual(llm._title_rule({**self.KIDS, "audience": "general"}),
                         llm._SEO_TITLE_RULE)
        self.assertEqual(llm._title_rule({**self.KIDS, "audience": "nonsense"}),
                         llm._SEO_TITLE_RULE)   # _channel falls unknown audiences back to general

    def test_the_planner_rows_are_written_in_the_same_shape(self):
        """The planner's title is the first draft of what write_seo polishes, so a kids week that
        starts with clever titles spends the SEO call fixing them instead of sharpening them."""
        kids = llm._plan_row_keys(self.KIDS)
        self.assertIn("the video title a parent would click", kids)
        self.assertIn("Clean Up", kids)
        general = llm._plan_row_keys({**self.KIDS, "audience": "general"})
        self.assertIn("the video title a viewer would click, max 12 words", general)
        self.assertNotIn("a parent would click", general)
        # the shape itself is unchanged: every row key still asked for
        for key in ('"title"', '"description"', '"thumbnail_idea"', '"hook"'):
            self.assertIn(key, kids)
            self.assertIn(key, general)


class CompilationSeo(unittest.TestCase):
    """A compilation is sold on its length and on what is inside it."""

    CH = {"name": "Bright Minds Town", "purpose": "", "audience": "kids",
          "video_type": "animated", "cast": []}

    def _call(self, out, titles=("Tilly Counts", "Ellie Waits", "Finn's Kite")):
        seen = {}

        def grab(msgs, **kw):
            seen["user"] = msgs[-1]["content"]
            return out

        with mock.patch.object(llm, "_chat", side_effect=grab):
            got = llm.write_compilation_seo(list(titles), "18 Minutes", channel=self.CH)
        return seen["user"], got

    def test_the_length_and_the_parts_reach_the_prompt(self):
        user, _got = self._call({"seo_title": "t", "seo_description": "d"})
        self.assertIn("18 Minutes", user)
        self.assertIn("- Tilly Counts", user)
        self.assertIn("- Finn's Kite", user)
        self.assertIn("Never promise more time", user)

    def test_the_chapter_list_is_not_asked_for_twice(self):
        """pipeline.upload_description appends the real chapter block afterwards; a second,
        invented one in the description body would fight it."""
        user, _got = self._call({"seo_title": "t", "seo_description": "d"})
        self.assertIn("Do NOT list the videos or their timestamps", user)

    def test_thumbnail_lines_come_back_capped_for_the_audience(self):
        _user, got = self._call({"seo_title": "t", "seo_description": "d",
                                 "thumbnail_texts": ["18 Minutes of Tilly the Train", "18 Minutes"]})
        self.assertEqual(got["thumbnail_texts"], ["18 Minutes of", "18 Minutes"])

    def test_keywords_and_hashtags_behave_like_the_normal_seo_call(self):
        _user, got = self._call({"seo_title": "t", "seo_description": "Body #KidsSongs",
                                 "seo_keywords": [f"kw{i}" for i in range(30)]})
        self.assertEqual(len(got["seo_keywords"]), 12)
        self.assertIn("#kidssongs", got["seo_description"])

    def test_an_empty_reply_yields_empty_fields_rather_than_raising(self):
        """The stitch is the expensive, irreplaceable part: pipeline falls back to a plain title
        rather than losing a finished compilation to a bad reply, so this must not raise."""
        _user, got = self._call({})
        self.assertEqual(got["seo_title"], "")
        self.assertEqual(got["thumbnail_texts"], [])


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
