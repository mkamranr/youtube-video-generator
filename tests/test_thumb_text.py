"""Thumbnail overlay text and variants: run with  python -m unittest discover -s tests -t .

Pillow only - no GPU, no LLM call, no network. The logo-corner assertions compare against the
JPEG-reencode noise floor rather than demanding exact equality, because saving at quality 92
perturbs every pixel even in a region that was pasted back untouched.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops, ImageDraw

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-thumb-"))
config.OUTPUT_DIR = TMP / "output"

from services import llm, thumbtext  # noqa: E402  (config patched first)

SIZE = (1280, 720)


def make_base(path: Path) -> Path:
    im = Image.new("RGB", SIZE, (40, 55, 95))
    d = ImageDraw.Draw(im)
    for y in range(SIZE[1]):                       # a gradient, so re-encoding is realistic
        d.line([(0, y), (SIZE[0], y)], fill=(30 + y // 8, 50 + y // 10, 90 + y // 6))
    d.ellipse((760, 150, 1180, 570), fill=(230, 180, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
    return path


def max_delta(a: Image.Image, b: Image.Image) -> int:
    return max(e[1] for e in ImageChops.difference(a, b).getextrema())


class Normalize(unittest.TestCase):

    def test_trims_to_five_words_and_strips_punctuation(self):
        self.assertEqual(thumbtext.normalize('"The  REAL cost of running LLMs locally."'),
                         "The REAL cost of running")

    def test_drops_smart_quotes_and_ellipsis(self):
        self.assertEqual(thumbtext.normalize("“Nobody tells you…”"),
                         "Nobody tells you")

    def test_respects_the_character_cap_on_a_word_boundary(self):
        out = thumbtext.normalize("Supercalifragilistic expialidocious wordy")
        self.assertLessEqual(len(out), 24)

    def test_empty_stays_empty(self):
        self.assertEqual(thumbtext.normalize(""), "")
        self.assertEqual(thumbtext.normalize(None), "")


class LlmField(unittest.TestCase):

    def _thumb(self, out):
        with mock.patch.object(llm, "_chat", return_value=out):
            return llm.write_thumbnail("T", "narration", "style")

    def test_texts_are_returned_and_normalized(self):
        got = self._thumb({"thumbnail_prompt": "a scene",
                           "thumbnail_texts": ['"40% cheaper."', "Nobody tells you this"]})
        self.assertEqual(got["thumbnail_texts"], ["40% cheaper", "Nobody tells you this"])

    def test_a_bare_string_is_accepted(self):
        got = self._thumb({"thumbnail_prompt": "a scene", "thumbnail_texts": "One line only"})
        self.assertEqual(got["thumbnail_texts"], ["One line only"])

    def test_duplicates_are_dropped(self):
        got = self._thumb({"thumbnail_prompt": "a scene",
                           "thumbnail_texts": ["Same line", "same LINE", "Other"]})
        self.assertEqual(got["thumbnail_texts"], ["Same line", "Other"])

    def test_missing_texts_is_not_an_error(self):
        self.assertEqual(self._thumb({"thumbnail_prompt": "a scene"})["thumbnail_texts"], [])

    def test_a_missing_prompt_still_raises(self):
        with self.assertRaises(RuntimeError):
            self._thumb({"thumbnail_texts": ["x"]})

    def test_the_prompt_still_forbids_letters_in_the_picture(self):
        """Adding a text field must never relax the no-typography rule on the image itself."""
        seen = {}

        def grab(msgs, **kw):
            seen["user"] = msgs[-1]["content"]
            return {"thumbnail_prompt": "a scene", "thumbnail_texts": []}

        with mock.patch.object(llm, "_chat", side_effect=grab):
            llm.write_thumbnail("T", "narration", "style")
        self.assertIn("No text, subtitles, captions, letters", seen["user"])
        self.assertIn("letters of any kind", seen["user"])


class KidsCaps(unittest.TestCase):
    """A kids thumbnail is scanned on a phone by an adult, for a child who cannot read at all, so
    the line has to work as a shape: two or three big words, not five small ones."""

    KIDS = {"name": "Bright Minds Town", "purpose": "", "audience": "kids",
            "video_type": "animated", "cast": []}
    GENERAL = {**KIDS, "audience": "general"}

    def test_caps_by_audience(self):
        self.assertEqual(thumbtext.caps(self.KIDS), (3, 18))
        self.assertEqual(thumbtext.caps(self.GENERAL), thumbtext.DEFAULT_CAPS)
        self.assertEqual(thumbtext.caps({}), thumbtext.DEFAULT_CAPS)
        self.assertEqual(thumbtext.caps(None), thumbtext.DEFAULT_CAPS)

    def _prompt_and_texts(self, channel, texts):
        seen = {}

        def grab(msgs, **kw):
            seen["user"] = msgs[-1]["content"]
            return {"thumbnail_prompt": "a scene", "thumbnail_texts": texts}

        with mock.patch.object(llm, "_chat", side_effect=grab):
            out = llm.write_thumbnail("T", "narration", "style", channel=channel)
        return seen["user"], out["thumbnail_texts"]

    def test_a_kids_line_is_cut_to_three_words(self):
        _u, texts = self._prompt_and_texts(self.KIDS, ["All aboard the little red train"])
        self.assertEqual(texts, ["All aboard the"])

    def test_a_general_line_keeps_the_original_caps(self):
        _u, texts = self._prompt_and_texts(self.GENERAL, ["All aboard the little red train"])
        self.assertEqual(texts, ["All aboard the little"])

    def test_the_prompt_asks_for_what_normalize_enforces(self):
        """One source of truth: if the prompt and the cap disagreed, every line would be written
        at one length and then silently chopped to another."""
        kids, _t = self._prompt_and_texts(self.KIDS, [])
        self.assertIn("at most\n  3 words and under 18 characters", kids)
        general, _t = self._prompt_and_texts(self.GENERAL, [])
        self.assertIn("at most\n  5 words and under 24 characters", general)

    def test_only_a_kids_thumbnail_gets_the_toddler_composition_rule(self):
        kids, _t = self._prompt_and_texts(self.KIDS, [])
        self.assertIn("TODDLER audience", kids)
        self.assertIn("filling most of the frame", kids)
        general, _t = self._prompt_and_texts(self.GENERAL, [])
        self.assertNotIn("TODDLER", general)
        # the rules it must not displace
        self.assertIn("TOP-RIGHT corner free", kids)
        self.assertIn("No text, subtitles, captions, letters", kids)


class Render(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = TMP / "render"
        cls.base = make_base(cls.dir / "thumb_base.jpg")

    def _corner(self, img: Image.Image) -> Image.Image:
        W = SIZE[0]
        return img.crop((W - int(0.26 * W), 0, W, int(0.26 * W)))

    def test_every_layout_keeps_the_size_and_mode(self):
        for layout in thumbtext.LAYOUTS:
            out = thumbtext.render(self.base, self.dir / f"{layout}.jpg", "The real cost", layout)
            with Image.open(out) as im:
                self.assertEqual(im.size, SIZE, layout)
                self.assertEqual(im.mode, "RGB", layout)

    def test_no_layout_draws_into_the_logo_corner(self):
        """The corner is restored from the original, so only JPEG noise may differ there."""
        with Image.open(self.base) as src:
            orig = src.convert("RGB")
        floor = 0
        for layout in thumbtext.LAYOUTS:
            out = thumbtext.render(self.base, self.dir / f"c_{layout}.jpg",
                                   "A very long line that wraps", layout)
            with Image.open(out) as v:
                v = v.convert("RGB")
                floor = max(floor, max_delta(self._corner(orig), self._corner(v)))
        self.assertLess(floor, 60, "a layout drew under the channel logo")

    def test_a_layout_actually_changes_the_picture(self):
        """Guards against the corner test passing because nothing was drawn at all."""
        with Image.open(self.base) as src:
            orig = src.convert("RGB")
        out = thumbtext.render(self.base, self.dir / "drawn.jpg", "The real cost", "band")
        with Image.open(out) as v:
            self.assertGreater(max_delta(orig, v.convert("RGB")), 100)

    def test_empty_text_is_refused(self):
        with self.assertRaises(ValueError):
            thumbtext.render(self.base, self.dir / "x.jpg", "   ", "band")

    def test_portrait_and_square_bases_render(self):
        for size in ((720, 1280), (1080, 1080)):
            b = make_base(self.dir / f"base_{size[0]}x{size[1]}.jpg")
            Image.new("RGB", size, (50, 60, 90)).save(b, "JPEG", quality=92)
            for layout in thumbtext.LAYOUTS:
                out = thumbtext.render(b, self.dir / f"{size[0]}_{layout}.jpg", "Short line", layout)
                with Image.open(out) as im:
                    self.assertEqual(im.size, size)


class Variants(unittest.TestCase):

    def setUp(self):
        self.dir = TMP / f"var_{self.id().rsplit('.', 1)[-1]}"
        self.base = make_base(self.dir / "thumb_base.jpg")

    def test_names_count_and_distinct_layouts(self):
        recs = thumbtext.variants(self.base, self.dir, ["One line", "Two line", "Three line"])
        self.assertEqual([r["file"] for r in recs],
                         ["thumb_var_01.jpg", "thumb_var_02.jpg", "thumb_var_03.jpg"])
        self.assertEqual(len({r["layout"] for r in recs}), 3)
        for r in recs:
            self.assertTrue((self.dir / r["file"]).exists())

    def test_a_failing_layout_never_costs_the_others(self):
        # _DRAW binds the function objects at import, so the dict entry is what has to be patched
        broken = dict(thumbtext._DRAW, stack=mock.Mock(side_effect=RuntimeError("boom")))
        with mock.patch.object(thumbtext, "_DRAW", broken):
            recs = thumbtext.variants(self.base, self.dir, ["One", "Two", "Three"])
        self.assertEqual(len(recs), 2)
        self.assertNotIn("stack", [r["layout"] for r in recs])

    def test_no_texts_gives_no_variants(self):
        self.assertEqual(thumbtext.variants(self.base, self.dir, []), [])

    def test_stale_variants_from_a_previous_pass_are_removed(self):
        """Clearing moved out of variants() when a second base (the presenter thumbnail) began
        numbering its files after the first: a pass that cleaned up after itself would have
        deleted the batch before it."""
        thumbtext.variants(self.base, self.dir, ["One", "Two", "Three"])
        recs = thumbtext.variants(self.base, self.dir, ["Only one"])
        self.assertEqual(len(recs), 1)
        thumbtext.drop_stale(self.dir, len(recs))
        self.assertFalse(thumbtext.variant_path(self.dir, 2).exists())

    def test_an_offset_numbers_a_second_base_after_the_first(self):
        first = thumbtext.variants(self.base, self.dir, ["One", "Two"])
        second = thumbtext.variants(self.base, self.dir, ["Three"], offset=len(first))
        self.assertEqual([r["file"] for r in first],
                         [thumbtext.variant_path(self.dir, k).name for k in (1, 2)])
        self.assertEqual(second[0]["file"], thumbtext.variant_path(self.dir, 3).name)
        for k in (1, 2, 3):
            self.assertTrue(thumbtext.variant_path(self.dir, k).exists())


class Activate(unittest.TestCase):

    def setUp(self):
        self.dir = TMP / f"act_{self.id().rsplit('.', 1)[-1]}"
        self.base = make_base(self.dir / "thumb_base.jpg")
        self.recs = thumbtext.variants(self.base, self.dir, ["One line", "Two line"])

    def test_the_chosen_variant_becomes_thumbnail_jpg(self):
        k = thumbtext.activate(self.dir, self.recs, 2)
        self.assertEqual(k, 2)
        self.assertEqual((self.dir / "thumbnail.jpg").read_bytes(),
                         (self.dir / self.recs[1]["file"]).read_bytes())

    def test_zero_means_the_clean_render_and_sticks(self):
        """0 is a deliberate 'no text please' and must not be clamped away to variant 1."""
        self.assertEqual(thumbtext.activate(self.dir, self.recs, 0), 0)
        self.assertEqual((self.dir / "thumbnail.jpg").read_bytes(), self.base.read_bytes())

    def test_auto_picks_the_models_own_first_line(self):
        self.assertEqual(thumbtext.activate(self.dir, self.recs, thumbtext.AUTO), 1)

    def test_an_out_of_range_choice_clamps_to_the_first(self):
        self.assertEqual(thumbtext.activate(self.dir, self.recs, 9), 1)

    def test_a_deleted_variant_falls_back(self):
        (self.dir / self.recs[1]["file"]).unlink()
        self.assertEqual(thumbtext.activate(self.dir, self.recs, 2), 1)

    def test_no_variants_at_all_falls_back_to_the_clean_render(self):
        self.assertEqual(thumbtext.activate(self.dir, [], 1), 0)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
