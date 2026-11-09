"""The reel cover: the poster-style thumbnail layout for 9:16 videos (services/thumbcover.py).

Pillow only - no GPU, no LLM call, no network. Run with
python -m unittest discover -s tests -t .  from the repo root.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops, ImageDraw

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-cover-"))
config.OUTPUT_DIR = TMP / "output"

from services import llm, thumbcover  # noqa: E402  (config patched first)
import pipeline  # noqa: E402

FULL = {"headline": "Xiaomi-*OCR-0*", "strip": "The 0.8B Model.",
        "lines": ["That Fixes Your", "Messy Documents"],
        "badge": {"big": "0.8B", "small": "parameters"},
        "checks": ["Parse Documents", "Answer Questions", "Run Locally"],
        "tag": "Run Locally on Your Laptop!"}


def make_base(path: Path, size=(720, 1280)) -> Path:
    im = Image.new("RGB", size, (30, 40, 70))
    d = ImageDraw.Draw(im)
    for y in range(size[1]):
        d.line([(0, y), (size[0], y)], fill=(30 + y // 20, 40 + y // 30, 70 + y // 15))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
    return path


class Normalize(unittest.TestCase):

    def test_a_full_cover_is_cleaned(self):
        c = thumbcover.normalize(FULL)
        self.assertEqual(c["headline"], "Xiaomi-*OCR-0*")
        self.assertEqual(c["strip"], "The 0.8B Model")             # full stop dropped
        self.assertEqual(c["badge"], {"big": "0.8B", "small": "PARAMETERS"})
        self.assertEqual(len(c["checks"]), 3)

    def test_fields_are_capped_on_word_boundaries(self):
        c = thumbcover.normalize({"strip": "One Two Three Four Five Six",
                                  "checks": ["a", "b", "c", "d", "e"],
                                  "lines": ["x", "y", "z"]})
        self.assertEqual(c["strip"], "One Two Three Four")
        self.assertEqual(len(c["checks"]), 3)
        self.assertEqual(len(c["lines"]), 2)

    def test_accent_markers_do_not_count_and_an_unclosed_one_is_dropped(self):
        self.assertEqual(thumbcover.normalize({"headline": "*Mistral* Large 4"})["headline"],
                         "*Mistral* Large 4")
        self.assertEqual(thumbcover.normalize({"headline": "*Mistral Large 4"})["headline"],
                         "Mistral Large 4")

    def test_without_headline_or_strip_there_is_no_cover(self):
        self.assertIsNone(thumbcover.normalize({"checks": ["a", "b"], "tag": "hi"}))
        self.assertIsNone(thumbcover.normalize(None))
        self.assertIsNone(thumbcover.normalize("a string"))

    def test_a_badge_needs_its_figure(self):
        self.assertIsNone(thumbcover.normalize({"strip": "S", "badge": {"small": "units"}})["badge"])

    def test_bare_strings_are_accepted_for_the_lists(self):
        c = thumbcover.normalize({"strip": "S", "lines": "Is Here", "checks": "Fast"})
        self.assertEqual((c["lines"], c["checks"]), (["Is Here"], ["Fast"]))


class Eligible(unittest.TestCase):

    def test_only_tall_frames(self):
        self.assertTrue(thumbcover.eligible_aspect("9:16 (Portrait Widescreen)"))
        for ar in ("16:9 (Widescreen)", "1:1 (Square)", "3:4 (Portrait Standard)"):
            self.assertFalse(thumbcover.eligible_aspect(ar), ar)


class Render(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(dir=TMP))
        self.base = make_base(self.d / "base.jpg")

    def test_keeps_the_size_and_changes_the_picture(self):
        out = thumbcover.render(self.base, self.d / "c.jpg", FULL)
        with Image.open(out) as a, Image.open(self.base) as b:
            self.assertEqual(a.size, b.size)
            self.assertGreater(max(e[1] for e in ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getextrema()), 100)

    def test_a_minimal_cover_renders(self):
        thumbcover.render(self.base, self.d / "m.jpg", {"strip": "Only A Strip"})

    def test_the_logo_corner_is_restored_without_a_logo_file(self):
        out = thumbcover.render(self.base, self.d / "c.jpg", FULL)
        side = int(0.19 * 720)
        box = (720 - side + 4, 0, 720, side - 4)
        with Image.open(out) as a, Image.open(self.base) as b:
            diff = ImageChops.difference(a.convert("RGB").crop(box), b.convert("RGB").crop(box))
            self.assertLess(max(e[1] for e in diff.getextrema()), 24)

    def test_a_logo_file_is_pasted_back_on_top(self):
        logo = self.d / "logo.png"
        Image.new("RGBA", (100, 100), (255, 0, 0, 255)).save(logo)
        out = thumbcover.render(self.base, self.d / "c.jpg", FULL, logo=logo)
        lw, margin = int(720 * 0.16), max(8, int(720 * 0.025))
        with Image.open(out) as a:
            r, g, b = a.convert("RGB").getpixel((720 - margin - lw // 2, margin + lw // 2))
        self.assertGreater(r, 200)
        self.assertLess(g, 60)

    def test_a_landscape_base_is_refused(self):
        wide = make_base(self.d / "wide.jpg", (1280, 720))
        with self.assertRaises(ValueError):
            thumbcover.render(wide, self.d / "w.jpg", FULL)

    def test_unusable_copy_is_refused(self):
        with self.assertRaises(ValueError):
            thumbcover.render(self.base, self.d / "x.jpg", {"tag": "only a tag"})


class LlmField(unittest.TestCase):

    def _thumb(self, out, cover):
        prompts = []

        def grab(msgs, **kw):
            prompts.append(msgs[-1]["content"])
            return out
        with mock.patch.object(llm, "_chat", side_effect=grab):
            return llm.write_thumbnail("T", "narration", "style", cover=cover), prompts[0]

    def test_cover_asked_and_parsed(self):
        res, prompt = self._thumb({"thumbnail_prompt": "p", "thumbnail_cover": FULL}, True)
        self.assertIn('"thumbnail_cover"', prompt)
        self.assertIn("MIDDLE BAND", prompt)
        self.assertEqual(res["thumbnail_cover"]["strip"], "The 0.8B Model")

    def test_no_cover_unless_asked(self):
        res, prompt = self._thumb({"thumbnail_prompt": "p", "thumbnail_cover": FULL}, False)
        self.assertNotIn('"thumbnail_cover"', prompt)
        self.assertIn("TOP THIRD", prompt)
        self.assertIsNone(res["thumbnail_cover"])

    def test_a_missing_cover_is_not_an_error(self):
        res, _ = self._thumb({"thumbnail_prompt": "p"}, True)
        self.assertIsNone(res["thumbnail_cover"])


class PipelineWiring(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="job-", dir=TMP))
        make_base(self.d / "thumb_base.jpg")
        self.job = mock.MagicMock()
        self.job.dir = self.d
        self.job.thumb_text = True
        self.job.thumb_variant = -1
        self.job.aspect_ratio = "9:16 (Portrait Widescreen)"
        self.job.channel = {"audience": "general"}
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        self.plan = {"thumbnail_texts": ["One", "Two"], "thumbnail_cover": thumbcover.normalize(FULL)}
        self.no_logo = mock.patch.object(pipeline, "_logo_for", return_value=None)

    def test_the_cover_is_the_first_variant(self):
        with self.no_logo, mock.patch.object(config, "THUMB_COVER_ENABLED", True):
            pipeline._thumb_variants(self.job, self.plan)
        recs = self.plan["thumbnail_variants"]
        self.assertEqual(recs[0]["layout"], "cover", self.logs)
        self.assertEqual(recs[0]["text"], "Xiaomi-OCR-0 The 0.8B Model")
        self.assertEqual(len(recs), 3)                       # cover + one per line
        self.assertTrue((self.d / recs[0]["file"]).exists())

    def test_switched_off_there_is_no_cover(self):
        with self.no_logo, mock.patch.object(config, "THUMB_COVER_ENABLED", False):
            pipeline._thumb_variants(self.job, self.plan)
        self.assertNotIn("cover", [r["layout"] for r in self.plan["thumbnail_variants"]])

    def test_thumbnail_text_off_means_no_cover_either(self):
        self.job.thumb_text = False
        with self.no_logo, mock.patch.object(config, "THUMB_COVER_ENABLED", True):
            pipeline._thumb_variants(self.job, self.plan)
        self.assertEqual(self.plan["thumbnail_variants"], [])

    def test_cover_wanted_only_for_tall_non_kids_videos(self):
        with mock.patch.object(config, "THUMB_COVER_ENABLED", True):
            self.assertTrue(pipeline._cover_wanted(self.job))
            self.job.channel = {"audience": "kids"}
            self.assertFalse(pipeline._cover_wanted(self.job))
            self.job.channel = {}
            self.job.aspect_ratio = "16:9 (Widescreen)"
            self.assertFalse(pipeline._cover_wanted(self.job))
        self.job.aspect_ratio = "9:16 (Portrait Widescreen)"
        with mock.patch.object(config, "THUMB_COVER_ENABLED", False):
            self.assertFalse(pipeline._cover_wanted(self.job))


if __name__ == "__main__":
    unittest.main()
