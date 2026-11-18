"""The modern thumbnail (services/thumbmodern.py) and the letterbox fill (services/video.py).

Pillow only - no GPU, no LLM call, no network. Run with
python -m unittest discover -s tests -t .  from the repo root.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops, ImageDraw

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-modern-"))
config.OUTPUT_DIR = TMP / "output"

from services import llm, thumbcover, thumbmodern, video  # noqa: E402  (config patched first)
import pipeline  # noqa: E402

FULL = {
    "title": "Local AI Coder", "title_icon": "spark", "strip": "Run Offline.", "line": "Air-Gapped",
    "split": ["vLLM", "Setup Guide"],
    "badge": {"icon": "wifi_off", "text": "No Internet Needed"},
    "cards": [{"icon": "shield", "text": "Secure & Local"}, {"icon": "gear", "text": "Go Bridge Integration"},
              {"icon": "check", "text": "Verify SHA256"}, {"icon": "terminal", "text": "Terminal & Desktop App"}],
    "banner": ["Self-Hosted AI Coding", "in Restricted Networks!"], "banner_icon": "rocket",
    "tagline": "AI • Tech • Innovation",
}


def make_base(path: Path, size=(720, 1280)) -> Path:
    im = Image.new("RGB", size, (30, 40, 70))
    d = ImageDraw.Draw(im)
    for y in range(size[1]):
        d.line([(0, y), (size[0], y)], fill=(30 + y // 20, 40 + y // 30, 70 + y // 15))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
    return path


def make_letterboxed(size=(720, 1280), top=420, bottom=400) -> Image.Image:
    im = Image.new("RGB", size, (20, 24, 30))
    d = ImageDraw.Draw(im)
    for y in range(top, size[1] - bottom):
        d.line([(0, y), (size[0], y)], fill=(80 + y % 90, 120, 200 - y % 70))
    d.ellipse((200, top + 60, 520, size[1] - bottom - 60), fill=(240, 200, 60))
    return im


class Normalize(unittest.TestCase):

    def test_full_copy_is_cleaned(self):
        c = thumbmodern.normalize(FULL)
        self.assertEqual(c["strip"], "Run Offline")                 # full stop dropped
        self.assertEqual(c["split"], ["vLLM", "Setup Guide"])
        self.assertEqual(len(c["cards"]), 4)
        self.assertEqual(c["cards"][3]["text"], "Terminal & Desktop App")
        self.assertEqual(c["tagline"], "AI • Tech • Innovation")
        self.assertEqual(c["badge"], {"icon": "wifi_off", "text": "No Internet Needed"})

    def test_unknown_icons_fall_back(self):
        c = thumbmodern.normalize({"title": "X", "cards": [{"icon": "unicorn", "text": "A"},
                                                          {"icon": "Wifi-Off", "text": "B"}],
                                   "badge": {"icon": "??", "text": "Fast"}})
        self.assertEqual([k["icon"] for k in c["cards"]], ["check", "wifi_off"])
        self.assertEqual(c["badge"]["icon"], "bolt")

    def test_a_single_card_is_dropped(self):
        self.assertEqual(thumbmodern.normalize({"title": "X", "cards": [{"icon": "gear", "text": "A"}]})["cards"], [])

    def test_without_title_or_strip_there_is_no_poster(self):
        self.assertIsNone(thumbmodern.normalize({"line": "x", "banner": ["y"]}))
        self.assertIsNone(thumbmodern.normalize("nope"))

    def test_a_split_string_is_accepted(self):
        self.assertEqual(thumbmodern.normalize({"title": "X", "split": "vLLM|Guide"})["split"], ["vLLM", "Guide"])


class Render(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="render-", dir=TMP))
        self.base = make_base(self.d / "base.jpg")

    def test_keeps_the_size_and_changes_the_picture(self):
        out = thumbmodern.render(self.base, self.d / "out.jpg", FULL, brand="Daily Tech Talks")
        with Image.open(out) as a, Image.open(self.base) as b:
            self.assertEqual(a.size, b.size)
            self.assertIsNotNone(ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox())

    def test_minimal_copy_renders(self):
        thumbmodern.render(self.base, self.d / "min.jpg", {"strip": "Just This"})

    def test_a_logo_goes_in_the_header(self):
        logo = self.d / "logo.png"
        Image.new("RGBA", (200, 200), (255, 0, 0, 255)).save(logo)
        out = thumbmodern.render(self.base, self.d / "logo.jpg", FULL, brand="X", logo=logo)
        with Image.open(out) as im:
            r, g, b = im.convert("RGB").getpixel((int(im.width * 0.1), int(im.height * 0.06)))
        self.assertGreater(r, 180)
        self.assertLess(g, 90)

    def test_every_icon_draws(self):
        for name in thumbmodern.ICONS:
            im = Image.new("RGBA", (120, 120))
            thumbmodern.draw_icon(ImageDraw.Draw(im), name, 60, 60, 90)
            self.assertIsNotNone(im.getbbox(), name)

    def test_a_landscape_base_is_refused(self):
        wide = make_base(self.d / "wide.jpg", size=(1280, 720))
        with self.assertRaises(ValueError):
            thumbmodern.render(wide, self.d / "w.jpg", FULL)


class Letterbox(unittest.TestCase):

    def test_bars_are_measured(self):
        top, bottom = video.letterbox_rows(make_letterboxed())
        self.assertEqual((top, bottom), (420, 400))

    def test_bars_are_filled(self):
        im = make_letterboxed()
        out, changed = video.fill_letterbox(im)
        self.assertTrue(changed)
        self.assertEqual(out.size, im.size)
        self.assertEqual(video.letterbox_rows(out), (0, 0))

    def test_a_full_frame_is_left_alone(self):
        im = Image.open(make_base(TMP / "full.jpg"))
        out, changed = video.fill_letterbox(im.convert("RGB"))
        self.assertFalse(changed)

    def test_a_thin_dark_edge_is_not_a_bar(self):
        out, changed = video.fill_letterbox(make_letterboxed(top=40, bottom=30))
        self.assertFalse(changed)

    def test_the_logo_corner_does_not_hide_a_bar(self):
        im = make_letterboxed()
        ImageDraw.Draw(im).ellipse((600, 10, 710, 120), fill=(30, 120, 220))
        self.assertEqual(video.letterbox_rows(im)[0], 420)


class LlmField(unittest.TestCase):

    def test_asked_and_parsed(self):
        prompts = []

        def grab(msgs, **kw):
            prompts.append(msgs[-1]["content"])
            return {"thumbnail_modern_prompt": "a laptop and a server", "thumbnail_modern": FULL}
        with mock.patch.object(llm, "_chat", side_effect=grab):
            res = llm.write_thumbnail_modern("T", "narration", "style")
        self.assertIn('"thumbnail_modern"', prompts[0])
        self.assertIn("no letterbox", prompts[0])
        self.assertEqual(res["thumbnail_modern"]["strip"], "Run Offline")

    def test_unusable_answer_is_an_error(self):
        with mock.patch.object(llm, "_chat", return_value={"thumbnail_modern_prompt": "p"}):
            with self.assertRaises(RuntimeError):
                llm.write_thumbnail_modern("T", "n", "s")


class PipelineWiring(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="job-", dir=TMP))
        make_base(self.d / "thumb_base.jpg")
        make_base(self.d / pipeline.MODERN_BASE)
        self.job = mock.MagicMock()
        self.job.dir = self.d
        self.job.thumb_text = True
        self.job.thumb_variant = -1
        self.job.aspect_ratio = "9:16 (Portrait Widescreen)"
        self.job.channel = {"audience": "general", "name": "Daily Tech Talks"}
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        self.job.set.side_effect = lambda *a, **kw: setattr(self.job, "thumb_variant",
                                                            kw.get("thumb_variant", self.job.thumb_variant))
        self.plan = {"thumbnail_texts": ["One", "Two"], "thumbnail_modern": thumbmodern.normalize(FULL)}
        self.no_logo = mock.patch.object(pipeline, "_logo_for", return_value=None)

    def test_modern_is_added_last_and_selected_on_request(self):
        with self.no_logo, mock.patch.object(config, "THUMB_COVER_ENABLED", False):
            k = pipeline._thumb_variants(self.job, self.plan, select=pipeline.MODERN)
        recs = self.plan["thumbnail_variants"]
        self.assertEqual(recs[-1]["layout"], "modern", self.logs)
        self.assertEqual(len(recs), 3)                     # two classic lines + the poster
        self.assertEqual(k, 3)
        self.assertTrue((self.d / recs[-1]["file"]).exists())

    def test_modern_stays_active_when_the_classic_lines_change(self):
        with self.no_logo, mock.patch.object(config, "THUMB_COVER_ENABLED", False):
            pipeline._thumb_variants(self.job, self.plan, select=pipeline.MODERN)
            self.plan["thumbnail_texts"] = ["One", "Two", "Three"]
            k = pipeline._thumb_variants(self.job, self.plan)
        self.assertEqual(self.plan["thumbnail_variants"][k - 1]["layout"], "modern")

    def test_modern_is_drawn_even_with_thumbnail_text_off(self):
        self.job.thumb_text = False
        with self.no_logo:
            pipeline._thumb_variants(self.job, self.plan)
        self.assertEqual([r["layout"] for r in self.plan["thumbnail_variants"]], ["modern"])

    def test_a_modern_only_job_works_without_a_classic_base(self):
        (self.d / "thumb_base.jpg").unlink()
        with self.no_logo:
            k = pipeline._thumb_variants(self.job, self.plan, select=pipeline.MODERN)
        self.assertEqual(k, 1)
        self.assertTrue((self.d / "thumbnail.jpg").exists())
        self.assertFalse((self.d / "thumb_base.jpg").exists())

    def test_a_letterboxed_base_is_repaired(self):
        make_letterboxed().save(self.d / "thumb_base.jpg", "JPEG", quality=92)
        with self.no_logo:
            pipeline._thumb_variants(self.job, self.plan)
        with Image.open(self.d / "thumb_base.jpg") as im:
            self.assertEqual(video.letterbox_rows(im.convert("RGB")), (0, 0))

    def test_modern_needs_a_portrait_video(self):
        mgr = pipeline.JobManager.__new__(pipeline.JobManager)
        mgr.lock = mock.MagicMock()
        self.job.aspect_ratio = "16:9 (Widescreen)"
        with mock.patch.object(mgr, "_maintenance_target", create=True, return_value=self.job):
            with self.assertRaises(ValueError):
                mgr.regen_thumbnail("x", style="modern")
            with self.assertRaises(ValueError):
                mgr.regen_thumbnail("x", style="fancy")


if __name__ == "__main__":
    unittest.main()
