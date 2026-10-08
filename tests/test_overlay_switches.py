"""Who decides whether the burned-in captions and the thumbnail text appear.

Three levels, and all three were broken in different ways before these tests existed:

  global   config.CAPTIONS_ENABLED / config.THUMB_TEXT_ENABLED are MASTER switches. They were
           defined, listed in EDITABLE_SETTINGS and drawn on the Settings page - and read by
           nothing at all, so turning captions off there did nothing whatsoever.
  channel  defaults.captions / .polish existed in the record but the channel editor never sent
           them, and defaults.thumb_text did not exist.
  video    create_prompt_job resolved captions and polish and then dropped them on the floor,
           so every topic/description/transcript/repo video was assembled with both off.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-switches-"))
config.OUTPUT_DIR = TMP / "output"
config.CHANNELS_DIR = TMP / "channels"
config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"

from services import channels  # noqa: E402
import pipeline  # noqa: E402


class ChannelDefaults(unittest.TestCase):
    """The per-channel layer: four flags now, and they do not all mean the same thing."""

    def test_the_overlays_default_on_for_tech_channels_only(self):
        for vt, want in (("tech", True), ("animated", False), ("real", False)):
            rec = channels._validate({"name": "C", "video_type": vt})
            for key in ("storyboard", "captions", "polish"):
                self.assertEqual(rec["defaults"][key], want, f"{key} on {vt}")

    def test_thumbnail_text_defaults_on_for_every_channel(self):
        """It is drawn on the thumbnail IMAGE, not the video, and always has been on every
        channel - deriving it from video_type would silently switch it off for most of them."""
        for vt in ("tech", "animated", "real"):
            rec = channels._validate({"name": "C", "video_type": vt})
            self.assertTrue(rec["defaults"]["thumb_text"], vt)

    def test_an_explicit_choice_survives_validation(self):
        rec = channels._validate({"name": "C", "video_type": "tech",
                                  "defaults": {"captions": False, "thumb_text": False}})
        self.assertFalse(rec["defaults"]["captions"])
        self.assertFalse(rec["defaults"]["thumb_text"])
        self.assertTrue(rec["defaults"]["storyboard"])      # untouched

    def test_captions_off_is_not_reset_by_a_later_save(self):
        """The regression that hid the bug: the channel editor never sent captions, so every
        save re-derived it from video_type and a tech channel could never keep it off."""
        rec = channels._validate({"name": "C", "video_type": "tech",
                                  "defaults": {"captions": False}})
        again = channels._validate(rec)
        self.assertFalse(again["defaults"]["captions"])


class ResolveFlags(unittest.TestCase):
    """request value -> channel default -> the flag's own default."""

    @staticmethod
    def _ch(video_type="tech", **defaults):
        """A real validated record, not a hand-built dict: _resolve_choices snapshots it."""
        rec = channels._validate({"name": "C", "video_type": video_type,
                                  "defaults": dict(defaults)})
        rec["id"] = "c1"
        return rec

    def _resolve(self, channel, captions_flag=None):
        out = pipeline.JobManager._resolve_choices(
            "9:16 (Portrait Widescreen)", "", "Auto", "narration+music", "", channel,
            None, None, "", "",
            captions_flag, None, None)
        return out[9]          # captions sits at index 9; [-1] is the resolution

    def test_the_request_beats_the_channel(self):
        ch = self._ch(captions=True)
        self.assertFalse(self._resolve(ch, captions_flag=False))

    def test_a_false_channel_default_is_honoured_not_treated_as_unset(self):
        ch = self._ch(captions=False)
        self.assertFalse(self._resolve(ch))

    def test_unset_falls_back_to_the_channel_type(self):
        self.assertTrue(self._resolve(self._ch("tech")))
        self.assertFalse(self._resolve(self._ch("animated")))

    def test_thumbnail_text_resolves_on_when_nothing_says_otherwise(self):
        self.assertTrue(pipeline.JobManager._resolve_thumb_text(None, self._ch("tech")))
        self.assertTrue(pipeline.JobManager._resolve_thumb_text(None, self._ch("animated")))

    def test_thumbnail_text_request_and_channel_layers(self):
        self.assertFalse(pipeline.JobManager._resolve_thumb_text(False, self._ch("tech")))
        ch = self._ch(thumb_text=False)
        self.assertFalse(pipeline.JobManager._resolve_thumb_text(None, ch))
        self.assertTrue(pipeline.JobManager._resolve_thumb_text(True, ch))


class MasterSwitches(unittest.TestCase):
    """Off means off, whatever the job chose - and it is read at render time, so reassembling
    an existing video strips what the switch now forbids."""

    def setUp(self):
        self.job = mock.MagicMock()
        self.job.dir = TMP / "job"
        self.job.dir.mkdir(parents=True, exist_ok=True)
        self.job.captions = True
        self.job.thumb_text = True
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        self.plan = {"caption_words": {"words": [{"w": "hi", "start": 0.0, "end": 0.4}]}}

    def test_captions_master_off_writes_no_track(self):
        with mock.patch.object(config, "CAPTIONS_ENABLED", False):
            self.assertIsNone(pipeline._caption_track(self.job, self.plan, (1080, 1920), [], 10.0))

    def test_captions_master_on_still_honours_the_video(self):
        self.job.captions = False
        with mock.patch.object(config, "CAPTIONS_ENABLED", True):
            self.assertIsNone(pipeline._caption_track(self.job, self.plan, (1080, 1920), [], 10.0))

    def test_thumbnail_text_master_off_builds_no_variants(self):
        with mock.patch.object(config, "THUMB_TEXT_ENABLED", False), \
             mock.patch.object(pipeline.thumbtext, "variants") as v, \
             mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            (self.job.dir / "thumb_base.jpg").write_bytes(b"x")
            plan = {}
            pipeline._thumb_variants(self.job, plan)
        v.assert_not_called()
        self.assertEqual(plan["thumbnail_variants"], [])

    def test_thumbnail_text_off_for_this_video_builds_no_variants(self):
        self.job.thumb_text = False
        with mock.patch.object(config, "THUMB_TEXT_ENABLED", True), \
             mock.patch.object(pipeline.thumbtext, "variants") as v, \
             mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            (self.job.dir / "thumb_base.jpg").write_bytes(b"x")
            pipeline._thumb_variants(self.job, {})
        v.assert_not_called()


class PresenterThumbnail(unittest.TestCase):
    """A video with a presenter offers its thumbnail both ways: with the face and without."""

    def setUp(self):
        from PIL import Image
        self.d = Path(tempfile.mkdtemp(prefix="job-", dir=TMP))
        self.base = self.d / "thumb_base.jpg"
        Image.new("RGB", (720, 1280), (30, 40, 60)).save(self.base, "JPEG")
        (self.d / "avatar_01.mp4").write_bytes(b"clip")
        self.job = mock.MagicMock()
        self.job.dir = self.d
        self.job.thumb_text = True
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        self.plan = {"avatar": {"segments": [{"file": "avatar_01.mp4", "matte": ""}]},
                     "thumbnail_texts": ["One", "Two"]}

    def _patched(self):
        """thumbnail_cutout and with_presenter are exercised for real elsewhere; here the point
        is the wiring, so they just produce a file."""
        def _cut(clip, dst, matte=None, at=0.4):
            Path(dst).write_bytes(b"png")
            return Path(dst)

        def _comp(base, cut, out, style="", corner=""):
            from PIL import Image
            Image.new("RGB", (720, 1280), (90, 20, 20)).save(out, "JPEG")
            return Path(out)
        return (mock.patch.object(pipeline.avatar, "thumbnail_cutout", side_effect=_cut),
                mock.patch.object(pipeline.thumbtext, "with_presenter", side_effect=_comp))

    def test_no_presenter_means_one_base_as_before(self):
        self.plan["avatar"] = {}
        with mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            pipeline._thumb_variants(self.job, self.plan)
        self.assertFalse(any(r.get("presenter") for r in self.plan["thumbnail_variants"]))
        self.assertFalse((self.d / "thumb_base_presenter.jpg").exists())

    def test_a_presenter_adds_a_second_base_and_its_own_text_free_choice(self):
        c, w = self._patched()
        with c, w, mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            pipeline._thumb_variants(self.job, self.plan)
        recs = self.plan["thumbnail_variants"]
        pres = [r for r in recs if r.get("presenter")]
        self.assertTrue((self.d / "thumb_base_presenter.jpg").exists())
        # the presenter base itself, plus one per text
        self.assertEqual(len(pres), 1 + len(self.plan["thumbnail_texts"]))
        self.assertEqual(pres[0]["file"], "thumb_base_presenter.jpg")
        self.assertEqual(pres[0]["text"], "")

    def test_the_two_sets_do_not_overwrite_each_others_files(self):
        c, w = self._patched()
        with c, w, mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            pipeline._thumb_variants(self.job, self.plan)
        files = [r["file"] for r in self.plan["thumbnail_variants"]]
        self.assertEqual(len(files), len(set(files)), files)

    def test_a_failed_presenter_thumbnail_is_logged_and_not_fatal(self):
        with mock.patch.object(pipeline.avatar, "thumbnail_cutout", side_effect=RuntimeError("boom")),              mock.patch.object(pipeline.thumbtext, "activate", return_value=0):
            pipeline._thumb_variants(self.job, self.plan)
        self.assertTrue(any("presenter thumbnail skipped" in m for m in self.logs))
        self.assertFalse(any(r.get("presenter") for r in self.plan["thumbnail_variants"]))


if __name__ == "__main__":
    unittest.main()
