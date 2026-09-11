"""Jobs created from a weekly plan: they wait for their setup and never start by themselves.

Run with  python -m unittest discover -s tests -t .  from the repo root. `config.OUTPUT_DIR` is
patched to a throw-away directory before pipeline is imported, and `_start_draft` is always
mocked so no LLM thread is ever launched."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-planned-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)

CHANNEL = {"id": "ch01", "name": "Bright Minds Town", "purpose": "", "rules": "", "owner": "u1",
           "audience": "kids", "video_type": "animated", "cast": [], "cast_opening": True,
           "defaults": {"art_style": "", "audio_mode": "narration+music", "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": config.DEFAULT_VIDEO_LENGTH, "storyboard": False,
                        "song_voice": "child"}}

PLAN = {"plan_id": "p1", "day": 2, "date": "2026-09-16",
        "hook": "Whose kite is this?", "thumb_idea": "Finn holding a red kite."}


def _manager():
    return pipeline.JobManager()


def _planned(mgr, **kw):
    return mgr.submit_prompt("Finn walks to the meadow and finds a lost kite.", "description",
                             channel=CHANNEL, owner="u1", start=False,
                             title="The Kite in the Meadow", plan=PLAN, **kw)


class PlannedJobs(unittest.TestCase):
    def setUp(self):
        # other test modules patch config.OUTPUT_DIR too, and the paths are read at call time:
        # claim it for the duration of this test and hand it back afterwards
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        patcher = mock.patch.object(pipeline.JobManager, "_start_draft")
        self.draft = patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = _manager()

    def test_a_planned_job_waits_and_starts_nothing(self):
        job = _planned(self.mgr)
        self.assertEqual(job.status, "planned")
        self.assertEqual(self.mgr.queue, [])
        self.draft.assert_not_called()
        self.assertEqual(job.title, "The Kite in the Meadow")
        self.assertEqual(job.hook, "Whose kite is this?")   # carried for the script writer
        self.assertEqual((job.plan_id, job.planned_day, job.planned_for), ("p1", 2, "2026-09-16"))
        self.assertEqual(job.thumb_idea, "Finn holding a red kite.")
        # the channel's defaults are resolved now, so the setup form opens prefilled
        self.assertEqual(job.aspect_ratio, config.DEFAULT_ASPECT_RATIO)
        self.assertEqual(job.audio_mode, "narration+music")
        self.assertEqual(job.channel["name"], "Bright Minds Town")
        self.assertTrue(job.megapixels and job.resolution)

    def test_a_restart_does_not_requeue_it(self):
        job = _planned(self.mgr)
        again = _manager()                                   # a second start over the same output dir
        self.assertEqual(again.queue, [])
        self.assertEqual(again.jobs[job.id].status, "planned")
        self.draft.assert_not_called()

    def test_setup_applies_the_choices_and_drafts(self):
        job = _planned(self.mgr)
        out = self.mgr.setup_planned(job.id, aspect_ratio="9:16 (Portrait Widescreen)",
                                     video_length="1-2 minutes", audio_mode="music",
                                     megapixels=0.4, resolution="2K (1440p)")
        self.assertEqual(out.status, "drafting")
        self.assertTrue(out.setup_done)
        self.assertEqual(out.aspect_ratio, "9:16 (Portrait Widescreen)")
        self.assertEqual(out.video_length, "1-2 minutes")
        self.assertEqual(out.audio_mode, "music")
        self.assertEqual(out.voice_id, "")                   # a song has no reference voice
        self.assertEqual((out.megapixels, out.resolution), (0.4, "2K (1440p)"))
        self.draft.assert_called_once()
        self.assertEqual(pipeline.Job.load(job.dir / "job.json").status, "drafting")

    def test_blank_fields_keep_what_the_job_had(self):
        job = _planned(self.mgr)
        before = (job.aspect_ratio, job.video_length, job.audio_mode, job.megapixels,
                  job.resolution, job.song_voice, job.storyboard)
        out = self.mgr.setup_planned(job.id)
        self.assertEqual((out.aspect_ratio, out.video_length, out.audio_mode, out.megapixels,
                          out.resolution, out.song_voice, out.storyboard), before)

    def test_a_bad_value_leaves_the_job_untouched(self):
        job = _planned(self.mgr)
        with self.assertRaises(ValueError):
            self.mgr.setup_planned(job.id, aspect_ratio="banana")
        self.assertEqual(self.mgr.jobs[job.id].status, "planned")
        self.draft.assert_not_called()

    def test_setup_runs_once(self):
        job = _planned(self.mgr)
        self.mgr.setup_planned(job.id)
        with self.assertRaises(ValueError):
            self.mgr.setup_planned(job.id)
        with self.assertRaises(KeyError):
            self.mgr.setup_planned("no-such-job")

    def test_setup_survives_a_deleted_channel(self):
        job = _planned(self.mgr)
        with mock.patch.object(pipeline.channels, "get_channel", return_value=None):
            out = self.mgr.setup_planned(job.id, video_length="1-2 minutes")
        self.assertEqual(out.status, "drafting")
        self.assertEqual(out.channel_id, "ch01")             # the snapshot from plan time is kept
        self.assertEqual(out.channel["name"], "Bright Minds Town")

    def test_discard_then_restore(self):
        job = _planned(self.mgr)
        self.mgr.cancel(job.id)
        self.assertEqual(job.status, "failed")
        self.assertFalse(job.cancel_requested)               # no latent flag to kill a later draft
        self.mgr.resume(job.id)
        self.assertEqual(job.status, "planned")
        self.assertEqual(self.mgr.queue, [])
        self.draft.assert_not_called()

    def test_deletable_but_not_archivable(self):
        job = _planned(self.mgr)
        with self.assertRaises(ValueError):
            self.mgr.set_archived(job.id, True)
        self.mgr.delete(job.id)
        self.assertNotIn(job.id, self.mgr.jobs)

    def test_a_normal_prompt_job_still_drafts_at_once(self):
        job = self.mgr.submit_prompt("How bees make honey", "topic", channel=CHANNEL, owner="u1")
        self.assertEqual(job.status, "drafting")
        self.assertEqual(job.plan_id, "")
        self.draft.assert_called_once()


if __name__ == "__main__":
    unittest.main()
