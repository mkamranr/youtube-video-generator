"""Per-stage job timing: run with  python -m unittest discover -s tests -t .  from the repo root.

Pure unit tests on `pipeline.Job` against a throw-away output directory; the clock is scripted so
active seconds can be asserted exactly."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-timing-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from pipeline import Job  # noqa: E402


class Clock:
    """Deterministic replacement for time.time(): every read advances by `step`."""
    def __init__(self, start=1_000.0, step=0.0):
        self.now, self.step = start, step

    def __call__(self):
        self.now += self.step
        return self.now


def _job(**kw):
    return Job(id="20260101-000000-abc123", url="", **kw)


class TimingsTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def setUp(self):
        config.OUTPUT_DIR = TMP / "output"     # other test modules patch config at run time
        self.clock = Clock()
        self._p = mock.patch("pipeline.time.time", self.clock)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def at(self, t):
        self.clock.now = t

    def test_full_run_records_each_stage(self):
        j = _job()
        self.at(100); j.set("planning", "plan")
        self.at(130); j.set("clips", "clips", clips_total=2, clips_done=0)
        self.at(190); j.set("assembling", "assemble")
        self.at(200); j.set("done", "finished")
        self.assertEqual(set(j.timings), {"planning", "clips", "assembling"})
        p, c, a = j.timings["planning"], j.timings["clips"], j.timings["assembling"]
        self.assertEqual((p["start"], p["end"], p["seconds"], p["running"]), (100, 130, 30.0, False))
        self.assertEqual((c["start"], c["end"], c["seconds"], c["running"]), (130, 190, 60.0, False))
        self.assertEqual((a["start"], a["end"], a["seconds"], a["running"]), (190, 200, 10.0, False))
        # persisted and reloaded intact
        again = Job.load(j.dir / "job.json")
        self.assertEqual(again.timings, j.timings)

    def test_park_and_resume_keeps_start_and_excludes_gap(self):
        j = _job()
        self.at(100); j.set("clips", "clips", clips_total=3)
        self.at(160); j.set("queued", "waiting for the GPU window")     # parked after 60 s
        c = j.timings["clips"]
        self.assertEqual((c["end"], c["seconds"], c["running"]), (160, 60.0, False))
        self.at(5000); j.set("clips", "clips")                           # window opened hours later
        self.assertTrue(c["running"])
        self.assertEqual(c["start"], 100)
        self.at(5040); j.set("assembling", "assemble")
        self.assertEqual((c["start"], c["end"], c["seconds"], c["running"]), (100, 5040, 100.0, False))

    def test_progress_updates_do_not_touch_timings(self):
        j = _job()
        self.at(100); j.set("clips", "clips")
        snap = dict(j.timings["clips"])
        self.at(120); j.set("clips", "clip 1/3: sampling 4/20")
        self.at(140); j.set(clips_done=1)
        self.assertEqual(j.timings["clips"], snap)

    def test_failure_closes_the_stage(self):
        j = _job()
        self.at(100); j.set("tts", "speak")
        self.at(107); j.set("failed", "boom", error="boom")
        self.assertEqual(j.timings["tts"]["end"], 107)
        self.assertFalse(j.timings["tts"]["running"])
        self.assertNotIn("failed", j.timings)

    def test_maintenance_is_timed_apart(self):
        j = _job(status="done", clips_total=3, clips_done=3,
                 timings={"clips": {"start": 100, "end": 400, "seconds": 300.0, "running": False, "last_start": 100}})
        j.regen_queue = [2]
        self.at(900); j.set("clips", "regenerating clip 2 (1/1)")
        self.assertEqual(j.timings["clips"]["end"], 400)          # original render untouched
        self.assertFalse(j.timings["clips"]["running"])
        self.assertTrue(j.timings["regen_clips"]["running"])
        j.regen_queue = []
        j.assemble_requested = True
        self.at(950); j.set("assembling", "re-merge")
        self.assertEqual(j.timings["regen_clips"]["seconds"], 50.0)
        self.assertTrue(j.timings["regen_assembling"]["running"])
        self.at(960); j.set("done", "finished", assemble_requested=False)
        self.assertEqual(j.timings["regen_assembling"]["seconds"], 10.0)
        self.assertEqual(j.timings["clips"]["seconds"], 300.0)

    def test_legacy_job_without_timings_loads_and_tracks(self):
        d = config.OUTPUT_DIR / "20250101-000000-legacy"
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps({"id": d.name, "url": "", "status": "queued",
                                                "created": 1.0, "updated": 1.0}), encoding="utf-8")
        j = Job.load(d / "job.json")
        self.assertEqual(j.timings, {})
        self.at(50); j.set("planning", "plan")
        self.assertEqual(j.timings["planning"]["start"], 50)

    def test_close_running_uses_given_time(self):
        """Restart path: the downtime between the last save and the restart is not active time."""
        j = _job()
        self.at(100); j.set("clips", "clips")
        self.at(130); j.set(clips_done=1)          # last save before the crash -> updated == 130
        self.at(99_999)
        j.close_running(j.updated)
        c = j.timings["clips"]
        self.assertEqual((c["end"], c["seconds"], c["running"]), (130, 30.0, False))
        j.set("queued", "resuming after server restart")   # nothing left running: no-op for timings
        self.assertEqual(c["seconds"], 30.0)


if __name__ == "__main__":
    unittest.main()
