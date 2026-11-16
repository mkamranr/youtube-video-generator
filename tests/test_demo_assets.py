"""Demo footage on a reel scene: a screen recording or screenshot replaces that scene's AI clips.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-demo-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import video  # noqa: E402
from services.ffbin import ffmpeg  # noqa: E402

HAVE_FFMPEG = bool(shutil.which(ffmpeg()) or Path(ffmpeg()).exists())
PORTRAIT = "9:16 (Portrait Widescreen)"
CHANNEL = {"name": "Daily Tech Talks", "audience": "general", "video_type": "tech",
           "cast": [], "characters_enabled": False, "reel_format": True}


def _ff(*args):
    subprocess.run([ffmpeg(), "-y", "-v", "error", *map(str, args)], check=True)


def _plan():
    return {"scenes": ["s1", "s2", "s3"], "clip_seconds": [2.0, 2.0, 2.0], "clip_scene": [0, 1, 2],
            "reel": {"scenes": [{"n": 1, "role": "hook", "voiceover": "a", "asset": ""},
                                {"n": 2, "role": "demo", "voiceover": "b", "asset": ""},
                                {"n": 3, "role": "cta", "voiceover": "c", "asset": ""}]}}


class Mapping(unittest.TestCase):

    def setUp(self):
        self.job = pipeline.Job(id="dm1", url="", aspect_ratio=PORTRAIT, audio_mode="narration+music",
                                source="topic", channel=dict(CHANNEL))
        shutil.rmtree(self.job.dir, ignore_errors=True)
        self.job.dir.mkdir(parents=True, exist_ok=True)

    def test_only_a_scene_with_a_file_on_disk_counts(self):
        plan = _plan()
        plan["reel"]["scenes"][1]["asset"] = "demo_02.png"
        self.assertIsNone(pipeline._asset_for_clip(self.job, plan, 2))     # not on disk yet
        (self.job.dir / "demo_02.png").write_bytes(b"x")
        self.assertEqual(pipeline._asset_for_clip(self.job, plan, 2), self.job.dir / "demo_02.png")
        self.assertIsNone(pipeline._asset_for_clip(self.job, plan, 1))
        self.job.channel = {**CHANNEL, "reel_format": False}
        self.assertIsNone(pipeline._asset_for_clip(self.job, plan, 2))

    def test_footage_clip_skips_the_gpu(self):
        plan = _plan()
        plan["reel"]["scenes"][1]["asset"] = "demo_02.png"
        (self.job.dir / "demo_02.png").write_bytes(b"x")
        client = mock.MagicMock()
        with mock.patch.object(pipeline, "_asset_clip") as ac:
            pipeline._generate_clip(self.job, client, {}, {}, plan, 2, self.job.dir / "clip_02.mp4", "c")
        ac.assert_called_once()
        client.build.assert_not_called()


class Upload(unittest.TestCase):

    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        self.mgr = pipeline.JobManager()
        self.job = pipeline.Job(id="dm2", url="", aspect_ratio=PORTRAIT, audio_mode="narration+music",
                                source="topic", channel=dict(CHANNEL), status="awaiting_approval")
        self.job.dir.mkdir(parents=True, exist_ok=True)
        (self.job.dir / "plan.json").write_text(json.dumps(_plan()), encoding="utf-8")
        self.mgr.jobs[self.job.id] = self.job

    def _plan(self):
        return json.loads((self.job.dir / "plan.json").read_text(encoding="utf-8"))

    def test_attach_and_remove(self):
        self.mgr.set_scene_asset(self.job.id, 2, b"png-bytes", "screen.PNG")
        self.assertEqual(self._plan()["reel"]["scenes"][1]["asset"], "demo_02.png")
        self.assertTrue((self.job.dir / "demo_02.png").exists())
        self.mgr.set_scene_asset(self.job.id, 2, None)
        self.assertEqual(self._plan()["reel"]["scenes"][1]["asset"], "")
        self.assertFalse((self.job.dir / "demo_02.png").exists())

    def test_rejects_bad_type_and_scene(self):
        with self.assertRaises(ValueError):
            self.mgr.set_scene_asset(self.job.id, 2, b"x", "notes.txt")
        with self.assertRaises(ValueError):
            self.mgr.set_scene_asset(self.job.id, 9, b"x", "a.png")

    def test_finished_video_marks_that_scenes_clips_stale(self):
        self.job.status = "done"
        self.job.publish_approved = True
        self.mgr.set_scene_asset(self.job.id, 2, b"x", "a.mp4")
        self.assertEqual(self.job.stale_clips, [2])
        self.assertFalse(self.job.publish_approved)

    def test_not_while_rendering(self):
        self.job.status = "clips"
        with self.assertRaises(ValueError):
            self.mgr.set_scene_asset(self.job.id, 2, b"x", "a.png")


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class Encode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp(prefix="videogen-demo-enc-"))
        cls.ref = cls.d / "ref.mp4"
        _ff("-f", "lavfi", "-i", "testsrc=s=352x624:r=24:d=1", "-f", "lavfi", "-i",
            "anullsrc=r=44100:cl=stereo", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", cls.ref)

    def _check(self, src):
        shape = video.stream_shape(self.ref)
        out = self.d / (src.stem + "_clip.mp4")
        video.clip_from_asset(src, out, 2.0, shape)
        got = video.stream_shape(out)
        for k in ("codec", "width", "height", "fps", "timescale", "audio"):
            self.assertEqual(got[k], shape[k], k)
        self.assertAlmostEqual(video.duration(out), 2.0, delta=0.15)
        return out

    def test_screenshot_becomes_a_matching_clip(self):
        png = self.d / "shot.png"
        _ff("-f", "lavfi", "-i", "testsrc=s=1280x720", "-frames:v", "1", png)
        self._check(png)

    def test_short_recording_loops_to_length(self):
        rec = self.d / "rec.mp4"
        _ff("-f", "lavfi", "-i", "testsrc=s=640x360:r=30:d=0.7", "-c:v", "libx264", rec)
        out = self._check(rec)
        joined = video.concat_clips([self.ref, out, self.ref], self.d / "joined.mp4")
        self.assertAlmostEqual(video.duration(joined), 4.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
