"""Automated checks of a finished video (services/qa.py) and the publish approval gate.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import json
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-qa-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import qa  # noqa: E402
from services.ffbin import ffmpeg  # noqa: E402

HAVE_FFMPEG = bool(shutil.which(ffmpeg()) or Path(ffmpeg()).exists())
PORTRAIT = "9:16 (Portrait Widescreen)"


def _ff(*args):
    subprocess.run([ffmpeg(), "-y", "-v", "error", *map(str, args)], check=True)


def _job(d: Path, final="final.mp4", aspect=PORTRAIT):
    return types.SimpleNamespace(dir=d, final=final, aspect_ratio=aspect, clips=[], captions=False,
                                 audio_mode="narration+music")


def _by_id(res):
    return {c["id"]: c for c in res["checks"]}


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class Checks(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="videogen-qa-case-"))

    def _make(self, video_src, audio_src, seconds=6, size="360x640", name="final.mp4"):
        out = self.d / name
        _ff("-f", "lavfi", "-i", f"{video_src}=s={size}:r=24:d={seconds}",
            "-f", "lavfi", "-i", audio_src, "-t", seconds, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out)
        return out

    def test_a_healthy_video_passes_the_media_checks(self):
        self._make("testsrc2", "sine=f=330:r=44100:d=6,volume=0.3")
        res = qa.run(_job(self.d), {"clip_seconds": [3.0, 3.0]})
        c = _by_id(res)
        for k in ("aspect", "first_frame", "black", "static", "edge_silence", "clipping", "duration", "assets"):
            self.assertEqual(c[k]["status"], "pass", (k, c[k]))
        self.assertEqual(c["review"]["status"], "manual")
        self.assertEqual(c["resolution"]["status"], "warn")      # 360x640 is below 1080x1920

    def test_black_opening_and_dead_air_are_flagged(self):
        out = self.d / "final.mp4"
        _ff("-f", "lavfi", "-i", "color=c=black:s=360x640:r=24:d=1.5",
            "-f", "lavfi", "-i", "testsrc2=s=360x640:r=24:d=4.5",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1.5",
            "-f", "lavfi", "-i", "sine=f=330:r=44100:d=4.5,volume=0.3",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a][3:a]concat=n=2:v=0:a=1[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
        c = _by_id(qa.run(_job(self.d), {"clip_seconds": [6.0]}))
        self.assertEqual(c["first_frame"]["status"], "fail")
        self.assertEqual(c["black"]["status"], "warn")
        self.assertEqual(c["edge_silence"]["status"], "warn")

    def test_frozen_picture_wrong_aspect_and_clipping(self):
        out = self.d / "final.mp4"
        _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=1", "-f", "lavfi",
            "-i", "sine=f=330:r=44100:d=6,volume=8", "-vf", "tpad=stop_mode=clone:stop_duration=5",
            "-t", 6, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
        with mock.patch.object(config, "QA_FREEZE_SECONDS", 2.0):
            c = _by_id(qa.run(_job(self.d), {"clip_seconds": [9.0]}))
        self.assertEqual(c["static"]["status"], "warn")
        self.assertEqual(c["aspect"]["status"], "fail")
        self.assertIn(c["clipping"]["status"], ("warn", "fail"))
        self.assertEqual(c["duration"]["status"], "warn")

    def test_speech_running_to_the_last_frame_is_a_cut_off(self):
        self._make("testsrc2", "sine=f=330:r=44100:d=6,volume=0.3")
        _ff("-f", "lavfi", "-i", "sine=f=220:r=44100:d=6", self.d / "narration.wav")
        c = _by_id(qa.run(_job(self.d), {}))
        self.assertEqual(c["end_cut"]["status"], "fail")

    def test_missing_final_and_missing_clip(self):
        job = _job(self.d)
        self.assertEqual(_by_id(qa.run(job, {}))["file"]["status"], "fail")
        self._make("testsrc2", "sine=f=330:r=44100:d=6,volume=0.3")
        job.clips = ["clip_01.mp4"]
        self.assertEqual(_by_id(qa.run(job, {}))["assets"]["status"], "fail")


class CaptionOverflow(unittest.TestCase):

    def test_a_word_too_wide_for_the_area_is_flagged(self):
        words = [{"word": "Supercalifragilisticexpialidociousness-and-more", "start": 0.0, "end": 1.0},
                 {"word": "ok", "start": 1.0, "end": 1.5}]
        n, bad = qa.caption_overflow({"caption_words": {"words": words}}, (360, 640), None)
        self.assertGreaterEqual(n, 1)
        self.assertTrue(bad)


class PublishGate(unittest.TestCase):

    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        self.mgr = pipeline.JobManager()
        self.job = pipeline.Job(id="pub1", url="", aspect_ratio=PORTRAIT, audio_mode="narration+music",
                                source="topic", status="done", final="final.mp4",
                                channel={"name": "Daily Tech Talks", "reel_format": True,
                                         "video_type": "tech", "audience": "general"})
        self.job.dir.mkdir(parents=True, exist_ok=True)
        (self.job.dir / "final.mp4").write_bytes(b"x" * 2000)
        (self.job.dir / "plan.json").write_text(json.dumps({}), encoding="utf-8")
        self.mgr.jobs[self.job.id] = self.job

    def test_upload_is_locked_until_approved(self):
        with self.assertRaises(ValueError) as e:
            self.mgr.start_youtube_upload(self.job.id)
        self.assertIn("approve", str(e.exception))
        self.mgr.approve_publish(self.job.id, True)
        self.assertTrue(self.job.publish_approved)
        with mock.patch("services.youtube.has_secret", return_value=False):
            with self.assertRaises(ValueError) as e:      # past the gate, stopped by the next check
                self.mgr.start_youtube_upload(self.job.id)
        self.assertIn("client_secret", str(e.exception))

    def test_cannot_approve_unassembled_changes(self):
        self.job.needs_assemble = True
        with self.assertRaises(ValueError):
            self.mgr.approve_publish(self.job.id, True)

    def test_reassembly_clears_the_approval(self):
        self.job.publish_approved = True
        with mock.patch.object(pipeline, "_assemble", return_value=self.job.dir / "final.mp4"), \
             mock.patch.object(pipeline, "_drop_upscales"), \
             mock.patch.object(pipeline, "_run_qa") as rq:
            pipeline._do_assemble(self.job)
        self.assertFalse(self.job.publish_approved)
        rq.assert_called_once()

    def test_non_reel_upload_is_not_gated(self):
        self.job.channel = {**self.job.channel, "reel_format": False}
        with mock.patch("services.youtube.has_secret", return_value=False):
            with self.assertRaises(ValueError) as e:
                self.mgr.start_youtube_upload(self.job.id)
        self.assertIn("client_secret", str(e.exception))


if __name__ == "__main__":
    unittest.main()
