"""Output-resolution upscale: run with  python -m unittest discover -s tests -t .  from the repo root.

Pure unit tests: the size maths, the per-video choice validation and the assembly wiring
(ffmpeg calls are mocked)."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-resolution-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from pipeline import Job, JobManager  # noqa: E402
from services import video  # noqa: E402


class TargetSize(unittest.TestCase):
    def test_landscape_keeps_the_real_clip_aspect(self):
        self.assertEqual(video.target_size(864, 480, 1440), (2592, 1440))
        self.assertEqual(video.target_size(864, 480, 1080), (1944, 1080))
        self.assertEqual(video.target_size(1344, 736, 1080), (1972, 1080))

    def test_portrait_and_square(self):
        self.assertEqual(video.target_size(480, 864, 1080), (1080, 1944))
        self.assertEqual(video.target_size(736, 736, 1440), (1440, 1440))

    def test_dimensions_are_even(self):
        w, h = video.target_size(900, 500, 1080)
        self.assertEqual((w % 2, h % 2), (0, 0))
        self.assertEqual(h, 1080)

    def test_never_downscales(self):
        self.assertIsNone(video.target_size(1344, 736, 720))
        self.assertIsNone(video.target_size(1920, 1080, 1080))
        self.assertIsNone(video.target_size(864, 480, 0))


class ResolveChoices(unittest.TestCase):
    def resolve(self, resolution):
        return JobManager._resolve_choices("", "", "", "", "", None, resolution=resolution)[-1]

    def test_every_option_is_accepted(self):
        for name in config.OUTPUT_RESOLUTIONS:
            self.assertEqual(self.resolve(name), name)

    def test_empty_falls_back_to_the_settings_default(self):
        with mock.patch.object(config, "DEFAULT_RESOLUTION", "2K (1440p)"):
            self.assertEqual(self.resolve(""), "2K (1440p)")
            self.assertEqual(self.resolve(None), "2K (1440p)")

    def test_unknown_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve("3K")


class Assemble(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def setUp(self):
        config.OUTPUT_DIR = TMP / "output"     # other test modules patch config at run time
        self.job = Job(id="20260101-000000-abc123", url="", clips=["clip_01.mp4", "clip_02.mp4"])
        self.job.dir.mkdir(parents=True, exist_ok=True)
        patches = {
            "concat_clips": mock.patch.object(pipeline.video, "concat_clips",
                                              side_effect=lambda clips, out: Path(out)),
            "clip_boundaries": mock.patch.object(pipeline.video, "clip_boundaries", return_value=[5.0, 10.0]),
            "frame_size": mock.patch.object(pipeline.video, "frame_size", return_value=(864, 480)),
            "upscale": mock.patch.object(pipeline.video, "upscale",
                                         side_effect=lambda v, out, w, h: Path(out)),
            "mix_audio": mock.patch.object(pipeline.video, "mix_audio",
                                           side_effect=lambda src, out, **kw: Path(out)),
        }
        self.m = {k: p.start() for k, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_2k_upscales_before_mixing(self):
        self.job.resolution = "2K (1440p)"
        final = pipeline._assemble(self.job, {}, None, None)
        d = self.job.dir
        self.m["upscale"].assert_called_once_with(d / "clips_joined.mp4", d / "clips_upscaled.mp4", 2592, 1440)
        self.assertEqual(self.m["mix_audio"].call_args.args[0], d / "clips_upscaled.mp4")
        self.assertEqual(final, d / "final.mp4")
        self.assertTrue(any("upscaled 864x480 -> 2592x1440" in e["msg"] if isinstance(e, dict) else
                            "upscaled 864x480 -> 2592x1440" in str(e) for e in self.job.log))

    def test_native_and_legacy_jobs_skip_the_upscale(self):
        for res in ("Native (clip size)", ""):
            self.m["upscale"].reset_mock(); self.m["mix_audio"].reset_mock()
            self.job.resolution = res
            pipeline._assemble(self.job, {}, None, None)
            self.m["upscale"].assert_not_called()
            self.assertEqual(self.m["mix_audio"].call_args.args[0], self.job.dir / "clips_joined.mp4")

    def test_clips_already_large_enough_are_left_alone(self):
        self.m["frame_size"].return_value = (1920, 1080)
        self.job.resolution = "1080p (Full HD)"
        pipeline._assemble(self.job, {}, None, None)
        self.m["upscale"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
