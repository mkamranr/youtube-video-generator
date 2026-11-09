"""How a character's portrait reaches clip 1, and the cut trimmed off a seeded clip's start.

"Reference only" exists because a fitness channel's portrait is a likeness sheet, not a shot: the
video used to open on the crossed-arm reference pose. In that mode clip 1 starts from the last frame
of a discarded lead-in render out of the portrait, so the look carries over and the portrait itself
is never on screen.

ffmpeg runs for real (the fake client writes tiny lavfi clips); nothing needs ComfyUI or a GPU.
Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import shutil
import subprocess
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-opening-lead-"))
config.OUTPUT_DIR = TMP / "output"
config.CHANNELS_DIR = TMP / "channels"
config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"

import pipeline  # noqa: E402
from services import channels, video  # noqa: E402


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def lavfi(path: Path, spec: str, secs: float, audio: bool = False) -> Path:
    cmd = [video.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", spec]
    if audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo"]
    cmd += ["-t", f"{secs}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True, capture_output=True)
    return path


def stub_then_cut(path: Path, stub_frames: int = 4, secs: float = 2.0) -> Path:
    """`stub_frames` of red, then blue: the shape of a seeded clip that cuts to a new angle."""
    stub = stub_frames / 24
    return lavfi(path, f"color=c=red:s=64x64:r=24:d={stub}[a];color=c=blue:s=64x64:r=24:d={secs - stub}[b];"
                       "[a][b]concat=n=2:v=1:a=0", secs)


@lru_cache(maxsize=1)
def ffmpeg_works() -> bool:
    try:
        lavfi(TMP / "probe.mp4", "color=c=black:s=32x32:r=5:d=0.2", 0.2)
        return True
    except Exception:  # noqa: BLE001
        return False


class PortraitMode(unittest.TestCase):

    def test_a_legacy_record_keeps_what_its_switch_meant(self):
        self.assertEqual(channels.cast_portrait({"cast_opening": True}), "opening")
        self.assertEqual(channels.cast_portrait({"cast_opening": False}), "off")
        self.assertEqual(channels.cast_portrait({}), "opening")

    def test_reference_only_still_counts_as_using_the_portrait(self):
        rec = channels._validate({"name": "Fit", "cast_portrait": "reference"})
        self.assertEqual(rec["cast_portrait"], "reference")
        self.assertTrue(rec["cast_opening"])           # opening_character still picks a portrait

    def test_off_turns_the_old_switch_off(self):
        rec = channels._validate({"name": "Fit", "cast_portrait": "off"})
        self.assertFalse(rec["cast_opening"])

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            channels._validate({"name": "Fit", "cast_portrait": "sometimes"})

    def test_a_legacy_base_with_the_switch_off_is_not_turned_back_on(self):
        rec = channels._validate({"name": "Fit"}, base={"name": "Fit", "cast_opening": False})
        self.assertEqual(rec["cast_portrait"], "off")

    def test_the_job_snapshot_carries_the_mode(self):
        rec = channels._validate({"name": "Fit", "cast_portrait": "reference"})
        self.assertEqual(channels.snapshot(rec)["cast_portrait"], "reference")


class FakeClient:
    """Records what clip 1 was seeded with; every render is a tiny real mp4."""

    def __init__(self):
        self.uploads, self.builds = [], []

    def upload_image(self, path, name):
        self.uploads.append(Path(path).name)
        return name

    def build(self, template, prompt, prefix, first_frame=None, **kw):
        self.builds.append({"prefix": prefix, "first_frame": first_frame, "prompt": prompt,
                            "seconds": kw.get("seconds")})
        return {}

    def run(self, wf, dest, **kw):
        lavfi(Path(dest), "color=c=green:s=64x64:r=24:d=1", 1.0)
        return dest

    def scrub_file(self, *a):
        pass


@unittest.skipUnless(ffmpeg_works(), "no usable ffmpeg/libx264 on this machine")
class ClipOneSeed(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))
        lavfi(self.dir / "portrait.mp4", "color=c=red:s=64x64:r=5:d=0.2", 0.2)
        video.last_frame(self.dir / "portrait.mp4", self.dir / "cast_opening.jpg")
        self.job = mock.MagicMock(dir=self.dir, id="j1", cancel_requested=False,
                                  channel={"video_type": "real", "cast": []},
                                  aspect_ratio="9:16 (Portrait Widescreen)", megapixels=0.4,
                                  video_model="")
        self.client = FakeClient()

    def plan(self, mode):
        return {"style": "bright gym", "scenes": ["Mia does a cable curl."], "cast_used": ["Mia"],
                "opening": {"name": "Mia", "look": "black set", "mode": mode}}

    def render(self, plan):
        with mock.patch.object(pipeline, "_gpu_gate"), \
                mock.patch.object(config, "COMFY_CLEANUP", False):
            pipeline._generate_clip(self.job, self.client, "t2v", "i2v", plan, 1,
                                    self.dir / "clip_01.mp4", "clip 1/1")

    def test_reference_only_never_seeds_clip_one_with_the_portrait(self):
        self.render(self.plan("reference"))
        lead, clip = self.client.builds
        self.assertTrue(lead["prefix"].endswith("opening_lead"))
        self.assertEqual(lead["seconds"], config.OPENING_LEAD_SECONDS)
        self.assertIn("Mia does a cable curl.", lead["prompt"])     # it heads for scene 1
        self.assertEqual(self.client.uploads, ["cast_opening.jpg", pipeline.OPENING_LEAD_JPG])
        self.assertNotEqual(clip["first_frame"], lead["first_frame"])
        self.assertNotIn("Bring that image to life", clip["prompt"])

    def test_open_on_the_portrait_is_unchanged(self):
        self.render(self.plan("opening"))
        self.assertEqual(len(self.client.builds), 1)
        self.assertEqual(self.client.uploads, ["cast_opening.jpg"])
        self.assertIn("Bring that image to life", self.client.builds[0]["prompt"])

    def test_a_plan_from_before_the_mode_opens_on_the_portrait(self):
        plan = self.plan("opening")
        del plan["opening"]["mode"]
        self.render(plan)
        self.assertEqual(len(self.client.builds), 1)

    def test_the_lead_in_takes_the_likeness_but_not_the_portrait_background(self):
        """A builder's portrait shot in a cornfield put the cornfield into a backyard build."""
        self.render(self.plan("reference"))
        lead = self.client.builds[0]["prompt"]
        self.assertIn("background, ground, sky and lighting do not carry over", lead)
        self.assertIn("nothing of the reference image's surroundings left", lead)

    def test_a_lead_in_already_on_disk_is_not_rendered_again(self):
        video.last_frame(self.dir / "portrait.mp4", self.dir / pipeline.OPENING_LEAD_JPG)
        (self.dir / pipeline.OPENING_LEAD_JPG).write_bytes(b"x" * 2000)
        self.render(self.plan("reference"))
        self.assertEqual(len(self.client.builds), 1)
        self.assertEqual(self.client.uploads, [pipeline.OPENING_LEAD_JPG])


@unittest.skipUnless(ffmpeg_works(), "no usable ffmpeg/libx264 on this machine")
class DiyBuilderLook(unittest.TestCase):
    """A DIY short shows its builder in every shot, but close-ups and "he drills..." lines do not
    name him. Those clips used to get no look at all and invented a different man."""
    CAST = [{"name": "Jake", "look": "denim overalls over a white tee, short brown hair"}]

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))
        lavfi(self.dir / "clip_01.mp4", "color=c=red:s=64x64:r=24:d=1", 1.0)
        self.client = FakeClient()

    def render(self, video_type):
        job = mock.MagicMock(dir=self.dir, id="j2", cancel_requested=False,
                             channel={"video_type": video_type, "cast": self.CAST},
                             aspect_ratio="9:16 (Portrait Widescreen)", megapixels=0.4, video_model="")
        plan = {"style": "phone footage", "cast_used": ["Jake"],
                "scenes": ["Jake carries a pallet.", "Close-up: his hands drive a screw into the slat."]}
        with mock.patch.object(pipeline, "_gpu_gate"), \
                mock.patch.object(config, "COMFY_CLEANUP", False):
            pipeline._generate_clip(job, self.client, "t2v", "i2v", plan, 2,
                                    self.dir / "clip_02.mp4", "clip 2/2")
        return self.client.builds[-1]["prompt"]

    def test_an_unnamed_diy_shot_still_carries_the_builder_look(self):
        self.assertIn("Jake: denim overalls", self.render("diy"))

    def test_other_types_only_get_a_look_when_the_scene_names_the_character(self):
        self.assertNotIn("denim overalls", self.render("real"))


class DiyPortraitIsAReference(unittest.TestCase):

    def test_the_writer_is_told_the_look_is_identity_not_a_place(self):
        from services import llm
        self.assertIn("IDENTITY only", llm.DIY_RULES)
        self.assertIn("build site", llm.DIY_RULES)


@unittest.skipUnless(ffmpeg_works(), "no usable ffmpeg/libx264 on this machine")
class TrimLeadCut(unittest.TestCase):

    def test_the_stub_before_a_cut_is_dropped_and_the_length_kept(self):
        clip = stub_then_cut(TMP / "stub.mp4")
        before = video.duration(clip)
        dropped = video.trim_lead_cut(clip)
        self.assertAlmostEqual(dropped, 4 / 24, places=2)
        self.assertAlmostEqual(video.duration(clip), before, places=1)
        self.assertEqual(video.first_cut(clip), 0.0)          # it now opens on the new shot

    def test_a_clip_without_a_cut_is_left_alone(self):
        clip = lavfi(TMP / "steady.mp4", "color=c=blue:s=64x64:r=24:d=2", 2.0)
        stamp = clip.stat().st_mtime_ns
        self.assertEqual(video.trim_lead_cut(clip), 0.0)
        self.assertEqual(clip.stat().st_mtime_ns, stamp)

    def test_a_cut_later_in_the_clip_is_a_real_edit_and_kept(self):
        clip = stub_then_cut(TMP / "late.mp4", stub_frames=36, secs=3.0)   # a cut at 1.5 s
        self.assertEqual(video.trim_lead_cut(clip), 0.0)

    def test_the_audio_keeps_its_length_too(self):
        clip = lavfi(TMP / "aud.mp4", "color=c=red:s=64x64:r=24:d=0.17[a];"
                                      "color=c=blue:s=64x64:r=24:d=1.83[b];[a][b]concat=n=2:v=1:a=0",
                     2.0, audio=True)
        before = video.duration(clip)
        self.assertGreater(video.trim_lead_cut(clip), 0)
        self.assertTrue(video.has_audio(clip))
        self.assertAlmostEqual(video.duration(clip), before, places=1)


if __name__ == "__main__":
    unittest.main()
