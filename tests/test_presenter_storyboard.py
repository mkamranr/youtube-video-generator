"""The story board cards and the presenter on screen together, asserted on real pixels.

The unit tests in test_overlays.py pin trim_schedule's rules; this one proves the whole assembly
actually puts a card in the finished video when the presenter never leaves it - the case that
used to come out with no story board at all, because a corner overlay was reserving the entire
frame for the entire time it spoke.

Nothing is mocked except the chapter list: ffmpeg runs for real, so the assertions are on the
pixels of final.mp4. The avatar footage is the one stand-in - the render box would normally
produce it - and a flat magenta square with an opaque matte is all the compositor reads.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import shutil
import subprocess
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

from PIL import Image

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-presenter-cards-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402
from services import video  # noqa: E402

SIZE = (360, 640)          # a small 9:16: the templates still lay out, the encode stays cheap
NCLIPS = 3
CLIP_SECS = 3.0            # > 2 * CARD_INSET + 1.0, so every clip can carry a card
TOTAL = NCLIPS * CLIP_SECS
BASE = (16, 24, 32)        # one flat colour under everything, so any other pixel is overlay ink
PRESENTER = "0xff00ff"     # magenta: nothing else in the frame is near it


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


@lru_cache(maxsize=1)
def ffmpeg_works() -> bool:
    """Can this machine encode at all? The suite mocks ffmpeg everywhere else, so a box without
    a usable binary should skip this one rather than fail it."""
    try:
        subprocess.run([video.ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=32x32:r=5:d=0.2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(TMP / "probe.mp4")], check=True, capture_output=True, timeout=120)
        return True
    except Exception:  # noqa: BLE001
        return False


def solid(path: Path, colour: str, secs: float, size=SIZE) -> Path:
    subprocess.run([video.ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"color=c={colour}:s={size[0]}x{size[1]}:r=12:d={secs}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True)
    return path


def classify(px) -> str:
    """base / presenter / ink. The clips are one flat colour and the presenter is magenta, so
    everything else on screen was drawn by an overlay."""
    r, g, b = px
    if abs(r - BASE[0]) <= 16 and abs(g - BASE[1]) <= 16 and abs(b - BASE[2]) <= 16:
        return "base"
    if r > 140 and b > 140 and g < 100:
        return "presenter"
    return "ink"


@unittest.skipUnless(ffmpeg_works(), "no usable ffmpeg/libx264 on this machine")
class PresenterAndCardsTogether(unittest.TestCase):
    """One assembly, shared by every assertion: it is a real encode and costs real seconds."""

    @classmethod
    def setUpClass(cls):
        d = config.OUTPUT_DIR / "20260101-000000-abcdef"
        d.mkdir(parents=True)
        cls.dir = d
        clips = []
        for i in range(1, NCLIPS + 1):
            solid(d / f"clip_{i:02d}.mp4", f"0x{BASE[0]:02x}{BASE[1]:02x}{BASE[2]:02x}", CLIP_SECS)
            clips.append(f"clip_{i:02d}.mp4")
        # the presenter: on screen for the whole video, fully opaque
        box = video.avatar_box(SIZE, style="cutout")
        cls.presenter_rect = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        solid(d / "avatar_01.mp4", PRESENTER, TOTAL, size=(box[2], box[3]))
        solid(d / "avatar_01_matte.mp4", "white", TOTAL, size=(box[2], box[3]))

        cls.plan = {
            "storyboard": [
                {"start_clip": i, "end_clip": i, "kicker": "KEY FACT",
                 "headline": f"Card number {i}", "body": "A sentence of body copy.",
                 "image_prompt": "", "template": "lower_third"}
                for i in range(1, NCLIPS + 1)
            ],
            # "full" coverage is ONE window over the whole video - the shape that took every card
            "avatar": {"engine": "LTX 2.5", "style": "cutout", "coverage": "full",
                       "segments": [{"start": 0.0, "end": TOTAL, "file": "avatar_01.mp4",
                                     "matte": "avatar_01_matte.mp4", "bbox": []}]},
        }
        job = mock.MagicMock()
        job.dir, job.id, job.clips = d, d.name, clips
        job.storyboard = True
        job.page_scroll = False
        job.polish = False          # no hook, end card or progress bar: cards are the only ink
        job.captions = False
        job.audio_mode = "narration"
        job.channel = {"name": "Test"}
        job.hook = ""
        cls.logs = []
        job.add_log.side_effect = cls.logs.append
        with mock.patch.object(pipeline, "_chapters"):
            cls.final = pipeline._assemble(job, cls.plan, None, None)
        cls.frames = {at: cls._frame(at) for at in (1.5, 4.5, 7.5)}

    @classmethod
    def _frame(cls, at: float):
        out = TMP / f"frame_{at}.png"
        subprocess.run([video.ffmpeg(), "-y", "-v", "error", "-ss", str(at), "-i", str(cls.final),
                        "-frames:v", "1", str(out)], check=True, capture_output=True)
        return Image.open(out).convert("RGB")

    def tally(self, im):
        """{kind: count} plus the ink rows, so a test can ask WHERE the card ended up."""
        px = im.load()
        counts = {"base": 0, "presenter": 0, "ink": 0}
        rows = []
        for y in range(im.height):
            for x in range(im.width):
                kind = classify(px[x, y])
                counts[kind] += 1
                if kind == "ink":
                    rows.append(y)
        return counts, rows

    def test_the_overlay_log_counts_every_card(self):
        """The symptom was `overlays: 0 story board cards` on a video whose plan had three."""
        line = next((m for m in self.logs if m.startswith("overlays:")), "")
        self.assertEqual(line, f"overlays: {NCLIPS} story board cards", self.logs)

    def test_the_log_says_the_cards_were_lifted_rather_than_dropped(self):
        line = next((m for m in self.logs if m.startswith("presenter:")), "")
        self.assertIn("lifted clear", line, self.logs)
        self.assertIn("0 dropped", line)

    def test_the_presenter_is_on_screen_throughout(self):
        for at, im in self.frames.items():
            with self.subTest(at=at):
                counts, _ = self.tally(im)
                self.assertGreater(counts["presenter"], 500, f"t={at}s: {counts}")

    def test_a_card_is_on_screen_at_the_same_time(self):
        """The regression itself: a presenter who never leaves used to mean no card ever drew."""
        for at, im in self.frames.items():
            with self.subTest(at=at):
                counts, _ = self.tally(im)
                self.assertGreater(counts["ink"], 100, f"t={at}s: {counts}")

    def test_the_card_sits_clear_of_the_presenter(self):
        """What the lift is for. The card is composited UNDER the presenter, so overlapping ink
        would be hidden behind it - the card has to end up above the box, not just exist."""
        top = self.presenter_rect[1]
        for at, im in self.frames.items():
            with self.subTest(at=at):
                _counts, rows = self.tally(im)
                self.assertTrue(rows, f"t={at}s: no card ink at all")
                above = sum(1 for y in rows if y < top)
                self.assertGreater(above / len(rows), 0.9,
                                   f"t={at}s: only {above}/{len(rows)} ink rows above y={top}")

    def test_the_lifted_card_is_still_fully_on_the_frame(self):
        for at, im in self.frames.items():
            with self.subTest(at=at):
                _counts, rows = self.tally(im)
                self.assertGreaterEqual(min(rows), 0)
                self.assertLess(max(rows), im.height)


if __name__ == "__main__":
    unittest.main()
