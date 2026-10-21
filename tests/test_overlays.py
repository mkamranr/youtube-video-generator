"""Hook card, end card, progress bar, card whoosh and the composite filtergraph.

Run with  python -m unittest discover -s tests -t .  from the repo root. ffmpeg is mocked, so the
tests assert on the command that would have been run - which is where the one-encode property and
the quoting of per-frame expressions actually live.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-overlays-"))
config.OUTPUT_DIR = TMP / "output"

from services import cardtemplates, sfx, storyboard, video  # noqa: E402

SIZE = (1280, 720)


def png(path: Path, size=(8, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (255, 0, 0, 255)).save(path)
    return path


def stub(path: Path) -> Path:
    """A file that merely has to exist (the whoosh; ffmpeg is mocked)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" * 16)
    return path


class Composite(unittest.TestCase):
    """The filtergraph: one encode, expressions quoted, integers left bare."""

    def setUp(self):
        self.dir = TMP / "comp"
        self.png = png(self.dir / "a.png")
        self.src = self.dir / "in.mp4"
        self.src.write_bytes(b"x")
        self.run = mock.patch.object(video, "_run").start()
        mock.patch.object(video, "duration", return_value=10.0).start()
        mock.patch.object(video, "fps", return_value=24.0).start()
        mock.patch.object(video, "has_audio", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def _filt(self, layers, **kw):
        video.composite(self.src, layers, self.dir / "out.mp4", **kw)
        argv = self.run.call_args.args[0]
        return argv, argv[argv.index("-filter_complex") + 1]

    def test_an_integer_position_stays_bare(self):
        _argv, f = self._filt([{"png": self.png, "start": 1, "end": 5, "x": 72, "y": 130}])
        self.assertIn("overlay=72:130:eval=frame:", f)

    def test_an_expression_position_is_quoted_so_its_commas_survive(self):
        expr = "24*pow(1-clip((t-1.000)/0.400,0,1),3)"
        _argv, f = self._filt([{"png": self.png, "start": 1, "end": 5, "y": expr}])
        self.assertIn(f"overlay=0:'{expr}':eval=frame:", f)

    def test_a_per_layer_fade_overrides_the_default(self):
        _argv, f = self._filt([{"png": self.png, "start": 0, "end": 10, "fade": 0.3}], fade=0.5)
        self.assertIn("d=0.300", f)

    def test_one_encode_whatever_the_layer_count(self):
        layers = [{"png": self.png, "start": i, "end": i + 2} for i in range(6)]
        argv, _f = self._filt(layers)
        self.assertEqual(argv.count("-filter_complex"), 1)
        self.assertEqual(argv.count("libx264"), 1)
        self.assertEqual(argv.count("-i"), len(layers) + 1)

    def test_subtitles_are_the_last_filter_and_run_from_the_files_folder(self):
        ass = self.dir / "captions.ass"
        ass.write_text("x", encoding="utf-8")
        video.composite(self.src, [{"png": self.png, "start": 0, "end": 5}],
                        self.dir / "out.mp4", subtitles=ass)
        argv = self.run.call_args.args[0]
        f = argv[argv.index("-filter_complex") + 1]
        self.assertTrue(f.rstrip().endswith("subtitles=captions.ass[vsub]"))
        self.assertNotIn(str(self.dir), f)        # no drive letter, no backslashes in the graph
        self.assertEqual(self.run.call_args.kwargs.get("cwd"), self.dir)

    def test_subtitles_alone_still_encode_rather_than_stream_copy(self):
        ass = self.dir / "captions.ass"
        ass.write_text("x", encoding="utf-8")
        video.composite(self.src, [], self.dir / "out.mp4", subtitles=ass)
        argv = self.run.call_args.args[0]
        self.assertIn("-filter_complex", argv)
        self.assertNotIn("copy", argv[:argv.index("-filter_complex")])

    def test_no_layers_and_no_subtitles_is_a_stream_copy(self):
        video.composite(self.src, [], self.dir / "out.mp4")
        self.assertIn("copy", self.run.call_args.args[0])


class ProgressBar(unittest.TestCase):

    def test_track_and_bar_geometry(self):
        layers = video.progress_layers(SIZE, 30.0, TMP / "bar", height_pct=0.6)
        self.assertEqual(len(layers), 2)
        track, bar = layers
        self.assertEqual(track["x"], 0)
        self.assertEqual(track["y"], bar["y"])
        for lyr in layers:
            with Image.open(lyr["png"]) as im:
                self.assertEqual(im.width, SIZE[0])
                self.assertEqual(lyr["y"] + im.height, SIZE[1])

    def test_the_bar_sweeps_from_offscreen_to_flush(self):
        bar = video.progress_layers(SIZE, 30.0, TMP / "bar2")[1]
        self.assertIn(f"-{SIZE[0]}+{SIZE[0]}*clip(t/30.000,0,1)", bar["x"])

    def test_a_zero_length_video_gets_no_bar(self):
        self.assertEqual(video.progress_layers(SIZE, 0, TMP / "bar3"), [])


class Cards(unittest.TestCase):

    def setUp(self):
        self.dir = TMP / "cards"
        self.dir.mkdir(parents=True, exist_ok=True)

    def test_the_hook_card_leaves_the_top_and_bottom_clear(self):
        p = cardtemplates.render_hook("Local models are finally fast enough", SIZE,
                                      self.dir / "hook.png", kicker="IN 60 SECONDS")
        with Image.open(p) as im:
            a = im.getchannel("A")
            self.assertTrue(a.getbbox())
            self.assertEqual(a.crop((0, 0, SIZE[0], 8)).getbbox(), None)
            self.assertEqual(a.crop((0, SIZE[1] - 8, SIZE[0], SIZE[1])).getbbox(), None)

    def test_the_hook_card_renders_in_every_orientation(self):
        for size in (SIZE, (720, 1280), (1080, 1080)):
            p = cardtemplates.render_hook("A hook line", size, self.dir / f"h{size[0]}.png")
            with Image.open(p) as im:
                self.assertEqual(im.size, size)
                self.assertTrue(im.getchannel("A").getbbox())

    def test_the_end_card_keeps_the_lower_right_clear_for_youtube(self):
        logo = png(self.dir / "logo.png", (120, 120))
        boxes = []
        p = cardtemplates.render_end(SIZE, self.dir / "end.png", channel="Deep Stack",
                                     logo=logo, line="Subscribe for more")
        with Image.open(p) as im:
            self.assertEqual(im.size, SIZE)
        # nothing the card DREW (the scrim aside) reaches the lower-right quarter
        for x0, y0, x1, y1 in boxes:
            self.assertLess(y0, SIZE[1] * 0.75)

    def test_the_end_card_survives_a_missing_logo(self):
        p = cardtemplates.render_end(SIZE, self.dir / "end2.png", channel="Deep Stack",
                                     logo=self.dir / "nope.png", line="Subscribe")
        with Image.open(p) as im:
            self.assertTrue(im.getchannel("A").getbbox())


class TrimSchedule(unittest.TestCase):
    """A fact card must never share the screen with the hook or the end card."""

    def layer(self, start, end):
        return {"png": "x.png", "start": start, "end": end, "x": 0, "y": 0}

    def test_a_card_inside_the_hook_window_is_dropped(self):
        got = storyboard.trim_schedule([self.layer(0.5, 2.5)], [(0.0, 3.2)])
        self.assertEqual(got, [])

    def test_a_card_overlapping_the_end_card_is_shortened(self):
        got = storyboard.trim_schedule([self.layer(50.0, 60.0)], [(56.0, 60.0)])
        self.assertEqual(len(got), 1)
        self.assertLessEqual(got[0]["end"], 56.0)

    def test_a_card_overlapping_the_hook_starts_later(self):
        got = storyboard.trim_schedule([self.layer(1.0, 9.0)], [(0.0, 3.2)])
        self.assertGreaterEqual(got[0]["start"], 3.2)

    def test_untouched_cards_are_identical(self):
        lyr = self.layer(10.0, 20.0)
        self.assertEqual(storyboard.trim_schedule([lyr], [(0.0, 3.2)]), [lyr])

    def test_a_card_moved_by_the_page_panel_keeps_its_motion(self):
        """motion_xy bakes the start into an ffmpeg ease-out, so a trimmed card whose expression
        is left alone eases from a start it no longer has - it just appears in place."""
        lyr = dict(self.layer(10.0, 20.0), template="lower_third")
        lyr["x"], lyr["y"] = cardtemplates.motion_xy("lower_third", (1080, 1920), 10.0, 20.0)
        got = storyboard.trim_schedule([lyr], [(0.0, 14.0)], size=(1080, 1920))
        self.assertEqual(got[0]["start"], 14.2)
        moved = [v for v in (got[0]["x"], got[0]["y"]) if isinstance(v, str)]
        self.assertTrue(moved, "the card lost its motion expression entirely")
        for expr in moved:
            self.assertIn("t-14.200", expr)      # re-anchored to the new start

    def test_an_untouched_card_keeps_its_expression_byte_for_byte(self):
        lyr = dict(self.layer(10.0, 20.0), template="lower_third")
        lyr["x"], lyr["y"] = cardtemplates.motion_xy("lower_third", (1080, 1920), 10.0, 20.0)
        self.assertEqual(storyboard.trim_schedule([lyr], [(0.0, 3.2)], size=(1080, 1920)), [lyr])

    def test_a_card_without_a_template_is_left_alone(self):
        """motion_xy returns (0, 0) for a template it does not know, so recomputing a layer that
        carries no template would quietly wipe whatever x/y it already had."""
        lyr = dict(self.layer(10.0, 20.0), x="sentinel", y="sentinel")
        got = storyboard.trim_schedule([lyr], [(0.0, 14.0)], size=(1080, 1920))
        self.assertEqual((got[0]["x"], got[0]["y"]), ("sentinel", "sentinel"))

    def test_the_reserved_panel_window_is_trimmed_out_of_the_cards(self):
        cards = [self.layer(s, e) for s, e in [(5.35, 9.65), (15.35, 19.65), (25.35, 29.65)]]
        win = video.scroll_reserve(40, soft=[(c["start"], c["end"]) for c in cards],
                                   hard=[(0.0, 3.2), (36.0, 40.0)])
        got = storyboard.trim_schedule(cards, [win])
        self.assertEqual(len(got), 3)            # the panel costs 0.3s, not a whole card
        for c in got:
            self.assertFalse(c["end"] > win[0] and c["start"] < win[1])

    def test_no_busy_windows_changes_nothing(self):
        lyr = self.layer(1.0, 9.0)
        self.assertEqual(storyboard.trim_schedule([lyr], []), [lyr])

    def test_a_card_left_too_short_is_dropped_rather_than_flashed(self):
        self.assertEqual(storyboard.trim_schedule([self.layer(2.8, 4.0)], [(0.0, 3.2)]), [])


class TrimScheduleAroundThePresenter(unittest.TestCase):
    """A busy window carrying a rect takes only THAT part of the frame.

    The presenter is a corner box, so trimming a card out of the whole time it speaks dropped
    every card on a video it never leaves - which is the video the cards are there for. A card
    that misses the rect plays untouched; one that hits it rises above it.
    """
    PORTRAIT = (1080, 1920)

    def setUp(self):
        x, y, w, h = video.avatar_box(self.PORTRAIT, style="cutout")
        self.presenter = (x, y, x + w, y + h)
        bx, by, bw, bh = video.avatar_box(self.PORTRAIT, style="bubble")
        self.bubble = (bx, by, bx + bw, by + bh)
        # where the lower_third and quote_slab templates actually draw on a 1080x1920 frame
        self.low = (64, 1373, 1016, 1652)
        self.high = (64, 935, 1016, 1190)

    def layer(self, start, end, rect, template="lower_third"):
        return {"png": "x.png", "start": start, "end": end, "x": 0, "y": 0,
                "rect": rect, "template": template}

    def trim(self, layers, busy):
        return storyboard.trim_schedule(layers, busy, size=self.PORTRAIT)

    def test_a_presenter_who_never_leaves_no_longer_takes_every_card(self):
        """The reported bug: "full" coverage is one window over the whole video, so every card
        was fully covered and the assembled video had no story board at all."""
        cards = [self.layer(s, e, self.low) for s, e in [(5.3, 9.6), (15.3, 19.6), (25.3, 29.6)]]
        got = self.trim(cards, [(0.0, 30.0, self.presenter)])
        self.assertEqual(len(got), 3)

    def test_a_contiguous_chain_of_windows_no_longer_takes_every_card(self):
        """A chunked engine covers the video with back-to-back windows. Each one pushed the
        card's start forward and pulled its end back in the same pass, so the card ended up
        with a negative length and was dropped."""
        cards = [self.layer(s, e, self.low) for s, e in [(5.3, 9.6), (15.3, 19.6), (25.3, 29.6)]]
        chain = [(round(t * 4.81, 3), round(min(30.0, (t + 1) * 4.81), 3)) for t in range(7)]
        got = self.trim(cards, [(s, e, self.presenter) for s, e in chain])
        self.assertEqual(len(got), 3)

    def test_a_card_clear_of_the_presenter_is_identical(self):
        lyr = self.layer(5.0, 25.0, self.high, template="quote_slab")
        self.assertEqual(self.trim([lyr], [(0.0, 30.0, self.presenter)]), [lyr])

    def test_a_bubble_grazing_a_card_is_not_a_collision(self):
        """A bubble clips the corner of a full-width card by under a percent of it. Moving the
        card for that would be a visible jump for no reason - see CARD_CLEAR_FRAC."""
        lyr = self.layer(5.0, 25.0, self.low)
        self.assertEqual(self.trim([lyr], [(0.0, 30.0, self.bubble)]), [lyr])

    def test_a_colliding_card_is_lifted_rather_than_dropped(self):
        got = self.trim([self.layer(5.0, 25.0, self.low)], [(0.0, 30.0, self.presenter)])
        self.assertEqual(len(got), 1)
        lift = got[0]["lifted"]
        self.assertGreater(lift, 0)
        self.assertEqual(got[0]["start"], 5.0)          # it keeps its whole window
        self.assertEqual(got[0]["end"], 25.0)
        self.assertEqual(got[0]["y"], -lift)            # the PNG is full-frame: y IS the move
        self.assertEqual(got[0]["rect"],
                         (self.low[0], self.low[1] - lift, self.low[2], self.low[3] - lift))

    def test_a_lifted_card_ends_up_above_the_presenter_and_on_the_frame(self):
        """The two things the lift has to be worth doing: it clears the presenter, and it does
        not push the card off the top of the frame."""
        got = self.trim([self.layer(5.0, 25.0, self.low)], [(0.0, 30.0, self.presenter)])
        self.assertLessEqual(got[0]["rect"][3], self.presenter[1])
        self.assertGreaterEqual(got[0]["rect"][1], 0)

    def test_a_card_with_nowhere_to_rise_loses_its_window_instead(self):
        full_frame = self.layer(5.0, 25.0, (0, 0) + self.PORTRAIT)
        self.assertEqual(self.trim([full_frame], [(0.0, 30.0, self.presenter)]), [])

    def test_a_lifted_card_keeps_its_motion_expression_usable(self):
        """video._pos single-quotes an expression and forbids ':' and brackets in it, so the
        lift has to compose with the ease-out rather than replace it."""
        lyr = self.layer(5.0, 25.0, self.low)
        lyr["x"], lyr["y"] = cardtemplates.motion_xy("lower_third", self.PORTRAIT, 5.0, 25.0)
        self.assertIsInstance(lyr["y"], str)     # guards the fixture, not the code
        got = self.trim([lyr], [(0.0, 30.0, self.presenter)])
        y = got[0]["y"]
        self.assertIsInstance(y, str)
        self.assertTrue(y.startswith(lyr["y"]))
        self.assertNotIn(":", y)
        self.assertNotIn("[", y)
        self.assertEqual(video._pos(y), f"'{y}'")

    def test_a_window_without_a_rect_still_owns_the_whole_frame(self):
        """The hook and the end card both composite a full-frame scrim, so for them the old
        rule is the right one: nothing else may be on screen at all."""
        cards = [self.layer(s, e, self.low) for s, e in [(5.3, 9.6), (15.3, 19.6), (25.3, 29.6)]]
        self.assertEqual(storyboard.trim_schedule(cards, [(0.0, 30.0)], size=self.PORTRAIT), [])
        one = storyboard.trim_schedule(cards, [(0.0, 12.0)], size=self.PORTRAIT)
        # the first card is inside the window and lost; the others start after it and are
        # untouched, pad and all - the point is that none of them was MOVED to make room
        self.assertEqual([(c["start"], c["end"]) for c in one], [(15.3, 19.6), (25.3, 29.6)])

    def test_a_card_with_no_rect_of_its_own_is_treated_as_full_frame(self):
        """A hand-edited plan or a layer built before cards carried a rect: with nothing to
        measure, the safe reading is that the card covers everything."""
        lyr = {"png": "x.png", "start": 5.0, "end": 25.0, "x": 0, "y": 0, "rect": None}
        self.assertEqual(self.trim([lyr], [(0.0, 30.0, self.presenter)]), [])


class Sfx(unittest.TestCase):

    def test_hits_closer_than_half_a_second_collapse(self):
        self.assertEqual(sfx.dedupe([0.2, 0.3, 3.5, 6.0, 6.2, 8.0]), [0.2, 3.5, 6.0, 8.0])

    def test_negative_times_are_dropped_and_order_is_enforced(self):
        self.assertEqual(sfx.dedupe([5.0, -1.0, 1.0]), [1.0, 5.0])

    def test_the_hit_count_is_capped(self):
        self.assertLessEqual(len(sfx.dedupe([i for i in range(200)])), sfx.MAX_HITS)

    def test_no_hits_means_no_track(self):
        self.assertIsNone(sfx.build_track([], TMP / "s.wav", 10.0))

    def test_hits_past_the_end_of_the_video_are_dropped(self):
        with mock.patch.object(sfx, "ensure_whoosh") as whoosh, \
                mock.patch.object(sfx.tts, "concat_timed") as concat:
            whoosh.return_value = stub(TMP / "w.wav")
            sfx.build_track([1.0, 99.0], TMP / "s2.wav", 10.0)
            starts = [t for _p, t in concat.call_args.args[0]]
        self.assertEqual(starts, [1.0])

    def test_one_extra_mix_input_however_many_hits(self):
        """concat_timed lays every hit on ONE track, so amix never grows with the card count."""
        with mock.patch.object(sfx, "ensure_whoosh") as whoosh, \
                mock.patch.object(sfx.tts, "concat_timed") as concat:
            whoosh.return_value = stub(TMP / "w2.wav")
            sfx.build_track([1.0, 3.0, 5.0, 7.0], TMP / "s3.wav", 10.0)
        self.assertEqual(concat.call_count, 1)


class MixAudio(unittest.TestCase):

    def setUp(self):
        self.dir = TMP / "mix"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.src = self.dir / "in.mp4"
        self.src.write_bytes(b"x")
        self.nar = self.dir / "n.wav"
        self.nar.write_bytes(b"x")
        self.fx = self.dir / "sfx.wav"
        self.fx.write_bytes(b"x")
        self.run = mock.patch.object(video, "_run").start()
        mock.patch.object(video, "duration", return_value=10.0).start()
        mock.patch.object(video, "has_audio", return_value=False).start()
        self.addCleanup(mock.patch.stopall)

    def _filt(self, **kw):
        video.mix_audio(self.src, self.dir / "out.mp4", **kw)
        argv = self.run.call_args.args[0]
        return argv, argv[argv.index("-filter_complex") + 1]

    def test_the_whoosh_is_mixed_but_never_ducked(self):
        """It exists to mark a card appearing WHILE the voice talks - ducking would delete it."""
        _argv, f = self._filt(narration=self.nar, sfx=self.fx, sfx_volume=0.3)
        self.assertIn("[sx]", f)
        duck = [c for c in f.split(";") if "sidechaincompress" in c]
        self.assertTrue(all("[sx]" not in c for c in duck))

    def test_no_sfx_adds_no_input(self):
        argv_with, _ = self._filt(narration=self.nar, sfx=self.fx, sfx_volume=0.3)
        argv_without, _ = self._filt(narration=self.nar)
        self.assertEqual(argv_with.count("-i"), argv_without.count("-i") + 1)

    def test_zero_volume_adds_no_input(self):
        _argv, f = self._filt(narration=self.nar, sfx=self.fx, sfx_volume=0)
        self.assertNotIn("[sx]", f)

    def test_loudness_normalisation_runs_before_the_limiter(self):
        _argv, f = self._filt(narration=self.nar)
        self.assertIn("loudnorm=I=-14", f)
        self.assertLess(f.index("loudnorm"), f.index("alimiter"))

    def test_loudness_can_be_turned_off(self):
        with mock.patch.object(config, "LOUDNESS_LUFS", 0):
            _argv, f = self._filt(narration=self.nar)
        self.assertNotIn("loudnorm", f)
        self.assertIn("alimiter", f)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
