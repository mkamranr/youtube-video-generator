"""Presenter avatar: the channel registry, the per-video resolve, segment planning,
resolution-independent placement and the video layer in the composite filtergraph.

Run with  python -m unittest discover -s tests -t .  from the repo root. ffmpeg is mocked, so
the compositing tests assert on the command that would have been run - which is where the
single-encode property and the alphamerge wiring actually live.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-avatar-"))
config.OUTPUT_DIR = TMP / "output"
config.CHANNELS_DIR = TMP / "channels"
config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"

from services import avatar, channels, video  # noqa: E402
import pipeline  # noqa: E402


class ChannelAvatars(unittest.TestCase):
    """The roster lives on the channel record; the image is a file keyed by its id."""

    def test_first_avatar_becomes_the_default(self):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}, {"name": "Guest"}]})
        self.assertTrue(rec["avatars"][0]["default"])
        self.assertFalse(rec["avatars"][1]["default"])
        self.assertEqual(channels.default_avatar(rec)["name"], "Host")

    def test_only_one_default_survives(self):
        rec = channels._validate({"name": "C", "avatars": [
            {"name": "A", "default": True}, {"name": "B", "default": True}]})
        self.assertEqual([a["default"] for a in rec["avatars"]], [True, False])

    def test_ids_are_stable_across_edits_so_the_png_still_resolves(self):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}]})
        aid = rec["avatars"][0]["id"]
        again = channels._validate({"name": "C", "avatars": [{"id": aid, "name": "Renamed"}]},
                                   base=rec)
        self.assertEqual(again["avatars"][0]["id"], aid)

    def test_an_unknown_id_is_reissued_rather_than_trusted(self):
        rec = channels._validate({"name": "C", "avatars": [{"id": "../etc", "name": "Host"}]})
        self.assertNotEqual(rec["avatars"][0]["id"], "../etc")

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            channels._validate({"name": "C", "avatars": [{"name": "Host"}, {"name": "host"}]})

    def test_default_avatar_id_follows_the_roster(self):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}]})
        self.assertEqual(rec["defaults"]["avatar_id"], rec["avatars"][0]["id"])

    def test_a_default_pointing_at_a_deleted_avatar_is_cleared_not_rejected(self):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}],
                                  "defaults": {"avatar_id": "gone"}})
        self.assertEqual(rec["defaults"]["avatar_id"], rec["avatars"][0]["id"])

    def test_avatars_travel_on_the_job_snapshot(self):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}]})
        self.assertEqual(len(channels.snapshot(rec)["avatars"]), 1)


class ResolveAvatar(unittest.TestCase):
    """Request value -> channel default -> global default, and degrade instead of raising."""

    def _channel(self, **defaults):
        rec = channels._validate({"name": "C", "avatars": [{"name": "Host"}],
                                  "defaults": defaults})
        rec["id"] = "c1"
        return rec

    def test_channel_default_is_used_when_the_request_is_blank(self):
        ch = self._channel(avatar_coverage="intro")
        _id, _model, cov = pipeline.JobManager._resolve_avatar("", None, "", ch)
        self.assertEqual(cov, "intro")

    def test_request_beats_the_channel(self):
        ch = self._channel(avatar_coverage="intro")
        _id, _model, cov = pipeline.JobManager._resolve_avatar("", None, "outro", ch)
        self.assertEqual(cov, "outro")

    def test_blank_engine_stays_blank_so_settings_decides_at_render_time(self):
        _id, model, _cov = pipeline.JobManager._resolve_avatar("", None, "", self._channel())
        self.assertEqual(model, "")

    def test_a_channel_with_no_avatars_cannot_have_a_presenter(self):
        ch = channels._validate({"name": "C", "defaults": {"avatar_coverage": "both"}})
        ch["id"] = "c1"
        aid, _model, cov = pipeline.JobManager._resolve_avatar("", None, "both", ch)
        self.assertEqual((aid, cov), ("", "none"))

    def test_an_unknown_avatar_id_falls_back_to_the_default_rather_than_raising(self):
        ch = self._channel()
        aid, _model, cov = pipeline.JobManager._resolve_avatar("nope", None, "both", ch)
        self.assertEqual(aid, ch["avatars"][0]["id"])
        self.assertEqual(cov, "both")

    def test_a_zero_budget_switches_the_presenter_off_everywhere(self):
        ch = self._channel()
        with mock.patch.object(config, "AVATAR_MAX_SECONDS", 0.0):
            _aid, _model, cov = pipeline.JobManager._resolve_avatar("", None, "both", ch)
        self.assertEqual(cov, "none")

    def test_an_unknown_coverage_is_an_error(self):
        with self.assertRaises(ValueError):
            pipeline.JobManager._resolve_avatar("", None, "sometimes", self._channel())


class CoverageSurvivesSubmit(unittest.TestCase):
    """The reported bug: "Whole video" was picked in the New Video dialog and the job came back
    with the presenter at the start and the end only, its log naming `both`. Everything between
    the request and the stored job is pinned here, for a repo job in particular - that was the
    source the report came from."""

    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = TMP / "submit-output"
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        p = mock.patch.object(pipeline.JobManager, "_start_draft")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(pipeline.comfy, "ComfyClient")
        p.start()
        self.addCleanup(p.stop)
        self.mgr = pipeline.JobManager()
        rec = channels._validate({"name": "Daily Tech Talks", "video_type": "tech",
                                  "audience": "general", "avatars": [{"name": "Host"}],
                                  "defaults": {"avatar_coverage": "both"}})
        rec["id"] = "c1"
        self.channel = rec

    def _submit(self, kind, text, **kw):
        return self.mgr.submit_prompt(text, kind, channel=self.channel, owner="u1", **kw)

    def test_full_survives_a_repo_submit(self):
        job = self._submit("repo", "https://huggingface.co/tiiuae/Falcon-OCR",
                           avatar_coverage="full")
        self.assertEqual(job.avatar_coverage, "full")

    def test_full_survives_a_topic_submit(self):
        job = self._submit("topic", "A short film about trains", avatar_coverage="full")
        self.assertEqual(job.avatar_coverage, "full")

    def test_the_choice_is_written_to_disk_so_a_resume_keeps_it(self):
        job = self._submit("repo", "https://github.com/tiiuae/falcon-ocr", avatar_coverage="full")
        saved = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["avatar_coverage"], "full")
        self.assertEqual(pipeline.Job.load(job.dir / "job.json").avatar_coverage, "full")

    def test_full_says_at_submit_what_the_budget_will_pay_for(self):
        job = self._submit("topic", "A short film about trains", avatar_coverage="full")
        line = next(l for l in job.log if "presenter:" in l)
        self.assertIn("Whole video", line)
        self.assertIn("the budgets pay for", line)
        self.assertIn("start and the end", line)      # the fallback, before anything renders
        self.assertIn("LTX 2.5", line)                # ...and the engine that would do better

    def test_a_coverage_inside_the_budget_just_states_its_cost(self):
        job = self._submit("topic", "A short film about trains", avatar_coverage="intro")
        line = next(l for l in job.log if "presenter:" in l)
        self.assertIn("Intro only", line)
        self.assertIn("of GPU", line)
        self.assertNotIn("start and the end", line)

    def test_no_presenter_logs_nothing(self):
        job = self._submit("topic", "A short film about trains", avatar_coverage="none")
        self.assertEqual([l for l in job.log if "presenter:" in l], [])

    def test_an_empty_request_still_takes_the_channel_default(self):
        job = self._submit("topic", "A short film about trains")
        self.assertEqual(job.avatar_coverage, "both")


class SegmentWindows(unittest.TestCase):
    """Where the presenter speaks, and what happens when the budget cannot pay for it."""

    def test_none_renders_nothing(self):
        self.assertEqual(avatar.segment_windows("none", 120.0, "LTX 2.5"), [])

    def test_intro_starts_at_zero_and_outro_ends_at_the_end(self):
        intro = avatar.segment_windows("intro", 120.0, "LTX 2.5")
        outro = avatar.segment_windows("outro", 120.0, "LTX 2.5")
        self.assertEqual(intro[0][0], 0.0)
        self.assertEqual(outro[0][1], 120.0)

    def test_wan_segments_are_capped_at_one_chunk(self):
        cap = config.AVATAR_MODELS["Wan 2.2 S2V"]["max_segment"]
        for s, e in avatar.segment_windows("both", 300.0, "Wan 2.2 S2V"):
            self.assertLessEqual(e - s, cap + 1e-6)

    def test_full_on_a_long_video_degrades_to_intro_plus_outro(self):
        wins = avatar.segment_windows("full", 600.0, "LTX 2.5")
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins[0][0], 0.0)
        self.assertEqual(wins[-1][1], 600.0)

    def test_full_on_a_short_video_covers_all_of_it(self):
        with mock.patch.object(config, "AVATAR_MAX_SECONDS", 60.0):
            wins = avatar.segment_windows("full", 20.0, "LTX 2.5")
        self.assertEqual(wins, [(0.0, 20.0)])

    def test_the_budget_is_never_exceeded(self):
        for engine in config.AVATAR_MODELS:
            budget = config.avatar_budget_seconds(engine)
            got = sum(e - s for s, e in avatar.segment_windows("both", 600.0, engine))
            self.assertLessEqual(got, budget + 1e-6, engine)

    def test_a_video_too_short_for_two_windows_gets_one(self):
        wins = avatar.segment_windows("both", 6.0, "LTX 2.5")
        self.assertEqual(len(wins), 1)

    def test_edges_snap_onto_word_boundaries(self):
        words = [{"word": "a", "start": 0.0, "end": 0.5},
                 {"word": "b", "start": 0.6, "end": 11.4},
                 {"word": "c", "start": 12.3, "end": 13.0}]
        wins = avatar.segment_windows("intro", 60.0, "LTX 2.5", words)
        self.assertAlmostEqual(wins[0][1], 11.4, places=3)   # not the raw 12.0

    def test_no_word_timings_is_not_an_error(self):
        self.assertTrue(avatar.segment_windows("intro", 60.0, "LTX 2.5", []))


    def test_full_tiles_a_chunked_engine_end_to_end(self):
        """A Wan segment is one 77-frame chunk, so "whole video" there is a chain of them. The
        chain has to be contiguous: a gap is the presenter blinking out mid-sentence."""
        cap = config.AVATAR_MODELS["Wan 2.2 S2V"]["max_segment"]
        total = cap * 3
        with mock.patch.multiple(config, AVATAR_MAX_SECONDS=120.0,
                                 AVATAR_MAX_RENDER_MINUTES=6000.0):
            wins = avatar.segment_windows("full", total, "Wan 2.2 S2V")
        self.assertEqual(len(wins), 3)
        self.assertEqual(wins[0][0], 0.0)
        self.assertAlmostEqual(wins[-1][1], total, delta=0.001)   # windows are rounded to ms
        for (_s0, e0), (s1, _e1) in zip(wins, wins[1:]):
            self.assertAlmostEqual(e0, s1, places=3)
        for s, e in wins:
            self.assertLessEqual(e - s, cap + 0.001)   # edges are quantised to the millisecond

    def test_a_chain_keeps_its_interior_edges_off_the_word_snap(self):
        """Snapping every edge to the nearest word boundary would pull a contiguous chain
        apart - the one place the snap must not apply."""
        cap = config.AVATAR_MODELS["Wan 2.2 S2V"]["max_segment"]
        words = [{"word": "a", "start": 0.0, "end": 3.2}, {"word": "b", "start": 3.4, "end": 9.0}]
        with mock.patch.multiple(config, AVATAR_MAX_SECONDS=120.0,
                                 AVATAR_MAX_RENDER_MINUTES=6000.0):
            wins = avatar.segment_windows("full", cap * 2, "Wan 2.2 S2V", words)
        self.assertEqual(len(wins), 2)
        self.assertAlmostEqual(wins[0][1], wins[1][0], places=3)

    def test_a_two_chunk_chain_is_not_collapsed_into_one(self):
        """The collapse rule is for an intro and an outro that overlap on a short video; a
        chain is contiguous by construction, so it would have halved every short full video."""
        cap = config.AVATAR_MODELS["Wan 2.2 S2V"]["max_segment"]
        with mock.patch.multiple(config, AVATAR_MAX_SECONDS=120.0,
                                 AVATAR_MAX_RENDER_MINUTES=6000.0):
            wins = avatar.segment_windows("full", cap * 2, "Wan 2.2 S2V")
        self.assertEqual(len(wins), 2)

    def test_a_tile_too_short_to_be_worth_a_pass_is_dropped(self):
        """A 0.7 s remainder still costs a whole 77-frame render and reads as a glitch."""
        cap = config.AVATAR_MODELS["Wan 2.2 S2V"]["max_segment"]
        with mock.patch.multiple(config, AVATAR_MAX_SECONDS=120.0,
                                 AVATAR_MAX_RENDER_MINUTES=6000.0):
            wins = avatar.segment_windows("full", cap * 2 + 0.7, "Wan 2.2 S2V")
        self.assertEqual(len(wins), 2)
        for s, e in wins:
            self.assertGreaterEqual(e - s, avatar.MIN_SEGMENT)

    def test_full_stays_one_window_where_the_engine_needs_no_chain(self):
        """LTX renders a whole segment in one pass, so chaining would buy nothing and add a
        pose reset at every seam."""
        with mock.patch.object(config, "AVATAR_MAX_SECONDS", 120.0):
            wins = avatar.segment_windows("full", 90.0, "LTX 2.5")
        self.assertEqual(wins, [(0.0, 90.0)])

    def test_full_over_budget_still_falls_back_to_the_two_bookends(self):
        """The no-regression case: a video the budget cannot cover behaves as it always did."""
        wins = avatar.segment_windows("full", 600.0, "Wan 2.2 S2V")
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins, avatar.segment_windows("both", 600.0, "Wan 2.2 S2V"))


class Describe(unittest.TestCase):
    """The phrase a finished job shows next to the coverage that was asked for."""

    def test_the_two_bookends(self):
        self.assertEqual(avatar.describe([(0.0, 4.8), (67.6, 72.3)], 72.3),
                         "the start and the end")

    def test_a_chain_that_covers_everything(self):
        wins = [(0.0, 4.8), (4.8, 9.6), (9.6, 14.4)]
        self.assertEqual(avatar.describe(wins, 14.4), "the whole video")

    def test_one_window_at_the_top(self):
        self.assertEqual(avatar.describe([(0.0, 12.0)], 90.0), "the start")

    def test_one_window_at_the_end(self):
        self.assertEqual(avatar.describe([(78.0, 90.0)], 90.0), "the end")

    def test_nothing_rendered(self):
        self.assertEqual(avatar.describe([], 90.0), "nothing")

    def test_a_chain_that_stops_short_is_not_called_whole(self):
        self.assertNotEqual(avatar.describe([(0.0, 4.8), (4.8, 9.6)], 72.3), "the whole video")


class Placement(unittest.TestCase):
    """One set of fractions has to work for every resolution and aspect ratio."""

    SIZES = [(1920, 1080), (3840, 2160), (2560, 1440), (1080, 1920), (720, 1280), (1080, 1080)]

    def test_the_presenter_always_fits_inside_the_frame(self):
        for size in self.SIZES:
            for style in ("bubble", "cutout"):
                x, y, w, h = video.avatar_box(size, style)
                self.assertGreaterEqual(x, 0, (size, style))
                self.assertGreaterEqual(y, 0, (size, style))
                self.assertLessEqual(x + w, size[0], (size, style))
                self.assertLessEqual(y + h, size[1], (size, style))

    def test_the_bubble_scales_with_the_short_side(self):
        _x, _y, w1080, _h = video.avatar_box((1920, 1080), "bubble")
        _x, _y, w4k, _h = video.avatar_box((3840, 2160), "bubble")
        self.assertAlmostEqual(w4k / w1080, 2.0, delta=0.05)

    def test_a_portrait_frame_does_not_get_a_presenter_across_its_width(self):
        W = 1080
        _x, _y, w, _h = video.avatar_box((W, 1920), "cutout")
        self.assertLessEqual(w, config.AVATAR_MAX_WIDTH_PCT * W + 2)

    def test_the_cutout_is_anchored_to_the_bottom_edge(self):
        # a cut-out lifted off the edge shows a sliced torso, which is the thing to avoid
        for size in self.SIZES:
            _x, y, _w, h = video.avatar_box(size, "cutout", clear_bottom=0.10)
            self.assertEqual(y + h, size[1], size)

    def test_the_bubble_clears_the_caption_band(self):
        size = (1920, 1080)
        _x, y, _w, h = video.avatar_box(size, "bubble", clear_bottom=0.10)
        self.assertLessEqual(y + h, size[1] * 0.90 + 1)

    def test_corners_are_honoured(self):
        size = (1920, 1080)
        left = video.avatar_box(size, "bubble", corner="bottom-left")
        right = video.avatar_box(size, "bubble", corner="bottom-right")
        self.assertLess(left[0], right[0])
        top = video.avatar_box(size, "bubble", corner="top-right")
        self.assertLess(top[1], right[1])


    def test_a_measured_subject_fills_the_box_instead_of_the_frame(self):
        """The engines centre a head-and-shoulders portrait in a square, so drawing the whole
        frame spends a quarter of the box on transparency and the presenter reads as small."""
        size = (1440, 2580)
        x, y, bw, bh = video.avatar_box(size, "cutout")
        segs = [{"video": Path("a.mp4"), "matte": None, "start": 0.0, "end": 4.0,
                 "bbox": [0.2, 0.15, 0.8, 1.0]}]
        lyr = video.avatar_layers(size, segs, TMP, style="cutout")[0]
        self.assertGreater(lyr["w"], bw)                      # the frame is enlarged...
        sub_w, sub_h = 0.6 * lyr["w"], 0.85 * lyr["h"]        # ...until the FIGURE spans the box
        self.assertAlmostEqual(max(sub_w, sub_h), max(bw, bh), delta=3)
        self.assertLessEqual(sub_w, bw + 2)
        self.assertLessEqual(sub_h, bh + 2)
        # and lands where the box was: bottom edge held, centred across it
        self.assertAlmostEqual(lyr["y"] + 1.0 * lyr["h"], y + bh, delta=2)
        self.assertAlmostEqual(lyr["x"] + 0.5 * lyr["w"], x + bw / 2, delta=2)

    def test_an_unmeasured_segment_draws_exactly_as_it_used_to(self):
        size = (1440, 2580)
        segs = [{"video": Path("a.mp4"), "matte": None, "start": 0.0, "end": 4.0}]
        lyr = video.avatar_layers(size, segs, TMP, style="cutout")[0]
        self.assertEqual((lyr["x"], lyr["y"], lyr["w"], lyr["h"]), video.avatar_box(size, "cutout"))

    def test_a_subject_measurement_never_blows_the_presenter_up(self):
        """A matte that found only a sliver must not scale the frame to fill the video."""
        size = (1440, 2580)
        _x, _y, bw, _bh = video.avatar_box(size, "cutout")
        segs = [{"video": Path("a.mp4"), "matte": None, "start": 0.0, "end": 4.0,
                 "bbox": [0.49, 0.49, 0.51, 0.51]}]
        lyr = video.avatar_layers(size, segs, TMP, style="cutout")[0]
        self.assertLessEqual(lyr["w"], bw * video.AVATAR_FILL_MAX + 2)

    def test_an_unreadable_matte_measures_nothing_rather_than_failing(self):
        self.assertIsNone(avatar.matte_bounds(TMP / "no-such-matte.mp4"))


class CompositeVideoLayer(unittest.TestCase):
    """The new layer kind must stay inside the one encode and wire alphamerge correctly."""

    def setUp(self):
        self.run = mock.patch.object(video, "_run").start()
        mock.patch.object(video, "duration", return_value=30.0).start()
        mock.patch.object(video, "fps", return_value=24.0).start()
        mock.patch.object(video, "has_audio", return_value=True).start()
        self.addCleanup(mock.patch.stopall)
        self.dir = TMP / "comp"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clip = self.dir / "avatar_01.mp4"
        self.matte = self.dir / "avatar_01_matte.mp4"
        for p in (self.clip, self.matte):
            p.write_bytes(b"x")

    def _graph(self, layers):
        video.composite(self.dir / "in.mp4", layers, self.dir / "out.mp4")
        cmd = self.run.call_args[0][0]
        return cmd, cmd[cmd.index("-filter_complex") + 1]

    def test_a_matte_becomes_alphamerge_and_stays_one_encode(self):
        layers = [{"video": self.clip, "matte": self.matte, "w": 200, "h": 200,
                   "x": 10, "y": 20, "start": 2.0, "end": 8.0}]
        cmd, fc = self._graph(layers)
        self.assertIn("alphamerge", fc)
        self.assertEqual(cmd.count("-filter_complex"), 1)
        self.assertEqual(sum(1 for a in cmd if a == "-i"), 3)   # base + clip + matte

    def test_the_clip_is_slid_to_its_window(self):
        layers = [{"video": self.clip, "matte": self.matte, "w": 200, "h": 200,
                   "start": 7.5, "end": 12.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("setpts=PTS-STARTPTS+7.500/TB", fc)
        self.assertIn("enable='between(t,7.500,12.000)'", fc)

    def test_the_clip_is_resampled_to_the_video_frame_rate(self):
        # Wan renders at 16 fps and LTX at 24; the job's rate is what the encode runs at
        layers = [{"video": self.clip, "matte": self.matte, "w": 64, "h": 64,
                   "start": 0.0, "end": 4.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("fps=24.0000", fc)

    def test_a_png_matte_is_looped_or_alphamerge_starves(self):
        mask = self.dir / "avatar_mask.png"
        mask.write_bytes(b"x")
        layers = [{"video": self.clip, "matte": mask, "w": 64, "h": 64,
                   "start": 0.0, "end": 4.0}]
        cmd, _fc = self._graph(layers)
        self.assertIn("-loop", cmd)

    def test_the_cutout_shadow_is_drawn_under_the_presenter(self):
        layers = [{"video": self.clip, "matte": self.matte, "w": 200, "h": 200,
                   "x": 100, "y": 50, "start": 0.0, "end": 5.0, "shadow": 12.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("gblur", fc)
        self.assertLess(fc.index("[sh1]overlay"), fc.index("[c1]overlay"))

    def test_png_only_layers_are_unchanged(self):
        png = self.dir / "card.png"
        png.write_bytes(b"x")
        cmd, fc = self._graph([{"png": png, "start": 1.0, "end": 4.0}])
        self.assertNotIn("alphamerge", fc)
        self.assertIn("-loop", cmd)

    def test_the_default_is_still_one_symmetric_fade(self):
        layers = [{"video": self.clip, "matte": self.matte, "w": 80, "h": 80,
                   "start": 2.0, "end": 8.0, "fade": 0.4}]
        _cmd, fc = self._graph(layers)
        self.assertIn("fade=t=in:st=2.000:d=0.400:alpha=1", fc)
        self.assertIn("fade=t=out:st=7.600:d=0.400:alpha=1", fc)

    def test_an_interior_seam_gets_no_ramp_at_all(self):
        """The whole point: a tile with a neighbour on both sides never dips."""
        layers = [{"video": self.clip, "matte": self.matte, "w": 80, "h": 80,
                   "start": 4.75, "end": 9.5, "fade": 0.4, "fade_in": 0.0, "fade_out": 0.0}]
        _cmd, fc = self._graph(layers)
        self.assertNotIn("fade=t=", fc)
        # and the filter chain is still well formed without them
        self.assertNotIn(",,", fc)
        self.assertNotIn(",[c1]", fc)
        self.assertIn("alphamerge[c1]", fc)

    def test_the_two_edges_are_independent(self):
        layers = [{"video": self.clip, "matte": self.matte, "w": 80, "h": 80,
                   "start": 0.0, "end": 4.75, "fade": 0.4, "fade_in": 0.4, "fade_out": 0.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("fade=t=in:st=0.000:d=0.400", fc)
        self.assertNotIn("fade=t=out", fc)

    def test_a_png_layer_with_no_ramp_is_also_well_formed(self):
        png = self.dir / "card.png"
        png.write_bytes(b"x")
        _cmd, fc = self._graph([{"png": png, "start": 1.0, "end": 4.0,
                                 "fade_in": 0.0, "fade_out": 0.0}])
        self.assertIn("format=rgba[c1]", fc)

    def test_the_shadow_follows_the_presenter_edge_for_edge(self):
        """It is the same matte blurred; a shadow that fades on its own would give the dip back."""
        layers = [{"video": self.clip, "matte": self.matte, "w": 200, "h": 200,
                   "start": 4.75, "end": 9.5, "shadow": 12.0,
                   "fade_in": 0.0, "fade_out": 0.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("[sh1]", fc)
        self.assertNotIn("fade=t=", fc)

    def test_layers_build_a_single_overlay_chain(self):
        png = self.dir / "card.png"
        png.write_bytes(b"x")
        layers = [{"png": png, "start": 0.0, "end": 9.0},
                  {"video": self.clip, "matte": self.matte, "w": 80, "h": 80,
                   "start": 1.0, "end": 5.0}]
        _cmd, fc = self._graph(layers)
        self.assertIn("[v1][c2]", fc)      # layer 2 composites onto layer 1's output


class AvatarLayers(unittest.TestCase):
    def setUp(self):
        self.dir = TMP / "layers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clip = self.dir / "a.mp4"
        self.clip.write_bytes(b"x")
        self.seg = [{"start": 0.0, "end": 5.0, "video": self.clip, "matte": None}]

    def test_the_bubble_draws_its_ring_after_the_presenter(self):
        out = video.avatar_layers((1920, 1080), self.seg, self.dir, style="bubble")
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].get("video"))
        self.assertTrue(str(out[1]["png"]).endswith("avatar_ring.png"))

    def test_the_bubble_mask_matches_the_bubble_size(self):
        from PIL import Image
        out = video.avatar_layers((1920, 1080), self.seg, self.dir, style="bubble")
        with Image.open(out[0]["matte"]) as im:
            self.assertEqual(im.size, (out[0]["w"], out[0]["h"]))

    def test_the_cutout_asks_for_a_shadow(self):
        out = video.avatar_layers((1920, 1080), self.seg, self.dir, style="cutout")
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0]["shadow"], 0)

    def _chain(self, n=3, cap=4.75, bboxes=None):
        return [{"start": round(i * cap, 3), "end": round((i + 1) * cap, 3),
                 "video": self.clip, "matte": None,
                 "bbox": (bboxes[i] if bboxes else [0.2, 0.1, 0.8, 1.0])}
                for i in range(n)]

    def test_a_chain_fades_in_once_and_out_once(self):
        out = video.avatar_layers((1080, 1920), self._chain(), self.dir, style="cutout")
        self.assertEqual([(l["fade_in"], l["fade_out"]) for l in out],
                         [(0.4, 0.0), (0.0, 0.0), (0.0, 0.4)])

    def test_separate_appearances_keep_both_ramps(self):
        segs = [{"start": 0.0, "end": 12.0, "video": self.clip, "matte": None},
                {"start": 58.0, "end": 70.0, "video": self.clip, "matte": None}]
        out = video.avatar_layers((1080, 1920), segs, self.dir, style="cutout")
        self.assertEqual([(l["fade_in"], l["fade_out"]) for l in out], [(0.4, 0.4), (0.4, 0.4)])

    def test_a_chained_bubble_does_not_blink_either(self):
        out = video.avatar_layers((1080, 1920), self._chain(2), self.dir, style="bubble")
        # two presenter layers then two ring layers, each pair sharing the seam
        self.assertEqual([(l["fade_in"], l["fade_out"]) for l in out],
                         [(0.4, 0.0), (0.0, 0.4)] * 2)

    def test_a_chain_is_drawn_at_one_size(self):
        """Per-tile bounds made the figure resize at every seam - the same places it blinked."""
        segs = self._chain(3, bboxes=[[0.30, 0.20, 0.70, 1.0],
                                      [0.25, 0.10, 0.75, 0.9],
                                      [0.35, 0.15, 0.65, 1.0]])
        out = video.avatar_layers((1080, 1920), segs, self.dir, style="cutout")
        self.assertEqual(len({(l["x"], l["y"], l["w"], l["h"]) for l in out}), 1)

    def test_the_shared_box_is_the_union_so_no_tile_is_cropped(self):
        segs = self._chain(2, bboxes=[[0.30, 0.20, 0.70, 0.90], [0.10, 0.05, 0.60, 1.0]])
        out = video.avatar_layers((1080, 1920), segs, self.dir, style="cutout")
        alone = video.avatar_layers(
            (1080, 1920), [dict(segs[0], start=0.0, end=4.75, bbox=[0.10, 0.05, 0.70, 1.0])],
            self.dir, style="cutout")
        self.assertEqual((out[0]["w"], out[0]["h"]), (alone[0]["w"], alone[0]["h"]))

    def test_two_appearances_keep_their_own_boxes(self):
        segs = [{"start": 0.0, "end": 12.0, "video": self.clip, "matte": None,
                 "bbox": [0.30, 0.20, 0.70, 0.90]},
                {"start": 58.0, "end": 70.0, "video": self.clip, "matte": None,
                 "bbox": [0.10, 0.05, 0.60, 1.00]}]
        out = video.avatar_layers((1080, 1920), segs, self.dir, style="cutout")
        self.assertNotEqual((out[0]["w"], out[0]["h"]), (out[1]["w"], out[1]["h"]))

    def test_no_segments_means_no_layers(self):
        self.assertEqual(video.avatar_layers((1920, 1080), [], self.dir), [])


class AvatarStage(unittest.TestCase):
    """_avatar_assets: resumable, budget-aware, and never fatal."""

    def setUp(self):
        # a fresh directory per test: these assert on whether a clip was RE-rendered, so a
        # leftover avatar_01.mp4 from another test would silently make that pass
        self.jdir = Path(tempfile.mkdtemp(prefix="job-", dir=TMP))
        (self.jdir / "narration.wav").write_bytes(b"x" * 2000)
        config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
        self.job = mock.MagicMock()
        self.job.dir = self.jdir
        self.job.id = "job1"
        self.job.channel_id = "c1"
        self.job.avatar_id = "a1"
        self.job.avatar_model = "LTX 2.5"
        self.job.avatar_coverage = "intro"
        self.job.cancel_requested = False
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        channels.avatar_path("c1", "a1").write_bytes(b"png")
        mock.patch.object(pipeline.tts, "audio_duration", return_value=90.0).start()
        mock.patch.object(pipeline, "_gpu_gate").start()
        mock.patch.object(pipeline, "_save_plan").start()
        self.client = mock.patch.object(pipeline.comfy, "ComfyClient").start()
        mock.patch.object(pipeline.avatar, "slice_audio", return_value=12.0).start()
        mock.patch.object(pipeline.avatar, "matte_segment").start()
        self.render = mock.patch.object(pipeline.avatar, "render_segment").start()

        def _render(_c, _e, _i, _a, _s, dest, **kw):
            Path(dest).write_bytes(b"clip")
            return dest
        self.render.side_effect = _render
        mock.patch.object(pipeline.avatar, "trim_lead_frame",
                          side_effect=lambda src, dst, eng: (Path(dst).write_bytes(b"c"), dst)[1]).start()
        self.addCleanup(mock.patch.stopall)

    def test_it_records_segments_on_the_plan(self):
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        self.assertEqual(plan["avatar"]["engine"], "LTX 2.5")
        self.assertEqual(len(plan["avatar"]["segments"]), 1)
        self.assertEqual(self.render.call_count, 1)

    def test_the_plan_records_what_was_asked_for_next_to_what_was_delivered(self):
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        av = plan["avatar"]
        self.assertEqual(av["coverage"], "intro")
        self.assertEqual(av["coverage_label"], "Intro only")
        self.assertEqual(av["delivered"], "the start")
        self.assertGreater(av["asked_seconds"], 0)

    def test_coverage_none_does_nothing(self):
        self.job.avatar_coverage = "none"
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        self.assertNotIn("avatar", plan)
        self.render.assert_not_called()

    def test_a_missing_channel_image_degrades_with_a_log(self):
        channels.avatar_path("c1", "a1").unlink()
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        self.assertNotIn("avatar", plan)
        self.assertTrue(any("image is gone" in m for m in self.logs))

    def test_a_render_failure_never_fails_the_job(self):
        self.render.side_effect = RuntimeError("boom")
        plan = {}
        pipeline._avatar_assets(self.job, plan)   # must not raise
        self.assertTrue(any("failed" in m for m in self.logs))

    def test_existing_segments_are_reused_not_re_rendered(self):
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        self.render.reset_mock()
        pipeline._avatar_assets(self.job, plan)
        self.render.assert_not_called()
        self.assertTrue(any("resume" in m for m in self.logs))

    def test_a_trim_is_reported(self):
        self.job.avatar_coverage = "both"
        self.job.avatar_model = "Wan 2.2 S2V"
        plan = {}
        with mock.patch.object(pipeline.avatar, "slice_audio", return_value=4.81):
            pipeline._avatar_assets(self.job, plan)
        self.assertTrue(any("presenter avatar" in m for m in self.logs))


    def test_a_presenter_is_pending_only_while_the_plan_has_no_segments(self):
        """request_assemble and _do_assemble both ask this, and have to agree: it decides
        whether a reassembly is pure ffmpeg or has to wait for the GPU."""
        self.assertTrue(pipeline._avatar_pending(self.job, {}))
        self.assertTrue(pipeline._avatar_pending(self.job, {"avatar": {"segments": []}}))
        self.assertFalse(pipeline._avatar_pending(
            self.job, {"avatar": {"segments": [{"file": "avatar_01.mp4"}]}}))
        self.job.avatar_coverage = "none"
        self.assertFalse(pipeline._avatar_pending(self.job, {}))
        self.job.avatar_coverage = "intro"
        self.job.avatar_id = ""
        self.assertFalse(pipeline._avatar_pending(self.job, {}))

    def test_a_clip_already_on_disk_is_reused_rather_than_re_rendered(self):
        """How a video that lost its presenter gets one back: the render survived in the job
        directory even though the failure meant the plan never recorded it."""
        (self.jdir / "avatar_01.mp4").write_bytes(b"clip")
        plan = {}
        pipeline._avatar_assets(self.job, plan)
        self.render.assert_not_called()
        self.assertEqual(plan["avatar"]["segments"][0]["file"], "avatar_01.mp4")


class ChainedChunks(unittest.TestCase):
    """A long appearance on a chunked engine is a chain of passes, each one conditioned on the
    tail of the one before it - the presenter carries his pose across the seam instead of
    snapping back to the portrait."""

    WAN = "Wan 2.2 S2V"

    def test_a_tile_is_as_long_as_the_clip_the_engine_delivers(self):
        """77 frames are rendered and 76 survive drop_first, so a 77-frame window would leave
        overlay repeating the last frame at every seam."""
        spec = config.avatar_model(self.WAN)
        self.assertAlmostEqual(spec["max_segment"], (spec["chunk"] - 1) / spec["fps"])

    def test_the_budget_buys_whole_tiles(self):
        spec = config.avatar_model(self.WAN)
        tiles = config.avatar_budget_seconds(self.WAN) / spec["max_segment"]
        self.assertAlmostEqual(tiles, round(tiles), places=6)

    def test_a_full_coverage_video_inside_the_budget_tiles_end_to_end(self):
        total = config.avatar_budget_seconds(self.WAN)
        wins = avatar.segment_windows("full", total, self.WAN, [])
        self.assertGreater(len(wins), 1)
        for a, b in zip(wins, wins[1:]):
            self.assertAlmostEqual(a[1], b[0], places=3)

    def test_the_chained_graph_is_a_separate_file(self):
        plain = config.avatar_workflow(self.WAN)
        chained = config.avatar_workflow(self.WAN, chain=True)
        self.assertNotEqual(plain, chained)
        self.assertTrue(chained.exists())

    def test_an_engine_that_cannot_chain_falls_back_to_its_own_graph(self):
        self.assertEqual(config.avatar_workflow("LTX 2.5", chain=True),
                         config.avatar_workflow("LTX 2.5"))

    def test_the_motion_clip_is_wired_into_ref_motion(self):
        from services import comfy
        tpl = comfy.ComfyClient.load_workflow(config.avatar_workflow(self.WAN, chain=True))
        wf = comfy.ComfyClient.build_avatar(tpl, self.WAN, image="p.png", audio="a.wav",
                                            seconds=4.75, filename_prefix="x",
                                            motion="tail.mp4")
        node, field = config.avatar_model(self.WAN)["motion"]
        self.assertEqual(wf[node]["inputs"][field], "tail.mp4")
        # ref_motion takes frames, so it is fed from the decoder - which is fed by the loader
        frames = wf["11"]["inputs"]["ref_motion"][0]
        self.assertEqual(wf[frames]["inputs"]["video"][0], node)

    def test_the_plain_graph_has_no_motion_at_all(self):
        """The first chunk of any appearance must render exactly as it did before."""
        from services import comfy
        tpl = comfy.ComfyClient.load_workflow(config.avatar_workflow(self.WAN))
        wf = comfy.ComfyClient.build_avatar(tpl, self.WAN, image="p.png", audio="a.wav",
                                            seconds=4.75, filename_prefix="x")
        self.assertNotIn("ref_motion", wf["11"]["inputs"])
        self.assertEqual(wf["11"]["inputs"]["ref_image"], ["7", 0])

    def test_the_two_graphs_differ_only_by_the_motion_input(self):
        from services import comfy
        plain = comfy.ComfyClient.load_workflow(config.avatar_workflow(self.WAN))
        chained = comfy.ComfyClient.load_workflow(config.avatar_workflow(self.WAN, chain=True))
        self.assertEqual(set(chained) - set(plain), {"16", "17"})
        for nid in plain:
            if nid != "11":
                self.assertEqual(plain[nid], chained[nid], nid)

    def test_the_tail_is_the_end_of_the_clip_and_no_longer_than_the_model_takes(self):
        spec = config.avatar_model(self.WAN)
        d = Path(tempfile.mkdtemp(prefix="tail-", dir=TMP))
        dest, calls = d / "t.mp4", []

        def _ffmpeg(cmd, **kw):
            calls.append(cmd)
            dest.write_bytes(b"x")

        with mock.patch.object(avatar.subprocess, "run", side_effect=_ffmpeg):
            avatar.motion_tail(d / "c1.mp4", dest, self.WAN)
        # resolving the ffmpeg binary runs its own probe, so pick ours out of the calls
        cmd = next(c for c in calls if isinstance(c, (list, tuple)) and "-sseof" in c)
        self.assertIn("-sseof", cmd)
        self.assertAlmostEqual(-float(cmd[cmd.index("-sseof") + 1]),
                               spec["motion_seconds"], places=3)
        self.assertLessEqual(spec["motion_seconds"], 73 / spec["fps"] + 1e-6)

    def test_an_engine_that_cannot_chain_refuses_to_make_a_tail(self):
        with self.assertRaises(RuntimeError):
            avatar.motion_tail(Path("c1.mp4"), Path("t.mp4"), "LTX 2.5")

    def test_render_segment_picks_the_graph_from_the_motion_clip(self):
        from services import comfy
        seen, real = {}, comfy.ComfyClient.load_workflow

        def _load(path):
            seen["path"] = Path(path).name
            return real(path)

        client = mock.MagicMock()
        with mock.patch.object(comfy.ComfyClient, "load_workflow", side_effect=_load):
            avatar.render_segment(client, self.WAN, "p.png", "a.wav", 4.75,
                                  Path("d.mp4"), prefix="x")
            self.assertEqual(seen["path"], config.avatar_model(self.WAN)["file"])
            avatar.render_segment(client, self.WAN, "p.png", "a.wav", 4.75,
                                  Path("d.mp4"), prefix="x", motion_name="t.mp4")
            self.assertEqual(seen["path"], config.avatar_model(self.WAN)["chain_file"])


class TheResultIsNeverTheGraphsOwnInput(unittest.TestCase):
    """LoadVideo reports the clip it opened in the prompt's outputs, tagged `type: "input"`, and
    it runs before SaveVideo. Taking the first file in the history would therefore hand the
    chained presenter back the tail it was given instead of the chunk it rendered - verified
    against the box: outputs carried `{"16": {"images": [{..., "type": "input"}]}}`."""

    def setUp(self):
        from services import comfy
        self.comfy = comfy
        self.client = comfy.ComfyClient()
        self.dir = Path(tempfile.mkdtemp(prefix="dl-", dir=TMP))
        self.history = {"outputs": {
            "16": {"images": [{"filename": "motion.mp4", "subfolder": "", "type": "input"}],
                   "animated": [True]},
            "15": {"videos": [{"filename": "render.mp4", "subfolder": "video",
                               "type": "output"}]},
        }}

    def _get(self, **kw):
        got = {}

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_content(self, _n):
                return [b"x"]

        def fake(url, params=None, **kwargs):
            got.update(params or {})
            return R()

        with mock.patch.object(self.comfy.requests, "get", side_effect=fake):
            self.client.download_output(self.history, self.dir / "out.mp4", **kw)
        return got

    def test_the_scan_skips_what_the_graph_only_read(self):
        self.assertEqual(self._get()["filename"], "render.mp4")

    def test_naming_the_save_node_is_honoured(self):
        self.assertEqual(self._get(node_id="15")["filename"], "render.mp4")

    def test_a_prompt_that_saved_nothing_still_raises(self):
        self.history["outputs"].pop("15")
        with self.assertRaises(self.comfy.ComfyError):
            self._get()

    def test_cleanup_leaves_the_input_alone(self):
        scrubbed = []
        with mock.patch.object(self.comfy.ComfyClient, "scrub_file",
                               side_effect=lambda t, s, f: scrubbed.append((t, f)) or True), \
                mock.patch.object(self.comfy.ComfyClient, "forget_history"):
            self.client.cleanup_outputs(self.history, "pid")
        self.assertEqual(scrubbed, [("output", "render.mp4")])

    def test_the_avatar_render_names_its_save_node(self):
        client = mock.MagicMock()
        avatar.render_segment(client, "Wan 2.2 S2V", "p.png", "a.wav", 4.75,
                              self.dir / "d.mp4", prefix="x")
        self.assertEqual(client.run.call_args.kwargs["node_id"],
                         config.avatar_model("Wan 2.2 S2V")["save"])


class ChainedChunksInThePipeline(unittest.TestCase):
    """_avatar_assets carries the tail from one tile to the next, and only within one
    appearance: a bookend pair is two separate appearances and must not be chained."""

    def setUp(self):
        self.jdir = Path(tempfile.mkdtemp(prefix="chainjob-", dir=TMP))
        (self.jdir / "narration.wav").write_bytes(b"x" * 2000)
        config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
        self.job = mock.MagicMock()
        self.job.dir = self.jdir
        self.job.id = "job9"
        self.job.channel_id = "c9"
        self.job.avatar_id = "a9"
        self.job.avatar_model = "Wan 2.2 S2V"
        self.job.avatar_coverage = "full"
        self.job.cancel_requested = False
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        channels.avatar_path("c9", "a9").write_bytes(b"png")
        # short enough that "full" tiles instead of degrading to bookends
        self.total = config.avatar_budget_seconds("Wan 2.2 S2V")
        mock.patch.object(pipeline.tts, "audio_duration", return_value=self.total).start()
        mock.patch.object(pipeline, "_gpu_gate").start()
        mock.patch.object(pipeline, "_save_plan").start()
        client = mock.patch.object(pipeline.comfy, "ComfyClient").start()
        client.return_value.upload_file.return_value = "uploaded.mp4"
        client.return_value.upload_image.return_value = "portrait.png"
        self.scrub = client.return_value.scrub_file
        mock.patch.object(pipeline.avatar, "slice_audio", return_value=4.8125).start()
        mock.patch.object(pipeline.avatar, "matte_segment").start()
        self.tail = mock.patch.object(pipeline.avatar, "motion_tail").start()
        self.render = mock.patch.object(pipeline.avatar, "render_segment").start()

        def _render(_c, _e, _i, _a, _s, dest, **kw):
            Path(dest).write_bytes(b"clip")
            return dest
        self.render.side_effect = _render
        mock.patch.object(
            pipeline.avatar, "trim_lead_frame",
            side_effect=lambda src, dst, eng: (Path(dst).write_bytes(b"c"), dst)[1]).start()
        self.addCleanup(mock.patch.stopall)

    def _motions(self):
        return [c.kwargs.get("motion_name", "") for c in self.render.call_args_list]

    def test_every_tile_but_the_first_continues_the_one_before_it(self):
        pipeline._avatar_assets(self.job, {})
        motions = self._motions()
        self.assertGreater(len(motions), 1)
        self.assertEqual(motions[0], "")
        self.assertTrue(all(m == "uploaded.mp4" for m in motions[1:]), motions)

    def test_the_tail_handed_over_is_the_previous_tile(self):
        pipeline._avatar_assets(self.job, {})
        srcs = [c.args[0].name for c in self.tail.call_args_list]
        self.assertEqual(srcs, [f"avatar_{i:02d}.mp4" for i in range(1, len(srcs) + 1)])

    def test_two_separate_appearances_are_not_chained(self):
        self.job.avatar_coverage = "both"
        with mock.patch.object(pipeline.tts, "audio_duration", return_value=300.0):
            pipeline._avatar_assets(self.job, {})
        self.assertEqual(self._motions(), ["", ""])
        self.tail.assert_not_called()

    def test_a_tail_that_cannot_be_cut_costs_only_the_continuity(self):
        self.tail.side_effect = RuntimeError("no keyframe")
        pipeline._avatar_assets(self.job, {})
        self.assertTrue(all(m == "" for m in self._motions()))
        self.assertTrue(any("could not continue" in m for m in self.logs))
        self.assertGreater(self.render.call_count, 1)      # the renders still happened

    def test_the_uploaded_tail_is_scrubbed_from_the_box(self):
        pipeline._avatar_assets(self.job, {})
        scrubbed = [c.args[2] for c in self.scrub.call_args_list]
        self.assertIn("uploaded.mp4", scrubbed)

    def test_a_resumed_chain_still_continues_from_what_is_on_disk(self):
        (self.jdir / "avatar_01.mp4").write_bytes(b"clip")
        pipeline._avatar_assets(self.job, {})
        self.assertEqual(self._motions()[0], "uploaded.mp4")   # tile 2 chained off the reused one
        self.assertEqual(self.tail.call_args_list[0].args[0].name, "avatar_01.mp4")


class ReassembleRouting(unittest.TestCase):
    """request_assemble: immediately in its own thread for pure ffmpeg work, but through the
    GPU queue when the video is still missing something that has to be rendered."""

    def setUp(self):
        for name in ("_loop", "_sweep_comfy", "_start_draft", "_assemble_now"):
            mock.patch.object(pipeline.JobManager, name).start()
        self.addCleanup(mock.patch.stopall)
        self.mgr = pipeline.JobManager()

    def _job(self, jid, plan=None, **kw):
        opts = {"status": "done", "avatar_id": "a1", "avatar_coverage": "intro"}
        opts.update(kw)
        job = pipeline.Job(id=jid, url="", clips=["clip_01.mp4"], **opts)
        job.save()
        if plan is not None:
            pipeline._save_plan(job, plan)
        self.mgr.jobs[jid] = job
        return job

    def test_a_pending_presenter_goes_to_the_front_of_the_gpu_queue(self):
        """The reason this matters: rendering a segment is ~12 min on the shared, serial box,
        so it cannot run from an unqueued side thread alongside whatever else is rendering."""
        job = self._job("asm-pending")
        self.mgr.request_assemble("asm-pending")
        self.assertEqual(self.mgr.queue[0], "asm-pending")
        self.assertEqual(job.status, "queued")
        self.assertTrue(job.assemble_requested)

    def test_a_presenter_already_on_the_plan_is_not_waited_for(self):
        job = self._job("asm-ready",
                        plan={"avatar": {"segments": [{"file": "avatar_01.mp4"}]}})
        self.mgr.request_assemble("asm-ready")
        self.assertNotIn("asm-ready", self.mgr.queue)
        self.assertEqual(job.status, "assembling")

    def test_a_video_that_never_wanted_a_presenter_is_not_waited_for(self):
        job = self._job("asm-nopresenter", avatar_coverage="none")
        self.mgr.request_assemble("asm-nopresenter")
        self.assertNotIn("asm-nopresenter", self.mgr.queue)
        self.assertEqual(job.status, "assembling")

    def test_only_a_finished_job_can_be_reassembled(self):
        self._job("asm-running", status="clips")
        with self.assertRaises(ValueError):
            self.mgr.request_assemble("asm-running")


class PortraitFraming(unittest.TestCase):
    """The reference image is fitted to the engine's square HERE, because ComfyUI's own fit is
    a centred cover-crop that takes the top off a tall headshot."""

    def setUp(self):
        from PIL import Image
        self.dir = TMP / "framing"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tall = self.dir / "tall.png"
        Image.new("RGB", (944, 1133), (20, 20, 20)).save(self.tall)

    def _fit(self, src, bounds):
        with mock.patch.object(avatar, "_subject_bounds", return_value=bounds):
            return avatar.portrait_square(src, self.dir / "out.png", "Wan 2.2 S2V")

    def test_the_fit_is_the_square_the_engine_renders(self):
        from PIL import Image
        out = self._fit(self.tall, (19, 60, 931, 1132))
        with Image.open(out) as im:
            self.assertEqual(im.size, (640, 640))

    def test_a_head_near_the_top_is_not_clipped(self):
        # the hologram case: 944x1133, hair starts at y=60. A centred crop would have taken
        # 94 px off the top; the window has to start at 0 instead.
        self.assertEqual(avatar._window(1133, 944, 60, 0.07), 0)

    def test_a_subject_lower_down_shifts_the_window_but_keeps_headroom(self):
        self.assertEqual(avatar._window(2000, 944, 400, 0.07), 400 - int(round(0.07 * 944)))

    def test_the_window_never_runs_past_the_image(self):
        self.assertEqual(avatar._window(1133, 944, 1100, 0.07), 1133 - 944)

    def test_no_matte_falls_back_to_the_top_rather_than_the_centre(self):
        self.assertEqual(avatar._window(1133, 944, None, 0.07), 0)

    def test_a_square_source_is_left_alone(self):
        from PIL import Image
        sq = self.dir / "sq.png"
        Image.new("RGB", (900, 900), (10, 10, 10)).save(sq)
        out = self._fit(sq, (0, 0, 899, 899))
        with Image.open(out) as im:
            self.assertEqual(im.size, (640, 640))

    def test_an_already_square_portrait_is_never_matted(self):
        """The whole point of squaring at upload: the render-path guard must then be free."""
        from PIL import Image
        with mock.patch.object(avatar, "_subject_bounds") as bounds:
            avatar.square(Image.new("RGB", (800, 800)))
        bounds.assert_not_called()

    def test_a_tall_portrait_is_squared(self):
        from PIL import Image
        with mock.patch.object(avatar, "_subject_bounds", return_value=(19, 60, 931, 1132)):
            out = avatar.square(Image.new("RGB", (944, 1133)))
        self.assertEqual(out.size, (944, 944))

    def test_an_unreadable_portrait_still_gets_a_presenter(self):
        """Framing is an improvement, not a requirement: it must not cost the whole overlay."""
        bad = self.dir / "bad.png"
        bad.write_bytes(b"not an image")
        with self.assertRaises(Exception):
            avatar.portrait_square(bad, self.dir / "out2.png", "Wan 2.2 S2V")


class ServerCleanup(unittest.TestCase):
    """Nothing the presenter stage puts on the shared box outlives its local copy."""

    def setUp(self):
        self.jdir = Path(tempfile.mkdtemp(prefix="job-", dir=TMP))
        (self.jdir / "narration.wav").write_bytes(b"x" * 2000)
        config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
        self.job = mock.MagicMock()
        self.job.dir = self.jdir
        self.job.id = "job9"
        self.job.channel_id, self.job.avatar_id = "c9", "a9"
        self.job.avatar_model, self.job.avatar_coverage = "LTX 2.5", "both"
        self.job.cancel_requested = False
        self.logs = []
        self.job.add_log.side_effect = self.logs.append
        channels.avatar_path("c9", "a9").write_bytes(b"png")
        mock.patch.object(pipeline.tts, "audio_duration", return_value=90.0).start()
        mock.patch.object(pipeline, "_gpu_gate").start()
        mock.patch.object(pipeline, "_save_plan").start()
        mock.patch.object(pipeline.avatar, "slice_audio", return_value=12.0).start()
        mock.patch.object(pipeline.avatar, "matte_segment").start()
        mock.patch.object(pipeline.avatar, "trim_lead_frame",
                          side_effect=lambda src, dst, eng: (Path(dst).write_bytes(b"c"), dst)[1]).start()
        r = mock.patch.object(pipeline.avatar, "render_segment").start()
        r.side_effect = lambda _c, _e, _i, _a, _s, dest, **kw: (Path(dest).write_bytes(b"clip"), dest)[1]
        self.client = mock.patch.object(pipeline.comfy, "ComfyClient").start().return_value
        self.client.upload_image.return_value = "videogen_avatar_job9.png"
        self.client.upload_file.side_effect = lambda w, name="": name
        self.addCleanup(mock.patch.stopall)

    def _scrubbed(self):
        return [c.args[2] for c in self.client.scrub_file.call_args_list]

    def test_each_narration_slice_is_scrubbed_once_its_clip_is_down(self):
        with mock.patch.object(config, "COMFY_CLEANUP", True):
            pipeline._avatar_assets(self.job, {})
        self.assertIn("videogen_avatar_job9_1.wav", self._scrubbed())

    def test_the_portrait_is_scrubbed_after_the_last_segment_not_before(self):
        with mock.patch.object(config, "COMFY_CLEANUP", True):
            pipeline._avatar_assets(self.job, {})
        scrubbed = self._scrubbed()
        self.assertIn("videogen_avatar_job9.png", scrubbed)
        # it is the reference image for every segment, so it must go last
        self.assertEqual(scrubbed[-1], "videogen_avatar_job9.png")

    def test_nothing_is_scrubbed_when_cleanup_is_off(self):
        with mock.patch.object(config, "COMFY_CLEANUP", False):
            pipeline._avatar_assets(self.job, {})
        self.client.scrub_file.assert_not_called()

    def test_the_rendered_clip_is_scrubbed_by_the_run_itself(self):
        """The output is cleaned up by ComfyClient.run, so render_segment has to ask for it -
        otherwise a 640x640 video sits on the shared box until the next orphan sweep."""
        from services import comfy as _comfy    # render_segment imports it lazily
        mock.patch.stopall()                      # this one needs the real render_segment
        client = mock.MagicMock()
        with mock.patch.object(_comfy.ComfyClient, "load_workflow", return_value={}),              mock.patch.object(_comfy.ComfyClient, "build_avatar", return_value={}),              mock.patch.object(config, "COMFY_CLEANUP", True):
            avatar.render_segment(client, "LTX 2.5", "i.png", "a.wav", 5.0,
                                  self.jdir / "raw.mp4", prefix="p")
        self.assertTrue(client.run.call_args.kwargs["cleanup"])


if __name__ == "__main__":
    unittest.main()
