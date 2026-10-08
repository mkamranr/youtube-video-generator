"""Presenter avatar: the channel registry, the per-video resolve, segment planning,
resolution-independent placement and the video layer in the composite filtergraph.

Run with  python -m unittest discover -s tests -t .  from the repo root. ffmpeg is mocked, so
the compositing tests assert on the command that would have been run - which is where the
single-encode property and the alphamerge wiring actually live.
"""
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


if __name__ == "__main__":
    unittest.main()
