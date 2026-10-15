"""Videos made from a GitHub repo or a Hugging Face model / dataset / space URL.

Covers the URL parser, the markdown stripper, the page-scroll filter and the draft flow. Nothing
here touches the network, ffmpeg or a browser - every boundary is mocked.

Run with  python -m unittest discover -s tests -t .  from the repo root."""
import json
import pathlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-repo-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from pipeline import Job  # noqa: E402
from services import repo, video  # noqa: E402

TECH = {"id": "t1", "name": "Daily Tech Talks", "purpose": "AI tools", "owner": "u1", "rules": "",
        "audience": "general", "video_type": "tech", "cast": [], "cast_opening": True,
        "characters_enabled": False, "defaults": {}}
KIDS = {**TECH, "id": "k1", "name": "Bright Minds", "audience": "kids", "video_type": "animated"}


def _png(path: Path, w: int = 1080, h: int = 6000) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), "white").save(path, "PNG")
    return path


class ParseUrl(unittest.TestCase):
    def test_accepted(self):
        cases = [
            ("https://github.com/psf/requests", "github", "psf/requests"),
            ("https://github.com/psf/requests.git", "github", "psf/requests"),
            ("https://www.github.com/psf/requests/tree/main/src", "github", "psf/requests"),
            ("https://github.com/psf/requests/blob/main/README.md", "github", "psf/requests"),
            ("https://github.com/psf/requests/issues/1", "github", "psf/requests"),
            ("git@github.com:psf/requests.git", "github", "psf/requests"),
            ("github.com/psf/requests/", "github", "psf/requests"),
            ("https://huggingface.co/openai/whisper-large-v3", "hf_model", "openai/whisper-large-v3"),
            ("https://huggingface.co/models/openai/whisper-large-v3", "hf_model", "openai/whisper-large-v3"),
            ("https://huggingface.co/bert-base-uncased", "hf_model", "bert-base-uncased"),
            ("https://hf.co/bert-base-uncased", "hf_model", "bert-base-uncased"),
            ("https://huggingface.co/datasets/rajpurkar/squad", "hf_dataset", "rajpurkar/squad"),
            ("https://huggingface.co/spaces/gradio/hello_world", "hf_space", "gradio/hello_world"),
            ("https://huggingface.co/openai/whisper-large-v3/blob/main/config.json",
             "hf_model", "openai/whisper-large-v3"),
        ]
        for url, kind, rid in cases:
            with self.subTest(url=url):
                out = repo.parse_url(url)
                self.assertEqual((out["kind"], out["id"]), (kind, rid))

    def test_rejected(self):
        for url in ("https://youtube.com/watch?v=abc", "https://gitlab.com/a/b",
                    "https://github.com/psf", "", "   ", "nonsense", "https://example.com/x/y"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                repo.parse_url(url)

    def test_normalize_returns_the_canonical_page(self):
        self.assertEqual(repo.normalize_url("github.com/psf/requests.git"),
                         "https://github.com/psf/requests")


class MarkdownToText(unittest.TestCase):
    def test_strips_card_front_matter(self):
        out = repo.markdown_to_text("---\nlicense: mit\ntags:\n- a\n---\nReal content here.")
        self.assertEqual(out, "Real content here.")
        self.assertNotIn("license", out)

    def test_strips_code_badges_and_tags(self):
        md = ('# Title\n\n[![build](https://img.shields.io/x.svg)](https://ci.example)\n\n'
              'See [the docs](https://example.com) for more.\n\n'
              '```python\nprint("hello")\n```\n\n<p align="center">centered</p>\n\n'
              '| a | b |\n| --- | --- |\n\n&amp; done')
        out = repo.markdown_to_text(md)
        self.assertIn("Title", out)
        self.assertIn("the docs", out)          # link text survives
        self.assertNotIn("img.shields.io", out)  # badge does not
        self.assertNotIn("print", out)           # fenced code does not
        self.assertNotIn("<p", out)
        self.assertNotIn("| a |", out)
        self.assertIn("& done", out)             # entity unescaped

    def test_an_unterminated_angle_bracket_cannot_hang(self):
        out = repo.markdown_to_text("text < " + ("x" * 5000) + " end")
        self.assertIn("end", out)


class FactsAndGate(unittest.TestCase):
    META = {"full_name": "psf/requests", "description": "An HTTP library.", "language": "Python",
            "topics": ["http"], "stargazers_count": 54307, "forks_count": 10173,
            "open_issues_count": 238, "license": {"spdx_id": "Apache-2.0"},
            "pushed_at": "2026-01-02T00:00:00Z"}

    def test_fact_block_then_readme(self):
        text = repo.build_text(repo._facts_github(self.META), "# Requests\n\nIt does HTTP.")
        self.assertTrue(text.startswith("GitHub repository: psf/requests"))
        self.assertIn("Stars: 54,307", text)
        self.assertIn("License: Apache-2.0", text)
        self.assertLess(text.index("Stars"), text.index("README / MODEL CARD"))
        self.assertIn("It does HTTP.", text)

    def test_capped_at_the_transcript_limit(self):
        text = repo.build_text(["x"], "y " * 200000)
        self.assertLessEqual(len(text), config.MAX_TRANSCRIPT_CHARS)

    def test_a_thin_readme_is_refused(self):
        with mock.patch.object(repo, "_get") as get, \
             mock.patch.object(repo, "_readme", return_value=("tiny readme", "")):
            get.return_value = mock.Mock(json=lambda: {"full_name": "a/b"})
            with self.assertRaises(RuntimeError) as cm:
                repo.fetch("https://github.com/a/b")
        self.assertIn("120", str(cm.exception))      # names the threshold

    def test_a_real_readme_is_accepted_and_canonicalised(self):
        with mock.patch.object(repo, "_get") as get, \
             mock.patch.object(repo, "_readme", return_value=("word " * 300, "")):
            get.return_value = mock.Mock(json=lambda: {"full_name": "ggml-org/llama.cpp",
                                                       "html_url": "https://github.com/ggml-org/llama.cpp"})
            out = repo.fetch("https://github.com/ggerganov/llama.cpp")
        self.assertEqual(out["id"], "ggml-org/llama.cpp")        # follows the rename
        self.assertEqual(out["page_url"], "https://github.com/ggml-org/llama.cpp")
        self.assertGreater(out["words"], repo.MIN_WORDS)


class ScrollWindow(unittest.TestCase):
    def test_middle_of_the_video(self):
        self.assertEqual(video.scroll_window(60), (13.2, 46.8))

    def test_short_videos_are_skipped(self):
        self.assertIsNone(video.scroll_window(18))
        self.assertIsNone(video.scroll_window(0))

    def test_always_inside_the_video(self):
        for total in range(20, 300, 7):
            win = video.scroll_window(total)
            self.assertIsNotNone(win)
            self.assertLess(0, win[0])
            self.assertLess(win[0], win[1])
            self.assertLess(win[1], total)

    def test_avoids_the_story_board_cards(self):
        win = video.scroll_window(60, busy=[(10.0, 30.0)])
        self.assertIsNotNone(win)
        self.assertGreaterEqual(win[0], 30.0)      # pushed past the card

    def test_none_when_the_cards_leave_no_gap(self):
        self.assertIsNone(video.scroll_window(60, busy=[(0.0, 60.0)]))

    def test_evenly_spread_cards_leave_no_gap_on_a_short_video(self):
        """The reported bug, frozen: three cards on clips 2, 4 and 6 of a 40s video.

        Not every short schedule fails - about 29% of the three-card ones do - and it is the
        evenly spread arrangements, which is what the LLM tends to write. A test built on "three
        cards somewhere" would pass against the old code and prove nothing."""
        cards = [(5.35, 9.65), (15.35, 19.65), (25.35, 29.65)]
        polish = [(0.0, 3.2), (36.0, 40.0)]
        self.assertIsNone(video.scroll_window(40, busy=cards + polish))
        # ... while a clustered schedule on the same video still finds its own gap
        self.assertIsNotNone(video.scroll_window(
            40, busy=[(0.35, 4.65), (15.35, 19.65), (30.35, 34.65)] + polish))


class ScrollReserve(unittest.TestCase):
    """The fallback when the cards leave no gap: take a window and move the cards instead."""

    CARDS = [(5.35, 9.65), (15.35, 19.65), (25.35, 29.65)]
    POLISH = [(0.0, 3.2), (36.0, 40.0)]

    def test_the_reported_case_gets_a_window(self):
        win = video.scroll_reserve(40, soft=self.CARDS, hard=self.POLISH)
        self.assertIsNotNone(win)
        self.assertAlmostEqual(win[1] - win[0], video.SCROLL_MIN_WINDOW, places=3)
        self.assertLessEqual(8.8, win[0])
        self.assertLessEqual(win[1], 31.2)

    def test_it_overlaps_the_cards_as_little_as_possible(self):
        """The three gaps inside the band are 5.7s, 5.7s and 1.55s, all under the window, so some
        overlap is unavoidable - 0.3s is the minimum, and the exact candidate set must find it
        rather than landing mid-card the way a 0.1s grid sweep would."""
        win = video.scroll_reserve(40, soft=self.CARDS, hard=self.POLISH)
        self.assertLessEqual(video._busy_overlap(win[0], win[1], self.CARDS), 0.31)

    def test_a_real_gap_costs_no_card_time(self):
        soft = [(9.0, 13.0), (25.0, 29.0)]       # a 12s hole in the middle
        win = video.scroll_reserve(40, soft=soft, hard=self.POLISH)
        self.assertEqual(video._busy_overlap(win[0], win[1], soft), 0.0)

    def test_the_hook_and_end_cards_are_never_covered(self):
        """They are composited after the panel, so a panel underneath one is simply invisible -
        and unlike a fact card they cannot be trimmed. END_CARD_SECONDS goes up to 10, which on a
        20s video starts the end card at 10.0, inside the band."""
        self.assertIsNone(video.scroll_reserve(20, hard=[(0.0, 3.2), (10.0, 20.0)]))
        win = video.scroll_reserve(20, hard=[(0.0, 3.2), (16.0, 20.0)])
        self.assertIsNotNone(win)
        self.assertGreaterEqual(win[0], 3.2)
        self.assertLessEqual(win[1], 16.0)

    def test_short_videos_are_still_skipped(self):
        self.assertIsNone(video.scroll_reserve(18))
        self.assertIsNone(video.scroll_reserve(0))

    def test_a_blocked_middle_still_yields_a_window_from_the_floor_up(self):
        """What makes the panel a guarantee: the band is 0.56*total, so it is 11.2s wide at the
        20s floor and the window always fits however busy the middle is."""
        for total in range(20, 300, 7):
            win = video.scroll_reserve(total, soft=[(0.0, float(total))])
            self.assertIsNotNone(win, f"no window at {total}s")
            self.assertLess(0, win[0])
            self.assertLess(win[0], win[1])
            self.assertLess(win[1], total)


class KilledFfmpeg(unittest.TestCase):
    """The reported job lost its overlays, and before that a whole three-hour run, to an ffmpeg
    that exited non-zero with an empty stderr - the signature of a process the kernel killed.
    The message said only "ffmpeg failed", and the identical command succeeded when the user
    re-ran it by hand."""

    def _result(self, code, err=""):
        return mock.Mock(returncode=code, stderr=err, stdout="")

    def test_a_signal_is_named_as_a_kill_and_a_memory_problem(self):
        with mock.patch.object(video.subprocess, "run", return_value=self._result(-9)),                 mock.patch.object(video, "KILL_RETRIES", 0):
            with self.assertRaises(RuntimeError) as cm:
                video._run(["ffmpeg", "-i", "x"])
        self.assertIn("killed by signal 9", str(cm.exception))
        self.assertIn("ran out of memory", str(cm.exception))

    def test_dockers_137_is_the_same_kill(self):
        self.assertEqual(video._killed(137), 9)
        self.assertEqual(video._killed(143), 15)
        self.assertEqual(video._killed(1), 0)      # an ordinary ffmpeg error
        self.assertEqual(video._killed(0), 0)

    def test_a_kill_is_tried_once_more(self):
        runs = [self._result(-9), self._result(0)]
        with mock.patch.object(video.subprocess, "run", side_effect=runs) as run,                 mock.patch.object(video.time, "sleep"):
            video._run(["ffmpeg", "-i", "x"])
        self.assertEqual(run.call_count, 2)

    def test_an_ffmpeg_error_is_never_retried(self):
        # retrying a bad filter graph only pays the wait twice and hides the real message
        bad = self._result(1, "Error reinitializing filters!")
        with mock.patch.object(video.subprocess, "run", return_value=bad) as run:
            with self.assertRaises(RuntimeError) as cm:
                video._run(["ffmpeg", "-i", "x"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("exit code 1", str(cm.exception))
        self.assertIn("reinitializing filters", str(cm.exception))

    def test_a_kill_that_keeps_happening_still_fails(self):
        with mock.patch.object(video.subprocess, "run", return_value=self._result(-9)) as run,                 mock.patch.object(video.time, "sleep"):
            with self.assertRaises(RuntimeError):
                video._run(["ffmpeg", "-i", "x"])
        self.assertEqual(run.call_count, video.KILL_RETRIES + 1)


class PageContentOffset(unittest.TestCase):
    """Where the README starts has to outlive the run that captured it. The reported job lost it
    to a failed draft: the screenshot survived, `_page_screenshot` returned 0 on the reuse path,
    and the panel started on the page header for every later run."""

    def setUp(self):
        config.OUTPUT_DIR = TMP / "offset-output"
        self.job = Job(id="20260101-000000-offs01", url="https://github.com/a/b")
        self.job.dir.mkdir(parents=True, exist_ok=True)
        for name in ("page.png", "page.json"):
            (self.job.dir / name).unlink(missing_ok=True)

    def _shoot(self, content_y):
        def fake(url, png):
            _png(pathlib.Path(png))
            return 1080, 6000, content_y
        with mock.patch.object(pipeline.browser, "screenshot", side_effect=fake):
            return pipeline._page_screenshot(self.job)

    def test_the_offset_is_kept_beside_the_picture(self):
        self.assertEqual(self._shoot(2460), 2460)
        meta = json.loads((self.job.dir / "page.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["content_y"], 2460)

    def test_a_reused_screenshot_keeps_its_offset(self):
        self._shoot(2460)
        # a second run finds page.png already there and never calls the browser again
        with mock.patch.object(pipeline.browser, "screenshot",
                               side_effect=AssertionError("must not re-shoot")):
            self.assertEqual(pipeline._page_screenshot(self.job), 2460)
        self.assertTrue(any("reusing the page screenshot" in str(e) for e in self.job.log))

    def test_a_screenshot_captured_before_the_sidecar_existed_still_works(self):
        _png(self.job.dir / "page.png")
        with mock.patch.object(pipeline.browser, "screenshot",
                               side_effect=AssertionError("must not re-shoot")):
            self.assertEqual(pipeline._page_screenshot(self.job), 0)


class ScrollFilter(unittest.TestCase):
    """The three traps: per-frame crop, quoted commas, and the image input's frame rate."""

    def setUp(self):
        self.png = _png(TMP / "page.png")
        self.patches = [
            mock.patch.object(video, "_run"),
            mock.patch.object(video, "duration", return_value=60.0),
            mock.patch.object(video, "frame_size", return_value=(480, 864)),
            mock.patch.object(video, "fps", return_value=24.0),
            mock.patch.object(video, "has_audio", return_value=True),
        ]
        self.m = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def _argv(self):
        lyr = video.scroll_layer(TMP / "base.mp4", self.png, 13.2, 46.8, content_y=2460)
        video.composite(TMP / "base.mp4", [lyr], TMP / "out.mp4")
        return self.m[0].call_args.args[0]

    def test_panel_is_70_percent_and_centred(self):
        lyr = video.scroll_layer(TMP / "base.mp4", self.png, 13.2, 46.8)
        self.assertEqual((lyr["x"], lyr["y"]), ((480 - 336) // 2, (864 - 604) // 2))
        self.assertIn("crop=w=336:h=604", lyr["pre"])

    def test_the_crop_y_is_a_quoted_expression_of_t(self):
        argv = self._argv()
        filt = argv[argv.index("-filter_complex") + 1]
        self.assertIn("crop=w=336:h=604:x=0:y='", filt)
        self.assertIn("clip((t-13.200)/33.600,0,1)", filt)   # window-relative, commas quoted
        # an integer position is still written bare, so the panel's centring is unaffected by
        # the story board cards' per-frame motion expressions
        self.assertIn("overlay=72:130:eval=frame:enable='between(t,13.200,46.800)'", filt)

    def test_the_image_input_gets_the_video_frame_rate(self):
        argv = self._argv()
        self.assertIn("-framerate", argv)
        self.assertEqual(argv[argv.index("-framerate") + 1], "24.0000")
        self.assertIn("-loop", argv)

    def test_the_scroll_starts_at_the_readme(self):
        lyr = video.scroll_layer(TMP / "base.mp4", self.png, 13.2, 46.8, content_y=2460)
        self.assertNotIn("y=0:", lyr["pre"])       # not the top of the page
        self.assertIn("crop=w=336:h=604:x=0:y='7", lyr["pre"])   # scaled offset, ~765

    def test_the_travel_follows_the_configured_speed(self):
        # the reported complaint: at 55 px/s an 11 s window moved the panel a third of its own
        # height and the viewer read it as a screenshot. The travel is speed x window, always -
        # the page is far taller than any window can cover - so the speed is the whole control.
        def travel(speed):
            with mock.patch.object(config, "PAGE_SCROLL_SPEED", speed):
                pre = video.scroll_layer(TMP / "base.mp4", self.png, 41.0, 52.0)["pre"]
            return int(pre.split("y='0+")[1].split("*")[0])

        self.assertEqual(travel(55.0), 605)          # what the videos shipped with
        self.assertEqual(travel(110.0), 1210)        # the new default: two thirds of the panel
        self.assertGreater(travel(config.PAGE_SCROLL_SPEED), travel(55.0))

    def test_a_window_can_never_outrun_the_page(self):
        # a short page and a long window: the geometry still binds, and no negative travel
        short = _png(TMP / "short.png", 1080, 700)
        with mock.patch.object(config, "PAGE_SCROLL_SPEED", 300.0):
            pre = video.scroll_layer(TMP / "base.mp4", short, 0.0, 60.0)["pre"]
        self.assertIn("crop=w=336:h=604:x=0:y=0", pre)   # nothing to scroll: a bare position

    def test_audio_is_copied_and_no_layers_is_a_passthrough(self):
        argv = self._argv()
        self.assertIn("-c:a", argv)
        self.m[0].reset_mock()
        video.composite(TMP / "base.mp4", [], TMP / "out.mp4")
        self.assertIn("copy", self.m[0].call_args.args[0])


class AssembleOrder(unittest.TestCase):
    """One composite pass, with the page scroll underneath the story board cards."""

    def setUp(self):
        config.OUTPUT_DIR = TMP / "output"
        self.job = Job(id="20260101-000000-repo01", url="https://github.com/a/b",
                       clips=["clip_01.mp4", "clip_02.mp4"])
        self.job.dir.mkdir(parents=True, exist_ok=True)
        (self.job.dir / 'page.png').unlink(missing_ok=True)   # these tests share one job dir
        patches = {
            "concat_clips": mock.patch.object(pipeline.video, "concat_clips",
                                              side_effect=lambda clips, out: Path(out)),
            "clip_boundaries": mock.patch.object(pipeline.video, "clip_boundaries",
                                                 return_value=[30.0, 60.0]),
            "frame_size": mock.patch.object(pipeline.video, "frame_size", return_value=(480, 864)),
            "mix_audio": mock.patch.object(pipeline.video, "mix_audio",
                                           side_effect=lambda src, out, **kw: Path(out)),
            "composite": mock.patch.object(pipeline.video, "composite",
                                           side_effect=lambda v, layers, out, **kw: Path(out)),
            "render_all": mock.patch.object(
                pipeline.storyboard, "render_all",
                return_value=[{"png": TMP / "card_01.png", "start": 2.0, "end": 8.0,
                               "x": 0, "y": 0, "template": "lower_third"}]),
        }
        self.m = {k: p.start() for k, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_scroll_layer_is_first_so_cards_draw_on_top(self):
        _png(self.job.dir / "page.png")
        self.job.page_scroll = True
        self.job.storyboard = True
        pipeline._assemble(self.job, {"storyboard": [{}]}, None, None)
        layers = self.m["composite"].call_args.args[1]
        self.assertEqual(len(layers), 2)
        self.assertIn("crop=", layers[0].get("pre", ""))      # the scroll comes first
        self.assertEqual(layers[1].get("pre"), None)          # the card has no pre-chain
        self.assertEqual(self.m["mix_audio"].call_args.args[0],
                         self.job.dir / "clips_overlay.mp4")

    def test_cards_alone_still_composite(self):
        self.job.storyboard = True
        pipeline._assemble(self.job, {"storyboard": [{}]}, None, None)
        self.assertEqual(len(self.m["composite"].call_args.args[1]), 1)

    def test_no_page_png_means_no_scroll_and_no_probe(self):
        self.job.page_scroll = True
        with mock.patch.object(pipeline.video, "duration") as dur:
            pipeline._assemble(self.job, {}, None, None)
            dur.assert_not_called()               # must not ffprobe a file that is not there
        self.m["composite"].assert_not_called()
        self.assertEqual(self.m["mix_audio"].call_args.args[0],
                         self.job.dir / "clips_joined.mp4")

    def test_a_failing_overlay_never_loses_the_video(self):
        _png(self.job.dir / "page.png")
        self.job.page_scroll = True
        self.m["composite"].side_effect = RuntimeError("ffmpeg exploded")
        pipeline._assemble(self.job, {}, None, None)
        self.assertEqual(self.m["mix_audio"].call_args.args[0],
                         self.job.dir / "clips_joined.mp4")
        self.assertTrue(any("overlay failed" in str(e) for e in self.job.log))

    def test_a_missing_page_screenshot_says_so(self):
        """The screenshot is only ever taken while the script is drafted, so a job reassembled
        without one can never grow a panel. It used to skip in total silence."""
        self.job.page_scroll = True
        pipeline._assemble(self.job, {}, None, None)
        self.assertTrue(any("no page screenshot" in str(e) for e in self.job.log))

    def _crowded(self):
        """A 40s video, 8 clips of 5s, cards on clips 2, 4 and 6 - the reported schedule."""
        self.m["clip_boundaries"].return_value = [5.0 * i for i in range(1, 9)]
        self.m["render_all"].return_value = [
            {"png": TMP / f"card_{i}.png", "start": s, "end": e,
             "x": 0, "y": 0, "template": "lower_third"}
            for i, (s, e) in enumerate([(5.35, 9.65), (15.35, 19.65), (25.35, 29.65)], 1)]
        _png(self.job.dir / "page.png")
        self.job.page_scroll = True
        self.job.storyboard = True
        self.job.polish = True
        # keeps _polish_windows real (hook_end=3.2, end_start=36.0) without the PIL card work
        return mock.patch.object(pipeline, "_polish_layers", return_value=([], []))

    def test_a_crowded_middle_reserves_the_panel_and_moves_the_cards(self):
        with self._crowded():
            pipeline._assemble(self.job, {"storyboard": [{}, {}, {}]}, None, None)
        layers = self.m["composite"].call_args.args[1]
        self.assertIn("crop=", layers[0].get("pre", ""))        # the panel, underneath
        cards = layers[1:]
        self.assertEqual(len(cards), 3)                         # every card survived
        win = (layers[0]["start"], layers[0]["end"])
        for c in cards:
            self.assertFalse(c["end"] > win[0] and c["start"] < win[1],
                             f"card {c['start']}-{c['end']} draws under the panel {win}")
        self.assertTrue(any("page scroll" in str(e) for e in self.job.log))

    def test_the_page_scroll_suffix_only_appears_with_a_panel(self):
        """`len(layers) > len(cards)` counted the polish layers, so the log claimed a page scroll
        on any video with polish on - whether or not a panel was ever drawn."""
        self.job.storyboard = True
        self.job.polish = True
        with mock.patch.object(pipeline, "_polish_layers",
                               return_value=([{"png": TMP / "hook.png", "start": 0.0,
                                               "end": 3.0, "x": 0, "y": 0}], [])):
            pipeline._assemble(self.job, {"storyboard": [{}]}, None, None)
        overlays = [str(e) for e in self.job.log if "overlays:" in str(e)]
        self.assertTrue(overlays)
        self.assertNotIn("page scroll", overlays[-1])


class _Sync:
    """threading.Thread stand-in: run the target inline so the test is deterministic."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class DraftFetch(unittest.TestCase):
    def setUp(self):
        config.OUTPUT_DIR = TMP / "output"
        self.mgr = pipeline.JobManager()
        self.fetched = {"kind": "github", "id": "psf/requests", "owner": "psf", "name": "requests",
                        "page_url": "https://github.com/psf/requests",
                        "text": "GitHub repository: psf/requests\n\nREADME / MODEL CARD\n\n" + "word " * 300,
                        "meta": {"page_url": "https://github.com/psf/requests"}, "words": 305}
        self.plan = {"title": "t", "narration": "You use HTTP every day. " * 20, "style": "s",
                     "issues": []}

    def _run(self, screenshot=None, fetch=None):
        patches = [
            mock.patch.object(pipeline.threading, "Thread", _Sync),
            mock.patch.object(pipeline.repo, "fetch", fetch or mock.Mock(return_value=self.fetched)),
            mock.patch.object(pipeline.browser, "screenshot",
                              screenshot or mock.Mock(return_value=(1080, 6000, 2460))),
            mock.patch.object(pipeline.llm, "write_script_from_prompt",
                              mock.Mock(return_value=dict(self.plan))),
            mock.patch.object(config, "SCRIPT_REVIEW", False),
        ]
        started = [p.start() for p in patches]
        try:
            job = self.mgr.submit_prompt("https://github.com/psf/requests", "repo",
                                         channel=TECH, owner="u1")
            return job, started[1], started[2], started[3]
        finally:
            for p in patches:
                p.stop()

    def test_the_llm_gets_the_readme_not_the_url(self):
        job, fetch, shot, write = self._run()
        self.assertEqual(job.status, "awaiting_approval")
        self.assertIn("README / MODEL CARD", write.call_args.args[0])
        self.assertEqual(write.call_args.args[1], "repo")
        self.assertEqual(job.url, "https://github.com/psf/requests")
        self.assertTrue(job.page_scroll)
        self.assertTrue((job.dir / "transcript.txt").exists())
        self.assertTrue((job.dir / "repo.json").exists())
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["page_content_y"], 2460)

    def test_a_failed_screenshot_is_not_fatal(self):
        job, _, _, _ = self._run(screenshot=mock.Mock(side_effect=RuntimeError("no chrome")))
        self.assertEqual(job.status, "awaiting_approval")
        self.assertFalse((job.dir / "page.png").exists())
        self.assertTrue(any("page screenshot failed" in str(e) for e in job.log))

    def test_a_failed_fetch_fails_the_job(self):
        job, _, _, _ = self._run(fetch=mock.Mock(side_effect=RuntimeError("404")))
        self.assertEqual(job.status, "failed")

    def test_a_redraft_does_not_refetch(self):
        job, fetch, _, _ = self._run()
        self.assertEqual(fetch.call_count, 1)
        with mock.patch.object(pipeline.threading, "Thread", _Sync), \
             mock.patch.object(pipeline.repo, "fetch", fetch), \
             mock.patch.object(pipeline.browser, "screenshot", mock.Mock(return_value=(1, 1, 0))), \
             mock.patch.object(pipeline.llm, "write_script_from_prompt",
                               mock.Mock(return_value=dict(self.plan))), \
             mock.patch.object(config, "SCRIPT_REVIEW", False):
            self.mgr.redraft_script(job.id, feedback="shorter please")
        self.assertEqual(fetch.call_count, 1)      # the cached README is reused


class TechOnly(unittest.TestCase):
    def test_a_non_tech_channel_is_refused(self):
        mgr = pipeline.JobManager()
        with self.assertRaises(ValueError):
            mgr.submit_prompt("https://github.com/psf/requests", "repo", channel=KIDS, owner="u1")

    def test_a_bad_url_is_refused_at_submit(self):
        mgr = pipeline.JobManager()
        with self.assertRaises(ValueError):
            mgr.submit_prompt("https://youtube.com/watch?v=x", "repo", channel=TECH, owner="u1")


class ViewerDocument(unittest.TestCase):
    """Which document a repo video ships, and which one everything else ships.

    A promised resource always wins; the guide is the fallback, and only for a repo job."""

    GUIDE = {"kind": "setup guide", "title": "Run It", "markdown": "# Run It\n\nbody"}
    PROMISED = {"promise": "check the description", "kind": "checklist", "title": "C",
                "markdown": "# C\n\nbody"}

    def setUp(self):
        config.OUTPUT_DIR = TMP / "output"
        self.job = Job(id="20260101-000000-doc001", url="https://github.com/a/b",
                       source="repo", channel=TECH)
        self.job.dir.mkdir(parents=True, exist_ok=True)
        (self.job.dir / "repo.json").write_text(
            json.dumps({"page_url": "https://github.com/a/b"}), encoding="utf-8")
        self.plan = {"title": "T", "narration": "the script", "seo_description": "Body.\n\n#ai"}

    def _run(self, promised, guide=None):
        with mock.patch.object(pipeline.llm, "write_resource", return_value=promised) as wr, \
             mock.patch.object(pipeline.llm, "write_repo_guide",
                               return_value=guide or self.GUIDE) as wg:
            out = pipeline._viewer_document(self.job, self.plan, "README text")
        return out, wr, wg

    def test_a_repo_video_with_no_promise_gets_the_guide(self):
        (res, origin, url), _, wg = self._run({})
        self.assertEqual(res, self.GUIDE)
        self.assertEqual(origin, pipeline.resource.GUIDE)
        self.assertEqual(url, "https://github.com/a/b")          # read back off repo.json
        self.assertEqual(wg.call_args.args[2], "README text")    # written from the README
        self.assertEqual(wg.call_args.kwargs["source_url"], "https://github.com/a/b")

    def test_a_promise_wins_so_there_is_never_a_second_document(self):
        (res, origin, url), _, wg = self._run(self.PROMISED)
        self.assertEqual(res, self.PROMISED)
        self.assertEqual(origin, pipeline.resource.PROMISE)
        self.assertEqual(url, "")
        wg.assert_not_called()

    def test_a_non_repo_video_never_gets_a_guide(self):
        self.job.source = "topic"
        (res, origin, _), _, wg = self._run({})
        self.assertEqual(res, {})
        self.assertEqual(origin, pipeline.resource.PROMISE)
        wg.assert_not_called()

    def test_a_missing_repo_json_costs_only_the_link(self):
        (self.job.dir / "repo.json").unlink()
        (res, origin, url), _, _ = self._run({})
        self.assertEqual(res, self.GUIDE)                        # still written
        self.assertEqual(origin, pipeline.resource.GUIDE)
        self.assertEqual(url, "")

    def test_the_guide_is_not_squeezed_into_the_description_budget(self):
        """write_resource is capped by what is left of the description; a guide is a download."""
        _, wr, wg = self._run({})
        self.assertLess(wr.call_args.kwargs["max_chars"], pipeline.resource.GUIDE_MAX_CHARS)
        self.assertEqual(wg.call_args.kwargs["max_chars"], pipeline.resource.GUIDE_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
