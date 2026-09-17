"""Videos made from a GitHub repo or a Hugging Face model / dataset / space URL.

Covers the URL parser, the markdown stripper, the page-scroll filter and the draft flow. Nothing
here touches the network, ffmpeg or a browser - every boundary is mocked.

Run with  python -m unittest discover -s tests -t .  from the repo root."""
import json
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
        self.assertIn("overlay=72:130:enable='between(t,13.200,46.800)'", filt)

    def test_the_image_input_gets_the_video_frame_rate(self):
        argv = self._argv()
        self.assertIn("-framerate", argv)
        self.assertEqual(argv[argv.index("-framerate") + 1], "24.0000")
        self.assertIn("-loop", argv)

    def test_the_scroll_starts_at_the_readme(self):
        lyr = video.scroll_layer(TMP / "base.mp4", self.png, 13.2, 46.8, content_y=2460)
        self.assertNotIn("y=0:", lyr["pre"])       # not the top of the page
        self.assertIn("crop=w=336:h=604:x=0:y='7", lyr["pre"])   # scaled offset, ~765

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
            "render_all": mock.patch.object(pipeline.storyboard, "render_all",
                                            return_value=[(TMP / "card_01.png", 2.0, 8.0)]),
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


if __name__ == "__main__":
    unittest.main()
