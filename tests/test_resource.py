"""The document that goes out with a video.

Two of them exist and they arrive for different reasons: a PROMISE KEPT, when a script says "the
exact prompt structure is in the description" and has to hand that prompt over, and a SETUP GUIDE,
which every video made from a GitHub repo or a Hugging Face page ships whether or not the script
promised anything.

These cover what can go wrong in each: deciding whether a promise was really made, writing a guide
only from the README it was given, turning either answer into files, and folding a promised
resource into the YouTube description exactly once - while never folding a guide into it at all.

Run with  python -m unittest discover -s tests -t .  from the repo root."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-resource-"))

from services import llm, resource  # noqa: E402

NARRATION = ("You are wasting four hours a day writing the same emails. Here is how to stop. "
             "If you want the exact prompt structure we used, check the link in the description. "
             "We break down the specific keywords that make this technique work.")

RES = {"offered": True, "kind": "prompt structure", "title": "The Inbox Prompt",
       "promise": "check the link in the description",
       "markdown": "# The Inbox Prompt\n\n## Use it\n\n- [ ] paste your thread\n\n"
                   "```\nRole: assistant\nTone: <professional but friendly>\n```\n"}


def _capture(reply):
    calls = []

    def fake(messages, max_tokens=8000):
        calls.append(messages)
        return dict(reply)
    return calls, fake


class Decide(unittest.TestCase):
    """The promise must be real. An invented one is the bug this feature exists to prevent."""

    def test_no_promise_returns_nothing(self):
        _, fake = _capture({"promise": "", "kind": "", "title": "", "resource": ""})
        with mock.patch.object(llm, "_chat", fake):
            self.assertEqual(llm.write_resource("t", NARRATION), {})

    def test_an_invented_promise_is_thrown_away(self):
        """The regression guard: the script never said it, so no resource is produced."""
        _, fake = _capture({"promise": "grab the free template linked below", "kind": "template",
                            "title": "T", "resource": "# T\n\nbody"})
        with mock.patch.object(llm, "_chat", fake):
            self.assertEqual(llm.write_resource("t", "A plain script with no offer at all."), {})

    def test_a_real_promise_survives_repunctuation(self):
        _, fake = _capture({"promise": "Check the link in the description.", "kind": " Prompt Structure ",
                            "title": "The Inbox Prompt", "resource": "# The Inbox Prompt\n\nbody"})
        with mock.patch.object(llm, "_chat", fake):
            out = llm.write_resource("t", NARRATION)
        self.assertEqual(out["kind"], "prompt structure")       # normalised
        self.assertEqual(out["title"], "The Inbox Prompt")
        self.assertIn("body", out["markdown"])

    def test_a_short_fragment_does_not_count(self):
        _, fake = _capture({"promise": "the video", "kind": "x", "title": "x", "resource": "# x\n\nb"})
        with mock.patch.object(llm, "_chat", fake):
            self.assertEqual(llm.write_resource("t", NARRATION), {})

    def test_an_empty_body_counts_as_no_promise(self):
        _, fake = _capture({"promise": "check the link in the description", "kind": "k",
                            "title": "T", "resource": "   "})
        with mock.patch.object(llm, "_chat", fake):
            self.assertEqual(llm.write_resource("t", NARRATION), {})

    def test_the_prompt_asks_for_the_thing_that_was_named(self):
        calls, fake = _capture({"promise": "", "kind": "", "title": "", "resource": ""})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_resource("t", NARRATION, source_text="README body here", max_chars=1500)
        user = calls[0][1]["content"]
        self.assertIn("COPIED WORD FOR WORD", user)
        self.assertIn("If you are not certain", user)
        self.assertIn("at most 1500 characters", user)
        self.assertIn("250 words", user)                 # 1500 // 6
        self.assertIn("README body here", user)          # the source reaches the writer

    def test_the_script_prompts_are_left_alone(self):
        """Decision: the writer is never nudged to promise anything."""
        calls, fake = _capture({"title": "t", "narration": "n", "style": "s", "issues": []})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", True):
            llm.write_script_from_prompt("a topic", "topic")
        blob = "\n".join(m["content"] for c in calls for m in c)
        for banned in ("in the description", "link below", "downloadable", "free template"):
            self.assertNotIn(banned, blob)


class Markdown(unittest.TestCase):
    def test_fenced_content_survives_byte_for_byte(self):
        out = resource.to_text("intro\n\n```\nRole: x\n# a comment\n```\n")
        self.assertIn("Role: x", out)
        self.assertIn("# a comment", out)

    def test_headings_lose_their_hash(self):
        """So YouTube cannot turn '## Steps' into the hashtag #Steps."""
        out = resource.to_text("# Title\n\n## Steps\n\n### Detail\n")
        self.assertIn("TITLE", out)
        self.assertIn("STEPS", out)
        self.assertNotIn("#", out)

    def test_bullets_and_checkboxes_normalise(self):
        out = resource.to_text("* one\n- [ ] two\n- [x] three\n[link](http://x)\n")
        self.assertIn("- one", out)
        self.assertIn("[ ] two", out)
        self.assertIn("[x] three", out)
        self.assertIn("link", out)
        self.assertNotIn("http://x", out)

    def test_description_safety(self):
        """'<' and '>' are rejected by the upload API; a '#word' would fight the real hashtags."""
        out = resource.for_description("Tone: <professional> and #keyword plus # spaced")
        self.assertNotIn("<", out)
        self.assertNotIn(">", out)
        self.assertIn("[professional]", out)
        self.assertIn("keyword", out)
        self.assertNotIn("#keyword", out)
        self.assertIn("# spaced", out)      # a lone '#' is not a hashtag


class Description(unittest.TestCase):
    def _plan(self, desc="Body text here.\n\n#ai #tools #prompts"):
        return {"seo_description": desc, "resource": dict(RES)}

    def test_appended_once_however_often_it_runs(self):
        plan = self._plan()
        self.assertTrue(resource.fold_into_description(plan))
        for _ in range(4):
            self.assertFalse(resource.fold_into_description(plan))
        self.assertEqual(plan["seo_description"].count(resource.DIVIDER), 2)
        self.assertEqual(plan["seo_description_base"], "Body text here.\n\n#ai #tools #prompts")

    def test_the_channel_hashtags_stay_first(self):
        """YouTube reads the FIRST three hashtags, so the block goes after them."""
        plan = self._plan()
        resource.fold_into_description(plan)
        d = plan["seo_description"]
        tags = [w for w in d.split() if w.startswith("#") and len(w) > 1]
        self.assertEqual(tags[:3], ["#ai", "#tools", "#prompts"])
        self.assertLess(d.index("#prompts"), d.index(resource.DIVIDER))

    def test_no_angle_brackets_reach_the_description(self):
        plan = self._plan()
        resource.fold_into_description(plan)
        self.assertNotIn("<", plan["seo_description"])
        self.assertNotIn(">", plan["seo_description"])

    def test_left_out_when_it_does_not_fit(self):
        plan = {"seo_description": "d" * 4700,
                "resource": {**RES, "markdown": "x " * 2000}}
        self.assertFalse(resource.fold_into_description(plan))
        self.assertEqual(len(plan["seo_description"]), 4700)     # untouched
        self.assertFalse(plan["resource"]["in_description"])
        self.assertGreater(plan["resource"]["description_over_by"], 0)
        self.assertNotIn("seo_description_base", plan)           # so a retry is still possible

    def test_it_fits_inside_the_upload_limit(self):
        plan = self._plan()
        resource.fold_into_description(plan)
        self.assertLessEqual(len(plan["seo_description"]), config.YT_DESCRIPTION_LIMIT)

    def test_nothing_happens_without_a_description(self):
        plan = {"seo_description": "", "resource": dict(RES)}
        self.assertFalse(resource.fold_into_description(plan))
        self.assertNotIn("seo_description_base", plan)

    def test_budget_shrinks_with_the_description(self):
        self.assertEqual(resource.markdown_budget("x" * 1000),
                         config.YT_DESCRIPTION_LIMIT - 1000 - resource._CHROME)
        self.assertEqual(resource.markdown_budget("x" * 4800), resource.RESOURCE_MIN_CHARS)


class Files(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))

    def test_attach_writes_the_files_and_the_record(self):
        plan = {"seo_description": "Body.\n\n#ai", "title": "The video"}
        record, logs = resource.attach(self.dir, plan, dict(RES), channel="Daily Tech Talks")
        self.assertTrue(record["offered"])
        self.assertIn("resource.md", record["files"])
        self.assertIn("resource.txt", record["files"])
        self.assertTrue((self.dir / "resource.md").exists())
        self.assertTrue((self.dir / "resource.txt").exists())
        self.assertIn(resource.DIVIDER, plan["seo_description"])
        self.assertTrue(any("resource:" in line for line in logs))

    def test_nothing_promised_writes_nothing_and_clears_old_files(self):
        (self.dir / "resource.md").write_text("stale", encoding="utf-8")
        plan = {"seo_description": "Body."}
        record, logs = resource.attach(self.dir, plan, {})
        self.assertEqual(record, {"offered": False, "checked": True})
        self.assertFalse((self.dir / "resource.md").exists())   # the writer owns the files
        self.assertNotIn(resource.DIVIDER, plan["seo_description"])
        self.assertIn("promises the viewer nothing", logs[0])

    def test_the_pdf_is_optional(self):
        plan = {"seo_description": "Body."}
        with mock.patch.object(resource, "_pdf_font", lambda bold: ""):
            record, logs = resource.attach(self.dir, plan, dict(RES))
        self.assertNotIn("resource.pdf", record["files"])
        self.assertFalse((self.dir / "resource.pdf").exists())
        self.assertTrue(record["pdf_error"])
        self.assertTrue((self.dir / "resource.txt").exists())   # still delivered
        self.assertTrue(any("resource.pdf skipped" in line for line in logs))

    @unittest.skipUnless(resource.pdf_available(), "fpdf2 or a usable TTF is missing")
    def test_the_pdf_renders_with_the_channel_branding(self):
        md = (RES["markdown"] + "\nA curly apostrophe ’ and an em-dash — and an emoji \U0001F600.\n"
              + "b" * 400 + "\n")
        warn = resource.to_pdf(md, self.dir / "r.pdf", title="The Inbox Prompt",
                               channel="Daily Tech Talks", video_title="Stop rewriting emails")
        self.assertEqual(warn, "")
        data = (self.dir / "r.pdf").read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertGreater(len(data), 1500)

    def test_a_long_unbroken_token_is_wrapped(self):
        self.assertIn("\n", resource._hard_wrap("a" * 300, 110))


GUIDE_MD = ("# Run PersonaPlex locally\n\n"
            "PersonaPlex is a speech-to-speech persona model.\n\n"
            "## Requirements\n\n- An NVIDIA GPU with 24 GB of VRAM\n- The libopus system package\n\n"
            "## Install\n\n```\ngit clone https://github.com/NVIDIA/personaplex\n"
            "pip install -r requirements.txt\n```\n\n"
            "## First run\n\n```\npython -m personaplex.server --port 8998 --cpu-offload\n```\n")
GUIDE = {"kind": "setup guide", "title": "Run PersonaPlex Locally", "markdown": GUIDE_MD}
README = ("GitHub repository NVIDIA/personaplex. 4200 stars. Apache-2.0.\n\n"
          "README / MODEL CARD\n\nInstall with pip install -r requirements.txt and start the "
          "server on port 8998. Use --cpu-offload when the model does not fit the card.")


class Guide(unittest.TestCase):
    """A repo video always ships a guide - and it is written from the README, never from memory."""

    def test_a_guide_needs_no_promise(self):
        _, fake = _capture({"title": "Run PersonaPlex Locally", "guide": GUIDE_MD})
        with mock.patch.object(llm, "_chat", fake):
            out = llm.write_repo_guide("t", "A script that offers the viewer nothing.", README)
        self.assertEqual(out["kind"], "setup guide")
        self.assertEqual(out["title"], "Run PersonaPlex Locally")
        self.assertIn("--cpu-offload", out["markdown"])

    def test_no_readme_no_guide(self):
        """Without the source the model would invent the install commands, so it is never asked."""
        calls, fake = _capture({"title": "T", "guide": GUIDE_MD})
        with mock.patch.object(llm, "_chat", fake):
            self.assertEqual(llm.write_repo_guide("t", "narration", "   "), {})
        self.assertEqual(calls, [])

    def test_an_empty_answer_is_not_a_guide(self):
        for reply in ({"title": "T", "guide": ""}, {"title": "T", "guide": "# T\n\ntoo short"}):
            _, fake = _capture(reply)
            with mock.patch.object(llm, "_chat", fake):
                self.assertEqual(llm.write_repo_guide("t", "n", README), {})

    def test_the_readme_and_the_page_reach_the_writer(self):
        calls, fake = _capture({"title": "T", "guide": GUIDE_MD})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_repo_guide("t", "the narration", README,
                                 source_url="https://github.com/NVIDIA/personaplex")
        blob = "\n".join(m["content"] for c in calls for m in c)
        self.assertIn("--cpu-offload", blob)                 # the README itself
        self.assertIn("the narration", blob)                 # the script, for wording
        self.assertIn("https://github.com/NVIDIA/personaplex", blob)
        self.assertIn("COPIED FROM THE SOURCE", blob)        # the never-invent rule

    def test_the_guide_is_never_sent_to_the_description(self):
        """It is a download. Telling the writer otherwise would put a promise in the wrong place."""
        calls, fake = _capture({"title": "T", "guide": GUIDE_MD})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_repo_guide("t", "n", README)
        blob = "\n".join(m["content"] for c in calls for m in c)
        for banned in ("in the description", "link below", "linked below", "free to download"):
            self.assertNotIn(banned, blob)

    def test_it_is_capped(self):
        _, fake = _capture({"title": "T", "guide": "# T\n\n" + "word " * 4000})
        with mock.patch.object(llm, "_chat", fake):
            out = llm.write_repo_guide("t", "n", README, max_chars=900)
        self.assertLessEqual(len(out["markdown"]), 900)


class GuideFiles(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))

    def _attach(self, desc="Body text here.\n\n#ai #tools"):
        plan = {"seo_description": desc, "title": "The video", "seo_title": "Run PersonaPlex"}
        record, logs = resource.attach(self.dir, plan, dict(GUIDE), channel="Daily Tech Talks",
                                       origin=resource.GUIDE,
                                       source_url="https://github.com/NVIDIA/personaplex")
        return plan, record, logs

    def test_the_files_are_written_and_the_origin_recorded(self):
        _, record, logs = self._attach()
        self.assertTrue(record["offered"])
        self.assertEqual(record["origin"], resource.GUIDE)
        self.assertNotIn("promise", record)                  # nothing was promised
        self.assertEqual(record["source_url"], "https://github.com/NVIDIA/personaplex")
        for name in ("resource.md", "resource.txt"):
            self.assertIn(name, record["files"])
            self.assertTrue((self.dir / name).exists())
        self.assertTrue(any("written from the README" in line for line in logs))

    def test_the_description_is_left_alone(self):
        """The regression that matters: nobody was sent to the description to find a guide."""
        plan, record, logs = self._attach()
        self.assertEqual(plan["seo_description"], "Body text here.\n\n#ai #tools")
        self.assertNotIn(resource.DIVIDER, plan["seo_description"])
        self.assertNotIn("seo_description_base", plan)       # nothing to undo on a re-run
        self.assertFalse(record["in_description"])
        self.assertFalse(resource.fold_into_description(plan))
        self.assertEqual(plan["seo_description"], "Body text here.\n\n#ai #tools")
        self.assertTrue(any("download only" in line for line in logs))

    def test_a_promised_resource_still_reaches_the_description(self):
        """The contrast: the guide rule must not have switched the promised path off."""
        plan = {"seo_description": "Body.\n\n#ai", "title": "The video"}
        resource.attach(self.dir, plan, dict(RES), channel="Daily Tech Talks")
        self.assertIn(resource.DIVIDER, plan["seo_description"])
        self.assertEqual(plan["resource"]["origin"], resource.PROMISE)
        self.assertTrue(plan["resource"]["in_description"])

    def test_the_channel_signs_the_text_versions_only(self):
        _, record, _ = self._attach()
        for name in ("resource.md", "resource.txt"):
            head = (self.dir / name).read_text(encoding="utf-8")
            self.assertTrue(head.startswith("Daily Tech Talks"), name)
            self.assertIn("From the video: Run PersonaPlex", head)
            self.assertIn("Project page: https://github.com/NVIDIA/personaplex", head)
            self.assertIn("--cpu-offload", head)             # the guide itself is still all there
        # ...but the stored markdown stays the document, so the description block never repeats it
        self.assertFalse(record["markdown"].startswith("Daily Tech Talks"))
        self.assertEqual(record["markdown"], GUIDE_MD)

    def test_the_masthead_is_dropped_when_there_is_nothing_to_say(self):
        self.assertEqual(resource.masthead("", "", ""), "")

    @unittest.skipUnless(resource.pdf_available(), "fpdf2 or a usable TTF is missing")
    def test_the_pdf_carries_the_channel_and_the_project_page(self):
        warn = resource.to_pdf(GUIDE_MD, self.dir / "g.pdf", title="Run PersonaPlex Locally",
                               channel="Daily Tech Talks", video_title="Run PersonaPlex",
                               source_url="https://github.com/NVIDIA/personaplex")
        self.assertEqual(warn, "")
        data = (self.dir / "g.pdf").read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertGreater(len(data), 1500)

    @unittest.skipUnless(resource.pdf_available(), "fpdf2 or a usable TTF is missing")
    def test_a_hard_wrapped_paragraph_is_not_broken_mid_sentence(self):
        """Markdown wraps a paragraph across source lines; the PDF must reflow it into one."""
        md = "# T\n\nThe first half of a sentence\nand the second half of it.\n"
        rendered = []
        with mock.patch.object(resource, "_pdf_line",
                               lambda pdf, raw: rendered.append(raw)):
            resource.to_pdf(md, self.dir / "w.pdf", title="T", channel="C")
        self.assertIn("The first half of a sentence and the second half of it.", rendered)


if __name__ == "__main__":
    unittest.main()
