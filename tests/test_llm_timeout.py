"""A wedged LLM must fail the job, and a cancel must land while a script is being written.

The remote vLLM can accept a request, answer /v1/models in milliseconds and then never decode a
token. The OpenAI client's own defaults are a 600-second read with two silent retries, so one such
call used to hold a job on "writing script with the LLM" for the better part of an hour, and the
Cancel button only ever reached "cancelling..." because the draft thread was blocked in that
request with no checkpoint to notice the flag.

Run with  python -m unittest discover -s tests -t .  from the repo root.
"""
import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-llmtimeout-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import llm  # noqa: E402

CHANNEL = {"id": "ch01", "name": "Bright Minds Town", "purpose": "", "rules": "", "owner": "u1",
           "audience": "kids", "video_type": "animated", "cast": [], "cast_opening": True,
           "defaults": {"art_style": "", "audio_mode": "narration+music", "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": config.DEFAULT_VIDEO_LENGTH, "storyboard": False,
                        "song_voice": "child"}}

REQUEST = httpx.Request("POST", "http://box:8000/v1/chat/completions")


def _answer(text: str):
    """The shape of a chat completion `_chat` reads: choices[0].message.content."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text),
                                       finish_reason="stop")])


class BoundedCall(unittest.TestCase):
    """One vLLM request can only ever take LLM_TIMEOUT seconds."""

    def test_the_client_carries_the_configured_timeout_and_no_retries(self):
        with mock.patch.object(config, "LLM_TIMEOUT", 123):
            c = llm._client()
        self.assertEqual(c.timeout.read, 123.0)
        self.assertEqual(c.timeout.connect, 10.0)   # an unreachable box fails in seconds
        self.assertEqual(c.max_retries, 0)          # re-sending a timed-out request only waits again

    def test_a_wedged_server_fails_with_a_message_naming_it(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APITimeoutError(request=REQUEST)
        with mock.patch.object(llm, "_client", return_value=client), \
                mock.patch.object(config, "VLLM_URL", "http://box:8000/v1"), \
                mock.patch.object(config, "VLLM_MODEL", "agents-a1"), \
                mock.patch.object(config, "LLM_TIMEOUT", 300):
            with self.assertRaises(RuntimeError) as cm:
                llm._chat([{"role": "user", "content": "hi"}])
        msg = str(cm.exception)
        self.assertIn("http://box:8000/v1", msg)
        self.assertIn("300s", msg)
        self.assertIn("agents-a1", msg)
        self.assertIn("resume", msg)
        # the timeout is paid once: the JSON-mode fallback must not re-send a request that timed out
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_an_unreachable_server_says_which_url(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=REQUEST)
        with mock.patch.object(llm, "_client", return_value=client), \
                mock.patch.object(config, "VLLM_URL", "http://box:8000/v1"):
            with self.assertRaises(RuntimeError) as cm:
                llm._chat([{"role": "user", "content": "hi"}])
        self.assertIn("could not reach", str(cm.exception))
        self.assertIn("http://box:8000/v1", str(cm.exception))
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_a_server_without_json_mode_is_still_retried_plainly(self):
        refused = APIStatusError("unsupported", response=httpx.Response(400, request=REQUEST),
                                 body=None)
        client = mock.Mock()
        client.chat.completions.create.side_effect = [refused, _answer('{"title": "The Kite"}')]
        with mock.patch.object(llm, "_client", return_value=client):
            self.assertEqual(llm._chat([{"role": "user", "content": "hi"}]), {"title": "The Kite"})
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertNotIn("response_format", client.chat.completions.create.call_args.kwargs)

    def test_a_timeout_on_the_plain_retry_is_reported_the_same_way(self):
        refused = APIStatusError("unsupported", response=httpx.Response(400, request=REQUEST),
                                 body=None)
        client = mock.Mock()
        client.chat.completions.create.side_effect = [refused, APITimeoutError(request=REQUEST)]
        with mock.patch.object(llm, "_client", return_value=client):
            with self.assertRaises(RuntimeError) as cm:
                llm._chat([{"role": "user", "content": "hi"}])
        self.assertIn("did not answer within", str(cm.exception))


class CancelWhileDrafting(unittest.TestCase):
    """Drafting runs in its own thread, blocked in a request nothing can interrupt, and a job
    awaiting approval is not running at all: both must be cancelled outright rather than left
    waiting for a checkpoint that never comes."""

    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        for target in ("_start_draft",):   # no LLM thread is ever launched
            p = mock.patch.object(pipeline.JobManager, target)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(pipeline.comfy, "ComfyClient")   # cancel pokes the GPU box otherwise
        p.start()
        self.addCleanup(p.stop)
        self.mgr = pipeline.JobManager()

    def _drafting(self):
        job = self.mgr.submit_prompt("Finn walks to the meadow and finds a lost kite.",
                                     "description", channel=CHANNEL, owner="u1",
                                     title="The Kite in the Meadow")
        self.assertEqual(job.status, "drafting")
        return job

    def test_cancel_lands_immediately_while_drafting(self):
        job = self._drafting()
        self.mgr.cancel(job.id)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "cancelled by user")
        self.assertNotIn("cancelling", job.detail)
        self.assertTrue(job.cancel_requested)   # the draft thread reads this and drops its result

    def test_cancel_lands_immediately_while_awaiting_approval(self):
        job = self._drafting()
        job.set("awaiting_approval", "script ready - review and approve")
        self.mgr.cancel(job.id)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "cancelled by user")
        self.assertNotIn("cancelling", job.detail)

    def test_a_draft_that_lands_after_the_cancel_is_discarded(self):
        job = self._drafting()
        answer = {"title": "The Kite in the Meadow", "narration": " ".join(["kite"] * 400),
                  "style": "flat pastel storybook", "cast_chosen": []}

        def cancelled_mid_call(*a, **kw):
            self.mgr.cancel(job.id)     # the user gives up while the writer is still working
            return answer

        with mock.patch.object(pipeline.llm, "write_script_from_prompt",
                               side_effect=cancelled_mid_call):
            self.mgr._draft_script(job)

        self.assertEqual(job.status, "failed")          # not revived to awaiting_approval
        self.assertEqual(job.error, "cancelled by user")
        self.assertFalse((job.dir / "plan.json").exists())

    def test_cancelling_a_rewrite_keeps_the_script_being_reviewed(self):
        job = self._drafting()
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / "plan.json").write_text(json.dumps(
            {"title": "The Kite in the Meadow", "narration": "Finn finds a lost kite.",
             "style": "flat pastel storybook"}), encoding="utf-8")
        job.set("awaiting_approval", "script ready - review and approve")

        self.mgr.redraft_script(job.id, feedback="make it gentler")
        self.assertEqual(job.status, "drafting")
        self.mgr.cancel(job.id)

        # the rewrite is abandoned, not the script: the user is back where they were
        self.assertEqual(job.status, "awaiting_approval")
        self.assertIn("earlier script is kept", job.detail)
        plan = json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["narration"], "Finn finds a lost kite.")
        self.assertTrue(job.cancel_requested)   # the abandoned rewrite still has to drop its result

    def test_approving_after_a_cancelled_rewrite_clears_the_flag(self):
        job = self._drafting()
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / "plan.json").write_text(json.dumps(
            {"title": "The Kite in the Meadow", "narration": "Finn finds a lost kite."}),
            encoding="utf-8")
        job.set("awaiting_approval", "script ready - review and approve")
        self.mgr.redraft_script(job.id)
        self.mgr.cancel(job.id)

        self.mgr.approve_script(job.id)
        self.assertEqual(job.status, "queued")
        # otherwise the worker would raise JobCancelled at its first checkpoint
        self.assertFalse(job.cancel_requested)

    def test_a_first_draft_has_nothing_to_keep(self):
        job = self._drafting()
        self.assertFalse((job.dir / "plan.json").exists())
        self.mgr.cancel(job.id)
        self.assertEqual(job.status, "failed")

    def test_a_failed_draft_keeps_its_own_reason(self):
        job = self._drafting()
        with mock.patch.object(pipeline.llm, "write_script_from_prompt",
                               side_effect=RuntimeError("the LLM did not answer within 300s")):
            self.mgr._draft_script(job)
        self.assertEqual(job.status, "failed")
        self.assertIn("did not answer within 300s", job.error)
        self.assertFalse(job.cancel_requested)   # nothing was cancelled, so Resume is offered


if __name__ == "__main__":
    unittest.main()
