"""YouTube videos: the script is drafted from the fetched transcript and approved before rendering,
and the user can edit the transcript and redraft from it.

Run with  python -m unittest discover -s tests -t .  from the repo root.
Pure unit tests: the transcript fetch and the LLM call are mocked.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-ytapprove-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import llm  # noqa: E402

CHANNEL = {"id": "ch01", "name": "Tech", "purpose": "", "rules": "", "owner": "u1",
           "audience": "adults", "video_type": "tech",
           "defaults": {"art_style": "", "audio_mode": "narration+music", "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": "45-60 seconds", "storyboard": False, "video_mode": ""}}

TRANSCRIPT = "This is the fetched transcript of a talk about vector databases and how they index. " * 3


class YoutubeApproval(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        patcher = mock.patch.object(pipeline.JobManager, "_start_draft")
        self.start_draft = patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = pipeline.JobManager()

    def _awaiting(self):
        job = self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1")
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / "transcript.txt").write_text(TRANSCRIPT, encoding="utf-8")
        (job.dir / "plan.json").write_text(json.dumps({"title": "t", "narration": "n", "style": "s"}),
                                           encoding="utf-8")
        job.set("awaiting_approval", "script ready - review and approve")
        return job

    def test_submit_drafts_instead_of_queueing(self):
        job = self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1")
        self.assertEqual(job.status, "drafting")
        self.assertTrue(job.script_review)
        self.assertEqual(self.mgr.queue, [])
        self.start_draft.assert_called_once()

    def test_draft_writes_the_script_from_the_transcript(self):
        job = self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1")
        plan = {"title": "Vectors", "narration": "word " * 120, "style": "clean"}
        with mock.patch.object(pipeline.transcript, "get_transcript_info",
                               return_value={"text": TRANSCRIPT * 5, "seconds": 30}), \
                mock.patch.object(llm, "write_script_from_prompt", return_value=plan) as write, \
                mock.patch.object(pipeline, "_fit_narration"):
            self.mgr._draft_script(job)
        self.assertEqual(job.status, "awaiting_approval", job.error)
        self.assertIn("vector databases", write.call_args.args[0])
        self.assertEqual(write.call_args.args[1], "youtube")
        self.assertEqual(self.mgr.queue, [])

    def test_youtube_source_task_carries_the_transcript(self):
        task, _ = llm._source_task(TRANSCRIPT, "youtube", CHANNEL)
        self.assertIn("transcript of a source video", task)
        self.assertIn("vector databases", task)

    def test_redraft_saves_the_edited_transcript(self):
        job = self._awaiting()
        edited = "My own edited transcript that keeps only the indexing part of the talk, nothing else."
        self.mgr.redraft_script(job.id, "", transcript=edited)
        self.assertEqual((job.dir / "transcript.txt").read_text(encoding="utf-8"), edited)
        self.assertEqual(job.status, "drafting")
        self.assertTrue(any("transcript edited" in line for line in job.log))
        self.assertEqual(pipeline._youtube_transcript(job), edited)   # no refetch

    def test_redraft_rejects_a_too_short_transcript(self):
        job = self._awaiting()
        with self.assertRaises(ValueError):
            self.mgr.redraft_script(job.id, "", transcript="too short")
        self.assertEqual((job.dir / "transcript.txt").read_text(encoding="utf-8"), TRANSCRIPT)
        self.assertEqual(job.status, "awaiting_approval")

    def test_redraft_without_transcript_keeps_it(self):
        job = self._awaiting()
        self.mgr.redraft_script(job.id, "shorter please")
        self.assertEqual((job.dir / "transcript.txt").read_text(encoding="utf-8"), TRANSCRIPT)

    def test_transcript_edit_is_only_for_youtube_jobs(self):
        job = self.mgr.submit_prompt("vector databases", "topic", channel=CHANNEL, owner="u1")
        job.set("awaiting_approval")
        with self.assertRaises(ValueError):
            self.mgr.redraft_script(job.id, "", transcript=TRANSCRIPT)

    def test_approve_queues_the_job(self):
        job = self._awaiting()
        self.mgr.approve_script(job.id)
        self.assertEqual(job.status, "queued")
        self.assertEqual(self.mgr.queue, [job.id])

    def test_resume_of_an_unapproved_job_redrafts(self):
        job = self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1")
        job.set("failed", "transcript fetch failed")
        self.start_draft.reset_mock()
        self.mgr.resume(job.id)
        self.assertEqual(job.status, "drafting")
        self.start_draft.assert_called_once()

    def test_resume_of_a_legacy_job_keeps_its_plan_and_queues(self):
        job = pipeline.Job(id="legacy-yt", url="https://youtu.be/x", status="failed")
        job.save()
        self.mgr.jobs[job.id] = job
        self.mgr.resume(job.id)
        self.assertEqual(job.status, "queued")
        self.assertEqual(self.mgr.queue, [job.id])
        self.start_draft.assert_not_called()


if __name__ == "__main__":
    unittest.main()
