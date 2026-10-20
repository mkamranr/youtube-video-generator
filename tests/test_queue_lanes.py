"""Two lanes over one GPU: run with  python -m unittest discover -s tests -t .  from the repo root.

The point of the feature is in LaneScheduling.test_a_held_gpu_does_not_stop_the_other_lane: a job
rendering clips must not stop the next job writing its script, and the only thing that waits is the
work that needs the hardware - ComfyUI renders and Fish TTS.

Pure unit tests: no ComfyUI, no Fish, no threads beyond the one each test starts itself."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-lanes-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from pipeline import Job, JobManager, JobWaitingGpu, JobWaitingWindow  # noqa: E402
from services import gpu  # noqa: E402


def _job(jid: str, **kw) -> Job:
    job = Job(id=jid, url="", **kw)
    job.dir.mkdir(parents=True, exist_ok=True)
    return job


class Lease(unittest.TestCase):
    def setUp(self):
        gpu.reset()
        self.addCleanup(gpu.reset)

    def test_one_holder_at_a_time_and_the_holder_can_retake_it(self):
        self.assertTrue(gpu.acquire("a"))
        self.assertTrue(gpu.acquire("a"))       # re-entrant per job: every gate of a run re-takes it
        self.assertFalse(gpu.acquire("b"))
        self.assertEqual(gpu.holder(), "a")
        self.assertTrue(gpu.busy())
        self.assertTrue(gpu.held_by("a"))
        self.assertFalse(gpu.held_by("b"))

    def test_only_the_holder_can_release_it(self):
        gpu.acquire("a")
        gpu.release("b")                        # a parked job on its way out
        self.assertEqual(gpu.holder(), "a")     # must not hand away someone else's GPU
        gpu.release("a")
        self.assertIsNone(gpu.holder())
        self.assertTrue(gpu.acquire("b"))

    def test_an_empty_id_never_holds_it(self):
        self.assertFalse(gpu.acquire(""))
        self.assertIsNone(gpu.holder())

    def test_wait_until_free_blocks_until_released(self):
        gpu.acquire("a")
        got, waited = [], []
        t = threading.Thread(target=lambda: got.append(
            gpu.wait_until_free("b", on_wait=waited.append, poll=0.01)), daemon=True)
        t.start()
        t.join(0.2)
        self.assertTrue(t.is_alive())           # still waiting: "a" has it
        self.assertEqual(gpu.holder(), "a")
        gpu.release("a")
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(got, [True])
        self.assertEqual(gpu.holder(), "b")
        self.assertEqual(waited, ["a"])         # told once who it is behind

    def test_wait_until_free_gives_up_when_asked(self):
        gpu.acquire("a")
        self.assertFalse(gpu.wait_until_free("b", should_stop=lambda: True, poll=0.01))
        self.assertEqual(gpu.holder(), "a")


class Gate(unittest.TestCase):
    """_gpu_gate is the checkpoint in front of every generation step."""

    def setUp(self):
        gpu.reset()
        self.addCleanup(gpu.reset)
        self.job = _job("20260101-000001-aaaaaa")

    def test_the_first_gate_takes_the_lease_and_later_ones_are_free(self):
        pipeline._gpu_gate(self.job)
        self.assertEqual(gpu.holder(), self.job.id)
        pipeline._gpu_gate(self.job)            # every clip re-gates; it must not park itself
        self.assertEqual(gpu.holder(), self.job.id)

    def test_a_gate_against_a_held_lease_parks(self):
        gpu.acquire("someone-else")
        with self.assertRaises(JobWaitingGpu):
            pipeline._gpu_gate(self.job)
        self.assertEqual(gpu.holder(), "someone-else")

    def test_tts_queues_for_the_gpu_but_ignores_the_schedule_window(self):
        """The window says when ComfyUI may run; narration was never gated by it and still is not.
        The lease is the separate thing, and Fish shares the GPU, so it does queue."""
        with mock.patch.object(pipeline.schedule, "is_open", lambda: False):
            pipeline._gpu_gate(self.job, window=False)          # speaks even outside the window
            self.assertEqual(gpu.holder(), self.job.id)
            with self.assertRaises(JobWaitingWindow):           # a render does not
                pipeline._gpu_gate(_job("20260101-000002-bbbbbb"))
        gpu.reset()
        gpu.acquire("someone-else")
        with self.assertRaises(JobWaitingGpu):
            pipeline._gpu_gate(self.job, window=False)


class LeaseIsAlwaysReturned(unittest.TestCase):
    """However a run ends, the next job gets the GPU."""

    def setUp(self):
        gpu.reset()
        self.addCleanup(gpu.reset)
        self.job = _job("20260101-000003-cccccc")

    def _run_with(self, side_effect):
        with mock.patch.object(pipeline, "_run", side_effect=side_effect):
            pipeline.run_job(self.job)

    def test_released_after_a_normal_run(self):
        self._run_with(lambda job: pipeline._gpu_gate(job))
        self.assertIsNone(gpu.holder())
        self.assertFalse(self.job.gpu_wait)

    def test_released_after_a_failure(self):
        self._run_with(lambda job: (pipeline._gpu_gate(job), 1 / 0))
        self.assertIsNone(gpu.holder())
        self.assertEqual(self.job.status, "failed")

    def test_released_when_the_window_shuts_mid_run(self):
        def park(job):
            pipeline._gpu_gate(job)
            raise JobWaitingWindow()
        self._run_with(park)
        self.assertIsNone(gpu.holder())
        self.assertTrue(self.job.gpu_wait)

    def test_a_parked_job_is_queued_again_and_says_why(self):
        gpu.acquire("someone-else")
        self._run_with(lambda job: pipeline._gpu_gate(job))
        self.assertTrue(self.job.gpu_wait)
        self.assertEqual(self.job.status, "queued")
        self.assertIn("another video is rendering", self.job.detail)
        self.assertEqual(gpu.holder(), "someone-else")   # it never had it, so it took nothing away


class LaneScheduling(unittest.TestCase):
    """Which job a free lane picks up."""

    def setUp(self):
        gpu.reset()
        self.addCleanup(gpu.reset)
        self.m = object.__new__(JobManager)
        self.m.lock = threading.RLock()
        self.a, self.b = _job("20260101-000004-dddddd"), _job("20260101-000005-eeeeee")
        self.m.jobs = {self.a.id: self.a, self.b.id: self.b}
        self.m.queue = [self.a.id, self.b.id]
        self.m.running = set()

    def test_a_held_gpu_does_not_stop_the_other_lane(self):
        """Job A is rendering and job B is parked behind it - but a third job with LLM work left
        is picked up at once. This is the whole feature."""
        self.a.gpu_wait = True
        gpu.acquire(self.a.id)
        self.m.running.add(self.a.id)
        self.assertEqual(self.m._next_job(), self.b.id)   # B still has its script to write
        self.assertEqual(self.m.queue, [self.a.id])       # A stays parked where it was

    def test_the_parked_job_is_picked_up_once_the_gpu_frees(self):
        self.a.gpu_wait = True
        gpu.acquire("someone-else")
        self.assertEqual(self.m._next_job(), self.b.id)
        gpu.release("someone-else")
        self.assertEqual(self.m._next_job(), self.a.id)

    def test_a_job_parked_for_the_window_still_waits_for_the_window(self):
        self.a.gpu_wait = True
        with mock.patch.object(pipeline.schedule, "is_open", lambda: False):
            self.assertEqual(self.m._next_job(), self.b.id)
            self.assertIsNone(self.m._next_job())         # B taken, A still parked: nothing to do

    def test_a_job_running_in_the_other_lane_is_never_picked_up_twice(self):
        """A priority regeneration re-queues the job it paused while that job is still finishing
        its current step."""
        self.m.running.add(self.a.id)
        self.assertEqual(self.m._next_job(), self.b.id)
        self.assertEqual(self.m.queue, [self.a.id])
        self.m.running.discard(self.a.id)
        self.assertEqual(self.m._next_job(), self.a.id)

    def test_current_names_the_job_that_owns_the_gpu(self):
        self.m.running.update({self.a.id, self.b.id})
        gpu.acquire(self.b.id)
        self.assertEqual(self.m.current, self.b.id)       # the one others wait behind
        gpu.release(self.b.id)
        self.assertIn(self.m.current, {self.a.id, self.b.id})
        self.m.running.clear()
        self.assertIsNone(self.m.current)


if __name__ == "__main__":
    unittest.main()
