"""Scene prompts are the most expensive text the pipeline writes: one LLM call per clip, two
with the review on, over twenty-odd clips. A vLLM whose engine dies in the middle of that used to
cost every scene already written, and a review that failed killed the whole job.

Run with  python -m unittest discover -s tests -t .  from the repo root. The chat call is mocked;
nothing here needs vLLM or a GPU.
"""
import unittest
from unittest import mock

import config
from services import llm


def _chat_returning(scenes):
    """A fake _chat that hands out the given scenes in order and records the calls."""
    calls, it = [], iter(scenes)
    def fake(messages, max_tokens=8000):
        calls.append(messages[-1]["content"])
        return {"scene": next(it)}
    return calls, fake


class ResumeFromWhatIsWritten(unittest.TestCase):
    def test_only_the_missing_scenes_are_written(self):
        calls, fake = _chat_returning(["three", "four"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            scenes = llm.write_scenes("a b c d", "style", 4, done=["one", "two"])
        self.assertEqual(scenes, ["one", "two", "three", "four"])
        self.assertEqual(len(calls), 2)          # not four: two were already paid for

    def test_the_first_new_scene_continues_from_the_last_kept_one(self):
        # continuity is the whole reason scenes are written one per call, so a resumed run must
        # still hand the writer the scene before it
        calls, fake = _chat_returning(["three"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("a b c", "style", 3, done=["one", "two"])
        self.assertIn('The previous clip\'s scene was:\n"two"', calls[0])

    def test_nothing_done_is_an_ordinary_run(self):
        calls, fake = _chat_returning(["one", "two"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            self.assertEqual(llm.write_scenes("a b", "style", 2), ["one", "two"])
        self.assertEqual(len(calls), 2)

    def test_more_done_than_clips_is_trimmed_rather_than_trusted(self):
        # the clip count can shrink between runs; a stale extra scene must not survive
        _calls, fake = _chat_returning([])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            self.assertEqual(llm.write_scenes("a", "style", 1, done=["one", "two"]), ["one"])

    def test_each_scene_is_reported_as_it_lands(self):
        seen = []
        _calls, fake = _chat_returning(["one", "two", "three"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("a b c", "style", 3, on_scene=lambda s: seen.append(list(s)))
        # the caller gets the growing list, so it can persist progress after every call
        self.assertEqual(seen, [["one"], ["one", "two"], ["one", "two", "three"]])

    def test_the_report_carries_a_copy_the_caller_can_keep(self):
        seen = []
        _calls, fake = _chat_returning(["one", "two"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("a b", "style", 2, on_scene=seen.append)
        self.assertEqual(seen[0], ["one"])        # not mutated into the final list afterwards


class AReviewThatFailsCostsOnlyTheReview(unittest.TestCase):
    """SCRIPT_REVIEW is a polish pass over a scene that already exists. Losing the LLM during it
    must not lose the scene, and certainly not the twenty scenes before it."""

    def test_the_unreviewed_scene_is_kept(self):
        _calls, fake = _chat_returning(["one", "two"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", True), \
                mock.patch.object(llm, "review_scene",
                                  side_effect=RuntimeError("EngineCore encountered an issue")):
            scenes = llm.write_scenes("a b", "style", 2)
        self.assertEqual(scenes, ["one", "two"])

    def test_the_caller_is_told_which_scene_went_unreviewed(self):
        told = []
        _calls, fake = _chat_returning(["one"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", True), \
                mock.patch.object(llm, "review_scene", side_effect=RuntimeError("500 boom")):
            llm.write_scenes("a", "style", 1, on_review_failed=lambda i, e: told.append((i, e)))
        self.assertEqual(len(told), 1)
        self.assertEqual(told[0][0], 1)
        self.assertIn("500 boom", told[0][1])

    def test_a_working_review_still_replaces_the_scene_and_reports_its_issues(self):
        issues = []
        _calls, fake = _chat_returning(["one"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", True), \
                mock.patch.object(llm, "review_scene", return_value=("fixed", ["said 'Step 1/50'"])):
            scenes = llm.write_scenes("a", "style", 1,
                                      on_review_issue=lambda i, iss: issues.append((i, iss)))
        self.assertEqual(scenes, ["fixed"])
        self.assertEqual(issues, [(1, ["said 'Step 1/50'"])])

    def test_a_failed_review_does_not_look_like_a_fixed_one(self):
        issues, failed = [], []
        _calls, fake = _chat_returning(["one"])
        with mock.patch.object(llm, "_chat", fake), \
                mock.patch.object(config, "SCRIPT_REVIEW", True), \
                mock.patch.object(llm, "review_scene", side_effect=RuntimeError("boom")):
            llm.write_scenes("a", "style", 1,
                             on_review_issue=lambda i, iss: issues.append((i, iss)),
                             on_review_failed=lambda i, e: failed.append(i))
        self.assertEqual(issues, [])      # "review fixed scene 1: ..." would be a lie
        self.assertEqual(failed, [1])

    def test_a_scene_the_writer_could_not_produce_is_still_fatal(self):
        # an empty scene is not a polish problem: the clip has nothing to render from
        with mock.patch.object(llm, "_chat", lambda *a, **kw: {"scene": "  "}), \
                mock.patch.object(config, "SCRIPT_REVIEW", False):
            with self.assertRaises(RuntimeError):
                llm.write_scenes("a", "style", 1)


if __name__ == "__main__":
    unittest.main()
