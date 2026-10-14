"""Compilations: which videos may be joined, what shape they are brought to, where the
chapters fall, the title the length goes into, and the stitch pipeline's wiring.

Run with  python -m unittest discover -s tests -t .  from the repo root. No server and no GPU;
services/compilation.py is pure logic, and the pipeline tests mock the ffmpeg calls.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-compilation-"))
config.OUTPUT_DIR = TMP / "output"

import pipeline  # noqa: E402  (config patched first)
from services import chapters, compilation, llm, video  # noqa: E402


def _job(jid="20260101-000000-aaaaaa", status="done", final="final.mp4", channel="ch01",
         aspect="9:16 (Portrait Widescreen)", archived=False, source="topic", title="A video"):
    return {"id": jid, "status": status, "final": final, "channel_id": channel,
            "aspect_ratio": aspect, "archived": archived, "source": source, "title": title}


class Eligible(unittest.TestCase):
    def test_a_finished_video_with_a_file_is_eligible(self):
        self.assertTrue(compilation.eligible(_job()))

    def test_unfinished_fileless_and_archived_videos_are_not(self):
        self.assertFalse(compilation.eligible(_job(status="clips")))
        self.assertFalse(compilation.eligible(_job(final="")))
        self.assertFalse(compilation.eligible(_job(archived=True)))

    def test_a_compilation_is_never_a_part_of_another(self):
        """Nesting would re-encode whole videos for nothing and its chapters would be the parts
        of a part, so the only interesting thing about the nested video is lost."""
        rec = _job(source=compilation.SOURCE)
        self.assertTrue(compilation.is_compilation(rec))
        self.assertFalse(compilation.eligible(rec))

    def test_an_object_reads_the_same_as_a_dict(self):
        class Rec:
            pass
        r = Rec()
        for k, v in _job().items():
            setattr(r, k, v)
        self.assertTrue(compilation.eligible(r))


class ValidateParts(unittest.TestCase):
    def _two(self, **kw):
        return [_job("a"), _job("b", **kw)]

    def test_the_order_given_is_the_order_kept(self):
        parts = [_job("b"), _job("a")]
        self.assertEqual([p["id"] for p in compilation.validate_parts(parts)], ["b", "a"])

    def test_too_few_and_too_many_are_refused(self):
        with self.assertRaises(ValueError):
            compilation.validate_parts([_job("a")])
        with self.assertRaises(ValueError):
            compilation.validate_parts([])
        many = [_job(f"j{i}") for i in range(compilation.MAX_PARTS + 1)]
        with self.assertRaises(ValueError):
            compilation.validate_parts(many)
        # exactly the cap is fine - MAX_PARTS is chapters.MAX_COUNT, not an arbitrary number
        ok = [_job(f"j{i}") for i in range(compilation.MAX_PARTS)]
        self.assertEqual(len(compilation.validate_parts(ok)), compilation.MAX_PARTS)

    def test_the_same_video_cannot_be_picked_twice(self):
        with self.assertRaises(ValueError):
            compilation.validate_parts([_job("a"), _job("a")])

    def test_mixed_aspect_ratios_are_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            compilation.validate_parts(self._two(aspect="16:9 (Widescreen)"))
        self.assertIn("same aspect ratio", str(cm.exception))
        self.assertIn("16:9 (Widescreen)", str(cm.exception))

    def test_two_channels_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            compilation.validate_parts(self._two(channel="ch02"))
        self.assertIn("same channel", str(cm.exception))

    def test_an_unfinished_part_is_named_in_the_error(self):
        with self.assertRaises(ValueError) as cm:
            compilation.validate_parts([_job("a"), _job("b", status="clips", title="Half done")])
        self.assertIn("Half done", str(cm.exception))


class TargetSize(unittest.TestCase):
    def test_the_largest_frame_wins_so_nothing_is_downscaled(self):
        self.assertEqual(compilation.target_size([(720, 1280), (1080, 1920)]), (1080, 1920))
        self.assertEqual(compilation.target_size([(1080, 1920), (720, 1280)]), (1080, 1920))

    def test_unreadable_sizes_are_skipped_not_fatal(self):
        self.assertEqual(compilation.target_size([None, (0, 0), ("x", "y"), (640, 360)]), (640, 360))
        with self.assertRaises(ValueError):
            compilation.target_size([None, (0, 0)])


class Marks(unittest.TestCase):
    def test_one_mark_per_part_starting_at_zero(self):
        """bounds are cumulative END times, so part i starts where part i-1 ended. The first
        chapter has to read exactly 0:00 or YouTube ignores the whole list."""
        marks = compilation.marks(["One", "Two", "Three"], [60.0, 150.0, 210.0])
        self.assertEqual([m["t"] for m in marks], [0.0, 60.0, 150.0])
        self.assertEqual([m["title"] for m in marks], ["One", "Two", "Three"])

    def test_an_untitled_part_contributes_no_chapter(self):
        self.assertEqual([m["title"] for m in compilation.marks(["One", "  ", "Three"],
                                                                [60.0, 150.0, 210.0])],
                         ["One", "Three"])

    def test_more_titles_than_bounds_stops_rather_than_inventing_times(self):
        self.assertEqual(len(compilation.marks(["a", "b", "c"], [60.0])), 2)


class ChapterSource(unittest.TestCase):
    """chapters.build gained `marks` as a source tried BEFORE the inferred ones."""

    def test_marks_beat_cards_beats_and_segments(self):
        bounds = [120.0, 240.0, 360.0]
        out = chapters.build(bounds, total=360.0,
                             marks=compilation.marks(["First video", "Second", "Third"], bounds),
                             cards=[{"start_clip": 1, "headline": "A card"}],
                             beats=[{"action": "a beat"}],
                             segments=["some narration words here"])
        self.assertEqual([c["title"] for c in out], ["First video", "Second", "Third"])
        self.assertEqual(out[0]["t"], 0.0)

    def test_without_marks_nothing_changed(self):
        bounds = [120.0, 240.0, 360.0]
        cards = [{"start_clip": i, "headline": f"Card {i}"} for i in (1, 2, 3)]
        out = chapters.build(bounds, total=360.0, cards=cards)
        self.assertEqual([c["title"] for c in out], ["Card 1", "Card 2", "Card 3"])
        # and marks=None / [] is the same as not passing it at all
        self.assertEqual(chapters.build(bounds, total=360.0, cards=cards, marks=[]), out)

    def test_marks_still_obey_youtubes_rules(self):
        """Three chapters minimum, each at least ten seconds, the first at 0:00 - a list that
        breaks one of them is ignored in full by YouTube, so build returns nothing instead."""
        two = chapters.build([120.0, 240.0], total=240.0,
                             marks=compilation.marks(["One", "Two"], [120.0, 240.0]))
        self.assertEqual(two, [])
        # a last part under ten seconds is dropped rather than making the list illegal
        out = chapters.build([120.0, 240.0, 360.0, 365.0], total=365.0,
                             marks=compilation.marks(["One", "Two", "Three", "Four"],
                                                     [120.0, 240.0, 360.0, 365.0]))
        self.assertEqual([c["title"] for c in out], ["One", "Two", "Three"])

    def test_a_hash_in_a_video_title_cannot_void_the_channels_hashtags(self):
        """YouTube ignores every hashtag in a description once there are more than 15, so a '#1'
        surviving out of a video title into a chapter line is not cosmetic."""
        out = chapters.build([120.0, 240.0, 360.0], total=360.0,
                             marks=compilation.marks(["#1 Tilly the Train", "Two", "Three"],
                                                     [120.0, 240.0, 360.0]))
        self.assertEqual(out[0]["title"], "1 Tilly the Train")


class Titles(unittest.TestCase):
    def test_the_length_is_rounded_down_never_overstated(self):
        self.assertEqual(compilation.fmt_length(1199), "19 Minutes")   # not 20
        self.assertEqual(compilation.fmt_length(60), "1 Minute")
        self.assertEqual(compilation.fmt_length(45), "45 Seconds")
        self.assertEqual(compilation.fmt_length(0), "0 Seconds")
        self.assertEqual(compilation.fmt_length(None), "0 Seconds")

    def test_the_fallback_title_states_the_length_and_the_lead(self):
        t = compilation.default_title(["Tilly the Train Counts to Ten", "Ellie Waits"], 1830.0)
        self.assertTrue(t.startswith("30 Minutes of Tilly the Train Counts to Ten"))
        self.assertTrue(t.endswith("& More"))

    def test_the_fallback_falls_back_to_the_channel(self):
        self.assertEqual(compilation.default_title([], 600.0, "Bright Minds Town"),
                         "10 Minutes of Bright Minds Town")
        self.assertEqual(compilation.default_title([], 600.0), "10 Minutes of Compilation")

    def test_the_fallback_title_is_clamped(self):
        t = compilation.default_title(["x" * 400], 600.0)
        self.assertLessEqual(len(t), compilation.MAX_TITLE_CHARS)


CHANNEL = {"name": "Bright Minds Town", "purpose": "", "audience": "kids",
           "video_type": "animated", "cast": []}
SEO = {"seo_title": "18 Minutes of Tilly the Train | Toddler Songs",
       "seo_keywords": ["tilly the train"], "seo_description": "Body text #trains",
       "thumbnail_texts": ["18 Minutes"]}


class Stitch(unittest.TestCase):
    """pipeline._run_compilation: the orchestration, with every ffmpeg call mocked away.

    The stitch itself is ffmpeg's job and is verified by hand; what matters here is that it is
    local-only, idempotent, and that a failure of the one LLM call cannot lose it."""

    PART_SECONDS = 200.0        # each source video
    TOTAL_SECONDS = 600.0       # the joined result

    def _duration(self, path):
        """Stand in for ffprobe: a part is PART_SECONDS, the joined video is TOTAL_SECONDS.

        One blanket number would make the chapter maths degenerate (every part starting at the
        total length), which is the opposite of what these tests are checking."""
        p = Path(path)
        if p.name.startswith("part_") or p.parent.name in self.part_ids:
            return self.PART_SECONDS
        return self.TOTAL_SECONDS

    def setUp(self):
        shutil.rmtree(config.OUTPUT_DIR, ignore_errors=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.part_ids = {f"20260101-00000{i}-aaaaa{i}" for i in (1, 2, 3)}
        for p in (mock.patch.object(video, "_run"),
                  mock.patch.object(video, "duration", side_effect=self._duration),
                  mock.patch.object(video, "frame_size", return_value=(720, 1280)),
                  mock.patch.object(video, "has_audio", return_value=True),
                  mock.patch.object(pipeline, "_thumb_variants", return_value=1),
                  # the submit starts the stitch in its own thread; these tests drive
                  # _run_compilation themselves so the assertions are not racing it
                  mock.patch.object(pipeline.JobManager, "_start_compilation")):
            p.start()
            self.addCleanup(p.stop)
        self.mgr = pipeline.JobManager()
        self.parts = [self._part(f"20260101-00000{i}-aaaaa{i}", f"Video {i}") for i in (1, 2, 3)]
        for p in self.parts:
            self.mgr.jobs[p.id] = p

    def _part(self, jid, title):
        job = pipeline.Job(id=jid, url="", source="topic", title=title, status="done",
                           final="final.mp4", aspect_ratio="9:16 (Portrait Widescreen)",
                           channel_id="ch01", owner="u1", channel=dict(CHANNEL))
        job.save()
        (job.dir / "final.mp4").write_bytes(b"x" * 20000)
        (job.dir / "thumb_base.jpg").write_bytes(b"jpeg")
        return job

    def _submit(self):
        return self.mgr.submit_compilation([p.id for p in self.parts], channel=None, owner="u1")

    def _run(self, seo=SEO):
        with mock.patch.object(llm, "write_compilation_seo", return_value=seo) as m:
            job = self._submit()
            pipeline._run_compilation(job)
        return job, m

    def _plan(self, job):
        return json.loads((job.dir / "plan.json").read_text(encoding="utf-8"))

    def test_the_job_is_a_normal_job_with_the_parts_on_it(self):
        job, _m = self._run()
        self.assertEqual(job.source, compilation.SOURCE)
        self.assertEqual(job.parts, [p.id for p in self.parts])
        self.assertEqual(job.status, "done")
        self.assertEqual(job.final, "final.mp4")
        self.assertEqual(job.aspect_ratio, "9:16 (Portrait Widescreen)")
        self.assertEqual(job.channel_id, "ch01")
        self.assertEqual(job.title, SEO["seo_title"])

    def test_it_never_touches_the_gpu_queue(self):
        """The whole job is ffmpeg over files that already exist; queueing it would mean waiting
        an hour for a minute of work."""
        job, _m = self._run()
        self.assertNotIn(job.id, self.mgr.queue)
        self.assertIsNone(self.mgr.current)

    def test_the_plan_gets_the_parts_the_chapters_and_the_seo(self):
        job, _m = self._run()
        plan = self._plan(job)
        self.assertEqual([p["job_id"] for p in plan["parts"]], [p.id for p in self.parts])
        self.assertEqual([p["title"] for p in plan["parts"]], ["Video 1", "Video 2", "Video 3"])
        self.assertEqual([c["title"] for c in plan["chapters"]],
                         ["Video 1", "Video 2", "Video 3"])
        self.assertEqual(plan["chapters"][0]["t"], 0.0)
        self.assertEqual(plan["seo_description"], SEO["seo_description"])
        self.assertEqual(plan["title"], plan["seo_title"])

    def test_the_description_that_is_published_carries_the_chapters(self):
        job, _m = self._run()
        desc = pipeline.upload_description(self._plan(job))
        self.assertIn("Body text #trains", desc)
        self.assertIn("0:00 Video 1", desc)
        self.assertIn("Video 3", desc)

    def test_the_parts_are_normalized_to_one_size_then_joined(self):
        with mock.patch.object(video, "normalize_for_concat") as norm, \
             mock.patch.object(video, "concat_clips") as cat, \
             mock.patch.object(llm, "write_compilation_seo", return_value=SEO):
            pipeline._run_compilation(self._submit())
        self.assertEqual(norm.call_count, 3)
        # every part to the same frame, which is the largest among them
        self.assertEqual({c.args[2:4] for c in norm.call_args_list}, {(720, 1280)})
        self.assertEqual([Path(c.args[1]).name for c in norm.call_args_list],
                         ["part_01.mp4", "part_02.mp4", "part_03.mp4"])
        self.assertEqual(Path(cat.call_args.args[1]).name, "final.mp4")

    def test_the_normalized_copies_are_cleaned_up(self):
        """They are a transcode of videos that still exist, so keeping them would double a long
        compilation's disk for something a re-stitch can make again."""
        job, _m = self._run()
        self.assertEqual(list(job.dir.glob("part_*")), [])
        self.assertFalse((job.dir / "final.txt").exists())   # the concat demuxer list

    def test_a_thumbnail_is_adopted_from_a_part_so_nothing_renders(self):
        job, _m = self._run()
        self.assertTrue((job.dir / "thumb_base.jpg").exists())
        self.assertTrue(any("thumbnail taken from Video 1" in line for line in job.log))
        pipeline._thumb_variants.assert_called()

    def test_no_part_has_a_thumbnail_is_not_an_error(self):
        for p in self.parts:
            (p.dir / "thumb_base.jpg").unlink()
        job, _m = self._run()
        self.assertEqual(job.status, "done")
        self.assertTrue(any("no part had a thumbnail" in line for line in job.log))

    def test_a_dead_llm_costs_the_title_not_the_compilation(self):
        """The stitch is the expensive, irreplaceable part of the job."""
        with mock.patch.object(llm, "write_compilation_seo", side_effect=RuntimeError("down")):
            job = self._submit()
            pipeline._run_compilation(job)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.title, "10 Minutes of Video 1 & More")
        self.assertEqual(self._plan(job)["seo_description"], "")
        self.assertEqual([c["title"] for c in self._plan(job)["chapters"]],
                         ["Video 1", "Video 2", "Video 3"])

    def test_a_resume_does_not_re_encode_what_is_already_there(self):
        job, _m = self._run()
        (job.dir / "final.mp4").write_bytes(b"x" * 20000)      # _run left it mocked away
        plan = self._plan(job)
        plan.pop("seo_title")
        (job.dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with mock.patch.object(video, "concat_clips") as cat, \
             mock.patch.object(video, "normalize_for_concat") as norm, \
             mock.patch.object(llm, "write_compilation_seo", return_value=SEO):
            pipeline._run_compilation(job)
        cat.assert_not_called()
        norm.assert_not_called()
        # the chapters still land, off the lengths measured on the first pass
        self.assertEqual(len(self._plan(job)["chapters"]), 3)

    def test_a_deleted_part_fails_the_job_naming_the_video(self):
        job = self._submit()
        shutil.rmtree(self.parts[1].dir)
        pipeline.run_job(job)       # the real dispatcher, so the error handling is exercised
        self.assertEqual(job.status, "failed")
        self.assertIn(self.parts[1].id, job.error)

    def test_a_compilation_cannot_be_built_from_an_unknown_or_unfinished_video(self):
        with self.assertRaises(ValueError):
            self.mgr.submit_compilation(["nope", self.parts[0].id], channel=None, owner="u1")
        self.parts[1].set("clips", "rendering")
        with self.assertRaises(ValueError):
            self.mgr.submit_compilation([p.id for p in self.parts], channel=None, owner="u1")

    def test_a_failed_compilation_resumes_without_queueing(self):
        job = self._submit()
        job.set("failed", "boom", error="boom")
        with mock.patch.object(pipeline.JobManager, "_start_compilation") as start:
            back = self.mgr.resume(job.id)
        self.assertEqual(back.status, "assembling")
        self.assertNotIn(job.id, self.mgr.queue)
        start.assert_called_once()

    def test_cancelling_a_running_compilation_stops_it(self):
        """It is not in the queue, so cancel sets the flag and _run_compilation's checkpoints
        between parts have to honour it."""
        job = self._submit()
        # cancel also pokes ComfyUI to abort an in-flight render; a compilation has none, and
        # letting it try would spend the test's time waiting for a GPU box that is not here
        with mock.patch.object(pipeline.comfy, "ComfyClient"):
            self.mgr.cancel(job.id)
        self.assertTrue(job.cancel_requested)
        with mock.patch.object(llm, "write_compilation_seo", return_value=SEO):
            pipeline.run_job(job)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "cancelled by user")

    def test_run_job_sends_a_compilation_down_the_stitch_path(self):
        """_load_existing re-queues every unfinished job after a restart, so the branch has to be
        in the dispatcher and not only at submit time."""
        job = self._submit()
        with mock.patch.object(pipeline, "_run_compilation") as run, \
             mock.patch.object(pipeline, "_run") as full:
            pipeline.run_job(job)
        run.assert_called_once()
        full.assert_not_called()


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
