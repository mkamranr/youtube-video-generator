"""Story videos: the clip grid, the spoken-line timing, the prompts and the job plumbing.

Run with  python -m unittest discover -s tests -t .  from the repo root.

Pure unit tests: no ComfyUI, no Fish Speech, no vLLM (the LLM chat call is mocked and so is every
TTS call). `config.OUTPUT_DIR` is patched to a throw-away directory before pipeline is imported.

The load-bearing assertion of the whole feature is in StoryPrompts: the writer that produces the
spoken lines must never be shown a character's look, because a writer that cannot see the look
cannot recite it into the voiceover.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-story-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import cast, channels, llm, story, tts  # noqa: E402

FULL = 124 / 24     # what the video model renders for a requested 5 s clip

ELLIE = {"id": "c1", "name": "Ellie", "kind": "5-year-old anthropomorphic baby elephant",
         "role": "main", "look": "Lavender-gray skin, teal overalls, a yellow sun hat.",
         "personality": "Gentle and curious.", "voice_id": "v-ellie"}
PIP = {"id": "c2", "name": "Pip", "kind": "Baby bluebird", "role": "supporting",
       "look": "Sky-blue feathers, a red knitted scarf.", "personality": "Cheerful.",
       "voice_id": ""}


def _channel(audience="kids", cast_list=None, characters=True, **kw):
    return {"id": "ch01", "name": "Bright Minds Town", "purpose": "", "rules": "", "owner": "u1",
            "audience": audience, "video_type": "animated", "cast_opening": True,
            "characters_enabled": characters,
            "cast": [dict(c) for c in (cast_list if cast_list is not None else [ELLIE, PIP])],
            "defaults": {"art_style": "", "audio_mode": "music", "voice_id": "",
                         "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                         "video_length": "45-60 seconds", "storyboard": False,
                         "song_voice": "child", "video_mode": ""},
            **kw}


BEATS = [{"setting": "on the sunny platform", "action": "Ellie waits by the yellow line",
          "characters": ["Ellie"]},
         {"setting": "on the sunny platform", "action": "The train pulls in and Ellie steps aboard",
          "characters": ["Ellie"]},
         {"setting": "inside the carriage", "action": "Ellie settles into a window seat",
          "characters": ["Ellie"]},
         {"setting": "inside the carriage", "action": "Pip lands on the seat beside Ellie",
          "characters": ["Ellie", "Pip"]}]

# one narrator line, one line by a character with a configured voice, one by a character without
LINES = [{"speaker": "Narrator", "text": "Ellie waited for the train to arrive."},
         {"speaker": "Narrator", "text": "All aboard!"},
         {"speaker": "Ellie", "text": "A window seat!"},
         {"speaker": "Pip", "text": "There you are!"}]


# ---------------------------------------------------------------- pure helpers
class StoryGrid(unittest.TestCase):
    def test_beat_count_follows_the_target_and_clamps(self):
        self.assertEqual(story.beat_count(52.5, 5), 10)        # the 45-60 s preset
        self.assertEqual(story.beat_count(270, 10), 27)        # 4-5 minutes at the 10 s rung
        self.assertEqual(story.beat_count(5, 5), config.STORY_MIN_BEATS)
        self.assertEqual(story.beat_count(100000, 5), config.STORY_MAX_BEATS)
        self.assertEqual(story.beat_count(0, 5), config.STORY_MIN_BEATS)

    def test_the_clip_count_stays_sane_across_every_preset(self):
        for name, v in config.VIDEO_LENGTHS.items():
            if not v:
                continue
            mid = (v[0] + v[1]) / 2.0
            n = story.beat_count(mid, config.video_length_clip_seconds(name))
            self.assertTrue(config.STORY_MIN_BEATS <= n <= 30, f"{name} gave {n} beats")

    def test_line_words_fit_the_clip(self):
        with mock.patch.object(config, "STORY_WPS", 2.2), \
                mock.patch.object(config, "STORY_LINE_LEAD", 0.2), \
                mock.patch.object(config, "STORY_LINE_GAP", 0.6):
            self.assertEqual(story.line_words(5), int((FULL - 0.8) * 2.2))
            self.assertGreater(story.line_words(10), story.line_words(5))
            self.assertGreaterEqual(story.max_line_words(), story.line_words(5))
            self.assertEqual(story.line_words(0.1), 4)      # the floor, never zero words

    def test_a_short_line_still_buys_a_full_clip(self):
        durs = story.clip_durations([0.4, 0.0, 1.2], 5)
        self.assertEqual(durs, [FULL, FULL, FULL])

    def test_a_long_line_stretches_only_its_own_clip(self):
        durs = story.clip_durations([1.0, 9.0, 1.0], 5)
        self.assertAlmostEqual(durs[0], FULL)
        self.assertAlmostEqual(durs[2], FULL)
        self.assertGreater(durs[1], 9.0)                   # the words always fit
        self.assertGreater(durs[1], FULL)

    def test_every_duration_is_a_legal_render_length(self):
        durs = story.clip_durations([0.3, 4.9, 7.7, 11.4], 5)
        for d in durs:
            self.assertAlmostEqual(config.actual_clip_seconds(d), d, places=9)
            self.assertEqual(config.clip_frames(d) % config.FRAME_QUANTUM, 5 % config.FRAME_QUANTUM)


class StoryTimeline(unittest.TestCase):
    def test_times_are_relative_to_the_clip_and_the_tail_is_silent(self):
        durs = story.clip_durations([2.0, 0.0], 5)
        timed = story.timeline(LINES[:2] and [{"text": "Ellie waited."}, {"text": ""}],
                               [2.0, 0.0], durs)
        self.assertEqual([t["clip"] for t in timed], [1, 2])
        self.assertEqual(timed[0]["text"], "Ellie waited.")
        first = timed[0]["lines"][0]
        self.assertAlmostEqual(first["start"], config.STORY_LINE_LEAD, places=3)
        self.assertAlmostEqual(first["end"], config.STORY_LINE_LEAD + 2.0, places=3)
        self.assertTrue(timed[0]["lines"][-1]["instrumental"])
        self.assertAlmostEqual(timed[1]["start"], durs[0], places=3)
        # clip 2 has no line at all
        self.assertEqual(timed[1]["text"], "")
        self.assertEqual([e.get("text") for e in timed[1]["lines"]], [""])

    def test_format_says_no_speech_not_instrumental(self):
        durs = story.clip_durations([2.0], 5)
        timed = story.timeline([{"text": "Ellie waited."}], [2.0], durs)
        s = story.format_timeline(timed[0])
        self.assertIn('"Ellie waited."', s)
        self.assertIn("(no speech)", s)
        self.assertNotIn("instrumental", s)
        silent = story.format_timeline(story.timeline([{"text": ""}], [0.0], durs)[0])
        self.assertIn("no speech during this clip", silent)

    def test_a_line_longer_than_its_clip_still_fits(self):
        durs = story.clip_durations([11.0], 5)
        timed = story.timeline([{"text": "a very long line"}], [11.0], durs)
        self.assertLessEqual(timed[0]["lines"][0]["end"], durs[0] + 1e-6)
        self.assertGreaterEqual(timed[0]["lines"][0]["end"], 11.0)


class DescribesLook(unittest.TestCase):
    CHARS = [ELLIE, PIP]

    def test_a_clean_story_line_passes(self):
        for line in ["Ellie waited for the train to arrive.",
                     "Ellie boarded the blue train.",          # a prop, not the character
                     "All aboard!",
                     "Pip flew down and landed beside her.",
                     "Ellie looked around at the green hills.",
                     ""]:
            self.assertEqual(story.describes_look(line, self.CHARS), [], line)

    def test_appearance_and_camera_words_are_caught(self):
        self.assertIn("elephant", story.describes_look("Ellie the elephant boarded.", self.CHARS))
        self.assertIn("bluebird", story.describes_look("Pip the bluebird sang.", self.CHARS))
        self.assertIn("yellow", story.describes_look("Ellie's yellow hat blew away.", self.CHARS))
        self.assertIn("wearing", story.describes_look("Ellie, wearing her hat, waved.", self.CHARS))
        self.assertIn("we see", story.describes_look("We see Ellie step out.", self.CHARS))
        self.assertIn("the camera", story.describes_look("The camera follows Ellie.", self.CHARS))
        self.assertEqual(story.describes_look("Ellie the elephant boarded.", []), [])

    def test_slug_is_filename_safe(self):
        self.assertEqual(story.slug("Mrs. Hazel"), "mrs_hazel")
        self.assertEqual(story.slug(""), "narrator")


# ---------------------------------------------------------------------- prompts
def _capture(reply):
    calls = []

    def fake(messages, max_tokens=8000):
        calls.append(messages)
        return dict(reply)
    return calls, fake


class StoryPrompts(unittest.TestCase):
    def test_the_beat_writer_is_told_not_to_describe_or_frame(self):
        calls, fake = _capture({"title": "T", "premise": "p", "style": "s",
                                "characters": ["Ellie"], "beats": BEATS})
        with mock.patch.object(llm, "_chat", fake):
            plan = llm.write_story_beats("Ellie takes the train", "topic", 4, 5.0,
                                         art_style="Low-Poly 3D", channel=_channel())
        sysmsg, user = calls[0][0]["content"], calls[0][1]["content"]
        self.assertEqual(len(plan["beats"]), 4)
        self.assertEqual(plan["cast_chosen"], ["Ellie"])
        self.assertIn("4 BEATS", user)
        self.assertIn("5.2-second", user)
        self.assertIn("NEVER describe how anyone or anything LOOKS", user)
        self.assertIn("NEVER write camera", user)
        self.assertIn("Low-Poly 3D", user)
        # the roster is there by name, but the looks are not
        self.assertIn("Ellie", sysmsg)
        self.assertIn("Pip", sysmsg)
        self.assertNotIn("Lavender-gray", sysmsg)
        self.assertNotIn("teal overalls", sysmsg)
        self.assertNotIn("knitted scarf", sysmsg)
        self.assertIn("short PLAN", sysmsg)

    def test_the_line_writer_never_sees_a_look(self):
        calls, fake = _capture({"lines": LINES})
        with mock.patch.object(llm, "_chat", fake):
            lines, issues = llm.write_story_lines(BEATS, ["Ellie", "Pip"], channel=_channel(),
                                                  used=["Ellie", "Pip"], max_words=9)
        sysmsg, user = calls[0][0]["content"], calls[0][1]["content"]
        self.assertEqual(issues, [])
        self.assertEqual([ln["speaker"] for ln in lines],
                         ["Narrator", "Narrator", "Ellie", "Pip"])
        self.assertNotIn("Lavender-gray", sysmsg)
        self.assertNotIn("teal overalls", sysmsg)
        self.assertNotIn("knitted scarf", sysmsg)
        self.assertIn("name only", sysmsg)
        # the user prompt carries the beats, the roster, the budget and the worked example
        self.assertIn("Ellie waits by the yellow line", user)
        self.assertIn("Ellie, Pip, Narrator", user)
        self.assertIn("at most 9 words", user)
        self.assertIn("WRONG:", user)
        self.assertIn("RIGHT:", user)
        self.assertIn("must NEVER describe that picture", user)

    def test_a_short_or_off_roster_reply_is_repaired(self):
        calls, fake = _capture({"lines": [{"speaker": "Conductor", "text": "All aboard!"}]})
        with mock.patch.object(llm, "_chat", fake):
            lines, issues = llm.write_story_lines(BEATS, ["Ellie", "Pip"], channel=_channel(),
                                                  used=["Ellie"], max_words=9)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], {"speaker": "Narrator", "text": "All aboard!"})
        self.assertEqual([ln["text"] for ln in lines[1:]], ["", "", ""])
        self.assertTrue(any("Conductor" in i for i in issues))
        self.assertTrue(any("4 beats" in i for i in issues))

    def test_too_few_beats_is_an_error_not_a_short_video(self):
        _calls, fake = _capture({"title": "T", "premise": "p", "style": "s", "beats": []})
        with mock.patch.object(llm, "_chat", fake), self.assertRaises(RuntimeError):
            llm.write_story_beats("x", "topic", 6, 5.0, channel=_channel())

    def test_a_rewrite_asks_for_one_line_only(self):
        calls, fake = _capture({"text": "Ellie waited for the train."})
        with mock.patch.object(llm, "_chat", fake):
            out = llm.rewrite_story_line({"speaker": "Narrator", "text": "Ellie the elephant waited"},
                                         BEATS[0], "describes", 9, channel=_channel(),
                                         used=["Ellie"], speakers=["Ellie"])
        user = calls[0][1]["content"]
        self.assertEqual(out["text"], "Ellie waited for the train.")
        self.assertIn("describes the picture", user)
        self.assertIn("Ellie waits by the yellow line", user)
        self.assertIn("at most 9 words", user)


class StoryScenes(unittest.TestCase):
    def test_a_scene_is_built_from_the_beat_not_the_words(self):
        calls, fake = _capture({"scene": "A scene. No text, subtitles, logos or watermarks of any kind."})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            scenes = llm.write_scenes("", "style", 2, kind="story being told",
                                      segments=["Ellie waited for the train.", ""],
                                      timelines=['[0.2s-2.2s] "Ellie waited for the train."  '
                                                 '[2.2s-5.2s] (no speech)', ""],
                                      clip_secs=[FULL, FULL], beats=BEATS[:2],
                                      channel=_channel())
        self.assertEqual(len(scenes), 2)
        first = calls[0][1]["content"]
        self.assertIn("BEAT 1 of the story", first)
        self.assertIn("Ellie waits by the yellow line", first)
        self.assertIn("on the sunny platform", first)
        self.assertIn('"Ellie waited for the train."', first)
        self.assertIn("spoken words are the SOUNDTRACK", first)
        self.assertIn("This clip is 5.2 seconds long", first)
        # none of the song-only wording leaks into a story
        self.assertNotIn("sung", first)
        self.assertNotIn("never invent a performer", first)
        second = calls[1][1]["content"]
        self.assertIn("No one speaks during this clip", second)
        self.assertIn("The train pulls in", second)

    def test_the_song_branch_is_untouched_when_there_are_no_beats(self):
        calls, fake = _capture({"scene": "s"})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("", "style", 1, kind="song being played", segments=["Good morning sun"],
                             timelines=['[0.0s-1.8s] "Good morning sun"'], clip_secs=[FULL],
                             channel=_channel())
        self.assertIn("never invent a performer", calls[0][1]["content"])

    def test_the_reviewer_checks_the_beat_and_drops_the_sung_wording(self):
        calls, fake = _capture({"issues": [], "scene": "same"})
        with mock.patch.object(llm, "_chat", fake):
            llm.review_scene("same", "Ellie waited.", "style", 1, 3, kind="story being told",
                             timeline='[0.2s-2.2s] "Ellie waited."',
                             beat="Ellie waits by the yellow line")
        user = calls[0][1]["content"]
        self.assertIn("one beat of a story", user)
        self.assertIn("Ellie waits by the yellow line", user)
        self.assertIn("5. TIMING", user)
        self.assertNotIn("sung", user)
        self.assertNotIn("mis-hearings", user)


# ------------------------------------------------------------- option resolving
class ResolveVideoMode(unittest.TestCase):
    def test_the_default_is_a_song_video(self):
        self.assertEqual(pipeline.resolve_video_mode("", _channel(), "topic"), "song")
        self.assertEqual(pipeline.resolve_video_mode("song", _channel(), "topic"), "song")
        self.assertEqual(pipeline.resolve_video_mode("story", _channel(), "topic"), "story")
        self.assertEqual(pipeline.resolve_video_mode("story", _channel(), "youtube"), "story")
        self.assertEqual(pipeline.resolve_video_mode("story", _channel(), "description"), "story")

    def test_every_rejection(self):
        for mode, ch, source, kw, needle in [
                ("quiz", _channel(), "topic", {}, "unsupported video type"),
                ("story", _channel("general"), "topic", {}, "kids channels"),
                ("story", _channel(cast_list=[]), "topic", {}, "characters"),
                ("story", _channel(characters=False), "topic", {}, "characters"),
                ("story", _channel(), "transcript", {}, "topic, a description or a YouTube"),
                ("story", _channel(), "repo", {}, "topic, a description or a YouTube"),
                ("story", _channel(), "topic", {"lyrics": "la la"}, "spoken, not sung"),
                ("story", _channel(), "topic", {"upload_id": "u1"}, "uploaded music track")]:
            with self.assertRaises(ValueError) as cm:
                pipeline.resolve_video_mode(mode, ch, source, **kw)
            self.assertIn(needle, str(cm.exception))

    def test_is_story_never_branches_on_truthiness(self):
        self.assertFalse(pipeline._is_story(pipeline.Job(id="t", url="")))
        self.assertFalse(pipeline._is_story(pipeline.Job(id="t", url="", video_mode="song")))
        self.assertTrue(pipeline._is_story(pipeline.Job(id="t", url="", video_mode="story")))

    def test_resolve_choices_signature_is_untouched(self):
        r = pipeline.JobManager._resolve_choices("", "", "", "", "", {"audience": "kids", "defaults": {}})
        self.assertEqual(len(r), 14)   # ..., polish, video_model, megapixels, resolution
        self.assertEqual(r[-1], config.DEFAULT_RESOLUTION)   # tests/test_resolution.py indexes [-1]
        self.assertEqual(r[-3], config.DEFAULT_VIDEO_MODEL)


# ------------------------------------------------------------------- job plumbing
class StoryJobs(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        patcher = mock.patch.object(pipeline.JobManager, "_start_draft")
        self.draft = patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = pipeline.JobManager()

    def test_a_story_keeps_its_narrator_voice_and_gets_a_voiceover_mix(self):
        job = self.mgr.submit_prompt("Ellie takes the train", "topic", channel=_channel(),
                                     owner="u1", video_mode="story", voice_id="narr1")
        self.assertEqual(job.video_mode, "story")
        self.assertEqual(job.audio_mode, "narration+music")   # never the channel default "music"
        self.assertEqual(job.voice_id, "narr1")               # _resolve_choices would blank it
        self.assertFalse(job.storyboard)                      # fact cards do not belong on a story
        self.assertEqual(job.status, "drafting")

    def test_a_url_story_job_drafts_instead_of_queueing(self):
        job = self.mgr.submit("https://youtu.be/x", channel=_channel(), owner="u1",
                              video_mode="story")
        self.assertEqual(job.status, "drafting")
        self.assertEqual(self.mgr.queue, [])                  # it must reach the approval step
        self.draft.assert_called_once()

    def test_a_song_url_job_still_queues(self):
        job = self.mgr.submit("https://youtu.be/x", channel=_channel(), owner="u1")
        self.assertEqual(job.status, "queued")
        self.assertEqual(self.mgr.queue, [job.id])
        self.draft.assert_not_called()

    def test_a_legacy_job_loads_as_a_song_video(self):
        job = pipeline.Job(id="legacy", url="")
        job.save()
        self.assertEqual(pipeline.Job.load(job.dir / "job.json").video_mode, "")
        self.assertFalse(pipeline._is_story(pipeline.Job.load(job.dir / "job.json")))

    def test_setup_planned_inherits_the_jobs_own_format(self):
        planned = self.mgr.submit_prompt("Ellie takes the train", "topic", channel=_channel(),
                                         owner="u1", start=False, video_mode="story")
        self.assertEqual(planned.status, "planned")
        self.assertEqual(planned.video_mode, "story")
        out = self.mgr.setup_planned(planned.id)              # the form left it blank
        self.assertEqual(out.video_mode, "story")
        self.assertEqual(out.audio_mode, "narration+music")
        self.assertEqual(out.status, "drafting")

    def test_setup_planned_can_switch_back_to_a_song(self):
        planned = self.mgr.submit_prompt("Ellie takes the train", "topic", channel=_channel(),
                                         owner="u1", start=False, video_mode="story")
        out = self.mgr.setup_planned(planned.id, video_mode="song", audio_mode="music")
        self.assertEqual(out.video_mode, "song")
        self.assertEqual(out.audio_mode, "music")


# ----------------------------------------------------------------- the approval
class StoryApproval(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        patcher = mock.patch.object(pipeline.JobManager, "_start_draft")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = pipeline.JobManager()
        self.job = self.mgr.submit_prompt("Ellie takes the train", "topic", channel=_channel(),
                                          owner="u1", video_mode="story")
        self.plan = {"title": "T", "style": "s", "narration": "", "cast_used": ["Ellie", "Pip"],
                     "story": {"premise": "p", "beats": BEATS,
                               "lines": [dict(ln) for ln in LINES], "issues": []}}
        self.job.dir.mkdir(parents=True, exist_ok=True)
        (self.job.dir / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        self.job.set("awaiting_approval", "ready")

    def _plan(self):
        return json.loads((self.job.dir / "plan.json").read_text(encoding="utf-8"))

    def test_validate_story_lines(self):
        out = pipeline._validate_story_lines(
            [{"speaker": "Ellie", "text": "  Here   it comes! "},
             {"speaker": "nobody", "text": "All aboard!"},
             {"speaker": "Pip", "text": "Hi!"},
             {"speaker": "", "text": ""}], LINES, ["Ellie", "Pip"])
        self.assertEqual([ln["speaker"] for ln in out],
                         ["Ellie", "Narrator", "Pip", "Narrator"])
        self.assertEqual(out[0]["text"], "Here it comes!")
        for bad, needle in [("nope", "list of spoken lines"),
                            ([{"text": "a"}], "4 scenes"),
                            (["w", "x", "y", "z"], "must be an object"),
                            ([{"text": "x" * 500}, {}, {}, {}], "too long")]:
            with self.assertRaises(ValueError) as cm:
                pipeline._validate_story_lines(bad, LINES, ["Ellie"])
            self.assertIn(needle, str(cm.exception))

    def test_approving_edited_lines_rewrites_the_narration(self):
        edited = [{"speaker": "Ellie", "text": "Here comes the train!"},
                  {"speaker": "Narrator", "text": "All aboard!"},
                  {"speaker": "Ellie", "text": ""},
                  {"speaker": "Pip", "text": ""}]
        out = self.mgr.approve_script(self.job.id, "New title", "IGNORED PROSE", edited)
        plan = self._plan()
        self.assertEqual(out.status, "queued")
        self.assertTrue(out.script_approved)
        self.assertEqual(plan["title"], "New title")
        self.assertEqual([ln["text"] for ln in plan["story"]["lines"]],
                         ["Here comes the train!", "All aboard!", "", ""])
        # the free-text box is never applied to a story: the lines ARE the narration
        self.assertEqual(plan["narration"], "Here comes the train!\nAll aboard!")
        self.assertEqual(plan["cast_used"], ["Ellie", "Pip"])   # Pip is still in the beats

    def test_approving_without_lines_keeps_the_draft(self):
        self.mgr.approve_script(self.job.id)
        plan = self._plan()
        self.assertEqual([ln["text"] for ln in plan["story"]["lines"]],
                         [ln["text"] for ln in LINES])
        self.assertIn("All aboard!", plan["narration"])

    def test_a_bad_edit_leaves_the_job_awaiting_approval(self):
        with self.assertRaises(ValueError):
            self.mgr.approve_script(self.job.id, "T", "", [{"text": "only one"}])
        self.assertEqual(self.job.status, "awaiting_approval")

    def test_an_edited_line_forgets_its_old_audio(self):
        vdir = self.job.dir / "voices"
        vdir.mkdir(parents=True, exist_ok=True)
        for i in (1, 2, 3, 4):
            (vdir / f"line_{i:02d}.wav").write_bytes(b"x" * 2000)
        lines = [dict(ln) for ln in LINES]
        for ln in lines:
            ln["spoken_text"] = ln["text"]
        (self.job.dir / "plan.json").write_text(
            json.dumps({**self.plan, "story": {**self.plan["story"], "lines": lines}}),
            encoding="utf-8")
        edited = [dict(ln) for ln in lines]
        edited[1]["text"] = "Climb aboard!"
        self.mgr.approve_script(self.job.id, "", "", edited)
        self.assertTrue((vdir / "line_01.wav").exists())
        self.assertFalse((vdir / "line_02.wav").exists())    # its words changed
        self.assertTrue((vdir / "line_03.wav").exists())


# ----------------------------------------------------------------- the spoken track
class StoryAudio(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        self.job = pipeline.Job(id="story1", url="", video_mode="story",
                                audio_mode="narration+music", voice_id="narr1",
                                channel=channels.snapshot(_channel()), video_length="45-60 seconds")
        self.job.dir.mkdir(parents=True, exist_ok=True)
        self.plan = {"title": "T", "style": "s", "narration": "n", "cast_used": ["Ellie", "Pip"],
                     "story": {"premise": "p", "beats": BEATS,
                               "lines": [dict(ln) for ln in LINES]}}
        self.spoken = []

        def fake_synth(text, out_wav, on_progress=None, ref_audio=None, ref_text="",
                       keep_voice=True, client=None):
            Path(out_wav).write_bytes(b"x" * 2000)
            self.spoken.append({"text": text, "ref": str(ref_audio) if ref_audio else None,
                                "ref_text": ref_text})
            return 2.0

        for name, val in [("synthesize", fake_synth), ("connect", lambda: object()),
                          ("audio_duration", lambda p: 2.0),
                          ("concat_timed", lambda parts, out, total=0.0:
                           (Path(out).write_bytes(b"x" * 4000), float(total))[1])]:
            patcher = mock.patch.object(tts, name, val)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _ref(self, voice_id):
        return mock.patch.object(pipeline.voices, "reference",
                                 lambda vid: (Path(f"/lib/{vid}.wav"), f"sample {vid}") if vid else None)

    def test_each_speaker_uses_its_own_voice(self):
        with self._ref("x"):
            _f, durs, secs = pipeline._story_audio(self.job, self.plan)
        self.assertEqual(len(durs), 4)
        self.assertEqual(secs, [2.0, 2.0, 2.0, 2.0])
        # the Narrator lines clone the job voice and Ellie her own; Pip has none configured, so a
        # voice is invented for her and then kept
        self.assertEqual(self.spoken[0]["ref"], str(Path("/lib/narr1.wav")))
        self.assertEqual(self.spoken[1]["ref"], str(Path("/lib/narr1.wav")))
        self.assertEqual(self.spoken[2]["ref"], str(Path("/lib/v-ellie.wav")))
        self.assertEqual(self.spoken[2]["ref_text"], "sample v-ellie")
        self.assertIsNone(self.spoken[3]["ref"])

    def test_a_speaker_without_a_voice_keeps_the_one_it_gets(self):
        plan = {**self.plan, "story": {**self.plan["story"],
                                       "lines": [{"speaker": "Pip", "text": "Hello!"},
                                                 {"speaker": "Pip", "text": "Over here!"},
                                                 {"speaker": "Pip", "text": "Wait for me!"},
                                                 {"speaker": "Pip", "text": "Ready!"}]}}
        with mock.patch.object(pipeline.voices, "reference", lambda vid: None):
            pipeline._story_audio(self.job, plan)
        self.assertIsNone(self.spoken[0]["ref"])                       # invented once
        seed = str(self.job.dir / "voices" / "line_01.wav")
        self.assertEqual(self.spoken[1]["ref"], seed)                  # then reused
        self.assertEqual(self.spoken[2]["ref"], seed)
        self.assertEqual(self.spoken[1]["ref_text"], "Hello!")
        self.assertEqual(plan["story"]["voice_pins"]["Pip"]["file"], "line_01.wav")

    def test_the_channel_default_dresses_every_voiceless_character(self):
        """A kids channel sounds like kids through this one setting: Fish Speech has no voice-type
        knob, so a child sample picked once is what every character without their own voice uses."""
        self.job.channel = channels.snapshot(
            _channel(defaults={**_channel()["defaults"], "character_voice": "v-child"}))
        with self._ref("x"):
            pipeline._story_audio(self.job, self.plan)
        self.assertEqual(self.spoken[2]["ref"], str(Path("/lib/v-ellie.wav")))   # her own wins
        self.assertEqual(self.spoken[3]["ref"], str(Path("/lib/v-child.wav")))   # Pip has none

    def test_a_deleted_channel_voice_still_clones(self):
        self.job.channel = channels.snapshot(
            _channel(defaults={**_channel()["defaults"], "character_voice": "gone"}))
        with mock.patch.object(pipeline.voices, "reference", lambda vid: None):
            pipeline._story_audio(self.job, self.plan)
        self.assertIn("default character voice no longer exists", "\n".join(self.job.log))

    def test_a_deleted_voice_falls_back_and_says_so(self):
        with mock.patch.object(pipeline.voices, "reference", lambda vid: None):
            pipeline._story_audio(self.job, self.plan)
        log = "\n".join(self.job.log)
        self.assertIn("Ellie's reference voice no longer exists", log)
        self.assertIn("reference voice narr1 no longer exists", log)

    def test_a_resume_speaks_nothing_again(self):
        with self._ref("x"):
            _f, durs, _s = pipeline._story_audio(self.job, self.plan)
        spoken_first = len(self.spoken)
        self.assertEqual(spoken_first, 4)
        with self._ref("x"), mock.patch.object(config, "CLIP_SECONDS", 10):
            _f2, durs2, _s2 = pipeline._story_audio(self.job, self.plan)
        self.assertEqual(len(self.spoken), spoken_first)      # nothing was spoken again
        self.assertEqual(durs2, durs)                         # and the grid is frozen

    def test_an_empty_line_is_a_silent_scene(self):
        plan = {**self.plan, "story": {**self.plan["story"],
                                       "lines": [{"speaker": "Narrator", "text": "Hello."},
                                                 {"speaker": "Narrator", "text": ""},
                                                 {"speaker": "Pip", "text": "Hi!"},
                                                 {"speaker": "Pip", "text": "Bye!"}]}}
        with self._ref("x"):
            _f, durs, secs = pipeline._story_audio(self.job, plan)
        self.assertEqual(secs[1], 0.0)
        self.assertAlmostEqual(durs[1], FULL)
        self.assertEqual(len(self.spoken), 3)
        self.assertFalse((self.job.dir / "voices" / "line_02.wav").exists())


# ------------------------------------------------------------- per-character voices
class CastVoices(unittest.TestCase):
    def test_a_cast_voice_survives_validation_and_the_snapshot(self):
        rec = channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                  "cast": [dict(ELLIE), dict(PIP)]})
        self.assertEqual(rec["cast"][0]["voice_id"], "v-ellie")
        self.assertEqual(rec["cast"][1]["voice_id"], "")
        snap = channels.snapshot({**rec, "id": "ch01"})
        self.assertEqual(snap["cast"][0]["voice_id"], "v-ellie")
        self.assertEqual(cast.voice_id_for(snap, "ellie"), "v-ellie")   # case-insensitive
        self.assertEqual(cast.voice_id_for(snap, "Pip"), "")
        self.assertEqual(cast.voice_id_for(snap, "Nobody"), "")

    def test_a_character_may_not_change_voice_every_line(self):
        with self.assertRaises(ValueError) as cm:
            channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                "cast": [{**ELLIE, "voice_id": "random"}]})
        self.assertIn("one steady voice", str(cm.exception))

    def test_characters_off_means_no_character_voices(self):
        snap = channels.snapshot({**_channel(), "characters_enabled": False})
        self.assertEqual(cast.voice_id_for(snap, "Ellie"), "")

    def test_an_old_channel_reads_back_with_no_voices(self):
        d = Path(tempfile.mkdtemp(prefix="videogen-chan-"))
        prev_dir, prev_idx = config.CHANNELS_DIR, config.CHANNELS_INDEX
        config.CHANNELS_DIR, config.CHANNELS_INDEX = d, d / "channels.json"
        self.addCleanup(setattr, config, "CHANNELS_INDEX", prev_idx)
        self.addCleanup(setattr, config, "CHANNELS_DIR", prev_dir)
        old = {"channels": [{"id": "c", "name": "N", "audience": "kids", "video_type": "animated",
                             "cast": [{"id": "1", "name": "Ellie", "look": "l"}]}]}
        config.CHANNELS_INDEX.write_text(json.dumps(old), encoding="utf-8")
        self.assertEqual(channels.list_channels()[0]["cast"][0]["voice_id"], "")
        self.assertEqual(channels.list_channels()[0]["defaults"]["song_voice"], "child")


class ChannelStoryDefault(unittest.TestCase):
    def test_a_story_default_forces_a_voiceover(self):
        rec = channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                  "defaults": {"video_mode": "story", "audio_mode": "music",
                                               "voice_id": "narr1"}})
        self.assertEqual(rec["defaults"]["audio_mode"], "narration+music")
        self.assertEqual(rec["defaults"]["voice_id"], "narr1")

    def test_leaving_the_kids_audience_demotes_the_default(self):
        kids = channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                   "defaults": {"video_mode": "story"}})
        grown = channels._validate({"audience": "general"}, base=kids)
        self.assertEqual(grown["defaults"]["video_mode"], "song")

    def test_the_default_character_voice_round_trips(self):
        rec = channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                  "defaults": {"character_voice": "v-child"}})
        self.assertEqual(rec["defaults"]["character_voice"], "v-child")
        self.assertEqual(channels.snapshot({**rec, "id": "x"})["character_voice"], "v-child")

    def test_the_default_character_voice_may_not_be_random(self):
        with self.assertRaises(ValueError) as cm:
            channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                "defaults": {"character_voice": "random"}})
        self.assertIn("cannot be the random voice", str(cm.exception))

    def test_an_unknown_default_is_rejected(self):
        with self.assertRaises(ValueError):
            channels._validate({"name": "K", "audience": "kids", "video_type": "animated",
                                "defaults": {"video_mode": "quiz"}})


if __name__ == "__main__":
    unittest.main()
