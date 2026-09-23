"""Lyric timing, kids-song pacing and the song voice: run with
python -m unittest discover -s tests -t .  from the repo root.

Pure unit tests: no ComfyUI, whisper or vLLM (the LLM chat call is mocked)."""
import unittest
from unittest import mock

import config
from services import channels, llm, lyrics

FULL = 124 / 24     # what the video model renders for a requested 5 s clip


def W(start, end, word):
    return {"start": start, "end": end, "word": word}


class ClipDurations(unittest.TestCase):
    def test_frame_quantiser_matches_the_workflow(self):
        self.assertEqual(config.clip_frames(5), 124)
        self.assertEqual(config.clip_frames(2), 56)
        self.assertEqual(config.clip_frames(0.1), 5)      # thumbnails: minimal 5-frame render
        self.assertAlmostEqual(config.actual_clip_seconds(5), FULL)

    def test_durations_cover_the_audio_with_a_short_tail(self):
        with mock.patch.object(config, "CLIP_SECONDS", 5):
            durs = config.clip_durations(90.0)
            self.assertEqual(len(durs), 18)
            self.assertTrue(all(abs(d - FULL) < 1e-9 for d in durs[:-1]))
            self.assertLess(durs[-1], FULL)
            self.assertGreaterEqual(sum(durs), 90.0)
            self.assertLess(sum(durs) - 90.0, 0.7)
            self.assertEqual(config.clip_durations(3.0), [config.actual_clip_seconds(3.0)])
            self.assertEqual(len(config.clip_durations(FULL * 3 + 0.2)), 3)   # tiny remainder: no extra clip
            b = config.boundaries(durs)
            self.assertEqual(len(b), 18)
            self.assertTrue(all(x < y for x, y in zip(b, b[1:])))
            self.assertAlmostEqual(b[-1], sum(durs), places=3)

    def test_long_targets_render_longer_clips(self):
        self.assertEqual(config.video_length_clip_seconds("4-5 minutes"), 10)
        self.assertEqual(config.video_length_clip_seconds("3-4 minutes"), 8)
        self.assertEqual(config.video_length_range("4-5 minutes"), (240, 300))
        with mock.patch.object(config, "CLIP_SECONDS", 5):
            self.assertEqual(config.video_length_clip_seconds("Auto"), 5)        # global default
            self.assertEqual(config.video_length_clip_seconds("9-10 minutes"), 5)  # unknown name
            self.assertIsNone(config.video_length_range("Auto"))
            ten = config.actual_clip_seconds(10)      # 243 frames after quantisation
            self.assertAlmostEqual(ten, 243 / 24)
            durs = config.clip_durations(270.0, 10)
            self.assertEqual(len(durs), 27)           # 59 clips at the 5 s default
            self.assertTrue(all(abs(d - ten) < 1e-9 for d in durs[:-1]))
            self.assertGreaterEqual(sum(durs), 270.0)
            self.assertEqual(config.clip_durations(90.0, None), config.clip_durations(90.0))


class Alignment(unittest.TestCase):
    LYRICS = "[Verse]\nGood morning sun\nWake up, little one!\n(oh oh)\n[Chorus]\nF is for fairy"

    def test_lyric_tokens_skip_tags_and_parentheticals(self):
        toks = lyrics.lyric_tokens(self.LYRICS)
        self.assertEqual([t[1] for t in toks],
                         ["Good", "morning", "sun", "Wake", "up,", "little", "one!", "F", "is", "for", "fairy"])
        self.assertEqual([t[2] for t in toks], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    def test_whisper_words_are_stripped(self):
        class Word:
            def __init__(self, start, end, word):
                self.start, self.end, self.word, self.probability = start, end, word, 0.9

        class Seg:
            words = [Word(0.0, 0.4, " Good"), Word(0.4, 0.9, " morning"), Word(0.9, 1.0, " ")]
        out = lyrics.words_from_segments([Seg()])
        self.assertEqual([w["word"] for w in out], ["Good", "morning"])

    def test_mis_hearings_take_the_lyric_text_and_keep_whisper_timing(self):
        words = [W(0.0, 0.4, "good"), W(0.4, 0.9, "morning"), W(0.9, 1.4, "son"),
                 W(2.0, 2.3, "wake"), W(2.3, 2.5, "up"), W(2.5, 2.9, "little"), W(2.9, 3.4, "won"),
                 W(4.0, 4.2, "a"), W(4.2, 4.4, "is"), W(4.4, 4.6, "for"), W(4.6, 5.1, "fairy")]
        out, ratio = lyrics.align_to_lyrics(words, self.LYRICS)
        self.assertGreater(ratio, config.LYRIC_ALIGN_MIN_RATIO)
        self.assertEqual([w["text"] for w in out],
                         ["Good", "morning", "sun", "Wake", "up,", "little", "one!", "F", "is", "for", "fairy"])
        self.assertEqual((out[2]["start"], out[2]["end"]), (0.9, 1.4))
        self.assertEqual(out[7]["text"], "F")             # "A is for fairy" -> the written letter wins
        self.assertEqual(out[7]["line"], 2)

    def test_missed_words_are_interpolated_and_extras_dropped(self):
        words = [W(0.0, 0.4, "good"), W(0.4, 0.9, "morning"), W(0.9, 1.4, "sun"), W(1.5, 1.7, "yeah"),
                 W(1.7, 1.9, "yeah"), W(2.0, 2.3, "wake"), W(2.3, 2.5, "up"), W(2.9, 3.4, "one"),
                 W(4.0, 4.2, "f"), W(4.2, 4.4, "is"), W(4.4, 4.6, "for"), W(4.6, 5.1, "fairy")]
        out, _ratio = lyrics.align_to_lyrics(words, self.LYRICS)
        texts = [w["text"] for w in out]
        self.assertNotIn("yeah", texts)
        little = next(w for w in out if w["text"] == "little")
        self.assertEqual(little["src"], "inferred")
        self.assertGreaterEqual(little["start"], 2.5)
        self.assertLessEqual(little["end"], 2.9)
        self.assertTrue(all(a["end"] <= b["start"] + 1e-9 for a, b in zip(out, out[1:])))

    def test_poor_match_falls_back_to_the_raw_words(self):
        words = [W(0.0, 0.4, "purple"), W(0.4, 0.9, "elephants"), W(0.9, 1.4, "dance"), W(1.5, 1.9, "tonight")]
        out, ratio = lyrics.align_to_lyrics(words, self.LYRICS)
        self.assertLess(ratio, config.LYRIC_ALIGN_MIN_RATIO)
        self.assertEqual([w["text"] for w in out], ["purple", "elephants", "dance", "tonight"])
        self.assertTrue(all(w["src"] == "whisper" for w in out))

    def test_empty_inputs(self):
        self.assertEqual(lyrics.align_to_lyrics([], self.LYRICS), ([], 0.0))
        out, ratio = lyrics.align_to_lyrics([W(0, 1, "hi")], "[Instrumental]")
        self.assertEqual((out[0]["text"], ratio), ("hi", 0.0))
        timed = lyrics.timeline([], config.boundaries([FULL, FULL]))
        self.assertEqual([t["text"] for t in timed], ["", ""])
        self.assertEqual(timed[1]["lines"], [])
        self.assertIn("instrumental for the whole clip", lyrics.format_timeline(timed[0]))


class Timeline(unittest.TestCase):
    def test_real_clip_bounds_fix_the_drift(self):
        words = [W(4.95, 5.25, "sun")]                    # sung at 5.1 s: inside clip 1 as really rendered
        self.assertEqual(lyrics.timeline(words, config.boundaries([FULL, FULL]))[0]["text"], "sun")
        self.assertEqual(lyrics.timeline(words, [5.0, 10.0])[1]["text"], "sun")   # the old 5.0 s windows

    def test_lines_gaps_and_relative_times(self):
        words = [dict(W(0.2, 0.5, "Good"), text="Good", line=0), dict(W(0.5, 0.9, "morning"), text="morning", line=0),
                 dict(W(3.0, 3.4, "Wake"), text="Wake", line=1), dict(W(3.4, 3.8, "up"), text="up", line=1)]
        clip = lyrics.timeline(words, [FULL])[0]
        self.assertEqual(clip["text"], "Good morning Wake up")
        self.assertEqual([bool(l.get("instrumental")) for l in clip["lines"]], [False, True, False, True])
        self.assertEqual(clip["lines"][0]["text"], "Good morning")
        self.assertEqual((clip["lines"][2]["start"], clip["lines"][2]["end"]), (3.0, 3.8))
        s = lyrics.format_timeline(clip)
        self.assertIn('[0.2s-0.9s] "Good morning"', s)
        self.assertIn("[0.9s-3.0s] (instrumental)", s)
        self.assertIn('[3.0s-3.8s] "Wake up"', s)
        # second clip: times restart from that clip's start
        two = lyrics.timeline([dict(W(FULL + 1.0, FULL + 1.5, "hi"), text="hi")], config.boundaries([FULL, FULL]))
        self.assertEqual(two[1]["lines"][0]["start"], 1.0)


class Pace(unittest.TestCase):
    def test_words_per_second(self):
        words = [W(i * 0.5, i * 0.5 + 0.3, "w") for i in range(10)]     # 0.3 s words, 0.2 s gaps
        self.assertAlmostEqual(lyrics.words_per_second(words), round(10 / 4.8, 2))
        self.assertEqual(lyrics.words_per_second(words[:5]), 0.0)
        slow = [W(i * 3.0, i * 3.0 + 0.4, "w") for i in range(10)]       # long pauses count 1.5 s at most
        self.assertAlmostEqual(lyrics.words_per_second(slow), round(10 / (10 * 0.4 + 9 * 1.5), 2))

    def test_needs_slowdown(self):
        with mock.patch.object(config, "SONG_PACE_TOLERANCE", 1.4):
            self.assertTrue(llm.needs_slowdown(1.6, 0.9))
            self.assertFalse(llm.needs_slowdown(1.2, 0.9))
            self.assertFalse(llm.needs_slowdown(0.0, 0.9))    # too few words to judge: no retry

    def test_retry_reason(self):
        with mock.patch.object(config, "SONG_PACE_TOLERANCE", 1.4), \
                mock.patch.object(config, "SONG_SHORT_TOLERANCE", 0.9):
            rng = (120, 180)
            # both true: pace wins, there is only one retry
            self.assertEqual(llm.retry_reason({"wps": 1.6, "duration": 100}, 0.9, rng), "fast")
            self.assertEqual(llm.retry_reason({"wps": 1.0, "duration": 100}, 0.9, rng), "short")
            self.assertEqual(llm.retry_reason({"wps": 1.0, "duration": 115}, 0.9, rng), "")   # inside tolerance
            self.assertEqual(llm.retry_reason({"wps": 1.0, "duration": 150}, 0.9, rng), "")
            self.assertEqual(llm.retry_reason({"wps": 1.0, "duration": 400}, 0.9, rng), "")   # too long: never retried
            self.assertEqual(llm.retry_reason({"wps": 1.0, "duration": 30}, 0.9, None), "")   # Auto: no range


class Caption(unittest.TestCase):
    CHILD = llm.SONG_VOICE_DESCRIPTIONS["child"]

    def test_tempo_and_voice_are_enforced(self):
        cap = ("Global Metadata: Lo-fi hip-hop. 78 BPM, D flat major.\\n\\nVocal Details: Soft androgynous "
               "vocal.\\n\\nArrangement: Dusty drums.")
        out = llm.finalize_caption(cap, bpm=66, voice_desc=self.CHILD)
        self.assertIn("66 BPM", out)
        self.assertNotIn("78 BPM", out)
        self.assertNotIn("\\n", out)
        self.assertIn("\n\nVocal Details: A single young child's voice", out)
        self.assertEqual(llm.finalize_caption(out, bpm=66, voice_desc=self.CHILD), out)   # idempotent

    def test_ranges_and_missing_sections(self):
        self.assertEqual(llm.finalize_caption("Global Metadata: Pop. 80-90 BPM.", bpm=66), "Global Metadata: Pop. 66 BPM.")
        self.assertTrue(llm.finalize_caption("Global Metadata: Pop, C major.", bpm=66)
                        .startswith("Global Metadata: 66 BPM. Pop"))
        self.assertTrue(llm.finalize_caption("Upbeat pop song.", bpm=70).startswith("Global Metadata: 70 BPM."))
        out = llm.finalize_caption("Global Metadata: Pop.", voice_desc=self.CHILD)
        self.assertTrue(out.endswith(f"Vocal Details: {self.CHILD[0].upper() + self.CHILD[1:]}."))
        self.assertEqual(llm.finalize_caption("x", bpm=None, voice_desc=None), "x")

    def test_resolve_song_voice(self):
        self.assertEqual(llm.resolve_song_voice("auto", True), "child")
        self.assertEqual(llm.resolve_song_voice("", False), "auto")
        self.assertEqual(llm.resolve_song_voice("male", True), "male")
        self.assertEqual(llm.resolve_song_voice("robot", True), "child")


def _channel(audience="kids"):
    return {"name": "Bright Minds", "purpose": "", "rules": "", "audience": audience,
            "video_type": "animated", "cast": [], "cast_opening": True}


class Prompts(unittest.TestCase):
    def _capture(self, reply):
        calls = []

        def fake(messages, max_tokens=8000):
            calls.append(messages[-1]["content"])
            return dict(reply)
        return calls, fake

    def test_write_scenes_quotes_the_timeline_and_the_real_clip_length(self):
        calls, fake = self._capture({"scene": "A scene. [0s-2s] x [2s-5.2s] y. No text, subtitles, logos or watermarks of any kind."})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            scenes = llm.write_scenes("", "style", 2, kind="song being played",
                                      segments=["Good morning sun", ""],
                                      timelines=['[0.0s-1.8s] "Good morning sun"  [1.8s-5.2s] (instrumental)', ""],
                                      clip_secs=[FULL, FULL], channel=_channel())
        self.assertEqual(len(scenes), 2)
        p = calls[0]
        self.assertIn("This clip is 5.2 seconds long", p)
        self.assertIn('[0.0s-1.8s] "Good morning sun"', p)
        self.assertIn("never invent a performer", p)
        self.assertIn("[4s-5.2s]", p)
        self.assertIn("INSTRUMENTAL - no words are heard", calls[1])
        self.assertNotIn("TIMELINE - what is sung", calls[1])

    def test_legacy_call_without_timelines_still_works(self):
        calls, fake = self._capture({"scene": "s"})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False), \
                mock.patch.object(config, "CLIP_SECONDS", 5):
            llm.write_scenes("one two three four", "style", 2)
        self.assertIn("This clip is 5.2 seconds long", calls[0])
        self.assertNotIn("TIMELINE - what is sung", calls[0])

    def test_review_scene_timing_check_and_mis_hearing_note(self):
        calls, fake = self._capture({"issues": [], "scene": "same"})
        with mock.patch.object(llm, "_chat", fake):
            llm.review_scene("same", "Good morning", "style", 1, 3, timeline='[0.0s-1.0s] "Good morning"', aligned=True)
            llm.review_scene("same", "Good morning", "style", 1, 3, aligned=False)
        self.assertIn("5. TIMING", calls[0])
        self.assertIn('TIMELINE (what is sung', calls[0])
        self.assertNotIn("mis-hearings", calls[0])
        self.assertIn("mis-hearings", calls[1])
        self.assertNotIn("TIMING", calls[1])

    def test_write_music_kids_forces_tempo_and_a_child_voice(self):
        calls, fake = self._capture({"caption": "Global Metadata: Pop. 120 BPM.\\n\\nVocal Details: Adult male "
                                                "belting.\\n\\nArrangement: drums.", "lyrics": "[Verse]\nla la"})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "KIDS_SONG_BPM", 66), \
                mock.patch.object(config, "KIDS_SONG_WPS", 0.9):
            out = llm.write_music("T", "story", "", standalone=True, seconds=60, channel=_channel(), song_voice="auto")
            out2 = llm.write_music("T", "story", "", standalone=True, seconds=60, channel=_channel(), song_voice="male",
                                   slow_down={"measured": 1.6, "budget": 0.9, "previous_lyrics": "la la la la"})
            bed = llm.write_music("T", "story", "", standalone=False, channel=_channel())
        self.assertIn("66 BPM", out["caption"])
        self.assertNotIn("120 BPM", out["caption"])
        self.assertIn("young child's voice", out["caption"])
        self.assertEqual(out["song_voice"], "child")
        self.assertIn("EXACTLY 66 BPM", calls[0])
        self.assertIn("0.9 sung words per second", calls[0])
        self.assertIn("about 54 words", calls[0])                 # 60 s * 0.9 words/s
        self.assertNotIn("TOO FAST", calls[0])
        self.assertIn("TOO FAST", calls[1])
        self.assertIn("Previous lyrics", calls[1])
        self.assertIn("one warm, friendly male singer", calls[1])
        self.assertEqual(out2["song_voice"], "male")
        self.assertNotIn("young child", bed["caption"])           # instrumental bed: no singer injected
        self.assertIn("66 BPM", bed["caption"])

    def test_write_music_asks_for_longer_lyrics_after_a_short_render(self):
        calls, fake = self._capture({"caption": "Global Metadata: Pop. 66 BPM.", "lyrics": "[Verse]\nla la"})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "KIDS_SONG_WPS", 0.9):
            llm.write_music("T", "story", "", standalone=True, seconds=270, seconds_range=(240, 300),
                            channel=_channel(),
                            lengthen={"measured": 150.0, "low": 240, "target": 270,
                                      "previous_lyrics": "[Verse]\n" + "la " * 100})
            # slow_down wins when both are somehow passed: the two instructions contradict each other
            llm.write_music("T", "story", "", standalone=True, seconds=270, seconds_range=(240, 300),
                            channel=_channel(),
                            slow_down={"measured": 1.6, "budget": 0.9, "previous_lyrics": "la la"},
                            lengthen={"measured": 150.0, "low": 240, "target": 270, "previous_lyrics": "la la"})
        self.assertIn("TOO SHORT", calls[0])
        self.assertIn("150 seconds where the song must last at least 240 seconds", calls[0])
        self.assertIn("at least 180 words", calls[0])      # 100 previous words * 270/150
        self.assertIn("Previous lyrics", calls[0])
        self.assertNotIn("TOO FAST", calls[0])
        self.assertIn("TOO FAST", calls[1])
        self.assertNotIn("TOO SHORT", calls[1])


class ChannelSongVoice(unittest.TestCase):
    def test_channel_default_follows_the_audience(self):
        rec = channels._validate({"name": "K", "audience": "kids", "video_type": "animated"})
        self.assertEqual(rec["defaults"]["song_voice"], "child")
        rec = channels._validate({"name": "G", "audience": "general", "video_type": "animated"})
        self.assertEqual(rec["defaults"]["song_voice"], "auto")
        rec = channels._validate({"name": "G", "audience": "general", "video_type": "animated",
                                  "defaults": {"song_voice": "girl", "audio_mode": "narration"}})
        self.assertEqual(rec["defaults"]["song_voice"], "girl")   # kept even when the default audio is a voiceover
        with self.assertRaises(ValueError):
            channels._validate({"name": "G", "audience": "general", "video_type": "animated",
                                "defaults": {"song_voice": "robot"}})

    def test_job_choice_resolution(self):
        import pipeline
        r = pipeline.JobManager._resolve_choices("", "", "", "", "", {"audience": "kids", "defaults": {}})
        self.assertEqual(r[5], "child")
        r = pipeline.JobManager._resolve_choices("", "", "", "", "", {"audience": "kids", "defaults": {"song_voice": "boy"}})
        self.assertEqual(r[5], "boy")
        r = pipeline.JobManager._resolve_choices("", "", "", "", "", None, song_voice="female")
        self.assertEqual(r[5], "female")
        with self.assertRaises(ValueError):
            pipeline.JobManager._resolve_choices("", "", "", "", "", None, song_voice="robot")


USER_LYRICS = "[Verse]\nTwinkle twinkle little star\nHow I wonder what you are\nUp above the world so high"


class GivenLyrics(unittest.TestCase):
    """The user's own song sheet: sung verbatim, and it - not the preset - sets the length."""

    def test_same_words_ignores_tags_case_and_punctuation(self):
        self.assertTrue(lyrics.same_words(USER_LYRICS,
                                          "[Chorus]\nTWINKLE, twinkle, little star!\n(oh oh)\nhow i wonder "
                                          "what you are\nup above the world so high"))
        self.assertFalse(lyrics.same_words(USER_LYRICS, USER_LYRICS + "\nand one more line here"))
        self.assertFalse(lyrics.same_words(USER_LYRICS, USER_LYRICS.replace("little star", "star little")))
        self.assertFalse(lyrics.same_words(USER_LYRICS, USER_LYRICS.replace("Twinkle twinkle", "Twinkle")))
        self.assertFalse(lyrics.same_words(USER_LYRICS, ""))

    def test_ensure_sections_only_tags_an_untagged_sheet(self):
        self.assertEqual(lyrics.ensure_sections("a b\nc d"), "[Verse]\na b\nc d")
        self.assertEqual(lyrics.ensure_sections(USER_LYRICS), USER_LYRICS)
        self.assertEqual(lyrics.ensure_sections("  [Intro]\nla  "), "[Intro]\nla")

    def test_clip_seconds_follow_the_measured_song(self):
        with mock.patch.object(config, "CLIP_SECONDS", 5):
            self.assertEqual(config.clip_seconds_for_duration(20), 5)      # under every band: the global default
            self.assertEqual(config.clip_seconds_for_duration(45), 5)
            self.assertEqual(config.clip_seconds_for_duration(150), 6)
            self.assertEqual(config.clip_seconds_for_duration(200), 8)
            self.assertEqual(config.clip_seconds_for_duration(278), 10)
            self.assertEqual(config.clip_seconds_for_duration(400), 10)    # past the ladder: the longest rung
            # the ~25-30 clip invariant holds across the whole legal lyric range
            for secs in (150, 200, 278):
                n = len(config.clip_durations(secs, config.clip_seconds_for_duration(secs)))
                self.assertTrue(22 <= n <= 30, f"{secs}s gave {n} clips")

    def _job(self, lyric_text=USER_LYRICS, audience="kids"):
        import pipeline
        return pipeline.Job(id="t", url="", channel=_channel(audience), lyrics=lyric_text)

    def test_length_comes_from_the_lyrics_and_has_no_range(self):
        import pipeline
        with mock.patch.object(config, "KIDS_SONG_WPS", 0.9):
            words, target, ceiling, rng = pipeline._length_params(self._job())
            self.assertIsNone(rng)                      # no range: the "too short" rewrite can never fire
            self.assertAlmostEqual(target, 16 / 0.9, places=3)
            self.assertAlmostEqual(ceiling, target * 2 + 20, places=3)
            self.assertEqual(words, int(target * 2.5))
            self.assertIsNone(pipeline._length_params(self._job(""))[3])     # no lyrics, Auto job
            self.assertEqual(pipeline._length_params(self._job("la " * 400))[2], config.MAX_SONG_SECONDS)
        self.assertTrue(pipeline._truncated({"duration": 99.5}, 100.0))
        self.assertFalse(pipeline._truncated({"duration": 60.0}, 100.0))

    def test_clip_seconds_helper_prefers_the_measured_duration(self):
        import pipeline
        self.assertEqual(pipeline._clip_seconds(self._job(), 200.0), 8)
        self.assertEqual(pipeline._clip_seconds(self._job()), config.video_length_clip_seconds("Auto"))
        plain = pipeline.Job(id="t", url="", channel=_channel(), video_length="4-5 minutes")
        self.assertEqual(pipeline._clip_seconds(plain, 45.0), 10)   # no lyrics: the preset still wins

    def test_resolve_lyrics_rejects_rather_than_trims(self):
        import pipeline
        kids, gen = _channel(), _channel("general")
        self.assertEqual(pipeline._resolve_lyrics("  " + USER_LYRICS + " ", kids, "music"), USER_LYRICS)
        self.assertEqual(pipeline._resolve_lyrics("   ", gen, "narration+music"), "")
        for text, ch, mode, needle in [
                (USER_LYRICS, gen, "music", "kids channels"),
                (USER_LYRICS, kids, "narration+music", "Generate Song"),
                ("la la la", kids, "music", "too short"),
                ("la " * 400, kids, "music", "250"),
                ("la " * 1200, kids, "music", "characters")]:
            with self.assertRaises(ValueError) as cm:
                pipeline._resolve_lyrics(text, ch, mode)
            self.assertIn(needle, str(cm.exception))

    def test_resolve_choices_signature_is_untouched(self):
        import pipeline
        r = pipeline.JobManager._resolve_choices("", "", "", "", "", {"audience": "kids", "defaults": {}})
        self.assertEqual(len(r), 11)
        self.assertEqual(r[5], "child")

    def _music(self, reply, **kw):
        calls = []

        def fake(messages, max_tokens=8000):
            calls.append(messages[-1]["content"])
            return dict(reply)
        with mock.patch.object(llm, "_chat", fake):
            out = llm.write_music("T", "story", "", standalone=True, seconds=18, max_seconds=56,
                                  channel=_channel(), given_lyrics=USER_LYRICS, **kw)
        return out, calls

    def test_the_prompt_asks_for_the_words_back_unchanged(self):
        tagged = ("[Verse]\nTwinkle, twinkle, little star!\n[Chorus]\nHow I wonder what you are\n"
                  "Up above the world so high")
        out, calls = self._music({"caption": "Global Metadata: Pop. 120 BPM.", "lyrics": tagged})
        p = calls[0]
        self.assertIn("VERBATIM", p)
        self.assertIn("The lyrics are FIXED: 16 sung words", p)
        self.assertIn("return these words unchanged", p)
        self.assertNotIn("words of lyrics in total", p)          # no word budget for words already fixed
        self.assertNotIn("short simple lines of 4-6 words", p)   # that would be writing the lines for them
        self.assertNotIn("TELL THE VIDEO", p)
        self.assertEqual(out["lyrics_kept"], "verbatim")
        self.assertEqual(out["lyrics"], tagged)                  # tags and punctuation may be added
        self.assertIn("66 BPM", out["caption"])                  # the kids rules still apply
        self.assertIn("young child", out["caption"])

    def test_an_llm_that_edits_the_words_loses_its_version(self):
        out, _ = self._music({"caption": "Global Metadata: Pop.", "lyrics": "Some entirely different words here"})
        self.assertEqual(out["lyrics_kept"], "verbatim-fallback")
        self.assertEqual(out["lyrics"], USER_LYRICS)
        out, _ = self._music({"caption": "Global Metadata: Pop.", "lyrics": ""})
        self.assertEqual(out["lyrics"], USER_LYRICS)             # an empty reply falls back too

    def test_a_fast_song_is_slowed_by_the_music_not_by_cutting_words(self):
        out, calls = self._music({"caption": "Global Metadata: Pop.", "lyrics": USER_LYRICS},
                                 slow_down={"measured": 1.6, "budget": 0.9, "previous_lyrics": USER_LYRICS})
        p = calls[0]
        self.assertIn("TOO FAST", p)
        self.assertIn("WORDS CANNOT CHANGE", p)
        self.assertNotIn("FEWER words", p)
        self.assertEqual(out["lyrics"], USER_LYRICS)

    def test_lengthen_is_never_asked_for_when_the_words_are_fixed(self):
        _out, calls = self._music({"caption": "Global Metadata: Pop.", "lyrics": USER_LYRICS},
                                  lengthen={"measured": 20.0, "low": 60, "target": 70,
                                            "previous_lyrics": USER_LYRICS})
        self.assertNotIn("TOO SHORT", calls[0])


if __name__ == "__main__":
    unittest.main()
