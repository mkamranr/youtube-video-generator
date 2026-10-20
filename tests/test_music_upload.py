"""Uploading your own music instead of generating a song, and the video taking that track's length.

Run with  python -m unittest discover -s tests -t .  from the repo root. `config.OUTPUT_DIR` is
patched to a throw-away directory before pipeline is imported (the staging area lives under it),
and `_start_draft` is mocked so no LLM thread is ever launched.
"""
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-musicup-"))
OUT = TMP / "output"
config.OUTPUT_DIR = OUT

import pipeline  # noqa: E402  (config patched first)
from services import llm, lyrics, music_upload  # noqa: E402
from services.ffbin import ffmpeg  # noqa: E402

CHANNEL = {"id": "ch01", "name": "Bright Minds Town", "purpose": "", "rules": "", "owner": "u1",
           "audience": "kids", "video_type": "animated", "cast": [], "cast_opening": True,
           "defaults": {"art_style": "", "audio_mode": "music", "voice_id": "",
                        "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
                        "video_length": "1-2 minutes", "storyboard": False,
                        "song_voice": "child"}}

_TONES = {}


def tone(seconds: int = 12) -> bytes:
    """A real decodable mp3, so the ffmpeg validation in stage() is exercised for real."""
    if seconds not in _TONES:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "tone.mp3"
            subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                            "-i", f"sine=frequency=440:duration={seconds}", str(f)], check=True)
            _TONES[seconds] = f.read_bytes()
    return _TONES[seconds]


class Staging(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)

    def test_a_real_track_is_normalized_staged_and_described(self):
        rec = music_upload.stage(tone(12), "u1", "My Song.mp3")
        self.assertRegex(rec["id"], r"^[0-9a-f]{8}$")
        self.assertEqual(rec["owner"], "u1")
        self.assertEqual(rec["name"], "My Song.mp3")
        self.assertAlmostEqual(rec["seconds"], 12.0, delta=0.5)
        self.assertTrue(music_upload.path(rec["id"]).exists())
        self.assertEqual(music_upload.get(rec["id"])["id"], rec["id"])

    def test_the_clients_filename_never_becomes_a_path(self):
        rec = music_upload.stage(tone(12), "u1", "../../evil.mp3")
        self.assertNotIn("/", rec["name"])
        self.assertNotIn("..", rec["name"])
        # the file on disk is named from the generated id, not from anything the client sent
        self.assertEqual(music_upload.path(rec["id"]).name, f"{rec['id']}.mp3")

    def test_anything_ffmpeg_cannot_decode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a valid audio file"):
            music_upload.stage(b"this is not audio at all", "u1", "track.mp3")

    def test_an_oversized_file_is_rejected_before_it_is_decoded(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            music_upload.stage(b"x" * (music_upload.MAX_BYTES + 1), "u1", "big.wav")

    def test_a_track_below_the_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            music_upload.stage(tone(3), "u1", "stub.mp3")

    def test_a_track_over_the_ceiling_is_rejected_not_trimmed(self):
        with mock.patch.object(music_upload, "audio_duration", return_value=400.0):
            with self.assertRaisesRegex(ValueError, "too long"):
                music_upload.stage(tone(12), "u1", "epic.mp3")

    def test_install_copies_the_track_and_consumes_the_upload(self):
        rec = music_upload.stage(tone(12), "u1", "My Song.mp3")
        size = music_upload.path(rec["id"]).stat().st_size
        dest = OUT / "job01" / "music.mp3"
        music_upload.install(rec["id"], dest)
        self.assertEqual(dest.stat().st_size, size)
        self.assertIsNone(music_upload.get(rec["id"]))       # single use
        with self.assertRaises(ValueError):
            music_upload.install(rec["id"], OUT / "job02" / "music.mp3")

    def test_sweep_drops_expired_uploads_and_leaves_job_folders_alone(self):
        fresh = music_upload.stage(tone(12), "u1", "fresh.mp3")
        stale = music_upload.stage(tone(12), "u1", "stale.mp3")
        side = music_upload._sidecar(stale["id"])
        side.write_text(side.read_text(encoding="utf-8").replace(
            str(stale["created"]), "1.0"), encoding="utf-8")
        job_dir = OUT / "20250101-000000-abcdef"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text("{}", encoding="utf-8")
        music_upload.sweep()
        self.assertIsNone(music_upload.get(stale["id"]))
        self.assertIsNotNone(music_upload.get(fresh["id"]))
        self.assertTrue((job_dir / "job.json").exists())


class Validation(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        self.rec = music_upload.stage(tone(12), "u1", "My Song.mp3")

    def test_no_upload_resolves_to_nothing(self):
        self.assertIsNone(pipeline._resolve_music_upload("", "music", ""))
        self.assertIsNone(pipeline._resolve_music_upload("   ", "narration", "la la la"))

    def test_a_track_needs_the_generate_song_mode(self):
        for mode in ("narration", "narration+music"):
            with self.assertRaisesRegex(ValueError, "Generate Song"):
                pipeline._resolve_music_upload(self.rec["id"], mode, "")

    def test_a_track_cannot_be_combined_with_lyrics(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            pipeline._resolve_music_upload(self.rec["id"], "music", "[Verse]\nTwinkle twinkle")

    def test_an_unknown_or_expired_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no longer available"):
            pipeline._resolve_music_upload("deadbeef", "music", "")
        # hiding the control in the UI is not enough: a malformed id is refused, not probed
        with self.assertRaisesRegex(ValueError, "no longer available"):
            pipeline._resolve_music_upload("../../etc/passwd", "music", "")

    def test_a_valid_upload_resolves_to_its_record(self):
        self.assertEqual(pipeline._resolve_music_upload(self.rec["id"], "music", "")["id"],
                         self.rec["id"])


class LengthFollowsTheTrack(unittest.TestCase):
    """The point of the feature: the measured track, not a preset, decides the video length."""

    @staticmethod
    def _job(**kw):
        return pipeline.Job(id="j1", url="", audio_mode="music", **kw)

    def test_an_uploaded_track_picks_the_clip_rung_from_its_own_length(self):
        job = self._job(music_upload="song.mp3", music_seconds=200.0, video_length="Auto")
        # pinned under the rung it is testing: the Settings clip length is a floor over it
        with mock.patch.object(config, "CLIP_SECONDS", 5):
            self.assertEqual(pipeline._clip_seconds(job, 200.0),
                             config.clip_seconds_for_duration(200.0))
            # and without the duration argument it still uses the stored length
            self.assertEqual(pipeline._clip_seconds(job), config.clip_seconds_for_duration(200.0))

    def test_a_preset_job_is_unaffected(self):
        job = self._job(video_length="1-2 minutes")
        with mock.patch.object(config, "CLIP_SECONDS", 5):
            self.assertEqual(pipeline._clip_seconds(job, 200.0),
                             config.video_length_clip_seconds("1-2 minutes"))

    def test_a_lyrics_job_without_a_duration_still_falls_back_to_the_preset_rung(self):
        job = self._job(lyrics="[Verse]\nTwinkle twinkle little star", video_length="Auto")
        self.assertEqual(pipeline._clip_seconds(job), config.video_length_clip_seconds("Auto"))

    def test_length_params_are_taken_from_the_track(self):
        job = self._job(music_upload="song.mp3", music_seconds=150.0, video_length="Auto")
        words, target, ceiling, rng = pipeline._length_params(job)
        self.assertEqual((target, ceiling), (150.0, 150.0))
        self.assertIsNone(rng)          # no range: nothing is ever rewritten to fit the track
        self.assertEqual(words, int(150.0 * 2.5))

    def test_the_clips_always_cover_the_whole_track(self):
        for seconds in (12.0, 30.0, 95.0, 187.4, 360.0):
            durs = config.clip_durations(seconds, config.clip_seconds_for_duration(seconds))
            self.assertGreaterEqual(sum(durs) + 1e-6, seconds,
                                    f"{seconds}s track would end before its music")
            self.assertLessEqual(len(durs), 40, f"{seconds}s track renders too many clips")

    def test_the_longest_allowed_track_stays_within_the_usual_clip_budget(self):
        durs = config.clip_durations(music_upload.MAX_SECONDS,
                                     config.clip_seconds_for_duration(music_upload.MAX_SECONDS))
        self.assertEqual(len(durs), math.ceil(music_upload.MAX_SECONDS
                                              / config.actual_clip_seconds(10)))


class Submission(unittest.TestCase):
    def setUp(self):
        prev = config.OUTPUT_DIR
        config.OUTPUT_DIR = OUT
        self.addCleanup(setattr, config, "OUTPUT_DIR", prev)
        shutil.rmtree(OUT, ignore_errors=True)
        patcher = mock.patch.object(pipeline.JobManager, "_start_draft")
        self.draft = patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = pipeline.JobManager()
        self.rec = music_upload.stage(tone(12), "u1", "My Song.mp3")

    def test_submit_installs_the_track_and_drops_the_length_preset(self):
        job = self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1",
                              audio_mode="music", video_length="2-3 minutes",
                              music_upload_id=self.rec["id"])
        self.assertEqual(job.music_upload, "My Song.mp3")
        self.assertAlmostEqual(job.music_seconds, 12.0, delta=0.5)
        self.assertEqual(job.video_length, "Auto")   # the track sets the length
        self.assertTrue((job.dir / "music.mp3").exists())
        self.assertIsNone(music_upload.get(self.rec["id"]))   # consumed
        # and it survives a reload from disk
        again = pipeline.Job.load(job.dir / "job.json")
        self.assertEqual(again.music_upload, "My Song.mp3")
        self.assertAlmostEqual(again.music_seconds, job.music_seconds, places=3)

    def test_submit_prompt_installs_the_track_too(self):
        job = self.mgr.submit_prompt("A kite in the meadow.", "description", channel=CHANNEL,
                                     owner="u1", audio_mode="music",
                                     music_upload_id=self.rec["id"])
        self.assertEqual(job.music_upload, "My Song.mp3")
        self.assertTrue((job.dir / "music.mp3").exists())

    def test_the_same_upload_cannot_be_used_twice(self):
        self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1", audio_mode="music",
                        music_upload_id=self.rec["id"])
        with self.assertRaises(ValueError):
            self.mgr.submit("https://youtu.be/y", channel=CHANNEL, owner="u1", audio_mode="music",
                            music_upload_id=self.rec["id"])

    def test_a_rejected_upload_creates_no_job(self):
        before = set(self.mgr.jobs)
        with self.assertRaises(ValueError):
            self.mgr.submit("https://youtu.be/x", channel=CHANNEL, owner="u1",
                            audio_mode="narration+music", music_upload_id=self.rec["id"])
        self.assertEqual(set(self.mgr.jobs), before)
        self.assertEqual(self.mgr.queue, [])

    def test_setup_planned_installs_the_track_and_nothing_is_mutated_when_it_is_refused(self):
        job = self.mgr.submit_prompt("A kite in the meadow.", "description", channel=CHANNEL,
                                     owner="u1", start=False, plan={"plan_id": "p1"})
        self.assertEqual(job.status, "planned")
        with self.assertRaises(ValueError):    # a voiceover cannot take an uploaded song
            self.mgr.setup_planned(job.id, audio_mode="narration+music",
                                   music_upload_id=self.rec["id"])
        self.assertEqual(job.status, "planned")
        self.assertEqual(job.music_upload, "")
        self.mgr.setup_planned(job.id, audio_mode="music", music_upload_id=self.rec["id"])
        self.assertEqual(job.music_upload, "My Song.mp3")
        self.assertEqual(job.video_length, "Auto")
        self.assertTrue((job.dir / "music.mp3").exists())


class ScenesForAnUploadedTrack(unittest.TestCase):
    """The track has no written lyrics, so whisper alone drives the per-clip text - and when it
    hears nothing, the story has to drive the scenes instead."""

    def test_timing_a_track_without_lyrics_keeps_the_raw_whisper_words(self):
        heard = [{"start": 0.0, "end": 0.4, "word": "Hello"},
                 {"start": 0.5, "end": 1.0, "word": "world"}]
        words, ratio = lyrics.align_to_lyrics(heard, "")
        self.assertEqual([w["text"] for w in words], ["Hello", "world"])
        self.assertEqual(ratio, 0.0)          # nothing to match against; nothing is invented

    def test_an_instrumental_track_yields_instrumental_clip_windows(self):
        timed = lyrics.timeline([], config.boundaries([5.0, 5.0]))
        self.assertEqual([t["text"] for t in timed], ["", ""])
        self.assertIn("instrumental", lyrics.format_timeline(timed[0]))

    def test_empty_segments_would_strand_the_story_so_None_is_passed_instead(self):
        """The reason the instrumental branch passes segments=None: with all-empty segments every
        clip takes the "no words are heard" prompt and the story never reaches a single scene."""
        calls = []

        def capture(messages, max_tokens=8000):
            calls.append(messages[-1]["content"])
            return {"scene": "s. No text, subtitles, logos or watermarks of any kind."}

        story = "Finn flies a kite over the meadow"
        with mock.patch.object(llm, "_chat", capture), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes(story, "style", 2, kind="instrumental music track", segments=["", ""])
        prompts = " ".join(calls)
        self.assertIn("INSTRUMENTAL - no words are heard", prompts)
        for word in ("Finn", "kite", "meadow"):
            self.assertNotIn(word, prompts, "empty segments strand the story")

        calls.clear()
        with mock.patch.object(llm, "_chat", capture), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes(story, "style", 2, kind="instrumental music track", segments=None)
        prompts = " ".join(calls)   # split_narration spreads the story over the clips
        for word in ("Finn", "kite", "meadow"):
            self.assertIn(word, prompts, "the story must reach the scene prompts")

    def test_the_instrumental_threshold_tolerates_a_couple_of_hallucinated_words(self):
        # whisper invents a word or two over music; `not words` would be the wrong test
        self.assertGreater(config.MIN_HEARD_WORDS, 2)


if __name__ == "__main__":
    unittest.main()
