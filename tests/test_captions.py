"""Burned-in captions: run with  python -m unittest discover -s tests -t .

Pure arithmetic and one text file - no whisper, no ffmpeg, no GPU.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-captions-"))
config.OUTPUT_DIR = TMP / "output"

from services import captions  # noqa: E402  (config patched first)


def words(spec) -> list:
    """[(word, start, end), ...] -> the shape lyrics.transcribe_words returns."""
    return [{"word": w, "start": float(a), "end": float(b)} for w, a, b in spec]


def evenly(text: str, step: float = 0.4, gap: float = 0.0) -> list:
    out, at = [], 0.0
    for w in text.split():
        out.append({"word": w, "start": round(at, 3), "end": round(at + step, 3)})
        at += step + gap
    return out


class Group(unittest.TestCase):

    def test_breaks_on_the_word_budget(self):
        got = captions.group(evenly("one two three four five six"), max_words=3, max_chars=99)
        self.assertEqual([p["text"] for p in got], ["one two three", "four five six"])

    def test_breaks_on_the_character_budget(self):
        got = captions.group(evenly("alpha bravo charlie delta"), max_words=9, max_chars=12)
        self.assertTrue(all(len(p["text"]) <= 12 or len(p["words"]) == 1 for p in got))

    def test_an_over_long_word_gets_its_own_group(self):
        got = captions.group(evenly("x " + "y" * 40 + " z"), max_words=3, max_chars=10)
        self.assertIn("y" * 40, [p["text"] for p in got])

    def test_breaks_on_a_real_silence(self):
        got = captions.group(words([("one", 0, .4), ("two", .4, .8), ("three", 2.5, 2.9)]),
                             max_words=9, max_chars=99)
        self.assertEqual([p["text"] for p in got], ["one two", "three"])

    def test_breaks_after_a_sentence_ends(self):
        got = captions.group(words([("done.", 0, .4), ("next", .45, .8), ("word", .85, 1.2)]),
                             max_words=9, max_chars=99)
        self.assertEqual(got[0]["text"], "done.")

    def test_a_comma_breaks_only_once_the_group_has_two_words(self):
        got = captions.group(words([("a,", 0, .3), ("b,", .3, .6), ("c", .6, .9), ("d", .9, 1.2)]),
                             max_words=9, max_chars=99)
        self.assertEqual(got[0]["text"], "a, b,")

    def test_a_very_short_group_is_stretched_but_never_over_the_next(self):
        got = captions.group(words([("hi", 0, .05), ("there", 2.0, 2.6)]), max_words=1)
        self.assertGreaterEqual(got[0]["end"] - got[0]["start"], 0.2)
        self.assertLessEqual(got[0]["end"], got[1]["start"])

    def test_a_tiny_gap_is_absorbed_so_captions_do_not_flicker(self):
        got = captions.group(words([("one", 0, 1.0), ("two", 1.1, 2.0)]), max_words=1)
        self.assertEqual(got[0]["end"], got[1]["start"])

    def test_blank_words_are_ignored(self):
        self.assertEqual(captions.group(words([("", 0, .4), ("ok", .4, .8)]))[0]["text"], "ok")

    def test_no_words_gives_no_groups(self):
        self.assertEqual(captions.group([]), [])
        self.assertEqual(captions.group(None), [])


class Clamp(unittest.TestCase):

    def test_trims_what_runs_past_the_end_of_the_video(self):
        """The last clip is quantised shorter than its audio, so this really happens."""
        got = captions.clamp(captions.group(evenly("one two three four"), max_words=1), 1.0)
        self.assertTrue(all(p["end"] <= 1.0 for p in got))

    def test_groups_entirely_past_the_end_are_dropped(self):
        got = captions.clamp(captions.group(evenly("one two three four"), max_words=1), 0.5)
        self.assertLess(len(got), 4)

    def test_word_timings_are_clamped_too(self):
        for p in captions.clamp(captions.group(evenly("one two three"), max_words=3), 0.9):
            for w in p["words"]:
                self.assertLessEqual(w["end"], 0.9)


class Spread(unittest.TestCase):

    def test_is_character_weighted_not_even(self):
        got = captions.spread_words("a extraordinary a", 0.0, 3.0)
        durs = [w["end"] - w["start"] for w in got]
        self.assertGreater(durs[1], durs[0])
        self.assertGreater(durs[1], durs[2])

    def test_is_monotonic_and_covers_the_whole_span(self):
        got = captions.spread_words("one two three four five", 2.0, 7.0)
        self.assertEqual(got[0]["start"], 2.0)
        self.assertEqual(got[-1]["end"], 7.0)
        for a, b in zip(got, got[1:]):
            self.assertLessEqual(a["start"], b["start"])

    def test_empty_text_gives_nothing(self):
        self.assertEqual(captions.spread_words("", 0, 1), [])


class FromLines(unittest.TestCase):
    """Story videos are timed by construction: every line is spoken alone on the clip grid."""

    def test_lines_land_on_their_own_clips(self):
        lines = [{"text": "First line here", "seconds": 1.5},
                 {"text": "", "seconds": 0},
                 {"text": "Third line", "seconds": 1.0}]
        got = captions.words_from_lines(lines, [5.0, 5.0, 5.0], lead=0.2)
        self.assertEqual(got[0]["start"], 0.2)
        self.assertGreaterEqual(got[-1]["start"], 10.0)
        self.assertLessEqual(got[-1]["end"], 11.3)

    def test_a_silent_line_contributes_nothing(self):
        got = captions.words_from_lines([{"text": "", "seconds": 0}], [5.0])
        self.assertEqual(got, [])


class Ass(unittest.TestCase):

    def setUp(self):
        self.dir = TMP / "ass"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.phrases = captions.group(evenly("one two three four five six"), max_words=3)

    def _write(self, size=(1920, 1080), **kw):
        p = captions.write_ass(self.phrases, size, self.dir / "c.ass", **kw)
        return p.read_text(encoding="utf-8")

    def test_times_format(self):
        self.assertEqual(captions.ass_time(0), "0:00:00.00")
        self.assertEqual(captions.ass_time(3661.23), "1:01:01.23")

    def test_colour_is_byte_reversed(self):
        self.assertEqual(captions.ass_colour("#2ba1e9"), "&H00E9A12B")

    def test_playres_matches_the_frame(self):
        t = self._write((3840, 2160))
        self.assertIn("PlayResX: 3840", t)
        self.assertIn("PlayResY: 2160", t)

    def test_one_event_per_word_state_when_highlighting(self):
        many = self._write(highlight="word").count("Dialogue:")
        one = self._write(highlight="off").count("Dialogue:")
        self.assertEqual(one, len(self.phrases))
        self.assertGreater(many, one)

    def test_highlight_changes_colour_only(self):
        """Resizing the spoken word reflows the centred line and makes it jump every word."""
        t = self._write(highlight="word")
        for tag in ("\\fs", "\\fscx", "\\fscy", "\\b1"):
            self.assertNotIn(tag, t, tag)

    def test_fades_only_on_the_edges_of_a_group(self):
        t = self._write(highlight="word")
        self.assertEqual(t.count("\\fad("), 2 * len(self.phrases))

    def test_portrait_never_goes_below_the_shorts_safe_area(self):
        t = self._write((720, 1280), bottom_margin=0.05)
        mv = int([l for l in t.splitlines() if l.startswith("Style: Cap")][0].split(",")[-2])
        self.assertGreaterEqual(mv, int(captions.SHORTS_SAFE * 1280))

    def test_braces_in_the_text_cannot_open_an_override_block(self):
        phr = captions.group(words([("{\\an8}evil", 0, .5), ("text", .5, 1.0)]), max_words=2)
        t = captions.write_ass(phr, (1920, 1080), self.dir / "e.ass",
                               highlight="off").read_text(encoding="utf-8")
        body = [l for l in t.splitlines() if l.startswith("Dialogue:")][0].split(",,", 1)[1]
        self.assertNotIn("{\\an8}", body)


class Lifts(unittest.TestCase):

    def test_a_card_raises_the_caption_above_it(self):
        got = captions.lift_spans([(2.0, 5.0, (40, 700, 900, 1000))], (1920, 1080))
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0][2], 1080 - 700)

    def test_a_card_with_no_rect_is_ignored(self):
        self.assertEqual(captions.lift_spans([(2.0, 5.0, None)], (1920, 1080)), [])

    def test_a_group_straddling_a_lift_boundary_is_split(self):
        phrases = captions.group(words([("a", 0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0)]),
                                 max_words=3)
        lifts = [(1.5, 4.0, 400)]
        t = captions.write_ass(phrases, (1920, 1080), TMP / "split.ass",
                               highlight="off", lifts=lifts).read_text(encoding="utf-8")
        ev = [l for l in t.splitlines() if l.startswith("Dialogue:")]
        self.assertGreater(len(ev), 1)
        self.assertTrue(any(",,0,0,400,," in l for l in ev))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
