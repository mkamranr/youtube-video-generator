"""Weekly plans: normalization, the store, and the prompts.

Run with  python -m unittest discover -s tests -t .  from the repo root. The store is pointed at
a throw-away directory, so nothing touches the real plans/plans.json."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-plans-"))
config.PLANS_DIR = TMP / "plans"
config.PLANS_INDEX = config.PLANS_DIR / "plans.json"

from services import bible, cast, llm, planner  # noqa: E402  (config patched first)

FIXTURE = Path(__file__).parent / "fixtures" / "bright_minds_town.md"


def _channel():
    parsed = bible.parse_bible(FIXTURE.read_text(encoding="utf-8"))
    for i, c in enumerate(parsed["cast"]):
        c["id"] = f"{i:08x}"
    return {"id": "ch01", "name": "Bright Minds Town", "purpose": "Kids stories", "owner": "u1",
            "rules": parsed["rules"], "audience": "kids", "video_type": "animated",
            "cast": parsed["cast"], "cast_opening": True}


def _row(day=1, title="A day out", actors=("Finn",)):
    return {"day": day, "actors": list(actors), "title": title,
            "description": f"{title}: Finn walks to the meadow and finds a lost kite.",
            "thumbnail_idea": "Finn holding a red kite.", "hook": "Whose kite is this?"}


class Normalize(unittest.TestCase):
    def setUp(self):
        self.ch = _channel()

    def test_dates_follow_the_week_start(self):
        rows = planner.normalize_days([_row(i + 1) for i in range(7)], "2026-09-15", self.ch)
        self.assertEqual([r["day"] for r in rows], list(range(1, 8)))
        self.assertEqual(rows[0]["date"], "2026-09-15")
        self.assertEqual(rows[6]["date"], "2026-09-21")

    def test_short_and_long_replies_become_exactly_seven_rows(self):
        short = planner.normalize_days([_row(1), _row(2)], "2026-09-15", self.ch)
        self.assertEqual(len(short), 7)
        self.assertEqual(short[6]["title"], "")          # padded: the user regenerates it
        long = planner.normalize_days([_row(i + 1) for i in range(12)], "2026-09-15", self.ch)
        self.assertEqual(len(long), 7)
        self.assertEqual(planner.normalize_days("not a list", "2026-09-15", self.ch)[0]["day"], 1)

    def test_unknown_actors_are_dropped(self):
        row = planner.validate_day({**_row(), "actors": ["Finn", "Gandalf", "pip"]}, 1,
                                   "2026-09-15", self.ch)
        self.assertEqual(row["actors"], ["Finn", "Pip"])   # matched case-insensitively, canonical case
        self.assertEqual(planner.validate_day({**_row(), "actors": "Finn and Nia"}, 1,
                                              "2026-09-15", self.ch)["actors"], ["Finn", "Nia"])

    def test_text_is_clamped_and_job_id_kept(self):
        row = planner.validate_day({**_row(), "title": "x" * 500}, 3, "2026-09-15", self.ch,
                                   base={"job_id": "20260915-000000-aaaaaa"})
        self.assertEqual(len(row["title"]), planner.MAX_TITLE_CHARS)
        self.assertEqual(row["job_id"], "20260915-000000-aaaaaa")
        self.assertEqual(row["date"], "2026-09-17")

    def test_week_start_falls_back_to_today(self):
        self.assertEqual(planner.parse_week_start("2026-09-15"), "2026-09-15")
        self.assertEqual(len(planner.parse_week_start("not a date")), 10)


class Store(unittest.TestCase):
    def setUp(self):
        self.ch = _channel()
        config.PLANS_INDEX.unlink(missing_ok=True)
        days = planner.normalize_days([_row(i + 1, f"Story {i + 1}") for i in range(7)],
                                      "2026-09-15", self.ch)
        self.plan = planner.create_plan("ch01", "u1", "2026-09-15", days, "gentle stories")

    def test_listing_is_filtered_by_owner_and_channel(self):
        planner.create_plan("ch02", "u2", "2026-09-15", [], "")
        self.assertEqual([p["id"] for p in planner.list_plans(owner="u1")], [self.plan["id"]])
        self.assertEqual(planner.list_plans(owner="u1", channel_id="ch02"), [])
        self.assertEqual(len(planner.list_plans()), 2)

    def test_set_day_keeps_the_day_and_its_job(self):
        planner.set_job(self.plan["id"], 2, "20260915-000000-bbbbbb")
        updated = planner.set_day(self.plan["id"], 2, planner.validate_day(
            _row(2, "A new idea"), 2, "2026-09-15", self.ch))
        row = updated["days"][1]
        self.assertEqual(row["title"], "A new idea")
        self.assertEqual(row["day"], 2)
        self.assertEqual(row["job_id"], "20260915-000000-bbbbbb")   # the job survives a rewrite
        with self.assertRaises(KeyError):
            planner.set_day(self.plan["id"], 99, _row(99))

    def test_previous_titles_and_delete(self):
        titles = planner.previous_titles("ch01")
        self.assertIn("Story 1", titles)
        self.assertTrue(planner.delete_plan(self.plan["id"]))
        self.assertFalse(planner.delete_plan(self.plan["id"]))
        self.assertEqual(planner.previous_titles("ch01"), [])

    def test_job_seed_carries_the_hook(self):
        # the hook is NOT folded into the text: it travels on Job.hook as its own instruction
        seed = planner.job_seed(self.plan["days"][0])
        self.assertIn("Finn walks to the meadow", seed)
        self.assertNotIn("Whose kite is this?", seed)


class Prompts(unittest.TestCase):
    """The plan text becomes the brief the narration is written from, so it must not carry looks."""

    def setUp(self):
        self.ch = _channel()

    def _capture(self, reply):
        calls = []

        def fake(messages, max_tokens=8000):
            calls.append(messages)
            return dict(reply)
        return calls, fake

    def test_week_prompt_has_no_looks(self):
        calls, fake = self._capture({"plan": [_row(i + 1) for i in range(7)]})
        with mock.patch.object(llm, "_chat", fake):
            rows = llm.write_week_plan(channel=self.ch, days=7, week_start="2026-09-15",
                                       notes="sharing", avoid=["An old title"])
        self.assertEqual(len(rows), 7)
        system, user = calls[0][0]["content"], calls[0][1]["content"]
        self.assertNotIn("Sunny-yellow T-shirt", system)
        self.assertIn(llm.SPOKEN_OVERRIDE, system)
        self.assertIn("- Piko (MAIN", system)          # names and kinds still reach the planner
        self.assertIn("An old title", user)            # do not repeat an earlier week
        self.assertIn("sharing", user)

    def test_single_day_prompt_lists_the_other_days(self):
        calls, fake = self._capture({"day": _row(3, "Another idea")})
        others = [_row(1, "First"), _row(2, "Second")]
        with mock.patch.object(llm, "_chat", fake):
            row = llm.write_plan_day(channel=self.ch, day=3, others=others, feedback="funnier")
        self.assertEqual(row["title"], "Another idea")
        user = calls[0][1]["content"]
        self.assertIn("day 1: First", user)
        self.assertIn("Rewrite DAY 3", user)
        self.assertIn("funnier", user)

    def test_empty_replies_raise(self):
        with mock.patch.object(llm, "_chat", lambda *a, **k: {"plan": []}):
            with self.assertRaises(RuntimeError):
                llm.write_week_plan(channel=self.ch)
        with mock.patch.object(llm, "_chat", lambda *a, **k: {"day": {"title": ""}}):
            with self.assertRaises(RuntimeError):
                llm.write_plan_day(channel=self.ch, day=2)


if __name__ == "__main__":
    unittest.main()
