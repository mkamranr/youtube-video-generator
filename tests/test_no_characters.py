"""Channels whose videos have no characters (characters_enabled = False).

Off means informative explainer: no protagonist, no recurring cast, only generic unnamed people on
screen. The cast and its portraits are kept, so the switch is reversible.

Run with  python -m unittest discover -s tests -t .  from the repo root. The stores are pointed at
throw-away directories, so nothing touches the real channels/ or plans/."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-nochar-"))
config.CHANNELS_DIR = TMP / "channels"
config.CHANNELS_INDEX = config.CHANNELS_DIR / "channels.json"
config.PLANS_DIR = TMP / "plans"
config.PLANS_INDEX = config.PLANS_DIR / "plans.json"

from services import bible, cast, channels, llm, planner  # noqa: E402  (config patched first)

FIXTURE = Path(__file__).parent / "fixtures" / "bright_minds_town.md"


def _cast():
    parsed = bible.parse_bible(FIXTURE.read_text(encoding="utf-8"))
    for i, c in enumerate(parsed["cast"]):
        c["id"] = f"{i:08x}"
    return parsed["cast"], parsed["rules"]


def _channel(characters: bool):
    roster, rules = _cast()
    return {"id": "ch01", "name": "Daily Tech Talks", "purpose": "AI skills for non-technical people",
            "owner": "u1", "rules": rules, "audience": "general", "video_type": "tech",
            "cast": roster, "cast_opening": True, "characters_enabled": characters}


class Flag(unittest.TestCase):
    """cast.enabled() is the single fallback: a record written before the flag existed keeps
    behaving exactly as it did."""

    def test_explicit_value_wins(self):
        roster, _ = _cast()
        self.assertFalse(cast.enabled({"cast": roster, "characters_enabled": False}))
        self.assertTrue(cast.enabled({"cast": [], "characters_enabled": True}))

    def test_legacy_record_falls_back_to_having_a_cast(self):
        roster, _ = _cast()
        self.assertTrue(cast.enabled({"cast": roster}))      # pre-flag kids channel keeps its cast
        self.assertFalse(cast.enabled({"cast": []}))
        self.assertFalse(cast.enabled(None))

    def test_active_hides_the_roster_but_all_characters_keeps_it(self):
        ch = _channel(False)
        self.assertEqual(len(cast.all_characters(ch)), len(ch["cast"]))   # editor still sees them
        self.assertEqual(cast.active(ch), [])                             # prompts do not


class GatedLookups(unittest.TestCase):
    def setUp(self):
        self.off = _channel(False)
        self.on = _channel(True)

    def test_every_prompt_side_lookup_is_empty(self):
        self.assertEqual(cast.sheet_text(self.off), "")
        self.assertEqual(cast.sheet_text(self.off, spoken=True), "")
        self.assertEqual(cast.main_characters(self.off), [])
        self.assertEqual(cast.used_characters(self.off, "Finn walks to the meadow"), [])
        self.assertEqual(cast.mentioned(self.off, "Finn walks"), [])
        self.assertEqual(cast.select_characters(self.off, ["Finn"]), [])
        self.assertEqual(cast.looks_prefix(self.off, ["Finn"]), "")
        self.assertEqual(cast.with_prefix(self.off, "Finn walks"), "Finn walks")

    def test_the_same_lookups_still_work_when_on(self):
        self.assertIn("Finn", cast.sheet_text(self.on))
        self.assertEqual(cast.names(cast.used_characters(self.on, "Finn walks")), ["Finn"])
        self.assertNotEqual(cast.looks_prefix(self.on, ["Finn"]), "")

    def test_no_opening_portrait(self):
        char_id = self.off["cast"][0]["id"]
        config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
        channels.portrait_path("ch01", char_id).write_bytes(b"not really a png")
        try:
            self.assertIsNone(cast.opening_character(self.off, "ch01"))
            self.assertIsNotNone(cast.opening_character(self.on, "ch01"))
        finally:
            channels.portrait_path("ch01", char_id).unlink(missing_ok=True)


class Store(unittest.TestCase):
    def test_backfill_derives_from_the_cast(self):
        roster, _ = _cast()
        config.CHANNELS_DIR.mkdir(parents=True, exist_ok=True)
        config.CHANNELS_INDEX.write_text(
            '{"channels": [{"id": "a", "name": "Story", "audience": "kids", "video_type": "animated",'
            ' "purpose": "", "cast": ' + repr(roster).replace("'", '"') + '},'
            ' {"id": "b", "name": "Explainer", "audience": "general", "video_type": "tech",'
            ' "purpose": "", "cast": []}]}', encoding="utf-8")
        by_id = {c["id"]: c for c in channels.list_channels()}
        self.assertTrue(by_id["a"]["characters_enabled"])    # had a cast -> keeps its characters
        self.assertFalse(by_id["b"]["characters_enabled"])   # no cast -> informative explainers
        config.CHANNELS_INDEX.unlink()

    def test_new_channel_derives_from_the_cast_it_was_given(self):
        roster, _ = _cast()
        self.assertTrue(channels._validate({"name": "X", "cast": roster})["characters_enabled"])
        self.assertFalse(channels._validate({"name": "X"})["characters_enabled"])
        self.assertTrue(channels._validate({"name": "X", "characters_enabled": True})["characters_enabled"])

    def test_partial_update_preserves_the_flag(self):
        base = channels._validate({"name": "X", "characters_enabled": True})
        self.assertTrue(channels._validate({"name": "Renamed"}, base=base)["characters_enabled"])
        self.assertFalse(channels._validate({"characters_enabled": False}, base=base)["characters_enabled"])

    def test_turning_it_off_keeps_the_cast(self):
        """The UI keeps sending the cast when the switch goes off; ids must survive, because
        update_channel deletes the portrait files of every id that disappears from the list."""
        roster, _ = _cast()
        base = channels._validate({"name": "X", "cast": roster})
        off = channels._validate({"characters_enabled": False, "cast": base["cast"]}, base=base)
        self.assertFalse(off["characters_enabled"])
        self.assertEqual([c["name"] for c in off["cast"]], [c["name"] for c in base["cast"]])
        self.assertEqual([c["id"] for c in off["cast"]], [c["id"] for c in base["cast"]])

    def test_turning_it_off_without_sending_the_cast_keeps_it_too(self):
        roster, _ = _cast()
        base = channels._validate({"name": "X", "cast": roster})
        off = channels._validate({"characters_enabled": False}, base=base)
        self.assertEqual([c["id"] for c in off["cast"]], [c["id"] for c in base["cast"]])

    def test_snapshot_carries_the_flag_and_tolerates_bare_dicts(self):
        roster, _ = _cast()
        self.assertFalse(channels.snapshot(_channel(False))["characters_enabled"])
        self.assertTrue(channels.snapshot(_channel(True))["characters_enabled"])
        bare = {"name": "Old", "purpose": "", "audience": "kids", "video_type": "animated"}
        self.assertFalse(channels.snapshot(bare)["characters_enabled"])
        self.assertTrue(channels.snapshot({**bare, "cast": roster})["characters_enabled"])


class SystemPrompt(unittest.TestCase):
    def test_explainer_framing_replaces_the_cast_sheet(self):
        s = llm._system(_channel(False))
        self.assertIn("expert explainer", s)
        self.assertIn("THIS CHANNEL HAS NO CHARACTERS", s)
        self.assertIn("second person", s)
        self.assertNotIn("RECURRING CAST", s)
        self.assertNotIn("Finn", s)

    def test_story_framing_is_untouched_when_on(self):
        s = llm._system(_channel(True))
        self.assertIn("creative director", s)
        self.assertNotIn("THIS CHANNEL HAS NO CHARACTERS", s)
        self.assertIn("Finn", s)

    def test_spoken_override_does_not_claim_a_character_sheet(self):
        s = llm._system(_channel(False), spoken=True)
        self.assertIn(llm.SPOKEN_NO_CHARACTER_OVERRIDE, s)
        self.assertNotIn(llm.SPOKEN_OVERRIDE, s)

    def test_bible_character_rules_are_cancelled(self):
        ch = _channel(False)
        self.assertIn("character", ch["rules"].lower())        # the fixture bible carries them
        s = llm._system(ch)
        self.assertIn("ignore every instruction there about characters", s)


class Prompts(unittest.TestCase):
    def _capture(self, reply):
        calls = []

        def fake(messages, max_tokens=8000):
            calls.append(messages)
            return dict(reply)
        return calls, fake

    def test_weekly_plan_never_offers_an_actors_slot(self):
        calls, fake = self._capture({"plan": [{"day": i + 1, "title": "t", "description": "d",
                                               "thumbnail_idea": "i", "hook": "h"} for i in range(7)]})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_week_plan(channel=_channel(False), days=7)
        user = calls[0][1]["content"]
        self.assertNotIn('"actors"', user)
        self.assertIn("NO characters", user)
        self.assertIn("Alex, a project manager", user)      # the anti-example is spelled out
        self.assertNotIn("characters in the lead", user)

    def test_weekly_plan_keeps_actors_when_characters_are_on(self):
        calls, fake = self._capture({"plan": [{"day": 1, "title": "t", "description": "d",
                                               "thumbnail_idea": "i", "hook": "h"}]})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_week_plan(channel=_channel(True), days=7)
        self.assertIn('"actors"', calls[0][1]["content"])

    def test_single_day_rewrite_drops_actors_too(self):
        calls, fake = self._capture({"day": {"day": 3, "title": "t", "description": "d",
                                             "thumbnail_idea": "i", "hook": "h"}})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_plan_day(channel=_channel(False), day=3)
        self.assertNotIn('"actors"', calls[0][1]["content"])

    def test_script_is_asked_for_an_explainer(self):
        calls, fake = self._capture({"title": "t", "narration": "You waste four hours a day.",
                                     "style": "s", "issues": []})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", True):
            llm.write_script_from_prompt("Drafting emails with AI", "topic", channel=_channel(False))
        draft, review = calls[0][1]["content"], calls[1][1]["content"]
        self.assertIn("INFORMATIVE video script", draft)
        self.assertIn("No protagonist, no named person, no story", draft)
        self.assertNotIn("Refer to any character by name only", draft)
        self.assertIn("invented characters", review)

    def test_pasted_transcript_is_still_left_alone(self):
        calls, fake = self._capture({"title": "t", "narration": "The user's own words.",
                                     "style": "s", "issues": []})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", True):
            llm.write_script_from_prompt("The user's own words.", "transcript", channel=_channel(False))
        self.assertNotIn("invented characters", calls[1][1]["content"])

    def test_description_is_not_asked_to_keep_characters(self):
        calls, fake = self._capture({"title": "t", "narration": "n", "style": "s", "issues": []})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_script_from_prompt("A brief from the planner.", "description",
                                         channel=_channel(False))
        self.assertIn("FAITHFUL to their content, structure and intent", calls[0][1]["content"])

    def test_scenes_allow_generic_unnamed_people_only(self):
        calls, fake = self._capture({"scene": "A scene."})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_scenes("You waste four hours a day.", "style", 1, channel=_channel(False))
        user = calls[0][1]["content"]
        self.assertIn("this video has no characters", user)
        self.assertIn("never name them", user)
        self.assertNotIn("\nCAST:", user)

    def test_thumbnail_shows_the_topic_not_a_character(self):
        calls, fake = self._capture({"thumbnail_prompt": "A prompt."})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_thumbnail("t", "n", "s", channel=_channel(False))
        user = calls[0][1]["content"]
        self.assertIn("the topic itself", user)
        self.assertNotIn("main character(s)", user)

    def test_song_lyrics_name_nobody(self):
        calls, fake = self._capture({"caption": "Global Metadata: pop. 90 BPM.", "lyrics": "[Verse] x"})
        with mock.patch.object(llm, "_chat", fake):
            llm.write_music("t", "n", "", channel=_channel(False), standalone=True)
        user = calls[0][1]["content"]
        self.assertIn("never invent a person", user)
        self.assertNotIn("name any character", user)


class PlanRows(unittest.TestCase):
    def test_actors_are_forced_empty(self):
        row = {"title": "t", "description": "d", "thumbnail_idea": "i", "hook": "h",
               "actors": ["Alex", "Sam"]}
        out = planner.validate_day(row, 1, "2026-09-14", _channel(False))
        self.assertEqual(out["actors"], [])

    def test_a_roster_channel_still_validates_against_it(self):
        row = {"title": "t", "description": "d", "thumbnail_idea": "i", "hook": "h",
               "actors": ["Finn", "Gandalf"]}
        out = planner.validate_day(row, 1, "2026-09-14", _channel(True))
        self.assertEqual(out["actors"], ["Finn"])

    def test_an_orphaned_plan_keeps_what_was_written(self):
        row = {"title": "t", "description": "d", "thumbnail_idea": "i", "hook": "h",
               "actors": ["Alex"]}
        self.assertEqual(planner.validate_day(row, 1, "2026-09-14", None)["actors"], ["Alex"])
        self.assertEqual(planner.validate_day(row, 1, "2026-09-14", {})["actors"], ["Alex"])


if __name__ == "__main__":
    unittest.main()
