"""Cast selection + prompt assembly: run with  python -m unittest discover -s tests -t .  from the repo root."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from services import bible, cast, channels, llm

FIXTURE = Path(__file__).parent / "fixtures" / "bright_minds_town.md"


def _channel():
    parsed = bible.parse_bible(FIXTURE.read_text(encoding="utf-8"))
    for i, c in enumerate(parsed["cast"]):
        c["id"] = f"{i:08x}"
    return {"name": "Bright Minds Town", "purpose": "Kids stories", "rules": parsed["rules"],
            "audience": "kids", "video_type": "animated", "cast": parsed["cast"], "cast_opening": True}


class Detection(unittest.TestCase):
    def setUp(self):
        self.ch = _channel()

    def test_used_characters_order_and_case(self):
        text = "Finn met Pip in the meadow and found a shiny penny while a daisy swayed. Then Piko arrived."
        self.assertEqual(cast.names(cast.used_characters(self.ch, text)), ["Finn", "Pip", "Piko"])
        self.assertEqual(cast.used_characters(self.ch, "the woolly sweater of pip"), [])
        self.assertEqual(cast.names(cast.used_characters(self.ch, "Title with Leo", "Body with Finn")),
                         ["Leo", "Finn"])

    def test_forced_names_from_request(self):
        rule = llm._cast_selection_rule(self.ch, "Use Finn as the main character and Pip as the supporting character")
        self.assertIn("The request names Finn, Pip", rule)
        self.assertIn('"characters"', rule)
        self.assertNotIn("The request names", llm._cast_selection_rule(self.ch, "How bees make honey"))
        self.assertIn("play in the visuals", llm._cast_selection_rule(self.ch, "x", verbatim=True))
        self.assertEqual(llm._cast_selection_rule({"cast": []}, "Finn"), "")

    def test_chosen_names_are_validated(self):
        self.assertEqual(llm._chosen(self.ch, ["finn", "Nobody", "Pip", "Pip"]), ["Finn", "Pip"])
        self.assertEqual(llm._chosen(self.ch, "Finn, Pip and Leo"), ["Finn", "Pip", "Leo"])
        self.assertEqual(llm._chosen(self.ch, None), [])


class SheetAndSystem(unittest.TestCase):
    def setUp(self):
        self.ch = _channel()

    def test_roster_mode(self):
        s = cast.sheet_text(self.ch)
        for c in self.ch["cast"]:
            self.assertIn(f"- {c['name']} (", s)
        self.assertIn("Pick only the characters this story needs", s)
        self.assertNotIn("EVERY video", s)

    def test_video_mode(self):
        s = cast.sheet_text(self.ch, used=["Finn", "Pip"])
        self.assertIn("CAST IN THIS VIDEO", s)
        self.assertIn("Sky-blue hoodie", s)                      # Finn's look
        self.assertNotIn("Sunny-yellow T-shirt, red shorts", s)  # Piko's look is not resent
        self.assertIn("Piko", s)                                 # ...but his name is in the do-not-add list
        self.assertLess(len(s), len(cast.sheet_text(self.ch)) / 3)
        none = cast.sheet_text(self.ch, used=[])
        self.assertIn("NONE of the channel's recurring cast", none)
        self.assertEqual(cast.sheet_text({"cast": []}, used=["Finn"]), "")

    def test_system_prompt_rules_toggle(self):
        with_rules = llm._system(self.ch, used=["Finn"])
        self.assertIn("CHANNEL STYLE & RULES", with_rules)
        self.assertIn("Never redesign an established character", with_rules)
        self.assertIn("Sky-blue hoodie", with_rules)
        without = llm._system(self.ch, used=["Finn"], rules=False)
        self.assertNotIn("CHANNEL STYLE & RULES", without)
        legacy = llm._system({"name": "Old", "audience": "kids", "video_type": "animated"})
        self.assertIn("Always answer with a single JSON object", legacy)

    def test_spoken_sheet_drops_the_looks(self):
        """The narration is heard, not seen: the writer must not get the looks at all."""
        roster = cast.sheet_text(self.ch, spoken=True)
        self.assertIn("- Piko (MAIN", roster)
        self.assertIn("Personality: Shy at first", roster)
        self.assertIn("Pick only the characters this story needs", roster)
        self.assertNotIn("Sunny-yellow T-shirt", roster)
        self.assertNotIn("black-and-white fur", roster)
        self.assertNotIn("repeat their full look", roster)
        video = cast.sheet_text(self.ch, used=["Piko"], spoken=True)
        self.assertIn("CAST IN THIS VIDEO", video)
        self.assertIn("Piko", video)
        self.assertNotIn("backpack", video)
        self.assertIn("Finn", video)                             # still in the do-not-add list
        self.assertEqual(cast.sheet_text({"cast": []}, spoken=True), "")

    def test_spoken_system_prompt_overrides_the_channel_rules(self):
        s = llm._system(self.ch, spoken=True)
        self.assertNotIn("Sunny-yellow T-shirt", s)
        self.assertNotIn("Sky-blue hoodie", s)
        self.assertIn("CHANNEL STYLE & RULES", s)                # tone and story rules still apply
        self.assertIn(llm.SPOKEN_OVERRIDE, s)
        self.assertGreater(s.index(llm.SPOKEN_OVERRIDE),         # an override comes last
                           s.index("Never redesign an established character"))
        self.assertIn("Sky-blue hoodie", llm._system(self.ch, used=["Finn"]))   # visuals unchanged

    def test_prefix_restricted_to_used(self):
        scene = "Finn walks past a daisy while Pip sings."
        full = cast.render_prefix(self.ch, scene)
        self.assertIn("Daisy:", full)                            # case-insensitive match pulls Daisy in
        only = cast.render_prefix(self.ch, scene, only=["Finn", "Pip"])
        self.assertIn("Finn:", only)
        self.assertIn("Pip:", only)
        self.assertNotIn("Daisy:", only)
        self.assertEqual(cast.with_prefix(self.ch, "no cast here", only=["Finn"]), "no cast here")


class Priority(unittest.TestCase):
    """The cast ranking: which characters a new video should be cast from.

    Orthogonal to `role` - a MAIN character the audience never watches and the channel's flagship
    are both `main` - and entered by hand, since the app's YouTube scope cannot read analytics."""

    def setUp(self):
        self.ch = _channel()
        self.by_name = {c["name"]: c for c in self.ch["cast"]}
        self.by_name["Piko"]["priority"] = "lead"
        self.by_name["Pip"]["priority"] = "resting"

    def test_an_unranked_roster_is_untouched(self):
        plain = _channel()
        self.assertEqual([c["name"] for c in cast.by_priority(plain["cast"])],
                         [c["name"] for c in plain["cast"]])
        self.assertEqual(cast.leads(plain), [])
        self.assertEqual(cast.resting(plain), [])
        # and the prompt gains not one word for a channel that never used the field
        self.assertNotIn("LEAD", cast.sheet_text(plain))
        self.assertNotIn("RESTING", cast.sheet_text(plain))

    def test_leads_sort_first_and_resting_last(self):
        order = [c["name"] for c in cast.by_priority(self.ch["cast"])]
        self.assertEqual(order[0], "Piko")
        self.assertEqual(order[-1], "Pip")
        self.assertEqual(cast.names(cast.leads(self.ch)), ["Piko"])
        self.assertEqual(cast.names(cast.resting(self.ch)), ["Pip"])

    def test_the_sort_is_stable_between_equals(self):
        rest = [c["name"] for c in self.ch["cast"] if c["name"] not in ("Piko", "Pip")]
        self.assertEqual([c["name"] for c in cast.by_priority(self.ch["cast"])][1:-1], rest)

    def test_the_roster_sheet_marks_the_ranking_and_says_what_it_means(self):
        s = cast.sheet_text(self.ch)
        self.assertIn("- Piko (MAIN, LEAD", s)
        self.assertIn(", RESTING", s)
        self.assertIn("cast one of them in the lead", s)
        self.assertIn("not to be cast at all unless the request names", s)
        self.assertLess(s.index("- Piko ("), s.index("- Pip ("))   # leads are listed first
        # normal characters carry no mark - the ranking is only worth saying where it differs
        self.assertIn("- Finn (MAIN,", s)
        self.assertNotIn("- Finn (MAIN, NORMAL", s)

    def test_the_video_sheet_carries_no_ranking(self):
        """By then the cast is chosen; repeating the ranking would only invite second-guessing."""
        s = cast.sheet_text(self.ch, used=["Piko", "Pip"])
        self.assertNotIn("LEAD", s)
        self.assertNotIn("RESTING", s)

    def test_a_resting_character_can_still_be_asked_for_by_name(self):
        self.assertEqual(cast.names(cast.select_characters(self.ch, ["Pip"])), ["Pip"])
        self.assertIn("The request names Pip", llm._cast_selection_rule(self.ch, "A story with Pip"))

    def test_resting_characters_stay_in_active_so_finished_videos_keep_their_looks(self):
        """active() also gates render_prefix and voice_id_for. Dropping a resting character there
        would strip its canonical look off a video that already used it the next time one of its
        clips was regenerated - the ranking is about what to cast next, not a retraction."""
        self.assertIn("Pip", cast.names(cast.active(self.ch)))
        self.assertIn("Pip:", cast.render_prefix(self.ch, "Pip waves from the gate", only=["Pip"]))

    def test_main_characters_lead_first_so_the_opening_portrait_does(self):
        self.assertEqual(cast.names(cast.main_characters(self.ch))[0], "Piko")

    def test_the_week_prompt_spends_the_week_on_the_leads(self):
        rules = llm._plan_rules(self.ch, 7, "2026-09-15", "", ())
        self.assertIn("Most of the week belongs to the roster's LEAD characters", rules)
        self.assertNotIn("Most of the week belongs", llm._plan_rules(_channel(), 7, "", "", ()))

    def test_the_snapshot_frozen_onto_a_job_carries_the_ranking(self):
        """Without this the whole feature is dropped at submit time: every prompt reads the cast
        off the job's channel snapshot, not off the live channel record."""
        snap = channels.snapshot({**self.ch, "id": "c1", "video_type": "animated"})
        got = {c["name"]: c["priority"] for c in snap["cast"]}
        self.assertEqual(got["Piko"], "lead")
        self.assertEqual(got["Pip"], "resting")
        self.assertEqual(got["Finn"], "normal")     # back-filled, never missing

    def test_the_store_validates_and_back_fills_it(self):
        clean = channels._validate_cast(
            [{"name": "Tilly", "look": "a red tank engine", "priority": "lead"},
             {"name": "Momo", "look": "a grey moth"}], [])
        self.assertEqual([c["priority"] for c in clean], ["lead", "normal"])
        with self.assertRaises(ValueError):
            channels._validate_cast([{"name": "Tilly", "look": "x", "priority": "favourite"}], [])

    def test_a_bible_may_carry_it_and_a_bad_value_does_not_fail_the_import(self):
        parsed = bible.parse_bible("## 1. Cast\n\n### Tilly\n- **Look:** a red tank engine\n"
                                   "- **Priority:** Lead\n\n### Momo\n- **Look:** a grey moth\n"
                                   "- **Standing:** whatever\n")
        got = {c["name"]: c["priority"] for c in parsed["cast"]}
        self.assertEqual(got, {"Tilly": "lead", "Momo": "normal"})


class Opening(unittest.TestCase):
    def test_opening_follows_cast_used(self):
        ch = _channel()
        ids = {c["name"]: c["id"] for c in ch["cast"]}
        old = config.CHANNELS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            config.CHANNELS_DIR = Path(tmp)
            try:
                for name in ("Pip", "Finn"):
                    channels.portrait_path("cid", ids[name]).write_bytes(b"png")
                self.assertEqual(cast.opening_character(ch, "cid", used=["Pip", "Finn"])["name"], "Pip")
                self.assertEqual(cast.opening_character(ch, "cid", used=["Leo", "Finn"])["name"], "Finn")
                self.assertIsNone(cast.opening_character(ch, "cid", used=[]))
                self.assertIsNone(cast.opening_character(ch, "cid", used=["Leo"]))
                self.assertEqual(cast.opening_character(ch, "cid")["name"], "Finn")   # legacy: first main
                ch["cast_opening"] = False
                self.assertIsNone(cast.opening_character(ch, "cid", used=["Pip"]))
            finally:
                config.CHANNELS_DIR = old


class Thumbnail(unittest.TestCase):
    def test_looks_prefix_is_forced_and_ordered(self):
        ch = _channel()
        p = cast.looks_prefix(ch, ["Pip", "Finn", "Nobody"])
        lines = p.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("Pip: Precise look:"))
        self.assertTrue(lines[1].startswith("Finn: Precise look:"))
        self.assertEqual(cast.looks_prefix(ch, []), "")
        self.assertEqual(cast.looks_prefix(ch, None), "")

    def test_build_thumbnail_seeds_from_a_portrait(self):
        from services.comfy import ComfyClient
        # a thumbnail always renders on the stills model (config.STILLS_VIDEO_MODEL), so its node
        # ids come from there rather than from whatever model a video picked
        stills = config.VIDEO_MODELS[config.STILLS_VIDEO_MODEL]
        t2v, i2v = stills["t2v"], stills["i2v"]
        prompt_n, prompt_f = t2v["prompt"]
        dur_n, dur_f = t2v["duration"]
        img_n = i2v["load_image"]

        def template(spec):
            """A stand-in graph holding exactly the nodes build() writes to."""
            nodes = {prompt_n, dur_n, img_n, spec["save"], spec["resolution"]}
            nodes |= {n for n, _ in spec["seeds"]}
            if spec["lightning"]:
                nodes.add(spec["lightning"])
            return {n: {"inputs": {}} for n in nodes}

        plain = ComfyClient.build_thumbnail(template(t2v), "p", "videogen/x/thumb",
                                            aspect_ratio=config.DEFAULT_ASPECT_RATIO)
        self.assertEqual(plain[dur_n]["inputs"][dur_f], 0.1)
        self.assertNotIn("image", plain[img_n]["inputs"])
        self.assertEqual(plain[t2v["resolution"]]["inputs"]["megapixels"], 1.0)
        seeded = ComfyClient.build_thumbnail(template(i2v), "p", "videogen/x/thumb",
                                             aspect_ratio=config.DEFAULT_ASPECT_RATIO,
                                             first_frame="videogen_x_cast.jpg",
                                             seconds=config.THUMB_SEED_SECONDS)
        self.assertEqual(seeded[dur_n]["inputs"][dur_f], config.THUMB_SEED_SECONDS)
        self.assertEqual(seeded[img_n]["inputs"]["image"], "videogen_x_cast.jpg")
        self.assertEqual(seeded[prompt_n]["inputs"][prompt_f], "p")


class SpokenCalls(unittest.TestCase):
    """Which calls see the looks: the ones writing pictures, never the ones writing audio."""

    def setUp(self):
        self.ch = _channel()

    def _capture(self, reply):
        calls = []

        def fake(messages, max_tokens=8000):
            calls.append(messages)
            return dict(reply)
        return calls, fake

    def test_script_draft_and_review_are_spoken(self):
        calls, fake = self._capture({"title": "t", "narration": "Piko is worried.", "style": "s",
                                     "issues": [], "characters": ["Piko"]})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", True):
            llm.write_script_from_prompt("Piko loses his way", "topic", channel=self.ch)
        self.assertEqual(len(calls), 2)                          # draft + review
        for messages in calls:
            self.assertNotIn("Sunny-yellow T-shirt", messages[0]["content"])
            self.assertIn(llm.SPOKEN_OVERRIDE, messages[0]["content"])
        self.assertIn("never describe how they look", calls[0][1]["content"])
        self.assertIn("character looks in the spoken words", calls[1][1]["content"])

    def test_pasted_transcript_keeps_the_users_words(self):
        calls, fake = self._capture({"title": "t", "narration": "Piko, a small round panda, is worried.",
                                     "style": "s", "issues": []})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", True):
            llm.write_script_from_prompt("Piko, a small round panda, is worried.", "transcript",
                                         channel=self.ch)
        self.assertNotIn("character looks in the spoken words", calls[1][1]["content"])

    def test_song_lyrics_are_spoken_and_scenes_are_not(self):
        calls, fake = self._capture({"caption": "Global Metadata: pop. 90 BPM.", "lyrics": "[Verse] Piko walks",
                                     "scene": "A scene."})
        with mock.patch.object(llm, "_chat", fake), mock.patch.object(config, "SCRIPT_REVIEW", False):
            llm.write_music("t", "Piko walks", "", standalone=True, channel=self.ch, used=["Piko"])
            llm.write_scenes("Piko walks", "style", 1, channel=self.ch, used=["Piko"])
        self.assertNotIn("Sunny-yellow T-shirt", calls[0][0]["content"])
        self.assertIn("never describe how they look", calls[0][1]["content"])
        self.assertIn("Sunny-yellow T-shirt", calls[1][0]["content"])   # the video model still gets it


class Validation(unittest.TestCase):
    def test_cast_cap_and_rules_cap(self):
        base = {"name": "X"}
        mk = lambda n: [{"name": f"C{i}", "look": "a look"} for i in range(n)]
        rec = channels._validate({**base, "cast": mk(24)})
        self.assertEqual(len(rec["cast"]), 24)
        with self.assertRaises(ValueError):
            channels._validate({**base, "cast": mk(25)})
        self.assertEqual(channels._validate({**base, "rules": "x" * 6000})["rules"], "x" * 6000)
        with self.assertRaises(ValueError):
            channels._validate({**base, "rules": "x" * 6001})

    def test_snapshot_without_rules(self):
        snap = channels.snapshot({"name": "X", "purpose": "", "audience": "kids", "video_type": "animated"})
        self.assertEqual(snap["rules"], "")
        self.assertEqual(snap["cast"], [])


if __name__ == "__main__":
    unittest.main()
