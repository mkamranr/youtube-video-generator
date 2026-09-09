"""Cast selection + prompt assembly: run with  python -m unittest discover -s tests -t .  from the repo root."""
import tempfile
import unittest
from pathlib import Path

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

    def test_prefix_restricted_to_used(self):
        scene = "Finn walks past a daisy while Pip sings."
        full = cast.render_prefix(self.ch, scene)
        self.assertIn("Daisy:", full)                            # case-insensitive match pulls Daisy in
        only = cast.render_prefix(self.ch, scene, only=["Finn", "Pip"])
        self.assertIn("Finn:", only)
        self.assertIn("Pip:", only)
        self.assertNotIn("Daisy:", only)
        self.assertEqual(cast.with_prefix(self.ch, "no cast here", only=["Finn"]), "no cast here")


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
        nodes = (config.NODE_PROMPT, config.NODE_SEED, config.NODE_DURATION, config.NODE_SAVE,
                 config.NODE_RESOLUTION, config.NODE_LOAD_IMAGE)
        template = {n: {"inputs": {}} for n in nodes}
        plain = ComfyClient.build_thumbnail(template, "p", "videogen/x/thumb", aspect_ratio=config.DEFAULT_ASPECT_RATIO)
        self.assertEqual(plain[config.NODE_DURATION]["inputs"]["value"], 0.1)
        self.assertNotIn("image", plain[config.NODE_LOAD_IMAGE]["inputs"])
        self.assertEqual(plain[config.NODE_RESOLUTION]["inputs"]["megapixels"], 1.0)
        seeded = ComfyClient.build_thumbnail(template, "p", "videogen/x/thumb", aspect_ratio=config.DEFAULT_ASPECT_RATIO,
                                             first_frame="videogen_x_cast.jpg", seconds=config.THUMB_SEED_SECONDS)
        self.assertEqual(seeded[config.NODE_DURATION]["inputs"]["value"], config.THUMB_SEED_SECONDS)
        self.assertEqual(seeded[config.NODE_LOAD_IMAGE]["inputs"]["image"], "videogen_x_cast.jpg")
        self.assertEqual(seeded[config.NODE_PROMPT]["inputs"]["prompt"], "p")


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
