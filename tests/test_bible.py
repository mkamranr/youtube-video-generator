"""Character-bible parser: run with  python -m unittest discover -s tests -t .  from the repo root."""
import unittest
from pathlib import Path

from services import bible

FIXTURE = Path(__file__).parent / "fixtures" / "bright_minds_town.md"


class SampleBible(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = bible.parse_bible(FIXTURE.read_text(encoding="utf-8"))
        cls.by_name = {c["name"]: c for c in cls.out["cast"]}

    def test_cast_count_roles_and_order(self):
        cast = self.out["cast"]
        self.assertEqual(len(cast), 16)
        self.assertEqual(sum(1 for c in cast if c["role"] == "main"), 9)
        self.assertEqual(cast[0]["name"], "Finn")
        self.assertEqual(cast[-1]["name"], "Woolly")
        self.assertEqual(self.out["warnings"], [])

    def test_fields(self):
        finn = self.by_name["Finn"]
        self.assertEqual(finn["kind"], "5-year-old anthropomorphic red fox")
        self.assertEqual(finn["role"], "main")
        self.assertTrue(finn["personality"].startswith("Curious, brave"))
        self.assertEqual(self.by_name["Pip"]["role"], "supporting")

    def test_look_keeps_labels_in_order(self):
        milo = self.by_name["Milo"]["look"]
        self.assertIn("Precise look: Slightly taller, slim gray rabbit", milo)
        self.assertIn("Clothes: Forest-green vest, cream-colored shirt, orange scarf.", milo)
        self.assertIn("Accessories: Tiny brown satchel across his body.", milo)
        self.assertIn("Distinguishing marks: Small notch near the tip of his left ear.", milo)
        self.assertLess(milo.index("Clothes:"), milo.index("Accessories:"))
        self.assertLess(milo.index("Accessories:"), milo.index("Distinguishing marks:"))
        self.assertIn("Details: Red chimney with yellow cap", self.by_name["Tilly"]["look"])
        for c in self.out["cast"]:
            self.assertNotIn("**", c["look"] + c["kind"] + c["personality"])
            self.assertNotIn("\n", c["look"])

    def test_rules_sections(self):
        rules = self.out["rules"]
        self.assertEqual(self.out["sections"], ["Global visual style", "Permanent character rules",
                                                "Episode generation rules", "Final consistency check"])
        for must in ("Global visual style:", "High-quality 3D animated cartoon style",
                     "1. Never redesign an established character.", "Final consistency check:"):
            self.assertIn(must, rules)
        for never in ("Scene prompt format", "Video prompt", "Precise look", "### ", "**",
                      "character-consistency director", "Established character cast"):
            self.assertNotIn(never, rules)
        self.assertLessEqual(len(rules), 6000)


class Robustness(unittest.TestCase):
    def test_plain_bullets_and_parenthetical_kind(self):
        md = "## Cast\n### Zed (small round robot)\n- Look: chrome body with a blue visor\n- Role: supporting\n"
        out = bible.parse_bible(md)
        self.assertEqual(len(out["cast"]), 1)
        z = out["cast"][0]
        self.assertEqual((z["name"], z["kind"], z["role"]), ("Zed", "small round robot", "supporting"))
        self.assertEqual(z["look"], "Precise look: chrome body with a blue visor.")
        self.assertEqual(out["rules"], "")

    def test_missing_look_is_skipped_with_warning(self):
        out = bible.parse_bible("## Cast\n### Ghost\n- Personality: shy\n### Ok\n- Appearance: a blue cat\n")
        self.assertEqual([c["name"] for c in out["cast"]], ["Ok"])
        self.assertTrue(any("Ghost" in w for w in out["warnings"]))

    def test_kind_is_truncated_at_word_boundary(self):
        long_kind = "a very " * 20 + "long kind"
        out = bible.parse_bible(f"### X\n- Kind: {long_kind}\n- Look: something\n", max_kind_chars=30)
        self.assertLessEqual(len(out["cast"][0]["kind"]), 30)
        self.assertFalse(out["cast"][0]["kind"].endswith(" "))
        self.assertTrue(any("kind shortened" in w for w in out["warnings"]))

    def test_no_h2_at_all(self):
        out = bible.parse_bible("### Finn\n- Kind: fox\n- Look: red fox\n### Pip\n- Look: blue bird\n")
        self.assertEqual([c["name"] for c in out["cast"]], ["Finn", "Pip"])
        self.assertEqual(out["rules"], "")

    def test_rules_only_without_headings(self):
        out = bible.parse_bible("Soft pastel colours.\n- No scary faces.\n")
        self.assertEqual(out["cast"], [])
        self.assertEqual(out["rules"], "Soft pastel colours.\n- No scary faces.")

    def test_format_section_is_dropped_and_continuation_lines_join(self):
        md = ("# T\nintro\n## Style\n- soft\n## Scene prompt format\n**Scene [n]**\n## Cast\n### A\n"
              "- Look: first line\n  continues here\n- Personality: kind\n")
        out = bible.parse_bible(md)
        self.assertEqual(out["rules"], "Style:\n- soft")
        self.assertEqual(out["cast"][0]["look"], "Precise look: first line continues here.")

    def test_duplicates_and_empty(self):
        out = bible.parse_bible("### A\n- Look: one\n### a\n- Look: two\n")
        self.assertEqual(len(out["cast"]), 1)
        self.assertEqual(out["cast"][0]["look"], "Precise look: two.")
        self.assertTrue(out["warnings"])
        self.assertEqual(bible.parse_bible(""), {"rules": "", "cast": [], "sections": [], "warnings": []})


if __name__ == "__main__":
    unittest.main()
