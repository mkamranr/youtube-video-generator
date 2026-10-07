"""Story board card templates: run with  python -m unittest discover -s tests -t .  from the root.

Pure Pillow and pure arithmetic - no GPU, no network, no ffmpeg. The rendering tests assert on
ctx.boxes (every rectangle a template placed) rather than on the pixels: Pillow silently clips at
the canvas edge, so a template that overflows still writes a perfectly valid PNG and only the
intended geometry gives it away.
"""
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-cards-"))
config.OUTPUT_DIR = TMP / "output"

from services import cardtemplates, storyboard  # noqa: E402  (config patched first)

SIZES = {"landscape": (1280, 720), "portrait": (720, 1280), "square": (1080, 1080)}


def card(**kw) -> dict:
    base = {"start_clip": 1, "end_clip": 1, "kicker": "KEY FACT",
            "headline": "Transformers changed everything",
            "body": "Attention replaced recurrence and training finally parallelised.",
            "image_prompt": ""}
    base.update(kw)
    return base


def cards(n: int, **kw) -> list:
    return [card(start_clip=i, end_clip=i, headline=f"Headline number {i}", **kw)
            for i in range(1, n + 1)]


class Family(unittest.TestCase):
    """The card's own content decides which family of layouts it may draw from."""

    def test_numbers_kicker_is_a_stat(self):
        self.assertEqual(cardtemplates._family(card(kicker="BY THE NUMBERS"), 2, 5), "stat")

    def test_a_figure_in_the_headline_is_a_stat_even_with_a_plain_kicker(self):
        c = card(kicker="KEY FACT", headline="Costs fell 42% in one year")
        self.assertEqual(cardtemplates._family(c, 2, 5), "stat")

    def test_takeaway_kickers_are_banners(self):
        for kick in ("TAKEAWAY", "IN SHORT", "WHY IT MATTERS"):
            self.assertEqual(cardtemplates._family(card(kicker=kick), 2, 5), "banner", kick)

    def test_the_opener_and_the_closer_are_banners(self):
        self.assertEqual(cardtemplates._family(card(), 1, 5), "banner")
        self.assertEqual(cardtemplates._family(card(), 5, 5), "banner")

    def test_an_image_outranks_every_other_signal(self):
        """Pins the documented priority order - this fails loudly if the rules are reordered."""
        c = card(kicker="BY THE NUMBERS", headline="40% faster", image_prompt="a server rack")
        self.assertEqual(cardtemplates._family(c, 5, 5), "visual")

    def test_a_blanked_image_prompt_is_never_visual(self):
        """ensure_assets blanks image_prompt when the ComfyUI render failed."""
        self.assertNotEqual(cardtemplates._family(card(image_prompt=""), 2, 5), "visual")

    def test_plain_middle_card_is_classic(self):
        c = card(kicker="SOMETHING ELSE", headline="No figures here at all", body="Nor here.")
        self.assertEqual(cardtemplates._family(c, 2, 5), "classic")


class Pick(unittest.TestCase):
    """The variant is drawn from a seeded RNG, so a job always renders identically."""

    def setUp(self):
        self._set = config.STORYBOARD_TEMPLATE_SET
        self.addCleanup(setattr, config, "STORYBOARD_TEMPLATE_SET", self._set)

    def test_same_seed_same_picks_and_no_global_random_state(self):
        cs = cards(8)
        first = cardtemplates.pick_templates(cs, "20260101-000000-abc123")
        [random.random() for _ in range(100)]          # disturb the global RNG
        self.assertEqual(cardtemplates.pick_templates(cs, "20260101-000000-abc123"), first)

    def test_different_seeds_produce_different_mixes(self):
        cs = cards(8)
        seen = {tuple(cardtemplates.pick_templates(cs, f"job-{i}")) for i in range(20)}
        self.assertGreater(len(seen), 1)

    def test_never_the_same_layout_twice_in_a_row(self):
        cs = cards(10)
        for i in range(30):
            picks = cardtemplates.pick_templates(cs, f"seed-{i}")
            self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])), picks)

    def test_every_pick_is_a_real_template(self):
        for name in cardtemplates.pick_templates(cards(12), "seed"):
            self.assertIn(name, cardtemplates.TEMPLATES)

    def test_an_edit_that_keeps_the_family_changes_nothing(self):
        """The RNG is seeded per card rather than walked across the list, so re-planning does not
        reshuffle the video."""
        a = cards(9)
        b = [dict(c) for c in a]
        b[4]["body"] = "A completely rewritten supporting sentence."
        self.assertEqual(cardtemplates.pick_templates(a, "seed"),
                         cardtemplates.pick_templates(b, "seed"))

    def test_a_family_change_never_reaches_the_cards_before_it(self):
        """A card whose family flips can cascade forward through the no-repeat rotation, but the
        cards already on screen before it must never move."""
        a = cards(9)                                     # "Headline number N" -> every card is stat
        b = [dict(c) for c in a]
        b[4]["headline"] = "A line carrying no figures"  # -> classic
        pa = cardtemplates.pick_templates(a, "seed")
        pb = cardtemplates.pick_templates(b, "seed")
        self.assertNotEqual(pa[4], pb[4])
        self.assertEqual(pa[:4], pb[:4])

    def test_classic_only_pins_every_card(self):
        config.STORYBOARD_TEMPLATE_SET = "Classic lower third only"
        picks = cardtemplates.pick_templates(cards(6), "seed")
        self.assertEqual(set(picks), {"lower_third"})  # the no-repeat rule degrades, never raises

    def test_empty_and_single_card_lists(self):
        self.assertEqual(cardtemplates.pick_templates([], "seed"), [])
        self.assertEqual(len(cardtemplates.pick_templates(cards(1), "seed")), 1)


class Render(unittest.TestCase):
    """Every template, at every orientation, keeps its content inside the safe area."""

    @classmethod
    def setUpClass(cls):
        cls.dir = TMP / "render"
        cls.dir.mkdir(parents=True, exist_ok=True)
        cls.img = cls.dir / "pic.jpg"
        Image.new("RGB", (160, 120), (80, 90, 110)).save(cls.img)

    def _render(self, name, size, c, image=None):
        boxes = []
        out = storyboard.render_card(c, size, self.dir / f"{name}.png", image=image,
                                     index=2, total=5, template=name, boxes=boxes)
        return out, boxes

    def _assert_inside(self, boxes, size, label):
        ctx = cardtemplates.context(size, None, 2, 5)
        slack = int(14 * ctx.s)            # _shadowed deliberately bleeds past the element
        for x0, y0, x1, y1 in boxes:
            self.assertGreaterEqual(x0, -slack, label)
            self.assertLessEqual(x1, size[0] + slack, label)
            self.assertGreaterEqual(y0, ctx.top_safe - slack, label)
            self.assertLessEqual(y1, ctx.floor + slack, label)

    def test_every_template_renders_in_every_orientation(self):
        for name in cardtemplates.TEMPLATES:
            for shape, size in SIZES.items():
                for has_img in (False, True):
                    label = f"{name}/{shape}/{'img' if has_img else 'text'}"
                    with self.subTest(label):
                        out, boxes = self._render(name, size, card(),
                                                  image=self.img if has_img else None)
                        with Image.open(out) as im:
                            self.assertEqual(im.size, size, label)
                            self.assertEqual(im.mode, "RGBA", label)
                            self.assertTrue(im.getchannel("A").getbbox(), f"{label}: nothing drawn")
                        self._assert_inside(boxes, size, label)

    def test_pathological_text_stays_inside_the_safe_area(self):
        """A single unbreakable word used to be drawn straight past the column edge."""
        c = card(headline="Supercalifragilisticexpialidocious" * 6, body="x" * 400)
        for name in cardtemplates.TEMPLATES:
            for shape, size in SIZES.items():
                with self.subTest(f"{name}/{shape}"):
                    _out, boxes = self._render(name, size, c)
                    self._assert_inside(boxes, size, f"{name}/{shape}")

    def test_headline_only_and_body_only_cards_render(self):
        for c in (card(kicker="", body=""), card(headline="", kicker="")):
            for name in cardtemplates.TEMPLATES:
                out, _ = self._render(name, SIZES["landscape"], c)
                with Image.open(out) as im:
                    self.assertTrue(im.getchannel("A").getbbox(), name)

    def test_a_corrupt_image_still_leaves_the_text_card(self):
        bad = self.dir / "bad.jpg"
        bad.write_bytes(b"not an image at all")
        for name in ("split", "photo_stack", "lower_third"):
            out, _ = self._render(name, SIZES["landscape"], card(), image=bad)
            with Image.open(out) as im:
                self.assertTrue(im.getchannel("A").getbbox(), name)

    def test_an_unknown_template_falls_back_instead_of_raising(self):
        """plan.json may name a layout a later version dropped."""
        a, _ = self._render("does_not_exist", SIZES["landscape"], card())
        b = storyboard.render_card(card(), SIZES["landscape"], self.dir / "fallback.png",
                                   index=2, total=5, template="lower_third")
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())

    def test_no_template_argument_is_the_classic_layout(self):
        """The default path must not drift: it is the 'classic only' escape hatch."""
        a = storyboard.render_card(card(), SIZES["landscape"], self.dir / "default.png",
                                   index=2, total=5)
        b = storyboard.render_card(card(), SIZES["landscape"], self.dir / "explicit.png",
                                   index=2, total=5, template="lower_third")
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())


class Motion(unittest.TestCase):
    """Cards ease into place with a per-frame overlay position - no second encode."""

    def setUp(self):
        self._on = config.STORYBOARD_MOTION
        config.STORYBOARD_MOTION = True
        self.addCleanup(setattr, config, "STORYBOARD_MOTION", self._on)

    def test_a_full_bleed_banner_only_fades(self):
        self.assertEqual(cardtemplates.motion_xy("full_bleed", (1920, 1080), 4.0, 12.0), (0, 0))

    def test_motion_off_returns_plain_zeros(self):
        config.STORYBOARD_MOTION = False
        self.assertEqual(cardtemplates.motion_xy("ribbon", (1920, 1080), 4.0, 12.0), (0, 0))

    def test_a_sliding_card_moves_on_one_axis_only(self):
        x, y = cardtemplates.motion_xy("ribbon", (1920, 1080), 4.0, 12.0)
        self.assertIsInstance(x, str)
        self.assertEqual(y, 0)
        self.assertIn("clip((t-4.000)", x)
        self.assertIn("pow(", x)

    def test_a_rising_card_moves_on_y(self):
        x, y = cardtemplates.motion_xy("lower_third", (1920, 1080), 0.0, 9.0)
        self.assertEqual(x, 0)
        self.assertIn("clip((t-0.000)", y)

    def test_the_travel_is_clamped_on_a_short_card(self):
        """Mirrors composite()'s own fade clamp, so a card is never still moving as it fades out."""
        _x, y = cardtemplates.motion_xy("lower_third", (1920, 1080), 2.0, 3.0)
        secs = float(y.split("/")[1].split(",")[0])
        self.assertLessEqual(secs, 1.0 / 3 + 1e-6)

    def test_the_expression_is_quoted_by_composite_so_its_commas_survive(self):
        from services import video
        expr = cardtemplates.motion_xy("lower_third", (1920, 1080), 0.0, 9.0)[1]
        self.assertTrue(video._pos(expr).startswith("'"))
        self.assertTrue(video._pos(expr).endswith("'"))
        self.assertEqual(video._pos(0), "0")       # integers stay bare for the page-scroll panel


class Freeze(unittest.TestCase):
    """The layout is decided once and frozen into plan.json, never re-decided."""

    def test_templates_are_assigned_once_and_kept(self):
        cs = cards(5)
        storyboard._freeze_templates(cs, "20260101-000000-abc123")
        first = [c["template"] for c in cs]
        cs[0]["template"] = "quote_slab"            # a hand-edited plan.json wins
        storyboard._freeze_templates(cs, "20260101-000000-abc123")
        self.assertEqual(cs[0]["template"], "quote_slab")
        self.assertEqual([c["template"] for c in cs[1:]], first[1:])

    def test_a_frozen_pick_matches_what_render_all_would_compute(self):
        """The two paths cannot disagree for a job planned by this version."""
        cs = cards(6)
        storyboard._freeze_templates(cs, "job-xyz")
        self.assertEqual([c["template"] for c in cs],
                         cardtemplates.pick_templates(cs, "job-xyz"))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
