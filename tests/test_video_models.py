"""The video-model registry: run with  python -m unittest discover -s tests -t .  from the repo root.

Pure unit tests: the workflow templates are read off disk, but nothing is submitted to ComfyUI.

The point of this file is that config.VIDEO_MODELS is a promise about files it does not contain -
node ids, input field names and frame arithmetic that live in the workflow JSONs. A workflow
re-exported from ComfyUI renumbers its nodes, and the first sign of that would otherwise be a
job failing after minutes of GPU time. TemplateIntegrity catches it in milliseconds.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config

TMP = Path(tempfile.mkdtemp(prefix="videogen-models-"))
config.OUTPUT_DIR = TMP / "output"

from pipeline import JobManager  # noqa: E402  (config patched first)
from services.comfy import ComfyClient  # noqa: E402

VARIANTS = [(name, variant) for name in config.VIDEO_MODELS for variant in ("t2v", "i2v")]


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TemplateIntegrity(unittest.TestCase):
    """Every node id and input field the registry names really exists in the workflow it names."""

    def test_every_variant_names_nodes_that_exist(self):
        for name, variant in VARIANTS:
            spec = config.VIDEO_MODELS[name][variant]
            path = config.WORKFLOWS_DIR / spec["file"]
            with self.subTest(model=name, variant=variant):
                self.assertTrue(path.is_file(), f"missing workflow {spec['file']}")
                wf = json.loads(path.read_text(encoding="utf-8"))
                # (node, field) pairs: the field has to be an input on that node
                pairs = [spec["prompt"], spec["duration"], *spec["seeds"]]
                if spec["negative"]:
                    pairs.append(spec["negative"])
                for node, field in pairs:
                    self.assertIn(node, wf)
                    self.assertIn(field, wf[node]["inputs"])
                # bare node ids: build() writes a fixed field name on these
                self.assertIn("filename_prefix", wf[spec["save"]]["inputs"])
                for field in ("aspect_ratio", "megapixels"):
                    self.assertIn(field, wf[spec["resolution"]]["inputs"])
                for key in ("load_image", "lightning"):
                    if spec[key]:
                        self.assertIn(spec[key], wf)

    def test_only_the_first_frame_variant_loads_an_image(self):
        for name in config.VIDEO_MODELS:
            with self.subTest(model=name):
                self.assertIsNone(config.VIDEO_MODELS[name]["t2v"]["load_image"])
                self.assertTrue(config.VIDEO_MODELS[name]["i2v"]["load_image"])

    def test_the_template_resolution_is_a_selectable_one(self):
        """A shipped template should open in ComfyUI at values the app itself allows."""
        for name, variant in VARIANTS:
            spec = config.VIDEO_MODELS[name][variant]
            wf = json.loads((config.WORKFLOWS_DIR / spec["file"]).read_text(encoding="utf-8"))
            res = wf[spec["resolution"]]["inputs"]
            with self.subTest(model=name, variant=variant):
                self.assertIn(res["aspect_ratio"], config.ASPECT_RATIOS)
                self.assertTrue(any(abs(res["megapixels"] - o) < 1e-6
                                    for o in config.MEGAPIXELS_OPTIONS))

    def test_model_workflow_picks_the_variant(self):
        for name in config.VIDEO_MODELS:
            with self.subTest(model=name):
                self.assertEqual(config.model_workflow(name, False).name,
                                 config.VIDEO_MODELS[name]["t2v"]["file"])
                self.assertEqual(config.model_workflow(name, True).name,
                                 config.VIDEO_MODELS[name]["i2v"]["file"])

    def test_an_unknown_or_blank_model_falls_back_to_the_settings_default(self):
        with mock.patch.object(config, "DEFAULT_VIDEO_MODEL", "LTX 2.5"):
            for name in ("", "Wan 2.2", None):
                with self.subTest(name=name):
                    self.assertIs(config.video_model(name or ""),
                                  config.VIDEO_MODELS["LTX 2.5"])


class FrameMath(unittest.TestCase):
    """Each model quantises a requested clip length the way its own graph does."""

    def test_minimax_h3_rounds_to_the_frame_quantum(self):
        # node 140:132: max(5, round(s * 24)) raised to the next f = 5 (mod 17)
        self.assertEqual(config.clip_frames(5, "MiniMax H3"), 124)
        self.assertEqual(config.clip_frames(2, "MiniMax H3"), 56)
        self.assertEqual(config.clip_frames(0.1, "MiniMax H3"), 5)
        self.assertEqual(config.clip_frames(10, "MiniMax H3"), 243)

    def test_ltx_renders_whole_seconds_plus_one_frame(self):
        # node 398:378: duration * 24 + 1
        self.assertEqual(config.clip_frames(5, "LTX 2.5"), 121)
        self.assertEqual(config.clip_frames(2, "LTX 2.5"), 49)
        self.assertEqual(config.clip_frames(10, "LTX 2.5"), 241)
        # its duration node is whole seconds, so a sub-second clip is not expressible: the
        # shortest it can render is one second, which is why stills stay on MiniMax H3
        self.assertEqual(config.clip_frames(0.1, "LTX 2.5"), 25)

    def test_a_fraction_buys_the_next_whole_second_on_ltx(self):
        """A clip must never come out SHORTER than asked: a spoken line would be cut off."""
        for secs in (2.1, 4.9, 5.0001, 7.5):
            with self.subTest(secs=secs):
                self.assertGreaterEqual(config.actual_clip_seconds(secs, "LTX 2.5"), secs)

    def test_actual_clip_seconds_is_a_fixed_point(self):
        """plan["clip_seconds"] is frozen at plan time and re-quantised by build() at render
        time, so a length that came out of this function has to survive going back in."""
        for name in config.VIDEO_MODELS:
            for req in (2, 5, 6, 8, 10, 12):
                real = config.actual_clip_seconds(req, name)
                with self.subTest(model=name, req=req):
                    self.assertEqual(config.actual_clip_seconds(real, name), real)

    def test_the_two_models_really_do_disagree(self):
        """If this ever stops being true the per-model threading can be dropped again."""
        self.assertNotEqual(config.actual_clip_seconds(5, "MiniMax H3"),
                            config.actual_clip_seconds(5, "LTX 2.5"))

    def test_the_default_model_keeps_the_legacy_numbers(self):
        """Every call site that does not know a model must behave exactly as it used to."""
        self.assertEqual(config.DEFAULT_VIDEO_MODEL, "MiniMax H3")
        self.assertEqual(config.clip_frames(5), 124)
        self.assertEqual(config.actual_clip_seconds(5), 124 / 24)
        self.assertEqual(config.clip_frames(5, ""), config.clip_frames(5, "MiniMax H3"))

    def test_duration_value_matches_what_each_node_accepts(self):
        h3 = config.model_duration_value(5, "MiniMax H3")
        self.assertIsInstance(h3, float)
        self.assertEqual(h3, 5.0)                      # the graph quantises it itself
        ltx = config.model_duration_value(5, "LTX 2.5")
        self.assertIsInstance(ltx, int)                # PrimitiveInt
        self.assertEqual(ltx, 5)
        # fed the real length back, LTX still asks the node for the same whole seconds
        self.assertEqual(config.model_duration_value(121 / 24, "LTX 2.5"), 5)

    def test_duration_value_reproduces_the_frame_count(self):
        """What the node is handed must render the length the planner froze, not a neighbour."""
        for req in (2, 5, 8, 10):
            frames = config.clip_frames(req, "LTX 2.5")
            secs = config.model_duration_value(req, "LTX 2.5")
            with self.subTest(req=req):
                self.assertEqual(secs * config.VIDEO_MODELS["LTX 2.5"]["fps"] + 1, frames)

    def test_clip_durations_cover_the_audio_on_both_models(self):
        for name in config.VIDEO_MODELS:
            durs = config.clip_durations(90.0, 5, name)
            with self.subTest(model=name):
                self.assertGreaterEqual(sum(durs), 90.0)
                self.assertEqual(durs[:-1], [config.actual_clip_seconds(5, name)] * (len(durs) - 1))


class BuildPatching(unittest.TestCase):
    """build() writes every per-render value into the nodes the registry names."""

    def _build(self, model, first_frame=None, **kw):
        spec = config.model_variant(model, first_frame is not None)
        template = ComfyClient.load_workflow(config.model_workflow(model, first_frame is not None))
        kw.setdefault("seconds", 5)
        kw.setdefault("aspect_ratio", "9:16 (Portrait Widescreen)")
        kw.setdefault("megapixels", 0.6)
        wf = ComfyClient.build(template, "a cat on a wall", "videogen/j/clip_01", seed=7,
                               first_frame=first_frame, model=model, negative="NEG", **kw)
        return wf, spec

    def test_every_model_gets_prompt_duration_resolution_and_save(self):
        for name, variant in VARIANTS:
            ff = "frame_01.jpg" if variant == "i2v" else None
            wf, spec = self._build(name, ff)
            res = wf[spec["resolution"]]["inputs"]
            with self.subTest(model=name, variant=variant):
                self.assertEqual(wf[spec["prompt"][0]]["inputs"][spec["prompt"][1]],
                                 "a cat on a wall")
                self.assertEqual(wf[spec["duration"][0]]["inputs"][spec["duration"][1]],
                                 config.model_duration_value(5, name))
                self.assertEqual(res["aspect_ratio"], "9:16 (Portrait Widescreen)")
                self.assertEqual(res["megapixels"], 0.6)
                self.assertEqual(wf[spec["save"]]["inputs"]["filename_prefix"],
                                 "videogen/j/clip_01")
                if ff:
                    self.assertEqual(wf[spec["load_image"]]["inputs"]["image"], "frame_01.jpg")

    def test_every_noise_node_gets_its_own_seed(self):
        """LTX samples in two passes; its second RandomNoise ships pinned at 42, which would
        make every render of a given prompt repeat the same refinement."""
        for name, variant in VARIANTS:
            spec = config.model_variant(name, variant == "i2v")
            wf, _ = self._build(name, "frame_01.jpg" if variant == "i2v" else None)
            seeds = [wf[n]["inputs"][f] for n, f in spec["seeds"]]
            with self.subTest(model=name, variant=variant):
                self.assertEqual(seeds, list(range(7, 7 + len(seeds))))
                self.assertEqual(len(seeds), len(set(seeds)))

    def test_the_negative_prompt_reaches_ltx_and_is_dropped_for_h3(self):
        for variant in ("t2v", "i2v"):
            ff = "frame_01.jpg" if variant == "i2v" else None
            wf, spec = self._build("LTX 2.5", ff)
            with self.subTest(variant=variant):
                self.assertEqual(wf[spec["negative"][0]]["inputs"][spec["negative"][1]], "NEG")
            # H3 has no negative input at all, so passing one must simply do nothing
            self.assertIsNone(config.model_variant("MiniMax H3", ff is not None)["negative"])

    def test_the_lightning_switch_is_h3_only(self):
        with mock.patch.object(config, "LIGHTNING_LORA", True):
            wf, spec = self._build("MiniMax H3")
            self.assertIs(wf[spec["lightning"]]["inputs"]["value"], True)
            # LTX 2.5 is already distilled: it has no LoRA node to switch
            self.assertIsNone(config.model_variant("LTX 2.5")["lightning"])

    def test_a_bad_aspect_ratio_is_refused_before_anything_is_patched(self):
        for name in config.VIDEO_MODELS:
            with self.subTest(model=name), self.assertRaises(ValueError):
                self._build(name, aspect_ratio="5:4 (Nope)")

    def test_the_template_itself_is_never_mutated(self):
        """build() is called once per clip off one loaded template, so a leak would make clip 2
        inherit clip 1's prompt and seed."""
        for name in config.VIDEO_MODELS:
            path = config.model_workflow(name)
            before = path.read_text(encoding="utf-8")
            template = ComfyClient.load_workflow(path)
            snapshot = json.dumps(template, sort_keys=True)
            ComfyClient.build(template, "x", "p", seed=1, seconds=5, model=name, negative="n")
            with self.subTest(model=name):
                self.assertEqual(json.dumps(template, sort_keys=True), snapshot)
                self.assertEqual(path.read_text(encoding="utf-8"), before)


class StillsIgnoreTheVideoModel(unittest.TestCase):
    """Thumbnails, cast portraits and story board images are 5-frame renders at 1 MP - a length
    only the H3 graph can express - so they must stay on it whatever a video picked."""

    def test_a_thumbnail_renders_on_the_stills_model(self):
        h3 = config.VIDEO_MODELS[config.STILLS_VIDEO_MODEL]["t2v"]
        with mock.patch.object(config, "DEFAULT_VIDEO_MODEL", "LTX 2.5"):
            wf = ComfyClient.build_thumbnail(ComfyClient.load_workflow(config.T2V_WORKFLOW),
                                             "p", "videogen/x/thumb")
            self.assertEqual(wf[h3["prompt"][0]]["inputs"][h3["prompt"][1]], "p")
            self.assertEqual(wf[h3["duration"][0]]["inputs"][h3["duration"][1]], 0.1)
            self.assertEqual(wf[h3["resolution"]]["inputs"]["megapixels"], 1.0)

    def test_the_stills_constants_point_at_the_stills_model(self):
        h3 = config.VIDEO_MODELS[config.STILLS_VIDEO_MODEL]
        self.assertEqual(config.T2V_WORKFLOW.name, h3["t2v"]["file"])
        self.assertEqual(config.I2V_WORKFLOW.name, h3["i2v"]["file"])
        self.assertEqual(config.clip_frames(0.1, config.STILLS_VIDEO_MODEL), 5)


class ResolveChoices(unittest.TestCase):
    """The per-video choice: request value -> Settings default, validated. No channel layer, the
    same as megapixels and the output resolution."""

    def resolve(self, video_model):
        return JobManager._resolve_choices("", "", "", "", "", None, video_model=video_model)

    def test_every_model_is_accepted(self):
        for name in config.VIDEO_MODELS:
            with self.subTest(model=name):
                self.assertEqual(self.resolve(name)[-3], name)

    def test_empty_falls_back_to_the_settings_default(self):
        with mock.patch.object(config, "DEFAULT_VIDEO_MODEL", "LTX 2.5"):
            self.assertEqual(self.resolve("")[-3], "LTX 2.5")
            self.assertEqual(self.resolve(None)[-3], "LTX 2.5")

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            self.resolve("Wan 2.2")

    def test_the_output_resolution_is_still_last(self):
        """The tuple is unpacked positionally in three places and indexed [-1] in the tests."""
        out = JobManager._resolve_choices("", "", "", "", "", None,
                                          resolution="2K (1440p)", megapixels=0.6,
                                          video_model="LTX 2.5")
        self.assertEqual(out[-1], "2K (1440p)")
        self.assertEqual(out[-2], 0.6)
        self.assertEqual(out[-3], "LTX 2.5")


class StoryGridFollowsTheModel(unittest.TestCase):
    """A story sizes its clips around spoken lines, so the model's quantiser decides the grid."""

    def test_the_clip_grid_is_quantised_by_the_given_model(self):
        from services import story
        for name in config.VIDEO_MODELS:
            durs = story.clip_durations([0.4, 9.0, 1.2], 5, name)
            with self.subTest(model=name):
                for dur in durs:
                    self.assertEqual(config.actual_clip_seconds(dur, name), dur)
                self.assertEqual(durs[0], config.actual_clip_seconds(5, name))

    def test_a_line_longer_than_the_clip_buys_the_room_it_needs(self):
        from services import story
        for name in config.VIDEO_MODELS:
            need = config.STORY_LINE_LEAD + 9.0 + config.STORY_LINE_GAP
            with self.subTest(model=name):
                self.assertGreaterEqual(story.clip_durations([9.0], 5, name)[0], need)

    def test_beat_count_and_line_words_take_a_model(self):
        from services import story
        for name in config.VIDEO_MODELS:
            with self.subTest(model=name):
                self.assertGreater(story.beat_count(60.0, 5, name), 0)
                self.assertGreater(story.line_words(5, name), 0)


if __name__ == "__main__":
    unittest.main()
