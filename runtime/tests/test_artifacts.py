from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.protocol import SongRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import load_latents, load_semantic, save_latents, save_semantic


class ExactStageArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        request = SongRequest(style="original piano pop", lyrics="[Verse]\nA new morning", seed=312)
        self.plan = SymbolicPlan(request, "X:1\nK:C\nCDEF|", [71, 71, 46, 93], [151643, 151847, 71, 71, 46, 93, 151848, 151851])
        self.semantic = SemanticResult(self.plan, [3, 3, 19, 32767], {"output_tokens": 4}, False)

    def tearDown(self):
        self.temp.cleanup()

    def test_symbolic_ids_survive_stage_boundary_exactly(self):
        save_semantic(self.semantic, self.directory)
        loaded = load_semantic(self.directory)
        self.assertEqual(loaded.plan.abc_ids, self.plan.abc_ids)
        self.assertEqual(loaded.plan.prefix, self.plan.prefix)
        self.assertEqual(loaded.tokens, self.semantic.tokens)
        self.assertEqual(loaded.plan.request, self.plan.request)

    def test_latents_roundtrip_without_precision_change(self):
        latents = np.arange(256, dtype=np.float32).reshape(4, 64) / 13
        save_latents(latents, self.semantic, self.directory, {"nar_seconds": 7.25})
        loaded, semantic, timing = load_latents(self.directory)
        np.testing.assert_array_equal(loaded, latents)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(semantic.tokens, self.semantic.tokens)
        self.assertEqual(timing["nar_seconds"], 7.25)

    def test_changed_score_is_rejected_without_retokenization(self):
        save_semantic(self.semantic, self.directory)
        (self.directory / "score.abc").write_text("X:1\nK:C\nGABC|", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Saved plan changed"):
            load_semantic(self.directory)

    def test_changed_semantic_tokens_are_rejected(self):
        save_semantic(self.semantic, self.directory)
        np.save(self.directory / "semantic.npy", np.asarray([0, 0], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "semantic tokens were changed"):
            load_semantic(self.directory)

    def test_changed_acoustic_latents_are_rejected(self):
        latents = np.zeros((4, 64), dtype=np.float32)
        save_latents(latents, self.semantic, self.directory, {})
        np.save(self.directory / "latent.npy", latents + 1)
        with self.assertRaisesRegex(ValueError, "acoustic latents were changed"):
            load_latents(self.directory)


if __name__ == "__main__":
    unittest.main()
