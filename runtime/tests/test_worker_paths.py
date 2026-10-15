from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np
from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.protocol import SongRequest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
spec = importlib.util.spec_from_file_location("isolated_yue2_worker_paths", ROOT / "runtime" / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FixturePipeline:
    def __init__(self):
        self.weights = {}
        self.load_timing = {}
        self.abc_ids = [71, 71, 46, 93]
        self.prefix = [151643, 151847, *self.abc_ids, 151848, 151851]
        self.tokens = [3, 3, 19, 32767]
        self.latents = np.arange(256, dtype=np.float32).reshape(4, 64) / 13
        self.decoded = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def plan(self, request, cancelled):
        return SymbolicPlan(request, "X:1\nK:C\nCDEF|", self.abc_ids, self.prefix)

    def generate_semantic(self, plan, cancelled):
        return SemanticResult(plan, self.tokens, {"output_tokens": 4}, False)

    def synthesize(self, semantic, cancelled):
        return self.latents.copy()

    def decode(self, latents, full):
        self.decoded.append(latents.copy())
        return np.zeros((480, 2), dtype=np.float32)

    def effective_config(self, request):
        return {}


class WorkerPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.artifacts = self.root / "comfy-temp" / "yue2"
        self.artifacts.mkdir(parents=True)
        self.directories = {
            "yue2": "diffusion_models/YuE2/YuE2-3B",
            "vae": "vae/YuE2/YuE2-Vae",
            "vae_legacy": "vae/YuE2/YuE2-Vae-legacy",
        }
        for relative in self.directories.values():
            directory = self.models / relative
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"fixture")
        (self.models / self.directories["yue2"] / "qwen.tiktoken").write_bytes(b"fixture")
        index = self.models / "diffusion_models/YuE2/official_models.json"
        index.write_text(json.dumps({"models": {key: {"directory": relative}
                                               for key, relative in self.directories.items()}}), encoding="utf-8")
        self.pipe = FixturePipeline()

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, action, input_artifact=None, vae_key="vae"):
        output = Path(tempfile.mkdtemp(prefix=action + "-", dir=self.artifacts))
        payload = {
            "action": action, "model_key": "yue2", "vae_key": vae_key,
            "models_root": str(self.models), "artifact_root": str(self.artifacts),
            "artifact_id": output.name, "device": "cpu", "backend": "torch-eager",
            "request": SongRequest(style="original piano pop", lyrics="[Verse]\nA new morning", seed=312).to_dict(),
        }
        if input_artifact is not None:
            payload["input_artifact"] = input_artifact
        return payload

    def assert_registered_models(self, calls, vae_key):
        for call in calls:
            self.assertEqual(call.args[0], str((self.models / self.directories["yue2"]).resolve()))
            self.assertEqual(call.kwargs["vae"], str((self.models / self.directories[vae_key]).resolve()))
            self.assertTrue(call.kwargs["local_files_only"])

    def test_stage_chain_preserves_exact_tokens_and_float32_latents_with_both_vaes(self):
        for vae_key in ("vae", "vae_legacy"):
            with self.subTest(vae_key=vae_key):
                source = None
                with patch.object(worker.ComfyProgressPipeline, "from_pretrained", return_value=self.pipe) as load:
                    for action in ("plan", "semantic", "synthesize", "decode"):
                        result = worker.execute(self.payload(action, source, vae_key))
                        self.assertEqual(set(result), {"artifact_id", "abc", "metadata"} | ({"sample_rate"} if action == "decode" else set()))
                        self.assertEqual(result["abc"], "X:1\nK:C\nCDEF|")
                        source = result["artifact_id"]
                        directory = self.artifacts / source
                        plan = SymbolicPlan.load(directory)
                        self.assertEqual(plan.abc_ids, self.pipe.abc_ids)
                        self.assertEqual(plan.prefix, self.pipe.prefix)
                        if action != "plan":
                            semantic = worker.load_semantic(directory)
                            self.assertEqual(semantic.tokens, self.pipe.tokens)
                        if action in {"synthesize", "decode"}:
                            latents, _, _ = worker.load_latents(directory)
                            np.testing.assert_array_equal(latents, self.pipe.latents)
                            self.assertEqual(latents.dtype, np.float32)
                    self.assert_registered_models(load.call_args_list, vae_key)
                    self.assertEqual(load.call_count, 4)
                audio = np.load(directory / "audio.npy", allow_pickle=False)
                self.assertEqual(audio.shape, (480, 2))
                self.assertEqual(audio.dtype, np.float32)
                self.assertTrue((directory / "audio.flac").is_file())
                self.assertEqual(result["sample_rate"], 48000)
                np.testing.assert_array_equal(self.pipe.decoded[-1], self.pipe.latents)

    def test_unc_input_artifact_is_rejected_before_model_loading(self):
        for action in ("semantic", "synthesize", "decode"):
            with self.subTest(action=action):
                payload = self.payload(action, "\\\\attacker.invalid\\share\\plan-12345678")
                with patch.object(worker, "check_models") as check, patch.object(worker.ComfyProgressPipeline, "from_pretrained") as load:
                    with self.assertRaisesRegex(ValueError, "artifact ID"):
                        worker.execute(payload)
                check.assert_not_called()
                load.assert_not_called()

    def test_raw_path_fields_cannot_override_registered_models_or_artifact_ids(self):
        forged = {key: "\\\\attacker.invalid\\share\\" + key
                  for key in ("model_dir", "vae_dir", "input_dir", "artifact_dir", "cancel_file")}
        source = None
        with patch.object(worker.ComfyProgressPipeline, "from_pretrained", return_value=self.pipe) as load:
            for action in ("plan", "semantic", "synthesize", "decode", "generate"):
                payload = {**self.payload(action, source), **forged}
                result = worker.execute(payload)
                self.assertEqual(result["artifact_id"], payload["artifact_id"])
                self.assertTrue((self.artifacts / result["artifact_id"] / "plan.json").is_file())
                source = result["artifact_id"]
            self.assert_registered_models(load.call_args_list, "vae")
            self.assertEqual(load.call_count, 5)
        self.assertEqual(result["sample_rate"], 48000)


if __name__ == "__main__":
    unittest.main()
