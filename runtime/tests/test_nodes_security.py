from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import importlib.util
import json
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_nodes():
    package_name = "isolated_yue2_nodes_security"
    package = ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = Mock()
    utils = ModuleType("comfy.utils")
    utils.ProgressBar = Mock()
    comfy.model_management = management
    comfy.utils = utils
    folder_paths = ModuleType("folder_paths")
    torch = ModuleType("torch")
    torch.device = lambda value: SimpleNamespace(type=value.split(":")[0])
    backend = ModuleType(package_name + ".backend")
    backend.run = Mock()
    modules = {
        package_name: package, package_name + ".backend": backend,
        "comfy": comfy, "comfy.model_management": management, "comfy.utils": utils,
        "folder_paths": folder_paths, "torch": torch,
        "numpy": ModuleType("numpy"), "soundfile": ModuleType("soundfile"),
    }
    spec = importlib.util.spec_from_file_location(package_name + ".nodes", ROOT / "nodes.py")
    nodes = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(nodes)
    return nodes


class NodeSecurityBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.nodes = load_nodes()
        self.nodes.folder_paths.models_dir = str(self.root / "models")
        self.nodes.folder_paths.get_temp_directory = lambda: str(self.root / "temp")
        self.models = {"model_key": "yue2", "vae_key": "vae", "device": "cpu",
                       "backend": "torch-eager", "quantization": "none", "offload_ar": False,
                       "memory_budget_gib": 16., "vae_core_frames": 1024, "verify_hashes": True}
        self.nodes.model_directory = Mock(return_value=str(self.root / "registered-model"))
        self.nodes.run.return_value = {"artifact_id": "plan-12345678", "abc": "X:1", "metadata": {}}

    def tearDown(self):
        self.temp.cleanup()

    def test_incoming_dictionaries_cannot_override_server_paths_or_model_keys(self):
        injected = {"model_dir": "\\\\attacker.invalid\\weights", "vae_dir": "C:/private",
                    "artifact_root": "\\\\attacker.invalid\\artifacts", "input_dir": "/private",
                    "models_root": "\\\\attacker.invalid\\models", "cancel_file": "/private/cancel"}
        models = {**self.models, **injected}
        options = {**injected, "model_key": "sheetsage2", "vae_key": "mert2",
                   "config": {"ode_steps": 8}, "decode_full": True}
        self.nodes.execute("plan", models, options, request={"style": "piano"})
        action, payload = self.nodes.run.call_args.args[:2]
        self.assertEqual(action, "plan")
        self.assertEqual(payload["artifact_root"], str(self.root / "temp" / "yue2"))
        self.assertEqual(payload["models_root"], str(self.root / "models"))
        self.assertEqual(payload["model_key"], "yue2")
        self.assertEqual(payload["vae_key"], "vae")
        self.assertEqual(payload["config"], {"ode_steps": 8})
        self.assertTrue(payload["decode_full"])
        for key in ("model_dir", "vae_dir", "input_dir", "cancel_file"):
            self.assertNotIn(key, payload)
        self.assertEqual([call.args[0] for call in self.nodes.model_directory.call_args_list], ["yue2", "vae"])

    def test_invalid_model_selections_never_launch_worker(self):
        for model_key, vae_key in (("forged", "vae"), ("vae", "vae"), ("yue2", "yue2"),
                                   ("yue2", "\\\\attacker.invalid\\vae")):
            with self.subTest(model_key=model_key, vae_key=vae_key):
                with self.assertRaises((ValueError, TypeError)):
                    self.nodes.execute("plan", {**self.models, "model_key": model_key, "vae_key": vae_key}, request={})
        self.nodes.run.assert_not_called()

    def test_stage_nodes_accept_only_local_artifact_ids(self):
        artifacts = self.root / "temp" / "yue2"
        artifacts.mkdir(parents=True)
        directory = artifacts / "plan-12345678"
        directory.mkdir()
        (directory / "plan.json").write_text(json.dumps({}), encoding="utf-8")
        stage = {"artifact_id": directory.name, "plan_dir": "\\\\attacker.invalid\\plan",
                 "semantic_dir": "C:/private", "artifact_dir": "/private"}
        with patch.object(self.nodes, "audio_result", return_value={"waveform": "fixture"}):
            self.nodes.YuE2SemanticTokens().generate(self.models, stage)
            self.nodes.YuE2Synthesize().synthesize(self.models, stage)
            self.nodes.YuE2Decode().decode(self.models, stage)
        self.assertEqual(self.nodes.run.call_count, 3)
        for call in self.nodes.run.call_args_list:
            payload = call.args[1]
            self.assertEqual(payload["input_artifact"], directory.name)
            for key in ("input_dir", "plan_dir", "semantic_dir", "artifact_dir"):
                self.assertNotIn(key, payload)

    def test_forged_stage_paths_never_launch_worker(self):
        for stage in ({"plan_dir": "\\\\attacker.invalid\\plan"},
                      {"artifact_id": "\\\\attacker.invalid\\plan"},
                      {"artifact_id": "C:plan-12345678"},
                      {"artifact_id": "../plan-12345678"},
                      {"artifact_id": "plan-12345678:stream"}):
            with self.subTest(stage=stage):
                with self.assertRaises((ValueError, TypeError)):
                    self.nodes.YuE2SemanticTokens().generate(self.models, stage)
        self.nodes.run.assert_not_called()

    def test_audio_is_loaded_only_from_server_artifact_directory(self):
        directory = self.root / "temp" / "yue2" / "decode-12345678"
        directory.mkdir(parents=True)
        (directory / "audio.npy").touch()
        result = {"artifact_id": directory.name, "audio_array": "\\\\attacker.invalid\\audio.npy",
                  "audio_path": "C:/private/audio.flac"}
        with patch.object(self.nodes.np, "load", create=True, side_effect=RuntimeError("stop before decoding")) as load:
            with self.assertRaisesRegex(RuntimeError, "stop before decoding"):
                self.nodes.audio_result(result)
        self.assertEqual(Path(load.call_args.args[0]), directory.resolve() / "audio.npy")
        self.assertFalse(load.call_args.kwargs["allow_pickle"])

    def test_audio_rejects_forged_artifact_before_reading_numpy(self):
        with patch.object(self.nodes.np, "load", create=True) as load:
            with self.assertRaises((ValueError, TypeError)):
                self.nodes.audio_result({"artifact_id": "\\\\attacker.invalid\\decode-12345678"})
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
