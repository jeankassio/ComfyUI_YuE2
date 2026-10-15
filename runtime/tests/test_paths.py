from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("isolated_yue2_paths", ROOT / "runtime" / "paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class ArtifactPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "comfy-temp" / "yue2"
        self.artifacts.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self, name="plan-abc123_4"):
        directory = self.artifacts / name
        directory.mkdir()
        (directory / "plan.json").write_text("{}", encoding="utf-8")
        return directory

    def symlink(self, link, target, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")

    def test_valid_stage_identifiers_resolve_under_server_root(self):
        for name in ("generate-12ab34cd", "plan-abc123_4", "semantic-0123_ab_",
                     "synthesize-12345678", "decode-89ab01cd", "transcribe-a1b2c3d4"):
            with self.subTest(name=name):
                directory = self.artifact(name)
                self.assertEqual(paths.artifact_directory(self.artifacts, name), directory.resolve())

    def test_untrusted_artifact_values_are_rejected_before_filesystem_resolution(self):
        invalid = (
            "/tmp/plan-abc123_4", "C:/plan-abc123_4", "C:\\plan-abc123_4", "C:plan-abc123_4",
            "\\\\attacker.invalid\\share\\plan-abc123_4", "//attacker.invalid/share/plan-abc123_4",
            "\\\\?\\C:\\plan-abc123_4", "\\\\.\\pipe\\plan-abc123_4", "\\plan-abc123_4",
            "plan-abc123_4:stream", "../plan-abc123_4", "..\\plan-abc123_4",
            "plan-abc123_4/../decode-89ab01cd", "plan-abc123_4\\../decode-89ab01cd",
            "plan-abc123_4\x00", "plan-abc123_4\n", "plan-abc123_4 ", " plan-abc123_4",
            "PLAN-abc123_4", "plan-ABC123_4", "other-abc123_4", "plan-abc1234", "plan-abc123456",
            "", ".", "..", None, 12, True, {}, [], Path("plan-abc123_4"),
        )
        for value in invalid:
            with self.subTest(value=value):
                with patch.object(paths.Path, "resolve", side_effect=AssertionError("Untrusted path reached filesystem")), \
                     patch.object(paths.Path, "lstat", side_effect=AssertionError("Untrusted path reached filesystem")), \
                     patch.object(paths.Path, "stat", side_effect=AssertionError("Untrusted path reached filesystem")):
                    with self.assertRaises((ValueError, TypeError)):
                        paths.artifact_directory(self.artifacts, value)

    def test_missing_artifact_is_not_created(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            paths.artifact_directory(self.artifacts, "plan-abc123_4")
        self.assertFalse((self.artifacts / "plan-abc123_4").exists())

    def test_file_is_not_an_artifact_directory(self):
        (self.artifacts / "plan-abc123_4").write_text("{}", encoding="utf-8")
        with self.assertRaises((ValueError, NotADirectoryError)):
            paths.artifact_directory(self.artifacts, "plan-abc123_4")

    def test_linked_artifact_directory_is_rejected(self):
        target = self.root / "outside"
        target.mkdir()
        link = self.artifacts / "plan-abc123_4"
        self.symlink(link, target, directory=True)
        with self.assertRaises(ValueError):
            paths.artifact_directory(self.artifacts, link.name)

    def test_link_to_another_artifact_inside_root_is_rejected(self):
        target = self.artifact("plan-00000000")
        link = self.artifacts / "plan-abc123_4"
        self.symlink(link, target, directory=True)
        with self.assertRaises(ValueError):
            paths.artifact_directory(self.artifacts, link.name)

    def test_linked_json_is_rejected(self):
        directory = self.artifact()
        target = self.root / "private.json"
        target.write_text("{}", encoding="utf-8")
        (directory / "plan.json").unlink()
        self.symlink(directory / "plan.json", target)
        with self.assertRaises(ValueError):
            paths.artifact_directory(self.artifacts, directory.name)

    def test_linked_numpy_array_is_rejected(self):
        directory = self.artifact("semantic-0123_ab_")
        target = self.root / "private.npy"
        target.write_bytes(b"not read by path validation")
        self.symlink(directory / "semantic.npy", target)
        with self.assertRaises(ValueError):
            paths.artifact_directory(self.artifacts, directory.name)

    def test_link_metadata_on_stage_files_is_rejected(self):
        directory = self.artifact()
        lstat = Path.lstat
        for filename in ("plan.json", "semantic.npy", "latent.npy", "audio.npy"):
            with self.subTest(filename=filename):
                linked = directory / filename
                linked.touch(exist_ok=True)

                def link_stat(path, *args, **kwargs):
                    if path == linked:
                        return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
                    return lstat(path, *args, **kwargs)

                with patch.object(paths.Path, "lstat", link_stat):
                    with self.assertRaises(ValueError):
                        paths.artifact_directory(self.artifacts, directory.name)

    def test_child_directory_link_is_rejected(self):
        directory = self.artifact()
        target = self.root / "outside"
        target.mkdir()
        self.symlink(directory / "contents", target, directory=True)
        with self.assertRaises(ValueError):
            paths.artifact_directory(self.artifacts, directory.name)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_junction_artifact_is_rejected(self):
        target = self.root / "outside"
        target.mkdir()
        junction = self.artifacts / "plan-abc123_4"
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                                 capture_output=True, check=False)
        if created.returncode:
            self.skipTest("Cannot create a junction in this environment")
        try:
            with self.assertRaises(ValueError):
                paths.artifact_directory(self.artifacts, junction.name)
        finally:
            os.rmdir(junction)
        self.assertTrue(target.is_dir())


class OfficialModelPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models_root = self.root / "models"
        self.index_file = self.models_root / "diffusion_models" / "YuE2" / "official_models.json"
        self.index_file.parent.mkdir(parents=True)
        self.models = {
            "yue2": {"directory": "diffusion_models/YuE2/YuE2-3B"},
            "vae": {"directory": "vae/YuE2/YuE2-Vae"},
            "vae_legacy": {"directory": "vae/YuE2/YuE2-Vae-legacy"},
            "sheetsage2": {"directory": "audio_encoders/YuE2/SheetSage2"},
            "mert2": {"directory": "audio_encoders/YuE2/MERT-v2-FullSong"},
        }
        for entry in self.models.values():
            directory = self.models_root / entry["directory"]
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"fixture weights")
        self.write_index()

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self):
        self.index_file.write_text(json.dumps({"models": self.models}), encoding="utf-8")

    def test_allowlisted_keys_use_registered_local_directories(self):
        for key, entry in self.models.items():
            with self.subTest(key=key):
                expected = str((self.models_root / entry["directory"]).resolve())
                self.assertEqual(paths.model_directory(self.models_root, key), expected)

    def test_unknown_keys_are_rejected_before_filesystem_resolution(self):
        self.models["forged"] = self.models["yue2"].copy()
        self.write_index()
        for key in ("forged", "../yue2", "C:\\weights", "\\\\attacker.invalid\\share", "", None, [], {}):
            with self.subTest(key=key):
                with patch.object(paths.Path, "resolve", side_effect=AssertionError("Untrusted key reached filesystem")):
                    with self.assertRaises((ValueError, TypeError, FileNotFoundError)):
                        paths.model_directory(self.models_root, key)

    def test_forged_index_directories_are_rejected_before_resolution(self):
        resolve = Path.resolve

        def resolve_trusted(path, *args, **kwargs):
            if path not in (self.models_root, self.index_file):
                raise AssertionError("Untrusted model path reached filesystem resolution")
            return resolve(path, *args, **kwargs)

        invalid = (
            "/tmp/weights", "C:/weights", "C:\\weights", "C:weights", "\\weights",
            "\\\\attacker.invalid\\share", "//attacker.invalid/share", "\\\\?\\C:\\weights",
            "\\\\.\\pipe\\weights", "vae/YuE2/weights:stream", "../weights", "vae/../weights",
            "vae\\..\\weights", "vae/..\\weights", "vae\\../weights", "weights\x00", "", ".", "..",
            None, 12, [], {},
        )
        for directory in invalid:
            with self.subTest(directory=directory):
                self.models["yue2"]["directory"] = directory
                self.write_index()
                with patch.object(paths.Path, "resolve", resolve_trusted):
                    with self.assertRaises((ValueError, TypeError)):
                        paths.model_directory(self.models_root, "yue2")

    def test_config_and_exact_weight_file_are_required(self):
        directory = self.models_root / self.models["yue2"]["directory"]
        for filename in ("config.json", "model.safetensors"):
            with self.subTest(filename=filename):
                original = (directory / filename).read_bytes()
                (directory / filename).unlink()
                (directory / "other.safetensors").touch()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    paths.model_directory(self.models_root, "yue2")
                (directory / filename).write_bytes(original)

    def test_model_directory_cannot_escape_through_symlink(self):
        target = self.root / "outside"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").touch()
        link = self.models_root / "linked-model"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")
        self.models["yue2"]["directory"] = link.name
        self.write_index()
        with self.assertRaises(ValueError):
            paths.model_directory(self.models_root, "yue2")

    def test_required_model_files_cannot_be_symlinks(self):
        directory = self.models_root / self.models["yue2"]["directory"]
        for filename in ("config.json", "model.safetensors"):
            with self.subTest(filename=filename):
                original = (directory / filename).read_bytes()
                target = self.root / ("outside-" + filename)
                target.write_bytes(original)
                (directory / filename).unlink()
                try:
                    (directory / filename).symlink_to(target)
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f"Symlinks unavailable: {exc}")
                try:
                    with self.assertRaises(ValueError):
                        paths.model_directory(self.models_root, "yue2")
                finally:
                    (directory / filename).unlink()
                    (directory / filename).write_bytes(original)


if __name__ == "__main__":
    unittest.main()
