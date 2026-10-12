"""Verify installer commands without downloading or changing Python environments."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_setup(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_setup("setup_runtime")
sheetsage = load_setup("setup_sheetsage")


class PlatformSetup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="YuE2 setup with spaces ")
        self.root = Path(self.temp.name)
        vendor = self.root / "runtime" / "vendor"
        vendor.mkdir(parents=True)
        source = b"official source fixture\n"
        (vendor / "source.py").write_bytes(source)
        (self.root / "runtime" / "official-source.json").write_text(
            json.dumps({"files": {"source.py": hashlib.sha256(source).hexdigest()}}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, module, system, args=(), *, machine="arm64", uv="/existing uv/bin/uv", failure=None):
        calls = []
        runner_name = "command" if module is runtime else "run"
        root_name = "ROOT" if module is runtime else "NODE_ROOT"

        def record(command, **kwargs):
            command = list(map(str, command))
            calls.append((command, kwargs))
            if failure and "venv" in command and command[:2] == [module.sys.executable, "-m"]:
                raise subprocess.CalledProcessError(1, command)

        with patch.object(module, root_name, self.root), patch.object(module.sys, "platform", system), \
                patch.object(module.platform, "machine", return_value=machine), \
                patch.object(module.sys, "argv", ["setup", *args]), \
                patch.object(module.shutil, "which", return_value=uv), \
                patch.object(module, runner_name, record), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module.main()
        return calls

    def torch_command(self, calls):
        return next(command for command, _ in calls if any(value.startswith("torch==") for value in command))

    def test_default_wheels_match_each_platform_and_preserve_pins(self):
        for module, version, cuda in ((runtime, "2.10.0", "cu128"), (sheetsage, "2.8.0", "cu126")):
            for system in ("win32", "linux", "darwin"):
                with self.subTest(installer=module.__name__, system=system):
                    calls = self.invoke(module, system)
                    command = self.torch_command(calls)
                    self.assertIn("torch==" + version, command)
                    expected_index = "https://pypi.org/simple" if system == "darwin" else "https://download.pytorch.org/whl/" + cuda
                    self.assertEqual(command[command.index("--index-url") + 1], expected_index)
                    python = command[command.index("--python") + 1]
                    self.assertEqual(python, str(self.root / (".venv" if module is runtime else ".venv-sheetsage") / ("Scripts/python.exe" if system == "win32" else "bin/python")))
                    if module is sheetsage:
                        self.assertIn("torchaudio==2.8.0", command)

    def test_cpu_uses_pypi_on_mac_and_cpu_index_elsewhere(self):
        for module in (runtime, sheetsage):
            for system in ("win32", "linux", "darwin"):
                with self.subTest(installer=module.__name__, system=system):
                    calls = self.invoke(module, system, ("--cpu", "--torch-index-url", "https://custom.invalid/unused"))
                    command = self.torch_command(calls)
                    expected_index = "https://pypi.org/simple" if system == "darwin" else "https://download.pytorch.org/whl/cpu"
                    self.assertEqual(command[command.index("--index-url") + 1], expected_index)

    def test_custom_torch_index_is_explicit(self):
        for module in (runtime, sheetsage):
            calls = self.invoke(module, "linux", ("--torch-index-url", "https://custom.invalid/wheels"))
            self.assertIn("https://custom.invalid/wheels", self.torch_command(calls))

    def test_unix_uv_bootstrap_never_installs_into_system_python(self):
        for module in (runtime, sheetsage):
            for system in ("linux", "darwin"):
                with self.subTest(installer=module.__name__, system=system):
                    calls = self.invoke(module, system, uv=None)
                    bootstrap = self.root / "runtime" / ".tools" / ("uv-bootstrap-yue2" if module is runtime else "uv-bootstrap-sheetsage")
                    self.assertEqual(calls[0][0], [module.sys.executable, "-m", "venv", str(bootstrap)])
                    self.assertEqual(calls[1][0], [str(bootstrap / "bin" / "python"), "-m", "pip", "install", "uv==0.12.17"])
                    self.assertFalse(any("--target" in command or "--break-system-packages" in command for command, _ in calls))
                    self.assertTrue(any(command[:3] == [str(bootstrap / "bin" / "python"), "-m", "uv"] for command, _ in calls[2:]))

    def test_windows_embedded_python_bootstrap_is_preserved(self):
        for module in (runtime, sheetsage):
            calls = self.invoke(module, "win32", uv=None)
            self.assertEqual(calls[0][0], [module.sys.executable, "-m", "pip", "install", "--target", str(self.root / "runtime" / ".tools"), "uv==0.12.17"])
            self.assertFalse(any(command[:3] == [module.sys.executable, "-m", "venv"] for command, _ in calls))

    def test_missing_venv_support_has_actionable_error(self):
        for module in (runtime, sheetsage):
            with self.subTest(installer=module.__name__):
                with self.assertRaisesRegex(RuntimeError, "python3-venv.*--uv"):
                    self.invoke(module, "linux", uv=None, failure=True)

    def test_fast_backend_is_only_allowed_on_gpu_linux(self):
        for system, args in (("win32", ("--fast",)), ("darwin", ("--fast",)), ("linux", ("--fast", "--cpu"))):
            with self.subTest(system=system, args=args), self.assertRaises(SystemExit) as error:
                self.invoke(runtime, system, args)
            self.assertEqual(error.exception.code, 2)
        calls = self.invoke(runtime, "linux", ("--fast",))
        self.assertTrue(any(str(self.root / "runtime" / "vendor") + "[fast]" in command for command, _ in calls))

    def test_intel_mac_is_rejected_before_installing(self):
        for module in (runtime, sheetsage):
            for args in ((), ("--cpu",)):
                with self.subTest(installer=module.__name__, args=args), self.assertRaises(SystemExit) as error:
                    self.invoke(module, "darwin", args, machine="x86_64")
                self.assertEqual(error.exception.code, 2)

    def test_existing_uv_keeps_managed_python_inside_the_node(self):
        for module in (runtime, sheetsage):
            calls = self.invoke(module, "linux")
            install, kwargs = next((command, kwargs) for command, kwargs in calls if command[1:3] == ["python", "install"])
            self.assertIn("--no-bin", install)
            self.assertIn("--no-registry", install)
            expected = self.root / "runtime" / ".tools" / ("python-yue2" if module is runtime else "python-sheetsage")
            self.assertEqual(kwargs["env"]["UV_PYTHON_INSTALL_DIR"], str(expected))
            self.assertEqual(kwargs["env"]["HF_HUB_DISABLE_TELEMETRY"], "1")


if __name__ == "__main__":
    unittest.main()
