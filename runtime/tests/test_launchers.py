"""Exercise POSIX launchers without installing packages or downloading models."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SHELL = shutil.which("sh")
if not SHELL and os.name == "nt":
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    if git_bash.is_file():
        SHELL = str(git_bash)
LAUNCHERS = (
    "setup-runtime.sh", "convert-models.sh", "scripts/setup-runtime.sh",
    "scripts/setup-sheetsage.sh", "scripts/download-models.sh",
)


def shell_path(path):
    value = Path(path).as_posix()
    if os.name == "nt":
        return "/" + value[0].lower() + value[2:]
    return value


@unittest.skipUnless(SHELL, "A POSIX shell or Git Bash is required.")
class PosixLaunchers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yue2 launchers ")
        self.base = Path(self.temp.name)
        self.comfy = self.base / "ComfyUI with spaces"
        self.plugin = self.comfy / "custom_nodes/ComfyUI_YuE2"
        self.plugin.mkdir(parents=True)
        for name in (*LAUNCHERS, "scripts/shell-common.sh"):
            target = self.plugin / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / name).read_text(), encoding="utf-8", newline="\n")
        self.python(self.plugin / ".venv/bin/python", "plugin")

    def tearDown(self):
        self.temp.cleanup()

    def python(self, path, label, exit_code=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/bin/sh\nprintf '%s\\n' '" + label + "'\n"
            "printf '<%s>\\n' \"$@\"\nexit " + str(exit_code) + "\n",
            encoding="utf-8", newline="\n",
        )
        path.chmod(0o755)

    def run_launcher(self, name, *arguments):
        return subprocess.run(
            [SHELL, "-c", 'exec /bin/sh "$@"', "launcher-test", shell_path(self.plugin / name), *arguments],
            cwd=self.base, capture_output=True, text=True, check=False,
        )

    def test_shell_syntax(self):
        for name in (*LAUNCHERS, "scripts/shell-common.sh"):
            with self.subTest(name=name):
                result = subprocess.run([SHELL, "-n", shell_path(ROOT / name)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_and_conversion_preserve_arguments_and_arbitrary_cwd(self):
        targets = {
            "setup-runtime.sh": "setup_runtime.py",
            "scripts/setup-runtime.sh": "setup_runtime.py",
            "scripts/setup-sheetsage.sh": "setup_sheetsage.py",
            "convert-models.sh": "convert_official_models.py",
        }
        for name, target in targets.items():
            with self.subTest(name=name):
                result = self.run_launcher(name, "--models-dir", "models with spaces", "$(touch unexpected)")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [
                    "plugin", "<" + shell_path(self.plugin / "scripts" / target) + ">",
                    "<--models-dir>", "<models with spaces>", "<$(touch unexpected)>",
                ])
        self.assertFalse((self.base / "unexpected").exists())

    def test_setup_uses_comfy_environment_when_runtime_does_not_exist(self):
        (self.plugin / ".venv/bin/python").unlink()
        self.python(self.comfy / ".venv/bin/python", "comfy")
        result = self.run_launcher("setup-runtime.sh", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], "comfy")

    def test_setup_uses_comfy_venv_alternative(self):
        (self.plugin / ".venv/bin/python").unlink()
        self.python(self.comfy / "venv/bin/python", "comfy-venv")
        result = self.run_launcher("scripts/setup-sheetsage.sh", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], "comfy-venv")

    def test_download_has_one_explicit_action_and_preserves_arguments(self):
        for arguments, expected in (
            (("--models", "yue2"), ("--download", "--models", "yue2")),
            (("--check", "--models-dir", "my models"), ("--check", "--models-dir", "my models")),
            (("--download", "--models", "vae"), ("--download", "--models", "vae")),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_launcher("scripts/download-models.sh", *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[2:], ["<" + item + ">" for item in expected])

    def test_download_does_not_fall_back_to_comfy_or_system_python(self):
        (self.plugin / ".venv/bin/python").unlink()
        self.python(self.comfy / ".venv/bin/python", "must-not-run")
        result = self.run_launcher("scripts/download-models.sh", "--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Run sh setup-runtime.sh first", result.stderr)

    def test_launchers_preserve_python_exit_status(self):
        self.python(self.plugin / ".venv/bin/python", "plugin", exit_code=7)
        for name in LAUNCHERS:
            with self.subTest(name=name):
                result = self.run_launcher(name, "--check")
                self.assertEqual(result.returncode, 7, result.stderr)


if __name__ == "__main__":
    unittest.main()
