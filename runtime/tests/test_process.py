from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("isolated_backend", ROOT / "backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class ChildProcessLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        python_path = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python_path.parent.mkdir(parents=True)
        python_path.touch()
        (self.root / "runtime").mkdir()
        self.worker = self.root / "runtime" / "worker.py"
        self.spawned = []
        self.real_popen = subprocess.Popen

    def tearDown(self):
        for process in self.spawned:
            if process.poll() is None:
                process.kill()
                process.wait()
        self.temp.cleanup()

    def launch(self, args, **kwargs):
        if args[0] == "taskkill":
            return self.real_popen(args, **kwargs)
        process = self.real_popen([sys.executable, *args[1:]], **kwargs)
        self.spawned.append(process)
        return process

    def test_results_and_native_counts_reach_caller(self):
        self.worker.write_text(
            "import json\nprint(json.dumps({'event':'progress','phase':'Synthesizing audio','completed':1,'total':32,'status':'running'}),flush=True)\n"
            "print(json.dumps({'event':'result','result':{'sample_rate':48000}}),flush=True)\n", encoding="utf-8")
        progress = []
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch):
            result = backend.run("generate", {"artifact_root": str(self.root / "artifacts")}, on_progress=progress.append)
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(progress[0]["completed"], 1)
        self.assertEqual(progress[0]["total"], 32)
        self.assertFalse(list((self.root / "artifacts").rglob("*.request.json")))

    def test_cancellation_kills_unresponsive_child_and_removes_request(self):
        self.worker.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        started = time.monotonic()
        def cancel():
            return time.monotonic() - started > 0.2
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch):
            with self.assertRaises(InterruptedError):
                backend.run("generate", {"artifact_root": str(self.root / "artifacts")}, check_cancelled=cancel)
        self.assertIsNotNone(self.spawned[0].poll())
        self.assertLess(time.monotonic() - started, 8)
        self.assertFalse(list((self.root / "artifacts").rglob("*.request.json")))

    def test_failure_preserves_actionable_stderr(self):
        self.worker.write_text("import sys\nprint('Official model config missing',file=sys.stderr)\nsys.exit(7)\n", encoding="utf-8")
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch):
            with self.assertRaisesRegex(RuntimeError, "Official model config missing"):
                backend.run("generate", {"artifact_root": str(self.root / "artifacts")})

    def test_transient_request_is_outside_official_artifact_manifest(self):
        self.worker.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "request=Path(sys.argv[2]); payload=json.loads(request.read_text())\n"
            "files=[p.name for p in Path(payload['artifact_dir']).rglob('*') if p.is_file()]\n"
            "print(json.dumps({'event':'result','result':{'files':files,'request_outside':request.parent!=Path(payload['artifact_dir'])}}),flush=True)\n",
            encoding="utf-8")
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch):
            result = backend.run("generate", {"artifact_root": str(self.root / "artifacts")})
        self.assertTrue(result["request_outside"])
        self.assertFalse(any(name.endswith(".request.json") for name in result["files"]))

    def test_console_counts_real_work_and_preserves_gui_events(self):
        events = [
            {"event": "progress", "phase": "Loading model", "completed": 0, "total": None, "status": "running"},
            {"event": "progress", "phase": "Loading model", "completed": 0, "total": None, "status": "completed"},
            {"event": "progress", "phase": "Planning score", "completed": 0, "total": 4096,
             "total_is_budget": True, "token_budget": 4096, "unit": "tokens", "status": "running"},
            {"event": "progress", "phase": "Planning score", "completed": 3, "total": 4096,
             "total_is_budget": True, "token_budget": 4096, "unit": "tokens", "status": "running"},
            {"event": "progress", "phase": "Planning score", "completed": 3, "total": 3,
             "total_is_budget": False, "token_budget": 4096, "unit": "tokens", "status": "completed"},
            {"event": "progress", "phase": "Synthesizing audio", "completed": 0, "total": None,
             "unit": "steps", "status": "running"},
            {"event": "progress", "phase": "Synthesizing audio", "completed": 16, "total": 32,
             "unit": "steps", "status": "running"},
            {"event": "progress", "phase": "Synthesizing audio", "completed": 32, "total": 32,
             "unit": "steps", "status": "completed"},
        ]
        self.worker.write_text(
            "import json,time\n"
            f"events = {events!r}\n"
            "for event in events:\n print(json.dumps(event),flush=True)\n time.sleep(0.3)\n"
            "print(json.dumps({'event':'result','result':{'sample_rate':48000}}),flush=True)\n",
            encoding="utf-8")
        output, gui = io.StringIO(), []
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch), redirect_stderr(output):
            result = backend.run("generate", {"artifact_root": str(self.root / "artifacts")}, on_progress=gui.append)
        rendered = output.getvalue()
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(gui, events)
        self.assertIn("YuE2", rendered)
        self.assertIn("Loading model", rendered)
        self.assertIn("Planning score", rendered)
        self.assertRegex(rendered, r"3\s*tokens")
        self.assertNotIn("/4096", rendered)
        self.assertIn("32/32", rendered)
        self.assertIn("100%", rendered)
        self.assertTrue(rendered.endswith("\n"))

    def test_console_progress_without_gui_callback_and_truncation(self):
        self.worker.write_text(
            "import json,time\n"
            "print(json.dumps({'event':'progress','phase':'Generating song','completed':5,'total':9000,"
            "'total_is_budget':True,'token_budget':9000,'unit':'tokens','status':'running'}),flush=True)\n"
            "time.sleep(0.2)\n"
            "print(json.dumps({'event':'progress','phase':'Generating song','completed':5,'total':5,"
            "'total_is_budget':False,'token_budget':9000,'unit':'tokens','status':'truncated'}),flush=True)\n"
            "print(json.dumps({'event':'result','result':{'truncated':True}}),flush=True)\n", encoding="utf-8")
        output = io.StringIO()
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch), redirect_stderr(output):
            result = backend.run("generate", {"artifact_root": str(self.root / "artifacts")})
        rendered = output.getvalue()
        self.assertTrue(result["truncated"])
        self.assertIn("Generating song", rendered)
        self.assertNotIn("/9000", rendered)
        self.assertRegex(rendered.lower(), r"truncated|limit reached|generation limit")
        self.assertTrue(rendered.endswith("\n"))

    def test_failure_closes_console_without_filling_remaining_steps(self):
        self.worker.write_text(
            "import json,sys\n"
            "print(json.dumps({'event':'progress','phase':'Synthesizing audio','completed':7,'total':32,"
            "'unit':'steps','status':'running'}),flush=True)\n"
            "print('Official model failed',file=sys.stderr)\nsys.exit(7)\n",
            encoding="utf-8")
        output = io.StringIO()
        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch), redirect_stderr(output):
            with self.assertRaisesRegex(RuntimeError, "Official model failed"):
                backend.run("generate", {"artifact_root": str(self.root / "artifacts")})
        rendered = output.getvalue()
        self.assertRegex(rendered.lower(), "failed")
        self.assertIn("7/32", rendered)
        self.assertNotIn("32/32", rendered)
        self.assertTrue(rendered.endswith("\n"))
        self.assertIsNotNone(self.spawned[0].poll())

    def test_comfy_base_exception_cancels_child_and_closes_console(self):
        class InterruptProcessingException(BaseException):
            pass

        self.worker.write_text(
            "import json,time\n"
            "print(json.dumps({'event':'progress','phase':'Generating song','completed':2,'total':9000,"
            "'total_is_budget':True,'unit':'tokens','status':'running'}),flush=True)\n"
            "time.sleep(60)\n", encoding="utf-8")
        output, progress = io.StringIO(), []

        def cancel():
            if progress:
                raise InterruptProcessingException()

        with patch.object(backend, "ROOT", self.root), patch.object(backend.subprocess, "Popen", self.launch), redirect_stderr(output):
            with self.assertRaises(InterruptProcessingException):
                backend.run("generate", {"artifact_root": str(self.root / "artifacts")},
                            on_progress=progress.append, check_cancelled=cancel)
        rendered = output.getvalue()
        self.assertRegex(rendered.lower(), "cancelled")
        self.assertNotIn("/9000", rendered)
        self.assertTrue(rendered.endswith("\n"))
        self.assertIsNotNone(self.spawned[0].poll())
        self.assertFalse(list((self.root / "artifacts").rglob("*.request.json")))


if __name__ == "__main__":
    unittest.main()
