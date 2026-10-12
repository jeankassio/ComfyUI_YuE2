"""Run the official engine in its isolated Python environment."""
from collections import deque
from pathlib import Path
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent


class _ConsoleProgress:
    def __init__(self):
        self.bar = None
        self.phase = None
        self.last_refresh = 0

    def update(self, event):
        phase = event["phase"]
        total = None if event.get("total_is_budget") else event.get("total")
        completed = event.get("completed", 0)
        if self.bar is None or phase != self.phase or completed < self.bar.n:
            self.close()
            self.phase = phase
            self.bar = tqdm(total=total, desc=f"YuE2 | {phase}", unit=event.get("unit") or "it",
                            dynamic_ncols=True, mininterval=0.25, leave=True)
        self.bar.total = total
        self.bar.update(completed - self.bar.n)
        if event.get("status") not in {None, "running"}:
            self.close(event["status"])

    def refresh(self):
        now = time.monotonic()
        if self.bar is not None and now - self.last_refresh >= 1:
            self.bar.refresh()
            self.last_refresh = now

    def close(self, status=None):
        if self.bar is not None:
            if status:
                self.bar.set_postfix_str(status, refresh=False)
            self.bar.close()
            self.bar = None


def _read_output(stream, messages):
    for line in stream:
        messages.put(line)
    messages.put(None)


def _read_errors(stream, errors):
    for line in stream:
        errors.append(line[-4096:].rstrip())


def _stop(process, cancel_file):
    cancel_file.touch()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()


def run(action, payload, on_progress=None, check_cancelled=None):
    if action not in {"generate", "plan", "semantic", "synthesize", "decode", "transcribe"}:
        raise ValueError(f"Unsupported YuE2 action: {action}")
    environment = ".venv-sheetsage" if action == "transcribe" else ".venv"
    python = ROOT / environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    worker = ROOT / "runtime" / ("sheetsage_worker.py" if action == "transcribe" else "worker.py")
    if not python.is_file():
        setup = "setup-sheetsage.bat" if action == "transcribe" else "setup-runtime.bat"
        raise RuntimeError(f"The isolated runtime is missing. Run {setup} in ComfyUI_YuE2 first.")
    artifact_root = Path(payload.get("artifact_root", Path(tempfile.gettempdir()) / "comfyui-yue2"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix=action + "-", dir=artifact_root))
    cancel_file = artifact_dir / ".cancel"
    request_file = artifact_root / (artifact_dir.name + ".request.json")
    request = {**payload, "action": action, "artifact_dir": str(artifact_dir), "cancel_file": str(cancel_file)}
    request_file.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    environment_vars = os.environ.copy()
    environment_vars.update({
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "runtime" / ".cache" / "inductor"),
    })
    # A node's embedded Python path must not leak into the isolated engine.
    environment_vars.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [str(python), str(worker), "--request", str(request_file)], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", env=environment_vars,
        creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    messages, errors = queue.Queue(), deque(maxlen=80)
    output_thread = threading.Thread(target=_read_output, args=(process.stdout, messages), daemon=True)
    error_thread = threading.Thread(target=_read_errors, args=(process.stderr, errors), daemon=True)
    output_thread.start()
    error_thread.start()
    result = None
    console = _ConsoleProgress()
    try:
        while True:
            if check_cancelled is not None and check_cancelled():
                raise InterruptedError("YuE2 generation cancelled")
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                console.refresh()
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(line[-4096:].rstrip())
                continue
            if event.get("event") == "progress":
                console.update(event)
                if on_progress is not None:
                    on_progress(event)
            elif event.get("event") == "result":
                result = event["result"]
        code = process.wait()
        error_thread.join(timeout=1)
        if code != 0 or result is None:
            detail = "\n".join(errors)
            if code == 130:
                raise InterruptedError("YuE2 generation cancelled")
            raise RuntimeError(f"Official YuE2 runtime failed (exit {code}).\n{detail}")
        console.close("completed")
        return result
    except BaseException as exc:
        cancelled = isinstance(exc, InterruptedError) or not isinstance(exc, Exception)
        console.close("cancelled" if cancelled else "failed")
        raise
    finally:
        console.close()
        if process.poll() is None:
            _stop(process, cancel_file)
        process.stdout.close()
        process.stderr.close()
        output_thread.join(timeout=1)
        error_thread.join(timeout=1)
        request_file.unlink(missing_ok=True)
        cancel_file.unlink(missing_ok=True)
