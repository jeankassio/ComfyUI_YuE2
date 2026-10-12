"""Install the official SheetSage2 requirements outside ComfyUI's Python."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


NODE_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = (
    "transformers==4.45.2", "huggingface-hub==0.36.0", "safetensors==0.5.3",
    "numpy==1.24.3", "scipy==1.13.1", "mir_eval==0.8.2", "pretty_midi==0.2.10",
    "mido==1.3.3", "setuptools==78.1.1",
)


def run(command, *, env=None):
    print("Running: " + subprocess.list2cmdline([str(value) for value in command]), flush=True)
    subprocess.run(command, check=True, cwd=NODE_ROOT, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", help="Path to an existing uv executable.")
    parser.add_argument("--python", choices=("3.10", "3.11"), default="3.11", help="Separate official runtime Python.")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu126")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    uv_path = args.uv or shutil.which("uv")
    tools = NODE_ROOT / "runtime" / ".tools"
    setup_env = os.environ.copy()
    setup_env.update({
        "HF_HUB_DISABLE_TELEMETRY": "1", "UV_NO_PROGRESS": "1",
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_INSTALL_DIR": str(tools / "python-sheetsage"),
    })
    if uv_path:
        uv = [str(uv_path)]
    else:
        run([sys.executable, "-m", "pip", "install", "--target", str(tools), "uv==0.12.17"])
        setup_env["PYTHONPATH"] = str(tools)
        uv = [sys.executable, "-m", "uv"]
    environment = NODE_ROOT / ".venv-sheetsage"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run(uv + ["python", "install", args.python, "--no-bin", "--no-registry"], env=setup_env)
    if not python.is_file():
        run(uv + ["venv", "--managed-python", "--python", args.python, str(environment)], env=setup_env)
    index_url = "https://download.pytorch.org/whl/cpu" if args.cpu else args.torch_index_url
    run(uv + ["pip", "install", "--python", str(python), "torch==2.8.0", "torchaudio==2.8.0", "--index-url", index_url], env=setup_env)
    run(uv + ["pip", "install", "--python", str(python), "--index-url", "https://pypi.org/simple", *REQUIREMENTS], env=setup_env)
    run([str(python), "-c", "import torch, torchaudio, transformers; print('SheetSage2 ready:', torch.__version__, torchaudio.__version__, transformers.__version__, 'CUDA:', torch.cuda.is_available())"], env=setup_env)
    print("SheetSage2 runtime is ready. ComfyUI's Python environment was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
