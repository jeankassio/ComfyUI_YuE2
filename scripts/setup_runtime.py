"""Install the pinned official engine without modifying ComfyUI's Python."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(args, **kwargs):
    print("Installing: " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, help="Path to an existing uv executable")
    parser.add_argument("--cpu", action="store_true", help="Install CPU Torch (macOS uses the standard PyPI wheel)")
    parser.add_argument("--torch-index-url", help="Override the Torch wheel index; ignored with --cpu")
    parser.add_argument("--fast", action="store_true", help="Install the official Linux vLLM optional backend")
    args = parser.parse_args()
    if sys.platform not in ("win32", "linux", "darwin"):
        parser.error("Setup supports Windows, Linux, and macOS.")
    if sys.platform == "darwin" and platform.machine().lower() not in ("arm64", "aarch64"):
        parser.error("The official Torch 2.10.0 pin has no Intel macOS wheel. Use an Apple Silicon Mac and an ARM64 Python, or Windows/Linux.")
    if args.fast and (sys.platform != "linux" or args.cpu):
        parser.error("The official vLLM backend requires Linux with a supported GPU; omit --fast on Windows, macOS, or CPU-only setups.")
    manifest = json.loads((ROOT / "runtime" / "official-source.json").read_text(encoding="utf-8"))
    vendor = ROOT / "runtime" / "vendor"
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((vendor / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"The vendored official source was changed: {name}")
    uv_path = args.uv or shutil.which("uv")
    tools = ROOT / "runtime" / ".tools"
    setup_env = os.environ.copy()
    setup_env.update({
        "HF_HUB_DISABLE_TELEMETRY": "1", "UV_NO_PROGRESS": "1", "UV_LINK_MODE": "copy",
        "UV_PYTHON_INSTALL_DIR": str(tools / "python-yue2"),
    })
    if uv_path:
        uv = [str(uv_path)]
    elif sys.platform == "win32":
        command([sys.executable, "-m", "pip", "install", "--target", tools, "uv==0.12.17"])
        setup_env["PYTHONPATH"] = str(tools)
        uv = [sys.executable, "-m", "uv"]
    else:
        bootstrap = tools / "uv-bootstrap-yue2"
        bootstrap_python = bootstrap / "bin" / "python"
        if not bootstrap_python.is_file():
            try:
                command([sys.executable, "-m", "venv", str(bootstrap)])
            except subprocess.CalledProcessError as error:
                raise RuntimeError("A private uv bootstrap needs Python's venv support. Install python3-venv on Debian/Ubuntu, or install uv separately and retry with --uv /path/to/uv.") from error
        command([bootstrap_python, "-m", "pip", "install", "uv==0.12.17"])
        uv = [str(bootstrap_python), "-m", "uv"]
    command(uv + ["python", "install", "3.12", "--no-bin", "--no-registry"], env=setup_env)
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        command(uv + ["venv", "--managed-python", "--python", "3.12", str(environment)], env=setup_env)
    if sys.platform == "darwin":
        torch_index = args.torch_index_url if not args.cpu and args.torch_index_url else "https://pypi.org/simple"
    else:
        torch_index = "https://download.pytorch.org/whl/cpu" if args.cpu else args.torch_index_url or "https://download.pytorch.org/whl/cu128"
    command(uv + ["pip", "install", "--python", str(python), "--index-url", torch_index, "torch==2.10.0"], env=setup_env)
    package = str(vendor) + ("[fast]" if args.fast else "")
    command(uv + ["pip", "install", "--python", str(python), "--index-url", "https://pypi.org/simple", package], env=setup_env)
    command([python, "-c", "import torch, transformers, yue2; print('Official YuE2 ready. Torch', torch.__version__, 'Transformers', transformers.__version__, 'CUDA', torch.cuda.is_available(), 'MPS', torch.backends.mps.is_available())"], env=setup_env)
    print("Setup complete. Restart ComfyUI to load the nodes.")


if __name__ == "__main__":
    main()
