"""Install the pinned official engine without modifying ComfyUI's Python."""
from pathlib import Path
import argparse
import hashlib
import json
import os
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
    parser.add_argument("--cpu", action="store_true", help="Install CPU Torch instead of the CUDA 12.8 wheel")
    parser.add_argument("--fast", action="store_true", help="Install the official Linux vLLM optional backend")
    args = parser.parse_args()
    if args.fast and os.name == "nt":
        parser.error("The official vLLM backend supports Linux; omit --fast on Windows.")
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
    else:
        command([sys.executable, "-m", "pip", "install", "--target", tools, "uv==0.12.17"])
        setup_env["PYTHONPATH"] = str(tools)
        uv = [sys.executable, "-m", "uv"]
    command(uv + ["python", "install", "3.12", "--no-bin", "--no-registry"], env=setup_env)
    environment = ROOT / ".venv"
    if not environment.exists():
        command(uv + ["venv", "--managed-python", "--python", "3.12", str(environment)], env=setup_env)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    torch_index = "https://download.pytorch.org/whl/cpu" if args.cpu else "https://download.pytorch.org/whl/cu128"
    command(uv + ["pip", "install", "--python", str(python), "--index-url", torch_index, "torch==2.10.0"], env=setup_env)
    package = str(vendor) + ("[fast]" if args.fast else "")
    command(uv + ["pip", "install", "--python", str(python), "--index-url", "https://pypi.org/simple", package], env=setup_env)
    command([python, "-c", "import torch, transformers, yue2; print('Official YuE2 ready. Torch', torch.__version__, 'Transformers', transformers.__version__, 'CUDA', torch.cuda.is_available())"], env=setup_env)
    print("Setup complete. Restart ComfyUI to load the nodes.")


if __name__ == "__main__":
    main()
