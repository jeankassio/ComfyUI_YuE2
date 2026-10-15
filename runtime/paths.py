"""Resolve node references inside server-owned artifact and model directories."""
import json
from pathlib import Path
import re
import stat


ARTIFACT_NAME = re.compile(r"(?:generate|plan|semantic|synthesize|decode|transcribe)-[a-z0-9_]{8}")
MODEL_KEYS = {"yue2", "vae", "vae_legacy", "sheetsage2", "mert2"}


def _reject_link(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"Linked YuE2 paths are not allowed: {path.name}")
    return info


def artifact_directory(root, artifact_id):
    if not isinstance(artifact_id, str) or ARTIFACT_NAME.fullmatch(artifact_id) is None:
        raise ValueError("Invalid YuE2 artifact ID; connect a freshly executed YuE2 stage")
    root = Path(root).resolve()
    directory = root / artifact_id
    info = _reject_link(directory)
    if not stat.S_ISDIR(info.st_mode) or not directory.resolve().is_relative_to(root):
        raise ValueError("YuE2 artifacts must stay inside the ComfyUI temporary directory")
    for child in directory.iterdir():
        _reject_link(child)
    return directory


def _model_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or "\x00" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Invalid registered YuE2 model path")
    path = root
    for part in relative.split("/"):
        path = path / part
        _reject_link(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError("YuE2 models must stay inside the ComfyUI models directory")
    return path


def model_directory(models_root, key):
    if not isinstance(key, str) or key not in MODEL_KEYS:
        raise ValueError("Select a registered official YuE2 model")
    root = Path(models_root).resolve()
    try:
        index = _model_path(root, "diffusion_models/YuE2/official_models.json")
    except FileNotFoundError as exc:
        raise FileNotFoundError("YuE2 official models are not adapted yet. Run the model download and conversion launchers for your platform.") from exc
    models = json.loads(index.read_text(encoding="utf-8"))["models"]
    if key not in models:
        raise FileNotFoundError(f"Official model '{key}' is missing. Download it and run the model conversion launcher for your platform.")
    directory = _model_path(root, models[key]["directory"])
    for filename in ("config.json", "model.safetensors"):
        if not stat.S_ISREG(_reject_link(directory / filename).st_mode):
            raise ValueError(f"Invalid official model file: {key}/{filename}")
    for child in directory.iterdir():
        _reject_link(child)
    return str(directory)
