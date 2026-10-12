"""Adapt verified official Hugging Face directories to this ComfyUI node's layout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

PLUGIN = Path(__file__).resolve().parents[1]
DESTINATIONS = {
    "yue2": "diffusion_models/YuE2/YuE2-3B",
    "vae": "vae/YuE2/YuE2-Vae",
    "vae_legacy": "vae/YuE2/YuE2-Vae-legacy",
    "sheetsage2": "audio_encoders/YuE2/SheetSage2",
    "mert2": "audio_encoders/YuE2/MERT-v2-FullSong",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt(models_dir):
    manifest = json.loads((PLUGIN / "model_manifest.json").read_text(encoding="utf-8"))
    models_dir = models_dir.resolve()
    converted = {}
    for key, relative in DESTINATIONS.items():
        model = manifest["models"][key]
        source = models_dir / model["directory"]
        target = models_dir / relative
        if not source.is_dir():
            if key in ("yue2", "vae"):
                raise FileNotFoundError(f"Run download-models.bat first: {source}")
            continue
        for artifact in model["files"]:
            src = source / artifact["filename"]
            dst = target / artifact["filename"]
            if not src.resolve().is_relative_to(models_dir) or not dst.resolve().is_relative_to(models_dir):
                raise ValueError("Model artifact must stay inside the models directory")
            if not src.is_file() or src.stat().st_size != artifact["bytes"] or sha256(src) != artifact["sha256"]:
                raise ValueError(f"Official artifact integrity check failed: {src}")
            if src.resolve() == dst.resolve():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if dst.stat().st_size == artifact["bytes"] and sha256(dst) == artifact["sha256"]:
                    continue
                raise FileExistsError(f"Refusing to overwrite a different model: {dst}")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        converted[key] = {"directory": relative, "repository": model["repository"], "revision": model["revision"]}
        print(f"Adapted official {model['repository']} -> {target}", flush=True)
    tokenizer = models_dir / DESTINATIONS["yue2"] / "qwen.tiktoken"
    text_encoder = models_dir / "text_encoders/YuE2/qwen.tiktoken"
    text_encoder.parent.mkdir(parents=True, exist_ok=True)
    if not text_encoder.exists():
        try:
            os.link(tokenizer, text_encoder)
        except OSError:
            shutil.copy2(tokenizer, text_encoder)
    elif sha256(text_encoder) != sha256(tokenizer):
        raise FileExistsError("A different YuE2 tokenizer is already installed")
    bundle = models_dir / "diffusion_models/YuE2/official_models.json"
    bundle.write_text(json.dumps({"format_version": 1, "models": converted,
                                 "tokenizer": "text_encoders/YuE2/qwen.tiktoken",
                                 "conversion": "Bit-identical official weights; ComfyUI folder layout adapter, no quantization or tensor remapping."}, indent=2), encoding="utf-8")
    print(f"Model index: {bundle}", flush=True)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=PLUGIN.parent.parent / "models")
    args = parser.parse_args()
    adapt(args.models_dir)
