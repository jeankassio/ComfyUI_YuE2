"""Download explicitly selected, revision-pinned official YuE2 model directories."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

PLUGIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PLUGIN_DIR.parent.parent / "models"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, artifact):
    return path.is_file() and path.stat().st_size == artifact["bytes"] and sha256_file(path) == artifact["sha256"]


def main():
    manifest = json.loads((PLUGIN_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Explicitly authorize downloading the selected model files")
    action.add_argument("--check", action="store_true", help="Verify local files without making network requests")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--models", nargs="+", choices=list(manifest["models"]), default=manifest["default_models"])
    parser.add_argument("--report", type=Path, help="Write local verification results to this JSON file")
    args = parser.parse_args()
    models_dir = args.models_dir.resolve()
    if not args.download and not args.check:
        for model_id in args.models:
            model = manifest["models"][model_id]
            total = sum(artifact["bytes"] for artifact in model["files"])
            print(f"{model_id}: {model['repository']} @ {model['revision']} ({total / 1e9:.3f} GB)")
            print(f"  {models_dir / model['directory']}")
        print("Use --download to start; --check only verifies local files.")
        return 0

    if args.download:
        from huggingface_hub import hf_hub_download

    results = []
    for model_id in args.models:
        model = manifest["models"][model_id]
        destination = (models_dir / model["directory"]).resolve()
        if not destination.is_relative_to(models_dir):
            raise ValueError(f"Model directory escapes models-dir: {model_id}")
        print(f"[{model_id}] {model['repository']} @ {model['revision']}", flush=True)
        for artifact in model["files"]:
            target = (destination / artifact["filename"]).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Artifact path escapes model directory: {artifact['filename']}")
            valid = verify_file(target, artifact)
            if valid:
                print(f"  Verified: {artifact['filename']}", flush=True)
                status = "existing_verified"
            elif args.check:
                print(f"  Missing or invalid: {artifact['filename']}", flush=True)
                status = "missing_or_invalid"
            else:
                if target.exists():
                    raise ValueError(f"Existing file failed SHA256 verification; move it aside before retrying: {target}")
                print(f"  Downloading: {artifact['filename']} ({artifact['bytes'] / 1e6:.1f} MB)", flush=True)
                hf_hub_download(
                    repo_id=model["repository"], filename=artifact["filename"],
                    revision=model["revision"], local_dir=destination, token=False,
                )
                if not verify_file(target, artifact):
                    raise ValueError(f"Downloaded file failed SHA256 verification: {target}")
                print(f"  Verified download: {artifact['filename']}", flush=True)
                status = "downloaded_verified"
            results.append({"model": model_id, "path": str(target), "sha256": artifact["sha256"], "bytes": artifact["bytes"], "status": status})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"files": results}, indent=2) + "\n", encoding="utf-8")
    return 1 if any(item["status"] == "missing_or_invalid" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
