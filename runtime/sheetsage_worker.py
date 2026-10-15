"""Run the official SheetSage2 transcription pipeline in its own environment."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from paths import artifact_directory, model_directory

NODE_ROOT = Path(__file__).resolve().parents[1]
JSON_OUTPUT = sys.stdout


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), file=JSON_OUTPUT, flush=True)


def check_cancelled(cancel_file):
    if cancel_file and Path(cancel_file).exists():
        raise InterruptedError("SheetSage2 transcription cancelled.")


def verify_model(directory, model_name):
    manifest = json.loads((NODE_ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["models"][model_name]["files"]:
        path = directory / item["filename"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise FileNotFoundError(f"Incomplete official {model_name} model: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Official {model_name} file failed integrity verification: {path}")


def transcribe(payload):
    if payload.get("model_key") != "sheetsage2" or payload.get("mert_key") != "mert2":
        raise ValueError("Select the official SheetSage2 transcription models")
    artifact_dir = artifact_directory(payload["artifact_root"], payload.get("artifact_id"))
    cancel_file = artifact_dir / ".cancel"
    check_cancelled(cancel_file)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    import torch
    from transformers import AutoModel

    model_dir = Path(model_directory(payload["models_root"], payload["model_key"]))
    mert_dir = Path(model_directory(payload["models_root"], payload["mert_key"]))
    audio_path = Path(payload["audio_path"]).resolve()
    check_cancelled(cancel_file)
    emit("progress", phase="Verifying SheetSage2 models", completed=0, total=2, status="running")
    verify_model(model_dir, "sheetsage2")
    check_cancelled(cancel_file)
    emit("progress", phase="Verifying SheetSage2 models", completed=1, total=2, status="running")
    verify_model(mert_dir, "mert2")
    check_cancelled(cancel_file)
    emit("progress", phase="Verifying SheetSage2 models", completed=2, total=2, status="completed")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Source music does not exist: {audio_path}")
    device = payload.get("device", "cuda:0")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    emit("progress", phase="Loading official SheetSage2", completed=0, total=None, status="running")
    model = AutoModel.from_pretrained(
        str(model_dir), base_model_path=str(mert_dir), trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(device)
    check_cancelled(cancel_file)
    emit("progress", phase="Loading official SheetSage2", completed=1, total=1, status="completed")

    def progress(value):
        check_cancelled(cancel_file)
        stage = value["stage"]
        if stage == "decoding":
            completed, total = value.get("tokens", 0), model.max_output_seq_len
            unit = "tokens"
        elif "windows" in value:
            completed = value["window"] if stage == "window_complete" else value["window"] - 1
            total = value["windows"]
            unit = "windows"
        else:
            completed, total = 0, None
            unit = "it"
        emit("progress", phase=f"SheetSage2 {stage}", completed=completed,
             total=total, total_is_budget=stage == "decoding", unit=unit,
             status="completed" if stage == "window_complete" else "running")

    keys = (
        "melody_only", "dtype", "preset", "max_seconds", "overlap_seconds",
        "lookahead_seconds", "prompts", "export_logits", "export_scores",
        "export_embeddings", "output_hidden_states", "render_audio", "render_score",
        "render_parts",
    )
    options = {key: payload[key] for key in keys if key in payload}
    options.setdefault("melody_only", True)
    options.setdefault("dtype", "bf16")
    options.setdefault("preset", "default")
    options["progress"] = progress
    result = model.transcribe(str(audio_path), output_dir=str(artifact_dir), **options)
    check_cancelled(cancel_file)
    score_path = artifact_dir / "score.abc"
    if result.get("abc_error") or not score_path.is_file():
        raise RuntimeError(f"SheetSage2 did not produce a usable score: {result.get('abc_error')}")
    abc = score_path.read_text(encoding="utf-8")
    if not abc.strip():
        raise RuntimeError("SheetSage2 produced an empty ABC score.")
    # Official output_dir mode persists metadata without binary MIDI or tensor payloads.
    metadata = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    metadata["model_directory"] = str(model_dir)
    metadata["mert_directory"] = str(mert_dir)
    metadata["device"] = device
    emit("progress", phase="SheetSage2 transcription", completed=1, total=1, status="completed")
    return {"artifact_id": artifact_dir.name, "abc": abc, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8")) if args.request else json.load(sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            result = transcribe(payload)
    except InterruptedError as exc:
        emit("progress", phase="SheetSage2 transcription", completed=0, total=None, status="cancelled")
        emit("error", error=str(exc), error_type=type(exc).__name__)
        return 1
    except Exception as exc:
        emit("error", error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc(file=sys.stderr)
        return 1
    emit("result", result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
