"""JSON-line adapter for the unmodified official YuE2 inference package."""
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch
from yue2.pipeline import SongResult, SymbolicPlan, YuE2Pipeline
from yue2.protocol import GenerationConfig, SongRequest
from yue2.storage import identity

from artifacts import load_latents, load_semantic, save_latents, save_semantic
from paths import artifact_directory, model_directory


JSON_STREAM = sys.stdout


def emit(event):
    JSON_STREAM.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    JSON_STREAM.flush()


class Stage:
    def __init__(self, label, total=None, total_is_budget=False, unit=None):
        self.label, self.total, self.completed = label, total, 0
        self.unit = unit
        self.total_is_budget = total_is_budget
        self.token_budget = total if total_is_budget else None
        self.last_report = 0
        self.finished = False

    def update(self, completed, total=None):
        self.completed = completed
        if total is not None:
            self.total = total
        now = time.monotonic()
        if now - self.last_report >= 0.15 or self.completed == self.total:
            self.report("running")
            self.last_report = now

    def advance(self, count=1):
        self.update(self.completed + count)

    def finish(self, status="completed"):
        if not self.finished:
            if self.total_is_budget:
                self.total = self.completed
                self.total_is_budget = False
            self.report(status)
            self.finished = True

    def report(self, status):
        emit({"event": "progress", "phase": self.label, "completed": self.completed,
              "total": self.total, "total_is_budget": self.total_is_budget,
              "token_budget": self.token_budget, "unit": self.unit, "status": status})


class ComfyProgressPipeline(YuE2Pipeline):
    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        # Only presentation is replaced. Native token, ODE and VAE callbacks drive counts.
        budget = {"Planning score": self.generation_config.abc.max_tokens,
                  "Generating song": self.generation_config.semantic.max_tokens}.get(label)
        stage = Stage(label, total if budget is None else budget, total_is_budget=budget is not None, unit=unit)
        stage.report("running")
        try:
            yield stage
        except InterruptedError:
            stage.finish("cancelled")
            raise
        except Exception:
            stage.finish("failed")
            raise
        else:
            stage.finish()


def check_models(payload):
    if payload.get("model_key") != "yue2" or payload.get("vae_key") not in ("vae", "vae_legacy"):
        raise ValueError("Select an official YuE2 model and decoder")
    model_dir = model_directory(payload["models_root"], payload["model_key"])
    vae_dir = model_directory(payload["models_root"], payload["vae_key"])
    if not (Path(model_dir) / "qwen.tiktoken").is_file():
        raise FileNotFoundError("The official qwen.tiktoken tokenizer is missing. Run the model download launcher for your platform.")
    if payload.get("backend") == "vllm" and sys.platform == "win32":
        raise RuntimeError("The official vLLM backend is Linux-only. Select torch or torch-eager on Windows.")
    device = payload.get("device", "auto")
    uses_cuda = str(device).startswith("cuda") or (device == "auto" and torch.cuda.is_available())
    if uses_cuda and payload.get("backend", "torch") == "torch" and payload.get("quantization", "none") == "none" and not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("This PyTorch build lacks the FlashAttention kernel required by the official CUDA-graph backend. Select torch-eager; it preserves official BF16 weights, sampling and acoustic generation.")
    return model_dir, vae_dir


def execute(payload):
    action = payload["action"]
    if action not in {"generate", "plan", "semantic", "synthesize", "decode"}:
        raise ValueError(f"Unsupported YuE2 action: {action}")
    output = artifact_directory(payload["artifact_root"], payload.get("artifact_id"))
    source = artifact_directory(payload["artifact_root"], payload.get("input_artifact")) if action in {"semantic", "synthesize", "decode"} else None
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_dir, vae_dir = check_models(payload)
    cancelled = lambda: (output / ".cancel").exists()
    if cancelled():
        raise InterruptedError("Cancelled before model loading")
    config = GenerationConfig.from_dict(payload.get("config", {}))
    options = {key: payload[key] for key in ("device", "memory_budget_gib", "backend", "quantization", "offload_ar", "vae_core_frames") if key in payload}
    result = {"artifact_id": output.name}
    start = time.perf_counter()
    with ComfyProgressPipeline.from_pretrained(
        model_dir, vae=vae_dir, local_files_only=True,
        generation_config=config, progress=True, verify_hashes=payload.get("verify_hashes", True), **options,
    ) as pipe:
        if action in {"generate", "plan"}:
            request = SongRequest(**payload["request"])
            plan = pipe.plan(request=request, cancelled=cancelled)
            plan.save(output)
        else:
            plan = SymbolicPlan.load(source)
            request = plan.request
        result["abc"] = plan.abc or ""
        if action == "plan":
            metadata = {"truncated": {"abc": plan.truncated}, "timing": plan.timing}
        else:
            if action in {"generate", "semantic"}:
                semantic = pipe.generate_semantic(plan, cancelled=cancelled)
            elif action in {"synthesize", "decode"}:
                semantic = load_semantic(source)
            else:
                raise ValueError(f"Unsupported YuE2 action: {action}")
            save_semantic(semantic, output)
            metadata = {"truncated": {"abc": plan.truncated, "semantic": semantic.truncated},
                        "semantic_frames": len(semantic.tokens), "duration_seconds": len(semantic.tokens) / 25,
                        "timing": {"abc": plan.timing, "semantic": semantic.timing}}
            if action in {"generate", "synthesize", "decode"}:
                if action == "decode":
                    latents, semantic, nar_timing = load_latents(source)
                else:
                    nar_start = time.perf_counter()
                    latents = pipe.synthesize(semantic, cancelled=cancelled)
                    nar_timing = {"nar_seconds": time.perf_counter() - nar_start}
                save_latents(latents, semantic, output, nar_timing)
                metadata["timing"].update(nar_timing)
                if action in {"generate", "decode"}:
                    if cancelled():
                        raise InterruptedError("Cancelled before audio decoding")
                    vae_start = time.perf_counter()
                    audio = pipe.decode(latents, full=payload.get("decode_full", False))
                    metadata["timing"].update({"vae_seconds": time.perf_counter() - vae_start,
                                              "e2e_seconds": time.perf_counter() - start,
                                              "load": dict(pipe.load_timing)})
                    effective = pipe.effective_config(request)
                    if payload.get("decode_full", False):
                        effective["vae_decode"] = "full"
                    request_identity = identity({"request": request.to_dict(), "config": effective, "weights": pipe.weights})
                    song = SongResult(audio, 48000, semantic, latents, effective, pipe.weights, metadata["timing"], request_identity)
                    official_result = song.save_artifacts(output)
                    np.save(output / "audio.npy", audio)
                    metadata.update({"official_result": official_result, "audio_seconds": len(audio) / 48000})
                    result["sample_rate"] = 48000
        settings = pipe.effective_config(request)
        if payload.get("decode_full", False):
            settings["vae_decode"] = "full"
        settings["verify_hashes"] = payload.get("verify_hashes", True)
        metadata.update({"request": request.to_dict(), "settings": settings,
                         "weights": pipe.weights, "runtime_seconds": time.perf_counter() - start})
        result["metadata"] = metadata
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        # Libraries may print diagnostics; the parent reserves stdout for JSON events.
        with redirect_stdout(sys.stderr):
            result = execute(payload)
        emit({"event": "result", "result": result})
        return 0
    except InterruptedError:
        emit({"event": "progress", "phase": "Cancelled", "completed": 0, "total": None, "status": "cancelled"})
        return 130
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
