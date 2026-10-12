"""ComfyUI nodes backed exclusively by the official YuE2 Python pipeline."""

import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .backend import run

PLUGIN_DIR = Path(__file__).resolve().parent
CATEGORY = "audio/YuE2 Official"
DEFAULT_CONFIG = {
    "abc": {"temperature": .7, "top_p": .9, "top_k": 30, "repetition_penalty": 1.005,
            "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096},
    "semantic": {"temperature": 1., "top_p": .95, "top_k": 100, "repetition_penalty": 1.2,
                 "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000},
    "ode_steps": 32, "ode_method": "midpoint", "context": 24576,
}


def model_index():
    path = Path(folder_paths.models_dir) / "diffusion_models/YuE2/official_models.json"
    if not path.is_file():
        raise FileNotFoundError("YuE2 official models are not adapted yet. Run the model download and conversion launchers for your platform.")
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def model_directory(key):
    root = Path(folder_paths.models_dir).resolve()
    models = model_index()
    if key not in models:
        raise FileNotFoundError(f"Official model '{key}' is missing. Download it and run the model conversion launcher for your platform.")
    path = (root / models[key]["directory"]).resolve()
    if not path.is_relative_to(root) or not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise ValueError(f"Invalid official model directory: {key}")
    return str(path)


def options_payload(options):
    return options or {"config": DEFAULT_CONFIG, "decode_full": False}


def execute(action, models, options=None, **values):
    progress = comfy.utils.ProgressBar(1)

    def update(event):
        if event.get("total") is None:
            return
        total = max(1, int(event["total"]))
        completed = min(total, max(0, int(event.get("completed") or 0)))
        progress.update_absolute(completed, total)

    payload = {**models, **options_payload(options), **values,
               "artifact_root": str(Path(folder_paths.get_temp_directory()) / "yue2")}
    device = models.get("device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cuda":
        with torch.cuda.device(target):
            comfy.model_management.free_memory(1e30, target)
            comfy.model_management.soft_empty_cache()
    elif target.type == "mps":
        comfy.model_management.free_memory(1e30, target)
    return run(action, payload, on_progress=update,
               check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted)


def report(result):
    return json.dumps(result.get("metadata", result), ensure_ascii=False, indent=2)


def audio_result(result):
    if result.get("audio_array"):
        audio = np.load(result["audio_array"], allow_pickle=False)
        sample_rate = result.get("sample_rate", 48000)
    else:
        audio, sample_rate = sf.read(result["audio_path"], dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[1] != 2 or sample_rate != 48000 or not np.isfinite(audio).all():
        raise ValueError("Official YuE2 runtime returned invalid stereo 48 kHz audio")
    return {"waveform": torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32)).unsqueeze(0),
            "sample_rate": sample_rate}


class YuE2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["YuE2-3B"],),
            "vae": (["YuE2-Vae", "YuE2-Vae-legacy"],),
            "device": (["cuda:0", "cuda:1", "auto", "cpu", "mps"],),
            "memory_budget_gib": ("FLOAT", {"default": 24., "min": 1., "max": 256., "step": .5}),
            "backend": (["torch-eager", "torch", "vllm"],),
            "quantization": (["none", "fp8"],),
            "offload_ar": ("BOOLEAN", {"default": False}),
            "vae_core_frames": ("INT", {"default": 1024, "min": 1, "max": 24576}),
            "verify_hashes": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("YUE2_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads official model locations and runtime settings. Heavy weights are loaded by the isolated official pipeline during generation. BF16 generation and FP32 VAE are preserved."

    def load(self, model, vae, device, memory_budget_gib, backend, quantization, offload_ar, vae_core_frames, verify_hashes):
        if model != "YuE2-3B" or vae not in {"YuE2-Vae", "YuE2-Vae-legacy"}:
            raise ValueError("Select an official YuE2 model and decoder")
        if backend not in {"torch", "torch-eager", "vllm"} or quantization not in {"none", "fp8"}:
            raise ValueError("Invalid official runtime option")
        return ({"model_dir": model_directory("yue2"), "vae_dir": model_directory("vae" if vae == "YuE2-Vae" else "vae_legacy"),
                 "device": device, "memory_budget_gib": memory_budget_gib, "backend": backend,
                 "quantization": quantization, "offload_ar": offload_ar,
                 "vae_core_frames": vae_core_frames, "verify_hashes": verify_hashes},)


class YuE2TextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "style": ("STRING", {"multiline": True, "default": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, 88 BPM"}),
            "lyrics": ("STRING", {"multiline": True, "default": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\nLet the day come into view\nEvery road begins with you"}),
            "cot": (["full", "melody", "off"],),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 2**63-1}),
            "automatic_cfg": ("BOOLEAN", {"default": True}),
            "cfg_scale": ("FLOAT", {"default": 1., "min": 0., "max": 20., "step": .01}),
            "song_id": ("STRING", {"default": "song"}),
        }, "optional": {"abc": ("STRING", {"multiline": True, "default": "", "forceInput": True})}}

    RETURN_TYPES = ("YUE2_INPUT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Builds the official SongRequest. The original tokenizer and exact prefix encoding run in the official pipeline. Use full for new songs, melody for covers, off for direct generation."

    def encode(self, style, lyrics, cot, seed, automatic_cfg, cfg_scale, song_id, abc=""):
        if cot not in {"full", "melody", "off"}:
            raise ValueError("cot must be full, melody or off")
        request = {"style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "id": song_id,
                   "cfg_scale": None if automatic_cfg else cfg_scale}
        if abc.strip():
            request["abc"] = abc
        return (request,)


class YuE2GenerationOptions:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        for phase, values in DEFAULT_CONFIG.items():
            if phase not in {"abc", "semantic"}:
                continue
            for key, value in values.items():
                if key == "temperature":
                    limits = {"min": 0., "max": 5., "step": .05}
                elif key == "top_p":
                    limits = {"min": .01, "max": 1., "step": .01}
                elif key == "repetition_penalty":
                    limits = {"min": .01, "max": 10., "step": .005}
                elif key == "penalty_window":
                    limits = {"min": 1, "max": 100}
                elif key == "min_tokens":
                    limits = {"min": 0, "max": 24576}
                else:
                    limits = {"min": 1, "max": 184704 if key == "top_k" else 24576}
                inputs[f"{phase}_{key}"] = ("FLOAT" if isinstance(value, float) else "INT", {"default": value, **limits})
        inputs["ode_steps"] = ("INT", {"default": 32, "min": 1, "max": 10000})
        inputs["decode_full"] = ("BOOLEAN", {"default": False})
        return {"required": inputs}

    RETURN_TYPES = ("YUE2_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY
    DESCRIPTION = "Every official ABC/semantic sampling control, acoustic midpoint steps and decoder mode. Official token rate is 25/s; max_semantic_tokens is a ceiling, not a forced duration."

    def configure(self, ode_steps, decode_full, **values):
        config = {"ode_steps": ode_steps, "ode_method": "midpoint", "context": 24576}
        for phase in ("abc", "semantic"):
            config[phase] = {key: values[f"{phase}_{key}"] for key in DEFAULT_CONFIG[phase]}
        return ({"config": config, "decode_full": decode_full},)


class YuE2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"models": ("YUE2_MODELS",), "text": ("YUE2_INPUT",)},
                "optional": {"options": ("YUE2_OPTIONS",)}}

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "abc", "report")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Runs the unchanged official pipeline: exact symbolic plan, semantic tokens, midpoint synthesis, FP32 audio decoding. Outputs stereo 48 kHz AUDIO without extra normalization or effects."

    def generate(self, models, text, options=None):
        result = execute("generate", models, options, request=text)
        return (audio_result(result), result.get("abc") or "", report(result))


class YuE2Plan:
    INPUT_TYPES = YuE2Generate.INPUT_TYPES
    RETURN_TYPES = ("YUE2_PLAN", "STRING", "STRING")
    RETURN_NAMES = ("plan", "abc", "report")
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = "Creates and saves a SymbolicPlan with exact ABC token IDs. Connect plan directly to Semantic Tokens for exact reuse; edit the ABC only when requesting a new composition."

    def plan(self, models, text, options=None):
        result = execute("plan", models, options, request=text)
        return (result, result.get("abc") or "", report(result))


class YuE2SemanticTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"models": ("YUE2_MODELS",), "plan": ("YUE2_PLAN",)}, "optional": {"options": ("YUE2_OPTIONS",)}}

    RETURN_TYPES = ("YUE2_SEMANTIC", "STRING")
    RETURN_NAMES = ("semantic", "report")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, models, plan, options=None):
        result = execute("semantic", models, options, input_dir=plan["plan_dir"])
        return (result, report(result))


class YuE2Synthesize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"models": ("YUE2_MODELS",), "semantic": ("YUE2_SEMANTIC",)}, "optional": {"options": ("YUE2_OPTIONS",)}}

    RETURN_TYPES = ("YUE2_LATENT", "STRING")
    RETURN_NAMES = ("latents", "report")
    FUNCTION = "synthesize"
    CATEGORY = CATEGORY

    def synthesize(self, models, semantic, options=None):
        result = execute("synthesize", models, options, input_dir=semantic["semantic_dir"])
        return (result, report(result))


class YuE2Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"models": ("YUE2_MODELS",), "latents": ("YUE2_LATENT",)}, "optional": {"options": ("YUE2_OPTIONS",)}}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, models, latents, options=None):
        result = execute("decode", models, options, input_dir=latents["artifact_dir"])
        return (audio_result(result), report(result))


class YuE2ABCInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, abc):
        return (abc,)


class YuE2Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "mode": (["melody", "full"],),
                "device": (["cuda:0", "cuda:1", "cpu"],), "dtype": (["bf16", "fp32"],),
                "preset": (["default", "paper"],),
                "max_seconds": ("FLOAT", {"default": 0., "min": 0., "max": 7200., "step": 1., "tooltip": "0 transcribes the entire source; a positive value limits its length in seconds."}),
                "overlap_seconds": ("FLOAT", {"default": 200., "min": 0., "max": 1000., "step": 1.}),
                "lookahead_seconds": ("FLOAT", {"default": 100., "min": 0., "max": 1000., "step": 1.})},
                "optional": {
                    "automatic_window_settings": ("BOOLEAN", {"default": True}),
                    "prompts": ("STRING", {"default": "", "tooltip": "Leave blank for all official tasks, or provide comma-separated task names including timestamp."}),
                    "export_logits": ("BOOLEAN", {"default": False}),
                    "export_scores": ("BOOLEAN", {"default": False}),
                    "export_embeddings": ("BOOLEAN", {"default": False}),
                    "output_hidden_states": ("BOOLEAN", {"default": False}),
                    "render_audio": ("BOOLEAN", {"default": False}),
                    "render_score": (["none", "pdf", "svg", "png", "pdf,svg,png"],),
                    "render_parts": ("STRING", {"default": "mix"}),
                }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc", "report")
    FUNCTION = "transcribe"
    CATEGORY = CATEGORY
    DESCRIPTION = "Official SheetSage2 transcription for score-conditioned covers. This extracts a musical score; it is not voice cloning or waveform-preserving voice conversion. Requires the separate SheetSage runtime."

    def transcribe(self, audio, mode, device, dtype, preset, max_seconds, overlap_seconds, lookahead_seconds,
                   automatic_window_settings=True, prompts="", export_logits=False, export_scores=False,
                   export_embeddings=False, output_hidden_states=False, render_audio=False,
                   render_score="none", render_parts="mix"):
        if mode not in {"melody", "full"}:
            raise ValueError("Transcription mode must be melody or full")
        waveform = audio["waveform"]
        if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[-1] == 0:
            raise ValueError("Provide one nonempty AUDIO recording")
        folder = Path(folder_paths.get_temp_directory()) / "yue2"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=folder, delete=False) as stream:
            path = Path(stream.name)
        try:
            sf.write(path, waveform[0].detach().cpu().T.numpy(), audio["sample_rate"], subtype="FLOAT")
            transcription_options = {"export_logits": export_logits, "export_scores": export_scores,
                                     "export_embeddings": export_embeddings, "output_hidden_states": output_hidden_states,
                                     "render_audio": render_audio, "render_score": False if render_score == "none" else render_score,
                                     "render_parts": render_parts}
            if prompts.strip():
                transcription_options["prompts"] = [item.strip() for item in prompts.split(",") if item.strip()]
            result = execute("transcribe", {"model_dir": model_directory("sheetsage2"), "mert_dir": model_directory("mert2"),
                             "device": device}, audio_path=str(path), melody_only=mode == "melody", dtype=dtype,
                             preset=preset, max_seconds=None if max_seconds == 0 else max_seconds,
                             overlap_seconds=None if automatic_window_settings else overlap_seconds,
                             lookahead_seconds=None if automatic_window_settings else lookahead_seconds,
                             **transcription_options)
            return (result["abc"], report(result))
        finally:
            path.unlink(missing_ok=True)


NODE_CLASS_MAPPINGS = {
    "SomoraYuE2ModelLoader": YuE2ModelLoader, "SomoraYuE2TextEncode": YuE2TextEncode,
    "SomoraYuE2GenerationOptions": YuE2GenerationOptions, "SomoraYuE2Generate": YuE2Generate,
    "SomoraYuE2Plan": YuE2Plan, "SomoraYuE2SemanticTokens": YuE2SemanticTokens,
    "SomoraYuE2Synthesize": YuE2Synthesize, "SomoraYuE2Decode": YuE2Decode,
    "SomoraYuE2ABCInput": YuE2ABCInput, "SomoraYuE2Transcribe": YuE2Transcribe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SomoraYuE2ModelLoader": "YuE2 Official Model Loader", "SomoraYuE2TextEncode": "YuE2 Official Text Encode",
    "SomoraYuE2GenerationOptions": "YuE2 Official Generation Options", "SomoraYuE2Generate": "YuE2 Official Generate Music",
    "SomoraYuE2Plan": "YuE2 Official Generate ABC", "SomoraYuE2SemanticTokens": "YuE2 Official Semantic Tokens",
    "SomoraYuE2Synthesize": "YuE2 Official Synthesize", "SomoraYuE2Decode": "YuE2 Official Decode Audio",
    "SomoraYuE2ABCInput": "YuE2 ABC Score Input", "SomoraYuE2Transcribe": "YuE2 Official SheetSage2 Transcribe",
}
