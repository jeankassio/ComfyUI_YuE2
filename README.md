# ComfyUI_YuE2

[Português brasileiro](README_PTBR.md)

ComfyUI nodes for **the official YuE2 Python pipeline**: model loading, style/lyrics input, symbolic composition, semantic generation, acoustic synthesis, stereo decoding and score-conditioned covers. The engine is pinned to [YuE2 v0.1.6](https://github.com/multimodal-art-projection/YuE/tree/1647252d68b70cbead046ffd3f7caf036dde47ee).

Inference uses the upstream Python implementation and official Hugging Face weights. ComfyUI supplies the graph, audio inputs/outputs and progress display. The model math, tokenizer, sampling, guidance and midpoint solver stay in the upstream engine.

## Install on Windows

Put this repository in `ComfyUI/custom_nodes/ComfyUI_YuE2`. A typical portable installation is:

```text
Comfy/
├── python_embeded/
└── ComfyUI/
    ├── models/
    └── custom_nodes/ComfyUI_YuE2/
```

Run these launchers in order; each keeps its console visible:

1. `setup-runtime.bat` — installs the pinned official engine in `.venv`, using Python 3.12 and its own Torch/Transformers versions.
2. `scripts/download-models.bat` — explicitly downloads the revision-pinned model files and verifies SHA256.
3. `convert-models.bat` — places the verified official artifacts in ComfyUI model categories and writes the local model index.
4. For audio-to-cover, also run `scripts/setup-sheetsage.bat` — installs the separate `.venv-sheetsage` environment. The installer prepares a private `uv` copy when needed; it also accepts `--uv C:\path\to\uv.exe`.
5. Restart ComfyUI and drag an example workflow onto the canvas.

The adapter uses NumPy, SoundFile and tqdm in ComfyUI's existing Python. If those are missing, install this repository's `requirements.txt` with that Python. Heavy generation/transcription dependencies belong to the isolated runtimes, so these installers do not replace ComfyUI's Torch or Transformers.

On Windows, keep `backend=torch-eager`: this official Torch distribution lacks the FlashAttention kernel required by the CUDA Graph backend. Eager retains the official weights, precision and algorithms. CUDA indices refer to devices visible to the process; use `--cuda-device all` to expose both GPUs in ComfyUI.

## Install on Linux and macOS

Open a terminal in `ComfyUI/custom_nodes/ComfyUI_YuE2`. A working Python 3 with `venv`/`pip` is needed for bootstrap; on Debian/Ubuntu, install `python3-venv` if the installer reports it missing. The launchers also find ComfyUI's `.venv` or `venv` automatically. Run:

```sh
sh setup-runtime.sh
sh scripts/download-models.sh
sh convert-models.sh
# Optional: transcription for audio-to-cover
sh scripts/setup-sheetsage.sh
```

The shell scripts preserve arguments, work from any current directory and use isolated environments. The download launcher explicitly downloads by default; `--check` verifies local files offline:

```sh
sh scripts/download-models.sh --models yue2 vae
sh scripts/download-models.sh --check
```

| Platform | Runtime installation and node settings |
|---|---|
| Linux with NVIDIA | Official CUDA 12.8 Torch wheel for YuE2; CUDA 12.6 Torch/Torchaudio for SheetSage2. Select the available CUDA device. The official recommended starting point is a BF16-capable GPU with 24 GB VRAM. |
| Linux without CUDA | Add `--cpu` to both setup launchers and select `device=cpu` in the nodes. |
| macOS with Apple Silicon | Official macOS wheels from PyPI, including CPU/MPS. In Model Loader select `device=auto` or `mps` and `backend=torch-eager`; for SheetSage2 Transcribe select `cpu`. MPS BF16 requires macOS 14 or later. |
| macOS with Intel/translated x86 Python | The pinned official Torch versions have no Intel macOS wheels; setup reports this explicitly. Use native ARM64 Python on Apple Silicon. |

The setup scripts keep Python and dependencies local, including a private `uv` bootstrap when needed, without changing system Python or ComfyUI's dependencies. Both accept `--torch-index-url URL` for an explicit wheel-index override. `--cpu` on macOS uses the same official CPU/MPS distribution; select `cpu` in the node to use CPU execution. The optional vLLM backend is Linux GPU only: `sh setup-runtime.sh --fast`.

Restart ComfyUI after installation and load a workflow from `example_workflows`. Linux/macOS setup choices and shell launchers were tested without installing on those systems; generation has been validated on Windows/RTX 3090.

## Models and folders

All five model sets and required supporting files total **11.08 GB**, before runtime dependencies.

| Model | Role | Adapted location beneath `ComfyUI/models/` |
|---|---|---|
| [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | Shared AR/NAR backbone: ABC planning, semantic tokens and acoustic latents | `diffusion_models/YuE2/YuE2-3B/` |
| `qwen.tiktoken` | Official text tokenizer; its learned model components are inside YuE2-3B | `text_encoders/YuE2/qwen.tiktoken` and the model directory |
| [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | Current listening decoder; default for generated music | `vae/YuE2/YuE2-Vae/` |
| [YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | Decoder for reproducing the published evaluation protocol | `vae/YuE2/YuE2-Vae-legacy/` |
| [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | Converts source music into a structured ABC score for covers | `audio_encoders/YuE2/SheetSage2/` |
| [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | SheetSage2's required audio backbone | `audio_encoders/YuE2/MERT-v2-FullSong/` |

YuE2 uses a joint transformer rather than an image UNet plus a separate text-encoder checkpoint. The folder adapter retains **bit-identical official weights and configs**; it does not quantize, rename tensors or substitute a native ComfyUI implementation. Hard links avoid a second copy on the same filesystem; otherwise the adapter copies verified files. Original downloads remain in the directories recorded in `model_manifest.json`.

Use these commands for a selected download or an offline integrity check:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --download --models yue2 vae
.\.venv\Scripts\python.exe scripts\download_models.py --check
```

Inference runs with local-only Hugging Face settings. Downloads occur only when you run the downloader. `model_manifest.json` pins repositories, revisions, file sizes and SHA256, including the exact MERT revision expected by SheetSage2.

## Example workflows

The **ComfyUI v1.0 canvas JSON** examples are in [example_workflows/](example_workflows/).

| Workflow | Use |
|---|---|
| [01-text-to-music.json](example_workflows/01-text-to-music.json) | Four nodes: Model Loader → Text Encode → Generate Music → Save Audio. Start here. |
| [02-staged-generation.json](example_workflows/02-staged-generation.json) | Full controls and exact stages: Generate ABC → Semantic Tokens → Synthesize → Decode Audio → Save Audio. |
| [03-edited-abc-score.json](example_workflows/03-edited-abc-score.json) | Edit a short original ABC composition, then generate a performance from it. |
| [04-audio-to-cover.json](example_workflows/04-audio-to-cover.json) | Load Audio → SheetSage2 Transcribe → Text Encode with melody ABC → Generate Music → Save Audio. |

For workflow 04, **upload and select a real source song in Load Audio before queuing**. `select-source-audio.wav` is a placeholder, not a bundled recording. Set target lyrics/style in Text Encode. Keep `mode=melody` in Transcribe and `cot=melody` in Text Encode for a cover with adaptable accompaniment.

The text sample comes from upstream [examples/song.json](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/examples/song.json). Workflow 03 contains a new short demonstration composition in its ABC Score Input node. Edit its pitches, durations, harmony or tempo deliberately; a supplied score conditions a new performance.

## Nodes and official controls

The nodes appear under **audio/YuE2 Official**; all node labels are English.

| Node | Inputs and result |
|---|---|
| Official Model Loader | YuE2 model, listening/legacy VAE, device, memory budget, `torch`/`torch-eager`/`vllm`, quantization, AR offload and VAE chunk size. Returns runtime/model settings. |
| Official Text Encode | Style, lyrics, planning mode, seed, automatic/manual CFG and song ID; optional ABC score. Returns the official song request. |
| Official Generation Options | Independent ABC and semantic sampling controls, acoustic ODE steps and full/chunked VAE decode. |
| Official Generate Music | Complete official pipeline; returns 48 kHz stereo AUDIO, ABC and a JSON report. |
| Official Generate ABC | Creates a saved symbolic plan with exact token IDs; returns plan, readable ABC and report. |
| Official Semantic Tokens | Consumes the saved plan directly, preserving its exact token IDs. |
| Official Synthesize | Turns semantic tokens into acoustic latents using the official midpoint solver. |
| Official Decode Audio | Decodes saved latents with the selected official VAE. |
| ABC Score Input | Editable native-dialect ABC text. |
| Official SheetSage2 Transcribe | Audio-to-score controls, melody/full mode, BF16/FP32, default/paper presets, automatic/manual window settings, task prompts and official export/render options. |

Planning modes follow the official API:

- `full`: creates melody and chord planning; the default for new songs.
- `melody`: melody planning with free accompaniment; recommended for covers.
- `off`: generates directly from lyrics and style.
- Supplied ABC requires `full` or `melody`. Use the native two-voice `Vocal`/`Ins` notation described in the [official ABC guide](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/skills/yue2-music/references/abc-editing.md).

Generation Options exposes `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens` and `max_tokens` **for both ABC and semantic stages**. Defaults match the official generation configuration, including **32 midpoint ODE steps**. Automatic CFG selects the engine's official default for the planning mode. The quality defaults use unquantized **BF16 generation and FP32 VAE decoding**, with the listening VAE. FP8 is an explicit optional selection; it is never enabled to rescue a memory failure.

Semantic audio runs at **25 tokens per second**. `semantic_max_tokens` is a duration ceiling, not an exact length control: the model can finish earlier. For example, 9000 tokens permit about 360 seconds. Put the intended tempo, language, vocal character, instruments and genre in `style`; put singable words and section tags in `lyrics`.

For SheetSage2, `max_seconds=0` processes the whole source recording; a positive value explicitly limits the input to that many seconds. `automatic_window_settings=true` lets the selected official preset choose window settings. Disable it to use the explicit seconds fields. Empty task prompts use official defaults. Extra export flags write artifacts; optional rendering requires the official SheetSage2 `render_assets/` files and dependencies installed by its `setup_render.py`. The model downloader supplies inference files; renderer assets are a separate optional setup.

## Output, progress and comparison

Connect AUDIO to ComfyUI's Save Audio (Advanced); examples select **FLAC**. Audio is stereo **48,000 Hz**. Normalization, effects and mastering are not added by this adapter. Official stage progress appears in both the ComfyUI interface and its console, using the same `tqdm` bars as ComfyUI samplers. Token generation shows actual counts and rate; acoustic steps and decoder chunks show their real totals. Loading stages show elapsed time. Cancellation interrupts the isolated process and closes the console bar.

Before launching the separate engine, the adapter releases cached ComfyUI models on the selected device to make room for the official weights.

Temporary generation artifacts are kept under `ComfyUI/temp/yue2/`, including the score, exact plan, semantic tokens, latents and generation metadata. Reports include token-limit truncation and timings. Playable audio can still have `truncated=true`; inspect the report before treating it as a complete song.

Stage nodes exchange only an artifact folder ID, resolved under ComfyUI's configured temporary directory. Model locations are resolved from registered official model keys, never from paths in incoming node dictionaries. Absolute, drive, UNC and traversal paths are rejected, as are symbolic links and Windows junctions in artifact/model paths. After updating, restart ComfyUI and rerun earlier stages to replace cached outputs from the previous format.

Reuse an unchanged `YUE2_PLAN` directly between stages. To edit a composition, copy/edit its ABC and submit it as a new request. To compare listening and legacy decoders, decode the **same latents** twice; do not generate two unrelated songs. Seeds aid comparison, but device/runtime/settings changes can affect reproducibility. Each queue execution generates one candidate; published best-of-N scores refer to a separate candidate-selection evaluation.

SheetSage2 covers regenerate a performance from a musical score. **They do not clone a speaker from a voice recording, nor replace only the singer while preserving the source waveform.** This node does not add a speaker-identity model to YuE2.

The downloaded model files pass SHA256 verification, and the example graphs pass the published ComfyUI v1.0 schema and connection checks. These checks do not establish a perceptual quality score or guarantee faithful score/lyric realization; audition your generated results.

Validated locally on Windows/RTX 3090: thirteen runtime tests, all four prompts with ComfyUI's actual validator, a 20-second generation through ComfyUI with native progress and FLAC output, full decoding of saved latents, and SheetSage2 transcription of that generated audio. Console tests cover actual counts, failure and ComfyUI cancellation. The short generation intentionally used a 500-token semantic ceiling; the shipped workflows retain the full official defaults.

## Licenses and attribution

The adapter is covered by this repository's existing [MIT license](LICENSE). The pinned upstream engine retains its [Apache 2.0 license](runtime/vendor/LICENSE), [model terms](runtime/vendor/MODEL_LICENSE) and [third-party notices](runtime/vendor/THIRD_PARTY_NOTICES.md).

Official YuE2 weights use **CC BY-NC 4.0 with additional individual-creator permission**. The licensors permit individuals to publish and monetize their generated outputs under those terms. That permission does not authorize companies to use the weights commercially; corporate use requires discussing a commercial license with the licensors. See the [official model license](https://huggingface.co/m-a-p/YuE2-3B/blob/8f24312f187763b9854adeec87e437e2b03bbff1/LICENSE) for the exact conditions. The transcription models retain their own downloaded licenses. Creating this local node does not deploy it to Somora.

Primary references: [official generation guide](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/docs/generation.md), [official release](https://github.com/multimodal-art-projection/YuE/releases/tag/yue2-v0.1.6), [SheetSage2](https://huggingface.co/m-a-p/SheetSage2), [ComfyUI workflow schema](https://docs.comfy.org/specs/workflow_json).
