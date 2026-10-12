# Official SheetSage2 cover transcription

This optional component uses the original [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) implementation and its pinned [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) encoder. It transcribes a recording into ABC melody/chords; YuE2 then generates a new performance from that score. It does not clone the singer or retain the original audio waveform.

Run `scripts/setup-sheetsage.bat` on Windows. The installer creates `.venv-sheetsage` with Python 3.11 and the official Torch/Torchaudio 2.8.0 and Transformers 4.45.2 requirements. YuE2 and ComfyUI retain their own environments. It keeps its Python installation in `runtime/.tools/python-sheetsage`, without registering it globally. It uses an existing `uv` executable on PATH; `--uv PATH` selects another one. If uv is missing, it installs a private copy in `runtime/.tools`. The default CUDA wheel is the official cu126 build; `--cpu` selects CPU wheels.

Models are downloaded by the project's model downloader. The worker loads both local model folders with `local_files_only=True` and verifies their manifest hashes before executing the official model code. Generation never downloads arbitrary model code.

Use **melody** for a cover with a new arrangement, or **full** to include transcribed harmony. Verify the score and lyrics before generating: transcription can make mistakes. The default transcription uses 300-second windows with 200-second overlap and 100-second lookahead. The official `paper` preset fixes overlap to 100 seconds and lookahead to zero.

The worker supports all official transcription controls: `melody_only`, `dtype` (`bf16` or `fp32`), `preset`, `max_seconds`, `overlap_seconds`, `lookahead_seconds`, `prompts`, logits/scores/embeddings export, all MERT layers, optional audio/score rendering, and render parts. Rendering requires the separate official rendering dependencies/assets. Basic ABC transcription does not need rendering.

Progress and cancellation are passed through to ComfyUI. Temporary audio, ABC, MIDI and metadata remain inside the supplied ComfyUI temporary artifact directory.

The official [cover guide](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md) specifies FFmpeg 6.1 and shared libraries for encoded audio. Put FFmpeg on PATH. The default transcription preset decodes input through the FFmpeg executable; the paper preset uses Torchaudio's FFmpeg backend and therefore also needs shared FFmpeg libraries.
