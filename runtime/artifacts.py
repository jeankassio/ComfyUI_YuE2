"""Persist stage boundaries without decoding and retokenizing symbolic plans."""
from pathlib import Path
import json

import numpy as np
from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.storage import sha256_file, write_json


def save_semantic(semantic, directory):
    directory = Path(directory)
    semantic.plan.save(directory)
    np.save(directory / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
    write_json(directory / "semantic.json", {
        "timing": semantic.timing, "truncated": semantic.truncated,
        "sha256": sha256_file(directory / "semantic.npy"),
    })


def load_semantic(directory):
    directory = Path(directory)
    plan = SymbolicPlan.load(directory)
    metadata = json.loads((directory / "semantic.json").read_text(encoding="utf-8"))
    path = directory / "semantic.npy"
    if path.is_symlink() or sha256_file(path) != metadata["sha256"]:
        raise ValueError("Saved semantic tokens were changed")
    tokens = np.load(path, allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("Saved semantic tokens must be an integer array")
    return SemanticResult(plan, tokens.tolist(), metadata["timing"], metadata["truncated"])


def save_latents(latents, semantic, directory, timing):
    directory = Path(directory)
    save_semantic(semantic, directory)
    np.save(directory / "latent.npy", latents)
    write_json(directory / "latent.json", {"sha256": sha256_file(directory / "latent.npy"), "timing": timing})


def load_latents(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "latent.json").read_text(encoding="utf-8"))
    path = directory / "latent.npy"
    if path.is_symlink() or sha256_file(path) != metadata["sha256"]:
        raise ValueError("Saved acoustic latents were changed")
    return np.load(path, allow_pickle=False), load_semantic(directory), metadata["timing"]
