from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from lightning.fabric.utilities.data import AttributeDict

from .model import LightMedVLM


def add_lightning_safe_globals() -> None:
    """Allowlist Lightning AttributeDict for torch checkpoint loading.

    This mirrors the notebook line:
        torch.serialization.add_safe_globals([AttributeDict])
    """
    torch.serialization.add_safe_globals([AttributeDict])


def resolve_checkpoint(ckpt_path: Optional[str], checkpoints_dir: str) -> str:
    """Resolve checkpoint path.

    - If `ckpt_path` is provided and exists, use it.
    - Else, pick the newest *.ckpt or *.pth in `checkpoints_dir`.
    """
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            return str(p)

    d = Path(checkpoints_dir)
    if not d.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {d}")

    candidates = list(d.glob("*.ckpt")) + list(d.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files (*.ckpt/*.pth) found in: {d}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_model_from_checkpoint(
    ckpt_path: Optional[str] = None,
    checkpoints_dir: str = "lightmedvlm/checkpoints",
    vision_model: str = "microsoft/swin-base-patch4-window7-224",
    llm_model: str = "Qwen/Qwen3-0.6B",
    device: str = "cuda",
) -> LightMedVLM:
    add_lightning_safe_globals()
    resolved = resolve_checkpoint(ckpt_path, checkpoints_dir)
    model = LightMedVLM.load_from_checkpoint(resolved, strict=False, vision_model=vision_model, llm_model=llm_model)
    model = model.to(device)
    model.eval()
    return model
