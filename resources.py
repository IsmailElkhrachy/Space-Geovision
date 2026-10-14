"""Portable resource lookup (icons, splash image, SAM weights) - no hard-coded developer paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parent
ICON_DIR = APP_DIR / "icons"

# Official Segment Anything checkpoint file names (https://github.com/facebookresearch/segment-anything)
SAM_CHECKPOINTS = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


def resource_path(name: str) -> str:
    """Path of a bundled icon/image. A missing file is fine: Qt then shows an empty icon."""
    return str(ICON_DIR / name)


def sam_search_dirs() -> list:
    dirs = []
    env = os.environ.get("SAM_CHECKPOINT_DIR")
    if env:
        dirs.append(Path(env))
    dirs += [APP_DIR / "models", APP_DIR, Path.home() / ".cache" / "space_geovision", Path.home() / ".cache" / "torch" / "hub" / "checkpoints"]
    return dirs


def model_type_from_filename(path: str) -> Optional[str]:
    n = Path(path).name.lower()
    for key in SAM_CHECKPOINTS:
        if key in n:
            return key
    return None


def find_sam_checkpoint(model_type: str = "vit_h", saved: Optional[str] = None) -> Optional[str]:
    """Return an existing checkpoint path for the model type, or None (the caller then asks the user)."""
    if saved and Path(saved).is_file():
        return str(saved)
    fname = SAM_CHECKPOINTS.get(model_type)
    if not fname:
        return None
    for d in sam_search_dirs():
        p = d / fname
        if p.is_file():
            return str(p)
    return None
