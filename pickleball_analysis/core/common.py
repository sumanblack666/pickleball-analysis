from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ..constants import DEFAULT_COURT_MODEL, ROLE_ALIASES


def to_numpy(value: object) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def find_latest_pose_model() -> Path | None:
    candidates = list(Path("runs/pose").glob("*/weights/best.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_default_court_model() -> Path:
    if DEFAULT_COURT_MODEL.exists():
        return DEFAULT_COURT_MODEL
    latest = find_latest_pose_model()
    return latest if latest is not None else Path("yolo26m-pose.pt")


def normalize_model_names(names_obj: object) -> dict[int, str]:
    class_map: dict[int, str] = {}
    if isinstance(names_obj, dict):
        for key, value in names_obj.items():
            try:
                class_map[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(names_obj, list):
        for index, value in enumerate(names_obj):
            class_map[index] = str(value)
    return class_map


def resolve_role_ids(class_map: dict[int, str]) -> dict[str, int | None]:
    role_ids: dict[str, int | None] = {"paddle": None, "person": None, "pickleball": None}
    for role, aliases in ROLE_ALIASES.items():
        for class_id, name in class_map.items():
            if any(alias in name.lower() for alias in aliases):
                role_ids[role] = class_id
                break
    return role_ids


def role_from_class_id(class_id: int, role_ids: dict[str, int | None]) -> str:
    for role, resolved_id in role_ids.items():
        if resolved_id is not None and class_id == resolved_id:
            return role
    return "unknown"


def is_probable_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
