from __future__ import annotations

from pathlib import Path

from ..types import AnalysisConfig
from .common import normalize_model_names, resolve_role_ids


class ModelManager:
    def __init__(self) -> None:
        self.court_model = None
        self.object_model = None
        self.loaded_paths: tuple[Path, Path] | None = None
        self.class_map: dict[int, str] = {}
        self.role_ids: dict[str, int | None] = {"paddle": None, "person": None, "pickleball": None}
        self._half_precision_available = True

    def load(self, court_model_path: Path, object_model_path: Path) -> None:
        court_path = court_model_path.expanduser()
        object_path = object_model_path.expanduser()
        if not court_path.exists():
            raise FileNotFoundError(f"Court model not found: {court_path}")
        if not object_path.exists():
            raise FileNotFoundError(f"Object model not found: {object_path}")

        requested_paths = (court_path.resolve(), object_path.resolve())
        if self.loaded_paths == requested_paths and self.court_model is not None and self.object_model is not None:
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"Failed to import ultralytics: {exc}") from exc

        self.court_model = YOLO(str(court_path))
        self.object_model = YOLO(str(object_path))
        self.loaded_paths = requested_paths
        self._half_precision_available = True
        self.class_map = normalize_model_names(getattr(self.object_model, "names", {}))
        if not self.class_map:
            self.class_map = {0: "ball", 1: "person", 2: "paddle"}
        self.role_ids = resolve_role_ids(self.class_map)

    def predict_court(self, frame, config: AnalysisConfig):
        return self._predict(self.court_model, frame, "pose", config.court_conf, config)

    def predict_objects(self, frame, config: AnalysisConfig):
        return self._predict(self.object_model, frame, "detect", config.object_conf, config)

    def class_description(self) -> str:
        return ", ".join(f"{class_id}:{name}" for class_id, name in sorted(self.class_map.items())) or "unknown"

    def _predict(self, model, frame, task: str, confidence: float, config: AnalysisConfig):
        if model is None:
            raise RuntimeError("Models are not loaded")

        device = config.device.strip() or None
        use_half = config.half_precision and device != "cpu" and self._half_precision_available
        try:
            return model.predict(
                source=frame,
                task=task,
                conf=float(confidence),
                imgsz=max(160, int(config.imgsz)),
                device=device,
                verbose=False,
                half=use_half,
            )
        except Exception:
            if not use_half:
                raise
            self._half_precision_available = False
            return model.predict(
                source=frame,
                task=task,
                conf=float(confidence),
                imgsz=max(160, int(config.imgsz)),
                device=device,
                verbose=False,
            )
