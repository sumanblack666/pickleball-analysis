from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriterService:
    def __init__(self) -> None:
        self.writer: cv2.VideoWriter | None = None

    def write(self, output_path: Path | None, fps: float, frame: np.ndarray) -> None:
        if output_path is None:
            return
        if self.writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError(f"Writer failed: {output_path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
