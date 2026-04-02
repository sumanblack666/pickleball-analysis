from __future__ import annotations

import numpy as np

from ..types import DetectedObject
from .common import role_from_class_id, to_numpy

try:
    import supervision as sv

    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False


class DetectionTracker:
    def __init__(self, class_map: dict[int, str], role_ids: dict[str, int | None]) -> None:
        self.class_map = class_map
        self.role_ids = role_ids
        self.ball_tracker = None
        self.player_tracker = None

    def update_mappings(self, class_map: dict[int, str], role_ids: dict[str, int | None]) -> None:
        self.class_map = class_map
        self.role_ids = role_ids

    def reset(self, frame_rate: float = 30.0) -> None:
        if not HAS_SUPERVISION:
            self.ball_tracker = None
            self.player_tracker = None
            return
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=30,
            minimum_matching_threshold=0.6,
            frame_rate=int(frame_rate),
        )
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=15,
            minimum_matching_threshold=0.7,
            frame_rate=int(frame_rate),
        )

    def extract_detections(self, result: object) -> list[DetectedObject]:
        boxes_data = to_numpy(getattr(getattr(result, "boxes", None), "data", None))
        if boxes_data.size == 0:
            return []
        if boxes_data.ndim == 1:
            boxes_data = np.expand_dims(boxes_data, 0)

        detections: list[DetectedObject] = []
        for row in boxes_data:
            if len(row) < 6:
                continue
            x1, y1, x2, y2 = (int(value) for value in row[:4])
            confidence, class_id = float(row[4]), int(row[5])
            label = self.class_map.get(class_id, f"class_{class_id}")
            role = role_from_class_id(class_id, self.role_ids)
            center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            anchor = (center_x, float(y2)) if role == "person" else (center_x, center_y)
            detections.append(
                DetectedObject(
                    class_id=class_id,
                    confidence=confidence,
                    label=label,
                    role=role,
                    box_xyxy=(x1, y1, x2, y2),
                    anchor_xy=anchor,
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def apply_tracking(self, detections: list[DetectedObject]) -> list[DetectedObject]:
        if not HAS_SUPERVISION or self.ball_tracker is None:
            return detections

        for role_name, tracker in (("pickleball", self.ball_tracker), ("person", self.player_tracker)):
            role_detections = [detection for detection in detections if detection.role == role_name]
            if not role_detections:
                continue
            xyxy = np.array([detection.box_xyxy for detection in role_detections], dtype=np.float32)
            confidences = np.array([detection.confidence for detection in role_detections], dtype=np.float32)
            class_ids = np.array([detection.class_id for detection in role_detections], dtype=int)
            tracked = tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids))
            if tracked.tracker_id is None:
                continue
            for index, tracker_id in enumerate(tracked.tracker_id):
                if index < len(role_detections):
                    role_detections[index].track_id = int(tracker_id)
        return detections


class TrajectoryInterpolator:
    def __init__(self, max_gap: int = 8) -> None:
        self.max_gap = max_gap
        self.last_point: tuple[int, int] | None = None
        self.second_last_point: tuple[int, int] | None = None
        self.gap = 0

    def reset(self) -> None:
        self.last_point = None
        self.second_last_point = None
        self.gap = 0

    def update(self, point: tuple[int, int] | None) -> tuple[tuple[int, int] | None, bool]:
        if point is not None:
            self.second_last_point = self.last_point
            self.last_point = point
            self.gap = 0
            return point, False

        self.gap += 1
        if self.gap > self.max_gap or self.last_point is None:
            return None, False
        if self.second_last_point is not None:
            delta_x = self.last_point[0] - self.second_last_point[0]
            delta_y = self.last_point[1] - self.second_last_point[1]
            interpolated_x = int(self.last_point[0] + delta_x * self.gap * 0.5)
            interpolated_y = int(self.last_point[1] + delta_y * self.gap * 0.5)
            return (interpolated_x, interpolated_y), True
        return self.last_point, True
