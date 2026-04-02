from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

SourceKind = Literal["file", "youtube"]
PacketType = Literal["frame", "error", "completed", "status"]


@dataclass(frozen=True, slots=True)
class AppConfig:
    court_model_default: Path
    object_model_default: Path
    source_default: str
    output_default: Path
    app_title: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    kind: SourceKind
    value: str

    @property
    def display_name(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    court_model_path: Path
    object_model_path: Path
    output_video_path: Path | None
    device: str
    imgsz: int
    court_conf: float
    object_conf: float
    kp_conf: float
    frame_skip: int
    half_precision: bool
    save_video: bool
    heatmap_enabled: bool
    export_data: bool
    infer_missing_court_keypoints: bool = True
    lock_static_court: bool = False
    panel_width: int = 360
    queue_backpressure: int = 3
    homography_stale_limit: int = 90
    homography_smoothing: float = 0.80
    static_court_stable_frames: int = 8
    static_court_motion_threshold_px: float = 1.5


@dataclass(slots=True)
class DetectedObject:
    class_id: int
    confidence: float
    label: str
    role: str
    box_xyxy: tuple[int, int, int, int]
    anchor_xy: tuple[float, float]
    track_id: int = -1


@dataclass(slots=True)
class ProjectedObject:
    role: str
    label: str
    confidence: float
    point_xy: tuple[int, int]
    track_id: int = -1
    interpolated: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    event_type: str
    frame_index: int
    timestamp_seconds: float
    title: str
    details: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionSummary:
    source_label: str
    status_message: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    rally_count: int
    shot_count: int
    far_distance_ft: float
    near_distance_ft: float
    far_speed_ft_s: float
    near_speed_ft_s: float
    ball_speed_ft_s: float
    ball_peak_speed_ft_s: float
    far_left_pct: float
    near_left_pct: float
    exported_files: tuple[Path, ...]
    recent_events: tuple[AnalysisEvent, ...] = ()


@dataclass(slots=True)
class FramePacket:
    packet_type: PacketType
    status_text: str
    frame_index: int = 0
    total_frames: int = 0
    fps: float = 0.0
    detections: int = 0
    projected: int = 0
    used_keypoints: int = 0
    homography_state: str = ""
    homography_quality: float = 0.0
    frame: np.ndarray | None = None
    recent_events: tuple[AnalysisEvent, ...] = ()
    summary: SessionSummary | None = None
    error_message: str | None = None
    # Live dashboard data
    live_lines: tuple[str, ...] = ()
    analytics_lines: tuple[str, ...] = ()
    detected_keypoints: int = 0
    inferred_keypoints: int = 0
    court_locked: bool = False
    rally_count: int = 0
    shot_count: int = 0
    ball_speed: float = 0.0
    ball_peak_speed: float = 0.0
    far_speed: float = 0.0
    far_distance: float = 0.0
    near_speed: float = 0.0
    near_distance: float = 0.0


@dataclass(slots=True)
class ResolvedSource:
    source_spec: SourceSpec
    video_path: Path
    display_label: str
    cleanup_paths: tuple[Path, ...] = ()
