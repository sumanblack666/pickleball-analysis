from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import AnalysisEvent, SessionSummary


@dataclass(slots=True)
class AnalyticsState:
    fps: float = 30.0
    px_per_ft: float = 1.0
    rally_count: int = 0
    shot_count: int = 0
    _ball_prev_y: float | None = None
    _ball_prev_side: str = ""
    _ball_dir_prev: int = 0
    far_speed: float = 0.0
    near_speed: float = 0.0
    ball_speed: float = 0.0
    ball_peak_speed: float = 0.0
    _far_prev: tuple[float, float] | None = None
    _near_prev: tuple[float, float] | None = None
    _ball_prev_pt: tuple[float, float] | None = None
    far_distance: float = 0.0
    near_distance: float = 0.0
    far_heatmap: np.ndarray | None = None
    near_heatmap: np.ndarray | None = None
    far_left_frames: int = 0
    near_left_frames: int = 0
    far_total_frames: int = 0
    near_total_frames: int = 0
    ball_crossings: int = 0
    events: list[AnalysisEvent] = field(default_factory=list)

    def _px_to_ft(self, distance_px: float) -> float:
        return distance_px / max(0.01, self.px_per_ft)

    def update_ball(self, point: tuple[int, int] | None, net_y: float, frame_index: int = 0) -> None:
        if point is None:
            return
        y_pos = float(point[1])
        side = "far" if y_pos < net_y else "near"
        if self._ball_prev_side and side != self._ball_prev_side:
            self.rally_count = max(1, self.rally_count)
            self.shot_count += 1
            self.ball_crossings += 1
            self.events.append(
                AnalysisEvent(
                    event_type="shot",
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / max(1e-6, self.fps),
                    title=f"Shot {self.shot_count}",
                    details=f"Ball crossed from {self._ball_prev_side} to {side}",
                    metadata={"to_side": side},
                )
            )
        ball_direction = 1 if (self._ball_prev_y is not None and y_pos > self._ball_prev_y) else -1
        if self._ball_dir_prev != 0 and ball_direction != self._ball_dir_prev:
            self.shot_count += 0
        self._ball_dir_prev = ball_direction
        self._ball_prev_side = side
        self._ball_prev_y = y_pos
        if self._ball_prev_pt is not None:
            distance = math.hypot(point[0] - self._ball_prev_pt[0], point[1] - self._ball_prev_pt[1])
            speed = self._px_to_ft(distance) * self.fps
            self.ball_speed = 0.7 * self.ball_speed + 0.3 * speed
            self.ball_peak_speed = max(self.ball_peak_speed, speed)
        self._ball_prev_pt = (float(point[0]), float(point[1]))

    def update_player(
        self,
        role: str,
        point: tuple[int, int] | None,
        map_shape: tuple[int, ...],
        frame_index: int = 0,
    ) -> None:
        del frame_index
        if point is None:
            return
        floating_point = (float(point[0]), float(point[1]))
        mid_x = max(1, map_shape[1]) / 2.0
        if role == "far":
            self.far_total_frames += 1
            if point[0] < mid_x:
                self.far_left_frames += 1
            if self._far_prev is not None:
                distance = math.hypot(floating_point[0] - self._far_prev[0], floating_point[1] - self._far_prev[1])
                speed = self._px_to_ft(distance) * self.fps
                self.far_speed = 0.7 * self.far_speed + 0.3 * speed
                self.far_distance += self._px_to_ft(distance)
            self._far_prev = floating_point
            if self.far_heatmap is not None:
                y_pos = int(np.clip(point[1], 0, map_shape[0] - 1))
                x_pos = int(np.clip(point[0], 0, map_shape[1] - 1))
                self.far_heatmap[y_pos, x_pos] += 1
        elif role == "near":
            self.near_total_frames += 1
            if point[0] < mid_x:
                self.near_left_frames += 1
            if self._near_prev is not None:
                distance = math.hypot(floating_point[0] - self._near_prev[0], floating_point[1] - self._near_prev[1])
                speed = self._px_to_ft(distance) * self.fps
                self.near_speed = 0.7 * self.near_speed + 0.3 * speed
                self.near_distance += self._px_to_ft(distance)
            self._near_prev = floating_point
            if self.near_heatmap is not None:
                y_pos = int(np.clip(point[1], 0, map_shape[0] - 1))
                x_pos = int(np.clip(point[0], 0, map_shape[1] - 1))
                self.near_heatmap[y_pos, x_pos] += 1


class AnalyticsEngine:
    def __init__(self, px_per_ft: float) -> None:
        self.px_per_ft = px_per_ft
        self.state = AnalyticsState(px_per_ft=px_per_ft)
        self.map_shape: tuple[int, int] = (1, 1)
        self.source_label = ""

    def reset(self, fps: float, map_shape: tuple[int, ...], source_label: str) -> None:
        self.map_shape = map_shape[:2]
        self.source_label = source_label
        self.state = AnalyticsState(fps=fps, px_per_ft=self.px_per_ft)
        self.state.far_heatmap = np.zeros(self.map_shape, dtype=np.float32)
        self.state.near_heatmap = np.zeros(self.map_shape, dtype=np.float32)

    def update_ball(self, point: tuple[int, int] | None, net_y: float, frame_index: int) -> None:
        self.state.update_ball(point, net_y, frame_index)

    def update_player(self, role: str, point: tuple[int, int] | None, frame_index: int) -> None:
        self.state.update_player(role, point, self.map_shape, frame_index)

    def heatmap_overlay(self) -> np.ndarray | None:
        if self.state.far_heatmap is None or self.state.near_heatmap is None:
            return None
        return self.state.far_heatmap + self.state.near_heatmap

    def recent_events(self, limit: int = 5) -> tuple[AnalysisEvent, ...]:
        return tuple(self.state.events[-limit:])

    def build_summary(
        self,
        total_frames: int,
        processed_frames: int,
        status_message: str,
        exported_files: tuple[Path, ...] = (),
    ) -> SessionSummary:
        return SessionSummary(
            source_label=self.source_label,
            status_message=status_message,
            total_frames=total_frames,
            processed_frames=processed_frames,
            fps=self.state.fps,
            duration_seconds=processed_frames / max(1e-6, self.state.fps),
            rally_count=self.state.rally_count,
            shot_count=self.state.shot_count,
            far_distance_ft=self.state.far_distance,
            near_distance_ft=self.state.near_distance,
            far_speed_ft_s=self.state.far_speed,
            near_speed_ft_s=self.state.near_speed,
            ball_speed_ft_s=self.state.ball_speed,
            ball_peak_speed_ft_s=self.state.ball_peak_speed,
            far_left_pct=self.state.far_left_frames / max(1, self.state.far_total_frames),
            near_left_pct=self.state.near_left_frames / max(1, self.state.near_total_frames),
            exported_files=exported_files,
            recent_events=tuple(self.state.events),
        )

    def export_summary_files(self, output_video_path: Path, summary: SessionSummary) -> tuple[Path, ...]:
        base_path = output_video_path.with_suffix("")
        json_path = base_path.parent / f"{base_path.name}_summary.json"
        csv_path = base_path.parent / f"{base_path.name}_events.csv"
        write_summary_json(json_path, summary)
        write_events_csv(csv_path, summary.recent_events)
        return json_path, csv_path


def write_summary_json(path: Path, summary: SessionSummary) -> None:
    payload = {
        "source_label": summary.source_label,
        "status_message": summary.status_message,
        "total_frames": summary.total_frames,
        "processed_frames": summary.processed_frames,
        "fps": summary.fps,
        "duration_seconds": summary.duration_seconds,
        "rally_count": summary.rally_count,
        "shot_count": summary.shot_count,
        "far_distance_ft": summary.far_distance_ft,
        "near_distance_ft": summary.near_distance_ft,
        "far_speed_ft_s": summary.far_speed_ft_s,
        "near_speed_ft_s": summary.near_speed_ft_s,
        "ball_speed_ft_s": summary.ball_speed_ft_s,
        "ball_peak_speed_ft_s": summary.ball_peak_speed_ft_s,
        "far_left_pct": summary.far_left_pct,
        "near_left_pct": summary.near_left_pct,
        "exported_files": [str(value) for value in summary.exported_files],
        "events": [
            {
                "event_type": event.event_type,
                "frame_index": event.frame_index,
                "timestamp_seconds": event.timestamp_seconds,
                "title": event.title,
                "details": event.details,
                "metadata": event.metadata,
            }
            for event in summary.recent_events
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_events_csv(path: Path, events: tuple[AnalysisEvent, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_type", "frame_index", "timestamp_seconds", "title", "details"],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_type": event.event_type,
                    "frame_index": event.frame_index,
                    "timestamp_seconds": f"{event.timestamp_seconds:.3f}",
                    "title": event.title,
                    "details": event.details,
                }
            )
