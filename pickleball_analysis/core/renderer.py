from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..constants import COURT_KEYPOINT_NAMES, COURT_SKELETON, ROLE_COLORS_BGR
from ..types import AnalysisEvent, DetectedObject, ProjectedObject


class FrameRenderer:
    def __init__(self, minimap_base: np.ndarray, panel_width: int = 360) -> None:
        self.minimap_base = minimap_base
        self.panel_width = panel_width

    def render_output(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray | None,
        inferred_keypoint_mask: np.ndarray | None,
        kp_conf_threshold: float,
        detections: list[DetectedObject],
        projected_objects: list[ProjectedObject],
        ball_trace: deque[tuple[int, int]],
        far_trace: deque[tuple[int, int]],
        near_trace: deque[tuple[int, int]],
        live_lines: list[str],
        analytics_lines: list[str],
        recent_events: tuple[AnalysisEvent, ...],
        heatmap_overlay: np.ndarray | None,
    ) -> np.ndarray:
        annotated = frame.copy()
        self.draw_court_keypoints_overlay(annotated, keypoints, inferred_keypoint_mask, kp_conf_threshold)
        self.draw_detection_overlay(annotated, detections)
        minimap = self.draw_minimap(projected_objects, ball_trace, far_trace, near_trace, heatmap_overlay)
        return self.compose_output_frame(annotated, minimap, live_lines, analytics_lines, recent_events)

    def draw_court_keypoints_overlay(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray | None,
        inferred_keypoint_mask: np.ndarray | None,
        kp_conf_threshold: float,
    ) -> None:
        if keypoints is None:
            return
        frame_height, frame_width = frame.shape[:2]
        inferred_count = int(np.count_nonzero(inferred_keypoint_mask)) if inferred_keypoint_mask is not None else 0
        for start, end in COURT_SKELETON:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            point_a, point_b = keypoints[start], keypoints[end]
            conf_a = float(point_a[2]) if len(point_a) > 2 else 1.0
            conf_b = float(point_b[2]) if len(point_b) > 2 else 1.0
            if conf_a < kp_conf_threshold or conf_b < kp_conf_threshold:
                continue
            is_inferred = bool(inferred_keypoint_mask is not None and (inferred_keypoint_mask[start] or inferred_keypoint_mask[end]))
            line_color = (0, 165, 255) if is_inferred else (64, 200, 255)
            thickness = 3 if is_inferred else 2
            draw_a = self._clamp_draw_point((int(point_a[0]), int(point_a[1])), frame_width, frame_height, margin=4)
            draw_b = self._clamp_draw_point((int(point_b[0]), int(point_b[1])), frame_width, frame_height, margin=4)
            cv2.line(frame, draw_a, draw_b, line_color, thickness, cv2.LINE_AA)
        for index, keypoint in enumerate(keypoints):
            confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
            if confidence < kp_conf_threshold:
                continue
            center = self._clamp_draw_point((int(keypoint[0]), int(keypoint[1])), frame_width, frame_height, margin=8)
            is_inferred = bool(inferred_keypoint_mask is not None and inferred_keypoint_mask[index])
            if is_inferred:
                cv2.circle(frame, center, 8, (20, 20, 20), -1)
                cv2.circle(frame, center, 6, (0, 165, 255), 2)
                cv2.circle(frame, center, 3, (0, 215, 255), -1)
            else:
                cv2.circle(frame, center, 5, (15, 15, 15), -1)
                cv2.circle(frame, center, 3, (0, 255, 0), -1)
            if index < len(COURT_KEYPOINT_NAMES):
                label = f"I{index}" if is_inferred else str(index)
                label_origin = self._clamp_label_origin(label, center, frame_width, frame_height)
                cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        if inferred_count > 0:
            self._draw_inferred_court_banner(frame, inferred_count)

    def draw_detection_overlay(self, frame: np.ndarray, detections: list[DetectedObject]) -> None:
        for detection in detections:
            x1, y1, x2, y2 = detection.box_xyxy
            color = ROLE_COLORS_BGR.get(detection.role, ROLE_COLORS_BGR["unknown"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tracker_text = f" #{detection.track_id}" if detection.track_id >= 0 else ""
            label = f"{detection.label}{tracker_text} {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(detection.anchor_xy[0]), int(detection.anchor_xy[1])), 4, color, -1)

    def draw_minimap(
        self,
        projected_objects: list[ProjectedObject],
        ball_trace: deque[tuple[int, int]],
        far_trace: deque[tuple[int, int]],
        near_trace: deque[tuple[int, int]],
        heatmap_overlay: np.ndarray | None = None,
    ) -> np.ndarray:
        minimap = self.minimap_base.copy()
        if heatmap_overlay is not None:
            heatmap = heatmap_overlay.astype(np.float32)
            maximum = heatmap.max()
            if maximum > 0:
                normalized = (heatmap / maximum * 255).astype(np.uint8)
                blurred = cv2.GaussianBlur(normalized, (15, 15), 0)
                colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
                mask = blurred > 10
                mask_3d = np.stack([mask] * 3, axis=-1)
                minimap = np.where(mask_3d, cv2.addWeighted(minimap, 0.5, colored, 0.5, 0), minimap)

        if len(ball_trace) >= 2:
            cv2.polylines(minimap, [polyline_points(ball_trace)], False, (0, 215, 255), 2, cv2.LINE_AA)
        if len(far_trace) >= 2:
            cv2.polylines(minimap, [polyline_points(far_trace)], False, (255, 140, 0), 2, cv2.LINE_AA)
        if len(near_trace) >= 2:
            cv2.polylines(minimap, [polyline_points(near_trace)], False, (0, 165, 255), 2, cv2.LINE_AA)

        for projected in projected_objects:
            color = ROLE_COLORS_BGR.get(projected.role, ROLE_COLORS_BGR["unknown"])
            if projected.interpolated:
                cv2.circle(minimap, projected.point_xy, 5, color, 1)
            else:
                cv2.circle(minimap, projected.point_xy, 5, (20, 20, 20), -1)
                cv2.circle(minimap, projected.point_xy, 4, color, -1)

        persons = sorted([projected for projected in projected_objects if projected.role == "person"], key=lambda projected: projected.point_xy[1])
        if persons:
            cv2.putText(minimap, "P_far", (persons[0].point_xy[0] + 6, persons[0].point_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 120), 1, cv2.LINE_AA)
        if len(persons) > 1:
            cv2.putText(minimap, "P_near", (persons[-1].point_xy[0] + 6, persons[-1].point_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1, cv2.LINE_AA)

        balls = [projected for projected in projected_objects if projected.role == "pickleball"]
        if balls:
            best_ball = max(balls, key=lambda projected: projected.confidence)
            cv2.putText(minimap, "Ball", (best_ball.point_xy[0] + 6, best_ball.point_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)
        return minimap

    def compose_output_frame(
        self,
        annotated_frame: np.ndarray,
        minimap: np.ndarray,
        live_lines: list[str],
        analytics_lines: list[str],
        recent_events: tuple[AnalysisEvent, ...],
    ) -> np.ndarray:
        frame_height = annotated_frame.shape[0]
        panel = np.full((frame_height, self.panel_width, 3), (28, 28, 28), dtype=np.uint8)

        map_height, map_width = minimap.shape[:2]
        scale = min((frame_height - 250) / max(1, map_height), (self.panel_width - 28) / max(1, map_width), 1.0)
        scaled_width, scaled_height = max(1, int(map_width * scale)), max(1, int(map_height * scale))
        minimap_resized = cv2.resize(minimap, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        top_margin, left_margin = 14, (self.panel_width - scaled_width) // 2
        panel[top_margin : top_margin + scaled_height, left_margin : left_margin + scaled_width] = minimap_resized

        text_y = top_margin + scaled_height + 16
        text_y = self._draw_section(panel, text_y, "Live Stats", live_lines, (100, 200, 255))
        text_y += 6
        text_y = self._draw_section(panel, text_y, "Analytics", analytics_lines, (100, 255, 150))
        text_y += 6
        event_lines = [f"{event.timestamp_seconds:6.2f}s  {event.title}" for event in recent_events[-4:]] or ["No events yet"]
        self._draw_section(panel, text_y, "Recent Events", event_lines, (255, 215, 120))

        spacer = np.full((frame_height, 6, 3), (12, 12, 12), dtype=np.uint8)
        return np.hstack((annotated_frame, spacer, panel))

    @staticmethod
    def _draw_section(panel: np.ndarray, y_pos: int, title: str, lines: list[str], title_color: tuple[int, int, int]) -> int:
        cv2.putText(panel, f"-- {title} --", (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, title_color, 1, cv2.LINE_AA)
        y_pos += 15
        for line in lines:
            cv2.putText(panel, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
            y_pos += 14
        return y_pos

    @staticmethod
    def _draw_inferred_court_banner(frame: np.ndarray, inferred_count: int) -> None:
        banner_lines = [
            f"Inferred court keypoints: {inferred_count}",
            "Orange skeleton/keypoints are homography-filled",
        ]
        x_pos, y_pos = 12, 18
        banner_width = 0
        for line in banner_lines:
            (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            banner_width = max(banner_width, width)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_pos - 6, y_pos - 14), (x_pos + banner_width + 8, y_pos + 24), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, banner_lines[0], (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, banner_lines[1], (x_pos, y_pos + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)

    @staticmethod
    def _clamp_draw_point(point: tuple[int, int], frame_width: int, frame_height: int, margin: int) -> tuple[int, int]:
        x_pos = int(np.clip(point[0], margin, max(margin, frame_width - margin - 1)))
        y_pos = int(np.clip(point[1], margin, max(margin, frame_height - margin - 1)))
        return x_pos, y_pos

    @staticmethod
    def _clamp_label_origin(
        label: str,
        center: tuple[int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        desired_x = center[0] + 6
        desired_y = center[1] - 6
        x_pos = int(np.clip(desired_x, 2, max(2, frame_width - label_width - 2)))
        min_y = label_height + baseline + 2
        max_y = max(min_y, frame_height - 2)
        y_pos = int(np.clip(desired_y, min_y, max_y))
        return x_pos, y_pos


def polyline_points(points: deque[tuple[int, int]]) -> np.ndarray:
    return np.asarray(list(points), dtype=np.int32).reshape(-1, 1, 2)
