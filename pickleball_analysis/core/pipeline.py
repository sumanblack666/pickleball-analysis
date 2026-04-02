from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2

from ..constants import COURT_KEYPOINT_NAMES
from ..types import AnalysisConfig, FramePacket, ProjectedObject, SessionSummary, SourceSpec
from .analytics import AnalyticsEngine
from .court_mapper import CourtMapper, extract_best_court_keypoints
from .model_manager import ModelManager
from .renderer import FrameRenderer
from .source_resolver import SourceResolver
from .tracking import DetectionTracker, TrajectoryInterpolator
from .video_writer import VideoWriterService


class AnalysisPipeline:
    def __init__(
        self,
        source_resolver: SourceResolver,
        model_manager: ModelManager,
        court_mapper: CourtMapper,
        detection_tracker: DetectionTracker,
        analytics: AnalyticsEngine,
        renderer: FrameRenderer,
        video_writer: VideoWriterService,
    ) -> None:
        self.source_resolver = source_resolver
        self.model_manager = model_manager
        self.court_mapper = court_mapper
        self.detection_tracker = detection_tracker
        self.analytics = analytics
        self.renderer = renderer
        self.video_writer = video_writer

    def run(self, source_spec: SourceSpec, config: AnalysisConfig, stop_event, emit_packet) -> SessionSummary | None:
        resolved = None
        cap = None
        processed_frames = 0
        total_frames = 0
        try:
            resolved = self.source_resolver.resolve(source_spec)
            self.model_manager.load(config.court_model_path, config.object_model_path)
            self.detection_tracker.update_mappings(self.model_manager.class_map, self.model_manager.role_ids)
            self.court_mapper.reset()

            cap = cv2.VideoCapture(str(resolved.video_path))
            if not cap.isOpened():
                emit_packet(FramePacket(packet_type="error", status_text="Cannot open video", error_message=f"Cannot open: {resolved.video_path}"))
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if source_fps <= 1e-3:
                source_fps = 30.0

            self.analytics.reset(source_fps, self.court_mapper.minimap_base.shape, resolved.display_label)
            self.detection_tracker.reset(source_fps)
            ball_trace: deque[tuple[int, int]] = deque(maxlen=60)
            far_trace: deque[tuple[int, int]] = deque(maxlen=80)
            near_trace: deque[tuple[int, int]] = deque(maxlen=80)
            ball_interpolator = TrajectoryInterpolator(max_gap=8)
            far_interpolator = TrajectoryInterpolator(max_gap=5)
            near_interpolator = TrajectoryInterpolator(max_gap=5)

            frame_index = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if config.frame_skip > 1 and frame_index % config.frame_skip != 0:
                    continue

                processed_frames += 1
                start_time = time.perf_counter()
                skip_court_inference = config.lock_static_court and self.court_mapper.court_locked
                try:
                    court_result = [] if skip_court_inference else self.model_manager.predict_court(frame, config)
                    object_result = self.model_manager.predict_objects(frame, config)
                except Exception as exc:
                    emit_packet(FramePacket(packet_type="error", status_text="Inference failed", error_message=f"Inference failed: {exc}"))
                    return None

                court_primary = court_result[0] if court_result else None
                object_primary = object_result[0] if object_result else None
                raw_keypoints = extract_best_court_keypoints(court_primary, len(COURT_KEYPOINT_NAMES)) if court_primary else None
                court_update = self.court_mapper.update(
                    keypoints=raw_keypoints,
                    kp_conf_threshold=config.kp_conf,
                    max_stale_frames=config.homography_stale_limit,
                    smooth_alpha=config.homography_smoothing,
                    infer_missing_keypoints=config.infer_missing_court_keypoints,
                    lock_static_court=config.lock_static_court,
                    static_court_stable_frames=config.static_court_stable_frames,
                    static_court_motion_threshold_px=config.static_court_motion_threshold_px,
                    inference_skipped=skip_court_inference,
                )
                keypoints = court_update.keypoints
                homography_state = "ok" if court_update.homography is not None else "miss"

                detections = self.detection_tracker.extract_detections(object_primary) if object_primary else []
                detections = self.detection_tracker.apply_tracking(detections)

                projected_objects: list[ProjectedObject] = []
                for detection in detections:
                    projected_point = self.court_mapper.project_point(detection.anchor_xy)
                    if projected_point is None:
                        continue
                    projected_objects.append(
                        ProjectedObject(
                            role=detection.role,
                            label=detection.label,
                            confidence=detection.confidence,
                            point_xy=projected_point,
                            track_id=detection.track_id,
                        )
                    )

                ball_candidates = [projected for projected in projected_objects if projected.role == "pickleball"]
                best_ball_point = max(ball_candidates, key=lambda item: item.confidence).point_xy if ball_candidates else None
                ball_point, ball_interpolated = ball_interpolator.update(best_ball_point)
                if ball_point:
                    ball_trace.append(ball_point)
                    if ball_interpolated and not ball_candidates:
                        projected_objects.append(ProjectedObject("pickleball", "ball", 0.3, ball_point, interpolated=True))
                    self.analytics.update_ball(ball_point, self.court_mapper.net_y, frame_index)

                persons = sorted([projected for projected in projected_objects if projected.role == "person"], key=lambda item: item.point_xy[1])
                far_point = persons[0].point_xy if persons else None
                near_point = persons[-1].point_xy if len(persons) > 1 else None
                far_point_i, far_interpolated = far_interpolator.update(far_point)
                near_point_i, near_interpolated = near_interpolator.update(near_point)

                if far_point_i:
                    far_trace.append(far_point_i)
                    if far_interpolated and not far_point:
                        projected_objects.append(ProjectedObject("person", "player", 0.3, far_point_i, interpolated=True))
                    self.analytics.update_player("far", far_point_i, frame_index)
                if near_point_i:
                    near_trace.append(near_point_i)
                    if near_interpolated and not near_point:
                        projected_objects.append(ProjectedObject("person", "player", 0.3, near_point_i, interpolated=True))
                    self.analytics.update_player("near", near_point_i, frame_index)

                recent_events = self.analytics.recent_events(5)
                state = self.analytics.state
                live_lines = [
                    f"Frame: {frame_index}/{total_frames if total_frames > 0 else '?'}",
                    f"Detections: {len(detections)}",
                    f"Projected: {len(projected_objects)}",
                    f"Homography: {homography_state} (Q:{court_update.homography_quality:.1f})",
                    f"KPs: det={court_update.detected_keypoints} inf={court_update.inferred_keypoints} used={court_update.used_keypoints}",
                    f"Court: {'locked' if court_update.court_locked else 'live'}{' (skip)' if court_update.inference_skipped else ''}  Stale: {court_update.stale_frames}",
                    f"Classes: {self.model_manager.class_description()}",
                ]
                analytics_lines = [
                    f"Rally: {state.rally_count}  Shots: {state.shot_count}",
                    f"P_far spd: {state.far_speed:.1f} ft/s",
                    f"P_far dist: {state.far_distance:.0f} ft",
                    f"P_near spd: {state.near_speed:.1f} ft/s",
                    f"P_near dist: {state.near_distance:.0f} ft",
                    f"Ball spd: {state.ball_speed:.1f} ft/s",
                    f"Ball peak: {state.ball_peak_speed:.1f} ft/s",
                ]
                output_frame = self.renderer.render_output(
                    frame=frame,
                    keypoints=keypoints,
                    inferred_keypoint_mask=court_update.inferred_mask,
                    kp_conf_threshold=config.kp_conf,
                    detections=detections,
                    projected_objects=projected_objects,
                    ball_trace=ball_trace,
                    far_trace=far_trace,
                    near_trace=near_trace,
                    live_lines=live_lines,
                    analytics_lines=analytics_lines,
                    recent_events=recent_events,
                    heatmap_overlay=self.analytics.heatmap_overlay() if config.heatmap_enabled else None,
                )

                try:
                    if config.save_video:
                        self.video_writer.write(config.output_video_path, source_fps, output_frame)
                except Exception as exc:
                    emit_packet(FramePacket(packet_type="error", status_text="Video writer failed", error_message=str(exc)))
                    return None

                elapsed = max(1e-6, time.perf_counter() - start_time)
                emit_packet(
                    FramePacket(
                        packet_type="frame",
                        status_text=f"det={len(detections)} H={homography_state} kp={court_update.detected_keypoints}+{court_update.inferred_keypoints} {'locked' if court_update.court_locked else 'live'}",
                        frame_index=frame_index,
                        total_frames=total_frames,
                        fps=1.0 / elapsed,
                        detections=len(detections),
                        projected=len(projected_objects),
                        used_keypoints=court_update.used_keypoints,
                        homography_state=homography_state,
                        homography_quality=court_update.homography_quality,
                        frame=output_frame,
                        recent_events=recent_events,
                        live_lines=tuple(live_lines),
                        analytics_lines=tuple(analytics_lines),
                        detected_keypoints=court_update.detected_keypoints,
                        inferred_keypoints=court_update.inferred_keypoints,
                        court_locked=court_update.court_locked,
                        rally_count=state.rally_count,
                        shot_count=state.shot_count,
                        ball_speed=state.ball_speed,
                        ball_peak_speed=state.ball_peak_speed,
                        far_speed=state.far_speed,
                        far_distance=state.far_distance,
                        near_speed=state.near_speed,
                        near_distance=state.near_distance,
                    )
                )

            status_message = "Stopped." if stop_event.is_set() else "Analysis complete"
            summary = self.analytics.build_summary(total_frames, processed_frames, status_message)
            exported_files: tuple[Path, ...] = ()
            if config.export_data and config.output_video_path is not None:
                exported_files = self.analytics.export_summary_files(config.output_video_path, summary)
                summary = replace(summary, exported_files=exported_files)
            if config.save_video and config.output_video_path is not None and not stop_event.is_set():
                status_message += f" Saved video: {config.output_video_path}"
            if exported_files and not stop_event.is_set():
                status_message += " Exported summary data."
            summary = replace(summary, status_message=status_message)
            emit_packet(
                FramePacket(
                    packet_type="completed",
                    status_text=status_message,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    summary=summary,
                    recent_events=self.analytics.recent_events(5),
                )
            )
            return summary
        finally:
            if cap is not None:
                cap.release()
            self.video_writer.close()
            if resolved is not None:
                self.source_resolver.cleanup(resolved)
