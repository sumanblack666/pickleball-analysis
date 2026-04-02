from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..constants import COURT_KEYPOINT_NAMES, COURT_SKELETON
from .common import to_numpy

DISPLAY_COMPLETION_CONF_THRESHOLD = 0.65


@dataclass(slots=True)
class CourtUpdateResult:
    keypoints: np.ndarray | None
    inferred_mask: np.ndarray | None
    homography: np.ndarray | None
    detected_keypoints: int
    used_keypoints: int
    inferred_keypoints: int
    stale_frames: int
    homography_quality: float
    inference_skipped: bool
    court_locked: bool


class CourtMapper:
    def __init__(self, minimap_height: int = 440, padding: int = 18) -> None:
        self.template_points, self.minimap_base, self.px_per_ft, self.net_y = self._build_court_template(
            minimap_height,
            padding,
        )
        self.last_h: np.ndarray | None = None
        self.stale_frames = 0
        self.last_detected_keypoints: np.ndarray | None = None
        self.court_locked = False
        self.static_streak = 0
        self.locked_keypoints: np.ndarray | None = None
        self.locked_inferred_mask: np.ndarray | None = None
        self.locked_detected_keypoints = 0
        self.locked_used_keypoints = 0
        self.locked_homography_quality = 0.0

    def reset(self) -> None:
        self.last_h = None
        self.stale_frames = 0
        self.last_detected_keypoints = None
        self.court_locked = False
        self.static_streak = 0
        self.locked_keypoints = None
        self.locked_inferred_mask = None
        self.locked_detected_keypoints = 0
        self.locked_used_keypoints = 0
        self.locked_homography_quality = 0.0

    def update(
        self,
        keypoints: np.ndarray | None,
        kp_conf_threshold: float,
        max_stale_frames: int,
        smooth_alpha: float,
        infer_missing_keypoints: bool,
        lock_static_court: bool,
        static_court_stable_frames: int,
        static_court_motion_threshold_px: float,
        inference_skipped: bool = False,
    ) -> CourtUpdateResult:
        if self.court_locked and lock_static_court and self.locked_keypoints is not None:
            return CourtUpdateResult(
                keypoints=self.locked_keypoints.copy(),
                inferred_mask=None if self.locked_inferred_mask is None else self.locked_inferred_mask.copy(),
                homography=self.last_h,
                detected_keypoints=self.locked_detected_keypoints,
                used_keypoints=self.locked_used_keypoints,
                inferred_keypoints=int(self.locked_inferred_mask.sum()) if self.locked_inferred_mask is not None else 0,
                stale_frames=0,
                homography_quality=self.locked_homography_quality,
                inference_skipped=True,
                court_locked=True,
            )

        normalized_keypoints = normalize_keypoints(keypoints, len(self.template_points))
        completion_conf_threshold = max(kp_conf_threshold, DISPLAY_COMPLETION_CONF_THRESHOLD)
        detected_keypoints = count_detected_keypoints(normalized_keypoints, completion_conf_threshold)
        self.last_h, used_kpts, self.stale_frames, quality = estimate_homography(
            keypoints=normalized_keypoints,
            template_points=self.template_points,
            prev_h=self.last_h,
            kp_conf_threshold=kp_conf_threshold,
            max_stale_frames=max_stale_frames,
            stale_frames=self.stale_frames,
            smooth_alpha=smooth_alpha,
        )
        completed_keypoints: np.ndarray | None = normalized_keypoints
        inferred_mask: np.ndarray | None = np.zeros(len(self.template_points), dtype=bool) if normalized_keypoints is not None else None
        if infer_missing_keypoints and self.last_h is not None:
            completed_keypoints, inferred_mask = complete_keypoints_with_homography(
                normalized_keypoints,
                self.template_points,
                self.last_h,
                completion_conf_threshold,
            )

        if lock_static_court and normalized_keypoints is not None and self.last_h is not None:
            motion = mean_keypoint_motion(self.last_detected_keypoints, normalized_keypoints, kp_conf_threshold)
            if motion is not None and motion <= static_court_motion_threshold_px:
                self.static_streak += 1
            else:
                self.static_streak = 0
            if (
                self.static_streak >= max(1, static_court_stable_frames)
                and completed_keypoints is not None
            ):
                self.court_locked = True
                self.locked_keypoints = completed_keypoints.copy()
                self.locked_inferred_mask = None if inferred_mask is None else inferred_mask.copy()
                self.locked_detected_keypoints = detected_keypoints
                self.locked_used_keypoints = used_kpts
                self.locked_homography_quality = quality
        else:
            self.static_streak = 0

        if normalized_keypoints is not None and detected_keypoints > 0:
            self.last_detected_keypoints = normalized_keypoints.copy()

        return CourtUpdateResult(
            keypoints=completed_keypoints,
            inferred_mask=inferred_mask,
            homography=self.last_h,
            detected_keypoints=detected_keypoints,
            used_keypoints=used_kpts,
            inferred_keypoints=int(inferred_mask.sum()) if inferred_mask is not None else 0,
            stale_frames=self.stale_frames,
            homography_quality=quality,
            inference_skipped=inference_skipped,
            court_locked=self.court_locked,
        )

    def project_point(self, point_xy: tuple[float, float]) -> tuple[int, int] | None:
        if self.last_h is None:
            return None
        src = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.last_h)
        x_pos, y_pos = float(dst[0, 0, 0]), float(dst[0, 0, 1])
        height, width = self.minimap_base.shape[:2]
        return int(round(np.clip(x_pos, 0, width - 1))), int(round(np.clip(y_pos, 0, height - 1)))

    @staticmethod
    def _build_court_template(minimap_height: int, padding: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        court_w_ft, court_l_ft = 20.0, 44.0
        net_y_ft, nvz_ft = 22.0, 7.0
        top_nvz, bot_nvz = net_y_ft - nvz_ft, net_y_ft + nvz_ft

        usable_height = max(120, minimap_height - 2 * padding)
        px_per_ft = usable_height / court_l_ft
        minimap_width = int(round(2 * padding + court_w_ft * px_per_ft))

        def ft_to_px(x_ft: float, y_ft: float) -> tuple[float, float]:
            return padding + x_ft * px_per_ft, padding + y_ft * px_per_ft

        points = {
            "outer_tl": ft_to_px(0, 0),
            "outer_tr": ft_to_px(court_w_ft, 0),
            "outer_br": ft_to_px(court_w_ft, court_l_ft),
            "outer_bl": ft_to_px(0, court_l_ft),
            "nvz_left_top": ft_to_px(0, top_nvz),
            "nvz_right_top": ft_to_px(court_w_ft, top_nvz),
            "nvz_left_bottom": ft_to_px(0, bot_nvz),
            "nvz_right_bottom": ft_to_px(court_w_ft, bot_nvz),
            "center_top": ft_to_px(court_w_ft / 2, 0),
            "center_top_nvz": ft_to_px(court_w_ft / 2, top_nvz),
            "center_bottom_nvz": ft_to_px(court_w_ft / 2, bot_nvz),
            "center_bottom": ft_to_px(court_w_ft / 2, court_l_ft),
            "net_left": ft_to_px(0, net_y_ft),
            "net_right": ft_to_px(court_w_ft, net_y_ft),
        }
        ordered = np.array([points[name] for name in COURT_KEYPOINT_NAMES], dtype=np.float32)
        net_y = padding + net_y_ft * px_per_ft

        base = np.full((minimap_height, minimap_width, 3), (31, 90, 45), dtype=np.uint8)
        line_color, net_color = (230, 230, 230), (255, 215, 0)
        for start, end in COURT_SKELETON:
            point_a = tuple(np.round(ordered[start]).astype(int))
            point_b = tuple(np.round(ordered[end]).astype(int))
            color = net_color if {start, end} == {12, 13} else line_color
            cv2.line(base, point_a, point_b, color, 3 if color == net_color else 2, cv2.LINE_AA)
        cv2.putText(base, "Top-Down Court", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        return ordered, base, px_per_ft, net_y


def extract_best_court_keypoints(result: object, expected_count: int = 14) -> np.ndarray | None:
    kp_obj = getattr(result, "keypoints", None)
    kp_data = to_numpy(getattr(kp_obj, "data", None))
    if kp_data.size == 0:
        return None
    if kp_data.ndim == 2:
        kp_data = np.expand_dims(kp_data, 0)
    boxes_data = to_numpy(getattr(getattr(result, "boxes", None), "data", None))
    if boxes_data.ndim == 1 and boxes_data.size > 0:
        boxes_data = np.expand_dims(boxes_data, 0)

    best_idx, best_score = -1, -1.0
    for index, keypoint_set in enumerate(kp_data):
        detection_conf = (
            float(boxes_data[index][4])
            if boxes_data.size > 0 and index < len(boxes_data) and len(boxes_data[index]) >= 5
            else 0.0
        )
        
        # Prioritize detections with higher number of confident keypoints
        confident_kps = np.sum(keypoint_set[:, 2] > 0.3) if keypoint_set.shape[1] >= 3 else 0
        score = confident_kps + (0.9 * detection_conf)
        
        if len(keypoint_set) <= expected_count and score > best_score:
            best_score, best_idx = score, index

    return np.asarray(kp_data[best_idx], dtype=np.float32) if best_idx >= 0 else None


def normalize_keypoints(keypoints: np.ndarray | None, expected_count: int) -> np.ndarray | None:
    if keypoints is None:
        return None
    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    normalized = np.zeros((expected_count, 3), dtype=np.float32)
    limit = min(len(array), expected_count)
    normalized[:limit, :2] = array[:limit, :2]
    if array.shape[1] >= 3:
        normalized[:limit, 2] = array[:limit, 2]
    else:
        normalized[:limit, 2] = 1.0
    return normalized


def count_detected_keypoints(keypoints: np.ndarray | None, kp_conf_threshold: float) -> int:
    if keypoints is None:
        return 0
    return int(np.count_nonzero(keypoints[:, 2] >= kp_conf_threshold))


def complete_keypoints_with_homography(
    keypoints: np.ndarray | None,
    template_points: np.ndarray,
    homography: np.ndarray,
    kp_conf_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if homography is None:
        return keypoints, None if keypoints is None else np.zeros(len(template_points), dtype=bool)

    completed = normalize_keypoints(keypoints, len(template_points))
    if completed is None:
        completed = np.zeros((len(template_points), 3), dtype=np.float32)
    inferred_mask = np.zeros(len(template_points), dtype=bool)

    try:
        inverse_h = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return completed, inferred_mask

    for index, template_point in enumerate(template_points):
        if completed[index, 2] >= kp_conf_threshold:
            continue
        projected = cv2.perspectiveTransform(np.array([[[template_point[0], template_point[1]]]], dtype=np.float32), inverse_h)
        completed[index, 0] = float(projected[0, 0, 0])
        completed[index, 1] = float(projected[0, 0, 1])
        completed[index, 2] = kp_conf_threshold
        inferred_mask[index] = True
    return completed, inferred_mask


def mean_keypoint_motion(
    previous_keypoints: np.ndarray | None,
    current_keypoints: np.ndarray | None,
    kp_conf_threshold: float,
) -> float | None:
    if previous_keypoints is None or current_keypoints is None:
        return None
        
    limit = min(len(previous_keypoints), len(current_keypoints))
    prev_kps = previous_keypoints[:limit]
    curr_kps = current_keypoints[:limit]
    
    valid_mask = (prev_kps[:, 2] >= kp_conf_threshold) & (curr_kps[:, 2] >= kp_conf_threshold)
    
    if np.sum(valid_mask) < 4:
        return None
        
    displacements = np.linalg.norm(curr_kps[valid_mask, :2] - prev_kps[valid_mask, :2], axis=1)
    return float(np.mean(displacements))


def estimate_homography(
    keypoints: np.ndarray | None,
    template_points: np.ndarray,
    prev_h: np.ndarray | None,
    kp_conf_threshold: float,
    max_stale_frames: int,
    stale_frames: int,
    smooth_alpha: float = 0.80,
) -> tuple[np.ndarray | None, int, int, float]:
    if keypoints is None:
        if prev_h is None:
            return None, 0, stale_frames + 1, 0.0
        next_stale = stale_frames + 1
        return (None if next_stale > max_stale_frames else prev_h), 0, next_stale, 0.0

    src_points: list[tuple[float, float]] = []
    dst_points: list[tuple[float, float]] = []
    limit = min(len(keypoints), len(template_points))
    for index in range(limit):
        keypoint = keypoints[index]
        confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
        if confidence < kp_conf_threshold:
            continue
        src_points.append((float(keypoint[0]), float(keypoint[1])))
        dst_points.append((float(template_points[index][0]), float(template_points[index][1])))

    used = len(src_points)
    if used < 4:
        if prev_h is None:
            return None, used, stale_frames + 1, 0.0
        next_stale = stale_frames + 1
        return (None if next_stale > max_stale_frames else prev_h), used, next_stale, 0.0

    src_np = np.asarray(src_points, dtype=np.float32)
    dst_np = np.asarray(dst_points, dtype=np.float32)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    threshold = 3.0 if method != cv2.RANSAC else 2.0
    homography, _ = cv2.findHomography(src_np, dst_np, method, threshold)

    quality = 0.0
    if homography is not None:
        reprojection = cv2.perspectiveTransform(src_np.reshape(-1, 1, 2), homography)
        errors = np.linalg.norm(reprojection.reshape(-1, 2) - dst_np, axis=1)
        quality = float(np.mean(errors))
        if quality > 15.0:
            if prev_h is None:
                return None, used, stale_frames + 1, quality
            next_stale = stale_frames + 1
            return (None if next_stale > max_stale_frames else prev_h), used, next_stale, quality

    if homography is None:
        if prev_h is None:
            return None, used, stale_frames + 1, 0.0
        next_stale = stale_frames + 1
        return (None if next_stale > max_stale_frames else prev_h), used, next_stale, 0.0

    if prev_h is not None:
        homography = smooth_alpha * prev_h + (1.0 - smooth_alpha) * homography
    if abs(float(homography[2, 2])) > 1e-8:
        homography = homography / float(homography[2, 2])
    return homography.astype(np.float32), used, 0, quality
