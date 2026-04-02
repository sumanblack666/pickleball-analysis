from __future__ import annotations

from pathlib import Path


DEFAULT_COURT_MODEL = Path("models/court_best.pt")
DEFAULT_OBJECT_MODEL = Path("models/ball,person,paddle.pt")
DEFAULT_VIDEO_PATH = Path("data/test.mp4")
DEFAULT_OUTPUT_PATH = Path("outputs/analysis_output.mp4")

COURT_KEYPOINT_NAMES = [
    "outer_tl",
    "outer_tr",
    "outer_br",
    "outer_bl",
    "nvz_left_top",
    "nvz_right_top",
    "nvz_left_bottom",
    "nvz_right_bottom",
    "center_top",
    "center_top_nvz",
    "center_bottom_nvz",
    "center_bottom",
    "net_left",
    "net_right",
]

COURT_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (4, 12),
    (12, 6),
    (5, 13),
    (13, 7),
]

ROLE_ALIASES = {
    "paddle": ("paddle", "racket"),
    "person": ("person", "player", "human"),
    "pickleball": ("pickleball", "ball"),
}

ROLE_COLORS_BGR = {
    "paddle": (219, 112, 147),
    "person": (255, 153, 51),
    "pickleball": (0, 215, 255),
    "unknown": (180, 180, 180),
}

APP_TITLE = "Advanced Pickleball Analysis"
