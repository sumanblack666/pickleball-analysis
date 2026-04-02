from __future__ import annotations

import argparse
import customtkinter as ctk
import tkinter as tk
from pathlib import Path

from pickleball_analysis import (
    APP_TITLE,
    DEFAULT_OBJECT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_VIDEO_PATH,
    AppConfig,
)
from pickleball_analysis.core.common import is_probable_url, resolve_default_court_model
from pickleball_analysis.gui import AppController, MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced pickleball analysis desktop app.")
    parser.add_argument("--court-model", type=Path, default=resolve_default_court_model())
    parser.add_argument("--object-model", type=Path, default=DEFAULT_OBJECT_MODEL)
    parser.add_argument("--source", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-kind", choices=("file", "youtube"), default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--court-conf", type=float, default=0.25)
    parser.add_argument("--object-conf", type=float, default=0.25)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--no-export-data", action="store_true")
    parser.add_argument("--no-infer-missing-keypoints", action="store_true")
    parser.add_argument("--lock-static-court", action="store_true")
    parser.add_argument("--auto-load-models", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_value = args.video if args.video is not None else args.source
    app_config = AppConfig(
        court_model_default=args.court_model,
        object_model_default=args.object_model,
        source_default=source_value,
        output_default=args.output,
        app_title=APP_TITLE,
    )
    root = ctk.CTk()
    window = MainWindow(root, app_config)
    control_panel = window.control_panel
    if args.source_kind is not None:
        control_panel.source_kind_var.set(args.source_kind)
    elif is_probable_url(source_value):
        control_panel.source_kind_var.set("youtube")
    control_panel.device_var.set(args.device)
    control_panel.imgsz_var.set(args.imgsz)
    control_panel.court_conf_var.set(args.court_conf)
    control_panel.object_conf_var.set(args.object_conf)
    control_panel.kp_conf_var.set(args.kp_conf)
    control_panel.frame_skip_var.set(args.frame_skip)
    control_panel.save_video_var.set(not args.no_save_video)
    control_panel.heatmap_var.set(args.heatmap)
    control_panel.export_data_var.set(not args.no_export_data)
    control_panel.infer_missing_keypoints_var.set(not args.no_infer_missing_keypoints)
    control_panel.lock_static_court_var.set(args.lock_static_court)

    controller = AppController(window)
    root.protocol("WM_DELETE_WINDOW", lambda: (_shutdown(controller, root)))
    if args.auto_load_models and args.court_model.exists() and args.object_model.exists():
        controller.load_models()
    root.mainloop()
    return 0


def _shutdown(controller: AppController, root: ctk.CTk) -> None:
    controller.stop_analysis()
    root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
