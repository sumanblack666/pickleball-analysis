from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk

from ..core.common import is_probable_url
from ..types import AnalysisConfig, AppConfig, SourceSpec


class ControlPanel(ctk.CTkFrame):
    def __init__(self, master, app_config: AppConfig) -> None:
        super().__init__(master, corner_radius=10, fg_color="transparent")
        default_source_kind = "youtube" if is_probable_url(app_config.source_default) else "file"
        self.court_model_path_var = tk.StringVar(value=str(app_config.court_model_default))
        self.object_model_path_var = tk.StringVar(value=str(app_config.object_model_default))
        self.source_kind_var = tk.StringVar(value=default_source_kind)
        self.source_value_var = tk.StringVar(value=str(app_config.source_default))
        self.output_path_var = tk.StringVar(value=str(app_config.output_default))
        self.device_var = tk.StringVar(value="0")
        self.imgsz_var = tk.IntVar(value=960)
        self.court_conf_var = tk.DoubleVar(value=0.25)
        self.object_conf_var = tk.DoubleVar(value=0.25)
        self.kp_conf_var = tk.DoubleVar(value=0.35)
        self.frame_skip_var = tk.IntVar(value=1)
        self.save_video_var = tk.BooleanVar(value=True)
        self.half_precision_var = tk.BooleanVar(value=True)
        self.heatmap_var = tk.BooleanVar(value=False)
        self.export_data_var = tk.BooleanVar(value=True)
        self.infer_missing_keypoints_var = tk.BooleanVar(value=True)
        self.lock_static_court_var = tk.BooleanVar(value=False)
        self.class_map_var = tk.StringVar(value="classes: not loaded")

        self.load_models_button: ctk.CTkButton | None = None
        self.start_button: ctk.CTkButton | None = None
        self.stop_button: ctk.CTkButton | None = None
        self.source_browse_button: ctk.CTkButton | None = None
        self._build()

    def _build(self) -> None:
        # Setup Frame
        config_frame = ctk.CTkFrame(self, corner_radius=8)
        config_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(config_frame, text="Configurations", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5))

        rows = [
            ("Court Pose Model", self.court_model_path_var, self._browse_court_model),
            ("Object Detect Model", self.object_model_path_var, self._browse_object_model),
            ("Source Value", self.source_value_var, self._browse_source),
            ("Output Base / Video", self.output_path_var, self._browse_output),
        ]
        
        for row_index, (label, variable, command) in enumerate(rows, start=1):
            ctk.CTkLabel(config_frame, text=label).grid(row=row_index, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkEntry(config_frame, textvariable=variable, width=600).grid(row=row_index, column=1, sticky="ew", padx=(10, 5), pady=5)
            button = ctk.CTkButton(config_frame, text="Browse", width=80, command=command)
            button.grid(row=row_index, column=2, padx=10, pady=5)
            if label == "Source Value":
                self.source_browse_button = button
                
        ctk.CTkLabel(config_frame, text="Source Type").grid(row=5, column=0, sticky="w", padx=10, pady=(5, 10))
        ctk.CTkOptionMenu(config_frame, variable=self.source_kind_var, values=["file", "youtube"], width=120).grid(row=5, column=1, sticky="w", padx=(10, 5), pady=(5, 10))
        config_frame.columnconfigure(1, weight=1)

        # Params Frame
        params_frame = ctk.CTkFrame(self, corner_radius=8)
        params_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(params_frame, text="Runtime Parameters", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=6, sticky="w", padx=10, pady=(10, 5))

        items = [
            ("Device", self.device_var, 80),
            ("ImgSz", self.imgsz_var, 80),
            ("Court Conf", self.court_conf_var, 80),
            ("Object Conf", self.object_conf_var, 80),
            ("KP Conf", self.kp_conf_var, 80),
            ("Frame Skip", self.frame_skip_var, 80),
        ]
        
        inputs_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        inputs_frame.grid(row=1, column=0, columnspan=6, sticky="w", padx=10, pady=5)
        
        for column, (label, variable, width) in enumerate(items):
            ctk.CTkLabel(inputs_frame, text=label).grid(row=0, column=column * 2, padx=(10, 5), pady=5)
            ctk.CTkEntry(inputs_frame, textvariable=variable, width=width).grid(row=0, column=column * 2 + 1, padx=(0, 10), pady=5)

        switches_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        switches_frame.grid(row=2, column=0, columnspan=6, sticky="w", padx=10, pady=(5, 10))
        
        toggles = [
            ("Save Video", self.save_video_var),
            ("FP16", self.half_precision_var),
            ("Heatmap", self.heatmap_var),
            ("Export JSON/CSV", self.export_data_var),
            ("Infer Missing Court KPs", self.infer_missing_keypoints_var),
            ("Lock Static Court", self.lock_static_court_var),
        ]
        
        for column, (text, var) in enumerate(toggles):
            ctk.CTkSwitch(switches_frame, text=text, variable=var).grid(row=0, column=column, padx=(10, 15), pady=5)

        ctk.CTkLabel(params_frame, textvariable=self.class_map_var, text_color="gray").grid(row=3, column=0, columnspan=6, sticky="w", padx=10, pady=(0, 10))

        # Actions Frame
        actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        actions_frame.pack(fill="x", pady=(5, 0))
        
        self.load_models_button = ctk.CTkButton(actions_frame, text="Load Models", fg_color="#3b82f6", hover_color="#2563eb", font=ctk.CTkFont(weight="bold"))
        self.load_models_button.pack(side="left", padx=(0, 10))
        
        self.start_button = ctk.CTkButton(actions_frame, text="Start Analysis", fg_color="#10b981", hover_color="#059669", font=ctk.CTkFont(weight="bold"))
        self.start_button.pack(side="left", padx=10)
        
        self.stop_button = ctk.CTkButton(actions_frame, text="Stop", fg_color="#ef4444", hover_color="#dc2626", state="disabled", font=ctk.CTkFont(weight="bold"))
        self.stop_button.pack(side="left", padx=10)

        self.source_kind_var.trace_add("write", lambda *_: self._refresh_source_controls())
        self._refresh_source_controls()

    def bind_callbacks(self, *, load_models, start_analysis, stop_analysis) -> None:
        assert self.load_models_button and self.start_button and self.stop_button
        self.load_models_button.configure(command=load_models)
        self.start_button.configure(command=start_analysis)
        self.stop_button.configure(command=stop_analysis)

    def set_running(self, running: bool) -> None:
        assert self.start_button and self.stop_button
        if running:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def set_class_description(self, description: str) -> None:
        self.class_map_var.set(f"classes: {description}")

    def build_source_spec(self) -> SourceSpec:
        val = self.source_kind_var.get()
        kind = val.strip() if val else "file"
        return SourceSpec(kind=kind, value=self.source_value_var.get().strip())

    def build_analysis_config(self) -> AnalysisConfig:
        output_value = self.output_path_var.get().strip()
        output_path = Path(output_value).expanduser() if output_value else None
        return AnalysisConfig(
            court_model_path=Path(self.court_model_path_var.get()).expanduser(),
            object_model_path=Path(self.object_model_path_var.get()).expanduser(),
            output_video_path=output_path,
            device=self.device_var.get().strip() or "0",
            imgsz=max(160, int(self.imgsz_var.get())),
            court_conf=float(self.court_conf_var.get()),
            object_conf=float(self.object_conf_var.get()),
            kp_conf=float(self.kp_conf_var.get()),
            frame_skip=max(1, int(self.frame_skip_var.get())),
            half_precision=bool(self.half_precision_var.get()),
            save_video=bool(self.save_video_var.get()),
            heatmap_enabled=bool(self.heatmap_var.get()),
            export_data=bool(self.export_data_var.get()),
            infer_missing_court_keypoints=bool(self.infer_missing_keypoints_var.get()),
            lock_static_court=bool(self.lock_static_court_var.get()),
        )

    def _refresh_source_controls(self) -> None:
        if self.source_browse_button is None:
            return
        if self.source_kind_var.get() == "file":
             self.source_browse_button.configure(state="normal")
        else:
             self.source_browse_button.configure(state="disabled")

    def _browse_court_model(self) -> None:
        path = filedialog.askopenfilename(title="Court pose model", filetypes=[("Model", "*.pt *.onnx"), ("All", "*.*")])
        if path:
            self.court_model_path_var.set(path)

    def _browse_object_model(self) -> None:
        path = filedialog.askopenfilename(title="Object model", filetypes=[("Model", "*.pt *.onnx"), ("All", "*.*")])
        if path:
            self.object_model_path_var.set(path)

    def _browse_source(self) -> None:
        if self.source_kind_var.get() != "file":
            return
        path = filedialog.askopenfilename(title="Input video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if path:
            self.source_value_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Output video", defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All", "*.*")])
        if path:
            self.output_path_var.set(path)
