from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

from ..core import (
    AnalyticsEngine,
    AnalysisPipeline,
    CourtMapper,
    DetectionTracker,
    FrameRenderer,
    ModelManager,
    SourceResolver,
    VideoWriterService,
    write_events_csv,
    write_summary_json,
)
from ..types import FramePacket


class AppController:
    def __init__(self, window) -> None:
        self.window = window
        self.model_manager = ModelManager()
        self.source_resolver = SourceResolver()
        self.packet_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=4)
        self.critical_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.current_summary = None

        self.window.control_panel.bind_callbacks(
            load_models=self.load_models,
            start_analysis=self.start_analysis,
            stop_analysis=self.stop_analysis,
        )
        self.window.summary_view.bind_export_callbacks(
            export_json=self.export_json,
            export_csv=self.export_csv,
        )
        self.window.root.after(30, self.poll_queues)

    def load_models(self) -> bool:
        config = self.window.control_panel.build_analysis_config()
        try:
            self.model_manager.load(config.court_model_path, config.object_model_path)
        except Exception as exc:
            self.window.live_view.set_status(str(exc))
            messagebox.showerror("Model Load Failed", str(exc))
            return False
        self.window.control_panel.set_class_description(self.model_manager.class_description())
        self.window.live_view.set_status("Models loaded.")
        return True

    def start_analysis(self) -> None:
        if self.worker_thread is not None:
            return
        source_spec = self.window.control_panel.build_source_spec()
        config = self.window.control_panel.build_analysis_config()
        if not source_spec.value:
            messagebox.showerror("Missing Source", "Set a local video path or YouTube URL.")
            return
        if (config.save_video or config.export_data) and config.output_video_path is None:
            messagebox.showerror("Missing Output", "Set an output path for the annotated video and exports.")
            return
        if not self.load_models():
            return

        self.current_summary = None
        self.window.summary_view.clear()
        self.window.live_view.reset()
        self.window.dashboard_view.reset()
        self.window.control_panel.set_running(True)
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._analysis_worker, args=(source_spec, config), daemon=True)
        self.worker_thread.start()
        # Auto-switch to Live tab
        try:
            self.window.tabview.set("\U0001f4f9  Live")
        except Exception:
            pass

    def stop_analysis(self) -> None:
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None
        self.window.control_panel.set_running(False)

    def _analysis_worker(self, source_spec, config) -> None:
        court_mapper = CourtMapper()
        analytics = AnalyticsEngine(court_mapper.px_per_ft)
        detection_tracker = DetectionTracker(self.model_manager.class_map, self.model_manager.role_ids)
        renderer = FrameRenderer(court_mapper.minimap_base, panel_width=config.panel_width)
        pipeline = AnalysisPipeline(
            source_resolver=self.source_resolver,
            model_manager=self.model_manager,
            court_mapper=court_mapper,
            detection_tracker=detection_tracker,
            analytics=analytics,
            renderer=renderer,
            video_writer=VideoWriterService(),
        )
        pipeline.run(source_spec, config, self.stop_event, self._enqueue_packet)

    def _enqueue_packet(self, packet: FramePacket) -> None:
        if packet.packet_type in {"error", "completed"}:
            try:
                self.critical_queue.put_nowait(packet)
            except queue.Full:
                pass
            return
        while self.packet_queue.full():
            try:
                self.packet_queue.get_nowait()
            except queue.Empty:
                break
        self.packet_queue.put_nowait(packet)

    def poll_queues(self) -> None:
        try:
            while True:
                packet = self.critical_queue.get_nowait()
                if packet.packet_type == "error":
                    self.window.live_view.set_status(packet.error_message or packet.status_text)
                    self.stop_analysis()
                    messagebox.showerror("Analysis Error", packet.error_message or packet.status_text)
                elif packet.packet_type == "completed":
                    self.worker_thread = None
                    self.window.control_panel.set_running(False)
                    self.window.live_view.set_status(packet.status_text)
                    if packet.summary is not None:
                        self.current_summary = packet.summary
                        self.window.summary_view.set_summary(packet.summary)
                        self.window.dashboard_view.set_events(packet.summary.recent_events[-5:])
                        # Auto-switch to Summary tab
                        try:
                            self.window.tabview.set("\U0001f4cb  Summary")
                        except Exception:
                            pass
        except queue.Empty:
            pass

        try:
            while True:
                packet = self.packet_queue.get_nowait()
                self.window.live_view.update_packet(packet)
                self.window.dashboard_view.update_packet(packet)
        except queue.Empty:
            pass
        finally:
            self.window.root.after(30, self.poll_queues)

    def export_json(self) -> None:
        if self.current_summary is None:
            return
        path = filedialog.asksaveasfilename(title="Export summary JSON", defaultextension=".json", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        write_summary_json(Path(path), self.current_summary)
        self.window.live_view.set_status(f"Exported JSON: {path}")

    def export_csv(self) -> None:
        if self.current_summary is None:
            return
        path = filedialog.asksaveasfilename(title="Export events CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        write_events_csv(Path(path), self.current_summary.recent_events)
        self.window.live_view.set_status(f"Exported CSV: {path}")
