from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from ..types import AnalysisEvent

_BG      = "#1e1e2e"
_TEXT    = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_GREEN   = "#a6e3a1"
_YELLOW  = "#f9e2af"
_BLUE    = "#89b4fa"
_SURFACE = "#313244"


class LiveView(ctk.CTkFrame):
    """Full-screen video canvas with a thin status header."""

    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")
        self.canvas_image: ImageTk.PhotoImage | None = None

        # ── Thin status header ────────────────────────────────────────
        header = ctk.CTkFrame(self, height=32, corner_radius=8, fg_color=_SURFACE)
        header.pack(fill="x", padx=8, pady=(8, 4))
        header.pack_propagate(False)

        self.status_var = tk.StringVar(value="Load models and select a source.")
        self.fps_var    = tk.StringVar(value="")
        self.prog_var   = tk.StringVar(value="")

        ctk.CTkLabel(header, textvariable=self.status_var, text_color=_YELLOW, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, textvariable=self.fps_var,    text_color=_GREEN,  font=ctk.CTkFont(size=12, weight="bold")).pack(side="right", padx=10)
        ctk.CTkLabel(header, textvariable=self.prog_var,   text_color=_SUBTEXT, font=ctk.CTkFont(size=11)).pack(side="right", padx=10)

        # ── Video canvas — fills EVERYTHING ───────────────────────────
        self.canvas = tk.Canvas(self, bg=_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Public API ────────────────────────────────────────────────────

    def reset(self) -> None:
        self.status_var.set("Running analysis…")
        self.fps_var.set("")
        self.prog_var.set("")
        self.canvas.delete("all")
        self.canvas_image = None

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def update_packet(self, packet) -> None:
        if packet.frame is not None:
            self._render_image(packet.frame)
        self.fps_var.set(f"FPS {packet.fps:.1f}" if packet.fps else "")
        if packet.total_frames > 0:
            pct = packet.frame_index / packet.total_frames * 100
            self.prog_var.set(f"{packet.frame_index}/{packet.total_frames}  ({pct:.0f}%)")
        else:
            self.prog_var.set(f"Frame {packet.frame_index}")
        self.status_var.set(packet.status_text)

    def set_events(self, events: tuple[AnalysisEvent, ...]) -> None:
        """Kept for backward compat — dashboard handles events now."""
        pass

    # ── Private ───────────────────────────────────────────────────────

    def _render_image(self, image) -> None:
        cw = max(64, self.canvas.winfo_width())
        ch = max(64, self.canvas.winfo_height())
        fh, fw = image.shape[:2]
        scale = min(cw / max(1, fw), ch / max(1, fh))
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.cvtColor(cv2.resize(image, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
        self.canvas_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=self.canvas_image)
