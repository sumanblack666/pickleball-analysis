from __future__ import annotations

import tkinter as tk
import customtkinter as ctk

from ..types import AnalysisEvent, FramePacket

# ── Catppuccin Mocha palette ──────────────────────────────────────────
_BG        = "#1e1e2e"
_SURFACE   = "#313244"
_CARD      = "#2a2a3c"
_BORDER    = "#45475a"
_TEXT      = "#cdd6f4"
_SUBTEXT   = "#a6adc8"
_BLUE      = "#89b4fa"
_GREEN     = "#a6e3a1"
_YELLOW    = "#f9e2af"
_RED       = "#f38ba8"
_PEACH     = "#fab387"
_MAUVE     = "#cba6f7"
_TEAL      = "#94e2d5"


class _StatCard(ctk.CTkFrame):
    """A single metric card with accent bar, large value, and label."""

    def __init__(
        self,
        master,
        label: str,
        initial_value: str = "—",
        accent: str = _BLUE,
        unit: str = "",
    ) -> None:
        super().__init__(master, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        self._unit = unit

        # Accent stripe (left edge)
        stripe = ctk.CTkFrame(self, width=4, corner_radius=2, fg_color=accent)
        stripe.pack(side="left", fill="y", padx=(6, 0), pady=8)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        self.value_label = ctk.CTkLabel(
            inner,
            text=initial_value,
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=accent,
            anchor="w",
        )
        self.value_label.pack(anchor="w")

        ctk.CTkLabel(
            inner,
            text=label,
            font=ctk.CTkFont(size=11),
            text_color=_SUBTEXT,
            anchor="w",
        ).pack(anchor="w")

    def set(self, value: str) -> None:
        display = f"{value}{self._unit}" if self._unit else value
        self.value_label.configure(text=display)


class _PlayerCard(ctk.CTkFrame):
    """Compact player stats card."""

    def __init__(self, master, title: str, accent: str) -> None:
        super().__init__(master, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        self._accent = accent

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        dot = ctk.CTkFrame(header, width=10, height=10, corner_radius=5, fg_color=accent)
        dot.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            header,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_TEXT,
        ).pack(side="left")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="x", padx=12, pady=(0, 10))
        body.grid_columnconfigure((0, 1), weight=1)

        self.speed_val = ctk.CTkLabel(body, text="0.0", font=ctk.CTkFont(size=20, weight="bold"), text_color=accent, anchor="w")
        self.speed_val.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(body, text="Speed (ft/s)", font=ctk.CTkFont(size=10), text_color=_SUBTEXT, anchor="w").grid(row=1, column=0, sticky="w")

        self.dist_val = ctk.CTkLabel(body, text="0", font=ctk.CTkFont(size=20, weight="bold"), text_color=accent, anchor="w")
        self.dist_val.grid(row=0, column=1, sticky="w", padx=(20, 0))
        ctk.CTkLabel(body, text="Distance (ft)", font=ctk.CTkFont(size=10), text_color=_SUBTEXT, anchor="w").grid(row=1, column=1, sticky="w", padx=(20, 0))

    def set(self, speed: float, distance: float) -> None:
        self.speed_val.configure(text=f"{speed:.1f}")
        self.dist_val.configure(text=f"{distance:.0f}")


class DashboardView(ctk.CTkFrame):
    """Real-time analytics dashboard — stat cards, player panels, event feed."""

    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # ── Row 0 : headline stat cards ──────────────────────────────
        cards_row = ctk.CTkFrame(self, fg_color="transparent")
        cards_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        cards_row.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        self.card_rally   = _StatCard(cards_row, "Rallies",     accent=_MAUVE)
        self.card_shots   = _StatCard(cards_row, "Total Shots", accent=_BLUE)
        self.card_fps     = _StatCard(cards_row, "FPS",         accent=_GREEN)
        self.card_ball_sp = _StatCard(cards_row, "Ball Speed",  accent=_YELLOW, unit=" ft/s")
        self.card_peak    = _StatCard(cards_row, "Peak Speed",  accent=_RED,    unit=" ft/s")
        self.card_hstate  = _StatCard(cards_row, "Court State", accent=_TEAL)

        for idx, card in enumerate(
            [self.card_rally, self.card_shots, self.card_fps, self.card_ball_sp, self.card_peak, self.card_hstate]
        ):
            card.grid(row=0, column=idx, sticky="nsew", padx=5, pady=5)

        # ── Row 1, left column : player panels + detection info ──────
        left_col = ctk.CTkFrame(self, fg_color="transparent")
        left_col.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left_col.grid_rowconfigure(2, weight=1)
        left_col.grid_columnconfigure((0, 1), weight=1)

        self.far_card  = _PlayerCard(left_col, "Far Player",  accent=_PEACH)
        self.near_card = _PlayerCard(left_col, "Near Player", accent=_BLUE)
        self.far_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        self.near_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))

        # Detection / Homography info box
        det_frame = ctk.CTkFrame(left_col, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        det_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))

        ctk.CTkLabel(det_frame, text="Detection Info", font=ctk.CTkFont(size=13, weight="bold"), text_color=_TEAL).pack(anchor="w", padx=12, pady=(10, 4))

        self.det_text = ctk.CTkTextbox(
            det_frame,
            fg_color=_SURFACE,
            text_color=_TEXT,
            font=ctk.CTkFont(family="Consolas", size=12),
            height=120,
            activate_scrollbars=False,
        )
        self.det_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.det_text.configure(state="disabled")

        # Progress
        progress_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky="sew", pady=(5, 0))

        self.progress_label = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=11), text_color=_SUBTEXT)
        self.progress_label.pack(side="left", padx=(0, 10))

        self.progressbar = ctk.CTkProgressBar(progress_frame, height=8)
        self.progressbar.set(0)
        self.progressbar.pack(side="left", fill="x", expand=True)

        # ── Row 1, right column : event feed ─────────────────────────
        event_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        event_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)

        ctk.CTkLabel(event_frame, text="Event Feed", font=ctk.CTkFont(size=13, weight="bold"), text_color=_YELLOW).pack(anchor="w", padx=12, pady=(10, 4))

        self.event_feed = ctk.CTkTextbox(
            event_frame,
            fg_color=_SURFACE,
            text_color=_TEXT,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.event_feed.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.event_feed.configure(state="disabled")

    # ── Public API ────────────────────────────────────────────────────

    def reset(self) -> None:
        for card in (self.card_rally, self.card_shots, self.card_fps, self.card_ball_sp, self.card_peak, self.card_hstate):
            card.set("—")
        self.far_card.set(0.0, 0.0)
        self.near_card.set(0.0, 0.0)
        self.progressbar.set(0)
        self.progress_label.configure(text="")
        self._clear_textbox(self.det_text)
        self._clear_textbox(self.event_feed)

    def update_packet(self, packet: FramePacket) -> None:
        self.card_rally.set(str(packet.rally_count))
        self.card_shots.set(str(packet.shot_count))
        self.card_fps.set(f"{packet.fps:.1f}" if packet.fps else "—")
        self.card_ball_sp.set(f"{packet.ball_speed:.1f}")
        self.card_peak.set(f"{packet.ball_peak_speed:.1f}")

        court_state = "Locked" if packet.court_locked else ("OK" if packet.homography_state == "ok" else "Miss")
        self.card_hstate.set(court_state)

        self.far_card.set(packet.far_speed, packet.far_distance)
        self.near_card.set(packet.near_speed, packet.near_distance)

        # Progress
        if packet.total_frames > 0:
            self.progressbar.set(packet.frame_index / packet.total_frames)
            pct = packet.frame_index / packet.total_frames * 100
            self.progress_label.configure(text=f"Frame {packet.frame_index}/{packet.total_frames}  ({pct:.0f}%)")
        else:
            self.progress_label.configure(text=f"Frame {packet.frame_index}")

        # Detection info
        det_lines = [
            f"Detections: {packet.detections}    Projected: {packet.projected}",
            f"Homography: {packet.homography_state}  (Q: {packet.homography_quality:.1f})",
            f"Keypoints: det={packet.detected_keypoints}  inf={packet.inferred_keypoints}  used={packet.used_keypoints}",
            f"Court: {'LOCKED' if packet.court_locked else 'live'}",
        ]
        self._set_textbox(self.det_text, "\n".join(det_lines))

        # Events
        self.set_events(packet.recent_events)

    def set_events(self, events: tuple[AnalysisEvent, ...]) -> None:
        lines = [f"  {e.timestamp_seconds:7.2f}s  │  {e.title:<14}  │  {e.details}" for e in events]
        self._set_textbox(self.event_feed, "\n".join(lines) if lines else "  No events yet")

    @staticmethod
    def _set_textbox(tb: ctk.CTkTextbox, text: str) -> None:
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        tb.insert("1.0", text)
        tb.configure(state="disabled")

    @staticmethod
    def _clear_textbox(tb: ctk.CTkTextbox) -> None:
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        tb.configure(state="disabled")
