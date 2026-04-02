from __future__ import annotations

import tkinter as tk
import customtkinter as ctk

from ..types import SessionSummary

_BG      = "#1e1e2e"
_SURFACE = "#313244"
_CARD    = "#2a2a3c"
_BORDER  = "#45475a"
_TEXT    = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_BLUE    = "#89b4fa"
_GREEN   = "#a6e3a1"
_YELLOW  = "#f9e2af"
_RED     = "#f38ba8"
_PEACH   = "#fab387"
_MAUVE   = "#cba6f7"
_TEAL    = "#94e2d5"


class _MiniStat(ctk.CTkFrame):
    """Summary stat card — title + value."""

    def __init__(self, master, title: str, value: str = "—", accent: str = _BLUE) -> None:
        super().__init__(master, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        stripe = ctk.CTkFrame(self, width=4, corner_radius=2, fg_color=accent)
        stripe.pack(side="left", fill="y", padx=(6, 0), pady=8)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        self.val = ctk.CTkLabel(inner, text=value, font=ctk.CTkFont(size=22, weight="bold"), text_color=accent, anchor="w")
        self.val.pack(anchor="w")
        ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=10), text_color=_SUBTEXT, anchor="w").pack(anchor="w")

    def set(self, value: str) -> None:
        self.val.configure(text=value)


class SummaryView(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")
        self.current_summary: SessionSummary | None = None

        # ── Header row: title + export buttons ────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text="Session Report", font=ctk.CTkFont(size=18, weight="bold"), text_color=_TEXT).pack(side="left")

        self.export_csv_button = ctk.CTkButton(header, text="Export CSV", width=110, state="disabled", fg_color=_SURFACE, hover_color=_BORDER)
        self.export_csv_button.pack(side="right", padx=5)
        self.export_json_button = ctk.CTkButton(header, text="Export JSON", width=110, state="disabled", fg_color=_SURFACE, hover_color=_BORDER)
        self.export_json_button.pack(side="right", padx=5)

        # ── Meta info ─────────────────────────────────────────────────
        self.source_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12), text_color=_SUBTEXT)
        self.source_label.pack(anchor="w", padx=16, pady=(0, 5))
        self.status_label = ctk.CTkLabel(self, text="No analysis completed yet.", font=ctk.CTkFont(size=12), text_color=_YELLOW)
        self.status_label.pack(anchor="w", padx=16, pady=(0, 8))

        # ── Stat cards grid ───────────────────────────────────────────
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=10, pady=(0, 5))
        cards.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.c_frames = _MiniStat(cards, "Processed Frames", accent=_TEAL)
        self.c_duration = _MiniStat(cards, "Duration", accent=_GREEN)
        self.c_rallies = _MiniStat(cards, "Rallies", accent=_MAUVE)
        self.c_shots   = _MiniStat(cards, "Total Shots", accent=_BLUE)
        self.c_peak    = _MiniStat(cards, "Peak Ball Speed", accent=_RED)

        for i, c in enumerate([self.c_frames, self.c_duration, self.c_rallies, self.c_shots, self.c_peak]):
            c.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)

        # ── Player comparison row ─────────────────────────────────────
        players = ctk.CTkFrame(self, fg_color="transparent")
        players.pack(fill="x", padx=10, pady=(0, 5))
        players.grid_columnconfigure((0, 1), weight=1)

        # Far player
        self.far_frame = ctk.CTkFrame(players, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        self.far_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        ctk.CTkLabel(self.far_frame, text="● Far Player", font=ctk.CTkFont(size=13, weight="bold"), text_color=_PEACH).pack(anchor="w", padx=12, pady=(10, 4))
        self.far_detail = ctk.CTkLabel(self.far_frame, text="—", font=ctk.CTkFont(size=12), text_color=_TEXT, justify="left")
        self.far_detail.pack(anchor="w", padx=16, pady=(0, 10))

        # Near player
        self.near_frame = ctk.CTkFrame(players, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        self.near_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        ctk.CTkLabel(self.near_frame, text="● Near Player", font=ctk.CTkFont(size=13, weight="bold"), text_color=_BLUE).pack(anchor="w", padx=12, pady=(10, 4))
        self.near_detail = ctk.CTkLabel(self.near_frame, text="—", font=ctk.CTkFont(size=12), text_color=_TEXT, justify="left")
        self.near_detail.pack(anchor="w", padx=16, pady=(0, 10))

        # ── Timeline ──────────────────────────────────────────────────
        tl_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=_CARD, border_width=1, border_color=_BORDER)
        tl_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        ctk.CTkLabel(tl_frame, text="Session Timeline", font=ctk.CTkFont(size=13, weight="bold"), text_color=_YELLOW).pack(anchor="w", padx=12, pady=(10, 4))
        self.timeline_text = ctk.CTkTextbox(tl_frame, fg_color=_SURFACE, text_color=_TEXT, font=ctk.CTkFont(family="Consolas", size=12))
        self.timeline_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.timeline_text.configure(state="disabled")

    def bind_export_callbacks(self, *, export_json, export_csv) -> None:
        self.export_json_button.configure(command=export_json)
        self.export_csv_button.configure(command=export_csv)

    def clear(self) -> None:
        self.current_summary = None
        self.source_label.configure(text="")
        self.status_label.configure(text="No analysis completed yet.")
        for c in (self.c_frames, self.c_duration, self.c_rallies, self.c_shots, self.c_peak):
            c.set("—")
        self.far_detail.configure(text="—")
        self.near_detail.configure(text="—")
        self.timeline_text.configure(state="normal")
        self.timeline_text.delete("1.0", "end")
        self.timeline_text.configure(state="disabled")
        self.export_json_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")

    def set_summary(self, summary: SessionSummary) -> None:
        self.current_summary = summary

        self.source_label.configure(text=f"Source: {summary.source_label}")
        self.status_label.configure(text=summary.status_message)

        self.c_frames.set(f"{summary.processed_frames}/{summary.total_frames}")
        self.c_duration.set(f"{summary.duration_seconds:.1f}s")
        self.c_rallies.set(str(summary.rally_count))
        self.c_shots.set(str(summary.shot_count))
        self.c_peak.set(f"{summary.ball_peak_speed_ft_s:.1f} ft/s")

        self.far_detail.configure(
            text=f"Speed: {summary.far_speed_ft_s:.1f} ft/s\n"
                 f"Distance: {summary.far_distance_ft:.1f} ft\n"
                 f"Left Occupancy: {summary.far_left_pct:.0%}"
        )
        self.near_detail.configure(
            text=f"Speed: {summary.near_speed_ft_s:.1f} ft/s\n"
                 f"Distance: {summary.near_distance_ft:.1f} ft\n"
                 f"Left Occupancy: {summary.near_left_pct:.0%}"
        )

        # Timeline
        self.timeline_text.configure(state="normal")
        self.timeline_text.delete("1.0", "end")
        header_line = f"  {'Time':<12} │ {'Event':<18} │ Details\n"
        header_line += "  " + "─" * 70 + "\n"
        self.timeline_text.insert("end", header_line)
        for event in summary.recent_events:
            line = f"  {event.timestamp_seconds:<10.2f}s │ {event.title:<18} │ {event.details}\n"
            self.timeline_text.insert("end", line)
        self.timeline_text.configure(state="disabled")

        # Exported files info
        if summary.exported_files:
            self.timeline_text.configure(state="normal")
            self.timeline_text.insert("end", "\n  Exported files:\n")
            for p in summary.exported_files:
                self.timeline_text.insert("end", f"    • {p}\n")
            self.timeline_text.configure(state="disabled")

        self.export_json_button.configure(state="normal")
        self.export_csv_button.configure(state="normal")
