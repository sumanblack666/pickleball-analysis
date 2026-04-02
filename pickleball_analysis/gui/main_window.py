from __future__ import annotations

import customtkinter as ctk

from ..types import AppConfig
from .control_panel import ControlPanel
from .dashboard_view import DashboardView
from .live_view import LiveView
from .summary_view import SummaryView
from .theme import apply_dark_theme

_BG    = "#1e1e2e"
_TEXT   = "#cdd6f4"
_BLUE  = "#89b4fa"


class MainWindow:
    def __init__(self, root: ctk.CTk, app_config: AppConfig) -> None:
        self.root = root
        self.root.title(app_config.app_title)
        self.root.geometry("1680x1020")
        self.root.minsize(1200, 700)
        apply_dark_theme(root)

        # Full-window tabview — no control panel eating vertical space
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(
            self.root,
            corner_radius=12,
            segmented_button_fg_color="#313244",
            segmented_button_selected_color=_BLUE,
            segmented_button_selected_hover_color="#74c7ec",
            segmented_button_unselected_color="#313244",
            segmented_button_unselected_hover_color="#45475a",
            text_color=_BG,
            text_color_disabled="#6c7086",
        )
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create tabs
        self.tabview.add("⚙  Setup")
        self.tabview.add("📹  Live")
        self.tabview.add("📊  Dashboard")
        self.tabview.add("📋  Summary")

        # ── Setup tab (control panel) ─────────────────────────────────
        self.control_panel = ControlPanel(self.tabview.tab("⚙  Setup"), app_config)
        self.control_panel.pack(fill="both", expand=True)

        # ── Live tab (full-screen video) ──────────────────────────────
        self.live_view = LiveView(self.tabview.tab("📹  Live"))
        self.live_view.pack(fill="both", expand=True)

        # ── Dashboard tab (real-time analytics) ───────────────────────
        self.dashboard_view = DashboardView(self.tabview.tab("📊  Dashboard"))
        self.dashboard_view.pack(fill="both", expand=True)

        # ── Summary tab (post-analysis) ───────────────────────────────
        self.summary_view = SummaryView(self.tabview.tab("📋  Summary"))
        self.summary_view.pack(fill="both", expand=True)

        # Default to Setup tab
        self.tabview.set("⚙  Setup")
