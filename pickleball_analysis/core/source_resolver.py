from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2

from ..types import ResolvedSource, SourceSpec


class SourceResolver:
    def resolve(self, source_spec: SourceSpec) -> ResolvedSource:
        if source_spec.kind == "file":
            video_path = Path(source_spec.value).expanduser()
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            self._validate_video(video_path)
            return ResolvedSource(source_spec=source_spec, video_path=video_path, display_label=str(video_path))
        return self._resolve_youtube(source_spec)

    def cleanup(self, resolved: ResolvedSource) -> None:
        for cleanup_path in resolved.cleanup_paths:
            if cleanup_path.is_dir():
                shutil.rmtree(cleanup_path, ignore_errors=True)
            elif cleanup_path.exists():
                cleanup_path.unlink(missing_ok=True)

    def _resolve_youtube(self, source_spec: SourceSpec) -> ResolvedSource:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required for YouTube URLs") from exc

        temp_dir = Path(tempfile.mkdtemp(prefix="pickleball-analysis-"))
        options = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(temp_dir / "%(title).80s-%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "restrictfilenames": True,
        }

        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(source_spec.value, download=True)

        candidates = [
            path
            for path in temp_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        ]
        if not candidates:
            raise RuntimeError("yt-dlp did not produce a playable video file")

        video_path = max(candidates, key=lambda path: path.stat().st_size)
        self._validate_video(video_path)
        title = str(info.get("title") or source_spec.value)
        return ResolvedSource(
            source_spec=source_spec,
            video_path=video_path,
            display_label=title,
            cleanup_paths=(temp_dir,),
        )

    @staticmethod
    def _validate_video(video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
        finally:
            cap.release()
