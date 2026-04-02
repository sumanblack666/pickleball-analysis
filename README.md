# Advanced Pickleball Analysis

A comprehensive desktop application designed for automated pickleball match analysis using computer vision and deep learning.

![Analysis Preview](outputs/analysis_output.mp4) <!-- Replace with a gif or image link if available -->

## Features

- **Object Detection:** Real-time tracking of players, paddles, and the pickleball using YOLOv8/v11.
- **Court Mapping:** Automated court keypoint detection and homography estimation for top-down analytics.
- **Live Analysis:** Integrated GUI for real-time video processing and visualization.
- **Analytics & Insights:**
  - Trajectory interpolation for missing ball detections.
  - Interactive heatmaps of player and ball movements.
  - Event logging and session summaries.
- **YouTube Support:** Directly analyze matches from YouTube URLs using `yt-dlp`.
- **Data Export:** Export results to MP4 (annotated video), CSV (event log), and JSON (summary).

## Repository Structure

```text
.
├── data/               # Sample videos and test data
├── models/             # Pre-trained YOLO models (.pt)
├── outputs/            # Generated analysis reports and videos
├── pickleball_analysis/ # Core package source code
│   ├── core/           # Analysis logic, models, and tracking
│   └── gui/            # CustomTkinter interface components
├── main.py             # Application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pickleball-analysis.git
   cd pickleball-analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure you have `ffmpeg` installed on your system for video processing.*

## Usage

Run the main application using the following command:

```bash
python main.py
```

### CLI Arguments

You can also customize the launch parameters via command line:

```bash
python main.py --source data/test.mp4 --device 0 --imgsz 960 --heatmap
```

- `--source`: Path to a video file or YouTube URL.
- `--device`: Torch device (e.g., `0`, `1`, or `cpu`).
- `--heatmap`: Enable heatmap generation by default.
- `--court-model`: Path to a custom court pose model.
- `--object-model`: Path to a custom object detection model.

## Core Dependencies

- [Ultralytics](https://github.com/ultralytics/ultralytics): Powering the YOLO object detection and pose models.
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter): Providing a modern desktop UI.
- [OpenCV](https://opencv.org/): For high-performance video frame processing.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp): For seamless YouTube video integration.

## License

[MIT](LICENSE) <!-- Update this based on your preference -->
