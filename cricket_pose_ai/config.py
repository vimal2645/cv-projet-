"""
Cricket Pose Analyzer AI - Configuration Module
================================================
Defines configuration settings, data structures, thresholds, color palettes,
and biomechanical reference benchmarks for cricket batting analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
ASSETS_DIR = BASE_DIR / "assets"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
FRAMES_DIR = OUTPUT_DIR / "annotated_frames"
SESSIONS_FILE = OUTPUT_DIR / "sessions.json"

# Ensure directories exist
for folder in [OUTPUT_DIR, TEMP_DIR, ASSETS_DIR, GRAPHS_DIR, HEATMAPS_DIR, FRAMES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class PoseConfig:
    """MediaPipe Pose detection configuration."""
    static_image_mode: bool = False
    model_complexity: int = 2  # 0, 1, or 2 (2 is most accurate)
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class BatConfig:
    """Bat detection & swing tracking configuration."""
    use_yolo: bool = True
    yolo_model_name: str = "yolov8n.pt"  # Will fallback to HSV/contour heuristics if YOLO model not loaded
    confidence_threshold: float = 0.35
    bat_color_hsv_min: Tuple[int, int, int] = (0, 0, 50)
    bat_color_hsv_max: Tuple[int, int, int] = (180, 50, 255)
    max_trail_points: int = 30


@dataclass
class VisualConfig:
    """Styling & visual rendering parameters."""
    # Dark modern theme palette (BGR format for OpenCV)
    PRIMARY_NEON: Tuple[int, int, int] = (255, 230, 0)      # Neon Cyan/Yellow (255, 230, 0)
    SKELETON_COLOR: Tuple[int, int, int] = (245, 190, 40)   # Cyan Accent
    JOINT_COLOR: Tuple[int, int, int] = (50, 230, 80)       # Emerald Green
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (30, 144, 255)  # Dodger Blue
    WARNING_COLOR: Tuple[int, int, int] = (50, 50, 255)     # Coral Red
    BAT_COLOR: Tuple[int, int, int] = (0, 215, 255)         # Bright Amber/Gold
    COM_COLOR: Tuple[int, int, int] = (255, 105, 180)       # Hot Pink
    HUD_BG_COLOR: Tuple[int, int, int] = (20, 24, 33)       # Dark Slate

    # Thickness and Fonts
    line_thickness: int = 2
    joint_radius: int = 5
    font_scale: float = 0.55
    hud_height: int = 70

    # Overlay Toggles
    draw_skeleton: bool = True
    draw_angles: bool = True
    draw_hud: bool = True
    draw_motion_trails: bool = True
    draw_com: bool = True
    draw_velocity_vectors: bool = True


@dataclass
class CricketBenchmarks:
    """Biomechanical reference benchmarks for Cricket Batting technique."""
    # Optimal Ranges in degrees or indices
    ideal_backlift_elbow_angle: Tuple[float, float] = (120.0, 160.0)  # High backlift elbow angle
    ideal_downswing_elbow_angle: Tuple[float, float] = (90.0, 140.0)
    ideal_impact_spine_angle: Tuple[float, float] = (10.0, 28.0)      # Balanced forward lean
    ideal_followthrough_extension: Tuple[float, float] = (145.0, 175.0)
    max_acceptable_head_drift_ratio: float = 0.06                   # 6% of image height
    ideal_stance_width_ratio: Tuple[float, float] = (0.8, 1.4)        # Stance width relative to shoulder width


@dataclass
class AnalysisConfig:
    """General analysis and scoring configuration."""
    pose: PoseConfig = field(default_factory=PoseConfig)
    bat: BatConfig = field(default_factory=BatConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    benchmarks: CricketBenchmarks = field(default_factory=CricketBenchmarks)
    smoothing_window: int = 7
    fps_default: float = 30.0


# Color Schemes for Dashboard & Reports (HEX)
THEME_COLORS = {
    "background": "#0F172A",
    "surface": "#1E293B",
    "primary": "#38BDF8",
    "secondary": "#34D399",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text": "#F8FAFC",
    "text_muted": "#94A3B8"
}
