"""
Cricket Pose Analyzer AI - Utility Functions Module
===================================================
Provides file I/O operations, formatting functions, ZIP archive generator,
temporary file cleaners, math helpers, and data serialization.
"""

import json
import logging
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cricket_pose_ai.config import (
    FRAMES_DIR,
    GRAPHS_DIR,
    HEATMAPS_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CricketPoseAI.Utils")


def setup_logger(name: str = "CricketPoseAI") -> logging.Logger:
    """Returns a configured logger instance."""
    return logging.getLogger(name)


def clean_temp_files():
    """Removes all temporary files from TEMP_DIR."""
    try:
        if TEMP_DIR.exists():
            for item in TEMP_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("Cleaned temporary directory.")
    except Exception as e:
        logger.warning(f"Failed to clean temp directory: {e}")


def export_to_csv(data: List[Dict[str, Any]], filepath: Path) -> bool:
    """Exports a list of frame-wise dictionary metrics to a CSV file."""
    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully exported CSV to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting CSV to {filepath}: {e}")
        return False


def export_to_json(data: Dict[str, Any], filepath: Path) -> bool:
    """Exports structured metrics/summary to a JSON file."""
    try:
        def convert_types(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            return obj

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=convert_types)
        logger.info(f"Successfully exported JSON to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting JSON to {filepath}: {e}")
        return False


def create_analysis_zip(output_zip_path: Path) -> Optional[Path]:
    """
    Packs all generated artifacts (processed video, reports, CSV, JSON,
    graphs, heatmaps, annotated frames) into a single downloadable ZIP archive.
    """
    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.resolve() == output_zip_path.resolve():
                        continue
                    arcname = file_path.relative_to(OUTPUT_DIR)
                    zipf.write(file_path, arcname)

        logger.info(f"Created comprehensive ZIP bundle at {output_zip_path}")
        return output_zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP bundle: {e}")
        return None


def calculate_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def smooth_series(series: List[float], window_size: int = 7) -> np.ndarray:
    """Applies moving average smoothing to a numeric array."""
    arr = np.array(series, dtype=float)
    if len(arr) < window_size or window_size < 3:
        return arr
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(arr, kernel, mode="same")
    # Restore boundary values
    half = window_size // 2
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return smoothed


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.ms string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


def ensure_h264_encoding(video_path: Path) -> Path:
    """
    Re-encodes output MP4 video to standard H.264 (avc1/yuv420p) video format
    so web browsers and Streamlit st.video can display it smoothly.
    """
    if not video_path.exists():
        return video_path

    # 1. Try FFmpeg CLI if available
    try:
        import subprocess
        temp_out = video_path.with_name(video_path.stem + "_h264.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-an", str(temp_out)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if res.returncode == 0 and temp_out.exists() and temp_out.stat().st_size > 0:
            shutil.move(str(temp_out), str(video_path))
            logger.info(f"FFmpeg re-encoded {video_path.name} to H.264 for HTML5 web playback.")
            return video_path
    except Exception:
        pass

    # 2. Try MoviePy fallback
    try:
        from moviepy.editor import VideoFileClip
        temp_out = video_path.with_name(video_path.stem + "_h264.mp4")
        clip = VideoFileClip(str(video_path))
        clip.write_videofile(str(temp_out), codec="libx264", audio=False, verbose=False, logger=None)
        clip.close()
        if temp_out.exists() and temp_out.stat().st_size > 0:
            shutil.move(str(temp_out), str(video_path))
            logger.info(f"MoviePy re-encoded {video_path.name} to H.264 for web playback.")
            return video_path
    except Exception:
        pass

    return video_path

