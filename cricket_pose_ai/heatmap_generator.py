"""
Cricket Pose Analyzer AI - Spatial Motion Heatmap Module
=========================================================
Generates 2D kernel density heatmaps for spatial trajectory tracking of
wrist, bat tip, head, foot placement, and Center of Mass movement.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from cricket_pose_ai.config import HEATMAPS_DIR
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.HeatmapGenerator")


class MotionHeatmapGenerator:
    """Renders high-resolution 2D motion density heatmaps."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or HEATMAPS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_heatmaps(
        self,
        wrist_pts: List[Tuple[int, int]],
        bat_pts: List[Tuple[int, int]],
        head_pts: List[Tuple[int, int]],
        foot_pts: List[Tuple[int, int]],
        com_pts: List[Tuple[int, int]],
        img_w: int,
        img_h: int,
        background_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Path]:
        """
        Renders and saves individual heatmap PNG files for Wrist, Bat, Head, Foot, and COM.
        Returns a dictionary mapping category names to generated file paths.
        """
        results = {}

        data_map = {
            "wrist_heatmap": ("Wrist Movement Heatmap", wrist_pts, "plasma"),
            "bat_heatmap": ("Bat Tip Trajectory Heatmap", bat_pts, "inferno"),
            "head_heatmap": ("Head Stability Heatmap", head_pts, "viridis"),
            "foot_heatmap": ("Foot Placement Heatmap", foot_pts, "hot"),
            "com_heatmap": ("Center of Mass Drift Heatmap", com_pts, "magma")
        }

        for key, (title, pts, cmap) in data_map.items():
            filepath = self.output_dir / f"{key}.png"
            saved_path = self._render_single_heatmap(
                title=title,
                pts=pts,
                cmap_name=cmap,
                img_w=img_w,
                img_h=img_h,
                output_path=filepath,
                background_frame=background_frame
            )
            if saved_path:
                results[key] = saved_path

        logger.info(f"Generated {len(results)} spatial motion heatmaps in {self.output_dir}")
        return results

    @staticmethod
    def _filter_batsman_outliers(pts: List[Tuple[int, int]], img_w: int, img_h: int) -> List[Tuple[int, int]]:
        """
        Filters out spatial outlier points belonging to non-batsman persons
        (e.g., bowler running in from boundary, keeper behind stumps).
        """
        if len(pts) < 5:
            return pts

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        med_x = np.median(xs)
        med_y = np.median(ys)

        mad_x = np.median(np.abs(xs - med_x)) + 1e-5
        mad_y = np.median(np.abs(ys - med_y)) + 1e-5

        max_dist_x = max(img_w * 0.28, 3.2 * mad_x)
        max_dist_y = max(img_h * 0.32, 3.2 * mad_y)

        filtered = [
            (x, y) for (x, y) in pts
            if abs(x - med_x) <= max_dist_x and abs(y - med_y) <= max_dist_y
        ]
        return filtered if len(filtered) >= 3 else pts

    def _render_single_heatmap(
        self,
        title: str,
        pts: List[Tuple[int, int]],
        cmap_name: str,
        img_w: int,
        img_h: int,
        output_path: Path,
        background_frame: Optional[np.ndarray] = None
    ) -> Optional[Path]:
        """Creates a smooth 2D Gaussian density map overlaid on background frame or dark canvas."""
        try:
            # Filter spatial outliers (non-batsman persons)
            clean_pts = self._filter_batsman_outliers(pts, img_w, img_h)

            # Create density grid
            density = np.zeros((img_h, img_w), dtype=np.float32)

            valid_pts = [(x, y) for (x, y) in clean_pts if 0 <= x < img_w and 0 <= y < img_h]
            if not valid_pts:
                logger.warning(f"No valid points for {title}")
                return None

            for x, y in valid_pts:
                density[y, x] += 1.0

            # Smooth density using Gaussian blur
            sigma = max(8.0, min(img_w, img_h) * 0.02)
            smoothed_density = gaussian_filter(density, sigma=sigma)

            # Normalize [0, 1]
            max_val = np.max(smoothed_density)
            if max_val > 0:
                smoothed_density /= max_val

            # Setup Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#0F172A')

            # Draw background image if provided
            if background_frame is not None:
                bg_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
                ax.imshow(bg_rgb, alpha=0.45)
            else:
                ax.imshow(np.zeros((img_h, img_w, 3), dtype=np.uint8))

            # Overlay heatmap density
            im = ax.imshow(smoothed_density, cmap=cmap_name, alpha=0.75, origin='upper')

            # Scatter trajectory points
            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]
            ax.scatter(xs, ys, color='#38BDF8', s=12, alpha=0.6, edgecolors='none')

            ax.set_title(title, color='#F8FAFC', fontsize=14, fontweight='bold', pad=12)
            ax.axis('off')

            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='#F8FAFC')
            plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#94A3B8')

            plt.tight_layout()
            plt.savefig(output_path, facecolor='#0F172A', edgecolor='none')
            plt.close(fig)
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate heatmap {title}: {e}")
            return None
