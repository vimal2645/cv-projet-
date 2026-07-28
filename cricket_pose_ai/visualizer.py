"""
Cricket Pose Analyzer AI - Visualizer & Graphics Module
========================================================
OpenCV HUD frame annotation (skeleton, glowing keypoints, motion trails, COM,
velocity vectors) and Plotly/Matplotlib high-res chart generation.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cricket_pose_ai.angle_calculator import FrameBiomechanics
from cricket_pose_ai.bat_detector import BatDetectionResult
from cricket_pose_ai.config import GRAPHS_DIR, VisualConfig
from cricket_pose_ai.motion_analyzer import MotionSummaryMetrics
from cricket_pose_ai.pose_detector import Landmark
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.Visualizer")


class PoseVisualizer:
    """Renders video overlays and high-resolution charts."""

    # Key skeletal connections (MediaPipe landmark pairs)
    SKELETON_CONNECTIONS = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("NOSE", "LEFT_SHOULDER"),
        ("NOSE", "RIGHT_SHOULDER")
    ]

    def __init__(self, config: Optional[VisualConfig] = None):
        self.config = config or VisualConfig()
        self.wrist_trail = deque(maxlen=25)
        self.bat_trail = deque(maxlen=25)

    def draw_frame_annotations(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[Dict[str, Landmark]],
        biomechanics: Optional[FrameBiomechanics],
        bat_result: Optional[BatDetectionResult],
        phase_label: str,
        frame_idx: int,
        total_frames: int,
        fps: float
    ) -> np.ndarray:
        """Annotates frame with pose skeleton, joint badges, HUD, motion trails, and bat vector."""
        if frame_bgr is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        if landmarks and self.config.draw_skeleton:
            # 1. Draw Skeleton Lines
            for p1_name, p2_name in self.SKELETON_CONNECTIONS:
                if p1_name in landmarks and p2_name in landmarks:
                    lm1 = landmarks[p1_name]
                    lm2 = landmarks[p2_name]
                    if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                        cv2.line(
                            annotated,
                            (lm1.px, lm1.py),
                            (lm2.px, lm2.py),
                            self.config.SKELETON_COLOR,
                            self.config.line_thickness,
                            cv2.LINE_AA
                        )

            # 2. Draw Keypoint Glowing Circles
            for name, lm in landmarks.items():
                if lm.visibility > 0.4:
                    cv2.circle(annotated, (lm.px, lm.py), self.config.joint_radius + 2, (0, 0, 0), -1)
                    cv2.circle(annotated, (lm.px, lm.py), self.config.joint_radius, self.config.JOINT_COLOR, -1)

            # 3. Motion Trails for Wrists & Bat
            if self.config.draw_motion_trails:
                rw = landmarks.get("RIGHT_WRIST")
                if rw:
                    self.wrist_trail.append((rw.px, rw.py))

                if bat_result and bat_result.detected:
                    self.bat_trail.append(bat_result.bat_tip)

                # Draw wrist trail
                for i in range(1, len(self.wrist_trail)):
                    alpha = i / len(self.wrist_trail)
                    thickness = int(1 + alpha * 3)
                    cv2.line(annotated, self.wrist_trail[i-1], self.wrist_trail[i], (255, 100, 0), thickness, cv2.LINE_AA)

                # Draw bat trail
                for i in range(1, len(self.bat_trail)):
                    alpha = i / len(self.bat_trail)
                    thickness = int(2 + alpha * 4)
                    cv2.line(annotated, self.bat_trail[i-1], self.bat_trail[i], self.config.BAT_COLOR, thickness, cv2.LINE_AA)

        # 4. Bat Vector & Box
        if bat_result and bat_result.detected:
            cv2.line(annotated, bat_result.bat_handle, bat_result.bat_tip, self.config.BAT_COLOR, 3, cv2.LINE_AA)
            cv2.circle(annotated, bat_result.bat_tip, 6, (0, 255, 255), -1)

            if bat_result.bbox:
                x1, y1, x2, y2 = bat_result.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.config.BAT_COLOR, 1)

        # 5. Center of Mass Indicator
        if biomechanics and self.config.draw_com:
            cx, cy = biomechanics.com_px, biomechanics.com_py
            cv2.circle(annotated, (cx, cy), 10, self.config.COM_COLOR, 2, cv2.LINE_AA)
            cv2.line(annotated, (cx - 14, cy), (cx + 14, cy), self.config.COM_COLOR, 2, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - 14), (cx, cy + 14), self.config.COM_COLOR, 2, cv2.LINE_AA)
            cv2.putText(annotated, "COM", (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.config.COM_COLOR, 1, cv2.LINE_AA)

        # 6. Joint Angle Overlay Badges
        if biomechanics and landmarks and self.config.draw_angles:
            re = landmarks.get("RIGHT_ELBOW")
            if re:
                self._draw_angle_badge(annotated, f"R-Elbow: {biomechanics.right_elbow_angle}°", (re.px + 10, re.py))

            rk = landmarks.get("RIGHT_KNEE")
            if rk:
                self._draw_angle_badge(annotated, f"R-Knee: {biomechanics.right_knee_angle}°", (rk.px + 10, rk.py))

            rs = landmarks.get("RIGHT_SHOULDER")
            if rs:
                self._draw_angle_badge(annotated, f"Spine: {biomechanics.spine_angle}°", (rs.px - 90, rs.py - 15))

        # 7. Professional Top HUD Overlay Bar
        if self.config.draw_hud:
            self._draw_hud_bar(annotated, phase_label, frame_idx, total_frames, fps, biomechanics)

        return annotated

    def _draw_angle_badge(self, img: np.ndarray, text: str, pos: Tuple[int, int]):
        """Draws small background pill with text."""
        x, y = pos
        (w_t, h_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x - 2, y - h_t - 4), (x + w_t + 4, y + 4), (15, 23, 42), -1)
        cv2.rectangle(img, (x - 2, y - h_t - 4), (x + w_t + 4, y + 4), (56, 189, 248), 1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud_bar(
        self,
        img: np.ndarray,
        phase: str,
        frame_idx: int,
        total_frames: int,
        fps: float,
        bm: Optional[FrameBiomechanics]
    ):
        """Draws top HUD bar with metrics."""
        h, w, _ = img.shape
        hud_h = self.config.hud_height

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), self.config.HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, hud_h), (w, hud_h), (56, 189, 248), 2)

        # Content Text
        t1 = f"CRICKET POSE ANALYZER AI  |  FRAME: {frame_idx}/{total_frames}  ({fps:.1f} FPS)"
        t2 = f"SHOT PHASE: {phase.upper()}"

        cv2.putText(img, t1, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, t2, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (56, 189, 248), 2, cv2.LINE_AA)

        if bm:
            t3 = f"Stance Ratio: {bm.stance_width_ratio:.2f} | Head Tilt: {bm.head_tilt}°"
            cv2.putText(img, t3, (w - 320, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 230, 80), 1, cv2.LINE_AA)

    def generate_analytics_charts(
        self,
        framewise_records: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Generates high-res PNG plots for joint angles, velocities, posture, head drift, and timeline."""
        out_folder = output_dir or GRAPHS_DIR
        out_folder.mkdir(parents=True, exist_ok=True)

        frames = [r["frame"] for r in framewise_records]
        r_elbows = [r["right_elbow_angle"] for r in framewise_records]
        r_knees = [r["right_knee_angle"] for r in framewise_records]
        spines = [r["spine_angle"] for r in framewise_records]
        wrist_vels = [r["wrist_velocity_px_s"] for r in framewise_records]
        bat_vels = [r["bat_velocity_px_s"] for r in framewise_records]
        head_xs = [r["head_x"] for r in framewise_records]
        head_ys = [r["head_y"] for r in framewise_records]

        generated = {}

        # 1. Joint Angles Plot
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        ax.plot(frames, r_elbows, label='Right Elbow Angle (°)', color='#38BDF8', linewidth=2)
        ax.plot(frames, r_knees, label='Right Knee Angle (°)', color='#34D399', linewidth=2)
        ax.set_title('Joint Angles vs Frame', color='#F8FAFC', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame Index', color='#94A3B8')
        ax.set_ylabel('Angle (degrees)', color='#94A3B8')
        ax.tick_params(colors='#94A3B8')
        ax.legend(facecolor='#0F172A', edgecolor='none', labelcolor='#F8FAFC')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        p1 = out_folder / "joint_angles.png"
        plt.tight_layout()
        plt.savefig(p1, facecolor='#0F172A')
        plt.close(fig)
        generated["joint_angles"] = p1

        # 2. Movement Speed Plot
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        ax.plot(frames, wrist_vels, label='Wrist Speed (px/s)', color='#F59E0B', linewidth=2)
        ax.plot(frames, bat_vels, label='Bat Tip Speed (px/s)', color='#EF4444', linewidth=2)
        ax.set_title('Movement Velocity Profile', color='#F8FAFC', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame Index', color='#94A3B8')
        ax.set_ylabel('Velocity (px/s)', color='#94A3B8')
        ax.tick_params(colors='#94A3B8')
        ax.legend(facecolor='#0F172A', edgecolor='none', labelcolor='#F8FAFC')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        p2 = out_folder / "movement_speed.png"
        plt.tight_layout()
        plt.savefig(p2, facecolor='#0F172A')
        plt.close(fig)
        generated["movement_speed"] = p2

        # 3. Body Lean Plot
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        ax.plot(frames, spines, label='Spine Lean Angle (°)', color='#A855F7', linewidth=2)
        ax.axhspan(10, 28, color='#34D399', alpha=0.15, label='Optimal Lean Zone')
        ax.set_title('Body Lean & Spine Inclination', color='#F8FAFC', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame Index', color='#94A3B8')
        ax.set_ylabel('Spine Angle (°)', color='#94A3B8')
        ax.tick_params(colors='#94A3B8')
        ax.legend(facecolor='#0F172A', edgecolor='none', labelcolor='#F8FAFC')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        p3 = out_folder / "body_lean.png"
        plt.tight_layout()
        plt.savefig(p3, facecolor='#0F172A')
        plt.close(fig)
        generated["body_lean"] = p3

        # 4. Head Movement Scatter
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        ax.scatter(head_xs, head_ys, c=frames, cmap='plasma', s=30, alpha=0.8)
        ax.invert_yaxis()
        ax.set_title('Head Drift Trajectory (2D Scatter)', color='#F8FAFC', fontsize=13, fontweight='bold')
        ax.set_xlabel('X Position (px)', color='#94A3B8')
        ax.set_ylabel('Y Position (px)', color='#94A3B8')
        ax.tick_params(colors='#94A3B8')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        p4 = out_folder / "head_movement.png"
        plt.tight_layout()
        plt.savefig(p4, facecolor='#0F172A')
        plt.close(fig)
        generated["head_movement"] = p4

        logger.info(f"Generated {len(generated)} analytics chart images in {out_folder}")
        return generated
