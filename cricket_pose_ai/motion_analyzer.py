"""
Cricket Pose Analyzer AI - Motion Analyzer & Shot Phase Module
================================================================
Calculates frame-to-frame joint displacements, velocities, accelerations,
signal smoothing, automatic shot phase segmentation, and overall scores.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from cricket_pose_ai.angle_calculator import FrameBiomechanics
from cricket_pose_ai.bat_detector import BatDetectionResult
from cricket_pose_ai.pose_detector import Landmark
from cricket_pose_ai.utils import smooth_series, setup_logger

logger = setup_logger("CricketPoseAI.MotionAnalyzer")


@dataclass
class ShotPhaseTimeline:
    """Timestamps and frame indices for cricket shot phases."""
    stance_start_frame: int = 0
    backlift_peak_frame: int = 0
    downswing_start_frame: int = 0
    impact_frame: int = 0
    followthrough_peak_frame: int = 0
    finish_frame: int = 0
    fps: float = 30.0

    @property
    def backlift_time_sec(self) -> float:
        return self.backlift_peak_frame / self.fps

    @property
    def impact_time_sec(self) -> float:
        return self.impact_frame / self.fps

    @property
    def followthrough_time_sec(self) -> float:
        return self.followthrough_peak_frame / self.fps


@dataclass
class MotionSummaryMetrics:
    """Aggregated biomechanical ratings and performance metrics."""
    total_frames: int
    fps: float
    duration_sec: float
    overall_technique_score: float
    balance_index: float
    head_stability_index: float
    weight_transfer_index: float
    rhythm_score: float
    posture_quality_score: float
    followthrough_quality_score: float
    max_wrist_velocity_px: float
    max_bat_speed_px: float
    max_spine_lean_deg: float
    head_drift_px: float
    stance_width_avg: float
    timeline: ShotPhaseTimeline = field(default_factory=ShotPhaseTimeline)


class MotionAnalyzer:
    """Analyzes kinematics, shot phases, and biomechanical scores."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def analyze_full_sequence(
        self,
        biomechanics_list: List[FrameBiomechanics],
        landmarks_history: List[Optional[Dict[str, Landmark]]],
        bat_history: List[BatDetectionResult],
        img_w: int,
        img_h: int
    ) -> Tuple[List[Dict[str, Any]], MotionSummaryMetrics]:
        """
        Executes sequence-level kinetic analysis, shot phase detection,
        and computes aggregate scores.
        """
        total_frames = len(biomechanics_list)
        dt = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

        if total_frames == 0:
            return [], self._empty_metrics(self.fps)

        # 1. Extract raw trajectories
        wrist_pts = []
        head_pts = []
        com_pts = []
        bat_pts = []

        for idx in range(total_frames):
            lms = landmarks_history[idx]
            bm = biomechanics_list[idx]
            bat = bat_history[idx]

            # Right wrist or midpoint wrist
            if lms and "RIGHT_WRIST" in lms:
                wrist_pts.append((lms["RIGHT_WRIST"].px, lms["RIGHT_WRIST"].py))
            else:
                wrist_pts.append((img_w // 2, img_h // 2))

            # Nose/Head
            if lms and "NOSE" in lms:
                head_pts.append((lms["NOSE"].px, lms["NOSE"].py))
            else:
                head_pts.append((img_w // 2, int(img_h * 0.2)))

            com_pts.append((bm.com_px, bm.com_py))

            if bat.detected:
                bat_pts.append(bat.bat_tip)
            else:
                bat_pts.append(wrist_pts[-1])

        # 2. Compute Frame-to-Frame Velocities & Accelerations
        wrist_vel = self._compute_velocity(wrist_pts, dt)
        bat_vel = self._compute_velocity(bat_pts, dt)
        head_vel = self._compute_velocity(head_pts, dt)
        com_vel = self._compute_velocity(com_pts, dt)

        # Smooth velocity signals
        wrist_vel_smooth = smooth_series(wrist_vel, window_size=7)
        bat_vel_smooth = smooth_series(bat_vel, window_size=7)

        # Accelerations
        wrist_accel = np.gradient(wrist_vel_smooth, dt)

        # 3. Shot Phase Detection
        timeline = self._detect_shot_phases(
            wrist_pts=wrist_pts,
            bat_pts=bat_pts,
            wrist_vel=wrist_vel_smooth,
            total_frames=total_frames,
            fps=self.fps
        )

        # 4. Compute Biomechanical Scores
        head_stability = self._calculate_head_stability(head_pts, timeline, img_h)
        balance_index = self._calculate_balance_index(com_pts, landmarks_history)
        weight_transfer = self._calculate_weight_transfer(com_pts, timeline)
        rhythm_score = self._calculate_rhythm_score(wrist_accel)
        posture_score = self._calculate_posture_score(biomechanics_list)
        followthrough_score = self._calculate_followthrough_score(biomechanics_list, timeline)

        # Overall Composite Score
        overall_score = float(np.clip(
            0.25 * balance_index +
            0.25 * head_stability +
            0.20 * posture_score +
            0.15 * weight_transfer +
            0.15 * followthrough_score,
            0.0, 100.0
        ))

        # 5. Build Frame-wise records
        framewise_records = []
        for idx in range(total_frames):
            bm = biomechanics_list[idx]
            bat = bat_history[idx]

            # Assign phase string label
            phase_label = self._get_phase_label(idx, timeline)

            record = {
                "frame": idx,
                "timestamp_sec": round(idx * dt, 3),
                "phase": phase_label,
                "left_elbow_angle": bm.left_elbow_angle,
                "right_elbow_angle": bm.right_elbow_angle,
                "left_knee_angle": bm.left_knee_angle,
                "right_knee_angle": bm.right_knee_angle,
                "spine_angle": bm.spine_angle,
                "neck_angle": bm.neck_angle,
                "head_tilt": bm.head_tilt,
                "body_lean": bm.body_lean,
                "stance_width_ratio": bm.stance_width_ratio,
                "wrist_velocity_px_s": round(float(wrist_vel_smooth[idx]), 1),
                "bat_velocity_px_s": round(float(bat_vel_smooth[idx]), 1),
                "head_x": head_pts[idx][0],
                "head_y": head_pts[idx][1],
                "com_x": com_pts[idx][0],
                "com_y": com_pts[idx][1],
                "bat_angle_deg": bat.angle_deg,
                "bat_lift_px": bat.lift_height_px
            }
            framewise_records.append(record)

        summary = MotionSummaryMetrics(
            total_frames=total_frames,
            fps=self.fps,
            duration_sec=round(total_frames * dt, 2),
            overall_technique_score=round(overall_score, 1),
            balance_index=round(balance_index, 1),
            head_stability_index=round(head_stability, 1),
            weight_transfer_index=round(weight_transfer, 1),
            rhythm_score=round(rhythm_score, 1),
            posture_quality_score=round(posture_score, 1),
            followthrough_quality_score=round(followthrough_score, 1),
            max_wrist_velocity_px=round(float(np.max(wrist_vel_smooth)), 1),
            max_bat_speed_px=round(float(np.max(bat_vel_smooth)), 1),
            max_spine_lean_deg=round(float(np.max([b.spine_angle for b in biomechanics_list])), 1),
            head_drift_px=round(float(np.std([p[0] for p in head_pts]) + np.std([p[1] for p in head_pts])), 1),
            stance_width_avg=round(float(np.mean([b.stance_width_ratio for b in biomechanics_list])), 2),
            timeline=timeline
        )

        return framewise_records, summary

    @staticmethod
    def _compute_velocity(pts: List[Tuple[int, int]], dt: float) -> np.ndarray:
        """Computes point-to-point velocity in pixels per second."""
        velocities = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            dist = np.hypot(dx, dy)
            velocities.append(dist / dt)
        return np.array(velocities, dtype=float)

    @staticmethod
    def _detect_shot_phases(
        wrist_pts: List[Tuple[int, int]],
        bat_pts: List[Tuple[int, int]],
        wrist_vel: np.ndarray,
        total_frames: int,
        fps: float
    ) -> ShotPhaseTimeline:
        """
        Segments video into stance, backlift peak, downswing, impact,
        follow-through peak, and finish.
        """
        if total_frames < 10:
            return ShotPhaseTimeline(0, 2, 4, 6, 8, total_frames - 1, fps)

        y_coords = [p[1] for p in wrist_pts]

        # 1. Backlift Peak: Lowest y-coordinate (highest point on image screen) before max velocity
        peak_vel_frame = int(np.argmax(wrist_vel))

        # Search for minimum Y (highest height) before peak velocity
        pre_peak_search = y_coords[:max(5, peak_vel_frame)]
        backlift_frame = int(np.argmin(pre_peak_search)) if len(pre_peak_search) > 0 else 0

        # Downswing start is right after backlift peak
        downswing_frame = min(total_frames - 1, backlift_frame + int(fps * 0.15))

        # Impact frame is near peak wrist velocity or lowest hand position post-downswing
        impact_frame = peak_vel_frame

        # Follow-through peak: Maximum extension/height after impact
        post_impact_search = y_coords[impact_frame:]
        if len(post_impact_search) > 0:
            followthrough_frame = impact_frame + int(np.argmin(post_impact_search))
        else:
            followthrough_frame = min(total_frames - 1, impact_frame + int(fps * 0.5))

        finish_frame = total_frames - 1

        return ShotPhaseTimeline(
            stance_start_frame=0,
            backlift_peak_frame=int(backlift_frame),
            downswing_start_frame=int(downswing_frame),
            impact_frame=int(impact_frame),
            followthrough_peak_frame=int(followthrough_frame),
            finish_frame=int(finish_frame),
            fps=fps
        )

    @staticmethod
    def _calculate_head_stability(
        head_pts: List[Tuple[int, int]],
        timeline: ShotPhaseTimeline,
        img_h: int
    ) -> float:
        """Evaluates head movement stability during downswing and impact."""
        start = timeline.backlift_peak_frame
        end = min(len(head_pts), timeline.followthrough_peak_frame + 1)

        if end <= start:
            return 85.0

        stroke_head_pts = head_pts[start:end]
        std_x = np.std([p[0] for p in stroke_head_pts])
        std_y = np.std([p[1] for p in stroke_head_pts])

        total_std_ratio = (std_x + std_y) / (img_h + 1e-5)

        # 0% drift -> 100 score, > 8% drift -> low score
        score = 100.0 - (total_std_ratio * 1000.0)
        return float(np.clip(score, 30.0, 98.0))

    @staticmethod
    def _calculate_balance_index(
        com_pts: List[Tuple[int, int]],
        landmarks_history: List[Optional[Dict[str, Landmark]]]
    ) -> float:
        """Evaluates if Center of Mass stays within foot support boundaries."""
        scores = []
        for idx, lms in enumerate(landmarks_history):
            if not lms or "LEFT_ANKLE" not in lms or "RIGHT_ANKLE" not in lms:
                scores.append(80.0)
                continue

            la_x = lms["LEFT_ANKLE"].px
            ra_x = lms["RIGHT_ANKLE"].px
            left_bound = min(la_x, ra_x) - 20
            right_bound = max(la_x, ra_x) + 20

            com_x = com_pts[idx][0]
            if left_bound <= com_x <= right_bound:
                scores.append(95.0)
            else:
                drift = min(abs(com_x - left_bound), abs(com_x - right_bound))
                scores.append(max(40.0, 95.0 - drift * 0.8))

        return float(np.mean(scores))

    @staticmethod
    def _calculate_weight_transfer(
        com_pts: List[Tuple[int, int]],
        timeline: ShotPhaseTimeline
    ) -> float:
        """Measures forward displacement of COM from backlift to impact."""
        if timeline.impact_frame >= len(com_pts) or timeline.backlift_peak_frame >= len(com_pts):
            return 80.0

        com_start = com_pts[timeline.backlift_peak_frame]
        com_impact = com_pts[timeline.impact_frame]

        dx = abs(com_impact[0] - com_start[0])
        # Moderate controlled shift (15-60 px) is ideal
        if 15 <= dx <= 70:
            return 92.0
        elif dx < 15:
            return 65.0  # Stationary / weight stuck back
        else:
            return 75.0  # Over-striding

    @staticmethod
    def _calculate_rhythm_score(accel_signal: np.ndarray) -> float:
        """Evaluates smoothness of acceleration (low jerk = high rhythm)."""
        jerk = np.diff(accel_signal)
        mean_jerk = np.mean(np.abs(jerk)) if len(jerk) > 0 else 1.0
        score = 100.0 - (mean_jerk * 0.05)
        return float(np.clip(score, 50.0, 96.0))

    @staticmethod
    def _calculate_posture_score(biomechanics_list: List[FrameBiomechanics]) -> float:
        """Evaluates spine inclination and knee flex quality."""
        spine_angles = [b.spine_angle for b in biomechanics_list]
        avg_spine = np.mean(spine_angles)
        # Optimal spine angle 10-25 deg
        if 10.0 <= avg_spine <= 25.0:
            return 94.0
        else:
            return max(50.0, 94.0 - abs(avg_spine - 17.5) * 2.0)

    @staticmethod
    def _calculate_followthrough_score(
        biomechanics_list: List[FrameBiomechanics],
        timeline: ShotPhaseTimeline
    ) -> float:
        """Evaluates elbow extension at follow-through peak."""
        idx = min(len(biomechanics_list) - 1, timeline.followthrough_peak_frame)
        if idx < 0:
            return 80.0
        bm = biomechanics_list[idx]
        max_elbow = max(bm.left_elbow_angle, bm.right_elbow_angle)
        if max_elbow >= 140.0:
            return 95.0
        else:
            return max(55.0, 95.0 - (140.0 - max_elbow) * 1.2)

    @staticmethod
    def _get_phase_label(frame_idx: int, timeline: ShotPhaseTimeline) -> str:
        """Returns human-readable shot phase label for a frame."""
        if frame_idx < timeline.backlift_peak_frame:
            return "Stance / Address"
        elif frame_idx < timeline.downswing_start_frame:
            return "Backlift Peak"
        elif frame_idx < timeline.impact_frame:
            return "Downswing"
        elif frame_idx == timeline.impact_frame:
            return "Impact"
        elif frame_idx <= timeline.followthrough_peak_frame:
            return "Follow-Through"
        else:
            return "Finish / Recovery"

    def _empty_metrics(self, fps: float) -> MotionSummaryMetrics:
        return MotionSummaryMetrics(
            total_frames=0, fps=fps, duration_sec=0.0,
            overall_technique_score=0.0, balance_index=0.0,
            head_stability_index=0.0, weight_transfer_index=0.0,
            rhythm_score=0.0, posture_quality_score=0.0,
            followthrough_quality_score=0.0, max_wrist_velocity_px=0.0,
            max_bat_speed_px=0.0, max_spine_lean_deg=0.0,
            head_drift_px=0.0, stance_width_avg=0.0
        )
