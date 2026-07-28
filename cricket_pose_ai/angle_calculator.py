"""
Cricket Pose Analyzer AI - Angle & Biomechanics Calculator Module
==================================================================
2D and 3D vector geometry, joint angle calculations, Center of Mass (COM)
segmental approximation, spine inclination, and body alignment metrics.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np

from cricket_pose_ai.pose_detector import Landmark
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.AngleCalculator")


@dataclass
class FrameBiomechanics:
    """Frame-level biomechanical angle and posture metrics."""
    frame_idx: int
    left_elbow_angle: float
    right_elbow_angle: float
    left_shoulder_angle: float
    right_shoulder_angle: float
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    spine_angle: float
    neck_angle: float
    head_tilt: float
    body_lean: float
    foot_distance_px: float
    stance_width_ratio: float
    shoulder_alignment_angle: float
    hip_alignment_angle: float
    com_x: float
    com_y: float
    com_px: int
    com_py: int


class BiomechanicsCalculator:
    """Vector math engine for calculating joint angles and posture."""

    @staticmethod
    def calculate_angle_3p(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float]
    ) -> float:
        """
        Calculates angle at vertex B formed by segments BA and BC in degrees [0, 180].
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        return float(np.degrees(angle_rad))

    @staticmethod
    def calculate_line_angle(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        reference_axis: str = "vertical"
    ) -> float:
        """
        Calculates angle of line p1->p2 relative to vertical or horizontal axis in degrees.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if reference_axis == "vertical":
            # Angle relative to vertical (upward = 0 deg)
            angle_rad = math.atan2(dx, -dy)
        else:
            # Angle relative to horizontal (right = 0 deg)
            angle_rad = math.atan2(dy, dx)

        return float(np.degrees(angle_rad))

    def compute_frame_biomechanics(
        self,
        landmarks: Dict[str, Landmark],
        frame_idx: int,
        img_w: int,
        img_h: int
    ) -> FrameBiomechanics:
        """Calculates all joint angles, COM, stance width, and alignments for a frame."""

        def get_pt(name: str) -> Optional[Tuple[float, float]]:
            lm = landmarks.get(name)
            return (lm.x, lm.y) if lm else None

        def get_px(name: str) -> Optional[Tuple[int, int]]:
            lm = landmarks.get(name)
            return (lm.px, lm.py) if lm else None

        # Key Landmarks
        ls, rs = get_pt("LEFT_SHOULDER"), get_pt("RIGHT_SHOULDER")
        le, re = get_pt("LEFT_ELBOW"), get_pt("RIGHT_ELBOW")
        lw, rw = get_pt("LEFT_WRIST"), get_pt("RIGHT_WRIST")
        lh, rh = get_pt("LEFT_HIP"), get_pt("RIGHT_HIP")
        lk, rk = get_pt("LEFT_KNEE"), get_pt("RIGHT_KNEE")
        la, ra = get_pt("LEFT_ANKLE"), get_pt("RIGHT_ANKLE")
        nose = get_pt("NOSE")
        lear, rear = get_pt("LEFT_EAR"), get_pt("RIGHT_EAR")

        # 1. Elbow Angles
        l_elbow = self.calculate_angle_3p(ls, le, lw) if (ls and le and lw) else 140.0
        r_elbow = self.calculate_angle_3p(rs, re, rw) if (rs and re and rw) else 140.0

        # 2. Knee Angles
        l_knee = self.calculate_angle_3p(lh, lk, la) if (lh and lk and la) else 165.0
        r_knee = self.calculate_angle_3p(rh, rk, ra) if (rh and rk and ra) else 165.0

        # 3. Shoulder Angles (Hip - Shoulder - Elbow)
        l_shoulder = self.calculate_angle_3p(lh, ls, le) if (lh and ls and le) else 45.0
        r_shoulder = self.calculate_angle_3p(rh, rs, re) if (rh and rs and re) else 45.0

        # 4. Hip Angles (Shoulder - Hip - Knee)
        l_hip = self.calculate_angle_3p(ls, lh, lk) if (ls and lh and lk) else 160.0
        r_hip = self.calculate_angle_3p(rs, rh, rk) if (rs and rh and rk) else 160.0

        # Midpoints
        mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if (ls and rs) else (0.5, 0.3)
        mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2) if (lh and rh) else (0.5, 0.6)
        mid_ankle = ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2) if (la and ra) else (0.5, 0.9)

        # 5. Spine Angle & Body Lean (Mid-Hip to Mid-Shoulder relative to vertical)
        spine_angle = abs(self.calculate_line_angle(mid_hip, mid_shoulder, "vertical"))
        body_lean = spine_angle

        # 6. Neck Angle & Head Tilt
        neck_angle = self.calculate_angle_3p(nose, mid_shoulder, mid_hip) if nose else 170.0
        head_tilt = abs(self.calculate_line_angle(lear, rear, "horizontal")) if (lear and rear) else 0.0

        # 7. Foot Distance & Stance Width Ratio
        la_px, ra_px = get_px("LEFT_ANKLE"), get_px("RIGHT_ANKLE")
        foot_dist_px = math.hypot(la_px[0] - ra_px[0], la_px[1] - ra_px[1]) if (la_px and ra_px) else 100.0

        ls_px, rs_px = get_px("LEFT_SHOULDER"), get_px("RIGHT_SHOULDER")
        shoulder_width_px = math.hypot(ls_px[0] - rs_px[0], ls_px[1] - rs_px[1]) if (ls_px and rs_px) else 80.0
        stance_width_ratio = foot_dist_px / (shoulder_width_px + 1e-5)

        # 8. Alignment Angles
        shoulder_align = self.calculate_line_angle(ls, rs, "horizontal") if (ls and rs) else 0.0
        hip_align = self.calculate_line_angle(lh, rh, "horizontal") if (lh and rh) else 0.0

        # 9. Center of Mass (COM) Approximation
        # Segment Weights: Head 8%, Torso 48%, Thighs 24%, Lower Legs 12%, Arms 8%
        com_x = 0.48 * mid_hip[0] + 0.32 * mid_shoulder[0] + 0.10 * (la[0] + ra[0])/2 + 0.10 * (lw[0] + rw[0])/2 if (la and ra and lw and rw) else mid_hip[0]
        com_y = 0.48 * mid_hip[1] + 0.32 * mid_shoulder[1] + 0.10 * (la[1] + ra[1])/2 + 0.10 * (lw[1] + rw[1])/2 if (la and ra and lw and rw) else mid_hip[1]

        com_px = int(np.clip(com_x * img_w, 0, img_w - 1))
        com_py = int(np.clip(com_y * img_h, 0, img_h - 1))

        return FrameBiomechanics(
            frame_idx=frame_idx,
            left_elbow_angle=round(l_elbow, 1),
            right_elbow_angle=round(r_elbow, 1),
            left_shoulder_angle=round(l_shoulder, 1),
            right_shoulder_angle=round(r_shoulder, 1),
            left_knee_angle=round(l_knee, 1),
            right_knee_angle=round(r_knee, 1),
            left_hip_angle=round(l_hip, 1),
            right_hip_angle=round(r_hip, 1),
            spine_angle=round(spine_angle, 1),
            neck_angle=round(neck_angle, 1),
            head_tilt=round(head_tilt, 1),
            body_lean=round(body_lean, 1),
            foot_distance_px=round(foot_dist_px, 1),
            stance_width_ratio=round(stance_width_ratio, 2),
            shoulder_alignment_angle=round(shoulder_align, 1),
            hip_alignment_angle=round(hip_align, 1),
            com_x=float(com_x),
            com_y=float(com_y),
            com_px=com_px,
            com_py=com_py
        )
