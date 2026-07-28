"""
Cricket Pose Analyzer AI - Bat Detector Module
===============================================
Hybrid Cricket Bat detection engine utilizing YOLO object detection, HSV color
segmentation, contour geometry analysis, and wrist trajectory fallback.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import math
import numpy as np

from cricket_pose_ai.config import BatConfig
from cricket_pose_ai.pose_detector import Landmark
from cricket_pose_ai.utils import calculate_euclidean_distance, setup_logger

logger = setup_logger("CricketPoseAI.BatDetector")


@dataclass
class BatDetectionResult:
    """Detection output for cricket bat in a frame."""
    detected: bool
    bat_tip: Tuple[int, int]
    bat_handle: Tuple[int, int]
    bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    angle_deg: float                           # Angle relative to vertical
    lift_height_px: float                      # Vertical distance from ankles/ground
    confidence: float
    detection_source: str                      # "yolo", "hsv_contour", "wrist_extension"


class CricketBatDetector:
    """Bat detection and swing path estimation engine."""

    def __init__(self, config: Optional[BatConfig] = None):
        self.config = config or BatConfig()
        self.yolo_model = None
        self._init_yolo()

    def _init_yolo(self):
        """Attempts to load YOLO model if available."""
        if not self.config.use_yolo:
            return

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.config.yolo_model_name)
            logger.info("YOLO model loaded for bat detection.")
        except Exception as e:
            logger.info(f"YOLO not available ({e}). Using HSV/wrist fallback for bat detection.")
            self.yolo_model = None

    def detect_bat(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[Dict[str, Landmark]]
    ) -> BatDetectionResult:
        """
        Detects bat location, orientation, tip, and handle.
        Tries YOLO -> HSV Contour near wrists -> Wrist-extension fallback.
        """
        if frame_bgr is None:
            return self._empty_result()

        h, w, _ = frame_bgr.shape

        # 1. Try YOLO detection if available
        if self.yolo_model is not None:
            res = self._detect_yolo(frame_bgr, landmarks)
            if res.detected:
                return res

        # 2. Try HSV Color & Contour Detection near Wrists
        if landmarks:
            res_hsv = self._detect_hsv(frame_bgr, landmarks)
            if res_hsv.detected:
                return res_hsv

        # 3. Fallback to Wrist Direction Vector Extension
        if landmarks and "RIGHT_WRIST" in landmarks and "LEFT_WRIST" in landmarks:
            return self._detect_wrist_extension(landmarks, w, h)

        return self._empty_result()

    def _detect_yolo(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[Dict[str, Landmark]]
    ) -> BatDetectionResult:
        """YOLO-based detection."""
        try:
            results = self.yolo_model(frame_bgr, verbose=False, conf=self.config.confidence_threshold)[0]
            boxes = results.boxes
            for box in boxes:
                # Check for bat class or elongated sports object
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id] if hasattr(self.yolo_model, 'names') else ""
                if "bat" in cls_name.lower() or "sports ball" in cls_name.lower() or cls_id == 38:  # 38 is baseball bat in COCO
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0])

                    handle = (int((x1 + x2) / 2), int(y1))
                    tip = (int((x1 + x2) / 2), int(y2))

                    angle = self._calculate_vector_angle(handle, tip)
                    lift = self._calculate_lift_height(tip, landmarks, frame_bgr.shape[0])

                    return BatDetectionResult(
                        detected=True,
                        bat_tip=tip,
                        bat_handle=handle,
                        bbox=(x1, y1, x2, y2),
                        angle_deg=angle,
                        lift_height_px=lift,
                        confidence=conf,
                        detection_source="yolo"
                    )
        except Exception:
            pass

        return self._empty_result()

    def _detect_hsv(
        self,
        frame_bgr: np.ndarray,
        landmarks: Dict[str, Landmark]
    ) -> BatDetectionResult:
        """Detects bat contour in HSV color space around wrists."""
        rw = landmarks.get("RIGHT_WRIST")
        lw = landmarks.get("LEFT_WRIST")
        if not rw or not lw:
            return self._empty_result()

        h, w, _ = frame_bgr.shape
        wrist_x = (rw.px + lw.px) // 2
        wrist_y = (rw.py + lw.py) // 2

        # Define Region of Interest (ROI) around wrists
        roi_size = int(max(w, h) * 0.25)
        x1 = max(0, wrist_x - roi_size)
        y1 = max(0, wrist_y - roi_size)
        x2 = min(w, wrist_x + roi_size)
        y2 = min(h, wrist_y + roi_size)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return self._empty_result()

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.bat_color_hsv_min, self.config.bat_color_hsv_max)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:
                rect = cv2.minAreaRect(cnt)
                (w_r, h_r) = rect[1]
                aspect_ratio = max(w_r, h_r) / (min(w_r, h_r) + 1e-5)
                if aspect_ratio > 2.0 and area > max_area:
                    max_area = area
                    best_cnt = cnt

        if best_cnt is not None:
            vx, vy, cx, cy = cv2.fitLine(best_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            # Offset to full image coordinates
            vx_f, vy_f = float(vx[0]), float(vy[0])
            cx_f, cy_f = float(cx[0]) + x1, float(cy[0]) + y1

            length = 80.0
            p1 = (int(cx_f - vx_f * length), int(cy_f - vy_f * length))
            p2 = (int(cx_f + vx_f * length), int(cy_f + vy_f * length))

            # Handle is closer to wrists
            d1 = calculate_euclidean_distance(p1, (wrist_x, wrist_y))
            d2 = calculate_euclidean_distance(p2, (wrist_x, wrist_y))

            handle = p1 if d1 < d2 else p2
            tip = p2 if d1 < d2 else p1
            angle = self._calculate_vector_angle(handle, tip)
            lift = self._calculate_lift_height(tip, landmarks, h)

            return BatDetectionResult(
                detected=True,
                bat_tip=tip,
                bat_handle=handle,
                bbox=(x1, y1, x2, y2),
                angle_deg=angle,
                lift_height_px=lift,
                confidence=0.75,
                detection_source="hsv_contour"
            )

        return self._empty_result()

    def _detect_wrist_extension(
        self,
        landmarks: Dict[str, Landmark],
        w: int,
        h: int
    ) -> BatDetectionResult:
        """Estimates bat trajectory as an extension from elbow through wrist."""
        rw = landmarks.get("RIGHT_WRIST")
        re = landmarks.get("RIGHT_ELBOW")
        la = landmarks.get("LEFT_ANKLE")
        ra = landmarks.get("RIGHT_ANKLE")

        if not rw or not re:
            return self._empty_result()

        # Vector from elbow to wrist extended outward
        dir_x = rw.px - re.px
        dir_y = rw.py - re.py
        norm = math.hypot(dir_x, dir_y) + 1e-5
        dir_x /= norm
        dir_y /= norm

        bat_length = int(min(w, h) * 0.22)
        handle = (rw.px, rw.py)
        tip = (int(rw.px + dir_x * bat_length), int(rw.py + dir_y * bat_length))

        angle = self._calculate_vector_angle(handle, tip)

        # Ground ankle level reference
        ankle_y = (la.py + ra.py) / 2 if la and ra else h * 0.9
        lift_height = max(0.0, ankle_y - tip[1])

        return BatDetectionResult(
            detected=True,
            bat_tip=tip,
            bat_handle=handle,
            bbox=None,
            angle_deg=angle,
            lift_height_px=lift_height,
            confidence=0.55,
            detection_source="wrist_extension"
        )

    @staticmethod
    def _calculate_vector_angle(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates angle of vector p1->p2 relative to vertical (upward = 0 deg)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dx, -dy)  # Upward is 0, right is positive
        return math.degrees(angle_rad)

    @staticmethod
    def _calculate_lift_height(
        tip: Tuple[int, int],
        landmarks: Optional[Dict[str, Landmark]],
        frame_height: int
    ) -> float:
        """Calculates vertical lift height of bat tip above ankle ground level."""
        if landmarks and "LEFT_ANKLE" in landmarks and "RIGHT_ANKLE" in landmarks:
            ground_y = (landmarks["LEFT_ANKLE"].py + landmarks["RIGHT_ANKLE"].py) / 2
        else:
            ground_y = frame_height * 0.9
        return float(max(0.0, ground_y - tip[1]))

    @staticmethod
    def _empty_result() -> BatDetectionResult:
        """Returns empty default result when bat is not detected."""
        return BatDetectionResult(
            detected=False,
            bat_tip=(0, 0),
            bat_handle=(0, 0),
            bbox=None,
            angle_deg=0.0,
            lift_height_px=0.0,
            confidence=0.0,
            detection_source="none"
        )
