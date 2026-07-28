"""
Cricket Pose Analyzer AI - Pose Detector Module
================================================
MediaPipe Pose wrapper for extracting 33 full-body anatomical landmarks,
landmark coordinate transformation, confidence thresholding, and smoothing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

from cricket_pose_ai.config import PoseConfig
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.PoseDetector")


@dataclass
class Landmark:
    """Represents a single detected body keypoint."""
    id: int
    name: str
    x: float          # Normalized [0, 1]
    y: float          # Normalized [0, 1]
    z: float          # Relative depth
    visibility: float # Confidence [0, 1]
    px: int           # Pixel X coordinate
    py: int           # Pixel Y coordinate


class CricketPoseDetector:
    """Wrapper class for MediaPipe Pose estimation engine."""

    LANDMARK_NAMES = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or PoseConfig()
        self.mp_pose = None
        self.pose_engine = None
        self._init_mediapipe()
        self.previous_landmarks: Optional[Dict[str, Landmark]] = None
        self.smoothing_alpha = 0.65  # Exponential smoothing factor

    def _init_mediapipe(self):
        """Initializes MediaPipe Pose detector with static linter compatibility."""
        if mp is None:
            logger.error("MediaPipe library is not installed! Falling back to mock landmark mode.")
            return

        try:
            mp_solutions = getattr(mp, "solutions", None)
            if mp_solutions is not None and hasattr(mp_solutions, "pose"):
                self.mp_pose = mp_solutions.pose
            else:
                import importlib
                self.mp_pose = importlib.import_module("mediapipe.solutions.pose")

            self.pose_engine = self.mp_pose.Pose(
                static_image_mode=self.config.static_image_mode,
                model_complexity=self.config.model_complexity,
                smooth_landmarks=self.config.smooth_landmarks,
                enable_segmentation=self.config.enable_segmentation,
                smooth_segmentation=self.config.smooth_segmentation,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            logger.info("MediaPipe Pose initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Pose initialization note: {e}. Utilizing pose estimation fallback.")
            self.pose_engine = None

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[Dict[str, Landmark]], Optional[np.ndarray]]:
        """
        Processes a BGR OpenCV frame and returns a dictionary of landmarks by name
        and the raw MediaPipe results object.
        """
        if frame_bgr is None:
            return None, None

        h, w, _ = frame_bgr.shape

        if self.pose_engine is None:
            # Fallback mock detector for testing if mediapipe is unavailable
            return self._generate_mock_landmarks(w, h), None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose_engine.process(frame_rgb)

        if not results.pose_landmarks:
            return None, results

        landmarks_dict: Dict[str, Landmark] = {}
        raw_landmarks = results.pose_landmarks.landmark

        for i, lm in enumerate(raw_landmarks):
            name = self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else f"LANDMARK_{i}"
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))

            cur_lm = Landmark(
                id=i,
                name=name,
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
                px=px,
                py=py
            )

            # Apply temporal smoothing if previous landmarks exist
            if self.previous_landmarks and name in self.previous_landmarks:
                prev_lm = self.previous_landmarks[name]
                alpha = self.smoothing_alpha
                cur_lm.x = float(alpha * cur_lm.x + (1 - alpha) * prev_lm.x)
                cur_lm.y = float(alpha * cur_lm.y + (1 - alpha) * prev_lm.y)
                cur_lm.px = int(np.clip(cur_lm.x * w, 0, w - 1))
                cur_lm.py = int(np.clip(cur_lm.y * h, 0, h - 1))

            landmarks_dict[name] = cur_lm

        self.previous_landmarks = landmarks_dict
        return landmarks_dict, results

    def _generate_mock_landmarks(self, w: int, h: int) -> Dict[str, Landmark]:
        """Generates synthetic landmarks for testing in environments without MediaPipe."""
        mock = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            x = 0.5 + 0.1 * np.sin(i + 0.1)
            y = 0.2 + 0.02 * i
            mock[name] = Landmark(
                id=i, name=name, x=x, y=y, z=0.0, visibility=0.9,
                px=int(x * w), py=int(y * h)
            )
        return mock

    def release(self):
        """Releases MediaPipe resources."""
        if self.pose_engine:
            self.pose_engine.close()
            logger.info("MediaPipe Pose engine closed.")
